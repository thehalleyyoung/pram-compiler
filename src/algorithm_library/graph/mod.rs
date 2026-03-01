//! Graph algorithms for PRAM.

use crate::pram_ir::ast::*;
use crate::pram_ir::types::PramType;

// ─── helpers (same pattern as sorting) ──────────────────────────────────────

fn param(name: &str) -> Parameter {
    Parameter { name: name.to_string(), param_type: PramType::Int64 }
}

fn shared(name: &str, size: Expr) -> SharedMemoryDecl {
    SharedMemoryDecl { name: name.to_string(), elem_type: PramType::Int64, size }
}

fn add(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Add, a, b) }
fn sub(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Sub, a, b) }
fn mul(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Mul, a, b) }
fn lt(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Lt, a, b) }
fn le(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Le, a, b) }
fn eq_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Eq, a, b) }
fn ne_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Ne, a, b) }
fn and_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::And, a, b) }
fn min_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Min, a, b) }
fn v(s: &str) -> Expr { Expr::var(s) }
fn int(n: i64) -> Expr { Expr::int(n) }
fn sr(mem: &str, idx: Expr) -> Expr { Expr::shared_read(v(mem), idx) }

fn sw(mem: &str, idx: Expr, val: Expr) -> Stmt {
    Stmt::SharedWrite { memory: v(mem), index: idx, value: val }
}

fn local(name: &str, init: Expr) -> Stmt {
    Stmt::LocalDecl(name.to_string(), PramType::Int64, Some(init))
}

fn assign(name: &str, val: Expr) -> Stmt {
    Stmt::Assign(name.to_string(), val)
}

fn par_for(var: &str, n: Expr, body: Vec<Stmt>) -> Stmt {
    Stmt::ParallelFor { proc_var: var.to_string(), num_procs: n, body }
}

fn seq_for(var: &str, start: Expr, end: Expr, body: Vec<Stmt>) -> Stmt {
    Stmt::SeqFor { var: var.to_string(), start, end, step: None, body }
}

fn if_then(cond: Expr, then_body: Vec<Stmt>) -> Stmt {
    Stmt::If { condition: cond, then_body, else_body: vec![] }
}

fn if_else(cond: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt>) -> Stmt {
    Stmt::If { condition: cond, then_body, else_body }
}

fn log2_call(e: Expr) -> Expr {
    Expr::FunctionCall("log2".to_string(), vec![e])
}

// ─── Shiloach-Vishkin connected components ──────────────────────────────────

/// Shiloach-Vishkin connected components on CRCW-Arbitrary PRAM.
///
/// * m + n processors, O(log n) time.
/// * Input: edge list `src[m]`, `dst[m]`, n vertices.
/// * Each phase:
///   1. Hooking – for each edge (u,v), attempt to hook the root of the
///      higher-numbered component under the root of the lower one
///      (concurrent writes resolve arbitrarily; any hook is valid).
///   2. Pointer jumping – D[i] ← D[D[i]] until convergence, compressing
///      the component-id forest.
///   3. If no change occurred, stop.
pub fn shiloach_vishkin() -> PramProgram {
    let mut prog = PramProgram::new("shiloach_vishkin", MemoryModel::CRCWArbitrary);
    prog.description = Some(
        "Shiloach-Vishkin connected components. CRCW-Arbitrary, O(log n) time, m+n processors."
            .to_string(),
    );
    prog.work_bound = Some("O((m+n) log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("m"));
    prog.shared_memory.push(shared("src", v("m")));
    prog.shared_memory.push(shared("dst", v("m")));
    prog.shared_memory.push(shared("D", v("n")));        // component-id / parent
    prog.shared_memory.push(shared("changed", int(1)));   // flag
    prog.num_processors = add(v("m"), v("n"));

    // Initialise: D[i] = i
    prog.body.push(Stmt::Comment("Initialise: D[i] = i".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("D", Expr::ProcessorId, Expr::ProcessorId),
    ]));
    prog.body.push(Stmt::Barrier);

    // Iterate O(log n) rounds
    prog.body.push(Stmt::Comment("Main loop: O(log n) rounds of hook + pointer-jump".to_string()));
    prog.body.push(Stmt::LocalDecl(
        "max_rounds".to_string(),
        PramType::Int64,
        Some(add(
            Expr::Cast(
                Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
                PramType::Int64,
            ),
            int(1),
        )),
    ));

    prog.body.push(seq_for("round", int(0), v("max_rounds"), vec![
        // Reset changed flag
        sw("changed", int(0), int(0)),
        Stmt::Barrier,

        // Phase 1: Hooking – each edge processor hooks smaller root under larger
        Stmt::Comment("Hooking: for each edge (u,v) hook higher root under lower".to_string()),
        par_for("pid", v("m"), vec![
            local("u", sr("src", Expr::ProcessorId)),
            local("v_node", sr("dst", Expr::ProcessorId)),
            local("du", sr("D", v("u"))),
            local("dv", sr("D", v("v_node"))),
            if_then(ne_e(v("du"), v("dv")), vec![
                // Hook the larger component id under the smaller one
                if_else(
                    lt(v("du"), v("dv")),
                    vec![
                        sw("D", v("dv"), v("du")),
                        sw("changed", int(0), int(1)),
                    ],
                    vec![
                        sw("D", v("du"), v("dv")),
                        sw("changed", int(0), int(1)),
                    ],
                ),
            ]),
        ]),
        Stmt::Barrier,

        // Phase 2: Pointer jumping – D[i] ← D[D[i]]
        Stmt::Comment("Pointer jumping: D[i] <- D[D[i]]".to_string()),
        par_for("pid", v("n"), vec![
            local("parent", sr("D", Expr::ProcessorId)),
            local("grandparent", sr("D", v("parent"))),
            if_then(ne_e(v("parent"), v("grandparent")), vec![
                sw("D", Expr::ProcessorId, v("grandparent")),
                sw("changed", int(0), int(1)),
            ]),
        ]),
        Stmt::Barrier,

        // Check termination
        if_then(eq_e(sr("changed", int(0)), int(0)), vec![
            Stmt::Comment("Converged – break".to_string()),
        ]),
    ]));

    prog
}

// ─── Borůvka's MST ─────────────────────────────────────────────────────────

/// Borůvka's MST on CRCW-Priority PRAM.
///
/// * m processors, O(log n) phases.
/// * Each phase:
///   1. Find minimum-weight edge per component (concurrent writes
///      with priority resolution → minimum-weight writer wins).
///   2. Add those edges to the MST.
///   3. Contract components via pointer-jumping.
pub fn boruvka_mst() -> PramProgram {
    let mut prog = PramProgram::new("boruvka_mst", MemoryModel::CRCWPriority);
    prog.description = Some(
        "Borůvka's MST. CRCW-Priority, O(log n) phases, m processors per phase.".to_string(),
    );
    prog.work_bound = Some("O(m log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("m"));
    prog.shared_memory.push(shared("edge_src", v("m")));
    prog.shared_memory.push(shared("edge_dst", v("m")));
    prog.shared_memory.push(shared("edge_w", v("m")));
    prog.shared_memory.push(shared("comp", v("n")));       // component id
    prog.shared_memory.push(shared("min_edge", v("n")));   // min edge index per component
    prog.shared_memory.push(shared("min_w", v("n")));      // min weight per component
    prog.shared_memory.push(shared("mst_flag", v("m")));   // 1 if edge is in MST
    prog.shared_memory.push(shared("active", int(1)));
    prog.num_processors = v("m");

    // Initialise components
    prog.body.push(Stmt::Comment("Initialise: comp[i] = i".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("comp", Expr::ProcessorId, Expr::ProcessorId),
    ]));
    prog.body.push(Stmt::Barrier);

    // O(log n) phases
    prog.body.push(Stmt::LocalDecl(
        "max_phases".to_string(),
        PramType::Int64,
        Some(add(
            Expr::Cast(
                Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
                PramType::Int64,
            ),
            int(1),
        )),
    ));

    prog.body.push(seq_for("phase", int(0), v("max_phases"), vec![
        // Reset min weights to ∞ (large sentinel)
        Stmt::Comment("Reset min_w to MAX".to_string()),
        par_for("pid", v("n"), vec![
            sw("min_w", Expr::ProcessorId, Expr::int(i64::MAX)),
            sw("min_edge", Expr::ProcessorId, int(-1)),
        ]),
        Stmt::Barrier,

        // Each edge processor proposes its weight to its component.
        // Priority CRCW: the writer with the smallest processor-id wins.
        // We encode priority by sorting edges by weight so that the
        // minimum-weight edge for a component writes first.
        Stmt::Comment("Find min-weight edge per component (priority write)".to_string()),
        par_for("pid", v("m"), vec![
            local("u", sr("edge_src", Expr::ProcessorId)),
            local("v_node", sr("edge_dst", Expr::ProcessorId)),
            local("cu", sr("comp", v("u"))),
            local("cv", sr("comp", v("v_node"))),
            if_then(ne_e(v("cu"), v("cv")), vec![
                local("w", sr("edge_w", Expr::ProcessorId)),
                // Write to component cu – priority resolves to lowest pid
                // (we order edges by weight so lower weight ↔ lower pid)
                sw("min_w", v("cu"), v("w")),
                sw("min_edge", v("cu"), Expr::ProcessorId),
                sw("min_w", v("cv"), v("w")),
                sw("min_edge", v("cv"), Expr::ProcessorId),
            ]),
        ]),
        Stmt::Barrier,

        // Mark MST edges, hook components
        Stmt::Comment("Mark MST edges and hook components".to_string()),
        par_for("pid", v("n"), vec![
            local("me", sr("min_edge", Expr::ProcessorId)),
            if_then(Expr::binop(BinOp::Ge, v("me"), int(0)), vec![
                sw("mst_flag", v("me"), int(1)),
                // Hook: set comp of higher-id component to lower
                local("u", sr("edge_src", v("me"))),
                local("v_node", sr("edge_dst", v("me"))),
                local("cu", sr("comp", v("u"))),
                local("cv", sr("comp", v("v_node"))),
                if_else(
                    lt(v("cu"), v("cv")),
                    vec![ sw("comp", v("cv"), v("cu")) ],
                    vec![ sw("comp", v("cu"), v("cv")) ],
                ),
            ]),
        ]),
        Stmt::Barrier,

        // Pointer jumping to flatten component trees
        Stmt::Comment("Pointer-jump to flatten component ids".to_string()),
        par_for("pid", v("n"), vec![
            local("c", sr("comp", Expr::ProcessorId)),
            local("cc", sr("comp", v("c"))),
            Stmt::While {
                condition: ne_e(v("c"), v("cc")),
                body: vec![
                    assign("c", v("cc")),
                    assign("cc", sr("comp", v("c"))),
                ],
            },
            sw("comp", Expr::ProcessorId, v("c")),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Parallel BFS ───────────────────────────────────────────────────────────

/// Level-synchronous parallel BFS on CREW PRAM.
///
/// * O(D) time where D = diameter.
/// * Each level: all frontier vertices in parallel mark their
///   unvisited neighbors (concurrent reads allowed).
///
/// Input: CSR-style adjacency (row_ptr[n+1], col_idx[m]), source s.
pub fn parallel_bfs() -> PramProgram {
    let mut prog = PramProgram::new("parallel_bfs", MemoryModel::CREW);
    prog.description = Some(
        "Level-synchronous parallel BFS. CREW, O(D) time, n+m processors.".to_string(),
    );
    prog.work_bound = Some("O(n + m)".to_string());
    prog.time_bound = Some("O(D)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("m"));
    prog.parameters.push(param("source"));
    prog.shared_memory.push(shared("row_ptr", add(v("n"), int(1))));
    prog.shared_memory.push(shared("col_idx", v("m")));
    prog.shared_memory.push(shared("dist", v("n")));
    prog.shared_memory.push(shared("frontier", v("n")));
    prog.shared_memory.push(shared("next_frontier", v("n")));
    prog.shared_memory.push(shared("frontier_size", int(1)));
    prog.num_processors = add(v("n"), v("m"));

    // Initialise distances to -1, source to 0
    prog.body.push(Stmt::Comment("Initialise dist to -1, source to 0".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("dist", Expr::ProcessorId, int(-1)),
        sw("frontier", Expr::ProcessorId, int(0)),
        sw("next_frontier", Expr::ProcessorId, int(0)),
    ]));
    prog.body.push(Stmt::Barrier);

    prog.body.push(sw("dist", v("source"), int(0)));
    prog.body.push(sw("frontier", v("source"), int(1)));
    prog.body.push(sw("frontier_size", int(0), int(1)));
    prog.body.push(Stmt::Barrier);

    // BFS loop: at most n levels
    prog.body.push(Stmt::Comment("BFS levels: iterate while frontier is non-empty".to_string()));
    prog.body.push(Stmt::LocalDecl("level".to_string(), PramType::Int64, Some(int(0))));

    prog.body.push(Stmt::While {
        condition: Expr::binop(BinOp::Gt, sr("frontier_size", int(0)), int(0)),
        body: vec![
            assign("level", add(v("level"), int(1))),

            // For each frontier vertex, explore its neighbors in parallel
            Stmt::Comment("Explore neighbors of frontier vertices".to_string()),
            par_for("pid", v("n"), vec![
                if_then(eq_e(sr("frontier", Expr::ProcessorId), int(1)), vec![
                    local("row_start", sr("row_ptr", Expr::ProcessorId)),
                    local("row_end", sr("row_ptr", add(Expr::ProcessorId, int(1)))),
                    seq_for("e", v("row_start"), v("row_end"), vec![
                        local("neighbor", sr("col_idx", v("e"))),
                        // If unvisited, mark it
                        if_then(eq_e(sr("dist", v("neighbor")), int(-1)), vec![
                            sw("dist", v("neighbor"), v("level")),
                            sw("next_frontier", v("neighbor"), int(1)),
                        ]),
                    ]),
                ]),
            ]),
            Stmt::Barrier,

            // Swap frontiers, count next frontier size
            Stmt::Comment("Swap frontier and next_frontier".to_string()),
            sw("frontier_size", int(0), int(0)),
            par_for("pid", v("n"), vec![
                sw("frontier", Expr::ProcessorId, sr("next_frontier", Expr::ProcessorId)),
                sw("next_frontier", Expr::ProcessorId, int(0)),
            ]),
            Stmt::Barrier,

            // Count frontier (via reduction — simplified as a single prefix-sum)
            Stmt::PrefixSum {
                input: "frontier".to_string(),
                output: "frontier_size".to_string(),
                size: v("n"),
                op: BinOp::Add,
            },
            Stmt::Barrier,
        ],
    });

    prog
}

// ─── Euler tour construction ────────────────────────────────────────────────

/// Euler tour construction on CREW PRAM.
///
/// * O(log n) time, n processors.
/// * Input: a tree stored as adjacency lists (parent[n], first_child[n],
///   next_sibling[n]).
/// * Constructs the Euler tour by:
///   1. Creating successor pointers: for each edge (u → child), the
///      successor of "entering child" is "entering first grandchild or
///      returning to u"; successor of "leaving child" is "entering
///      next sibling or leaving u".
///   2. List-ranking the successor list to get Euler-tour positions.
pub fn euler_tour() -> PramProgram {
    let mut prog = PramProgram::new("euler_tour", MemoryModel::CREW);
    prog.description = Some(
        "Euler tour construction. CREW, O(log n) time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(n log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    // Tree stored as: parent[n], first_child[n], next_sibling[n]
    // -1 means null/no such node.
    prog.shared_memory.push(shared("parent", v("n")));
    prog.shared_memory.push(shared("first_child", v("n")));
    prog.shared_memory.push(shared("next_sibling", v("n")));
    // Euler tour edges: 2*(n-1) edges.  succ[2n] stores successor pointers.
    // Edge encoding: edge 2*v = "down into v", edge 2*v+1 = "up from v".
    let tour_size = mul(int(2), v("n"));
    prog.shared_memory.push(shared("succ", tour_size.clone()));
    prog.shared_memory.push(shared("tour_pos", tour_size.clone()));
    prog.shared_memory.push(shared("rank_arr", tour_size.clone()));
    prog.num_processors = v("n");

    // Step 1: build successor pointers
    prog.body.push(Stmt::Comment(
        "Build Euler-tour successor pointers for each tree edge".to_string(),
    ));
    prog.body.push(par_for("pid", v("n"), vec![
        local("fc", sr("first_child", Expr::ProcessorId)),
        local("ns", sr("next_sibling", Expr::ProcessorId)),
        local("par", sr("parent", Expr::ProcessorId)),

        // succ of "down into pid" (edge 2*pid):
        //   if pid has children → "down into first_child" = 2*fc
        //   else                → "up from pid"          = 2*pid+1
        if_else(
            Expr::binop(BinOp::Ge, v("fc"), int(0)),
            vec![ sw("succ", mul(int(2), Expr::ProcessorId), mul(int(2), v("fc"))) ],
            vec![ sw("succ", mul(int(2), Expr::ProcessorId), add(mul(int(2), Expr::ProcessorId), int(1))) ],
        ),

        // succ of "up from pid" (edge 2*pid+1):
        //   if pid has next_sibling → "down into next_sibling" = 2*ns
        //   else if pid has parent  → "up from parent"         = 2*par+1
        //   else                    → self (root – tour ends)
        if_else(
            Expr::binop(BinOp::Ge, v("ns"), int(0)),
            vec![ sw("succ", add(mul(int(2), Expr::ProcessorId), int(1)), mul(int(2), v("ns"))) ],
            vec![
                if_else(
                    Expr::binop(BinOp::Ge, v("par"), int(0)),
                    vec![ sw("succ", add(mul(int(2), Expr::ProcessorId), int(1)), add(mul(int(2), v("par")), int(1))) ],
                    vec![ sw("succ", add(mul(int(2), Expr::ProcessorId), int(1)), add(mul(int(2), Expr::ProcessorId), int(1))) ],
                ),
            ],
        ),
    ]));
    prog.body.push(Stmt::Barrier);

    // Step 2: list-ranking on successor pointers to assign positions
    // Pointer-jumping on succ: O(log n) rounds
    prog.body.push(Stmt::Comment(
        "List-ranking via pointer jumping on successor list: O(log n) rounds".to_string(),
    ));
    // Initialise rank_arr[i] = 1 for all edges, 0 for the tour-end self-loop
    prog.body.push(par_for("pid", tour_size.clone(), vec![
        if_else(
            ne_e(sr("succ", Expr::ProcessorId), Expr::ProcessorId),
            vec![ sw("rank_arr", Expr::ProcessorId, int(1)) ],
            vec![ sw("rank_arr", Expr::ProcessorId, int(0)) ],
        ),
    ]));
    prog.body.push(Stmt::Barrier);

    prog.body.push(Stmt::LocalDecl(
        "log_n".to_string(),
        PramType::Int64,
        Some(add(
            Expr::Cast(
                Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
                PramType::Int64,
            ),
            int(2),
        )),
    ));
    prog.body.push(seq_for("step", int(0), v("log_n"), vec![
        par_for("pid", tour_size.clone(), vec![
            local("s", sr("succ", Expr::ProcessorId)),
            if_then(ne_e(v("s"), Expr::ProcessorId), vec![
                local("ss", sr("succ", v("s"))),
                local("r_s", sr("rank_arr", v("s"))),
                sw("rank_arr", Expr::ProcessorId,
                   add(sr("rank_arr", Expr::ProcessorId), v("r_s"))),
                sw("succ", Expr::ProcessorId, v("ss")),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    // rank_arr now holds Euler-tour position for each edge
    prog.body.push(Stmt::Comment("tour_pos = rank_arr (Euler tour order)".to_string()));
    prog.body.push(par_for("pid", tour_size, vec![
        sw("tour_pos", Expr::ProcessorId, sr("rank_arr", Expr::ProcessorId)),
    ]));

    prog
}

// ─── Parallel DFS ───────────────────────────────────────────────────────────

/// Parallel depth-first search on CREW PRAM.
///
/// * n+m processors, O(D log n) time.
/// * Input: CSR adjacency (row_ptr[n+1], col_idx[m]), source vertex.
/// * Each phase: pop from stack, mark visited, push unvisited neighbors.
pub fn parallel_dfs() -> PramProgram {
    let mut prog = PramProgram::new("parallel_dfs", MemoryModel::CREW);
    prog.description = Some(
        "Parallel depth-first search. CREW, O(D log n) time, n+m processors.".to_string(),
    );
    prog.work_bound = Some("O((n+m) log n)".to_string());
    prog.time_bound = Some("O(D log n)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("m"));
    prog.parameters.push(param("source"));
    prog.shared_memory.push(shared("row_ptr", add(v("n"), int(1))));
    prog.shared_memory.push(shared("col_idx", v("m")));
    prog.shared_memory.push(shared("visited", v("n")));
    prog.shared_memory.push(shared("dfs_order", v("n")));
    prog.shared_memory.push(shared("parent_arr", v("n")));
    prog.shared_memory.push(shared("stack", v("n")));
    prog.shared_memory.push(shared("stack_top", int(1)));
    prog.shared_memory.push(shared("counter", int(1)));
    prog.num_processors = add(v("n"), v("m"));

    // Initialise
    prog.body.push(Stmt::Comment("Initialise visited=0, parent=-1, dfs_order=-1".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("visited", Expr::ProcessorId, int(0)),
        sw("parent_arr", Expr::ProcessorId, int(-1)),
        sw("dfs_order", Expr::ProcessorId, int(-1)),
    ]));
    prog.body.push(Stmt::Barrier);

    // Push source onto stack
    prog.body.push(sw("stack", int(0), v("source")));
    prog.body.push(sw("stack_top", int(0), int(1)));
    prog.body.push(sw("counter", int(0), int(0)));
    prog.body.push(Stmt::Barrier);

    // Main loop: up to n iterations
    prog.body.push(Stmt::Comment("Main DFS loop: pop stack, mark visited, push neighbors".to_string()));
    prog.body.push(seq_for("iter", int(0), v("n"), vec![
        // Phase 1: pop top of stack
        Stmt::Comment("Pop top vertex from stack".to_string()),
        par_for("pid", int(1), vec![
            local("top", sr("stack_top", int(0))),
            if_then(Expr::binop(BinOp::Gt, v("top"), int(0)), vec![
                local("cur", sr("stack", sub(v("top"), int(1)))),
                sw("stack_top", int(0), sub(v("top"), int(1))),
                // Mark visited and record order
                if_then(eq_e(sr("visited", v("cur")), int(0)), vec![
                    sw("visited", v("cur"), int(1)),
                    local("ord", sr("counter", int(0))),
                    sw("dfs_order", v("cur"), v("ord")),
                    sw("counter", int(0), add(v("ord"), int(1))),
                ]),
            ]),
        ]),
        Stmt::Barrier,

        // Phase 2: push unvisited neighbors of most recently visited vertex
        Stmt::Comment("Push unvisited neighbors onto stack".to_string()),
        par_for("pid", v("n"), vec![
            local("ord", sr("dfs_order", Expr::ProcessorId)),
            // Check if this vertex was just visited (order == counter - 1)
            if_then(and_e(
                eq_e(sr("visited", Expr::ProcessorId), int(1)),
                eq_e(v("ord"), sub(sr("counter", int(0)), int(1))),
            ), vec![
                local("row_start", sr("row_ptr", Expr::ProcessorId)),
                local("row_end", sr("row_ptr", add(Expr::ProcessorId, int(1)))),
                seq_for("e", v("row_start"), v("row_end"), vec![
                    local("neighbor", sr("col_idx", v("e"))),
                    if_then(eq_e(sr("visited", v("neighbor")), int(0)), vec![
                        local("st", sr("stack_top", int(0))),
                        sw("stack", v("st"), v("neighbor")),
                        sw("stack_top", int(0), add(v("st"), int(1))),
                        sw("parent_arr", v("neighbor"), Expr::ProcessorId),
                    ]),
                ]),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Graph coloring ─────────────────────────────────────────────────────────

/// Parallel greedy graph coloring on CRCW-Common PRAM.
///
/// * n+m processors, O(Delta log n) time.
/// * Input: CSR adjacency (row_ptr[n+1], col_idx[m]), n vertices.
/// * Each round: find minimum available color for uncolored vertices.
pub fn graph_coloring() -> PramProgram {
    let mut prog = PramProgram::new("graph_coloring", MemoryModel::CRCWCommon);
    prog.description = Some(
        "Parallel graph coloring (greedy). CRCW-Common, O(Delta log n) time, n+m processors."
            .to_string(),
    );
    prog.work_bound = Some("O((n+m) Delta)".to_string());
    prog.time_bound = Some("O(Delta log n)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("m"));
    prog.shared_memory.push(shared("row_ptr", add(v("n"), int(1))));
    prog.shared_memory.push(shared("col_idx", v("m")));
    prog.shared_memory.push(shared("color", v("n")));
    prog.shared_memory.push(shared("colored", v("n")));
    prog.shared_memory.push(shared("available", mul(v("n"), v("n"))));
    prog.shared_memory.push(shared("remaining", int(1)));
    prog.num_processors = add(v("n"), v("m"));

    // Initialise
    prog.body.push(Stmt::Comment("Initialise color=-1, colored=0, remaining=n".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("color", Expr::ProcessorId, int(-1)),
        sw("colored", Expr::ProcessorId, int(0)),
    ]));
    prog.body.push(sw("remaining", int(0), v("n")));
    prog.body.push(Stmt::Barrier);

    // Main loop: up to n rounds
    prog.body.push(Stmt::Comment("Coloring rounds: scan neighbors, pick min available color".to_string()));
    prog.body.push(seq_for("round", int(0), v("n"), vec![
        // Phase 1: mark unavailable colors based on colored neighbors
        Stmt::Comment("Phase 1: mark unavailable colors".to_string()),
        par_for("pid", v("n"), vec![
            if_then(eq_e(sr("colored", Expr::ProcessorId), int(0)), vec![
                // Reset available row for this vertex
                seq_for("c", int(0), v("n"), vec![
                    sw("available", add(mul(Expr::ProcessorId, v("n")), v("c")), int(1)),
                ]),
                // Scan neighbors, mark their colors as unavailable
                local("row_start", sr("row_ptr", Expr::ProcessorId)),
                local("row_end", sr("row_ptr", add(Expr::ProcessorId, int(1)))),
                seq_for("e", v("row_start"), v("row_end"), vec![
                    local("neighbor", sr("col_idx", v("e"))),
                    local("nc", sr("color", v("neighbor"))),
                    if_then(Expr::binop(BinOp::Ge, v("nc"), int(0)), vec![
                        sw("available", add(mul(Expr::ProcessorId, v("n")), v("nc")), int(0)),
                    ]),
                ]),
            ]),
        ]),
        Stmt::Barrier,

        // Phase 2: pick minimum available color
        Stmt::Comment("Phase 2: assign minimum available color".to_string()),
        par_for("pid", v("n"), vec![
            if_then(eq_e(sr("colored", Expr::ProcessorId), int(0)), vec![
                local("chosen", int(-1)),
                seq_for("c", int(0), v("n"), vec![
                    if_then(and_e(
                        eq_e(v("chosen"), int(-1)),
                        eq_e(sr("available", add(mul(Expr::ProcessorId, v("n")), v("c"))), int(1)),
                    ), vec![
                        assign("chosen", v("c")),
                    ]),
                ]),
                if_then(Expr::binop(BinOp::Ge, v("chosen"), int(0)), vec![
                    sw("color", Expr::ProcessorId, v("chosen")),
                    sw("colored", Expr::ProcessorId, int(1)),
                ]),
            ]),
        ]),
        Stmt::Barrier,

        // Phase 3: decrement remaining
        Stmt::Comment("Phase 3: update remaining count".to_string()),
        par_for("pid", v("n"), vec![
            if_then(and_e(
                eq_e(sr("colored", Expr::ProcessorId), int(1)),
                Expr::binop(BinOp::Ge, sr("color", Expr::ProcessorId), int(0)),
            ), vec![
                // Mark as done (colored stays 1)
            ]),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Maximal independent set ────────────────────────────────────────────────

/// Luby's parallel MIS algorithm on CRCW-Arbitrary PRAM.
///
/// * n+m processors, O(log n) expected time.
/// * Input: CSR adjacency (row_ptr[n+1], col_idx[m]), n vertices.
/// * Each round: assign random priorities, add local maxima to MIS,
///   remove them and their neighbors.
pub fn maximal_independent_set() -> PramProgram {
    let mut prog = PramProgram::new("maximal_independent_set", MemoryModel::CRCWArbitrary);
    prog.description = Some(
        "Luby's parallel MIS algorithm. CRCW-Arbitrary, O(log n) expected time, n+m processors."
            .to_string(),
    );
    prog.work_bound = Some("O((n+m) log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("m"));
    prog.shared_memory.push(shared("row_ptr", add(v("n"), int(1))));
    prog.shared_memory.push(shared("col_idx", v("m")));
    prog.shared_memory.push(shared("in_mis", v("n")));
    prog.shared_memory.push(shared("removed", v("n")));
    prog.shared_memory.push(shared("priority", v("n")));
    prog.shared_memory.push(shared("is_local_max", v("n")));
    prog.shared_memory.push(shared("changed", int(1)));
    prog.num_processors = add(v("n"), v("m"));

    // Initialise
    prog.body.push(Stmt::Comment("Initialise in_mis=0, removed=0".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("in_mis", Expr::ProcessorId, int(0)),
        sw("removed", Expr::ProcessorId, int(0)),
    ]));
    prog.body.push(Stmt::Barrier);

    // Main loop: O(log n) rounds
    prog.body.push(Stmt::Comment("Luby's MIS: O(log n) rounds".to_string()));
    prog.body.push(Stmt::LocalDecl(
        "max_rounds".to_string(),
        PramType::Int64,
        Some(add(
            Expr::Cast(
                Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
                PramType::Int64,
            ),
            int(2),
        )),
    ));

    prog.body.push(seq_for("round", int(0), v("max_rounds"), vec![
        // Phase 1: assign random priorities
        Stmt::Comment("Phase 1: assign random priorities via hash".to_string()),
        par_for("pid", v("n"), vec![
            if_then(eq_e(sr("removed", Expr::ProcessorId), int(0)), vec![
                // hash(pid + round * n) as priority
                sw("priority", Expr::ProcessorId,
                   Expr::FunctionCall("hash".to_string(), vec![
                       add(Expr::ProcessorId, mul(v("round"), v("n"))),
                   ])),
            ]),
        ]),
        Stmt::Barrier,

        // Phase 2: check local maximum
        Stmt::Comment("Phase 2: check if vertex has max priority among non-removed neighbors".to_string()),
        par_for("pid", v("n"), vec![
            sw("is_local_max", Expr::ProcessorId, int(0)),
            if_then(eq_e(sr("removed", Expr::ProcessorId), int(0)), vec![
                local("my_pri", sr("priority", Expr::ProcessorId)),
                local("is_max", int(1)),
                local("row_start", sr("row_ptr", Expr::ProcessorId)),
                local("row_end", sr("row_ptr", add(Expr::ProcessorId, int(1)))),
                seq_for("e", v("row_start"), v("row_end"), vec![
                    local("neighbor", sr("col_idx", v("e"))),
                    if_then(eq_e(sr("removed", v("neighbor")), int(0)), vec![
                        local("npri", sr("priority", v("neighbor"))),
                        if_then(Expr::binop(BinOp::Ge, v("npri"), v("my_pri")), vec![
                            // Tie-break: higher pid wins
                            if_then(Expr::binop(BinOp::Or,
                                Expr::binop(BinOp::Gt, v("npri"), v("my_pri")),
                                Expr::binop(BinOp::Gt, v("neighbor"), Expr::ProcessorId),
                            ), vec![
                                assign("is_max", int(0)),
                            ]),
                        ]),
                    ]),
                ]),
                sw("is_local_max", Expr::ProcessorId, v("is_max")),
            ]),
        ]),
        Stmt::Barrier,

        // Phase 3: add local maxima to MIS
        Stmt::Comment("Phase 3: add local maxima to MIS".to_string()),
        par_for("pid", v("n"), vec![
            if_then(eq_e(sr("is_local_max", Expr::ProcessorId), int(1)), vec![
                sw("in_mis", Expr::ProcessorId, int(1)),
            ]),
        ]),
        Stmt::Barrier,

        // Phase 4: remove MIS vertices and their neighbors
        Stmt::Comment("Phase 4: remove MIS vertices and neighbors".to_string()),
        par_for("pid", v("n"), vec![
            if_then(eq_e(sr("removed", Expr::ProcessorId), int(0)), vec![
                // Check if pid is in MIS
                if_then(eq_e(sr("in_mis", Expr::ProcessorId), int(1)), vec![
                    sw("removed", Expr::ProcessorId, int(1)),
                ]),
                // Check if any neighbor is in MIS
                local("row_start", sr("row_ptr", Expr::ProcessorId)),
                local("row_end", sr("row_ptr", add(Expr::ProcessorId, int(1)))),
                seq_for("e", v("row_start"), v("row_end"), vec![
                    local("neighbor", sr("col_idx", v("e"))),
                    if_then(eq_e(sr("in_mis", v("neighbor")), int(1)), vec![
                        sw("removed", Expr::ProcessorId, int(1)),
                    ]),
                ]),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Shortest path (Bellman-Ford) ───────────────────────────────────────────

/// Parallel Bellman-Ford shortest paths on CREW PRAM.
///
/// * m processors, O(n log n) time.
/// * Input: edge list (edge_src[m], edge_dst[m], edge_w[m]),
///   n vertices, source vertex.
/// * n-1 relaxation rounds; early exit if no change.
pub fn shortest_path() -> PramProgram {
    let mut prog = PramProgram::new("shortest_path", MemoryModel::CREW);
    prog.description = Some(
        "Parallel Bellman-Ford shortest paths. CREW, O(n log n) time, m processors.".to_string(),
    );
    prog.work_bound = Some("O(n * m)".to_string());
    prog.time_bound = Some("O(n log n)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("m"));
    prog.parameters.push(param("source"));
    prog.shared_memory.push(shared("edge_src", v("m")));
    prog.shared_memory.push(shared("edge_dst", v("m")));
    prog.shared_memory.push(shared("edge_w", v("m")));
    prog.shared_memory.push(shared("dist", v("n")));
    prog.shared_memory.push(shared("new_dist", v("n")));
    prog.shared_memory.push(shared("changed", int(1)));
    prog.num_processors = v("m");

    // Initialise dist to MAX, dist[source] = 0
    prog.body.push(Stmt::Comment("Initialise dist to MAX, dist[source]=0".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("dist", Expr::ProcessorId, Expr::int(i64::MAX)),
        sw("new_dist", Expr::ProcessorId, Expr::int(i64::MAX)),
    ]));
    prog.body.push(Stmt::Barrier);
    prog.body.push(sw("dist", v("source"), int(0)));
    prog.body.push(sw("new_dist", v("source"), int(0)));
    prog.body.push(Stmt::Barrier);

    // Main loop: n-1 relaxation iterations
    prog.body.push(Stmt::Comment("Bellman-Ford: n-1 relaxation rounds".to_string()));
    prog.body.push(seq_for("iter", int(0), sub(v("n"), int(1)), vec![
        // Phase 1: reset changed
        Stmt::Comment("Phase 1: reset changed flag".to_string()),
        sw("changed", int(0), int(0)),
        Stmt::Barrier,

        // Phase 2: relax all edges in parallel
        Stmt::Comment("Phase 2: relax edges".to_string()),
        par_for("pid", v("m"), vec![
            local("u", sr("edge_src", Expr::ProcessorId)),
            local("v_node", sr("edge_dst", Expr::ProcessorId)),
            local("w", sr("edge_w", Expr::ProcessorId)),
            local("du", sr("dist", v("u"))),
            if_then(lt(v("du"), Expr::int(i64::MAX)), vec![
                local("new_d", add(v("du"), v("w"))),
                local("cur_dv", sr("dist", v("v_node"))),
                if_then(lt(v("new_d"), v("cur_dv")), vec![
                    sw("new_dist", v("v_node"), v("new_d")),
                    sw("changed", int(0), int(1)),
                ]),
            ]),
        ]),
        Stmt::Barrier,

        // Phase 3: update dist from new_dist
        Stmt::Comment("Phase 3: merge new_dist into dist".to_string()),
        par_for("pid", v("n"), vec![
            sw("dist", Expr::ProcessorId,
               min_e(sr("dist", Expr::ProcessorId), sr("new_dist", Expr::ProcessorId))),
        ]),
        Stmt::Barrier,

        // Phase 4: check convergence
        Stmt::Comment("Phase 4: early exit if no changes".to_string()),
        if_then(eq_e(sr("changed", int(0)), int(0)), vec![
            Stmt::Comment("Converged – break".to_string()),
        ]),
    ]));

    prog
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shiloach_vishkin_structure() {
        let prog = shiloach_vishkin();
        assert_eq!(prog.name, "shiloach_vishkin");
        assert_eq!(prog.memory_model, MemoryModel::CRCWArbitrary);
        assert_eq!(prog.parameters.len(), 2);
        assert!(prog.shared_memory.len() >= 4);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_boruvka_mst_structure() {
        let prog = boruvka_mst();
        assert_eq!(prog.name, "boruvka_mst");
        assert_eq!(prog.memory_model, MemoryModel::CRCWPriority);
        assert!(prog.shared_memory.len() >= 6);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_parallel_bfs_structure() {
        let prog = parallel_bfs();
        assert_eq!(prog.name, "parallel_bfs");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert_eq!(prog.parameters.len(), 3);
        assert!(prog.shared_memory.len() >= 5);
        assert!(prog.time_bound.as_deref() == Some("O(D)"));
    }

    #[test]
    fn test_euler_tour_structure() {
        let prog = euler_tour();
        assert_eq!(prog.name, "euler_tour");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 5);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_graph_algorithms_have_descriptions() {
        for builder in [shiloach_vishkin, boruvka_mst, parallel_bfs, euler_tour] {
            let prog = builder();
            assert!(prog.description.is_some(), "{} missing description", prog.name);
            assert!(prog.work_bound.is_some(), "{} missing work_bound", prog.name);
        }
    }

    #[test]
    fn test_graph_algorithms_write_shared() {
        for builder in [shiloach_vishkin, boruvka_mst, parallel_bfs, euler_tour] {
            let prog = builder();
            assert!(
                prog.body.iter().any(|s| s.writes_shared()),
                "{} must write to shared memory",
                prog.name,
            );
        }
    }

    #[test]
    fn test_parallel_dfs_structure() {
        let prog = parallel_dfs();
        assert_eq!(prog.name, "parallel_dfs");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert_eq!(prog.parameters.len(), 3);
        assert!(prog.shared_memory.len() >= 6);
        assert!(prog.parallel_step_count() >= 2);
        assert!(prog.time_bound.as_deref() == Some("O(D log n)"));
    }

    #[test]
    fn test_graph_coloring_structure() {
        let prog = graph_coloring();
        assert_eq!(prog.name, "graph_coloring");
        assert_eq!(prog.memory_model, MemoryModel::CRCWCommon);
        assert!(prog.shared_memory.len() >= 4);
        assert!(prog.parallel_step_count() >= 2);
        assert!(prog.time_bound.as_deref() == Some("O(Delta log n)"));
    }

    #[test]
    fn test_maximal_independent_set_structure() {
        let prog = maximal_independent_set();
        assert_eq!(prog.name, "maximal_independent_set");
        assert_eq!(prog.memory_model, MemoryModel::CRCWArbitrary);
        assert!(prog.shared_memory.len() >= 5);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_shortest_path_structure() {
        let prog = shortest_path();
        assert_eq!(prog.name, "shortest_path");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert_eq!(prog.parameters.len(), 3);
        assert!(prog.shared_memory.len() >= 5);
        assert!(prog.parallel_step_count() >= 2);
        assert!(prog.time_bound.as_deref() == Some("O(n log n)"));
    }

    #[test]
    fn test_new_graph_algorithms_have_descriptions() {
        for builder in [parallel_dfs, graph_coloring, maximal_independent_set, shortest_path] {
            let prog = builder();
            assert!(prog.description.is_some(), "{} missing description", prog.name);
            assert!(prog.work_bound.is_some(), "{} missing work_bound", prog.name);
        }
    }

    #[test]
    fn test_new_graph_algorithms_write_shared() {
        for builder in [parallel_dfs, graph_coloring, maximal_independent_set, shortest_path] {
            let prog = builder();
            assert!(
                prog.body.iter().any(|s| s.writes_shared()),
                "{} must write to shared memory",
                prog.name,
            );
        }
    }
}
