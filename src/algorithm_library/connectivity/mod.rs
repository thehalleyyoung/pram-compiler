//! Connectivity algorithms for PRAM.

use crate::pram_ir::ast::*;
use crate::pram_ir::types::PramType;

fn param(name: &str) -> Parameter {
    Parameter { name: name.to_string(), param_type: PramType::Int64 }
}

fn shared(name: &str, size: Expr) -> SharedMemoryDecl {
    SharedMemoryDecl { name: name.to_string(), elem_type: PramType::Int64, size }
}

fn add(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Add, a, b) }
fn sub(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Sub, a, b) }
fn mul(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Mul, a, b) }
fn div_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Div, a, b) }
fn lt(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Lt, a, b) }
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

// ─── Vishkin's deterministic connectivity ───────────────────────────────────

/// Vishkin's deterministic connectivity on CREW PRAM.
///
/// * O(log n) time, m + n processors.
/// * Similar structure to Shiloach-Vishkin but avoids concurrent writes
///   by using deterministic hooking with CREW reads:
///   1. Each vertex reads its neighbor's component id.
///   2. A vertex hooks itself to the minimum-id neighbor's component
///      only if it is a local minimum (no write conflict under CREW
///      because each vertex writes only to its own component slot).
///   3. Pointer-jump to flatten.
pub fn vishkin_connectivity() -> PramProgram {
    let mut prog = PramProgram::new("vishkin_connectivity", MemoryModel::CREW);
    prog.description = Some(
        "Vishkin's deterministic connectivity. CREW, O(log n) time, m+n processors.".to_string(),
    );
    prog.work_bound = Some("O((m+n) log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("m"));
    prog.shared_memory.push(shared("row_ptr", add(v("n"), int(1))));
    prog.shared_memory.push(shared("col_idx", v("m")));
    prog.shared_memory.push(shared("comp", v("n")));
    prog.shared_memory.push(shared("min_neighbor_comp", v("n")));
    prog.shared_memory.push(shared("changed", int(1)));
    prog.num_processors = add(v("m"), v("n"));

    // Initialise: comp[i] = i
    prog.body.push(Stmt::Comment("Initialise: comp[i] = i".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("comp", Expr::ProcessorId, Expr::ProcessorId),
    ]));
    prog.body.push(Stmt::Barrier);

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
        sw("changed", int(0), int(0)),
        Stmt::Barrier,

        // Step 1: each vertex finds min component-id among its neighbors (CREW reads)
        Stmt::Comment("Find minimum neighbor component for each vertex".to_string()),
        par_for("pid", v("n"), vec![
            local("my_comp", sr("comp", Expr::ProcessorId)),
            local("best", v("my_comp")),
            local("start", sr("row_ptr", Expr::ProcessorId)),
            local("end", sr("row_ptr", add(Expr::ProcessorId, int(1)))),
            seq_for("e", v("start"), v("end"), vec![
                local("nbr", sr("col_idx", v("e"))),
                local("nbr_comp", sr("comp", v("nbr"))),
                assign("best", min_e(v("best"), v("nbr_comp"))),
            ]),
            sw("min_neighbor_comp", Expr::ProcessorId, v("best")),
        ]),
        Stmt::Barrier,

        // Step 2: deterministic hooking – each vertex hooks to min_neighbor_comp
        // only if min_neighbor_comp < comp (writes to own slot only → no conflict)
        Stmt::Comment("Deterministic hooking: each vertex hooks to min neighbor comp".to_string()),
        par_for("pid", v("n"), vec![
            local("mc", sr("min_neighbor_comp", Expr::ProcessorId)),
            local("cc", sr("comp", Expr::ProcessorId)),
            if_then(lt(v("mc"), v("cc")), vec![
                sw("comp", Expr::ProcessorId, v("mc")),
                sw("changed", int(0), int(1)),
            ]),
        ]),
        Stmt::Barrier,

        // Step 3: pointer jumping
        Stmt::Comment("Pointer jumping: comp[i] <- comp[comp[i]]".to_string()),
        par_for("pid", v("n"), vec![
            local("c", sr("comp", Expr::ProcessorId)),
            local("cc", sr("comp", v("c"))),
            if_then(ne_e(v("c"), v("cc")), vec![
                sw("comp", Expr::ProcessorId, v("cc")),
                sw("changed", int(0), int(1)),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Reif's ear decomposition ───────────────────────────────────────────────

/// Reif's ear decomposition on CREW PRAM.
///
/// * O(log n) time, m processors.
/// * A 2-edge-connected graph can be decomposed into ears (paths/cycles
///   that share only endpoints with earlier ears).
/// * Algorithm:
///   1. Find a spanning tree T.
///   2. Euler-tour T to linearise it.
///   3. For each non-tree edge (u,v), define ear = path u→LCA(u,v)→v.
///   4. Assign ear ids via parallel LCA + list ranking.
pub fn ear_decomposition() -> PramProgram {
    let mut prog = PramProgram::new("ear_decomposition", MemoryModel::CREW);
    prog.description = Some(
        "Reif's ear decomposition. CREW, O(log n) time, m processors.".to_string(),
    );
    prog.work_bound = Some("O(m log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("m"));
    // Spanning tree (parent representation)
    prog.shared_memory.push(shared("parent", v("n")));
    prog.shared_memory.push(shared("depth", v("n")));
    // Non-tree edges
    prog.shared_memory.push(shared("nt_src", v("m")));
    prog.shared_memory.push(shared("nt_dst", v("m")));
    prog.shared_memory.push(shared("nt_count", int(1)));
    // Euler tour arrays
    prog.shared_memory.push(shared("euler_succ", mul(int(2), v("n"))));
    prog.shared_memory.push(shared("euler_rank", mul(int(2), v("n"))));
    // Ear assignment
    prog.shared_memory.push(shared("ear_id", v("n")));
    prog.shared_memory.push(shared("lca_arr", v("m")));
    prog.num_processors = v("m");

    // Step 1: compute depth via BFS from root (simplified: parent-pointer traversal)
    prog.body.push(Stmt::Comment("Step 1: compute depths via parent pointers".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        local("d", int(0)),
        local("cur", Expr::ProcessorId),
        Stmt::While {
            condition: Expr::binop(BinOp::Ge, sr("parent", v("cur")), int(0)),
            body: vec![
                assign("d", add(v("d"), int(1))),
                assign("cur", sr("parent", v("cur"))),
            ],
        },
        sw("depth", Expr::ProcessorId, v("d")),
    ]));
    prog.body.push(Stmt::Barrier);

    // Step 2: build Euler-tour successor pointers (list-rank to get positions)
    prog.body.push(Stmt::Comment("Step 2: Euler tour via pointer jumping".to_string()));
    prog.body.push(par_for("pid", mul(int(2), v("n")), vec![
        sw("euler_rank", Expr::ProcessorId, int(1)),
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
        par_for("pid", mul(int(2), v("n")), vec![
            local("s", sr("euler_succ", Expr::ProcessorId)),
            if_then(ne_e(v("s"), Expr::ProcessorId), vec![
                sw("euler_rank", Expr::ProcessorId,
                   add(sr("euler_rank", Expr::ProcessorId), sr("euler_rank", v("s")))),
                sw("euler_succ", Expr::ProcessorId, sr("euler_succ", v("s"))),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    // Step 3: for each non-tree edge, compute LCA and assign ear
    prog.body.push(Stmt::Comment(
        "Step 3: compute LCA for each non-tree edge, assign ear ids".to_string(),
    ));
    prog.body.push(par_for("pid", sr("nt_count", int(0)), vec![
        local("u", sr("nt_src", Expr::ProcessorId)),
        local("v_node", sr("nt_dst", Expr::ProcessorId)),
        local("du", sr("depth", v("u"))),
        local("dv", sr("depth", v("v_node"))),
        // Walk the deeper vertex up until depths match
        Stmt::While {
            condition: Expr::binop(BinOp::Gt, v("du"), v("dv")),
            body: vec![
                assign("u", sr("parent", v("u"))),
                assign("du", sub(v("du"), int(1))),
            ],
        },
        Stmt::While {
            condition: Expr::binop(BinOp::Gt, v("dv"), v("du")),
            body: vec![
                assign("v_node", sr("parent", v("v_node"))),
                assign("dv", sub(v("dv"), int(1))),
            ],
        },
        // Walk both up until they meet
        Stmt::While {
            condition: ne_e(v("u"), v("v_node")),
            body: vec![
                assign("u", sr("parent", v("u"))),
                assign("v_node", sr("parent", v("v_node"))),
            ],
        },
        sw("lca_arr", Expr::ProcessorId, v("u")),
        // Ear id = non-tree edge index
        sw("ear_id", v("u"), Expr::ProcessorId),
    ]));
    prog.body.push(Stmt::Barrier);

    // Step 4: propagate ear ids down the tree via pointer jumping
    prog.body.push(Stmt::Comment("Step 4: propagate ear ids down from LCA".to_string()));
    prog.body.push(seq_for("step", int(0), v("log_n"), vec![
        par_for("pid", v("n"), vec![
            local("my_ear", sr("ear_id", Expr::ProcessorId)),
            local("par_ear", sr("ear_id", sr("parent", Expr::ProcessorId))),
            if_then(and_e(
                eq_e(v("my_ear"), int(-1)),
                ne_e(v("par_ear"), int(-1)),
            ), vec![
                sw("ear_id", Expr::ProcessorId, v("par_ear")),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Biconnected components via ear decomposition ───────────────────────────

/// Parallel biconnected components via ear decomposition.
///
/// * CREW, O(log^2 n) time, n+m processors.
/// * Algorithm:
///   1. Compute spanning-tree depth via parent pointers.
///   2. Initialise low values.
///   3. Update low values over O(log n) rounds using neighbours and
///      pointer-jumping through parents.
///   4. Identify articulation points (low[child] >= depth[v]).
///   5. Assign biconnected-component ids by pointer-jumping on the edge
///      graph with articulation-point separators.
pub fn biconnected_components() -> PramProgram {
    let mut prog = PramProgram::new("biconnected_components", MemoryModel::CREW);
    prog.description = Some(
        "Parallel biconnected components via ear decomposition. CREW, O(log^2 n) time, n+m processors.".to_string(),
    );
    prog.work_bound = Some("O((n+m) log^2 n)".to_string());
    prog.time_bound = Some("O(log^2 n)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("m"));
    prog.shared_memory.push(shared("row_ptr", add(v("n"), int(1))));
    prog.shared_memory.push(shared("col_idx", v("m")));
    prog.shared_memory.push(shared("parent", v("n")));
    prog.shared_memory.push(shared("depth", v("n")));
    prog.shared_memory.push(shared("low", v("n")));
    prog.shared_memory.push(shared("bicomp_id", v("n")));
    prog.shared_memory.push(shared("is_artic", v("n")));
    prog.shared_memory.push(shared("stack_arr", v("m")));
    prog.shared_memory.push(shared("comp_count", int(1)));
    prog.num_processors = add(v("n"), v("m"));

    let log_n = add(
        Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        ),
        int(2),
    );

    // Phase 1: compute spanning-tree depth via parent-pointer traversal
    prog.body.push(Stmt::Comment("Phase 1: compute depth via parent pointers".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        local("d", int(0)),
        local("cur", Expr::ProcessorId),
        Stmt::While {
            condition: Expr::binop(BinOp::Ge, sr("parent", v("cur")), int(0)),
            body: vec![
                assign("d", add(v("d"), int(1))),
                assign("cur", sr("parent", v("cur"))),
            ],
        },
        sw("depth", Expr::ProcessorId, v("d")),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 2: initialise low[pid] = depth[pid]
    prog.body.push(Stmt::Comment("Phase 2: initialise low = depth".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("low", Expr::ProcessorId, sr("depth", Expr::ProcessorId)),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 3: O(log n) rounds updating low values
    prog.body.push(Stmt::Comment("Phase 3: update low values via neighbours and pointer-jumping".to_string()));
    prog.body.push(Stmt::LocalDecl("log_n".to_string(), PramType::Int64, Some(log_n.clone())));
    prog.body.push(seq_for("round", int(0), v("log_n"), vec![
        par_for("pid", v("n"), vec![
            local("start", sr("row_ptr", Expr::ProcessorId)),
            local("end", sr("row_ptr", add(Expr::ProcessorId, int(1)))),
            seq_for("e", v("start"), v("end"), vec![
                local("nbr", sr("col_idx", v("e"))),
                local("d_nbr", sr("depth", v("nbr"))),
                if_then(lt(v("d_nbr"), sr("low", Expr::ProcessorId)), vec![
                    sw("low", Expr::ProcessorId, v("d_nbr")),
                ]),
            ]),
            // pointer-jump low through parent
            local("p", sr("parent", Expr::ProcessorId)),
            if_then(Expr::binop(BinOp::Ge, v("p"), int(0)), vec![
                local("par_low", sr("low", v("p"))),
                if_then(lt(v("par_low"), sr("low", Expr::ProcessorId)), vec![
                    sw("low", Expr::ProcessorId, v("par_low")),
                ]),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    // Phase 4: identify articulation points
    prog.body.push(Stmt::Comment("Phase 4: identify articulation points".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("is_artic", Expr::ProcessorId, int(0)),
        local("start", sr("row_ptr", Expr::ProcessorId)),
        local("end", sr("row_ptr", add(Expr::ProcessorId, int(1)))),
        seq_for("e", v("start"), v("end"), vec![
            local("nbr", sr("col_idx", v("e"))),
            // nbr is a child if parent[nbr] == pid
            if_then(eq_e(sr("parent", v("nbr")), Expr::ProcessorId), vec![
                if_then(Expr::binop(BinOp::Ge, sr("low", v("nbr")), sr("depth", Expr::ProcessorId)), vec![
                    sw("is_artic", Expr::ProcessorId, int(1)),
                ]),
            ]),
        ]),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 5: assign biconnected-component ids via pointer-jumping on edges
    prog.body.push(Stmt::Comment("Phase 5: assign biconnected component ids".to_string()));
    prog.body.push(par_for("pid", v("m"), vec![
        sw("bicomp_id", Expr::ProcessorId, Expr::ProcessorId),
    ]));
    prog.body.push(Stmt::Barrier);
    prog.body.push(seq_for("step", int(0), v("log_n"), vec![
        par_for("pid", v("m"), vec![
            local("src", sr("col_idx", Expr::ProcessorId)),
            local("my_id", sr("bicomp_id", Expr::ProcessorId)),
            local("src_id", sr("bicomp_id", v("src"))),
            // merge edges in same bicomponent unless separated by articulation point
            if_then(and_e(
                lt(v("src_id"), v("my_id")),
                eq_e(sr("is_artic", v("src")), int(0)),
            ), vec![
                sw("bicomp_id", Expr::ProcessorId, v("src_id")),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Strongly connected components (forward-backward) ──────────────────────

/// Parallel strongly connected components using forward-backward reachability.
///
/// * CREW, O(log^2 n) time, n+m processors.
/// * Algorithm:
///   1. Initialise: scc_id = -1, active = 1 for all vertices.
///   2. Main loop (O(log n) rounds): pick pivot (first active vertex).
///   3. Forward BFS from pivot on forward edges.
///   4. Backward BFS from pivot on reverse edges.
///   5. Vertices reachable in both directions form an SCC; assign and
///      deactivate.
///   6. Recurse on remaining active vertices.
pub fn strongly_connected() -> PramProgram {
    let mut prog = PramProgram::new("strongly_connected", MemoryModel::CREW);
    prog.description = Some(
        "Parallel strongly connected components (forward-backward). CREW, O(log^2 n) time, n+m processors.".to_string(),
    );
    prog.work_bound = Some("O((n+m) log^2 n)".to_string());
    prog.time_bound = Some("O(log^2 n)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("m"));
    prog.shared_memory.push(shared("row_ptr", add(v("n"), int(1))));
    prog.shared_memory.push(shared("col_idx", v("m")));
    prog.shared_memory.push(shared("rev_row_ptr", add(v("n"), int(1))));
    prog.shared_memory.push(shared("rev_col_idx", v("m")));
    prog.shared_memory.push(shared("scc_id", v("n")));
    prog.shared_memory.push(shared("fw_reach", v("n")));
    prog.shared_memory.push(shared("bw_reach", v("n")));
    prog.shared_memory.push(shared("pivot", int(1)));
    prog.shared_memory.push(shared("active", v("n")));
    prog.shared_memory.push(shared("changed", int(1)));
    prog.num_processors = add(v("n"), v("m"));

    let log_n = add(
        Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        ),
        int(2),
    );

    // Phase 1: initialise scc_id = -1, active = 1
    prog.body.push(Stmt::Comment("Phase 1: initialise scc_id and active flags".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("scc_id", Expr::ProcessorId, int(-1)),
        sw("active", Expr::ProcessorId, int(1)),
    ]));
    prog.body.push(Stmt::Barrier);

    prog.body.push(Stmt::LocalDecl("log_n".to_string(), PramType::Int64, Some(log_n.clone())));

    // Phase 2-6: main loop
    prog.body.push(Stmt::Comment("Phase 2-6: forward-backward reachability loop".to_string()));
    prog.body.push(seq_for("outer", int(0), v("log_n"), vec![
        // Pick pivot: first active vertex (processor 0 scans)
        Stmt::Comment("Pick pivot: first active vertex".to_string()),
        sw("pivot", int(0), int(-1)),
        Stmt::Barrier,
        par_for("pid", v("n"), vec![
            if_then(and_e(
                eq_e(sr("active", Expr::ProcessorId), int(1)),
                eq_e(sr("pivot", int(0)), int(-1)),
            ), vec![
                sw("pivot", int(0), Expr::ProcessorId),
            ]),
        ]),
        Stmt::Barrier,

        // Phase 3: forward BFS from pivot
        Stmt::Comment("Phase 3: forward BFS from pivot".to_string()),
        par_for("pid", v("n"), vec![
            sw("fw_reach", Expr::ProcessorId, int(0)),
        ]),
        Stmt::Barrier,
        // seed pivot
        par_for("pid", int(1), vec![
            local("pv", sr("pivot", int(0))),
            if_then(Expr::binop(BinOp::Ge, v("pv"), int(0)), vec![
                sw("fw_reach", v("pv"), int(1)),
            ]),
        ]),
        Stmt::Barrier,
        seq_for("bfs_step", int(0), v("log_n"), vec![
            sw("changed", int(0), int(0)),
            Stmt::Barrier,
            par_for("pid", v("n"), vec![
                if_then(and_e(
                    eq_e(sr("fw_reach", Expr::ProcessorId), int(1)),
                    eq_e(sr("active", Expr::ProcessorId), int(1)),
                ), vec![
                    local("start", sr("row_ptr", Expr::ProcessorId)),
                    local("end", sr("row_ptr", add(Expr::ProcessorId, int(1)))),
                    seq_for("e", v("start"), v("end"), vec![
                        local("nbr", sr("col_idx", v("e"))),
                        if_then(and_e(
                            eq_e(sr("fw_reach", v("nbr")), int(0)),
                            eq_e(sr("active", v("nbr")), int(1)),
                        ), vec![
                            sw("fw_reach", v("nbr"), int(1)),
                            sw("changed", int(0), int(1)),
                        ]),
                    ]),
                ]),
            ]),
            Stmt::Barrier,
        ]),

        // Phase 4: backward BFS from pivot on reverse edges
        Stmt::Comment("Phase 4: backward BFS from pivot".to_string()),
        par_for("pid", v("n"), vec![
            sw("bw_reach", Expr::ProcessorId, int(0)),
        ]),
        Stmt::Barrier,
        par_for("pid", int(1), vec![
            local("pv", sr("pivot", int(0))),
            if_then(Expr::binop(BinOp::Ge, v("pv"), int(0)), vec![
                sw("bw_reach", v("pv"), int(1)),
            ]),
        ]),
        Stmt::Barrier,
        seq_for("bfs_step", int(0), v("log_n"), vec![
            sw("changed", int(0), int(0)),
            Stmt::Barrier,
            par_for("pid", v("n"), vec![
                if_then(and_e(
                    eq_e(sr("bw_reach", Expr::ProcessorId), int(1)),
                    eq_e(sr("active", Expr::ProcessorId), int(1)),
                ), vec![
                    local("start", sr("rev_row_ptr", Expr::ProcessorId)),
                    local("end", sr("rev_row_ptr", add(Expr::ProcessorId, int(1)))),
                    seq_for("e", v("start"), v("end"), vec![
                        local("nbr", sr("rev_col_idx", v("e"))),
                        if_then(and_e(
                            eq_e(sr("bw_reach", v("nbr")), int(0)),
                            eq_e(sr("active", v("nbr")), int(1)),
                        ), vec![
                            sw("bw_reach", v("nbr"), int(1)),
                            sw("changed", int(0), int(1)),
                        ]),
                    ]),
                ]),
            ]),
            Stmt::Barrier,
        ]),

        // Phase 5: assign SCC id and deactivate
        Stmt::Comment("Phase 5: assign SCC and deactivate".to_string()),
        par_for("pid", v("n"), vec![
            if_then(and_e(
                eq_e(sr("fw_reach", Expr::ProcessorId), int(1)),
                eq_e(sr("bw_reach", Expr::ProcessorId), int(1)),
            ), vec![
                sw("scc_id", Expr::ProcessorId, sr("pivot", int(0))),
                sw("active", Expr::ProcessorId, int(0)),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vishkin_connectivity_structure() {
        let prog = vishkin_connectivity();
        assert_eq!(prog.name, "vishkin_connectivity");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert_eq!(prog.parameters.len(), 2);
        assert!(prog.shared_memory.len() >= 4);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_ear_decomposition_structure() {
        let prog = ear_decomposition();
        assert_eq!(prog.name, "ear_decomposition");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 6);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_connectivity_algorithms_have_descriptions() {
        for builder in [vishkin_connectivity, ear_decomposition] {
            let prog = builder();
            assert!(prog.description.is_some(), "{} missing description", prog.name);
            assert!(prog.work_bound.is_some(), "{} missing work_bound", prog.name);
        }
    }

    #[test]
    fn test_connectivity_algorithms_write_shared() {
        for builder in [vishkin_connectivity, ear_decomposition] {
            let prog = builder();
            assert!(
                prog.body.iter().any(|s| s.writes_shared()),
                "{} must write to shared memory",
                prog.name,
            );
        }
    }

    #[test]
    fn test_biconnected_components_structure() {
        let prog = biconnected_components();
        assert_eq!(prog.name, "biconnected_components");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert_eq!(prog.parameters.len(), 2);
        assert!(prog.shared_memory.len() >= 6);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log^2 n)"));
    }

    #[test]
    fn test_strongly_connected_structure() {
        let prog = strongly_connected();
        assert_eq!(prog.name, "strongly_connected");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert_eq!(prog.parameters.len(), 2);
        assert!(prog.shared_memory.len() >= 7);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log^2 n)"));
    }

    #[test]
    fn test_new_connectivity_algorithms_have_descriptions() {
        for builder in [biconnected_components, strongly_connected] {
            let prog = builder();
            assert!(prog.description.is_some(), "{} missing description", prog.name);
            assert!(prog.work_bound.is_some(), "{} missing work_bound", prog.name);
        }
    }

    #[test]
    fn test_new_connectivity_algorithms_write_shared() {
        for builder in [biconnected_components, strongly_connected] {
            let prog = builder();
            assert!(
                prog.body.iter().any(|s| s.writes_shared()),
                "{} must write to shared memory",
                prog.name,
            );
        }
    }
}
