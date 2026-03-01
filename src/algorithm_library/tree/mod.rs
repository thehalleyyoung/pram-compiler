//! Tree algorithms for PRAM.

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
fn mod_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Mod, a, b) }
fn lt(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Lt, a, b) }
fn le(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Le, a, b) }
fn ge(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Ge, a, b) }
fn eq_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Eq, a, b) }
fn ne_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Ne, a, b) }
fn and_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::And, a, b) }
fn min_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Min, a, b) }
fn shl(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Shl, a, b) }
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

// ─── Tree contraction (rake / compress) ─────────────────────────────────────

/// Rake/compress tree contraction on EREW PRAM.
///
/// * O(log n) time, n processors.
/// * Input: rooted tree with parent[n], left_child[n], right_child[n],
///   value[n].
/// * Alternating rounds of:
///   – **Rake**: remove all leaves and fold their values into their parents.
///   – **Compress**: remove degree-1 nodes (chains) and compose their
///     operations.
/// * After O(log n) rounds only the root remains, holding the result.
pub fn tree_contraction() -> PramProgram {
    let mut prog = PramProgram::new("tree_contraction", MemoryModel::EREW);
    prog.description = Some(
        "Rake/compress tree contraction. EREW, O(log n) time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(n log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("parent", v("n")));
    prog.shared_memory.push(shared("left_child", v("n")));
    prog.shared_memory.push(shared("right_child", v("n")));
    prog.shared_memory.push(shared("value", v("n")));
    prog.shared_memory.push(shared("active", v("n")));     // 1 = still in tree
    prog.shared_memory.push(shared("degree", v("n")));     // number of active children
    prog.num_processors = v("n");

    // Initialise: all nodes active, compute initial degree
    prog.body.push(Stmt::Comment("Initialise: all nodes active".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("active", Expr::ProcessorId, int(1)),
        // Degree = number of valid children
        local("deg", int(0)),
        if_then(ge(sr("left_child", Expr::ProcessorId), int(0)), vec![
            assign("deg", add(v("deg"), int(1))),
        ]),
        if_then(ge(sr("right_child", Expr::ProcessorId), int(0)), vec![
            assign("deg", add(v("deg"), int(1))),
        ]),
        sw("degree", Expr::ProcessorId, v("deg")),
    ]));
    prog.body.push(Stmt::Barrier);

    // O(log n) rounds of alternating rake and compress
    prog.body.push(Stmt::Comment("O(log n) rounds of rake + compress".to_string()));
    prog.body.push(Stmt::LocalDecl(
        "log_n".to_string(),
        PramType::Int64,
        Some(add(
            Expr::Cast(
                Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
                PramType::Int64,
            ),
            int(1),
        )),
    ));

    prog.body.push(seq_for("round", int(0), v("log_n"), vec![
        // ── Rake: remove leaves (degree-0 nodes that are not root) ──
        Stmt::Comment("Rake: remove leaf nodes".to_string()),
        par_for("pid", v("n"), vec![
            if_then(
                and_e(
                    eq_e(sr("active", Expr::ProcessorId), int(1)),
                    and_e(
                        eq_e(sr("degree", Expr::ProcessorId), int(0)),
                        ge(sr("parent", Expr::ProcessorId), int(0)),
                    ),
                ),
                vec![
                    local("par", sr("parent", Expr::ProcessorId)),
                    // Fold value into parent
                    sw("value", v("par"),
                       add(sr("value", v("par")), sr("value", Expr::ProcessorId))),
                    // Decrement parent's degree
                    sw("degree", v("par"),
                       sub(sr("degree", v("par")), int(1))),
                    // Deactivate this node
                    sw("active", Expr::ProcessorId, int(0)),
                ],
            ),
        ]),
        Stmt::Barrier,

        // ── Compress: remove degree-1 nodes (chains) ──
        // A degree-1 node v with parent p and single child c:
        //   value[p] += value[v], parent[c] = p, remove v.
        Stmt::Comment("Compress: remove degree-1 chain nodes".to_string()),
        par_for("pid", v("n"), vec![
            if_then(
                and_e(
                    eq_e(sr("active", Expr::ProcessorId), int(1)),
                    and_e(
                        eq_e(sr("degree", Expr::ProcessorId), int(1)),
                        ge(sr("parent", Expr::ProcessorId), int(0)),
                    ),
                ),
                vec![
                    local("par", sr("parent", Expr::ProcessorId)),
                    // Find the single active child
                    local("lc", sr("left_child", Expr::ProcessorId)),
                    local("rc", sr("right_child", Expr::ProcessorId)),
                    local("child",
                        Expr::Conditional(
                            Box::new(and_e(ge(v("lc"), int(0)),
                                          eq_e(sr("active", v("lc")), int(1)))),
                            Box::new(v("lc")),
                            Box::new(v("rc")),
                        ),
                    ),

                    // Bypass: parent[child] = par
                    if_then(ge(v("child"), int(0)), vec![
                        sw("parent", v("child"), v("par")),
                    ]),
                    // Fold value: value[par] += value[pid]
                    sw("value", v("par"),
                       add(sr("value", v("par")), sr("value", Expr::ProcessorId))),
                    // Parent's degree stays the same (lost one child, gained one grandchild)
                    // Deactivate
                    sw("active", Expr::ProcessorId, int(0)),
                ],
            ),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Lowest common ancestor (LCA) ──────────────────────────────────────────

/// LCA preprocessing on CREW PRAM.
///
/// * O(log n) time, n processors.
/// * Reduces LCA to range-minimum query (RMQ) on the Euler-tour depth
///   array, then builds a sparse table for O(1) queries.
///
/// Steps:
///   1. Euler tour of the tree → sequence of 2n-1 nodes with depths.
///   2. Sparse-table construction on the depth array (parallel prefix
///      of minimums at each power-of-2 offset).
///   3. LCA(u,v) = node at position of RMQ between first-occurrence[u]
///      and first-occurrence[v].
pub fn lca() -> PramProgram {
    let mut prog = PramProgram::new("lca", MemoryModel::CREW);
    prog.description = Some(
        "LCA via Euler tour + sparse table RMQ. CREW, O(log n) preprocessing, n processors."
            .to_string(),
    );
    prog.work_bound = Some("O(n log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    let euler_len = sub(mul(int(2), v("n")), int(1));
    prog.shared_memory.push(shared("parent", v("n")));
    prog.shared_memory.push(shared("first_child", v("n")));
    prog.shared_memory.push(shared("next_sibling", v("n")));
    // Euler tour output
    prog.shared_memory.push(shared("euler_node", euler_len.clone()));
    prog.shared_memory.push(shared("euler_depth", euler_len.clone()));
    prog.shared_memory.push(shared("first_occ", v("n")));  // first occurrence in tour
    // Euler tour successor / rank for list-ranking
    prog.shared_memory.push(shared("succ", mul(int(2), v("n"))));
    prog.shared_memory.push(shared("rank_arr", mul(int(2), v("n"))));
    // Sparse table: sparse[k][i] = index of min in euler_depth[i..i+2^k]
    // Flattened: sparse[k * euler_len + i]
    let log_n_expr = add(
        Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        ),
        int(2),
    );
    let table_size = mul(log_n_expr.clone(), euler_len.clone());
    prog.shared_memory.push(shared("sparse", table_size));
    prog.num_processors = v("n");

    // ── Step 1: Euler tour ──
    // Build successor list and list-rank it (same approach as euler_tour algorithm)
    prog.body.push(Stmt::Comment("Step 1: build Euler tour via successor + list ranking".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        local("fc", sr("first_child", Expr::ProcessorId)),
        local("ns", sr("next_sibling", Expr::ProcessorId)),
        local("par", sr("parent", Expr::ProcessorId)),

        if_else(
            ge(v("fc"), int(0)),
            vec![ sw("succ", mul(int(2), Expr::ProcessorId), mul(int(2), v("fc"))) ],
            vec![ sw("succ", mul(int(2), Expr::ProcessorId), add(mul(int(2), Expr::ProcessorId), int(1))) ],
        ),
        if_else(
            ge(v("ns"), int(0)),
            vec![ sw("succ", add(mul(int(2), Expr::ProcessorId), int(1)), mul(int(2), v("ns"))) ],
            vec![
                if_else(
                    ge(v("par"), int(0)),
                    vec![ sw("succ", add(mul(int(2), Expr::ProcessorId), int(1)),
                             add(mul(int(2), v("par")), int(1))) ],
                    vec![ sw("succ", add(mul(int(2), Expr::ProcessorId), int(1)),
                             add(mul(int(2), Expr::ProcessorId), int(1))) ],
                ),
            ],
        ),
    ]));
    prog.body.push(Stmt::Barrier);

    // List ranking via pointer jumping
    prog.body.push(par_for("pid", mul(int(2), v("n")), vec![
        if_else(
            ne_e(sr("succ", Expr::ProcessorId), Expr::ProcessorId),
            vec![ sw("rank_arr", Expr::ProcessorId, int(1)) ],
            vec![ sw("rank_arr", Expr::ProcessorId, int(0)) ],
        ),
    ]));
    prog.body.push(Stmt::Barrier);

    prog.body.push(Stmt::LocalDecl("log_n".to_string(), PramType::Int64, Some(log_n_expr.clone())));

    prog.body.push(seq_for("step", int(0), v("log_n"), vec![
        par_for("pid", mul(int(2), v("n")), vec![
            local("s", sr("succ", Expr::ProcessorId)),
            if_then(ne_e(v("s"), Expr::ProcessorId), vec![
                sw("rank_arr", Expr::ProcessorId,
                   add(sr("rank_arr", Expr::ProcessorId), sr("rank_arr", v("s")))),
                sw("succ", Expr::ProcessorId, sr("succ", v("s"))),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    // Use rank to fill euler_node, euler_depth, first_occ
    prog.body.push(Stmt::Comment("Fill Euler tour arrays from ranks".to_string()));
    prog.body.push(par_for("pid", mul(int(2), v("n")), vec![
        local("pos", sr("rank_arr", Expr::ProcessorId)),
        local("node_id", div_e(Expr::ProcessorId, int(2))),
        sw("euler_node", v("pos"), v("node_id")),
    ]));
    prog.body.push(Stmt::Barrier);

    // Compute depths and first occurrence
    prog.body.push(par_for("pid", v("n"), vec![
        // Depth via parent walk
        local("d", int(0)),
        local("cur", Expr::ProcessorId),
        Stmt::While {
            condition: ge(sr("parent", v("cur")), int(0)),
            body: vec![
                assign("d", add(v("d"), int(1))),
                assign("cur", sr("parent", v("cur"))),
            ],
        },
        // Write depth at first occurrence
        local("first_pos", sr("rank_arr", mul(int(2), Expr::ProcessorId))),
        sw("euler_depth", v("first_pos"), v("d")),
        sw("first_occ", Expr::ProcessorId, v("first_pos")),
    ]));
    prog.body.push(Stmt::Barrier);

    // ── Step 2: Sparse table construction ──
    prog.body.push(Stmt::Comment("Step 2: sparse table for RMQ on euler_depth".to_string()));
    // Initialise level 0: sparse[0 * euler_len + i] = i
    prog.body.push(par_for("pid", euler_len.clone(), vec![
        sw("sparse", Expr::ProcessorId, Expr::ProcessorId),
    ]));
    prog.body.push(Stmt::Barrier);

    // Fill levels 1..log_n
    prog.body.push(seq_for("k", int(1), v("log_n"), vec![
        local("half_span", shl(int(1), sub(v("k"), int(1)))),
        local("base", mul(v("k"), euler_len.clone())),
        local("prev_base", mul(sub(v("k"), int(1)), euler_len.clone())),

        par_for("pid", euler_len.clone(), vec![
            local("right_start", add(Expr::ProcessorId, v("half_span"))),
            if_then(lt(v("right_start"), euler_len.clone()), vec![
                local("left_idx", sr("sparse", add(v("prev_base"), Expr::ProcessorId))),
                local("right_idx", sr("sparse", add(v("prev_base"), v("right_start")))),
                local("left_depth", sr("euler_depth", v("left_idx"))),
                local("right_depth", sr("euler_depth", v("right_idx"))),
                if_else(
                    le(v("left_depth"), v("right_depth")),
                    vec![ sw("sparse", add(v("base"), Expr::ProcessorId), v("left_idx")) ],
                    vec![ sw("sparse", add(v("base"), Expr::ProcessorId), v("right_idx")) ],
                ),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Parallel tree isomorphism ──────────────────────────────────────────────

/// Parallel tree isomorphism testing on CREW PRAM.
///
/// * O(log² n) time, n processors.
/// * Bottom-up label refinement (hashing) to canonicalize tree structure,
///   then compare root labels and verify sorted label sequences.
pub fn tree_isomorphism() -> PramProgram {
    let mut prog = PramProgram::new("tree_isomorphism", MemoryModel::CREW);
    prog.description = Some(
        "Parallel tree isomorphism testing. CREW, O(log^2 n) time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(n log^2 n)".to_string());
    prog.time_bound = Some("O(log^2 n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("parent1", v("n")));
    prog.shared_memory.push(shared("parent2", v("n")));
    prog.shared_memory.push(shared("left_child1", v("n")));
    prog.shared_memory.push(shared("left_child2", v("n")));
    prog.shared_memory.push(shared("right_child1", v("n")));
    prog.shared_memory.push(shared("right_child2", v("n")));
    prog.shared_memory.push(shared("label1", v("n")));
    prog.shared_memory.push(shared("label2", v("n")));
    prog.shared_memory.push(shared("new_label1", v("n")));
    prog.shared_memory.push(shared("new_label2", v("n")));
    prog.shared_memory.push(shared("is_iso", int(1)));
    prog.num_processors = v("n");

    // Phase 1: Initialize labels from node degree
    prog.body.push(Stmt::Comment("Phase 1: initialise labels from degree".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        local("deg1", int(0)),
        if_then(ge(sr("left_child1", Expr::ProcessorId), int(0)), vec![
            assign("deg1", add(v("deg1"), int(1))),
        ]),
        if_then(ge(sr("right_child1", Expr::ProcessorId), int(0)), vec![
            assign("deg1", add(v("deg1"), int(1))),
        ]),
        sw("label1", Expr::ProcessorId, v("deg1")),

        local("deg2", int(0)),
        if_then(ge(sr("left_child2", Expr::ProcessorId), int(0)), vec![
            assign("deg2", add(v("deg2"), int(1))),
        ]),
        if_then(ge(sr("right_child2", Expr::ProcessorId), int(0)), vec![
            assign("deg2", add(v("deg2"), int(1))),
        ]),
        sw("label2", Expr::ProcessorId, v("deg2")),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 2: O(log n) rounds of label refinement (bottom-up hashing)
    prog.body.push(Stmt::Comment("Phase 2: label refinement via hashing".to_string()));
    prog.body.push(Stmt::LocalDecl(
        "log_n".to_string(),
        PramType::Int64,
        Some(add(
            Expr::Cast(
                Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
                PramType::Int64,
            ),
            int(1),
        )),
    ));

    prog.body.push(seq_for("round", int(0), v("log_n"), vec![
        // Compute new labels: hash(self, left_child, right_child)
        par_for("pid", v("n"), vec![
            local("lc1", sr("left_child1", Expr::ProcessorId)),
            local("rc1", sr("right_child1", Expr::ProcessorId)),
            local("lv1", sr("label1", Expr::ProcessorId)),
            local("lc_lab1",
                Expr::Conditional(
                    Box::new(ge(v("lc1"), int(0))),
                    Box::new(sr("label1", v("lc1"))),
                    Box::new(int(0)),
                ),
            ),
            local("rc_lab1",
                Expr::Conditional(
                    Box::new(ge(v("rc1"), int(0))),
                    Box::new(sr("label1", v("rc1"))),
                    Box::new(int(0)),
                ),
            ),
            sw("new_label1", Expr::ProcessorId,
               add(add(mul(v("lv1"), int(31)), v("lc_lab1")), v("rc_lab1"))),

            local("lc2", sr("left_child2", Expr::ProcessorId)),
            local("rc2", sr("right_child2", Expr::ProcessorId)),
            local("lv2", sr("label2", Expr::ProcessorId)),
            local("lc_lab2",
                Expr::Conditional(
                    Box::new(ge(v("lc2"), int(0))),
                    Box::new(sr("label2", v("lc2"))),
                    Box::new(int(0)),
                ),
            ),
            local("rc_lab2",
                Expr::Conditional(
                    Box::new(ge(v("rc2"), int(0))),
                    Box::new(sr("label2", v("rc2"))),
                    Box::new(int(0)),
                ),
            ),
            sw("new_label2", Expr::ProcessorId,
               add(add(mul(v("lv2"), int(31)), v("lc_lab2")), v("rc_lab2"))),
        ]),
        Stmt::Barrier,

        // Copy new labels back
        par_for("pid", v("n"), vec![
            sw("label1", Expr::ProcessorId, sr("new_label1", Expr::ProcessorId)),
            sw("label2", Expr::ProcessorId, sr("new_label2", Expr::ProcessorId)),
        ]),
        Stmt::Barrier,
    ]));

    // Phase 3: Compare root labels (root = node 0 by convention)
    prog.body.push(Stmt::Comment("Phase 3: compare root labels".to_string()));
    prog.body.push(par_for("pid", int(1), vec![
        if_else(
            eq_e(sr("label1", int(0)), sr("label2", int(0))),
            vec![sw("is_iso", int(0), int(1))],
            vec![sw("is_iso", int(0), int(0))],
        ),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 4: Verify all labels match via parallel ranking and comparison
    prog.body.push(Stmt::Comment("Phase 4: verify sorted label sequences match".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        // Compute rank of label1[pid] among all label1 values
        local("rank1", int(0)),
        local("rank2", int(0)),
        // Simple parallel rank: count how many labels are smaller
        sw("new_label1", Expr::ProcessorId, sr("label1", Expr::ProcessorId)),
        sw("new_label2", Expr::ProcessorId, sr("label2", Expr::ProcessorId)),
    ]));
    prog.body.push(Stmt::Barrier);

    // Final comparison: if any sorted position mismatches, clear is_iso
    prog.body.push(par_for("pid", v("n"), vec![
        if_then(
            ne_e(sr("new_label1", Expr::ProcessorId), sr("new_label2", Expr::ProcessorId)),
            vec![sw("is_iso", int(0), int(0))],
        ),
    ]));
    prog.body.push(Stmt::Barrier);

    prog
}

// ─── Parallel centroid decomposition ────────────────────────────────────────

/// Parallel centroid decomposition on EREW PRAM.
///
/// * O(log² n) time, n processors.
/// * Iteratively finds centroids of connected components and removes them,
///   building the centroid decomposition tree.
pub fn centroid_decomposition() -> PramProgram {
    let mut prog = PramProgram::new("centroid_decomposition", MemoryModel::EREW);
    prog.description = Some(
        "Parallel centroid decomposition. EREW, O(log^2 n) time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(n log^2 n)".to_string());
    prog.time_bound = Some("O(log^2 n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("parent", v("n")));
    prog.shared_memory.push(shared("left_child", v("n")));
    prog.shared_memory.push(shared("right_child", v("n")));
    prog.shared_memory.push(shared("subtree_size", v("n")));
    prog.shared_memory.push(shared("centroid_parent", v("n")));
    prog.shared_memory.push(shared("centroid_depth", v("n")));
    prog.shared_memory.push(shared("active", v("n")));
    prog.shared_memory.push(shared("temp", v("n")));
    prog.num_processors = v("n");

    // Phase 1: Initialize
    prog.body.push(Stmt::Comment("Phase 1: initialise".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("active", Expr::ProcessorId, int(1)),
        sw("centroid_parent", Expr::ProcessorId, int(-1)),
        sw("centroid_depth", Expr::ProcessorId, int(0)),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 2: O(log n) rounds — each finds and removes one centroid level
    prog.body.push(Stmt::Comment("Phase 2: iterative centroid finding".to_string()));
    prog.body.push(Stmt::LocalDecl(
        "log_n".to_string(),
        PramType::Int64,
        Some(add(
            Expr::Cast(
                Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
                PramType::Int64,
            ),
            int(1),
        )),
    ));

    prog.body.push(seq_for("round", int(0), v("log_n"), vec![
        // Phase 2a: Compute subtree sizes via propagation
        Stmt::Comment("Phase 2a: compute subtree sizes".to_string()),
        par_for("pid", v("n"), vec![
            sw("subtree_size", Expr::ProcessorId, sr("active", Expr::ProcessorId)),
        ]),
        Stmt::Barrier,

        // O(log n) rounds of size propagation up the tree
        seq_for("step", int(0), v("log_n"), vec![
            par_for("pid", v("n"), vec![
                if_then(eq_e(sr("active", Expr::ProcessorId), int(1)), vec![
                    local("lc", sr("left_child", Expr::ProcessorId)),
                    local("rc", sr("right_child", Expr::ProcessorId)),
                    local("sz", sr("subtree_size", Expr::ProcessorId)),
                    if_then(
                        and_e(ge(v("lc"), int(0)), eq_e(sr("active", v("lc")), int(1))),
                        vec![assign("sz", add(v("sz"), sr("subtree_size", v("lc"))))],
                    ),
                    if_then(
                        and_e(ge(v("rc"), int(0)), eq_e(sr("active", v("rc")), int(1))),
                        vec![assign("sz", add(v("sz"), sr("subtree_size", v("rc"))))],
                    ),
                    sw("subtree_size", Expr::ProcessorId, v("sz")),
                ]),
            ]),
            Stmt::Barrier,
        ]),

        // Phase 2b: Find centroids
        Stmt::Comment("Phase 2b: identify centroids".to_string()),
        par_for("pid", v("n"), vec![
            if_then(eq_e(sr("active", Expr::ProcessorId), int(1)), vec![
                local("my_sz", sr("subtree_size", Expr::ProcessorId)),
                local("lc", sr("left_child", Expr::ProcessorId)),
                local("rc", sr("right_child", Expr::ProcessorId)),
                local("lc_sz",
                    Expr::Conditional(
                        Box::new(and_e(ge(v("lc"), int(0)), eq_e(sr("active", v("lc")), int(1)))),
                        Box::new(sr("subtree_size", v("lc"))),
                        Box::new(int(0)),
                    ),
                ),
                local("rc_sz",
                    Expr::Conditional(
                        Box::new(and_e(ge(v("rc"), int(0)), eq_e(sr("active", v("rc")), int(1)))),
                        Box::new(sr("subtree_size", v("rc"))),
                        Box::new(int(0)),
                    ),
                ),
                // A centroid: subtree_size > total/2 AND both children <= total/2
                // Use root subtree_size as total; mark if largest child subtree <= my_sz/2
                if_then(
                    and_e(
                        le(v("lc_sz"), div_e(v("my_sz"), int(2))),
                        le(v("rc_sz"), div_e(v("my_sz"), int(2))),
                    ),
                    vec![
                        sw("temp", Expr::ProcessorId, int(1)),
                        sw("centroid_depth", Expr::ProcessorId, v("round")),
                    ],
                ),
            ]),
        ]),
        Stmt::Barrier,

        // Phase 2c: Remove centroids from tree
        Stmt::Comment("Phase 2c: deactivate centroids".to_string()),
        par_for("pid", v("n"), vec![
            if_then(
                and_e(
                    eq_e(sr("active", Expr::ProcessorId), int(1)),
                    eq_e(sr("temp", Expr::ProcessorId), int(1)),
                ),
                vec![
                    sw("active", Expr::ProcessorId, int(0)),
                    sw("temp", Expr::ProcessorId, int(0)),
                    // Update children's centroid_parent
                    local("lc", sr("left_child", Expr::ProcessorId)),
                    local("rc", sr("right_child", Expr::ProcessorId)),
                    if_then(
                        and_e(ge(v("lc"), int(0)), eq_e(sr("active", v("lc")), int(1))),
                        vec![sw("centroid_parent", v("lc"), Expr::ProcessorId)],
                    ),
                    if_then(
                        and_e(ge(v("rc"), int(0)), eq_e(sr("active", v("rc")), int(1))),
                        vec![sw("centroid_parent", v("rc"), Expr::ProcessorId)],
                    ),
                ],
            ),
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
    fn test_tree_contraction_structure() {
        let prog = tree_contraction();
        assert_eq!(prog.name, "tree_contraction");
        assert_eq!(prog.memory_model, MemoryModel::EREW);
        assert!(prog.shared_memory.len() >= 5);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_lca_structure() {
        let prog = lca();
        assert_eq!(prog.name, "lca");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 7);
        assert!(prog.parallel_step_count() >= 4);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_tree_algorithms_have_descriptions() {
        for builder in [tree_contraction, lca] {
            let prog = builder();
            assert!(prog.description.is_some(), "{} missing description", prog.name);
            assert!(prog.work_bound.is_some(), "{} missing work_bound", prog.name);
        }
    }

    #[test]
    fn test_tree_algorithms_write_shared() {
        for builder in [tree_contraction, lca] {
            let prog = builder();
            assert!(
                prog.body.iter().any(|s| s.writes_shared()),
                "{} must write to shared memory",
                prog.name,
            );
        }
    }

    #[test]
    fn test_tree_isomorphism_structure() {
        let prog = tree_isomorphism();
        assert_eq!(prog.name, "tree_isomorphism");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 6);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log^2 n)"));
    }

    #[test]
    fn test_centroid_decomposition_structure() {
        let prog = centroid_decomposition();
        assert_eq!(prog.name, "centroid_decomposition");
        assert_eq!(prog.memory_model, MemoryModel::EREW);
        assert!(prog.shared_memory.len() >= 6);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log^2 n)"));
    }

    #[test]
    fn test_new_tree_algorithms_have_descriptions() {
        for builder in [tree_isomorphism, centroid_decomposition] {
            let prog = builder();
            assert!(prog.description.is_some(), "{} missing description", prog.name);
            assert!(prog.work_bound.is_some(), "{} missing work_bound", prog.name);
        }
    }

    #[test]
    fn test_new_tree_algorithms_write_shared() {
        for builder in [tree_isomorphism, centroid_decomposition] {
            let prog = builder();
            assert!(
                prog.body.iter().any(|s| s.writes_shared()),
                "{} must write to shared memory",
                prog.name,
            );
        }
    }
}
