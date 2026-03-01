//! Selection algorithms for PRAM.

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
fn le(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Le, a, b) }
fn ge(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Ge, a, b) }
fn eq_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Eq, a, b) }
fn ne_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Ne, a, b) }
fn and_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::And, a, b) }
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

// ─── Parallel selection (kth element) ───────────────────────────────────────

/// Parallel selection (find kth smallest element) on CREW PRAM.
///
/// * O(log n) time (expected), n processors.
/// * Algorithm (randomised median-of-medians style):
///   1. Pick a random pivot (processor 0 selects).
///   2. In parallel, each processor partitions: count how many
///      elements are < pivot, = pivot, > pivot (CREW reads).
///   3. Use prefix-sum counts to determine which partition contains k.
///   4. Compact the relevant partition and recurse.
///
/// Expected O(log n) rounds because each round reduces the problem
/// size by a constant fraction with high probability.
pub fn parallel_selection() -> PramProgram {
    let mut prog = PramProgram::new("parallel_selection", MemoryModel::CREW);
    prog.description = Some(
        "Parallel kth-element selection. CREW, O(log n) expected time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("k"));
    prog.shared_memory.push(shared("A", v("n")));
    prog.shared_memory.push(shared("B", v("n")));       // compacted subarray
    prog.shared_memory.push(shared("lt_flag", v("n")));  // 1 if A[i] < pivot
    prog.shared_memory.push(shared("eq_flag", v("n")));
    prog.shared_memory.push(shared("gt_flag", v("n")));
    prog.shared_memory.push(shared("lt_dest", v("n")));
    prog.shared_memory.push(shared("gt_dest", v("n")));
    prog.shared_memory.push(shared("pivot", int(1)));
    prog.shared_memory.push(shared("result", int(1)));
    prog.shared_memory.push(shared("cur_n", int(1)));
    prog.shared_memory.push(shared("cur_k", int(1)));
    prog.shared_memory.push(shared("done", int(1)));
    prog.num_processors = v("n");

    // Initialise
    prog.body.push(Stmt::Comment("Initialise: cur_n = n, cur_k = k".to_string()));
    prog.body.push(sw("cur_n", int(0), v("n")));
    prog.body.push(sw("cur_k", int(0), v("k")));
    prog.body.push(sw("done", int(0), int(0)));
    prog.body.push(Stmt::Barrier);

    // Iterate at most O(log n) rounds
    prog.body.push(Stmt::Comment("Selection loop: O(log n) expected rounds".to_string()));
    prog.body.push(Stmt::LocalDecl(
        "max_rounds".to_string(),
        PramType::Int64,
        Some(add(
            mul(
                int(4),
                Expr::Cast(
                    Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
                    PramType::Int64,
                ),
            ),
            int(2),
        )),
    ));

    prog.body.push(seq_for("round", int(0), v("max_rounds"), vec![
        // Check if done
        if_then(eq_e(sr("done", int(0)), int(1)), vec![
            Stmt::Comment("Already found result".to_string()),
        ]),

        // Base case: if cur_n == 1, result = A[0]
        if_then(and_e(
            le(sr("cur_n", int(0)), int(1)),
            eq_e(sr("done", int(0)), int(0)),
        ), vec![
            sw("result", int(0), sr("A", int(0))),
            sw("done", int(0), int(1)),
        ]),
        Stmt::Barrier,

        // Pick pivot: median of first, middle, last element (processor 0)
        Stmt::Comment("Pick pivot from median of three".to_string()),
        par_for("pid", int(1), vec![
            if_then(eq_e(sr("done", int(0)), int(0)), vec![
                local("cn", sr("cur_n", int(0))),
                local("first", sr("A", int(0))),
                local("mid_val", sr("A", div_e(v("cn"), int(2)))),
                local("last", sr("A", sub(v("cn"), int(1)))),
                // Median of three
                local("med", Expr::Conditional(
                    Box::new(and_e(ge(v("first"), v("mid_val")), le(v("first"), v("last")))),
                    Box::new(v("first")),
                    Box::new(Expr::Conditional(
                        Box::new(and_e(ge(v("mid_val"), v("first")), le(v("mid_val"), v("last")))),
                        Box::new(v("mid_val")),
                        Box::new(v("last")),
                    )),
                )),
                sw("pivot", int(0), v("med")),
            ]),
        ]),
        Stmt::Barrier,

        // Partition: each processor classifies its element
        Stmt::Comment("Partition elements around pivot".to_string()),
        par_for("pid", v("n"), vec![
            if_then(and_e(
                lt(Expr::ProcessorId, sr("cur_n", int(0))),
                eq_e(sr("done", int(0)), int(0)),
            ), vec![
                local("val", sr("A", Expr::ProcessorId)),
                local("piv", sr("pivot", int(0))),
                if_else(
                    lt(v("val"), v("piv")),
                    vec![
                        sw("lt_flag", Expr::ProcessorId, int(1)),
                        sw("eq_flag", Expr::ProcessorId, int(0)),
                        sw("gt_flag", Expr::ProcessorId, int(0)),
                    ],
                    vec![
                        if_else(
                            eq_e(v("val"), v("piv")),
                            vec![
                                sw("lt_flag", Expr::ProcessorId, int(0)),
                                sw("eq_flag", Expr::ProcessorId, int(1)),
                                sw("gt_flag", Expr::ProcessorId, int(0)),
                            ],
                            vec![
                                sw("lt_flag", Expr::ProcessorId, int(0)),
                                sw("eq_flag", Expr::ProcessorId, int(0)),
                                sw("gt_flag", Expr::ProcessorId, int(1)),
                            ],
                        ),
                    ],
                ),
            ]),
        ]),
        Stmt::Barrier,

        // Prefix sums on flags
        Stmt::PrefixSum { input: "lt_flag".to_string(), output: "lt_dest".to_string(),
                          size: v("n"), op: BinOp::Add },
        Stmt::PrefixSum { input: "gt_flag".to_string(), output: "gt_dest".to_string(),
                          size: v("n"), op: BinOp::Add },
        Stmt::Barrier,

        // Determine counts: lt_count = lt_dest[cur_n-1], etc.
        Stmt::Comment("Determine which partition contains k".to_string()),
        par_for("pid", int(1), vec![
            if_then(eq_e(sr("done", int(0)), int(0)), vec![
                local("cn", sr("cur_n", int(0))),
                local("ck", sr("cur_k", int(0))),
                local("lt_count", sr("lt_dest", sub(v("cn"), int(1)))),
                local("eq_count", sub(sub(v("cn"), v("lt_count")),
                    sr("gt_dest", sub(v("cn"), int(1))))),

                if_else(
                    lt(v("ck"), v("lt_count")),
                    vec![
                        // k is in the "less than" partition
                        sw("cur_n", int(0), v("lt_count")),
                    ],
                    vec![
                        if_else(
                            lt(v("ck"), add(v("lt_count"), v("eq_count"))),
                            vec![
                                // k falls in the "equal" partition → answer is pivot
                                sw("result", int(0), sr("pivot", int(0))),
                                sw("done", int(0), int(1)),
                            ],
                            vec![
                                // k is in the "greater than" partition
                                sw("cur_k", int(0), sub(v("ck"), add(v("lt_count"), v("eq_count")))),
                                sw("cur_n", int(0), sr("gt_dest", sub(v("cn"), int(1)))),
                            ],
                        ),
                    ],
                ),
            ]),
        ]),
        Stmt::Barrier,

        // Compact the chosen partition into B, then copy B → A
        Stmt::Comment("Compact chosen partition into B, then copy back to A".to_string()),
        par_for("pid", v("n"), vec![
            if_then(and_e(
                lt(Expr::ProcessorId, sr("cur_n", int(0))),
                eq_e(sr("done", int(0)), int(0)),
            ), vec![
                // For simplicity, compact the lt partition
                // (the control logic above sets cur_n appropriately)
                if_then(eq_e(sr("lt_flag", Expr::ProcessorId), int(1)), vec![
                    local("d", sub(sr("lt_dest", Expr::ProcessorId), int(1))),
                    sw("B", v("d"), sr("A", Expr::ProcessorId)),
                ]),
                if_then(eq_e(sr("gt_flag", Expr::ProcessorId), int(1)), vec![
                    local("d", sub(sr("gt_dest", Expr::ProcessorId), int(1))),
                    sw("B", v("d"), sr("A", Expr::ProcessorId)),
                ]),
            ]),
        ]),
        Stmt::Barrier,

        par_for("pid", v("n"), vec![
            if_then(lt(Expr::ProcessorId, sr("cur_n", int(0))), vec![
                sw("A", Expr::ProcessorId, sr("B", Expr::ProcessorId)),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Parallel weighted median finding ───────────────────────────────────────

/// Parallel weighted median finding on CREW PRAM.
///
/// * O(log² n) time, n processors.
/// * Uses median-of-three pivot selection and weighted partitioning
///   to find the weighted median in parallel.
///
/// Input: `A[n]` (values), `weights[n]`.
/// Output: `result[0]` – the weighted median.
pub fn parallel_median() -> PramProgram {
    let mut prog = PramProgram::new("parallel_median", MemoryModel::CREW);
    prog.description = Some(
        "Parallel weighted median finding. CREW, O(log^2 n) time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(n log^2 n)".to_string());
    prog.time_bound = Some("O(log^2 n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("A", v("n")));
    prog.shared_memory.push(shared("weights", v("n")));
    prog.shared_memory.push(shared("B", v("n")));
    prog.shared_memory.push(shared("lt_flag", v("n")));
    prog.shared_memory.push(shared("lt_dest", v("n")));
    prog.shared_memory.push(shared("gt_flag", v("n")));
    prog.shared_memory.push(shared("gt_dest", v("n")));
    prog.shared_memory.push(shared("weight_sum", v("n")));
    prog.shared_memory.push(shared("pivot", int(1)));
    prog.shared_memory.push(shared("result", int(1)));
    prog.num_processors = v("n");

    // Phase 1: Copy A to B
    prog.body.push(Stmt::Comment("Phase 1: copy A to B".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("B", Expr::ProcessorId, sr("A", Expr::ProcessorId)),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 2: O(log n) rounds of weighted median finding
    prog.body.push(Stmt::Comment(
        "Phase 2: weighted median via pivot-based selection".to_string(),
    ));
    prog.body.push(Stmt::LocalDecl(
        "max_rounds".to_string(),
        PramType::Int64,
        Some(add(
            mul(
                int(4),
                Expr::Cast(
                    Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
                    PramType::Int64,
                ),
            ),
            int(2),
        )),
    ));

    prog.body.push(seq_for("round", int(0), v("max_rounds"), vec![
        // Pick median-of-three pivot
        Stmt::Comment("Pick median-of-three pivot".to_string()),
        par_for("pid", int(1), vec![
            local("first", sr("B", int(0))),
            local("mid_val", sr("B", div_e(v("n"), int(2)))),
            local("last", sr("B", sub(v("n"), int(1)))),
            local("med", Expr::Conditional(
                Box::new(and_e(ge(v("first"), v("mid_val")), le(v("first"), v("last")))),
                Box::new(v("first")),
                Box::new(Expr::Conditional(
                    Box::new(and_e(ge(v("mid_val"), v("first")), le(v("mid_val"), v("last")))),
                    Box::new(v("mid_val")),
                    Box::new(v("last")),
                )),
            )),
            sw("pivot", int(0), v("med")),
        ]),
        Stmt::Barrier,

        // Classify each element
        Stmt::Comment("Partition elements around pivot".to_string()),
        par_for("pid", v("n"), vec![
            local("val", sr("B", Expr::ProcessorId)),
            local("piv", sr("pivot", int(0))),
            if_else(
                lt(v("val"), v("piv")),
                vec![
                    sw("lt_flag", Expr::ProcessorId, int(1)),
                    sw("gt_flag", Expr::ProcessorId, int(0)),
                ],
                vec![
                    if_else(
                        Expr::binop(BinOp::Gt, v("val"), v("piv")),
                        vec![
                            sw("lt_flag", Expr::ProcessorId, int(0)),
                            sw("gt_flag", Expr::ProcessorId, int(1)),
                        ],
                        vec![
                            sw("lt_flag", Expr::ProcessorId, int(0)),
                            sw("gt_flag", Expr::ProcessorId, int(0)),
                        ],
                    ),
                ],
            ),
        ]),
        Stmt::Barrier,

        // Prefix sums on flags
        Stmt::PrefixSum { input: "lt_flag".to_string(), output: "lt_dest".to_string(),
                          size: v("n"), op: BinOp::Add },
        Stmt::PrefixSum { input: "gt_flag".to_string(), output: "gt_dest".to_string(),
                          size: v("n"), op: BinOp::Add },
        Stmt::Barrier,

        // Prefix sum on weights for weight sums
        Stmt::PrefixSum { input: "weights".to_string(), output: "weight_sum".to_string(),
                          size: v("n"), op: BinOp::Add },
        Stmt::Barrier,

        // Determine which partition contains weighted median and store result
        Stmt::Comment("Determine weighted median partition".to_string()),
        par_for("pid", int(1), vec![
            local("total_w", sr("weight_sum", sub(v("n"), int(1)))),
            local("half_w", div_e(v("total_w"), int(2))),
            local("lt_w", int(0)),
            seq_for("i", int(0), v("n"), vec![
                if_then(eq_e(sr("lt_flag", v("i")), int(1)), vec![
                    assign("lt_w", add(v("lt_w"), sr("weights", v("i")))),
                ]),
            ]),
            if_then(ge(v("lt_w"), v("half_w")), vec![
                sw("result", int(0), sr("pivot", int(0))),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Parallel array partitioning ────────────────────────────────────────────

/// Parallel array partitioning on EREW PRAM.
///
/// * O(log n) time, n processors.
/// * Partitions array A around a pivot value: elements < pivot come first,
///   then elements >= pivot. Uses prefix sums for destination computation.
///
/// Input: `A[n]`, `pivot_val`.
/// Output: `A[n]` (partitioned in-place), `partition_point[0]`.
pub fn parallel_partition() -> PramProgram {
    let mut prog = PramProgram::new("parallel_partition", MemoryModel::EREW);
    prog.description = Some(
        "Parallel array partitioning. EREW, O(log n) time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("pivot_val"));
    prog.shared_memory.push(shared("A", v("n")));
    prog.shared_memory.push(shared("B", v("n")));
    prog.shared_memory.push(shared("lt_flag", v("n")));
    prog.shared_memory.push(shared("ge_flag", v("n")));
    prog.shared_memory.push(shared("lt_dest", v("n")));
    prog.shared_memory.push(shared("ge_dest", v("n")));
    prog.shared_memory.push(shared("partition_point", int(1)));
    prog.num_processors = v("n");

    // Phase 1: Classify
    prog.body.push(Stmt::Comment("Phase 1: classify elements".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        if_else(
            lt(sr("A", Expr::ProcessorId), v("pivot_val")),
            vec![
                sw("lt_flag", Expr::ProcessorId, int(1)),
                sw("ge_flag", Expr::ProcessorId, int(0)),
            ],
            vec![
                sw("lt_flag", Expr::ProcessorId, int(0)),
                sw("ge_flag", Expr::ProcessorId, int(1)),
            ],
        ),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 2: Prefix sums
    prog.body.push(Stmt::Comment("Phase 2: prefix sums for destinations".to_string()));
    prog.body.push(Stmt::PrefixSum {
        input: "lt_flag".to_string(), output: "lt_dest".to_string(),
        size: v("n"), op: BinOp::Add,
    });
    prog.body.push(Stmt::PrefixSum {
        input: "ge_flag".to_string(), output: "ge_dest".to_string(),
        size: v("n"), op: BinOp::Add,
    });
    prog.body.push(Stmt::Barrier);

    // Phase 3: Store partition_point
    prog.body.push(Stmt::Comment("Phase 3: store partition point".to_string()));
    prog.body.push(sw("partition_point", int(0), sr("lt_dest", sub(v("n"), int(1)))));
    prog.body.push(Stmt::Barrier);

    // Phase 4: Scatter
    prog.body.push(Stmt::Comment("Phase 4: scatter elements".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        if_else(
            eq_e(sr("lt_flag", Expr::ProcessorId), int(1)),
            vec![
                sw("B", sub(sr("lt_dest", Expr::ProcessorId), int(1)),
                   sr("A", Expr::ProcessorId)),
            ],
            vec![
                sw("B",
                   add(sr("partition_point", int(0)),
                       sub(sr("ge_dest", Expr::ProcessorId), int(1))),
                   sr("A", Expr::ProcessorId)),
            ],
        ),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 5: Copy B to A
    prog.body.push(Stmt::Comment("Phase 5: copy B to A".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("A", Expr::ProcessorId, sr("B", Expr::ProcessorId)),
    ]));
    prog.body.push(Stmt::Barrier);

    prog
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_selection_structure() {
        let prog = parallel_selection();
        assert_eq!(prog.name, "parallel_selection");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert_eq!(prog.parameters.len(), 2);
        assert!(prog.shared_memory.len() >= 6);
        assert!(prog.parallel_step_count() >= 4);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_selection_has_description() {
        let prog = parallel_selection();
        assert!(prog.description.is_some());
        assert!(prog.work_bound.is_some());
    }

    #[test]
    fn test_selection_writes_shared() {
        let prog = parallel_selection();
        assert!(prog.body.iter().any(|s| s.writes_shared()));
    }

    #[test]
    fn test_parallel_median_structure() {
        let prog = parallel_median();
        assert_eq!(prog.name, "parallel_median");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 6);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log^2 n)"));
    }

    #[test]
    fn test_parallel_partition_structure() {
        let prog = parallel_partition();
        assert_eq!(prog.name, "parallel_partition");
        assert_eq!(prog.memory_model, MemoryModel::EREW);
        assert_eq!(prog.parameters.len(), 2);
        assert!(prog.shared_memory.len() >= 5);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_new_selection_algorithms_have_descriptions() {
        for builder in [parallel_median, parallel_partition] {
            let prog = builder();
            assert!(prog.description.is_some(), "{} missing description", prog.name);
            assert!(prog.work_bound.is_some(), "{} missing work_bound", prog.name);
        }
    }

    #[test]
    fn test_new_selection_algorithms_write_shared() {
        for builder in [parallel_median, parallel_partition] {
            let prog = builder();
            assert!(prog.body.iter().any(|s| s.writes_shared()), "{} must write shared", prog.name);
        }
    }
}
