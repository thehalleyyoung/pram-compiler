//! Search algorithms for PRAM.

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

// ─── Parallel binary search ────────────────────────────────────────────────

/// Parallel binary search on CREW PRAM.
///
/// * O(log n / log p) time with p processors (p = log n gives O(log n / log log n)).
/// * Input: sorted array `A[n]`, search key.
/// * Strategy: p processors divide the current search interval into p+1
///   sub-intervals, each checking a probe point.  After one round the
///   interval shrinks by a factor of p.  After O(log n / log p) rounds
///   the element is found.
///
/// With p = ⌈log n⌉ processors this achieves O(log n / log log n) time.
pub fn parallel_binary_search() -> PramProgram {
    let mut prog = PramProgram::new("parallel_binary_search", MemoryModel::CREW);
    prog.description = Some(
        "Parallel binary search. CREW, O(log n / log log n) time, log n processors.".to_string(),
    );
    prog.work_bound = Some("O(log n)".to_string());
    prog.time_bound = Some("O(log n / log log n)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("key"));
    // Number of processors p = ceil(log n)
    let p = add(
        Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        ),
        int(1),
    );

    prog.shared_memory.push(shared("A", v("n")));
    prog.shared_memory.push(shared("probe_result", p.clone()));  // comparison results
    prog.shared_memory.push(shared("lo", int(1)));               // current lower bound
    prog.shared_memory.push(shared("hi", int(1)));               // current upper bound
    prog.shared_memory.push(shared("found_idx", int(1)));        // result index
    prog.shared_memory.push(shared("done", int(1)));
    prog.num_processors = p.clone();

    // Initialise
    prog.body.push(Stmt::Comment("Initialise search bounds".to_string()));
    prog.body.push(sw("lo", int(0), int(0)));
    prog.body.push(sw("hi", int(0), sub(v("n"), int(1))));
    prog.body.push(sw("found_idx", int(0), int(-1)));
    prog.body.push(sw("done", int(0), int(0)));
    prog.body.push(Stmt::Barrier);

    // Number of rounds = ceil(log n / log p)
    // Conservatively use log_n rounds.
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
    prog.body.push(Stmt::LocalDecl("p".to_string(), PramType::Int64, Some(p.clone())));

    prog.body.push(Stmt::Comment(
        "Search loop: O(log n / log p) rounds; each round shrinks interval by factor p".to_string(),
    ));
    prog.body.push(seq_for("round", int(0), v("max_rounds"), vec![
        // Each processor probes a point in [lo, hi]
        par_for("pid", v("p"), vec![
            if_then(eq_e(sr("done", int(0)), int(0)), vec![
                local("cur_lo", sr("lo", int(0))),
                local("cur_hi", sr("hi", int(0))),
                local("range", sub(v("cur_hi"), v("cur_lo"))),
                // Probe point = lo + (pid + 1) * range / (p + 1)
                local("probe", add(v("cur_lo"),
                    div_e(
                        mul(add(Expr::ProcessorId, int(1)), v("range")),
                        add(v("p"), int(1)),
                    ),
                )),
                // Clamp to valid range
                local("probe_clamped", Expr::Conditional(
                    Box::new(lt(v("probe"), v("cur_lo"))),
                    Box::new(v("cur_lo")),
                    Box::new(Expr::Conditional(
                        Box::new(Expr::binop(BinOp::Gt, v("probe"), v("cur_hi"))),
                        Box::new(v("cur_hi")),
                        Box::new(v("probe")),
                    )),
                )),

                local("val", sr("A", v("probe_clamped"))),
                // probe_result[pid]: -1 if val < key, 0 if val == key, 1 if val > key
                if_else(
                    eq_e(v("val"), v("key")),
                    vec![
                        sw("probe_result", Expr::ProcessorId, int(0)),
                        sw("found_idx", int(0), v("probe_clamped")),
                        sw("done", int(0), int(1)),
                    ],
                    vec![
                        if_else(
                            lt(v("val"), v("key")),
                            vec![ sw("probe_result", Expr::ProcessorId, int(-1)) ],
                            vec![ sw("probe_result", Expr::ProcessorId, int(1)) ],
                        ),
                    ],
                ),
            ]),
        ]),
        Stmt::Barrier,

        // Narrow the interval: find the transition point in probe_result
        // The new interval is between the last probe < key and the first probe > key.
        if_then(eq_e(sr("done", int(0)), int(0)), vec![
            par_for("pid", int(1), vec![
                local("cur_lo", sr("lo", int(0))),
                local("cur_hi", sr("hi", int(0))),
                local("range", sub(v("cur_hi"), v("cur_lo"))),
                local("new_lo", v("cur_lo")),
                local("new_hi", v("cur_hi")),

                // Scan probe results to find transition
                seq_for("i", int(0), v("p"), vec![
                    local("pr", sr("probe_result", v("i"))),
                    local("probe_pos", add(v("cur_lo"),
                        div_e(
                            mul(add(v("i"), int(1)), v("range")),
                            add(v("p"), int(1)),
                        ),
                    )),
                    if_then(eq_e(v("pr"), int(-1)), vec![
                        // This probe is less than key → move lo up
                        assign("new_lo", add(v("probe_pos"), int(1))),
                    ]),
                    if_then(eq_e(v("pr"), int(1)), vec![
                        // This probe is greater than key → move hi down
                        assign("new_hi", sub(v("probe_pos"), int(1))),
                    ]),
                ]),

                sw("lo", int(0), v("new_lo")),
                sw("hi", int(0), v("new_hi")),
            ]),
        ]),
        Stmt::Barrier,

        // If lo > hi and not found, element is absent
        if_then(and_e(
            Expr::binop(BinOp::Gt, sr("lo", int(0)), sr("hi", int(0))),
            eq_e(sr("done", int(0)), int(0)),
        ), vec![
            sw("done", int(0), int(1)),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Parallel interpolation search ─────────────────────────────────────────

/// Parallel interpolation search on CREW PRAM.
///
/// * O(log log n) expected time on uniformly distributed data, p processors.
/// * Like parallel binary search, but each probe point is chosen by
///   interpolation rather than uniform subdivision:
///     probe = lo + (key - A[lo]) / (A[hi] - A[lo]) * (hi - lo)
///   With p processors probing in a neighbourhood around the
///   interpolation estimate, the search interval shrinks super-
///   exponentially.
pub fn parallel_interpolation_search() -> PramProgram {
    let mut prog = PramProgram::new("parallel_interpolation_search", MemoryModel::CREW);
    prog.description = Some(
        "Parallel interpolation search. CREW, O(log log n) expected time.".to_string(),
    );
    prog.work_bound = Some("O(sqrt(n))".to_string());
    prog.time_bound = Some("O(log log n)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("key"));

    let p = Expr::FunctionCall("sqrt".to_string(), vec![
        Expr::Cast(Box::new(v("n")), PramType::Float64),
    ]);
    let p_int = Expr::Cast(Box::new(p.clone()), PramType::Int64);

    prog.shared_memory.push(shared("A", v("n")));
    prog.shared_memory.push(shared("probe_result", add(p_int.clone(), int(1))));
    prog.shared_memory.push(shared("lo", int(1)));
    prog.shared_memory.push(shared("hi", int(1)));
    prog.shared_memory.push(shared("found_idx", int(1)));
    prog.shared_memory.push(shared("done", int(1)));
    prog.num_processors = add(p_int.clone(), int(1));

    // Initialise
    prog.body.push(Stmt::Comment("Initialise search bounds".to_string()));
    prog.body.push(sw("lo", int(0), int(0)));
    prog.body.push(sw("hi", int(0), sub(v("n"), int(1))));
    prog.body.push(sw("found_idx", int(0), int(-1)));
    prog.body.push(sw("done", int(0), int(0)));
    prog.body.push(Stmt::Barrier);

    // At most O(log log n) rounds
    prog.body.push(Stmt::LocalDecl(
        "max_rounds".to_string(),
        PramType::Int64,
        Some(add(
            Expr::Cast(
                Box::new(log2_call(log2_call(
                    Expr::Cast(Box::new(add(v("n"), int(1))), PramType::Float64),
                ))),
                PramType::Int64,
            ),
            int(4),
        )),
    ));
    prog.body.push(Stmt::LocalDecl("p_count".to_string(), PramType::Int64, Some(p_int.clone())));

    prog.body.push(Stmt::Comment(
        "Interpolation search loop: O(log log n) rounds".to_string(),
    ));
    prog.body.push(seq_for("round", int(0), v("max_rounds"), vec![
        par_for("pid", v("p_count"), vec![
            if_then(eq_e(sr("done", int(0)), int(0)), vec![
                local("cur_lo", sr("lo", int(0))),
                local("cur_hi", sr("hi", int(0))),

                if_then(le(v("cur_lo"), v("cur_hi")), vec![
                    local("a_lo", sr("A", v("cur_lo"))),
                    local("a_hi", sr("A", v("cur_hi"))),
                    local("range", sub(v("cur_hi"), v("cur_lo"))),

                    // Interpolation estimate
                    local("est", Expr::Conditional(
                        Box::new(eq_e(v("a_hi"), v("a_lo"))),
                        Box::new(v("cur_lo")),
                        Box::new(add(v("cur_lo"),
                            div_e(
                                mul(sub(v("key"), v("a_lo")), v("range")),
                                sub(v("a_hi"), v("a_lo")),
                            ),
                        )),
                    )),

                    // Each processor probes est + (pid - p/2) to cover a neighbourhood
                    local("offset", sub(Expr::ProcessorId, div_e(v("p_count"), int(2)))),
                    local("probe", add(v("est"), v("offset"))),
                    // Clamp
                    local("probe_c", Expr::Conditional(
                        Box::new(lt(v("probe"), v("cur_lo"))),
                        Box::new(v("cur_lo")),
                        Box::new(Expr::Conditional(
                            Box::new(Expr::binop(BinOp::Gt, v("probe"), v("cur_hi"))),
                            Box::new(v("cur_hi")),
                            Box::new(v("probe")),
                        )),
                    )),

                    local("val", sr("A", v("probe_c"))),
                    if_else(
                        eq_e(v("val"), v("key")),
                        vec![
                            sw("probe_result", Expr::ProcessorId, int(0)),
                            sw("found_idx", int(0), v("probe_c")),
                            sw("done", int(0), int(1)),
                        ],
                        vec![
                            if_else(
                                lt(v("val"), v("key")),
                                vec![ sw("probe_result", Expr::ProcessorId, int(-1)) ],
                                vec![ sw("probe_result", Expr::ProcessorId, int(1)) ],
                            ),
                        ],
                    ),
                ]),
            ]),
        ]),
        Stmt::Barrier,

        // Narrow interval
        if_then(eq_e(sr("done", int(0)), int(0)), vec![
            par_for("pid", int(1), vec![
                local("cur_lo", sr("lo", int(0))),
                local("cur_hi", sr("hi", int(0))),
                local("new_lo", v("cur_lo")),
                local("new_hi", v("cur_hi")),
                local("range", sub(v("cur_hi"), v("cur_lo"))),

                // Compute interpolation estimate for probe positions
                local("a_lo", sr("A", v("cur_lo"))),
                local("a_hi", sr("A", v("cur_hi"))),
                local("est", Expr::Conditional(
                    Box::new(eq_e(v("a_hi"), v("a_lo"))),
                    Box::new(v("cur_lo")),
                    Box::new(add(v("cur_lo"),
                        div_e(
                            mul(sub(v("key"), v("a_lo")), v("range")),
                            sub(v("a_hi"), v("a_lo")),
                        ),
                    )),
                )),

                seq_for("i", int(0), v("p_count"), vec![
                    local("pr", sr("probe_result", v("i"))),
                    local("offset", sub(v("i"), div_e(v("p_count"), int(2)))),
                    local("probe_pos", add(v("est"), v("offset"))),
                    local("probe_clamped", Expr::Conditional(
                        Box::new(lt(v("probe_pos"), v("cur_lo"))),
                        Box::new(v("cur_lo")),
                        Box::new(Expr::Conditional(
                            Box::new(Expr::binop(BinOp::Gt, v("probe_pos"), v("cur_hi"))),
                            Box::new(v("cur_hi")),
                            Box::new(v("probe_pos")),
                        )),
                    )),
                    if_then(eq_e(v("pr"), int(-1)), vec![
                        assign("new_lo", add(v("probe_clamped"), int(1))),
                    ]),
                    if_then(eq_e(v("pr"), int(1)), vec![
                        assign("new_hi", sub(v("probe_clamped"), int(1))),
                    ]),
                ]),

                sw("lo", int(0), v("new_lo")),
                sw("hi", int(0), v("new_hi")),
            ]),
        ]),
        Stmt::Barrier,

        if_then(and_e(
            Expr::binop(BinOp::Gt, sr("lo", int(0)), sr("hi", int(0))),
            eq_e(sr("done", int(0)), int(0)),
        ), vec![
            sw("done", int(0), int(1)),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Parallel batch search ──────────────────────────────────────────────────

/// Parallel batch search for multiple keys on CREW PRAM.
///
/// * O(log n) time, k processors (one per key).
/// * Each processor performs binary search on the same sorted array A
///   for its assigned key. All k searches proceed simultaneously.
///
/// Input: `A[n]` (sorted), `keys[k]` (search keys).
/// Output: `results[k]` – index of key in A, or -1 if not found.
pub fn parallel_batch_search() -> PramProgram {
    let mut prog = PramProgram::new("parallel_batch_search", MemoryModel::CREW);
    prog.description = Some(
        "Parallel batch search for multiple keys. CREW, O(log n) time, n*k processors.".to_string(),
    );
    prog.work_bound = Some("O(k * log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("k"));
    prog.shared_memory.push(shared("A", v("n")));
    prog.shared_memory.push(shared("keys", v("k")));
    prog.shared_memory.push(shared("results", v("k")));
    prog.shared_memory.push(shared("lo_arr", v("k")));
    prog.shared_memory.push(shared("hi_arr", v("k")));
    prog.shared_memory.push(shared("done_arr", v("k")));
    prog.num_processors = v("k");

    // Phase 1: Initialize
    prog.body.push(Stmt::Comment("Phase 1: initialise search state".to_string()));
    prog.body.push(par_for("pid", v("k"), vec![
        sw("lo_arr", Expr::ProcessorId, int(0)),
        sw("hi_arr", Expr::ProcessorId, sub(v("n"), int(1))),
        sw("done_arr", Expr::ProcessorId, int(0)),
        sw("results", Expr::ProcessorId, int(-1)),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 2: O(log n) rounds of binary search
    prog.body.push(Stmt::Comment(
        "Phase 2: O(log n) rounds of parallel binary search".to_string(),
    ));
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

    prog.body.push(seq_for("round", int(0), v("log_n"), vec![
        par_for("pid", v("k"), vec![
            if_then(eq_e(sr("done_arr", Expr::ProcessorId), int(0)), vec![
                local("lo", sr("lo_arr", Expr::ProcessorId)),
                local("hi", sr("hi_arr", Expr::ProcessorId)),
                if_else(
                    Expr::binop(BinOp::Gt, v("lo"), v("hi")),
                    vec![
                        sw("done_arr", Expr::ProcessorId, int(1)),
                    ],
                    vec![
                        local("mid", add(v("lo"), div_e(sub(v("hi"), v("lo")), int(2)))),
                        local("mid_val", sr("A", v("mid"))),
                        local("my_key", sr("keys", Expr::ProcessorId)),
                        if_else(
                            eq_e(v("mid_val"), v("my_key")),
                            vec![
                                sw("results", Expr::ProcessorId, v("mid")),
                                sw("done_arr", Expr::ProcessorId, int(1)),
                            ],
                            vec![
                                if_else(
                                    lt(v("mid_val"), v("my_key")),
                                    vec![
                                        sw("lo_arr", Expr::ProcessorId, add(v("mid"), int(1))),
                                    ],
                                    vec![
                                        sw("hi_arr", Expr::ProcessorId, sub(v("mid"), int(1))),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
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
    fn test_parallel_binary_search_structure() {
        let prog = parallel_binary_search();
        assert_eq!(prog.name, "parallel_binary_search");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert_eq!(prog.parameters.len(), 2);
        assert!(prog.shared_memory.len() >= 5);
        assert!(prog.parallel_step_count() >= 2);
        assert!(prog.time_bound.as_deref() == Some("O(log n / log log n)"));
    }

    #[test]
    fn test_parallel_interpolation_search_structure() {
        let prog = parallel_interpolation_search();
        assert_eq!(prog.name, "parallel_interpolation_search");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 5);
        assert!(prog.parallel_step_count() >= 2);
        assert!(prog.time_bound.as_deref() == Some("O(log log n)"));
    }

    #[test]
    fn test_search_algorithms_have_descriptions() {
        for builder in [parallel_binary_search, parallel_interpolation_search] {
            let prog = builder();
            assert!(prog.description.is_some(), "{} missing description", prog.name);
            assert!(prog.work_bound.is_some(), "{} missing work_bound", prog.name);
        }
    }

    #[test]
    fn test_search_algorithms_write_shared() {
        for builder in [parallel_binary_search, parallel_interpolation_search] {
            let prog = builder();
            assert!(
                prog.body.iter().any(|s| s.writes_shared()),
                "{} must write to shared memory",
                prog.name,
            );
        }
    }

    #[test]
    fn test_parallel_batch_search_structure() {
        let prog = parallel_batch_search();
        assert_eq!(prog.name, "parallel_batch_search");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert_eq!(prog.parameters.len(), 2);
        assert!(prog.shared_memory.len() >= 4);
        assert!(prog.parallel_step_count() >= 2);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_new_search_algorithms_have_descriptions() {
        let prog = parallel_batch_search();
        assert!(prog.description.is_some());
        assert!(prog.work_bound.is_some());
    }

    #[test]
    fn test_new_search_algorithms_write_shared() {
        let prog = parallel_batch_search();
        assert!(prog.body.iter().any(|s| s.writes_shared()));
    }
}
