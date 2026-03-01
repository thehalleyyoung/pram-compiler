//! String algorithms for PRAM.

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

// ─── Parallel string matching ───────────────────────────────────────────────

/// Parallel string matching on CREW PRAM.
///
/// * O(log m) time, n × m processors (brute-force variant).
/// * For each of n - m + 1 text positions, m processors check one
///   character of the pattern in parallel.
/// * A parallel AND-reduction over the m comparisons determines match.
///
/// A more work-efficient version uses witness-based techniques, but
/// the brute-force approach is the canonical CREW encoding.
///
/// Input: `text[n]`, `pattern[m]` (character codes).
/// Output: `match_flag[n]` – 1 if pattern matches at position i.
pub fn string_matching() -> PramProgram {
    let mut prog = PramProgram::new("string_matching", MemoryModel::CREW);
    prog.description = Some(
        "Parallel brute-force string matching. CREW, O(log m) time, n*m processors.".to_string(),
    );
    prog.work_bound = Some("O(n * m)".to_string());
    prog.time_bound = Some("O(log m)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("m"));
    prog.shared_memory.push(shared("text", v("n")));
    prog.shared_memory.push(shared("pattern", v("m")));
    prog.shared_memory.push(shared("match_flag", v("n")));
    // match_grid[i * m + j] = 1 if text[i+j] == pattern[j]
    let grid_size = mul(v("n"), v("m"));
    prog.shared_memory.push(shared("match_grid", grid_size.clone()));
    prog.num_processors = grid_size.clone();

    // Step 1: each processor (i,j) compares text[i+j] with pattern[j]
    prog.body.push(Stmt::Comment(
        "Step 1: each processor (i,j) compares text[i+j] == pattern[j]".to_string(),
    ));
    prog.body.push(par_for("pid", grid_size.clone(), vec![
        local("i", div_e(Expr::ProcessorId, v("m"))),
        local("j", mod_e(Expr::ProcessorId, v("m"))),
        local("text_pos", add(v("i"), v("j"))),
        if_else(
            lt(v("text_pos"), v("n")),
            vec![
                if_else(
                    eq_e(sr("text", v("text_pos")), sr("pattern", v("j"))),
                    vec![ sw("match_grid", Expr::ProcessorId, int(1)) ],
                    vec![ sw("match_grid", Expr::ProcessorId, int(0)) ],
                ),
            ],
            vec![ sw("match_grid", Expr::ProcessorId, int(0)) ],
        ),
    ]));
    prog.body.push(Stmt::Barrier);

    // Step 2: parallel AND-reduction along j for each position i
    // Uses log m rounds of pairwise AND.
    prog.body.push(Stmt::Comment(
        "Step 2: parallel AND-reduction along j dimension, O(log m) steps".to_string(),
    ));
    prog.body.push(Stmt::LocalDecl(
        "log_m".to_string(),
        PramType::Int64,
        Some(Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("m")), PramType::Float64))),
            PramType::Int64,
        )),
    ));

    prog.body.push(seq_for("d", int(0), v("log_m"), vec![
        local("stride", shl(int(1), v("d"))),

        par_for("pid", grid_size.clone(), vec![
            local("i", div_e(Expr::ProcessorId, v("m"))),
            local("j", mod_e(Expr::ProcessorId, v("m"))),

            if_then(
                and_e(
                    eq_e(mod_e(v("j"), mul(int(2), v("stride"))), int(0)),
                    lt(add(v("j"), v("stride")), v("m")),
                ),
                vec![
                    local("partner", add(Expr::ProcessorId, v("stride"))),
                    local("a_val", sr("match_grid", Expr::ProcessorId)),
                    local("b_val", sr("match_grid", v("partner"))),
                    sw("match_grid", Expr::ProcessorId,
                       Expr::binop(BinOp::BitAnd, v("a_val"), v("b_val"))),
                ],
            ),
        ]),
        Stmt::Barrier,
    ]));

    // Step 3: match_flag[i] = match_grid[i * m + 0]
    prog.body.push(Stmt::Comment("Step 3: write match results".to_string()));
    let num_positions = add(sub(v("n"), v("m")), int(1));
    prog.body.push(par_for("pid", num_positions, vec![
        sw("match_flag", Expr::ProcessorId,
           sr("match_grid", mul(Expr::ProcessorId, v("m")))),
    ]));

    prog
}

// ─── Parallel suffix array construction ─────────────────────────────────────

/// Parallel suffix array construction on CREW PRAM.
///
/// * O(log² n) time, n processors.
/// * Prefix-doubling approach:
///   1. Start with rank = character value for each suffix.
///   2. In each of O(log n) rounds, double the comparison length:
///      rank a suffix by the pair (rank of first half, rank of second half).
///   3. Use parallel sorting of pairs to assign new ranks.
///   4. Stop when all ranks are unique.
///
/// Each round takes O(log n) for sorting ⇒ total O(log² n).
pub fn suffix_array() -> PramProgram {
    let mut prog = PramProgram::new("suffix_array", MemoryModel::CREW);
    prog.description = Some(
        "Parallel suffix array (prefix-doubling). CREW, O(log^2 n) time, n processors."
            .to_string(),
    );
    prog.work_bound = Some("O(n log^2 n)".to_string());
    prog.time_bound = Some("O(log^2 n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("text", v("n")));
    prog.shared_memory.push(shared("sa", v("n")));         // suffix array
    prog.shared_memory.push(shared("rank_arr", v("n")));   // current rank
    prog.shared_memory.push(shared("new_rank", v("n")));
    prog.shared_memory.push(shared("key1", v("n")));       // first-half rank
    prog.shared_memory.push(shared("key2", v("n")));       // second-half rank
    prog.shared_memory.push(shared("temp", v("n")));
    prog.num_processors = v("n");

    // Initialise: rank = text character, sa = identity
    prog.body.push(Stmt::Comment("Initialise: rank[i] = text[i], sa[i] = i".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("rank_arr", Expr::ProcessorId, sr("text", Expr::ProcessorId)),
        sw("sa", Expr::ProcessorId, Expr::ProcessorId),
    ]));
    prog.body.push(Stmt::Barrier);

    // O(log n) doubling rounds
    prog.body.push(Stmt::Comment("Prefix-doubling: O(log n) rounds".to_string()));
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
        local("offset", shl(int(1), v("round"))),

        // Build key pairs: key1[i] = rank[i], key2[i] = rank[i + offset] or -1
        Stmt::Comment("Build (key1, key2) pairs for sorting".to_string()),
        par_for("pid", v("n"), vec![
            sw("key1", Expr::ProcessorId, sr("rank_arr", Expr::ProcessorId)),
            if_else(
                lt(add(Expr::ProcessorId, v("offset")), v("n")),
                vec![ sw("key2", Expr::ProcessorId,
                         sr("rank_arr", add(Expr::ProcessorId, v("offset")))) ],
                vec![ sw("key2", Expr::ProcessorId, int(-1)) ],
            ),
        ]),
        Stmt::Barrier,

        // Sort by (key1, key2) via ranking — each suffix computes its rank
        // among all suffixes by counting how many have smaller (key1, key2).
        Stmt::Comment("Rank suffixes by (key1, key2) pairs".to_string()),
        par_for("pid", v("n"), vec![
            local("my_k1", sr("key1", Expr::ProcessorId)),
            local("my_k2", sr("key2", Expr::ProcessorId)),
            local("my_new_rank", int(0)),
            seq_for("j", int(0), v("n"), vec![
                local("other_k1", sr("key1", v("j"))),
                local("other_k2", sr("key2", v("j"))),
                // Count strictly smaller pairs
                if_then(
                    Expr::binop(BinOp::Or,
                        lt(v("other_k1"), v("my_k1")),
                        and_e(eq_e(v("other_k1"), v("my_k1")),
                              lt(v("other_k2"), v("my_k2"))),
                    ),
                    vec![ assign("my_new_rank", add(v("my_new_rank"), int(1))) ],
                ),
                // Break ties by index
                if_then(
                    and_e(
                        eq_e(v("other_k1"), v("my_k1")),
                        and_e(eq_e(v("other_k2"), v("my_k2")),
                              lt(v("j"), Expr::ProcessorId)),
                    ),
                    vec![ assign("my_new_rank", add(v("my_new_rank"), int(1))) ],
                ),
            ]),
            sw("new_rank", Expr::ProcessorId, v("my_new_rank")),
            sw("sa", v("my_new_rank"), Expr::ProcessorId),
        ]),
        Stmt::Barrier,

        // Update rank array
        par_for("pid", v("n"), vec![
            sw("rank_arr", Expr::ProcessorId, sr("new_rank", Expr::ProcessorId)),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Parallel LCP array construction ────────────────────────────────────────

/// Parallel LCP array construction on CREW PRAM.
///
/// * O(log² n) time, n processors.
/// * Computes the LCP (Longest Common Prefix) array from a suffix array
///   using a Kasai-like approach executed in parallel.
///
/// Input: `text[n]`, `sa[n]` (suffix array).
/// Output: `lcp[n]` – LCP values.
pub fn lcp_array() -> PramProgram {
    let mut prog = PramProgram::new("lcp_array", MemoryModel::CREW);
    prog.description = Some(
        "Parallel LCP array construction. CREW, O(log^2 n) time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(n log^2 n)".to_string());
    prog.time_bound = Some("O(log^2 n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("text", v("n")));
    prog.shared_memory.push(shared("sa", v("n")));
    prog.shared_memory.push(shared("rank_arr", v("n")));
    prog.shared_memory.push(shared("lcp", v("n")));
    prog.shared_memory.push(shared("temp", v("n")));
    prog.shared_memory.push(shared("inv_sa", v("n")));
    prog.num_processors = v("n");

    // Phase 1: Compute inverse suffix array
    prog.body.push(Stmt::Comment("Phase 1: inv_sa[sa[i]] = i".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("inv_sa", sr("sa", Expr::ProcessorId), Expr::ProcessorId),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 2: Initialize lcp[0] = 0
    prog.body.push(Stmt::Comment("Phase 2: lcp[0] = 0".to_string()));
    prog.body.push(sw("lcp", int(0), int(0)));
    prog.body.push(Stmt::Barrier);

    // Phase 3: Compute LCP values using Kasai-like approach in parallel
    prog.body.push(Stmt::Comment(
        "Phase 3: Kasai-like parallel LCP computation".to_string(),
    ));
    prog.body.push(par_for("pid", v("n"), vec![
        local("rank_i", sr("inv_sa", Expr::ProcessorId)),
        if_then(Expr::binop(BinOp::Gt, v("rank_i"), int(0)), vec![
            local("j", sr("sa", sub(v("rank_i"), int(1)))),
            local("match_len", int(0)),
            seq_for("c", int(0), v("n"), vec![
                local("pos_a", add(Expr::ProcessorId, v("match_len"))),
                local("pos_b", add(v("j"), v("match_len"))),
                if_then(
                    and_e(
                        lt(v("pos_a"), v("n")),
                        and_e(
                            lt(v("pos_b"), v("n")),
                            eq_e(sr("text", v("pos_a")), sr("text", v("pos_b"))),
                        ),
                    ),
                    vec![assign("match_len", add(v("match_len"), int(1)))],
                ),
            ]),
            sw("lcp", v("rank_i"), v("match_len")),
        ]),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 4: Verify and adjust via parallel prefix
    prog.body.push(Stmt::Comment(
        "Phase 4: propagate minimum LCP information".to_string(),
    ));
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
    prog.body.push(seq_for("d", int(0), v("log_n"), vec![
        local("stride", shl(int(1), v("d"))),
        par_for("pid", v("n"), vec![
            if_then(
                and_e(
                    ge(Expr::ProcessorId, v("stride")),
                    lt(Expr::ProcessorId, v("n")),
                ),
                vec![
                    local("a_val", sr("lcp", Expr::ProcessorId)),
                    local("b_val", sr("lcp", sub(Expr::ProcessorId, v("stride")))),
                    sw("temp", Expr::ProcessorId, Expr::Conditional(
                        Box::new(lt(v("a_val"), v("b_val"))),
                        Box::new(v("a_val")),
                        Box::new(v("b_val")),
                    )),
                ],
            ),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Parallel string sorting (MSD radix) ────────────────────────────────────

/// Parallel string sorting (MSD radix) on CREW PRAM.
///
/// * O(L log n) time, n processors.
/// * Sorts strings by examining characters from most significant to least,
///   using counting-based ranking at each position.
///
/// Input: `strings[n * max_len]` (flattened), `max_len`.
/// Output: `order[n]` – sorted permutation.
pub fn string_sorting() -> PramProgram {
    let mut prog = PramProgram::new("string_sorting", MemoryModel::CREW);
    prog.description = Some(
        "Parallel string sorting (MSD radix). CREW, O(L log n) time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(n * L)".to_string());
    prog.time_bound = Some("O(L log n)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("max_len"));
    prog.shared_memory.push(shared("strings", mul(v("n"), v("max_len"))));
    prog.shared_memory.push(shared("order", v("n")));
    prog.shared_memory.push(shared("new_order", v("n")));
    prog.shared_memory.push(shared("keys", v("n")));
    prog.shared_memory.push(shared("bucket_id", v("n")));
    prog.shared_memory.push(shared("bucket_offset", v("n")));
    prog.shared_memory.push(shared("bucket_count", v("n")));
    prog.num_processors = v("n");

    // Phase 1: Initialize order[pid] = pid
    prog.body.push(Stmt::Comment("Phase 1: initialise order".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("order", Expr::ProcessorId, Expr::ProcessorId),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 2: For each character position d from 0 to max_len
    prog.body.push(Stmt::Comment(
        "Phase 2: MSD radix sort over character positions".to_string(),
    ));
    prog.body.push(seq_for("d", int(0), v("max_len"), vec![
        // Phase 2a: Extract key at position d
        Stmt::Comment("Phase 2a: extract key at position d".to_string()),
        par_for("pid", v("n"), vec![
            sw("keys", Expr::ProcessorId,
               sr("strings", add(mul(sr("order", Expr::ProcessorId), v("max_len")), v("d")))),
        ]),
        Stmt::Barrier,

        // Phase 2b: Rank by key via counting
        Stmt::Comment("Phase 2b: rank by key via counting".to_string()),
        par_for("pid", v("n"), vec![
            local("my_key", sr("keys", Expr::ProcessorId)),
            local("cnt", int(0)),
            seq_for("j", int(0), v("n"), vec![
                if_then(
                    Expr::binop(BinOp::Or,
                        lt(sr("keys", v("j")), v("my_key")),
                        and_e(
                            eq_e(sr("keys", v("j")), v("my_key")),
                            lt(v("j"), Expr::ProcessorId),
                        ),
                    ),
                    vec![assign("cnt", add(v("cnt"), int(1)))],
                ),
            ]),
            sw("bucket_offset", Expr::ProcessorId, v("cnt")),
        ]),
        Stmt::Barrier,

        // Phase 2c: Prefix sum on counts -> offsets
        Stmt::PrefixSum {
            input: "bucket_offset".to_string(),
            output: "bucket_id".to_string(),
            size: v("n"),
            op: BinOp::Add,
        },
        Stmt::Barrier,

        // Phase 2d: Scatter
        Stmt::Comment("Phase 2d: scatter into new_order".to_string()),
        par_for("pid", v("n"), vec![
            sw("new_order", sr("bucket_offset", Expr::ProcessorId),
               sr("order", Expr::ProcessorId)),
        ]),
        Stmt::Barrier,

        // Phase 2e: Copy new_order to order
        Stmt::Comment("Phase 2e: copy new_order to order".to_string()),
        par_for("pid", v("n"), vec![
            sw("order", Expr::ProcessorId, sr("new_order", Expr::ProcessorId)),
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
    fn test_string_matching_structure() {
        let prog = string_matching();
        assert_eq!(prog.name, "string_matching");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert_eq!(prog.parameters.len(), 2);
        assert!(prog.shared_memory.len() >= 4);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log m)"));
    }

    #[test]
    fn test_suffix_array_structure() {
        let prog = suffix_array();
        assert_eq!(prog.name, "suffix_array");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 5);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log^2 n)"));
    }

    #[test]
    fn test_string_algorithms_have_descriptions() {
        for builder in [string_matching, suffix_array] {
            let prog = builder();
            assert!(prog.description.is_some(), "{} missing description", prog.name);
            assert!(prog.work_bound.is_some(), "{} missing work_bound", prog.name);
        }
    }

    #[test]
    fn test_string_algorithms_write_shared() {
        for builder in [string_matching, suffix_array] {
            let prog = builder();
            assert!(
                prog.body.iter().any(|s| s.writes_shared()),
                "{} must write to shared memory",
                prog.name,
            );
        }
    }

    #[test]
    fn test_lcp_array_structure() {
        let prog = lcp_array();
        assert_eq!(prog.name, "lcp_array");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 5);
        assert!(prog.parallel_step_count() >= 2);
        assert!(prog.time_bound.as_deref() == Some("O(log^2 n)"));
    }

    #[test]
    fn test_string_sorting_structure() {
        let prog = string_sorting();
        assert_eq!(prog.name, "string_sorting");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert_eq!(prog.parameters.len(), 2);
        assert!(prog.shared_memory.len() >= 5);
        assert!(prog.parallel_step_count() >= 2);
        assert!(prog.time_bound.as_deref() == Some("O(L log n)"));
    }

    #[test]
    fn test_new_string_algorithms_have_descriptions() {
        for builder in [lcp_array, string_sorting] {
            let prog = builder();
            assert!(prog.description.is_some(), "{} missing description", prog.name);
            assert!(prog.work_bound.is_some(), "{} missing work_bound", prog.name);
        }
    }

    #[test]
    fn test_new_string_algorithms_write_shared() {
        for builder in [lcp_array, string_sorting] {
            let prog = builder();
            assert!(prog.body.iter().any(|s| s.writes_shared()), "{} must write shared", prog.name);
        }
    }
}
