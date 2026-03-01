//! List / sequence algorithms for PRAM.

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
fn div_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Div, a, b) }
fn lt(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Lt, a, b) }
fn ge(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Ge, a, b) }
fn eq_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Eq, a, b) }
fn ne_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Ne, a, b) }
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

fn mul(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Mul, a, b) }
fn mod_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Mod, a, b) }
fn bit_or(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::BitOr, a, b) }
fn bit_xor(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::BitXor, a, b) }
fn bit_and(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::BitAnd, a, b) }
fn le(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Le, a, b) }
fn and_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::And, a, b) }
fn shr(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Shr, a, b) }

// ─── List ranking via pointer jumping ───────────────────────────────────────

/// Pointer-jumping list ranking on EREW PRAM.
///
/// * n processors, O(log n) time.
/// * Input: `next[n]` — linked-list successor pointers (–1 = end).
/// * Output: `rank[n]` — distance from each node to the end of the list.
/// * Each round: every processor jumps its pointer two hops while
///   accumulating the weight/rank of the skipped node.
pub fn list_ranking() -> PramProgram {
    let mut prog = PramProgram::new("list_ranking", MemoryModel::EREW);
    prog.description = Some(
        "Pointer-jumping list ranking. EREW, O(log n) time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(n log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("next", v("n")));
    prog.shared_memory.push(shared("rank", v("n")));
    prog.shared_memory.push(shared("succ", v("n")));   // working copy of next
    prog.num_processors = v("n");

    // Initialise: rank = 1 if has successor, else 0; succ = next
    prog.body.push(Stmt::Comment("Initialise rank and working successor array".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        local("nxt", sr("next", Expr::ProcessorId)),
        sw("succ", Expr::ProcessorId, v("nxt")),
        if_else(
            ge(v("nxt"), int(0)),
            vec![ sw("rank", Expr::ProcessorId, int(1)) ],
            vec![ sw("rank", Expr::ProcessorId, int(0)) ],
        ),
    ]));
    prog.body.push(Stmt::Barrier);

    // O(log n) rounds of pointer jumping
    prog.body.push(Stmt::Comment("Pointer jumping: O(log n) rounds".to_string()));
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

    prog.body.push(seq_for("step", int(0), v("log_n"), vec![
        par_for("pid", v("n"), vec![
            local("s", sr("succ", Expr::ProcessorId)),
            if_then(ge(v("s"), int(0)), vec![
                local("ss", sr("succ", v("s"))),
                local("rank_s", sr("rank", v("s"))),
                // rank[pid] += rank[s]
                sw("rank", Expr::ProcessorId,
                   add(sr("rank", Expr::ProcessorId), v("rank_s"))),
                // succ[pid] = succ[s]   (jump two hops)
                sw("succ", Expr::ProcessorId, v("ss")),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Parallel prefix sum (Blelloch scan) ────────────────────────────────────

/// Parallel prefix sum (inclusive scan) on EREW PRAM.
///
/// * n/2 processors, O(log n) time.
/// * Up-sweep (reduce) phase: build partial sums bottom-up.
/// * Down-sweep phase: propagate prefix sums top-down.
///
/// Input: `A[n]`   Output: `B[n]` where B[i] = A[0] + … + A[i].
pub fn prefix_sum() -> PramProgram {
    let mut prog = PramProgram::new("prefix_sum", MemoryModel::EREW);
    prog.description = Some(
        "Parallel prefix sum (Blelloch scan). EREW, O(log n) time, n/2 processors.".to_string(),
    );
    prog.work_bound = Some("O(n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("A", v("n")));
    prog.shared_memory.push(shared("B", v("n")));
    prog.num_processors = div_e(v("n"), int(2));

    // Copy A → B
    prog.body.push(Stmt::Comment("Copy input to working array".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("B", Expr::ProcessorId, sr("A", Expr::ProcessorId)),
    ]));
    prog.body.push(Stmt::Barrier);

    prog.body.push(Stmt::LocalDecl(
        "log_n".to_string(),
        PramType::Int64,
        Some(Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        )),
    ));

    // ── Up-sweep (reduce) ──
    // For d = 0 .. log_n-1:
    //   stride = 2^(d+1)
    //   Each processor pid (0 .. n/(2*stride)-1):
    //     idx = (pid+1)*stride - 1
    //     B[idx] += B[idx - stride/2]
    prog.body.push(Stmt::Comment("Up-sweep (reduce) phase".to_string()));
    prog.body.push(seq_for("d", int(0), v("log_n"), vec![
        local("stride", shl(int(2), v("d"))),           // 2^(d+1)
        local("half_stride", shl(int(1), v("d"))),      // 2^d
        local("num_active", div_e(v("n"), v("stride"))),

        par_for("pid", v("num_active"), vec![
            local("idx", sub(
                Expr::binop(BinOp::Mul, add(Expr::ProcessorId, int(1)), v("stride")),
                int(1),
            )),
            local("left_idx", sub(v("idx"), v("half_stride"))),
            if_then(lt(v("idx"), v("n")), vec![
                local("left_val", sr("B", v("left_idx"))),
                local("right_val", sr("B", v("idx"))),
                sw("B", v("idx"), add(v("left_val"), v("right_val"))),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    // ── Down-sweep ──
    // For d = log_n-2 .. 0:
    //   stride = 2^(d+1)
    //   Each processor pid (0 .. n/stride - 1):
    //     idx = (pid+1)*stride + stride/2 - 1
    //     B[idx] += B[idx - stride/2]
    prog.body.push(Stmt::Comment("Down-sweep phase".to_string()));
    prog.body.push(seq_for("d_fwd", int(0), sub(v("log_n"), int(1)), vec![
        local("d_actual", sub(sub(v("log_n"), int(2)), v("d_fwd"))),
        local("stride", shl(int(2), v("d_actual"))),
        local("half_stride", shl(int(1), v("d_actual"))),
        local("num_active", div_e(v("n"), v("stride"))),

        par_for("pid", v("num_active"), vec![
            local("base", sub(
                Expr::binop(BinOp::Mul, add(Expr::ProcessorId, int(1)), v("stride")),
                int(1),
            )),
            local("target", add(v("base"), v("half_stride"))),
            if_then(lt(v("target"), v("n")), vec![
                local("parent_val", sr("B", v("base"))),
                local("cur_val", sr("B", v("target"))),
                sw("B", v("target"), add(v("parent_val"), v("cur_val"))),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Parallel compaction ────────────────────────────────────────────────────

/// Parallel compaction on CREW PRAM.
///
/// * O(log n) time, n processors.
/// * Given `flags[n]` (0/1), compact elements of `A[n]` where flag=1
///   into `B[count]` preserving order.
/// * Uses prefix sum on flags to compute destination indices,
///   then each flagged element writes to its destination.
pub fn compact() -> PramProgram {
    let mut prog = PramProgram::new("compact", MemoryModel::CREW);
    prog.description = Some(
        "Parallel compaction. CREW, O(log n) time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("A", v("n")));
    prog.shared_memory.push(shared("flags", v("n")));
    prog.shared_memory.push(shared("dest", v("n")));
    prog.shared_memory.push(shared("B", v("n")));
    prog.num_processors = v("n");

    // Step 1: prefix sum on flags → destination indices
    prog.body.push(Stmt::Comment("Step 1: prefix sum on flags to get destination indices".to_string()));
    prog.body.push(Stmt::PrefixSum {
        input: "flags".to_string(),
        output: "dest".to_string(),
        size: v("n"),
        op: BinOp::Add,
    });
    prog.body.push(Stmt::Barrier);

    // Step 2: scatter flagged elements
    prog.body.push(Stmt::Comment("Step 2: scatter flagged elements to their destinations".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        if_then(eq_e(sr("flags", Expr::ProcessorId), int(1)), vec![
            // dest index is prefix_sum[pid] - 1 (0-based)
            local("d", sub(sr("dest", Expr::ProcessorId), int(1))),
            sw("B", v("d"), sr("A", Expr::ProcessorId)),
        ]),
    ]));

    prog
}

// ─── Segmented scan ─────────────────────────────────────────────────────────

/// Parallel segmented prefix sum on EREW PRAM.
///
/// * n processors, O(log n) time.
/// * Input: `A[n]` — values, `seg_head[n]` — 1 at each segment start.
/// * Output: `B[n]` — prefix sums restarting at each segment boundary.
pub fn segmented_scan() -> PramProgram {
    let mut prog = PramProgram::new("segmented_scan", MemoryModel::EREW);
    prog.description = Some(
        "Parallel segmented prefix sum. EREW, O(log n) time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(n log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("A", v("n")));
    prog.shared_memory.push(shared("B", v("n")));
    prog.shared_memory.push(shared("seg_head", v("n")));
    prog.shared_memory.push(shared("temp", v("n")));
    prog.num_processors = v("n");

    // Phase 1: Copy A to B, initialise temp (seg_flag) from seg_head
    prog.body.push(Stmt::Comment("Phase 1: initialise B and segment flags".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("B", Expr::ProcessorId, sr("A", Expr::ProcessorId)),
        sw("temp", Expr::ProcessorId, sr("seg_head", Expr::ProcessorId)),
    ]));
    prog.body.push(Stmt::Barrier);

    // Compute log_n
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

    // Phase 2: O(log n) rounds of segmented pointer jumping
    prog.body.push(Stmt::Comment("Phase 2: segmented prefix sum via pointer jumping".to_string()));
    prog.body.push(seq_for("d", int(0), v("log_n"), vec![
        local("stride", shl(int(1), v("d"))),
        par_for("pid", v("n"), vec![
            if_then(ge(Expr::ProcessorId, v("stride")), vec![
                local("src", sub(Expr::ProcessorId, v("stride"))),
                local("flag_src", sr("temp", v("src"))),
                // Only add if no segment boundary blocks
                if_then(eq_e(v("flag_src"), int(0)), vec![
                    sw("B", Expr::ProcessorId,
                       add(sr("B", Expr::ProcessorId), sr("B", v("src")))),
                ]),
                // Propagate segment flags: temp[pid] |= temp[src]
                sw("temp", Expr::ProcessorId,
                   bit_or(sr("temp", Expr::ProcessorId), v("flag_src"))),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── List splitting ─────────────────────────────────────────────────────────

/// Parallel list splitting on EREW PRAM.
///
/// * n processors, O(log n) time.
/// * Input: `next[n]` — linked-list successor pointers, `color[n]` — 0 or 1.
/// * Output: `next_even[n]`, `next_odd[n]` — sublist pointers,
///           `rank_even[n]`, `rank_odd[n]` — ranks within sublists.
pub fn list_split() -> PramProgram {
    let mut prog = PramProgram::new("list_split", MemoryModel::EREW);
    prog.description = Some(
        "Parallel list splitting. EREW, O(log n) time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(n log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("next", v("n")));
    prog.shared_memory.push(shared("color", v("n")));
    prog.shared_memory.push(shared("next_even", v("n")));
    prog.shared_memory.push(shared("next_odd", v("n")));
    prog.shared_memory.push(shared("rank_even", v("n")));
    prog.shared_memory.push(shared("rank_odd", v("n")));
    prog.num_processors = v("n");

    // Phase 1: Build sublist pointers by skipping nodes of opposite color
    prog.body.push(Stmt::Comment("Phase 1: build same-color sublist pointers".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        local("nxt", sr("next", Expr::ProcessorId)),
        local("my_color", sr("color", Expr::ProcessorId)),
        // Initialise ranks to 1 if node has a successor, else 0
        if_else(
            ge(v("nxt"), int(0)),
            vec![
                // Check if immediate successor has same color
                local("nxt_color", sr("color", v("nxt"))),
                if_else(
                    eq_e(v("nxt_color"), v("my_color")),
                    // Same color: sublist next is immediate next
                    vec![
                        if_else(eq_e(v("my_color"), int(0)),
                            vec![ sw("next_even", Expr::ProcessorId, v("nxt")) ],
                            vec![ sw("next_odd", Expr::ProcessorId, v("nxt")) ],
                        ),
                    ],
                    // Different color: skip to next-next
                    vec![
                        local("nxt2", sr("next", v("nxt"))),
                        if_else(eq_e(v("my_color"), int(0)),
                            vec![ sw("next_even", Expr::ProcessorId, v("nxt2")) ],
                            vec![ sw("next_odd", Expr::ProcessorId, v("nxt2")) ],
                        ),
                    ],
                ),
                if_else(eq_e(v("my_color"), int(0)),
                    vec![ sw("rank_even", Expr::ProcessorId, int(1)) ],
                    vec![ sw("rank_odd", Expr::ProcessorId, int(1)) ],
                ),
            ],
            vec![
                // End of list
                if_else(eq_e(v("my_color"), int(0)),
                    vec![
                        sw("next_even", Expr::ProcessorId, int(-1)),
                        sw("rank_even", Expr::ProcessorId, int(0)),
                    ],
                    vec![
                        sw("next_odd", Expr::ProcessorId, int(-1)),
                        sw("rank_odd", Expr::ProcessorId, int(0)),
                    ],
                ),
            ],
        ),
    ]));
    prog.body.push(Stmt::Barrier);

    // Compute log_n
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

    // Phase 2: Pointer jumping to rank even sublist
    prog.body.push(Stmt::Comment("Phase 2: pointer jumping to rank even sublist".to_string()));
    prog.body.push(seq_for("step", int(0), v("log_n"), vec![
        par_for("pid", v("n"), vec![
            if_then(eq_e(sr("color", Expr::ProcessorId), int(0)), vec![
                local("s", sr("next_even", Expr::ProcessorId)),
                if_then(ge(v("s"), int(0)), vec![
                    sw("rank_even", Expr::ProcessorId,
                       add(sr("rank_even", Expr::ProcessorId),
                           sr("rank_even", v("s")))),
                    sw("next_even", Expr::ProcessorId,
                       sr("next_even", v("s"))),
                ]),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    // Phase 3: Pointer jumping to rank odd sublist
    prog.body.push(Stmt::Comment("Phase 3: pointer jumping to rank odd sublist".to_string()));
    prog.body.push(seq_for("step", int(0), v("log_n"), vec![
        par_for("pid", v("n"), vec![
            if_then(eq_e(sr("color", Expr::ProcessorId), int(1)), vec![
                local("s", sr("next_odd", Expr::ProcessorId)),
                if_then(ge(v("s"), int(0)), vec![
                    sw("rank_odd", Expr::ProcessorId,
                       add(sr("rank_odd", Expr::ProcessorId),
                           sr("rank_odd", v("s")))),
                    sw("next_odd", Expr::ProcessorId,
                       sr("next_odd", v("s"))),
                ]),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Symmetry breaking (Cole-Vishkin) ───────────────────────────────────────

/// Deterministic symmetry breaking (Cole-Vishkin) on EREW PRAM.
///
/// * n processors, O(log* n) time.
/// * Input: `next[n]` — linked-list successor pointers (–1 = end).
/// * Output: `color[n]` — O(1)-coloring of the list.
/// * Reduces color space from n to O(1) via iterated XOR-based recoloring.
pub fn symmetry_breaking() -> PramProgram {
    let mut prog = PramProgram::new("symmetry_breaking", MemoryModel::EREW);
    prog.description = Some(
        "Deterministic symmetry breaking (Cole-Vishkin). EREW, O(log* n) time, n processors."
            .to_string(),
    );
    prog.work_bound = Some("O(n log* n)".to_string());
    prog.time_bound = Some("O(log* n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("next", v("n")));
    prog.shared_memory.push(shared("color", v("n")));
    prog.shared_memory.push(shared("new_color", v("n")));
    prog.shared_memory.push(shared("num_rounds", int(1)));
    prog.num_processors = v("n");

    // Phase 1: Initialise color[pid] = pid
    prog.body.push(Stmt::Comment("Phase 1: unique initial coloring".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("color", Expr::ProcessorId, Expr::ProcessorId),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 2: Compute number of rounds = 3 * log(log(n) + 1) + 6
    prog.body.push(Stmt::Comment("Phase 2: compute number of Cole-Vishkin rounds".to_string()));
    prog.body.push(Stmt::LocalDecl(
        "nr".to_string(),
        PramType::Int64,
        Some(add(
            mul(
                int(3),
                Expr::Cast(
                    Box::new(log2_call(add(
                        log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64)),
                        Expr::float(1.0),
                    ))),
                    PramType::Int64,
                ),
            ),
            int(6),
        )),
    ));
    prog.body.push(sw("num_rounds", int(0), v("nr")));
    prog.body.push(Stmt::Barrier);

    // Phase 3: Main Cole-Vishkin loop
    prog.body.push(Stmt::Comment("Phase 3: Cole-Vishkin color reduction".to_string()));
    prog.body.push(seq_for("round", int(0), v("nr"), vec![
        // Each processor computes new color
        par_for("pid", v("n"), vec![
            local("c1", sr("color", Expr::ProcessorId)),
            local("nxt", sr("next", Expr::ProcessorId)),
            // Read successor's color; use own color if end of list
            if_else(
                ge(v("nxt"), int(0)),
                vec![ local("c2", sr("color", v("nxt"))) ],
                vec![ local("c2", v("c1")) ],
            ),
            // XOR to find differing bits
            local("diff", bit_xor(v("c1"), v("c2"))),
            // Find lowest set bit position via iterative check:
            // We compute bit_pos as the position of the lowest set bit
            local("bit_pos", int(0)),
            local("tmp_diff", v("diff")),
            // If diff == 0, colors are same (end of list case), keep bit_pos = 0
            if_then(ne_e(v("tmp_diff"), int(0)), vec![
                // Count trailing zeros: shift right until LSB is 1
                Stmt::While {
                    condition: eq_e(bit_and(v("tmp_diff"), int(1)), int(0)),
                    body: vec![
                        assign("tmp_diff", shr(v("tmp_diff"), int(1))),
                        assign("bit_pos", add(v("bit_pos"), int(1))),
                    ],
                },
            ]),
            // new_color = 2 * bit_pos + bit_value_of_c1_at_that_position
            local("bit_val", bit_and(shr(v("c1"), v("bit_pos")), int(1))),
            sw("new_color", Expr::ProcessorId,
               add(mul(int(2), v("bit_pos")), v("bit_val"))),
        ]),
        Stmt::Barrier,
        // Copy new_color back to color
        par_for("pid", v("n"), vec![
            sw("color", Expr::ProcessorId, sr("new_color", Expr::ProcessorId)),
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
    fn test_list_ranking_structure() {
        let prog = list_ranking();
        assert_eq!(prog.name, "list_ranking");
        assert_eq!(prog.memory_model, MemoryModel::EREW);
        assert_eq!(prog.parameters.len(), 1);
        assert!(prog.shared_memory.len() >= 3);
        assert!(prog.parallel_step_count() >= 2);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_prefix_sum_structure() {
        let prog = prefix_sum();
        assert_eq!(prog.name, "prefix_sum");
        assert_eq!(prog.memory_model, MemoryModel::EREW);
        assert!(prog.shared_memory.len() >= 2);
        // up-sweep + down-sweep: at least 2 parallel phases
        assert!(prog.parallel_step_count() >= 2);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_compact_structure() {
        let prog = compact();
        assert_eq!(prog.name, "compact");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 3);
        assert!(prog.parallel_step_count() >= 1);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_list_algorithms_have_descriptions() {
        for builder in [list_ranking, prefix_sum, compact] {
            let prog = builder();
            assert!(prog.description.is_some(), "{} missing description", prog.name);
            assert!(prog.work_bound.is_some(), "{} missing work_bound", prog.name);
        }
    }

    #[test]
    fn test_list_algorithms_write_shared() {
        for builder in [list_ranking, prefix_sum, compact] {
            let prog = builder();
            assert!(
                prog.body.iter().any(|s| s.writes_shared()),
                "{} must write to shared memory",
                prog.name,
            );
        }
    }

    #[test]
    fn test_segmented_scan_structure() {
        let prog = segmented_scan();
        assert_eq!(prog.name, "segmented_scan");
        assert_eq!(prog.memory_model, MemoryModel::EREW);
        assert!(prog.shared_memory.len() >= 3);
        assert!(prog.parallel_step_count() >= 2);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_list_split_structure() {
        let prog = list_split();
        assert_eq!(prog.name, "list_split");
        assert_eq!(prog.memory_model, MemoryModel::EREW);
        assert!(prog.shared_memory.len() >= 4);
        assert!(prog.parallel_step_count() >= 2);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_symmetry_breaking_structure() {
        let prog = symmetry_breaking();
        assert_eq!(prog.name, "symmetry_breaking");
        assert_eq!(prog.memory_model, MemoryModel::EREW);
        assert!(prog.shared_memory.len() >= 3);
        assert!(prog.parallel_step_count() >= 2);
        assert!(prog.time_bound.as_deref() == Some("O(log* n)"));
    }

    #[test]
    fn test_new_list_algorithms_have_descriptions() {
        for builder in [segmented_scan, list_split, symmetry_breaking] {
            let prog = builder();
            assert!(prog.description.is_some(), "{} missing description", prog.name);
            assert!(prog.work_bound.is_some(), "{} missing work_bound", prog.name);
        }
    }

    #[test]
    fn test_new_list_algorithms_write_shared() {
        for builder in [segmented_scan, list_split, symmetry_breaking] {
            let prog = builder();
            assert!(
                prog.body.iter().any(|s| s.writes_shared()),
                "{} must write to shared memory",
                prog.name,
            );
        }
    }
}
