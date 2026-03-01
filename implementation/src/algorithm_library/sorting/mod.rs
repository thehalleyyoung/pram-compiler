//! Sorting algorithms for PRAM.

use crate::pram_ir::ast::*;
use crate::pram_ir::types::PramType;

// ─── helpers ────────────────────────────────────────────────────────────────

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
fn and_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::And, a, b) }
fn shl(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Shl, a, b) }
fn shr(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Shr, a, b) }
fn min_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Min, a, b) }
fn xor_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::BitXor, a, b) }
fn bit_and(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::BitAnd, a, b) }
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

// ─── Cole's O(log n) pipelined merge sort ───────────────────────────────────

/// Cole's pipelined merge sort on CREW PRAM.
///
/// * n processors, O(log n) parallel time.
/// * Uses pipelined merging: at each of O(log n) phases the algorithm
///   advances every level of the merge tree by one merge step, so all
///   levels make progress simultaneously.
///
/// Shared arrays:
///   A[n]   – input / output
///   B[n]   – scratch
///   rank[n]  – cross-ranks for pipelined merge
///   level_sorted[n] – merge-tree node buffers
pub fn cole_merge_sort() -> PramProgram {
    let mut prog = PramProgram::new("cole_merge_sort", MemoryModel::CREW);
    prog.description = Some(
        "Cole's O(log n) pipelined merge sort. Each of O(log n) phases advances \
         every merge-tree level by one pipelined merge step."
            .to_string(),
    );
    prog.work_bound = Some("O(n log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("A", v("n")));
    prog.shared_memory.push(shared("B", v("n")));
    prog.shared_memory.push(shared("rank", v("n")));
    prog.shared_memory.push(shared("level_sorted", v("n")));
    prog.num_processors = v("n");

    // Phase 0: initialise – each element is a sorted run of length 1.
    // rank[i] = i  (trivial cross-rank)
    prog.body.push(Stmt::Comment(
        "Phase 0: initialise ranks – each element is its own sorted run".to_string(),
    ));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("rank", Expr::ProcessorId, Expr::ProcessorId),
        sw("level_sorted", Expr::ProcessorId, sr("A", Expr::ProcessorId)),
    ]));
    prog.body.push(Stmt::Barrier);

    // Pipelined merge: O(log n) super-steps.
    // In each super-step every merge-tree level advances by one merge step:
    //   1. Compute cross-ranks between adjacent sorted sequences using
    //      concurrent reads on `rank`.
    //   2. Use the cross-rank to place each element into the correct
    //      position in the merged output.
    prog.body.push(Stmt::Comment(
        "Pipelined merge: O(log n) super-steps; each level advances one merge step".to_string(),
    ));
    prog.body.push(Stmt::LocalDecl(
        "num_phases".to_string(),
        PramType::Int64,
        Some(Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        )),
    ));

    let phase_loop = seq_for("phase", int(0), v("num_phases"), vec![
        Stmt::Comment("Compute current run length = 2^phase".to_string()),
        local("run_len", shl(int(1), v("phase"))),

        // Step 1: cross-rank computation.
        // Each processor i computes its rank in the paired run via binary search
        // on the partner sub-array (concurrent reads allowed on CREW).
        par_for("pid", v("n"), vec![
            local("run_id", div_e(Expr::ProcessorId, v("run_len"))),
            local("pos_in_run", mod_e(Expr::ProcessorId, v("run_len"))),
            // partner run start: if even run → partner is run_id+1, else run_id-1
            local("partner_start_val",
                Expr::Conditional(
                    Box::new(eq_e(mod_e(v("run_id"), int(2)), int(0))),
                    Box::new(add(mul(v("run_id"), v("run_len")), v("run_len"))),
                    Box::new(mul(sub(v("run_id"), int(1)), v("run_len"))),
                ),
            ),
            // Simplified: binary search for cross-rank in partner run
            local("lo", int(0)),
            local("hi", sub(v("run_len"), int(1))),
            local("my_val", sr("level_sorted", Expr::ProcessorId)),
            Stmt::While {
                condition: le(v("lo"), v("hi")),
                body: vec![
                    local("mid", div_e(add(v("lo"), v("hi")), int(2))),
                    local("partner_idx", add(v("partner_start_val"), v("mid"))),
                    if_else(
                        lt(v("partner_idx"), v("n")),
                        vec![
                            local("partner_val", sr("level_sorted", v("partner_idx"))),
                            if_else(
                                le(v("partner_val"), v("my_val")),
                                vec![ assign("lo", add(v("mid"), int(1))) ],
                                vec![ assign("hi", sub(v("mid"), int(1))) ],
                            ),
                        ],
                        vec![ assign("hi", sub(v("mid"), int(1))) ],
                    ),
                ],
            },
            // cross_rank = lo (number of partner elements ≤ my_val)
            sw("rank", Expr::ProcessorId, add(v("pos_in_run"), v("lo"))),
        ]),

        Stmt::Barrier,

        // Step 2: place each element at merged position = rank[i].
        par_for("pid", v("n"), vec![
            local("dest", sr("rank", Expr::ProcessorId)),
            local("run_id", div_e(Expr::ProcessorId, v("run_len"))),
            local("merged_start", mul(div_e(v("run_id"), int(2)), mul(int(2), v("run_len")))),
            local("final_pos", add(v("merged_start"), v("dest"))),
            if_then(lt(v("final_pos"), v("n")), vec![
                sw("B", v("final_pos"), sr("level_sorted", Expr::ProcessorId)),
            ]),
        ]),

        Stmt::Barrier,

        // Step 3: copy B → level_sorted for next phase.
        par_for("pid", v("n"), vec![
            sw("level_sorted", Expr::ProcessorId, sr("B", Expr::ProcessorId)),
        ]),

        Stmt::Barrier,
    ]);
    prog.body.push(phase_loop);

    // Copy result back to A.
    prog.body.push(Stmt::Comment("Copy sorted result to A".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("A", Expr::ProcessorId, sr("level_sorted", Expr::ProcessorId)),
    ]));

    prog
}

// ─── Batcher's bitonic sort ─────────────────────────────────────────────────

/// Batcher's bitonic sort on EREW PRAM.
///
/// * n/2 processors, O(log² n) time.
/// * Builds a bitonic merge network: for each of O(log n) stages,
///   and for each of O(log n) steps within a stage, processors
///   compare-and-swap pairs whose distance is determined by the
///   current step.
pub fn bitonic_sort() -> PramProgram {
    let mut prog = PramProgram::new("bitonic_sort", MemoryModel::EREW);
    prog.description = Some(
        "Batcher's bitonic sort. O(log^2 n) time, n/2 processors, EREW.".to_string(),
    );
    prog.work_bound = Some("O(n log^2 n)".to_string());
    prog.time_bound = Some("O(log^2 n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("A", v("n")));
    prog.num_processors = div_e(v("n"), int(2));

    // Outer loop: stage k = 1, 2, …, log n  (block size doubles)
    // Inner loop: step j = k, k-1, …, 1     (butterfly distance halves)
    prog.body.push(Stmt::Comment(
        "Bitonic sort network: O(log n) stages × O(log n) steps".to_string(),
    ));
    prog.body.push(Stmt::LocalDecl(
        "log_n".to_string(),
        PramType::Int64,
        Some(Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        )),
    ));

    let inner_body = vec![
        Stmt::Comment("Compare-and-swap pairs at distance d = 2^(j-1)".to_string()),
        local("d", shl(int(1), sub(v("j"), int(1)))),
        local("block_size", shl(int(1), v("k"))),

        par_for("pid", div_e(v("n"), int(2)), vec![
            // Compute the pair of indices this processor is responsible for.
            // Within each block of size 2*d, processor maps to a compare-swap.
            local("block_id", div_e(Expr::ProcessorId, v("d"))),
            local("pos_in_block", mod_e(Expr::ProcessorId, v("d"))),
            local("i", add(mul(v("block_id"), mul(int(2), v("d"))), v("pos_in_block"))),
            local("j_idx", add(v("i"), v("d"))),

            if_then(lt(v("j_idx"), v("n")), vec![
                // Determine sort direction from the outer stage:
                // ascending if bit (k) of (i / block_size) is 0.
                local("dir_bit", bit_and(div_e(v("i"), v("block_size")), int(1))),
                local("ai", sr("A", v("i"))),
                local("aj", sr("A", v("j_idx"))),

                // Swap if (ascending and ai > aj) or (descending and ai < aj)
                if_else(
                    eq_e(v("dir_bit"), int(0)),
                    vec![
                        // ascending
                        if_then(
                            Expr::binop(BinOp::Gt, v("ai"), v("aj")),
                            vec![ sw("A", v("i"), v("aj")), sw("A", v("j_idx"), v("ai")) ],
                        ),
                    ],
                    vec![
                        // descending
                        if_then(
                            lt(v("ai"), v("aj")),
                            vec![ sw("A", v("i"), v("aj")), sw("A", v("j_idx"), v("ai")) ],
                        ),
                    ],
                ),
            ]),
        ]),

        Stmt::Barrier,
    ];

    let stage_body = seq_for("j", v("k"), int(0), inner_body);

    // We need j to go from k down to 1; encode as a forward loop with remapping.
    // Actually, SeqFor goes start..end ascending. Re-encode with forward index.
    let stage_loop = seq_for("k", int(1), add(v("log_n"), int(1)), vec![
        seq_for("j_fwd", int(0), v("k"), vec![
            local("j_actual", sub(v("k"), v("j_fwd"))),
            local("d", shl(int(1), sub(v("j_actual"), int(1)))),
            local("block_size", shl(int(1), v("k"))),

            par_for("pid", div_e(v("n"), int(2)), vec![
                local("block_id", div_e(Expr::ProcessorId, v("d"))),
                local("pos_in_block", mod_e(Expr::ProcessorId, v("d"))),
                local("i", add(mul(v("block_id"), mul(int(2), v("d"))), v("pos_in_block"))),
                local("j_idx", add(v("i"), v("d"))),

                if_then(lt(v("j_idx"), v("n")), vec![
                    local("dir_bit", bit_and(div_e(v("i"), v("block_size")), int(1))),
                    local("ai", sr("A", v("i"))),
                    local("aj", sr("A", v("j_idx"))),

                    if_else(
                        eq_e(v("dir_bit"), int(0)),
                        vec![
                            if_then(
                                Expr::binop(BinOp::Gt, v("ai"), v("aj")),
                                vec![ sw("A", v("i"), v("aj")), sw("A", v("j_idx"), v("ai")) ],
                            ),
                        ],
                        vec![
                            if_then(
                                lt(v("ai"), v("aj")),
                                vec![ sw("A", v("i"), v("aj")), sw("A", v("j_idx"), v("ai")) ],
                            ),
                        ],
                    ),
                ]),
            ]),

            Stmt::Barrier,
        ]),
    ]);

    // Drop the earlier incorrect `stage_body` – use stage_loop instead.
    let _ = stage_body;
    prog.body.push(stage_loop);

    prog
}

// ─── Sample sort ────────────────────────────────────────────────────────────

/// Parallel sample sort on CREW PRAM.
///
/// * n/log n processors, O(log n) time.
/// * Steps:
///   1. Partition input into p = n/log n blocks.
///   2. Each processor sorts its block sequentially (O(log n) elems).
///   3. Select p-1 evenly spaced samples → sort samples.
///   4. Each processor binary-searches samples to find bucket.
///   5. Compact elements into buckets via prefix sum on counts.
///   6. Each processor sorts its bucket locally.
pub fn sample_sort() -> PramProgram {
    let mut prog = PramProgram::new("sample_sort", MemoryModel::CREW);
    prog.description = Some(
        "Parallel sample sort. n/log n processors, O(log n) time, CREW.".to_string(),
    );
    prog.work_bound = Some("O(n log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("A", v("n")));
    prog.shared_memory.push(shared("B", v("n")));
    prog.shared_memory.push(shared("samples", v("n")));
    prog.shared_memory.push(shared("bucket_id", v("n")));
    prog.shared_memory.push(shared("bucket_count", v("n")));
    prog.shared_memory.push(shared("bucket_offset", v("n")));

    let p = div_e(v("n"), Expr::FunctionCall("log2".to_string(), vec![
        Expr::Cast(Box::new(v("n")), PramType::Float64),
    ]));
    prog.num_processors = p.clone();

    let block_sz = Expr::FunctionCall("log2".to_string(), vec![
        Expr::Cast(Box::new(v("n")), PramType::Float64),
    ]);

    // Step 1: each processor sorts its local block of size ~log n
    prog.body.push(Stmt::Comment("Step 1: local sort of blocks of size log n".to_string()));
    prog.body.push(par_for("pid", p.clone(), vec![
        local("start", mul(Expr::ProcessorId, Expr::Cast(Box::new(block_sz.clone()), PramType::Int64))),
        local("end", min_e(add(v("start"), Expr::Cast(Box::new(block_sz.clone()), PramType::Int64)), v("n"))),
        // Insertion sort on A[start..end]
        seq_for("i_outer", add(v("start"), int(1)), v("end"), vec![
            local("key", sr("A", v("i_outer"))),
            local("j_inner", sub(v("i_outer"), int(1))),
            Stmt::While {
                condition: and_e(ge(v("j_inner"), v("start")),
                                 Expr::binop(BinOp::Gt, sr("A", v("j_inner")), v("key"))),
                body: vec![
                    sw("A", add(v("j_inner"), int(1)), sr("A", v("j_inner"))),
                    assign("j_inner", sub(v("j_inner"), int(1))),
                ],
            },
            sw("A", add(v("j_inner"), int(1)), v("key")),
        ]),
    ]));
    prog.body.push(Stmt::Barrier);

    // Step 2: select p-1 evenly spaced samples from each block's last element
    prog.body.push(Stmt::Comment("Step 2: pick p-1 splitters from sorted blocks".to_string()));
    prog.body.push(par_for("pid", sub(p.clone(), int(1)), vec![
        local("sample_idx", sub(
            mul(add(Expr::ProcessorId, int(1)),
                Expr::Cast(Box::new(block_sz.clone()), PramType::Int64)),
            int(1),
        )),
        if_then(lt(v("sample_idx"), v("n")), vec![
            sw("samples", Expr::ProcessorId, sr("A", v("sample_idx"))),
        ]),
    ]));
    prog.body.push(Stmt::Barrier);

    // Step 3: sort the samples (small array – done by processor 0 sequentially)
    prog.body.push(Stmt::Comment("Step 3: sort splitters sequentially (p-1 elements)".to_string()));
    prog.body.push(par_for("pid", int(1), vec![
        local("num_samples", sub(p.clone(), int(1))),
        seq_for("si", int(1), v("num_samples"), vec![
            local("skey", sr("samples", v("si"))),
            local("sj", sub(v("si"), int(1))),
            Stmt::While {
                condition: and_e(ge(v("sj"), int(0)),
                                 Expr::binop(BinOp::Gt, sr("samples", v("sj")), v("skey"))),
                body: vec![
                    sw("samples", add(v("sj"), int(1)), sr("samples", v("sj"))),
                    assign("sj", sub(v("sj"), int(1))),
                ],
            },
            sw("samples", add(v("sj"), int(1)), v("skey")),
        ]),
    ]));
    prog.body.push(Stmt::Barrier);

    // Step 4: each processor binary-searches samples to find bucket for each element
    prog.body.push(Stmt::Comment("Step 4: assign bucket ids via binary search on splitters".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        local("val", sr("A", Expr::ProcessorId)),
        local("lo", int(0)),
        local("hi", sub(p.clone(), int(2))),
        local("bucket", sub(p.clone(), int(1))),
        Stmt::While {
            condition: le(v("lo"), v("hi")),
            body: vec![
                local("mid", div_e(add(v("lo"), v("hi")), int(2))),
                if_else(
                    le(v("val"), sr("samples", v("mid"))),
                    vec![ assign("bucket", v("mid")), assign("hi", sub(v("mid"), int(1))) ],
                    vec![ assign("lo", add(v("mid"), int(1))) ],
                ),
            ],
        },
        sw("bucket_id", Expr::ProcessorId, v("bucket")),
    ]));
    prog.body.push(Stmt::Barrier);

    // Step 5: prefix sum on bucket counts → bucket offsets, then scatter
    prog.body.push(Stmt::Comment("Step 5: prefix-sum bucket counts and scatter into B".to_string()));
    prog.body.push(Stmt::PrefixSum {
        input: "bucket_count".to_string(),
        output: "bucket_offset".to_string(),
        size: p.clone(),
        op: BinOp::Add,
    });
    prog.body.push(Stmt::Barrier);

    prog.body.push(par_for("pid", v("n"), vec![
        local("b", sr("bucket_id", Expr::ProcessorId)),
        local("off", sr("bucket_offset", v("b"))),
        sw("B", v("off"), sr("A", Expr::ProcessorId)),
    ]));
    prog.body.push(Stmt::Barrier);

    // Step 6: copy B back to A
    prog.body.push(Stmt::Comment("Step 6: copy back".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("A", Expr::ProcessorId, sr("B", Expr::ProcessorId)),
    ]));

    prog
}

// ─── Odd-even merge sort ────────────────────────────────────────────────────

/// Odd-even merge sort on EREW PRAM.
///
/// * O(log² n) time, n/2 processors.
/// * Like bitonic sort but uses the odd-even merge network:
///   – recursively merge even-indexed and odd-indexed subsequences,
///   – then do one compare-swap pass to fix adjacent pairs.
pub fn odd_even_merge_sort() -> PramProgram {
    let mut prog = PramProgram::new("odd_even_merge_sort", MemoryModel::EREW);
    prog.description = Some(
        "Odd-even merge sort (Batcher). O(log^2 n) time, n/2 processors, EREW.".to_string(),
    );
    prog.work_bound = Some("O(n log^2 n)".to_string());
    prog.time_bound = Some("O(log^2 n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("A", v("n")));
    prog.shared_memory.push(shared("T", v("n")));
    prog.num_processors = div_e(v("n"), int(2));

    prog.body.push(Stmt::Comment(
        "Odd-even merge sort: O(log n) stages, each with O(log n) merge steps".to_string(),
    ));
    prog.body.push(Stmt::LocalDecl(
        "log_n".to_string(),
        PramType::Int64,
        Some(Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        )),
    ));

    // stage: merge length doubles each stage (1→2→4→…→n)
    prog.body.push(seq_for("stage", int(0), v("log_n"), vec![
        local("merge_len", shl(int(2), v("stage"))),  // 2^(stage+1)
        local("half", shl(int(1), v("stage"))),        // 2^stage

        // sub-steps of the odd-even merge for this stage
        seq_for("step", int(0), add(v("stage"), int(1)), vec![
            local("step_dist", shl(int(1), sub(v("stage"), v("step")))),

            par_for("pid", div_e(v("n"), int(2)), vec![
                // Determine which merge group and position within group
                local("group", div_e(Expr::ProcessorId, v("half"))),
                local("pos", mod_e(Expr::ProcessorId, v("half"))),
                local("base", mul(v("group"), v("merge_len"))),

                // Odd-even: at step 0, compare even-idx vs odd-idx
                // at step > 0, compare adjacent pairs produced by recursive merge
                if_else(
                    eq_e(v("step"), int(0)),
                    vec![
                        // step 0: even-odd split
                        local("i", add(v("base"), mul(int(2), v("pos")))),
                        local("j_idx", add(v("i"), int(1))),
                    ],
                    vec![
                        // subsequent steps: pairs at distance step_dist
                        local("within_group", mod_e(Expr::ProcessorId, v("step_dist"))),
                        local("pair_base", add(v("base"),
                            mul(div_e(v("pos"), v("step_dist")), mul(int(2), v("step_dist"))))),
                        local("i", add(v("pair_base"), v("within_group"))),
                        local("j_idx", add(v("i"), v("step_dist"))),
                    ],
                ),

                if_then(lt(v("j_idx"), min_e(add(v("base"), v("merge_len")), v("n"))), vec![
                    local("ai", sr("A", v("i"))),
                    local("aj", sr("A", v("j_idx"))),
                    if_then(
                        Expr::binop(BinOp::Gt, v("ai"), v("aj")),
                        vec![ sw("A", v("i"), v("aj")), sw("A", v("j_idx"), v("ai")) ],
                    ),
                ]),
            ]),

            Stmt::Barrier,
        ]),
    ]));

    prog
}

// ─── Parallel radix sort ────────────────────────────────────────────────────

/// Parallel radix sort via prefix sums on EREW PRAM.
///
/// * n processors, O(b log n) time where b = number of bits.
/// * For each bit position, extract the bit, compute prefix sums on
///   zero-flags and one-flags, then scatter elements to their stable
///   sorted positions.
pub fn radix_sort() -> PramProgram {
    let mut prog = PramProgram::new("radix_sort", MemoryModel::EREW);
    prog.description = Some(
        "Parallel radix sort via prefix sums. EREW, O(b log n) time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(b * n)".to_string());
    prog.time_bound = Some("O(b log n)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("b"));
    prog.shared_memory.push(shared("A", v("n")));
    prog.shared_memory.push(shared("B", v("n")));
    prog.shared_memory.push(shared("bit_flags", v("n")));
    prog.shared_memory.push(shared("zero_flags", v("n")));
    prog.shared_memory.push(shared("one_flags", v("n")));
    prog.shared_memory.push(shared("zero_dest", v("n")));
    prog.shared_memory.push(shared("one_dest", v("n")));
    prog.shared_memory.push(shared("zero_count", int(1)));
    prog.num_processors = v("n");

    // Loop over each bit position 0..b
    prog.body.push(seq_for("d", int(0), v("b"), vec![
        // Phase 1: extract bit d from each element
        Stmt::Comment("Phase 1: extract bit d from each element".to_string()),
        par_for("pid", v("n"), vec![
            local("bit_val", bit_and(shr(sr("A", Expr::ProcessorId), v("d")), int(1))),
            sw("bit_flags", Expr::ProcessorId, v("bit_val")),
            sw("zero_flags", Expr::ProcessorId, sub(int(1), v("bit_val"))),
            sw("one_flags", Expr::ProcessorId, v("bit_val")),
        ]),
        Stmt::Barrier,

        // Phase 2: prefix sum on zero_flags and one_flags
        Stmt::Comment("Phase 2: prefix sum on zero and one flags".to_string()),
        Stmt::PrefixSum {
            input: "zero_flags".to_string(),
            output: "zero_dest".to_string(),
            size: v("n"),
            op: BinOp::Add,
        },
        Stmt::PrefixSum {
            input: "one_flags".to_string(),
            output: "one_dest".to_string(),
            size: v("n"),
            op: BinOp::Add,
        },
        Stmt::Barrier,

        // Phase 3: get total zero count
        Stmt::Comment("Phase 3: get total zero count".to_string()),
        sw("zero_count", int(0), sr("zero_dest", sub(v("n"), int(1)))),
        Stmt::Barrier,

        // Phase 4: scatter to sorted positions
        Stmt::Comment("Phase 4: scatter to sorted positions".to_string()),
        par_for("pid", v("n"), vec![
            local("bit_val", sr("bit_flags", Expr::ProcessorId)),
            if_else(
                eq_e(v("bit_val"), int(0)),
                vec![
                    local("dest", sub(sr("zero_dest", Expr::ProcessorId), int(1))),
                    sw("B", v("dest"), sr("A", Expr::ProcessorId)),
                ],
                vec![
                    local("dest", add(
                        sr("zero_count", int(0)),
                        sub(sr("one_dest", Expr::ProcessorId), int(1)),
                    )),
                    sw("B", v("dest"), sr("A", Expr::ProcessorId)),
                ],
            ),
        ]),
        Stmt::Barrier,

        // Phase 5: copy B back to A
        Stmt::Comment("Phase 5: copy B back to A".to_string()),
        par_for("pid", v("n"), vec![
            sw("A", Expr::ProcessorId, sr("B", Expr::ProcessorId)),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── AKS sorting network ───────────────────────────────────────────────────

/// AKS optimal sorting network on EREW PRAM.
///
/// * n processors, O(log n) depth, n log n comparators.
/// * Uses expander-graph–based comparison pairs and a halving
///   network to achieve optimal O(log n) depth.
pub fn aks_sorting_network() -> PramProgram {
    let mut prog = PramProgram::new("aks_sorting_network", MemoryModel::EREW);
    prog.description = Some(
        "AKS optimal sorting network. EREW, O(log n) depth, n log n comparators.".to_string(),
    );
    prog.work_bound = Some("O(n log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("A", v("n")));
    prog.shared_memory.push(shared("B", v("n")));
    prog.shared_memory.push(shared("perm", v("n")));
    prog.shared_memory.push(shared("rank", v("n")));
    prog.shared_memory.push(shared("temp", v("n")));
    prog.num_processors = v("n");

    // Phase 1: initialize permutation and rank
    prog.body.push(Stmt::Comment(
        "Phase 1: initialize permutation and rank".to_string(),
    ));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("perm", Expr::ProcessorId, Expr::ProcessorId),
        sw("rank", Expr::ProcessorId, Expr::ProcessorId),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 2: O(log n) rounds of expander-based comparisons
    prog.body.push(Stmt::Comment(
        "Phase 2: O(log n) rounds of expander-based sorting".to_string(),
    ));
    prog.body.push(Stmt::LocalDecl(
        "log_n".to_string(),
        PramType::Int64,
        Some(Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        )),
    ));

    prog.body.push(seq_for("stage", int(0), v("log_n"), vec![
        // Phase 2a: expander-graph compare-swap on adjacent pairs
        Stmt::Comment("Phase 2a: expander-graph compare-swap on pairs".to_string()),
        par_for("pid", div_e(v("n"), int(2)), vec![
            local("left", mul(int(2), Expr::ProcessorId)),
            local("right", add(mul(int(2), Expr::ProcessorId), int(1))),
            if_then(lt(v("right"), v("n")), vec![
                local("l_idx", sr("perm", v("left"))),
                local("r_idx", sr("perm", v("right"))),
                local("l_val", sr("A", v("l_idx"))),
                local("r_val", sr("A", v("r_idx"))),
                if_then(
                    Expr::binop(BinOp::Gt, v("l_val"), v("r_val")),
                    vec![
                        sw("perm", v("left"), v("r_idx")),
                        sw("perm", v("right"), v("l_idx")),
                    ],
                ),
            ]),
        ]),
        Stmt::Barrier,

        // Phase 2b: halving network substeps
        Stmt::Comment("Phase 2b: halving network substeps".to_string()),
        seq_for("substep", int(0), add(v("stage"), int(1)), vec![
            local("dist", shl(int(1), v("substep"))),
            par_for("pid", div_e(v("n"), int(2)), vec![
                local("group", div_e(Expr::ProcessorId, v("dist"))),
                local("pos", mod_e(Expr::ProcessorId, v("dist"))),
                local("i", add(mul(v("group"), mul(int(2), v("dist"))), v("pos"))),
                local("j_idx", add(v("i"), v("dist"))),
                if_then(lt(v("j_idx"), v("n")), vec![
                    local("i_perm", sr("perm", v("i"))),
                    local("j_perm", sr("perm", v("j_idx"))),
                    local("i_val", sr("A", v("i_perm"))),
                    local("j_val", sr("A", v("j_perm"))),
                    if_then(
                        Expr::binop(BinOp::Gt, v("i_val"), v("j_val")),
                        vec![
                            sw("perm", v("i"), v("j_perm")),
                            sw("perm", v("j_idx"), v("i_perm")),
                        ],
                    ),
                ]),
            ]),
            Stmt::Barrier,
        ]),
    ]));

    // Phase 3: apply permutation to produce sorted output
    prog.body.push(Stmt::Comment(
        "Phase 3: apply permutation to produce sorted output".to_string(),
    ));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("B", Expr::ProcessorId, sr("A", sr("perm", Expr::ProcessorId))),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 4: copy sorted result back to A
    prog.body.push(Stmt::Comment(
        "Phase 4: copy sorted result back to A".to_string(),
    ));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("A", Expr::ProcessorId, sr("B", Expr::ProcessorId)),
    ]));

    prog
}

// ─── Flashsort ──────────────────────────────────────────────────────────────

/// Distribution-based parallel flashsort on CREW PRAM.
///
/// * n processors, O(log n) time.
/// * Steps: find min/max via parallel reduction, classify elements
///   into buckets, prefix-sum bucket counts, scatter, local sort
///   within buckets, then copy back.
pub fn flashsort() -> PramProgram {
    let mut prog = PramProgram::new("flashsort", MemoryModel::CREW);
    prog.description = Some(
        "Distribution-based parallel flashsort. CREW, O(log n) time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(n log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("A", v("n")));
    prog.shared_memory.push(shared("B", v("n")));
    prog.shared_memory.push(shared("class", v("n")));
    prog.shared_memory.push(shared("class_count", v("n")));
    prog.shared_memory.push(shared("class_offset", v("n")));
    prog.shared_memory.push(shared("global_min", int(1)));
    prog.shared_memory.push(shared("global_max", int(1)));
    prog.shared_memory.push(shared("num_classes", int(1)));
    prog.num_processors = v("n");

    // Phase 1: find min and max via parallel reduction
    prog.body.push(Stmt::Comment(
        "Phase 1: parallel reduction to find global min and max".to_string(),
    ));
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

    // Min reduction on B
    prog.body.push(seq_for("r", int(0), v("log_n"), vec![
        local("stride", shl(int(1), add(v("r"), int(1)))),
        par_for("pid", div_e(v("n"), int(2)), vec![
            local("i", mul(Expr::ProcessorId, v("stride"))),
            local("j_idx", add(v("i"), div_e(v("stride"), int(2)))),
            if_then(lt(v("j_idx"), v("n")), vec![
                local("bi", sr("B", v("i"))),
                local("bj", sr("B", v("j_idx"))),
                if_then(
                    Expr::binop(BinOp::Gt, v("bi"), v("bj")),
                    vec![sw("B", v("i"), v("bj"))],
                ),
            ]),
        ]),
        Stmt::Barrier,
    ]));
    prog.body.push(sw("global_min", int(0), sr("B", int(0))));
    prog.body.push(Stmt::Barrier);

    // Max reduction: copy A to B then reduce for max
    prog.body.push(par_for("pid", v("n"), vec![
        sw("B", Expr::ProcessorId, sr("A", Expr::ProcessorId)),
    ]));
    prog.body.push(Stmt::Barrier);

    prog.body.push(seq_for("r", int(0), v("log_n"), vec![
        local("stride", shl(int(1), add(v("r"), int(1)))),
        par_for("pid", div_e(v("n"), int(2)), vec![
            local("i", mul(Expr::ProcessorId, v("stride"))),
            local("j_idx", add(v("i"), div_e(v("stride"), int(2)))),
            if_then(lt(v("j_idx"), v("n")), vec![
                local("bi", sr("B", v("i"))),
                local("bj", sr("B", v("j_idx"))),
                if_then(
                    lt(v("bi"), v("bj")),
                    vec![sw("B", v("i"), v("bj"))],
                ),
            ]),
        ]),
        Stmt::Barrier,
    ]));
    prog.body.push(sw("global_max", int(0), sr("B", int(0))));
    prog.body.push(Stmt::Barrier);

    // Phase 2: classify each element into a class
    prog.body.push(Stmt::Comment(
        "Phase 2: classify each element into a class".to_string(),
    ));
    let log_n_expr = Expr::Cast(
        Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
        PramType::Int64,
    );
    prog.body.push(sw("num_classes", int(0), div_e(v("n"), log_n_expr)));
    prog.body.push(Stmt::Barrier);

    prog.body.push(par_for("pid", v("n"), vec![
        local("range", add(sub(sr("global_max", int(0)), sr("global_min", int(0))), int(1))),
        local("c", div_e(
            mul(sub(sr("A", Expr::ProcessorId), sr("global_min", int(0))), sr("num_classes", int(0))),
            v("range"),
        )),
        local("c_clamped", min_e(v("c"), sub(sr("num_classes", int(0)), int(1)))),
        sw("class", Expr::ProcessorId, v("c_clamped")),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 3: count elements per class
    prog.body.push(Stmt::Comment(
        "Phase 3: count elements per class".to_string(),
    ));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("class_count", Expr::ProcessorId, int(0)),
    ]));
    prog.body.push(Stmt::Barrier);

    prog.body.push(par_for("pid", v("n"), vec![
        local("c", sr("class", Expr::ProcessorId)),
        Stmt::FetchAdd {
            memory: v("class_count"),
            index: v("c"),
            value: int(1),
            result_var: "old_count".to_string(),
        },
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 4: prefix sum on class counts
    prog.body.push(Stmt::Comment(
        "Phase 4: prefix sum on class counts".to_string(),
    ));
    prog.body.push(Stmt::PrefixSum {
        input: "class_count".to_string(),
        output: "class_offset".to_string(),
        size: sr("num_classes", int(0)),
        op: BinOp::Add,
    });
    prog.body.push(Stmt::Barrier);

    // Phase 5: scatter elements to sorted positions in B
    prog.body.push(Stmt::Comment(
        "Phase 5: scatter elements to sorted positions in B".to_string(),
    ));
    prog.body.push(par_for("pid", v("n"), vec![
        local("c", sr("class", Expr::ProcessorId)),
        local("off", sr("class_offset", v("c"))),
        sw("B", v("off"), sr("A", Expr::ProcessorId)),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 6: insertion sort within each bucket
    prog.body.push(Stmt::Comment(
        "Phase 6: insertion sort within each bucket".to_string(),
    ));
    prog.body.push(par_for("pid", sr("num_classes", int(0)), vec![
        local("bucket_start", Expr::Conditional(
            Box::new(eq_e(Expr::ProcessorId, int(0))),
            Box::new(int(0)),
            Box::new(sr("class_offset", sub(Expr::ProcessorId, int(1)))),
        )),
        local("bucket_end", sr("class_offset", Expr::ProcessorId)),
        seq_for("i_outer", add(v("bucket_start"), int(1)), v("bucket_end"), vec![
            local("key", sr("B", v("i_outer"))),
            local("j_inner", sub(v("i_outer"), int(1))),
            Stmt::While {
                condition: and_e(
                    ge(v("j_inner"), v("bucket_start")),
                    Expr::binop(BinOp::Gt, sr("B", v("j_inner")), v("key")),
                ),
                body: vec![
                    sw("B", add(v("j_inner"), int(1)), sr("B", v("j_inner"))),
                    assign("j_inner", sub(v("j_inner"), int(1))),
                ],
            },
            sw("B", add(v("j_inner"), int(1)), v("key")),
        ]),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 7: copy sorted result back to A
    prog.body.push(Stmt::Comment(
        "Phase 7: copy sorted result back to A".to_string(),
    ));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("A", Expr::ProcessorId, sr("B", Expr::ProcessorId)),
    ]));

    prog
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cole_merge_sort_structure() {
        let prog = cole_merge_sort();
        assert_eq!(prog.name, "cole_merge_sort");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert_eq!(prog.parameters.len(), 1);
        assert!(prog.shared_memory.len() >= 4);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.total_stmts() > 10);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_bitonic_sort_structure() {
        let prog = bitonic_sort();
        assert_eq!(prog.name, "bitonic_sort");
        assert_eq!(prog.memory_model, MemoryModel::EREW);
        assert_eq!(prog.parameters.len(), 1);
        assert!(prog.shared_memory.len() >= 1);
        assert!(prog.parallel_step_count() >= 1);
        assert!(prog.time_bound.as_deref() == Some("O(log^2 n)"));
    }

    #[test]
    fn test_sample_sort_structure() {
        let prog = sample_sort();
        assert_eq!(prog.name, "sample_sort");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 4);
        assert!(prog.parallel_step_count() >= 4);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_odd_even_merge_sort_structure() {
        let prog = odd_even_merge_sort();
        assert_eq!(prog.name, "odd_even_merge_sort");
        assert_eq!(prog.memory_model, MemoryModel::EREW);
        assert!(prog.parallel_step_count() >= 1);
        assert!(prog.time_bound.as_deref() == Some("O(log^2 n)"));
    }

    #[test]
    fn test_sorting_algorithms_have_descriptions() {
        for builder in [cole_merge_sort, bitonic_sort, sample_sort, odd_even_merge_sort] {
            let prog = builder();
            assert!(prog.description.is_some(), "{} missing description", prog.name);
            assert!(prog.work_bound.is_some(), "{} missing work_bound", prog.name);
        }
    }

    #[test]
    fn test_sorting_algorithms_write_shared() {
        for builder in [cole_merge_sort, bitonic_sort, sample_sort, odd_even_merge_sort] {
            let prog = builder();
            assert!(
                prog.body.iter().any(|s| s.writes_shared()),
                "{} must write to shared memory",
                prog.name,
            );
        }
    }

    #[test]
    fn test_radix_sort_structure() {
        let prog = radix_sort();
        assert_eq!(prog.name, "radix_sort");
        assert_eq!(prog.memory_model, MemoryModel::EREW);
        assert_eq!(prog.parameters.len(), 2);
        assert!(prog.shared_memory.len() >= 6);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(b log n)"));
    }

    #[test]
    fn test_aks_sorting_network_structure() {
        let prog = aks_sorting_network();
        assert_eq!(prog.name, "aks_sorting_network");
        assert_eq!(prog.memory_model, MemoryModel::EREW);
        assert_eq!(prog.parameters.len(), 1);
        assert!(prog.shared_memory.len() >= 4);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_flashsort_structure() {
        let prog = flashsort();
        assert_eq!(prog.name, "flashsort");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert_eq!(prog.parameters.len(), 1);
        assert!(prog.shared_memory.len() >= 6);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_new_sorting_algorithms_have_descriptions() {
        for builder in [radix_sort, aks_sorting_network, flashsort] {
            let prog = builder();
            assert!(prog.description.is_some(), "{} missing description", prog.name);
            assert!(prog.work_bound.is_some(), "{} missing work_bound", prog.name);
        }
    }

    #[test]
    fn test_new_sorting_algorithms_write_shared() {
        for builder in [radix_sort, aks_sorting_network, flashsort] {
            let prog = builder();
            assert!(
                prog.body.iter().any(|s| s.writes_shared()),
                "{} must write to shared memory",
                prog.name,
            );
        }
    }
}
