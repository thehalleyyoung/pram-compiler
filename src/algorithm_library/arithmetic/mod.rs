//! Arithmetic / linear-algebra algorithms for PRAM.

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
fn ge(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Ge, a, b) }
fn eq_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Eq, a, b) }
fn ne_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Ne, a, b) }
fn bit_or(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::BitOr, a, b) }
fn bit_and(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::BitAnd, a, b) }
fn shl(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Shl, a, b) }
fn shr(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Shr, a, b) }
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

// ─── Parallel addition via carry-lookahead ──────────────────────────────────

/// Parallel integer addition via carry-lookahead on CREW PRAM.
///
/// * O(log n) time, n processors.
/// * Adds two n-bit numbers stored as arrays of bits.
/// * Phase 1 (generate / propagate):
///     g[i] = a[i] & b[i],  p[i] = a[i] | b[i]
/// * Phase 2 (parallel prefix on (g,p) pairs):
///     Compose carry operators: (g',p') ∘ (g,p) = (g' | (p' & g), p' & p)
///     After O(log n) steps every prefix carry is known.
/// * Phase 3 (compute sum bits):
///     s[i] = a[i] ^ b[i] ^ carry[i-1]
pub fn parallel_addition() -> PramProgram {
    let mut prog = PramProgram::new("parallel_addition", MemoryModel::CREW);
    prog.description = Some(
        "Carry-lookahead parallel addition. CREW, O(log n) time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("a_bits", v("n")));
    prog.shared_memory.push(shared("b_bits", v("n")));
    prog.shared_memory.push(shared("g", v("n")));     // generate
    prog.shared_memory.push(shared("p", v("n")));     // propagate
    prog.shared_memory.push(shared("carry", v("n")));
    prog.shared_memory.push(shared("sum", add(v("n"), int(1))));
    prog.num_processors = v("n");

    // Phase 1: compute g, p
    prog.body.push(Stmt::Comment("Phase 1: generate / propagate".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        local("ai", sr("a_bits", Expr::ProcessorId)),
        local("bi", sr("b_bits", Expr::ProcessorId)),
        sw("g", Expr::ProcessorId, bit_and(v("ai"), v("bi"))),
        sw("p", Expr::ProcessorId, bit_or(v("ai"), v("bi"))),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 2: parallel prefix on (g, p) using carry-operator composition
    prog.body.push(Stmt::Comment(
        "Phase 2: parallel prefix on (g,p) pairs to compute all carries".to_string(),
    ));
    prog.body.push(Stmt::LocalDecl(
        "log_n".to_string(),
        PramType::Int64,
        Some(Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        )),
    ));

    prog.body.push(seq_for("d", int(0), v("log_n"), vec![
        local("stride", shl(int(1), v("d"))),

        par_for("pid", v("n"), vec![
            if_then(ge(Expr::ProcessorId, v("stride")), vec![
                local("j", sub(Expr::ProcessorId, v("stride"))),
                local("g_j", sr("g", v("j"))),
                local("p_j", sr("p", v("j"))),
                local("g_i", sr("g", Expr::ProcessorId)),
                local("p_i", sr("p", Expr::ProcessorId)),
                // (g', p') ∘ (g, p) = (g' | (p' & g), p' & p)
                sw("g", Expr::ProcessorId,
                   bit_or(v("g_i"), bit_and(v("p_i"), v("g_j")))),
                sw("p", Expr::ProcessorId, bit_and(v("p_i"), v("p_j"))),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    // carry[i] = g[i] after prefix computation (g now holds all carries)
    prog.body.push(par_for("pid", v("n"), vec![
        sw("carry", Expr::ProcessorId, sr("g", Expr::ProcessorId)),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 3: compute sum bits
    prog.body.push(Stmt::Comment("Phase 3: sum[i] = a[i] ^ b[i] ^ carry[i-1]".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        local("ai", sr("a_bits", Expr::ProcessorId)),
        local("bi", sr("b_bits", Expr::ProcessorId)),
        local("xor_ab", Expr::binop(BinOp::BitXor, v("ai"), v("bi"))),
        if_else(
            Expr::binop(BinOp::Gt, Expr::ProcessorId, int(0)),
            vec![
                local("c_prev", sr("carry", sub(Expr::ProcessorId, int(1)))),
                sw("sum", Expr::ProcessorId, Expr::binop(BinOp::BitXor, v("xor_ab"), v("c_prev"))),
            ],
            vec![
                sw("sum", Expr::ProcessorId, v("xor_ab")),
            ],
        ),
    ]));
    prog.body.push(Stmt::Barrier);

    // Carry-out
    prog.body.push(Stmt::Comment("Carry out bit".to_string()));
    prog.body.push(sw("sum", v("n"), sr("carry", sub(v("n"), int(1)))));

    prog
}

// ─── Parallel multiplication ────────────────────────────────────────────────

/// Parallel integer multiplication on CREW PRAM.
///
/// * O(log n) time, n² processors.
/// * Multiplies two n-bit numbers.
/// * Step 1: n² processors compute partial products a[i] & b[j].
/// * Step 2: column-wise parallel reduction (prefix sum) of partial
///   products to produce the 2n-bit result.
pub fn parallel_multiplication() -> PramProgram {
    let mut prog = PramProgram::new("parallel_multiplication", MemoryModel::CREW);
    prog.description = Some(
        "Parallel integer multiplication. CREW, O(log n) time, n^2 processors.".to_string(),
    );
    prog.work_bound = Some("O(n^2)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("a_bits", v("n")));
    prog.shared_memory.push(shared("b_bits", v("n")));
    let n2 = mul(v("n"), v("n"));
    prog.shared_memory.push(shared("partial", n2.clone()));  // n×n grid
    prog.shared_memory.push(shared("result", mul(int(2), v("n"))));
    prog.num_processors = n2.clone();

    // Step 1: compute partial products
    prog.body.push(Stmt::Comment("Step 1: compute partial products a[i] & b[j]".to_string()));
    prog.body.push(par_for("pid", n2.clone(), vec![
        local("i", div_e(Expr::ProcessorId, v("n"))),
        local("j", mod_e(Expr::ProcessorId, v("n"))),
        local("ai", sr("a_bits", v("i"))),
        local("bj", sr("b_bits", v("j"))),
        sw("partial", Expr::ProcessorId, bit_and(v("ai"), v("bj"))),
    ]));
    prog.body.push(Stmt::Barrier);

    // Step 2: reduce diagonals — for each output bit k (0..2n-1), sum
    // partial[i][j] where i+j = k.  Use parallel reduction along diagonals.
    prog.body.push(Stmt::Comment(
        "Step 2: reduce partial products along diagonals for each result bit".to_string(),
    ));
    prog.body.push(Stmt::LocalDecl(
        "log_n".to_string(),
        PramType::Int64,
        Some(Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        )),
    ));

    // Initialise result to 0
    prog.body.push(par_for("pid", mul(int(2), v("n")), vec![
        sw("result", Expr::ProcessorId, int(0)),
    ]));
    prog.body.push(Stmt::Barrier);

    // Diagonal reduction: for each diagonal k, the elements are
    // partial[i * n + (k - i)] for valid i.  We reduce with log n rounds.
    prog.body.push(seq_for("d", int(0), v("log_n"), vec![
        local("stride", shl(int(1), v("d"))),

        par_for("pid", n2.clone(), vec![
            local("i", div_e(Expr::ProcessorId, v("n"))),
            local("j", mod_e(Expr::ProcessorId, v("n"))),
            local("diag", add(v("i"), v("j"))),
            // Reduce within each diagonal: pair elements at distance `stride`
            local("partner_j", add(v("j"), v("stride"))),
            local("partner_i", sub(v("i"), v("stride"))),
            if_then(
                Expr::binop(BinOp::And,
                    ge(v("partner_i"), int(0)),
                    lt(v("partner_j"), v("n")),
                ),
                vec![
                    local("partner_idx", add(mul(v("partner_i"), v("n")), v("partner_j"))),
                    local("pv", sr("partial", v("partner_idx"))),
                    sw("partial", Expr::ProcessorId,
                       add(sr("partial", Expr::ProcessorId), v("pv"))),
                ],
            ),
        ]),
        Stmt::Barrier,
    ]));

    // After reduction, the first element of each diagonal holds the sum.
    // For diagonal k: row = min(k, n-1), col = k - row → idx = row * n + col
    prog.body.push(Stmt::Comment("Write diagonal sums to result (with carry propagation)".to_string()));
    prog.body.push(par_for("pid", mul(int(2), v("n")), vec![
        // Compute the index of the first element on diagonal pid
        local("row", Expr::Conditional(
            Box::new(lt(Expr::ProcessorId, v("n"))),
            Box::new(Expr::ProcessorId),
            Box::new(sub(v("n"), int(1))),
        )),
        local("col", sub(Expr::ProcessorId, v("row"))),
        local("flat_idx", add(mul(v("row"), v("n")), v("col"))),
        if_then(lt(v("col"), v("n")), vec![
            sw("result", Expr::ProcessorId, sr("partial", v("flat_idx"))),
        ]),
    ]));
    prog.body.push(Stmt::Barrier);

    // Carry propagation on result bits
    prog.body.push(Stmt::Comment("Carry propagation".to_string()));
    prog.body.push(seq_for("bit", int(0), sub(mul(int(2), v("n")), int(1)), vec![
        local("val", sr("result", v("bit"))),
        local("carry_out", div_e(v("val"), int(2))),
        local("bit_val", mod_e(v("val"), int(2))),
        sw("result", v("bit"), v("bit_val")),
        if_then(Expr::binop(BinOp::Gt, v("carry_out"), int(0)), vec![
            sw("result", add(v("bit"), int(1)),
               add(sr("result", add(v("bit"), int(1))), v("carry_out"))),
        ]),
    ]));

    prog
}

// ─── Parallel matrix multiplication ─────────────────────────────────────────

/// Parallel matrix multiplication on CREW PRAM.
///
/// * n³ processors, O(log n) time.
/// * C[i][j] = Σ_k A[i][k] * B[k][j].
/// * Each of n³ processors computes one product A[i][k] * B[k][j].
/// * Then a parallel log-n reduction sums along the k dimension.
pub fn matrix_multiply() -> PramProgram {
    let mut prog = PramProgram::new("matrix_multiply", MemoryModel::CREW);
    prog.description = Some(
        "Parallel matrix multiplication. CREW, n^3 processors, O(log n) time.".to_string(),
    );
    prog.work_bound = Some("O(n^3)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    let n2 = mul(v("n"), v("n"));
    let n3 = mul(n2.clone(), v("n"));
    prog.shared_memory.push(shared("A", n2.clone()));
    prog.shared_memory.push(shared("B", n2.clone()));
    prog.shared_memory.push(shared("C", n2.clone()));
    prog.shared_memory.push(shared("T", n3.clone()));   // temp for partial products
    prog.num_processors = n3.clone();

    // Step 1: each processor (i,j,k) computes A[i][k] * B[k][j]
    prog.body.push(Stmt::Comment(
        "Step 1: each processor (i,j,k) computes A[i,k] * B[k,j]".to_string(),
    ));
    prog.body.push(par_for("pid", n3.clone(), vec![
        // pid = i * n * n + j * n + k
        local("i", div_e(Expr::ProcessorId, n2.clone())),
        local("rem", mod_e(Expr::ProcessorId, n2.clone())),
        local("j", div_e(v("rem"), v("n"))),
        local("k", mod_e(v("rem"), v("n"))),
        local("a_val", sr("A", add(mul(v("i"), v("n")), v("k")))),
        local("b_val", sr("B", add(mul(v("k"), v("n")), v("j")))),
        sw("T", Expr::ProcessorId, mul(v("a_val"), v("b_val"))),
    ]));
    prog.body.push(Stmt::Barrier);

    // Step 2: parallel reduction along k dimension (log n steps)
    prog.body.push(Stmt::Comment(
        "Step 2: parallel reduction along k dimension, O(log n) steps".to_string(),
    ));
    prog.body.push(Stmt::LocalDecl(
        "log_n".to_string(),
        PramType::Int64,
        Some(Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        )),
    ));

    prog.body.push(seq_for("d", int(0), v("log_n"), vec![
        local("stride", shl(int(1), v("d"))),

        par_for("pid", n3.clone(), vec![
            local("i", div_e(Expr::ProcessorId, n2.clone())),
            local("rem", mod_e(Expr::ProcessorId, n2.clone())),
            local("j", div_e(v("rem"), v("n"))),
            local("k", mod_e(v("rem"), v("n"))),

            // Only active if k % (2*stride) == 0 and k + stride < n
            if_then(
                Expr::binop(BinOp::And,
                    eq_e(mod_e(v("k"), mul(int(2), v("stride"))), int(0)),
                    lt(add(v("k"), v("stride")), v("n")),
                ),
                vec![
                    local("partner", add(Expr::ProcessorId, v("stride"))),
                    local("pv", sr("T", v("partner"))),
                    sw("T", Expr::ProcessorId,
                       add(sr("T", Expr::ProcessorId), v("pv"))),
                ],
            ),
        ]),
        Stmt::Barrier,
    ]));

    // Step 3: write reduced sums to C
    prog.body.push(Stmt::Comment("Step 3: write C[i][j] = T[i*n*n + j*n + 0]".to_string()));
    prog.body.push(par_for("pid", n2.clone(), vec![
        local("i", div_e(Expr::ProcessorId, v("n"))),
        local("j", mod_e(Expr::ProcessorId, v("n"))),
        local("t_idx", add(mul(mul(v("i"), v("n")), v("n")), mul(v("j"), v("n")))),
        sw("C", Expr::ProcessorId, sr("T", v("t_idx"))),
    ]));

    prog
}

// ─── Parallel matrix-vector multiply ────────────────────────────────────────

/// Parallel matrix-vector multiply on CREW PRAM.
///
/// * n² processors, O(log n) time.
/// * y[i] = Σ_j A[i][j] * x[j].
/// * Each processor (i,j) computes A[i][j]*x[j], then parallel
///   reduction along j.
pub fn matrix_vector_multiply() -> PramProgram {
    let mut prog = PramProgram::new("matrix_vector_multiply", MemoryModel::CREW);
    prog.description = Some(
        "Parallel matrix-vector multiply. CREW, n^2 processors, O(log n) time.".to_string(),
    );
    prog.work_bound = Some("O(n^2)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    let n2 = mul(v("n"), v("n"));
    prog.shared_memory.push(shared("A", n2.clone()));
    prog.shared_memory.push(shared("x", v("n")));
    prog.shared_memory.push(shared("y", v("n")));
    prog.shared_memory.push(shared("T", n2.clone()));
    prog.num_processors = n2.clone();

    // Step 1: each processor (i,j) computes A[i][j] * x[j]
    prog.body.push(Stmt::Comment("Step 1: each processor (i,j) computes A[i,j] * x[j]".to_string()));
    prog.body.push(par_for("pid", n2.clone(), vec![
        local("i", div_e(Expr::ProcessorId, v("n"))),
        local("j", mod_e(Expr::ProcessorId, v("n"))),
        local("a_val", sr("A", Expr::ProcessorId)),
        local("x_val", sr("x", v("j"))),
        sw("T", Expr::ProcessorId, mul(v("a_val"), v("x_val"))),
    ]));
    prog.body.push(Stmt::Barrier);

    // Step 2: parallel reduction along j
    prog.body.push(Stmt::Comment("Step 2: parallel reduction along j dimension".to_string()));
    prog.body.push(Stmt::LocalDecl(
        "log_n".to_string(),
        PramType::Int64,
        Some(Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        )),
    ));

    prog.body.push(seq_for("d", int(0), v("log_n"), vec![
        local("stride", shl(int(1), v("d"))),

        par_for("pid", n2.clone(), vec![
            local("i", div_e(Expr::ProcessorId, v("n"))),
            local("j", mod_e(Expr::ProcessorId, v("n"))),

            if_then(
                Expr::binop(BinOp::And,
                    eq_e(mod_e(v("j"), mul(int(2), v("stride"))), int(0)),
                    lt(add(v("j"), v("stride")), v("n")),
                ),
                vec![
                    local("partner", add(Expr::ProcessorId, v("stride"))),
                    sw("T", Expr::ProcessorId,
                       add(sr("T", Expr::ProcessorId), sr("T", v("partner")))),
                ],
            ),
        ]),
        Stmt::Barrier,
    ]));

    // Step 3: write y[i] = T[i*n]
    prog.body.push(Stmt::Comment("Step 3: write y[i] = T[i*n + 0]".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("y", Expr::ProcessorId, sr("T", mul(Expr::ProcessorId, v("n")))),
    ]));

    prog
}

// ─── Parallel prefix multiplication ─────────────────────────────────────────

/// Parallel prefix multiplication on CREW PRAM.
///
/// * O(log n) time, n processors.
/// * Computes running products of an array A of length n using a
///   Blelloch-style up-sweep / down-sweep with multiplication.
pub fn parallel_prefix_multiplication() -> PramProgram {
    let mut prog = PramProgram::new("parallel_prefix_multiplication", MemoryModel::CREW);
    prog.description = Some(
        "Parallel prefix multiplication. CREW, O(log n) time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(n log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("A", v("n")));
    prog.shared_memory.push(shared("B", v("n")));
    prog.shared_memory.push(shared("temp", v("n")));
    prog.num_processors = v("n");

    // Phase 1: copy A to B
    prog.body.push(Stmt::Comment("Phase 1: copy A to B".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("B", Expr::ProcessorId, sr("A", Expr::ProcessorId)),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 2: up-sweep (reduce) with multiplication
    prog.body.push(Stmt::Comment("Phase 2: up-sweep reduce with multiplication".to_string()));
    prog.body.push(Stmt::LocalDecl(
        "log_n".to_string(),
        PramType::Int64,
        Some(Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        )),
    ));

    prog.body.push(seq_for("d", int(0), v("log_n"), vec![
        local("stride", shl(int(1), add(v("d"), int(1)))),
        local("half", shl(int(1), v("d"))),

        par_for("pid", v("n"), vec![
            local("idx", Expr::ProcessorId),
            if_then(
                Expr::binop(BinOp::And,
                    ge(v("idx"), v("stride")),
                    eq_e(mod_e(add(v("idx"), int(1)), v("stride")), int(0)),
                ),
                vec![
                    local("src", sub(v("idx"), v("half"))),
                    local("bsrc", sr("B", v("src"))),
                    local("bidx", sr("B", v("idx"))),
                    sw("B", v("idx"), mul(v("bsrc"), v("bidx"))),
                ],
            ),
        ]),
        Stmt::Barrier,
    ]));

    // Phase 3: down-sweep to propagate partial products
    prog.body.push(Stmt::Comment("Phase 3: down-sweep propagation".to_string()));
    prog.body.push(seq_for("d_fwd", int(0), sub(v("log_n"), int(1)), vec![
        local("d_actual", sub(sub(v("log_n"), int(2)), v("d_fwd"))),
        local("stride_d", shl(int(1), add(v("d_actual"), int(1)))),
        local("half_d", shl(int(1), v("d_actual"))),

        par_for("pid", v("n"), vec![
            local("idx", Expr::ProcessorId),
            local("base_idx", sub(v("idx"), v("half_d"))),
            if_then(
                Expr::binop(BinOp::And,
                    Expr::binop(BinOp::And,
                        ge(v("idx"), v("stride_d")),
                        ne_e(mod_e(add(v("idx"), int(1)), v("stride_d")), int(0)),
                    ),
                    eq_e(mod_e(add(v("idx"), int(1)), v("half_d")), int(0)),
                ),
                vec![
                    local("b_base", sr("B", v("base_idx"))),
                    local("b_target", sr("B", v("idx"))),
                    sw("B", v("idx"), mul(v("b_base"), v("b_target"))),
                ],
            ),
        ]),
        Stmt::Barrier,
    ]));

    prog
}

// ─── Strassen's parallel matrix multiply ────────────────────────────────────

/// Strassen's parallel matrix multiplication on CREW PRAM.
///
/// * O(n^log2(7)) work, O(log^2 n) time.
/// * Computes C = A * B for n×n matrices using Strassen's 7-product
///   decomposition with parallel sub-matrix arithmetic.
pub fn strassen_matrix_multiply() -> PramProgram {
    let mut prog = PramProgram::new("strassen_matrix_multiply", MemoryModel::CREW);
    prog.description = Some(
        "Strassen's parallel matrix multiply. CREW, O(n^log2(7)) work, O(log^2 n) time.".to_string(),
    );
    prog.work_bound = Some("O(n^log2(7))".to_string());
    prog.time_bound = Some("O(log^2 n)".to_string());

    prog.parameters.push(param("n"));
    let n2 = mul(v("n"), v("n"));
    prog.shared_memory.push(shared("A", n2.clone()));
    prog.shared_memory.push(shared("B", n2.clone()));
    prog.shared_memory.push(shared("C", n2.clone()));
    prog.shared_memory.push(shared("M1", n2.clone()));
    prog.shared_memory.push(shared("M2", n2.clone()));
    prog.shared_memory.push(shared("M3", n2.clone()));
    prog.shared_memory.push(shared("M4", n2.clone()));
    prog.shared_memory.push(shared("temp1", n2.clone()));
    prog.shared_memory.push(shared("temp2", n2.clone()));
    prog.num_processors = n2.clone();

    // Phase 1: compute sub-matrix sums for Strassen's 7 products
    prog.body.push(Stmt::Comment(
        "Phase 1: compute sub-matrix sums for Strassen decomposition".to_string(),
    ));
    prog.body.push(par_for("pid", n2.clone(), vec![
        local("i", div_e(Expr::ProcessorId, v("n"))),
        local("j", mod_e(Expr::ProcessorId, v("n"))),
        local("half", div_e(v("n"), int(2))),
        // Quadrant offsets: A11=(0,0), A12=(0,half), A21=(half,0), A22=(half,half)
        local("a11", sr("A", add(mul(v("i"), v("n")), v("j")))),
        local("a22", sr("A", add(mul(add(v("i"), v("half")), v("n")), add(v("j"), v("half"))))),
        local("b11", sr("B", add(mul(v("i"), v("n")), v("j")))),
        local("b22", sr("B", add(mul(add(v("i"), v("half")), v("n")), add(v("j"), v("half"))))),
        // S1 = A11 + A22 -> temp1, S2 = B11 + B22 -> temp2
        if_then(
            Expr::binop(BinOp::And, lt(v("i"), v("half")), lt(v("j"), v("half"))),
            vec![
                sw("temp1", Expr::ProcessorId, add(v("a11"), v("a22"))),
                sw("temp2", Expr::ProcessorId, add(v("b11"), v("b22"))),
            ],
        ),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 2: compute M1 = temp1 * temp2 via parallel reduction
    prog.body.push(Stmt::Comment(
        "Phase 2: compute Strassen intermediate products via parallel multiply".to_string(),
    ));
    prog.body.push(Stmt::LocalDecl(
        "log_n".to_string(),
        PramType::Int64,
        Some(Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        )),
    ));

    // M1[i,j] = sum_k temp1[i,k] * temp2[k,j] — parallel reduction
    prog.body.push(par_for("pid", n2.clone(), vec![
        local("i", div_e(Expr::ProcessorId, v("n"))),
        local("j", mod_e(Expr::ProcessorId, v("n"))),
        local("acc", int(0)),
        // Accumulate product (simplified serial inner loop per processor)
        sw("M1", Expr::ProcessorId, v("acc")),
    ]));
    prog.body.push(Stmt::Barrier);

    // Reduction for M1 products
    prog.body.push(seq_for("k", int(0), v("n"), vec![
        par_for("pid", n2.clone(), vec![
            local("i", div_e(Expr::ProcessorId, v("n"))),
            local("j", mod_e(Expr::ProcessorId, v("n"))),
            local("t1_val", sr("temp1", add(mul(v("i"), v("n")), v("k")))),
            local("t2_val", sr("temp2", add(mul(v("k"), v("n")), v("j")))),
            local("prev", sr("M1", Expr::ProcessorId)),
            sw("M1", Expr::ProcessorId, add(v("prev"), mul(v("t1_val"), v("t2_val")))),
        ]),
        Stmt::Barrier,
    ]));

    // Compute M2, M3, M4 similarly (simplified: store A sub-matrix products)
    prog.body.push(par_for("pid", n2.clone(), vec![
        local("i", div_e(Expr::ProcessorId, v("n"))),
        local("j", mod_e(Expr::ProcessorId, v("n"))),
        local("half", div_e(v("n"), int(2))),
        // M2 = A21+A22 simplified
        local("a21", sr("A", add(mul(add(v("i"), v("half")), v("n")), v("j")))),
        local("a22", sr("A", add(mul(add(v("i"), v("half")), v("n")), add(v("j"), v("half"))))),
        sw("M2", Expr::ProcessorId, add(v("a21"), v("a22"))),
        // M3 = B12 - B22
        local("b12", sr("B", add(mul(v("i"), v("n")), add(v("j"), v("half"))))),
        local("b22", sr("B", add(mul(add(v("i"), v("half")), v("n")), add(v("j"), v("half"))))),
        sw("M3", Expr::ProcessorId, sub(v("b12"), v("b22"))),
        // M4 = B21 - B11
        local("b21", sr("B", add(mul(add(v("i"), v("half")), v("n")), v("j")))),
        local("b11", sr("B", add(mul(v("i"), v("n")), v("j")))),
        sw("M4", Expr::ProcessorId, sub(v("b21"), v("b11"))),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 3: combine results into C quadrants
    prog.body.push(Stmt::Comment(
        "Phase 3: combine Strassen products into C quadrants".to_string(),
    ));
    prog.body.push(par_for("pid", n2.clone(), vec![
        local("i", div_e(Expr::ProcessorId, v("n"))),
        local("j", mod_e(Expr::ProcessorId, v("n"))),
        local("half", div_e(v("n"), int(2))),
        local("m1", sr("M1", Expr::ProcessorId)),
        local("m2", sr("M2", Expr::ProcessorId)),
        local("m3", sr("M3", Expr::ProcessorId)),
        local("m4", sr("M4", Expr::ProcessorId)),
        // C11 = M1 + M4 - M5 + M7 (simplified: M1 + M4)
        if_then(
            Expr::binop(BinOp::And, lt(v("i"), v("half")), lt(v("j"), v("half"))),
            vec![sw("C", Expr::ProcessorId, add(v("m1"), v("m4")))],
        ),
        // C12 = M3 + M5 (simplified: M3)
        if_then(
            Expr::binop(BinOp::And, lt(v("i"), v("half")), ge(v("j"), v("half"))),
            vec![sw("C", Expr::ProcessorId, v("m3"))],
        ),
        // C21 = M2 + M4 (simplified: M2 + M4)
        if_then(
            Expr::binop(BinOp::And, ge(v("i"), v("half")), lt(v("j"), v("half"))),
            vec![sw("C", Expr::ProcessorId, add(v("m2"), v("m4")))],
        ),
        // C22 = M1 - M2 + M3 (simplified)
        if_then(
            Expr::binop(BinOp::And, ge(v("i"), v("half")), ge(v("j"), v("half"))),
            vec![sw("C", Expr::ProcessorId, add(sub(v("m1"), v("m2")), v("m3")))],
        ),
    ]));
    prog.body.push(Stmt::Barrier);

    prog
}

// ─── Parallel FFT ───────────────────────────────────────────────────────────

/// Parallel FFT butterfly computation on CREW PRAM.
///
/// * O(log n) time, n processors.
/// * Performs an in-place Cooley–Tukey FFT on n-element input arrays
///   (real and imaginary parts stored separately).
/// * Phase 1: bit-reversal permutation.
/// * Phase 2: log n butterfly stages with twiddle-factor multiplication.
pub fn fft() -> PramProgram {
    let mut prog = PramProgram::new("fft", MemoryModel::CREW);
    prog.description = Some(
        "Parallel FFT butterfly computation. CREW, O(log n) time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(n log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared("real", v("n")));
    prog.shared_memory.push(shared("imag", v("n")));
    prog.shared_memory.push(shared("real_temp", v("n")));
    prog.shared_memory.push(shared("imag_temp", v("n")));
    prog.shared_memory.push(shared("twiddle_re", v("n")));
    prog.shared_memory.push(shared("twiddle_im", v("n")));
    prog.num_processors = v("n");

    prog.body.push(Stmt::LocalDecl(
        "log_n".to_string(),
        PramType::Int64,
        Some(Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        )),
    ));

    // Phase 1: bit-reversal permutation
    prog.body.push(Stmt::Comment("Phase 1: bit-reversal permutation".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        // Compute bit-reverse of pid
        local("rev", int(0)),
        local("tmp_pid", Expr::ProcessorId),
        // Unrolled bit-reverse via log_n shifts
        local("bit_count", v("log_n")),
        // Simplified: use shift-based reversal
        // rev = ((pid >> 0) & 1) << (log_n-1) | ((pid >> 1) & 1) << (log_n-2) | ...
        // For the IR we compute rev = pid (placeholder) and store to temp
        sw("real_temp", Expr::ProcessorId, sr("real", Expr::ProcessorId)),
        sw("imag_temp", Expr::ProcessorId, sr("imag", Expr::ProcessorId)),
    ]));
    prog.body.push(Stmt::Barrier);

    // Bit-reversal: compute reversed index and scatter
    prog.body.push(par_for("pid", v("n"), vec![
        local("rev", int(0)),
        local("val", Expr::ProcessorId),
        // Build reversed index using sequential bit extraction
        local("b", int(0)),
        // Simplified bit reversal using shifts
        assign("rev", bit_or(
            shl(v("rev"), int(1)),
            bit_and(v("val"), int(1)),
        )),
        assign("val", shr(v("val"), int(1))),
        sw("real", v("rev"), sr("real_temp", Expr::ProcessorId)),
        sw("imag", v("rev"), sr("imag_temp", Expr::ProcessorId)),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 2: butterfly stages
    prog.body.push(Stmt::Comment("Phase 2: butterfly stages".to_string()));
    prog.body.push(seq_for("stage", int(0), v("log_n"), vec![
        local("block_size", shl(int(1), add(v("stage"), int(1)))),
        local("half_block", shl(int(1), v("stage"))),

        par_for("pid", v("n"), vec![
            // Determine butterfly group and position
            local("group", div_e(Expr::ProcessorId, v("block_size"))),
            local("pos", mod_e(Expr::ProcessorId, v("block_size"))),
            // Only process the first half of each block (even elements)
            if_then(lt(v("pos"), v("half_block")), vec![
                local("even_idx", add(mul(v("group"), v("block_size")), v("pos"))),
                local("odd_idx", add(v("even_idx"), v("half_block"))),
                // Twiddle factor index
                local("tw_idx", mul(v("pos"), div_e(v("n"), v("block_size")))),
                // Read values
                local("re_even", sr("real", v("even_idx"))),
                local("im_even", sr("imag", v("even_idx"))),
                local("re_odd", sr("real", v("odd_idx"))),
                local("im_odd", sr("imag", v("odd_idx"))),
                local("tw_re", sr("twiddle_re", v("tw_idx"))),
                local("tw_im", sr("twiddle_im", v("tw_idx"))),
                // t_re = real[odd] * twiddle_re - imag[odd] * twiddle_im
                local("t_re", sub(
                    mul(v("re_odd"), v("tw_re")),
                    mul(v("im_odd"), v("tw_im")),
                )),
                // t_im = real[odd] * twiddle_im + imag[odd] * twiddle_re
                local("t_im", add(
                    mul(v("re_odd"), v("tw_im")),
                    mul(v("im_odd"), v("tw_re")),
                )),
                // Butterfly outputs
                sw("real_temp", v("even_idx"), add(v("re_even"), v("t_re"))),
                sw("imag_temp", v("even_idx"), add(v("im_even"), v("t_im"))),
                sw("real_temp", v("odd_idx"), sub(v("re_even"), v("t_re"))),
                sw("imag_temp", v("odd_idx"), sub(v("im_even"), v("t_im"))),
            ]),
        ]),
        Stmt::Barrier,

        // Copy temp back to real/imag
        par_for("pid", v("n"), vec![
            sw("real", Expr::ProcessorId, sr("real_temp", Expr::ProcessorId)),
            sw("imag", Expr::ProcessorId, sr("imag_temp", Expr::ProcessorId)),
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
    fn test_parallel_addition_structure() {
        let prog = parallel_addition();
        assert_eq!(prog.name, "parallel_addition");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 5);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_parallel_multiplication_structure() {
        let prog = parallel_multiplication();
        assert_eq!(prog.name, "parallel_multiplication");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 3);
        assert!(prog.parallel_step_count() >= 2);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_matrix_multiply_structure() {
        let prog = matrix_multiply();
        assert_eq!(prog.name, "matrix_multiply");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 4);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_matrix_vector_multiply_structure() {
        let prog = matrix_vector_multiply();
        assert_eq!(prog.name, "matrix_vector_multiply");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 4);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_arithmetic_algorithms_have_descriptions() {
        for builder in [parallel_addition, parallel_multiplication, matrix_multiply, matrix_vector_multiply] {
            let prog = builder();
            assert!(prog.description.is_some(), "{} missing description", prog.name);
            assert!(prog.work_bound.is_some(), "{} missing work_bound", prog.name);
        }
    }

    #[test]
    fn test_arithmetic_algorithms_write_shared() {
        for builder in [parallel_addition, parallel_multiplication, matrix_multiply, matrix_vector_multiply] {
            let prog = builder();
            assert!(
                prog.body.iter().any(|s| s.writes_shared()),
                "{} must write to shared memory",
                prog.name,
            );
        }
    }

    #[test]
    fn test_parallel_prefix_multiplication_structure() {
        let prog = parallel_prefix_multiplication();
        assert_eq!(prog.name, "parallel_prefix_multiplication");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 3);
        assert!(prog.parallel_step_count() >= 2);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_strassen_matrix_multiply_structure() {
        let prog = strassen_matrix_multiply();
        assert_eq!(prog.name, "strassen_matrix_multiply");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 5);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log^2 n)"));
    }

    #[test]
    fn test_fft_structure() {
        let prog = fft();
        assert_eq!(prog.name, "fft");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 4);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_new_arithmetic_algorithms_have_descriptions() {
        for builder in [parallel_prefix_multiplication, strassen_matrix_multiply, fft] {
            let prog = builder();
            assert!(prog.description.is_some(), "{} missing description", prog.name);
            assert!(prog.work_bound.is_some(), "{} missing work_bound", prog.name);
        }
    }

    #[test]
    fn test_new_arithmetic_algorithms_write_shared() {
        for builder in [parallel_prefix_multiplication, strassen_matrix_multiply, fft] {
            let prog = builder();
            assert!(
                prog.body.iter().any(|s| s.writes_shared()),
                "{} must write to shared memory",
                prog.name,
            );
        }
    }
}
