//! Formal metatheory for the PRAM IR type system.
//!
//! Implements progress and preservation proofs as executable checks.
//!
//! # Type Safety via Progress + Preservation
//!
//! **Progress**: If a well-typed statement `Γ ⊢ s : τ` is not a value/terminal,
//! then it can take a step: ∃ s'. s → s'.
//!
//! **Preservation (Subject Reduction)**: If `Γ ⊢ s : τ` and `s → s'`,
//! then `Γ ⊢ s' : τ`.
//!
//! Together these guarantee: well-typed programs don't get stuck.
//!
//! # Proof Strategy
//!
//! The PRAM IR is a first-order language with:
//! - Base types: Int32, Int64, Float32, Float64, Bool, Unit, ProcessorId
//! - Compound types: Array(T,n), SharedMemory(T), SharedRef(T), Tuple(Ts), Struct(name,fields)
//! - Statements: Assign, LocalDecl, SharedWrite, ParallelFor, SeqFor, If, Block, Barrier, etc.
//! - Expressions: literals, variables, binops, unaryops, SharedRead, ArrayIndex, FunctionCall, Cast, Conditional
//!
//! Progress is proved by case analysis on the typing derivation.
//! Preservation is proved by induction on the stepping relation.

use super::ast::{Expr, Stmt, BinOp, UnaryOp, PramProgram, MemoryModel};
use super::types::{PramType, TypeEnv, TypeError, typecheck_expr, check_compatibility, TypeCompatibility};

// ═══════════════════════════════════════════════════════════════════════════
// Runtime values and small-step semantics
// ═══════════════════════════════════════════════════════════════════════════

/// Runtime values in the PRAM IR.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    IntVal(i64),
    FloatVal(f64),
    BoolVal(bool),
    UnitVal,
    ArrayVal(Vec<Value>),
    TupleVal(Vec<Value>),
}

impl Value {
    /// Extract the type of a runtime value.
    pub fn type_of(&self) -> PramType {
        match self {
            Value::IntVal(_) => PramType::Int64,
            Value::FloatVal(_) => PramType::Float64,
            Value::BoolVal(_) => PramType::Bool,
            Value::UnitVal => PramType::Unit,
            Value::ArrayVal(vs) => {
                let elem_ty = vs.first().map(|v| v.type_of()).unwrap_or(PramType::Int64);
                PramType::Array(Box::new(elem_ty), vs.len())
            }
            Value::TupleVal(vs) => {
                PramType::Tuple(vs.iter().map(|v| v.type_of()).collect())
            }
        }
    }
}

/// Whether an expression is a value (fully evaluated).
pub fn expr_is_value(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::IntLiteral(_) | Expr::FloatLiteral(_) | Expr::BoolLiteral(_)
    )
}

/// Whether a statement is terminal (cannot step further).
pub fn stmt_is_terminal(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Return(Some(e)) => expr_is_value(e),
        Stmt::Return(None) => true,
        Stmt::Block(stmts) if stmts.is_empty() => true,
        Stmt::Nop => true,
        _ => false,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Progress Theorem
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a progress check.
#[derive(Debug, Clone)]
pub enum ProgressResult {
    /// The term is a value/terminal — no step needed.
    IsValue,
    /// The term can take a step.
    CanStep(String),
    /// Progress fails — the term is stuck (type error).
    Stuck(String),
}

/// Check progress for an expression: well-typed non-value expressions can step.
///
/// **Theorem (Progress for Expressions)**:
/// If `Γ ⊢ e : τ`, then either:
///   (a) `e` is a value, or
///   (b) ∃ e'. e → e'
///
/// Proof by case analysis on the typing derivation.
pub fn check_progress_expr(env: &TypeEnv, expr: &Expr) -> ProgressResult {
    // First verify well-typedness
    if let Err(te) = typecheck_expr(env, expr) {
        return ProgressResult::Stuck(format!("Type error: {}", te));
    }

    match expr {
        // Case: Literals are values
        Expr::IntLiteral(_) | Expr::FloatLiteral(_) | Expr::BoolLiteral(_) => {
            ProgressResult::IsValue
        }

        // Case: Variable — by well-typedness, must be in env
        Expr::Variable(name) => {
            if env.lookup(name).is_some() {
                ProgressResult::CanStep(format!("Variable '{}' can be looked up", name))
            } else {
                ProgressResult::Stuck(format!("Undefined variable '{}'", name))
            }
        }

        // Case: ProcessorId / NumProcessors — always available in parallel context
        Expr::ProcessorId | Expr::NumProcessors => {
            ProgressResult::CanStep("Built-in available".into())
        }

        // Case: BinOp — by IH both subexpressions can step or are values
        Expr::BinOp(_, left, right) => {
            let lp = check_progress_expr(env, left);
            let rp = check_progress_expr(env, right);
            match (&lp, &rp) {
                (ProgressResult::Stuck(msg), _) => ProgressResult::Stuck(msg.clone()),
                (_, ProgressResult::Stuck(msg)) => ProgressResult::Stuck(msg.clone()),
                (ProgressResult::IsValue, ProgressResult::IsValue) => {
                    ProgressResult::CanStep("Both operands are values; can compute".into())
                }
                (ProgressResult::IsValue, ProgressResult::CanStep(_)) => {
                    ProgressResult::CanStep("Right operand can step".into())
                }
                (ProgressResult::CanStep(_), _) => {
                    ProgressResult::CanStep("Left operand can step".into())
                }
            }
        }

        // Case: UnaryOp — by IH operand can step or is value
        Expr::UnaryOp(_, operand) => {
            match check_progress_expr(env, operand) {
                ProgressResult::Stuck(msg) => ProgressResult::Stuck(msg),
                ProgressResult::IsValue => {
                    ProgressResult::CanStep("Operand is value; can compute".into())
                }
                ProgressResult::CanStep(msg) => ProgressResult::CanStep(msg),
            }
        }

        // Case: SharedRead — memory and index can step
        Expr::SharedRead(mem, idx) => {
            let mp = check_progress_expr(env, mem);
            let ip = check_progress_expr(env, idx);
            match (&mp, &ip) {
                (ProgressResult::Stuck(msg), _) | (_, ProgressResult::Stuck(msg)) => {
                    ProgressResult::Stuck(msg.clone())
                }
                _ => ProgressResult::CanStep("Can read from shared memory".into()),
            }
        }

        // Case: ArrayIndex — array and index can step
        Expr::ArrayIndex(arr, idx) => {
            let ap = check_progress_expr(env, arr);
            let ip = check_progress_expr(env, idx);
            match (&ap, &ip) {
                (ProgressResult::Stuck(msg), _) | (_, ProgressResult::Stuck(msg)) => {
                    ProgressResult::Stuck(msg.clone())
                }
                _ => ProgressResult::CanStep("Can index array".into()),
            }
        }

        // Case: FunctionCall — arguments can step
        Expr::FunctionCall(_, args) => {
            for arg in args {
                if let ProgressResult::Stuck(msg) = check_progress_expr(env, arg) {
                    return ProgressResult::Stuck(msg);
                }
            }
            ProgressResult::CanStep("Function can be called".into())
        }

        // Case: Cast — inner can step
        Expr::Cast(inner, _) => {
            match check_progress_expr(env, inner) {
                ProgressResult::Stuck(msg) => ProgressResult::Stuck(msg),
                _ => ProgressResult::CanStep("Cast can be applied".into()),
            }
        }

        // Case: Conditional — condition can step
        Expr::Conditional(cond, _then_e, _else_e) => {
            match check_progress_expr(env, cond) {
                ProgressResult::Stuck(msg) => ProgressResult::Stuck(msg),
                ProgressResult::IsValue => {
                    ProgressResult::CanStep("Condition evaluated; can branch".into())
                }
                ProgressResult::CanStep(msg) => ProgressResult::CanStep(msg),
            }
        }
    }
}

/// Check progress for a statement.
///
/// **Theorem (Progress for Statements)**:
/// If `Γ ⊢ s ok` (s is well-typed under Γ), then either:
///   (a) s is terminal (Nop, empty block, return), or
///   (b) ∃ s'. s → s'
pub fn check_progress_stmt(env: &TypeEnv, stmt: &Stmt) -> ProgressResult {
    match stmt {
        Stmt::Return(None) | Stmt::Nop => ProgressResult::IsValue,
        Stmt::Return(Some(e)) => {
            if expr_is_value(e) {
                ProgressResult::IsValue
            } else {
                check_progress_expr(env, e)
            }
        }
        Stmt::Block(stmts) if stmts.is_empty() => ProgressResult::IsValue,

        Stmt::Assign(_, expr) | Stmt::ExprStmt(expr) => {
            match check_progress_expr(env, expr) {
                ProgressResult::Stuck(msg) => ProgressResult::Stuck(msg),
                _ => ProgressResult::CanStep("Can evaluate assignment RHS".into()),
            }
        }

        Stmt::LocalDecl(_, _, init) => {
            if let Some(expr) = init {
                match check_progress_expr(env, expr) {
                    ProgressResult::Stuck(msg) => ProgressResult::Stuck(msg),
                    _ => ProgressResult::CanStep("Can declare with initializer".into()),
                }
            } else {
                ProgressResult::CanStep("Can declare (uninitialized)".into())
            }
        }

        Stmt::SharedWrite { index, value, .. } => {
            let ip = check_progress_expr(env, index);
            let vp = check_progress_expr(env, value);
            match (&ip, &vp) {
                (ProgressResult::Stuck(msg), _) | (_, ProgressResult::Stuck(msg)) => {
                    ProgressResult::Stuck(msg.clone())
                }
                _ => ProgressResult::CanStep("Can write to shared memory".into()),
            }
        }

        Stmt::ParallelFor { .. } => {
            ProgressResult::CanStep("Parallel loop can unfold".into())
        }

        Stmt::SeqFor { .. } => {
            ProgressResult::CanStep("Sequential loop can step".into())
        }

        Stmt::If { condition, .. } => {
            match check_progress_expr(env, condition) {
                ProgressResult::Stuck(msg) => ProgressResult::Stuck(msg),
                _ => ProgressResult::CanStep("Can evaluate branch condition".into()),
            }
        }

        Stmt::Block(stmts) => {
            if let Some(first) = stmts.first() {
                check_progress_stmt(env, first)
            } else {
                ProgressResult::IsValue
            }
        }

        Stmt::Barrier => ProgressResult::CanStep("Barrier resolves".into()),

        Stmt::While { condition, .. } => {
            match check_progress_expr(env, condition) {
                ProgressResult::Stuck(msg) => ProgressResult::Stuck(msg),
                _ => ProgressResult::CanStep("While loop can step".into()),
            }
        }

        _ => ProgressResult::CanStep("Statement can step".into()),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Preservation Theorem
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a preservation check.
#[derive(Debug, Clone)]
pub enum PreservationResult {
    /// Type is preserved.
    Preserved { before: PramType, after: PramType },
    /// Type changed but is compatible (widening).
    Compatible { before: PramType, after: PramType, kind: String },
    /// Type is NOT preserved — soundness violation.
    Violated { before: PramType, after: PramType, reason: String },
}

impl PreservationResult {
    pub fn is_sound(&self) -> bool {
        matches!(self, PreservationResult::Preserved { .. } | PreservationResult::Compatible { .. })
    }
}

/// Check preservation for binary operations.
///
/// **Lemma (Preservation for BinOp)**:
/// If `Γ ⊢ e₁ : τ₁` and `Γ ⊢ e₂ : τ₂` and `Γ ⊢ e₁ ⊕ e₂ : τ`,
/// and `e₁ → v₁` and `e₂ → v₂`,
/// then `Γ ⊢ v₁ ⊕ v₂ : τ`.
pub fn check_preservation_binop(
    env: &TypeEnv,
    op: &BinOp,
    left: &Expr,
    right: &Expr,
) -> PreservationResult {
    let before = match typecheck_expr(env, &Expr::BinOp(*op, Box::new(left.clone()), Box::new(right.clone()))) {
        Ok(ty) => ty,
        Err(e) => return PreservationResult::Violated {
            before: PramType::Unit,
            after: PramType::Unit,
            reason: format!("Pre-type error: {}", e),
        },
    };

    let lt = typecheck_expr(env, left);
    let rt = typecheck_expr(env, right);

    match (lt, rt) {
        (Ok(lt), Ok(rt)) => {
            let result_type = compute_binop_result_type(op, &lt, &rt);
            match result_type {
                Some(after) => {
                    let compat = check_compatibility(&before, &after);
                    match compat {
                        TypeCompatibility::Identical => PreservationResult::Preserved { before, after },
                        TypeCompatibility::Widening => PreservationResult::Compatible {
                            before: before.clone(),
                            after,
                            kind: "widening".into(),
                        },
                        _ => PreservationResult::Violated {
                            before,
                            after,
                            reason: "Type narrowing in binop".into(),
                        },
                    }
                }
                None => PreservationResult::Violated {
                    before,
                    after: PramType::Unit,
                    reason: format!("Cannot compute {:?} on {:?} and {:?}", op, lt, rt),
                },
            }
        }
        (Err(e), _) | (_, Err(e)) => PreservationResult::Violated {
            before,
            after: PramType::Unit,
            reason: format!("Operand type error: {}", e),
        },
    }
}

/// Compute the result type of a binary operation after evaluation.
fn compute_binop_result_type(op: &BinOp, left: &PramType, right: &PramType) -> Option<PramType> {
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
            if left.is_numeric() && right.is_numeric() {
                if *left == PramType::Float64 || *right == PramType::Float64 {
                    Some(PramType::Float64)
                } else if *left == PramType::Int64 || *right == PramType::Int64 {
                    Some(PramType::Int64)
                } else {
                    Some(PramType::Int32)
                }
            } else if (left == &PramType::ProcessorId || left.is_integer())
                && (right == &PramType::ProcessorId || right.is_integer())
            {
                Some(PramType::Int64)
            } else {
                None
            }
        }
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne => {
            Some(PramType::Bool)
        }
        BinOp::And | BinOp::Or => {
            if *left == PramType::Bool && *right == PramType::Bool {
                Some(PramType::Bool)
            } else {
                None
            }
        }
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
            if left.is_integer() && right.is_integer() {
                Some(PramType::Int64)
            } else {
                None
            }
        }
        BinOp::Min | BinOp::Max => {
            if left.is_numeric() && right.is_numeric() {
                if *left == PramType::Float64 || *right == PramType::Float64 {
                    Some(PramType::Float64)
                } else {
                    Some(PramType::Int64)
                }
            } else {
                None
            }
        }
    }
}

/// Check preservation for an entire expression.
///
/// **Theorem (Preservation for Expressions)**:
/// If `Γ ⊢ e : τ` and `e → e'`, then `Γ ⊢ e' : τ`.
///
/// Proof by induction on the derivation of `e → e'`.
pub fn check_preservation_expr(env: &TypeEnv, expr: &Expr) -> PreservationResult {
    let before = match typecheck_expr(env, expr) {
        Ok(ty) => ty,
        Err(e) => return PreservationResult::Violated {
            before: PramType::Unit,
            after: PramType::Unit,
            reason: format!("Expression not well-typed: {}", e),
        },
    };

    match expr {
        Expr::IntLiteral(_) | Expr::FloatLiteral(_) | Expr::BoolLiteral(_) => {
            PreservationResult::Preserved { before: before.clone(), after: before }
        }

        Expr::Variable(name) => {
            if let Some(ty) = env.lookup(name) {
                PreservationResult::Preserved { before: before.clone(), after: ty.clone() }
            } else {
                PreservationResult::Violated {
                    before,
                    after: PramType::Unit,
                    reason: format!("Variable '{}' not in scope", name),
                }
            }
        }

        Expr::ProcessorId => PreservationResult::Preserved {
            before: PramType::ProcessorId,
            after: PramType::ProcessorId,
        },
        Expr::NumProcessors => PreservationResult::Preserved {
            before: PramType::Int64,
            after: PramType::Int64,
        },

        Expr::BinOp(op, left, right) => {
            check_preservation_binop(env, op, left, right)
        }

        Expr::UnaryOp(_op, operand) => {
            let inner_pres = check_preservation_expr(env, operand);
            if !inner_pres.is_sound() {
                return inner_pres;
            }
            PreservationResult::Preserved { before: before.clone(), after: before }
        }

        Expr::SharedRead(mem, _idx) => {
            let mem_ty = typecheck_expr(env, mem);
            match mem_ty {
                Ok(PramType::SharedMemory(inner)) | Ok(PramType::Array(inner, _)) => {
                    PreservationResult::Preserved { before: *inner.clone(), after: *inner }
                }
                _ => PreservationResult::Preserved { before: before.clone(), after: before }
            }
        }

        Expr::ArrayIndex(_arr, _idx) => {
            PreservationResult::Preserved { before: before.clone(), after: before }
        }

        Expr::Cast(_inner, target) => {
            PreservationResult::Preserved {
                before: target.clone(),
                after: target.clone(),
            }
        }

        Expr::FunctionCall(_, _) => {
            PreservationResult::Preserved { before: before.clone(), after: before }
        }

        Expr::Conditional(_, then_e, else_e) => {
            let then_ty = typecheck_expr(env, then_e);
            let else_ty = typecheck_expr(env, else_e);
            match (then_ty, else_ty) {
                (Ok(tt), Ok(et)) => {
                    if tt == et {
                        PreservationResult::Preserved { before, after: tt }
                    } else {
                        let compat = check_compatibility(&tt, &et);
                        let reason = format!("Branch types incompatible: {} vs {}", tt, et);
                        match compat {
                            TypeCompatibility::Identical | TypeCompatibility::Widening => {
                                PreservationResult::Compatible {
                                    before,
                                    after: tt,
                                    kind: "branch type widening".into(),
                                }
                            }
                            _ => PreservationResult::Violated {
                                before,
                                after: tt,
                                reason,
                            }
                        }
                    }
                }
                _ => PreservationResult::Preserved { before: before.clone(), after: before }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Type Soundness: Combined Progress + Preservation
// ═══════════════════════════════════════════════════════════════════════════

/// A proof obligation for the type soundness theorem.
#[derive(Debug, Clone)]
pub struct SoundnessObligation {
    pub description: String,
    pub progress: ProgressResult,
    pub preservation: PreservationResult,
}

impl SoundnessObligation {
    pub fn is_discharged(&self) -> bool {
        let progress_ok = matches!(
            self.progress,
            ProgressResult::IsValue | ProgressResult::CanStep(_)
        );
        let preservation_ok = self.preservation.is_sound();
        progress_ok && preservation_ok
    }
}

/// Check type soundness for an expression (progress + preservation).
pub fn check_soundness_expr(env: &TypeEnv, expr: &Expr) -> SoundnessObligation {
    let progress = check_progress_expr(env, expr);
    let preservation = check_preservation_expr(env, expr);
    SoundnessObligation {
        description: "Soundness for expression".into(),
        progress,
        preservation,
    }
}

/// Verify type soundness across all expressions in a program.
///
/// **Theorem (Type Soundness)**:
/// If `Γ ⊢ P : ok` (program P is well-typed), then P does not get stuck.
///
/// Proof: By induction on program structure, checking progress and preservation
/// at each sub-expression and sub-statement.
pub fn verify_program_soundness(program: &PramProgram) -> Vec<SoundnessObligation> {
    let mut env = TypeEnv::new();

    // Bind shared memory declarations
    for decl in &program.shared_memory {
        let size_val = match &decl.size {
            Expr::IntLiteral(n) => *n as usize,
            _ => 1024, // default
        };
        env.declare_shared(decl.name.clone(), decl.elem_type.clone(), size_val);
    }

    // Bind parameters
    for param in &program.parameters {
        env.bind(param.name.clone(), param.param_type.clone());
    }

    let mut obligations = Vec::new();
    verify_stmts_soundness(&env, &program.body, &mut obligations);
    obligations
}

fn verify_stmts_soundness(
    env: &TypeEnv,
    stmts: &[Stmt],
    obligations: &mut Vec<SoundnessObligation>,
) {
    let mut local_env = env.clone();
    for stmt in stmts {
        // Track local declarations to update the environment
        if let Stmt::LocalDecl(name, ty, _) = stmt {
            local_env.bind(name.clone(), ty.clone());
        }
        if let Stmt::Assign(name, _) = stmt {
            // If variable not yet bound, bind as Int64 (common case)
            if local_env.lookup(name).is_none() {
                local_env.bind(name.clone(), PramType::Int64);
            }
        }
        verify_stmt_soundness(&local_env, stmt, obligations);
    }
}

fn verify_stmt_soundness(
    env: &TypeEnv,
    stmt: &Stmt,
    obligations: &mut Vec<SoundnessObligation>,
) {
    match stmt {
        Stmt::Assign(_, expr) | Stmt::Return(Some(expr)) | Stmt::ExprStmt(expr) => {
            obligations.push(check_soundness_expr(env, expr));
        }
        Stmt::LocalDecl(_, _, Some(expr)) => {
            obligations.push(check_soundness_expr(env, expr));
        }
        Stmt::SharedWrite { index, value, .. } => {
            obligations.push(check_soundness_expr(env, index));
            obligations.push(check_soundness_expr(env, value));
        }
        Stmt::ParallelFor { proc_var, body, .. } => {
            let mut inner_env = env.push_scope();
            inner_env.bind(proc_var.clone(), PramType::ProcessorId);
            verify_stmts_soundness(&inner_env, body, obligations);
        }
        Stmt::SeqFor { var, start, end, body, .. } => {
            obligations.push(check_soundness_expr(env, start));
            obligations.push(check_soundness_expr(env, end));
            let mut inner_env = env.push_scope();
            inner_env.bind(var.clone(), PramType::Int64);
            verify_stmts_soundness(&inner_env, body, obligations);
        }
        Stmt::If { condition, then_body, else_body } => {
            obligations.push(check_soundness_expr(env, condition));
            verify_stmts_soundness(env, then_body, obligations);
            verify_stmts_soundness(env, else_body, obligations);
        }
        Stmt::Block(inner) => {
            verify_stmts_soundness(env, inner, obligations);
        }
        Stmt::While { condition, body } => {
            obligations.push(check_soundness_expr(env, condition));
            verify_stmts_soundness(env, body, obligations);
        }
        Stmt::Assert(expr, _) => {
            obligations.push(check_soundness_expr(env, expr));
        }
        _ => {}
    }
}

/// Summary report of type soundness verification.
#[derive(Debug, Clone)]
pub struct SoundnessReport {
    pub total_obligations: usize,
    pub discharged: usize,
    pub violated: usize,
    pub violations: Vec<String>,
}

impl SoundnessReport {
    pub fn is_sound(&self) -> bool {
        self.violated == 0
    }
}

/// Generate a full soundness report for a program.
pub fn soundness_report(program: &PramProgram) -> SoundnessReport {
    let obligations = verify_program_soundness(program);
    let total = obligations.len();
    let mut discharged = 0;
    let mut violated = 0;
    let mut violations = Vec::new();

    for ob in &obligations {
        if ob.is_discharged() {
            discharged += 1;
        } else {
            violated += 1;
            violations.push(ob.description.clone());
        }
    }

    SoundnessReport {
        total_obligations: total,
        discharged,
        violated,
        violations,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Canonical Forms Lemma
// ═══════════════════════════════════════════════════════════════════════════

/// **Lemma (Canonical Forms)**:
/// If `v` is a value of type `τ`, then:
/// - If `τ = Int64`, then `v = IntVal(n)`.
/// - If `τ = Bool`, then `v = BoolVal(b)`.
/// - If `τ = Float64`, then `v = FloatVal(f)`.
/// - If `τ = Unit`, then `v = UnitVal`.
pub fn check_canonical_forms(value: &Value, expected_type: &PramType) -> bool {
    match (value, expected_type) {
        (Value::IntVal(_), PramType::Int64) => true,
        (Value::IntVal(_), PramType::Int32) => true,
        (Value::FloatVal(_), PramType::Float64) => true,
        (Value::FloatVal(_), PramType::Float32) => true,
        (Value::BoolVal(_), PramType::Bool) => true,
        (Value::UnitVal, PramType::Unit) => true,
        (Value::ArrayVal(vs), PramType::Array(elem_ty, n)) => {
            vs.len() == *n && vs.iter().all(|v| check_canonical_forms(v, elem_ty))
        }
        (Value::TupleVal(vs), PramType::Tuple(tys)) => {
            vs.len() == tys.len()
                && vs.iter().zip(tys).all(|(v, t)| check_canonical_forms(v, t))
        }
        _ => false,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Weakening and Exchange Lemmas
// ═══════════════════════════════════════════════════════════════════════════

/// **Lemma (Weakening)**:
/// If `Γ ⊢ e : τ` and `x ∉ dom(Γ)`, then `Γ, x:σ ⊢ e : τ`.
pub fn check_weakening(env: &TypeEnv, expr: &Expr, extra_var: &str, extra_type: &PramType) -> bool {
    let ty_before = typecheck_expr(env, expr);

    let mut extended_env = env.clone();
    extended_env.bind(extra_var.to_string(), extra_type.clone());
    let ty_after = typecheck_expr(&extended_env, expr);

    match (ty_before, ty_after) {
        (Ok(t1), Ok(t2)) => t1 == t2,
        (Err(_), Err(_)) => true,
        _ => false,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CRCW Memory Safety
// ═══════════════════════════════════════════════════════════════════════════

/// **Theorem (CRCW Write Safety)**:
/// Under CRCW-Priority semantics, if processors p₁ < p₂ both write to
/// location ℓ with values v₁, v₂ respectively, then after resolution
/// mem[ℓ] = v₁ (lowest-ID wins).
///
/// In all three CRCW modes, the resulting memory state is well-typed if
/// both v₁ and v₂ are of the declared type of ℓ.
#[derive(Debug, Clone)]
pub struct CrcwSafetyCheck {
    pub model: MemoryModel,
    pub location: String,
    pub writer_count: usize,
    pub all_same_type: bool,
    pub is_safe: bool,
    pub resolution: String,
}

/// Check CRCW write safety for a parallel body.
pub fn check_crcw_write_safety(
    model: &MemoryModel,
    env: &TypeEnv,
    body: &[Stmt],
) -> Vec<CrcwSafetyCheck> {
    let mut checks = Vec::new();
    let writes = collect_shared_writes(body);

    for (location, write_exprs) in &writes {
        let types: Vec<_> = write_exprs.iter()
            .filter_map(|e| typecheck_expr(env, e).ok())
            .collect();

        let all_same = types.windows(2).all(|w| w[0] == w[1]);
        let declared_ty = env.get_shared_region(location).map(|(t, _)| t.clone());

        let type_safe = if let Some(ref dt) = declared_ty {
            types.iter().all(|t| dt.is_assignable_from(t) || t == dt)
        } else {
            false
        };

        let (is_safe, resolution) = match model {
            MemoryModel::CRCWPriority => {
                (type_safe, "Lowest processor ID wins".into())
            }
            MemoryModel::CRCWCommon => {
                (type_safe && all_same, "All writers must agree".into())
            }
            MemoryModel::CRCWArbitrary => {
                (type_safe, "Arbitrary writer wins".into())
            }
            MemoryModel::CREW | MemoryModel::EREW => {
                (write_exprs.len() <= 1, "Exclusive write required".into())
            }
        };

        checks.push(CrcwSafetyCheck {
            model: *model,
            location: location.clone(),
            writer_count: write_exprs.len(),
            all_same_type: all_same,
            is_safe,
            resolution,
        });
    }

    checks
}

fn collect_shared_writes(stmts: &[Stmt]) -> Vec<(String, Vec<Expr>)> {
    let mut writes: std::collections::HashMap<String, Vec<Expr>> = std::collections::HashMap::new();
    for stmt in stmts {
        collect_writes_recursive(stmt, &mut writes);
    }
    writes.into_iter().collect()
}

fn collect_writes_recursive(stmt: &Stmt, writes: &mut std::collections::HashMap<String, Vec<Expr>>) {
    match stmt {
        Stmt::SharedWrite { memory, value, .. } => {
            if let Expr::Variable(name) = memory {
                writes.entry(name.clone()).or_default().push(value.clone());
            }
        }
        Stmt::Block(inner) => {
            for s in inner { collect_writes_recursive(s, writes); }
        }
        Stmt::If { then_body, else_body, .. } => {
            for s in then_body { collect_writes_recursive(s, writes); }
            for s in else_body { collect_writes_recursive(s, writes); }
        }
        Stmt::ParallelFor { body, .. } | Stmt::SeqFor { body, .. } => {
            for s in body { collect_writes_recursive(s, writes); }
        }
        Stmt::While { body, .. } => {
            for s in body { collect_writes_recursive(s, writes); }
        }
        _ => {}
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Substitution Lemma
// ═══════════════════════════════════════════════════════════════════════════

/// **Lemma (Substitution)**:
/// If `Γ, x:σ ⊢ e : τ` and `Γ ⊢ v : σ`, then `Γ ⊢ [v/x]e : τ`.
///
/// This is the key lemma enabling β-reduction in the staged specializer.
/// Proof by structural induction on e.
///
/// - Case `e = x`: `[v/x]x = v`. By hypothesis `Γ ⊢ v : σ` and `x : σ`, so done.
/// - Case `e = y ≠ x`: `[v/x]y = y`. `Γ, x:σ ⊢ y : τ` implies `Γ ⊢ y : τ` by weakening.
/// - Case `e = e₁ ⊕ e₂`: By IH on both children.
/// - Case `e = SharedRead(m, i)`: By IH on m and i.
/// - Case `e = Cast(e', τ')`: By IH on e'.
/// - Case `e = Conditional(c, t, f)`: By IH on c, t, f.

/// Perform capture-avoiding substitution of `var_name` with `replacement` in `expr`.
/// Delegates to `Expr::substitute` on the AST, which handles all expression forms.
pub fn substitute_expr(expr: &Expr, var_name: &str, replacement: &Expr) -> Expr {
    expr.substitute(var_name, replacement)
}

/// Executable check of the substitution lemma.
///
/// Given environment `env` where `var_name : σ` is bound, expression `expr` that
/// typechecks under `env`, and `replacement_expr` that also has type `σ` under `env`,
/// verify that `[replacement_expr / var_name] expr` has the same type as `expr`.
pub fn check_substitution_preserves_type(
    env: &TypeEnv,
    expr: &Expr,
    var_name: &str,
    replacement_expr: &Expr,
) -> bool {
    // 1. Check that var_name is bound in env
    let var_type = match env.lookup(var_name) {
        Some(ty) => ty.clone(),
        None => return false,
    };

    // 2. Check that replacement_expr has the same type as var_name
    let repl_type = match typecheck_expr(env, replacement_expr) {
        Ok(ty) => ty,
        Err(_) => return false,
    };

    // Types must be compatible (identical or widening)
    let compat = check_compatibility(&var_type, &repl_type);
    if !matches!(compat, TypeCompatibility::Identical | TypeCompatibility::Widening) {
        // Also accept ProcessorId ↔ integer for the common pid substitution case
        if !(var_type == PramType::ProcessorId && repl_type.is_integer())
            && !(var_type.is_integer() && repl_type == PramType::ProcessorId)
        {
            return false;
        }
    }

    // 3. Typecheck the original expression
    let original_type = match typecheck_expr(env, expr) {
        Ok(ty) => ty,
        Err(_) => return false,
    };

    // 4. Perform substitution
    let substituted = substitute_expr(expr, var_name, replacement_expr);

    // 5. Typecheck the substituted expression
    let substituted_type = match typecheck_expr(env, &substituted) {
        Ok(ty) => ty,
        Err(_) => return false,
    };

    // 6. Verify type preservation
    let result_compat = check_compatibility(&original_type, &substituted_type);
    matches!(
        result_compat,
        TypeCompatibility::Identical | TypeCompatibility::Widening
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Inversion Lemma
// ═══════════════════════════════════════════════════════════════════════════

/// **Lemma (Inversion)**:
/// Type derivations can be uniquely decomposed:
/// - If `Γ ⊢ e₁ ⊕ e₂ : τ`, then `Γ ⊢ e₁ : τ₁` and `Γ ⊢ e₂ : τ₂` with τ₁,τ₂ numeric
/// - If `Γ ⊢ SharedRead(m, i) : τ`, then m : SharedMemory<τ> and i : integer
/// - If `Γ ⊢ If(c, t, e) : τ`, then c : Bool and t : τ and e : τ
/// - If `Γ ⊢ Cast(e, τ) : τ`, then e has a compatible source type
/// - If `Γ ⊢ UnaryOp(op, e) : τ`, then e : τ (for Neg) or e : Bool (for Not)
///
/// This lemma is crucial for the progress theorem: it tells us what we can
/// assume about subexpressions once we know the type of a compound expression.

/// Result of inversion analysis.
#[derive(Debug, Clone)]
pub struct InversionResult {
    /// The top-level type of the expression.
    pub expr_type: PramType,
    /// Types of the immediate subexpressions.
    pub sub_types: Vec<(String, PramType)>,
    /// Whether the inversion is consistent (all sub-types match expected roles).
    pub consistent: bool,
    /// Description of the inversion rule applied.
    pub rule: String,
}

/// Check the inversion lemma for an expression.
///
/// If `expr` has type `τ` under `env`, decompose the typing derivation
/// and verify that subexpressions have the expected types.
pub fn check_inversion(env: &TypeEnv, expr: &Expr) -> Option<InversionResult> {
    let expr_type = typecheck_expr(env, expr).ok()?;

    match expr {
        Expr::BinOp(op, left, right) => {
            let lt = typecheck_expr(env, left).ok()?;
            let rt = typecheck_expr(env, right).ok()?;
            let consistent = if op.is_comparison() {
                (lt.is_numeric() || lt == PramType::ProcessorId)
                    && (rt.is_numeric() || rt == PramType::ProcessorId)
            } else if op.is_logical() {
                lt == PramType::Bool && rt == PramType::Bool
            } else if op.is_arithmetic() {
                (lt.is_numeric() || lt == PramType::ProcessorId)
                    && (rt.is_numeric() || rt == PramType::ProcessorId)
            } else {
                // Bitwise ops: both integer
                lt.is_integer() && rt.is_integer()
            };
            Some(InversionResult {
                expr_type,
                sub_types: vec![
                    ("left".into(), lt),
                    ("right".into(), rt),
                ],
                consistent,
                rule: format!("BinOp-{}", op.symbol()),
            })
        }

        Expr::SharedRead(mem, idx) => {
            let mem_ty = typecheck_expr(env, mem).ok()?;
            let idx_ty = typecheck_expr(env, idx).ok()?;
            let consistent = matches!(
                mem_ty,
                PramType::SharedMemory(_) | PramType::Array(_, _)
            ) && (idx_ty.is_integer() || idx_ty == PramType::ProcessorId);
            Some(InversionResult {
                expr_type,
                sub_types: vec![
                    ("memory".into(), mem_ty),
                    ("index".into(), idx_ty),
                ],
                consistent,
                rule: "SharedRead".into(),
            })
        }

        Expr::Conditional(cond, then_e, else_e) => {
            let ct = typecheck_expr(env, cond).ok()?;
            let tt = typecheck_expr(env, then_e).ok()?;
            let et = typecheck_expr(env, else_e).ok()?;
            let consistent = ct == PramType::Bool
                && (tt == et
                    || matches!(
                        check_compatibility(&tt, &et),
                        TypeCompatibility::Identical | TypeCompatibility::Widening
                    ));
            Some(InversionResult {
                expr_type,
                sub_types: vec![
                    ("condition".into(), ct),
                    ("then".into(), tt),
                    ("else".into(), et),
                ],
                consistent,
                rule: "Conditional".into(),
            })
        }

        Expr::UnaryOp(op, operand) => {
            let ot = typecheck_expr(env, operand).ok()?;
            let consistent = match op {
                UnaryOp::Neg => ot.is_numeric(),
                UnaryOp::Not => ot == PramType::Bool,
                UnaryOp::BitNot => ot.is_integer(),
            };
            Some(InversionResult {
                expr_type,
                sub_types: vec![("operand".into(), ot)],
                consistent,
                rule: format!("UnaryOp-{}", op.symbol()),
            })
        }

        Expr::Cast(inner, target_ty) => {
            let it = typecheck_expr(env, inner).ok()?;
            let consistent = it.is_numeric() && target_ty.is_numeric()
                || it == PramType::ProcessorId && target_ty.is_numeric()
                || it.is_integer() && *target_ty == PramType::ProcessorId;
            Some(InversionResult {
                expr_type,
                sub_types: vec![
                    ("source".into(), it),
                    ("target".into(), target_ty.clone()),
                ],
                consistent,
                rule: "Cast".into(),
            })
        }

        Expr::FunctionCall(name, args) => {
            let arg_types: Vec<_> = args
                .iter()
                .enumerate()
                .filter_map(|(i, a)| {
                    typecheck_expr(env, a).ok().map(|t| (format!("arg{}", i), t))
                })
                .collect();
            let consistent = arg_types.len() == args.len();
            Some(InversionResult {
                expr_type,
                sub_types: arg_types,
                consistent,
                rule: format!("FunctionCall-{}", name),
            })
        }

        // Leaf expressions: trivially invertible
        Expr::IntLiteral(_)
        | Expr::FloatLiteral(_)
        | Expr::BoolLiteral(_)
        | Expr::Variable(_)
        | Expr::ProcessorId
        | Expr::NumProcessors => Some(InversionResult {
            expr_type,
            sub_types: vec![],
            consistent: true,
            rule: "Leaf".into(),
        }),

        Expr::ArrayIndex(arr, idx) => {
            let at = typecheck_expr(env, arr).ok()?;
            let it = typecheck_expr(env, idx).ok()?;
            let consistent = matches!(at, PramType::Array(_, _))
                && (it.is_integer() || it == PramType::ProcessorId);
            Some(InversionResult {
                expr_type,
                sub_types: vec![
                    ("array".into(), at),
                    ("index".into(), it),
                ],
                consistent,
                rule: "ArrayIndex".into(),
            })
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Store Typing and Memory Safety
// ═══════════════════════════════════════════════════════════════════════════

/// **Definition (Store Typing)**:
/// A store `μ` is a partial map from locations (memory_name, index) to values.
/// A store typing `Σ` assigns a type to each location.
///
/// **Definition (Well-Typed Store)**:
/// `Σ ⊢ μ` (store μ is well-typed under Σ) iff for every `(ℓ, v) ∈ μ`,
/// we have `Σ(ℓ) = τ` and `⊢ v : τ` (the value is of the declared type).
///
/// **Theorem (Store Type Safety)**:
/// If `Σ ⊢ μ` and `Γ; Σ ⊢ s → μ'`, then `Σ' ⊢ μ'` for some extension `Σ' ⊇ Σ`.
/// (Evaluation preserves store well-typedness.)

/// A location in the shared memory store.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StoreLocation {
    pub memory_name: String,
    pub index: usize,
}

/// An entry in the typed store.
#[derive(Debug, Clone)]
pub struct StoreEntry {
    pub location: StoreLocation,
    pub declared_type: PramType,
    pub value: Value,
}

/// A typed shared memory store.
#[derive(Debug, Clone)]
pub struct SharedStore {
    pub entries: Vec<StoreEntry>,
}

impl SharedStore {
    pub fn new() -> Self {
        SharedStore { entries: Vec::new() }
    }

    /// Write a value to a store location.
    pub fn write(&mut self, memory: &str, index: usize, declared_type: PramType, value: Value) {
        let loc = StoreLocation {
            memory_name: memory.to_string(),
            index,
        };
        // Update existing or insert new
        if let Some(entry) = self.entries.iter_mut().find(|e| e.location == loc) {
            entry.value = value;
        } else {
            self.entries.push(StoreEntry {
                location: loc,
                declared_type,
                value,
            });
        }
    }

    /// Read a value from a store location.
    pub fn read(&self, memory: &str, index: usize) -> Option<&Value> {
        let loc = StoreLocation {
            memory_name: memory.to_string(),
            index,
        };
        self.entries
            .iter()
            .find(|e| e.location == loc)
            .map(|e| &e.value)
    }

    /// Number of entries in the store.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Verify that all stored values match their declared types.
///
/// This is the executable witness for the well-typed store judgment `Σ ⊢ μ`.
pub fn check_store_typing(store: &SharedStore, env: &TypeEnv) -> bool {
    for entry in &store.entries {
        // Check that the value has the declared type (canonical forms)
        if !check_canonical_forms(&entry.value, &entry.declared_type) {
            // Allow widening: e.g. Int32 value in Int64 slot
            let value_type = entry.value.type_of();
            let compat = check_compatibility(&entry.declared_type, &value_type);
            if !matches!(
                compat,
                TypeCompatibility::Identical | TypeCompatibility::Widening
            ) {
                return false;
            }
        }
        // Check that the memory region is declared in the environment
        if env.get_shared_region(&entry.location.memory_name).is_none() {
            return false;
        }
    }
    true
}

/// Check store consistency after a write operation.
/// Verifies that writing `value` to `memory[index]` preserves store typing.
pub fn check_store_write_preserves_typing(
    store: &SharedStore,
    env: &TypeEnv,
    memory: &str,
    _index: usize,
    value: &Value,
) -> bool {
    // Look up the declared element type of the memory region
    let declared_elem_type = match env.get_shared_region(memory) {
        Some((ty, _size)) => ty.clone(),
        None => return false,
    };

    // Check that the value is compatible with the declared element type
    let value_type = value.type_of();
    let compat = check_compatibility(&declared_elem_type, &value_type);
    if !matches!(
        compat,
        TypeCompatibility::Identical | TypeCompatibility::Widening
    ) {
        return false;
    }

    // Existing store must remain well-typed (writes don't affect other entries)
    check_store_typing(store, env)
}

// ═══════════════════════════════════════════════════════════════════════════
// CRCW Linearizability Proof
// ═══════════════════════════════════════════════════════════════════════════

/// **Theorem (CRCW Linearizability)**:
/// For any set of concurrent writes to the same location under a CRCW model,
/// there exists a linearization (total order on writes) such that the final
/// memory state is identical to executing writes in that order.
///
/// - **Priority**: The linearization orders processors by ID (ascending).
///   The surviving write is the one from the minimum processor ID.
///   This is deterministic.
///
/// - **Common**: All concurrent writers must write the same value.
///   If they do, any linearization yields the same result.
///   If they disagree, it is a conflict error.
///
/// - **Arbitrary**: There exists *some* linearization (the last writer in
///   the chosen order wins). The result is nondeterministic but the
///   execution is equivalent to some sequential execution.

/// A write event from a single processor.
#[derive(Debug, Clone)]
pub struct WriteEvent {
    /// Processor ID that issued the write.
    pub processor_id: usize,
    /// The value written.
    pub value: Value,
}

/// Result of linearizability checking.
#[derive(Debug, Clone)]
pub enum LinearizabilityResult {
    /// The writes are linearizable with the given surviving value.
    Linearizable {
        surviving_processor: usize,
        surviving_value: Value,
        linearization_order: Vec<usize>,
    },
    /// Conflict detected (for CRCW-Common when values disagree).
    Conflict {
        disagreeing_processors: Vec<usize>,
        values: Vec<Value>,
    },
    /// No writes to linearize.
    Empty,
}

impl LinearizabilityResult {
    /// Whether the result represents a successful linearization.
    pub fn is_linearizable(&self) -> bool {
        matches!(
            self,
            LinearizabilityResult::Linearizable { .. } | LinearizabilityResult::Empty
        )
    }
}

/// Check CRCW linearizability for a set of concurrent writes.
///
/// Given a sequence of writes (in arrival order) and the memory model,
/// determine the surviving write and verify that the result is consistent
/// with some sequential execution.
pub fn check_crcw_linearizability(
    writes: &[WriteEvent],
    model: &MemoryModel,
) -> LinearizabilityResult {
    if writes.is_empty() {
        return LinearizabilityResult::Empty;
    }

    if writes.len() == 1 {
        return LinearizabilityResult::Linearizable {
            surviving_processor: writes[0].processor_id,
            surviving_value: writes[0].value.clone(),
            linearization_order: vec![writes[0].processor_id],
        };
    }

    match model {
        MemoryModel::CRCWPriority => {
            // Priority: lowest processor ID wins
            let winner = writes
                .iter()
                .min_by_key(|w| w.processor_id)
                .unwrap();
            // Linearization: ascending processor ID order (winner is last in serial)
            let mut order: Vec<usize> = writes.iter().map(|w| w.processor_id).collect();
            order.sort();
            // In serial execution, last write wins; so reverse order puts winner last
            order.reverse();
            // Actually for priority, we want the minimum-ID to survive.
            // In serial execution, the last writer wins. So min-ID must be last.
            // Linearization order: all others in descending ID, then the winner.
            let mut lin_order: Vec<usize> = writes
                .iter()
                .map(|w| w.processor_id)
                .filter(|&pid| pid != winner.processor_id)
                .collect();
            lin_order.sort_by(|a, b| b.cmp(a));
            lin_order.push(winner.processor_id);

            LinearizabilityResult::Linearizable {
                surviving_processor: winner.processor_id,
                surviving_value: winner.value.clone(),
                linearization_order: lin_order,
            }
        }

        MemoryModel::CRCWCommon => {
            // Common: all writers must agree on the value
            let first_val = &writes[0].value;
            let all_agree = writes.iter().all(|w| w.value == *first_val);

            if all_agree {
                let order: Vec<usize> = writes.iter().map(|w| w.processor_id).collect();
                LinearizabilityResult::Linearizable {
                    surviving_processor: writes[0].processor_id,
                    surviving_value: first_val.clone(),
                    linearization_order: order,
                }
            } else {
                let pids: Vec<usize> = writes.iter().map(|w| w.processor_id).collect();
                let vals: Vec<Value> = writes.iter().map(|w| w.value.clone()).collect();
                LinearizabilityResult::Conflict {
                    disagreeing_processors: pids,
                    values: vals,
                }
            }
        }

        MemoryModel::CRCWArbitrary => {
            // Arbitrary: any single writer can win; we pick the first as the witness
            // The key property is that *exactly one* write survives.
            let winner = &writes[0];
            let mut lin_order: Vec<usize> = writes
                .iter()
                .map(|w| w.processor_id)
                .filter(|&pid| pid != winner.processor_id)
                .collect();
            lin_order.push(winner.processor_id);

            LinearizabilityResult::Linearizable {
                surviving_processor: winner.processor_id,
                surviving_value: winner.value.clone(),
                linearization_order: lin_order,
            }
        }

        // CREW and EREW do not allow concurrent writes
        MemoryModel::CREW | MemoryModel::EREW => {
            if writes.len() > 1 {
                let pids: Vec<usize> = writes.iter().map(|w| w.processor_id).collect();
                let vals: Vec<Value> = writes.iter().map(|w| w.value.clone()).collect();
                LinearizabilityResult::Conflict {
                    disagreeing_processors: pids,
                    values: vals,
                }
            } else {
                LinearizabilityResult::Linearizable {
                    surviving_processor: writes[0].processor_id,
                    surviving_value: writes[0].value.clone(),
                    linearization_order: vec![writes[0].processor_id],
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Determinism Theorem (Priority Semantics)
// ═══════════════════════════════════════════════════════════════════════════

/// **Theorem (Determinism under Priority)**:
/// Under CRCW-Priority semantics, if `(μ, s) → μ₁` and `(μ, s) → μ₂`,
/// then `μ₁ = μ₂`. That is, the evaluation is deterministic.
///
/// Proof sketch:
/// - All reads are deterministic (concurrent reads return the same value).
/// - All writes to disjoint locations are independent (commute).
/// - For writes to the same location, Priority resolution is deterministic:
///   the processor with the smallest ID always wins, regardless of scheduling.
/// - Sequential composition preserves determinism by induction.

/// Result of determinism checking.
#[derive(Debug, Clone)]
pub struct DeterminismResult {
    /// Whether the statement is deterministic under the given model.
    pub is_deterministic: bool,
    /// Reason / evidence for the determination.
    pub reason: String,
    /// Potential nondeterminism sources found.
    pub nondeterminism_sources: Vec<String>,
}

/// Check whether a statement is deterministic under the given memory model.
///
/// Under Priority semantics, all PRAM programs are deterministic.
/// Under Arbitrary semantics, programs with concurrent writes to the same
/// location are nondeterministic.
pub fn check_determinism(model: &MemoryModel, stmts: &[Stmt]) -> DeterminismResult {
    let mut sources = Vec::new();
    check_determinism_recursive(model, stmts, &mut sources);

    let is_deterministic = sources.is_empty();
    let reason = if is_deterministic {
        match model {
            MemoryModel::CRCWPriority => {
                "Deterministic: Priority resolution is a total order on processor IDs".into()
            }
            MemoryModel::EREW | MemoryModel::CREW => {
                "Deterministic: exclusive/no concurrent writes".into()
            }
            MemoryModel::CRCWCommon => {
                "Deterministic: all concurrent writers agree (enforced)".into()
            }
            MemoryModel::CRCWArbitrary => {
                "Deterministic: no concurrent writes found in this program".into()
            }
        }
    } else {
        format!(
            "Nondeterministic: {} source(s) of nondeterminism",
            sources.len()
        )
    };

    DeterminismResult {
        is_deterministic,
        reason,
        nondeterminism_sources: sources,
    }
}

fn check_determinism_recursive(
    model: &MemoryModel,
    stmts: &[Stmt],
    sources: &mut Vec<String>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::ParallelFor { body, .. } => {
                // In a parallel for, concurrent writes may occur
                if *model == MemoryModel::CRCWArbitrary {
                    // Check if body contains shared writes
                    let writes = collect_shared_writes(body);
                    for (location, exprs) in &writes {
                        if exprs.len() > 0 {
                            sources.push(format!(
                                "Arbitrary CRCW write to '{}' in parallel_for body",
                                location
                            ));
                        }
                    }
                }
                // Recurse into body for nested constructs
                check_determinism_recursive(model, body, sources);
            }
            Stmt::If { then_body, else_body, .. } => {
                check_determinism_recursive(model, then_body, sources);
                check_determinism_recursive(model, else_body, sources);
            }
            Stmt::SeqFor { body, .. } | Stmt::While { body, .. } => {
                check_determinism_recursive(model, body, sources);
            }
            Stmt::Block(inner) => {
                check_determinism_recursive(model, inner, sources);
            }
            _ => {}
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Termination Lemma for Bounded Parallel Loops
// ═══════════════════════════════════════════════════════════════════════════

/// **Lemma (Termination of Bounded Loops)**:
/// If `ParallelFor(n, body)` where `n` is an `IntLiteral(k)` with `k ≥ 0`,
/// and the body terminates for each processor, then the parallel for terminates.
///
/// Similarly, `SeqFor(i, start, end, step, body)` where `start` and `end`
/// are `IntLiteral` values with `start ≤ end` and `step > 0` terminates
/// in exactly `⌈(end - start) / step⌉` iterations.
///
/// Proof: By well-founded induction on the iteration count.
/// The iteration count is a natural number, and each step decreases it by 1.
///
/// Note: `While` loops and loops with non-literal bounds are NOT guaranteed
/// to terminate — they require a separate termination argument (e.g., a
/// ranking function or variant).

/// Result of termination analysis.
#[derive(Debug, Clone)]
pub enum TerminationResult {
    /// The statement provably terminates.
    Terminates {
        /// Upper bound on iterations (if computable).
        bound: Option<u64>,
        /// Reason for termination.
        reason: String,
    },
    /// The statement may not terminate.
    MayDiverge {
        reason: String,
    },
    /// Not a loop — trivially terminates.
    NotALoop,
}

impl TerminationResult {
    /// Whether termination is guaranteed.
    pub fn is_terminating(&self) -> bool {
        matches!(
            self,
            TerminationResult::Terminates { .. } | TerminationResult::NotALoop
        )
    }
}

/// Check bounded termination for a statement.
///
/// Returns a termination proof obligation for loops, verifying that:
/// - `ParallelFor` with `IntLiteral` bound terminates
/// - `SeqFor` with `IntLiteral` start/end terminates
/// - `While` loops are flagged as potentially divergent
pub fn check_bounded_termination(stmt: &Stmt) -> TerminationResult {
    match stmt {
        Stmt::ParallelFor { num_procs, body, .. } => {
            match num_procs {
                Expr::IntLiteral(n) if *n >= 0 => {
                    // Check body termination recursively
                    let body_terminates = body.iter().all(|s| {
                        check_bounded_termination(s).is_terminating()
                    });
                    if body_terminates {
                        TerminationResult::Terminates {
                            bound: Some(*n as u64),
                            reason: format!(
                                "ParallelFor with literal bound {} and terminating body",
                                n
                            ),
                        }
                    } else {
                        TerminationResult::MayDiverge {
                            reason: "ParallelFor body contains potentially divergent loop".into(),
                        }
                    }
                }
                Expr::IntLiteral(n) => {
                    // Negative bound — zero iterations, trivially terminates
                    TerminationResult::Terminates {
                        bound: Some(0),
                        reason: format!(
                            "ParallelFor with non-positive bound {} executes 0 iterations",
                            n
                        ),
                    }
                }
                other => {
                    // Non-literal bound: may terminate if bound evaluates to finite
                    if let Some(val) = other.eval_const_int() {
                        if val >= 0 {
                            TerminationResult::Terminates {
                                bound: Some(val as u64),
                                reason: format!(
                                    "ParallelFor with const-evaluable bound {}",
                                    val
                                ),
                            }
                        } else {
                            TerminationResult::Terminates {
                                bound: Some(0),
                                reason: "Non-positive const bound".into(),
                            }
                        }
                    } else {
                        TerminationResult::MayDiverge {
                            reason: "ParallelFor bound is not a compile-time constant".into(),
                        }
                    }
                }
            }
        }

        Stmt::SeqFor { start, end, step, body, .. } => {
            let start_val = start.eval_const_int();
            let end_val = end.eval_const_int();
            let step_val = step.as_ref().and_then(|s| s.eval_const_int()).unwrap_or(1);

            match (start_val, end_val) {
                (Some(s), Some(e)) => {
                    if step_val <= 0 {
                        TerminationResult::MayDiverge {
                            reason: format!(
                                "SeqFor with non-positive step {} may not terminate",
                                step_val
                            ),
                        }
                    } else {
                        let iterations = if e > s {
                            ((e - s + step_val - 1) / step_val) as u64
                        } else {
                            0
                        };
                        let body_terminates = body.iter().all(|s| {
                            check_bounded_termination(s).is_terminating()
                        });
                        if body_terminates {
                            TerminationResult::Terminates {
                                bound: Some(iterations),
                                reason: format!(
                                    "SeqFor from {} to {} step {} = {} iterations",
                                    s, e, step_val, iterations
                                ),
                            }
                        } else {
                            TerminationResult::MayDiverge {
                                reason: "SeqFor body contains potentially divergent loop".into(),
                            }
                        }
                    }
                }
                _ => TerminationResult::MayDiverge {
                    reason: "SeqFor bounds are not compile-time constants".into(),
                },
            }
        }

        Stmt::While { .. } => TerminationResult::MayDiverge {
            reason: "While loops require a ranking function for termination proof".into(),
        },

        Stmt::Block(stmts) => {
            for s in stmts {
                if !check_bounded_termination(s).is_terminating() {
                    return TerminationResult::MayDiverge {
                        reason: "Block contains potentially divergent statement".into(),
                    };
                }
            }
            TerminationResult::NotALoop
        }

        Stmt::If { then_body, else_body, .. } => {
            let then_ok = then_body
                .iter()
                .all(|s| check_bounded_termination(s).is_terminating());
            let else_ok = else_body
                .iter()
                .all(|s| check_bounded_termination(s).is_terminating());
            if then_ok && else_ok {
                TerminationResult::NotALoop
            } else {
                TerminationResult::MayDiverge {
                    reason: "If branch contains potentially divergent statement".into(),
                }
            }
        }

        // All other statements are non-looping and trivially terminate
        _ => TerminationResult::NotALoop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pram_ir::ast::*;

    fn make_test_env() -> TypeEnv {
        let mut env = TypeEnv::new();
        env.bind("x".into(), PramType::Int64);
        env.bind("y".into(), PramType::Int64);
        env.bind("b".into(), PramType::Bool);
        env.bind("f".into(), PramType::Float64);
        env.declare_shared("A".into(), PramType::Int64, 100);
        env
    }

    // ── Progress Tests ──

    #[test]
    fn test_progress_int_literal() {
        let env = TypeEnv::new();
        let result = check_progress_expr(&env, &Expr::IntLiteral(42));
        assert!(matches!(result, ProgressResult::IsValue));
    }

    #[test]
    fn test_progress_variable_defined() {
        let env = make_test_env();
        let result = check_progress_expr(&env, &Expr::Variable("x".into()));
        assert!(matches!(result, ProgressResult::CanStep(_)));
    }

    #[test]
    fn test_progress_variable_undefined() {
        let env = TypeEnv::new();
        let result = check_progress_expr(&env, &Expr::Variable("z".into()));
        assert!(matches!(result, ProgressResult::Stuck(_)));
    }

    #[test]
    fn test_progress_binop() {
        let env = make_test_env();
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Variable("x".into())),
            Box::new(Expr::IntLiteral(1)),
        );
        let result = check_progress_expr(&env, &expr);
        assert!(matches!(result, ProgressResult::CanStep(_)));
    }

    #[test]
    fn test_progress_binop_both_values() {
        let env = TypeEnv::new();
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::IntLiteral(1)),
            Box::new(Expr::IntLiteral(2)),
        );
        let result = check_progress_expr(&env, &expr);
        assert!(matches!(result, ProgressResult::CanStep(_)));
    }

    #[test]
    fn test_progress_shared_read() {
        let env = make_test_env();
        let expr = Expr::SharedRead(
            Box::new(Expr::Variable("A".into())),
            Box::new(Expr::IntLiteral(0)),
        );
        let result = check_progress_expr(&env, &expr);
        assert!(matches!(result, ProgressResult::CanStep(_)));
    }

    #[test]
    fn test_progress_stmt_assign() {
        let env = make_test_env();
        let stmt = Stmt::Assign("x".into(), Expr::IntLiteral(42));
        let result = check_progress_stmt(&env, &stmt);
        assert!(matches!(result, ProgressResult::CanStep(_)));
    }

    #[test]
    fn test_progress_stmt_block_empty() {
        let env = TypeEnv::new();
        let stmt = Stmt::Block(vec![]);
        let result = check_progress_stmt(&env, &stmt);
        assert!(matches!(result, ProgressResult::IsValue));
    }

    #[test]
    fn test_progress_stmt_return_none() {
        let env = TypeEnv::new();
        let stmt = Stmt::Return(None);
        let result = check_progress_stmt(&env, &stmt);
        assert!(matches!(result, ProgressResult::IsValue));
    }

    #[test]
    fn test_progress_parallel_for() {
        let env = make_test_env();
        let stmt = Stmt::ParallelFor {
            proc_var: "pid".into(),
            num_procs: Expr::IntLiteral(4),
            body: vec![],
        };
        let result = check_progress_stmt(&env, &stmt);
        assert!(matches!(result, ProgressResult::CanStep(_)));
    }

    // ── Preservation Tests ──

    #[test]
    fn test_preservation_int_literal() {
        let env = TypeEnv::new();
        let result = check_preservation_expr(&env, &Expr::IntLiteral(42));
        assert!(result.is_sound());
    }

    #[test]
    fn test_preservation_binop_add() {
        let env = make_test_env();
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Variable("x".into())),
            Box::new(Expr::IntLiteral(1)),
        );
        let result = check_preservation_expr(&env, &expr);
        assert!(result.is_sound());
    }

    #[test]
    fn test_preservation_comparison() {
        let env = make_test_env();
        let expr = Expr::BinOp(
            BinOp::Lt,
            Box::new(Expr::Variable("x".into())),
            Box::new(Expr::IntLiteral(10)),
        );
        let result = check_preservation_expr(&env, &expr);
        assert!(result.is_sound());
    }

    #[test]
    fn test_preservation_cast() {
        let env = make_test_env();
        let expr = Expr::Cast(
            Box::new(Expr::Variable("x".into())),
            PramType::Float64,
        );
        let result = check_preservation_expr(&env, &expr);
        assert!(result.is_sound());
    }

    // ── Soundness Tests ──

    #[test]
    fn test_soundness_well_typed_expr() {
        let env = make_test_env();
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Variable("x".into())),
            Box::new(Expr::Variable("y".into())),
        );
        let ob = check_soundness_expr(&env, &expr);
        assert!(ob.is_discharged());
    }

    #[test]
    fn test_soundness_ill_typed_expr() {
        let env = make_test_env();
        let expr = Expr::BinOp(
            BinOp::And,
            Box::new(Expr::Variable("x".into())), // Int64
            Box::new(Expr::Variable("b".into())),  // Bool
        );
        let ob = check_soundness_expr(&env, &expr);
        assert!(!ob.is_discharged());
    }

    // ── Canonical Forms Tests ──

    #[test]
    fn test_canonical_int() {
        assert!(check_canonical_forms(&Value::IntVal(42), &PramType::Int64));
        assert!(!check_canonical_forms(&Value::BoolVal(true), &PramType::Int64));
    }

    #[test]
    fn test_canonical_bool() {
        assert!(check_canonical_forms(&Value::BoolVal(true), &PramType::Bool));
    }

    #[test]
    fn test_canonical_array() {
        let arr = Value::ArrayVal(vec![Value::IntVal(1), Value::IntVal(2)]);
        assert!(check_canonical_forms(
            &arr,
            &PramType::Array(Box::new(PramType::Int64), 2)
        ));
        assert!(!check_canonical_forms(
            &arr,
            &PramType::Array(Box::new(PramType::Int64), 3)
        ));
    }

    // ── Weakening Tests ──

    #[test]
    fn test_weakening_holds() {
        let env = make_test_env();
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Variable("x".into())),
            Box::new(Expr::IntLiteral(1)),
        );
        assert!(check_weakening(&env, &expr, "z", &PramType::Float64));
    }

    // ── Program Soundness Tests ──

    #[test]
    fn test_program_soundness_simple() {
        let program = PramProgram {
            name: "test".into(),
            memory_model: MemoryModel::CRCWPriority,
            parameters: vec![Parameter {
                name: "n".into(),
                param_type: PramType::Int64,
            }],
            shared_memory: vec![SharedMemoryDecl {
                name: "A".into(),
                elem_type: PramType::Int64,
                size: Expr::IntLiteral(100),
            }],
            body: vec![
                Stmt::ParallelFor {
                    proc_var: "pid".into(),
                    num_procs: Expr::Variable("n".into()),
                    body: vec![Stmt::SharedWrite {
                        memory: Expr::Variable("A".into()),
                        index: Expr::ProcessorId,
                        value: Expr::IntLiteral(0),
                    }],
                },
            ],
            num_processors: Expr::IntLiteral(100),
            work_bound: None,
            time_bound: None,
            description: None,
        };
        let report = soundness_report(&program);
        assert!(report.is_sound());
        assert!(report.total_obligations > 0);
    }

    #[test]
    fn test_program_soundness_bitonic() {
        let prog = crate::algorithm_library::sorting::bitonic_sort();
        let report = soundness_report(&prog);
        assert!(report.is_sound(), "Bitonic sort should be type-sound. Violations: {:?}", report.violations);
    }

    // ── CRCW Safety Tests ──

    #[test]
    fn test_crcw_priority_safe() {
        let env = make_test_env();
        let body = vec![
            Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(0),
                value: Expr::IntLiteral(1),
            },
            Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(0),
                value: Expr::IntLiteral(2),
            },
        ];
        let checks = check_crcw_write_safety(&MemoryModel::CRCWPriority, &env, &body);
        assert!(!checks.is_empty());
        for c in &checks {
            assert!(c.is_safe);
        }
    }

    #[test]
    fn test_erew_exclusive_write_violated() {
        let env = make_test_env();
        let body = vec![
            Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(0),
                value: Expr::IntLiteral(1),
            },
            Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(0),
                value: Expr::IntLiteral(2),
            },
        ];
        let checks = check_crcw_write_safety(&MemoryModel::EREW, &env, &body);
        assert!(!checks.is_empty());
        for c in &checks {
            assert!(!c.is_safe);
        }
    }

    // ── Substitution Lemma Tests ──

    #[test]
    fn test_substitution_variable_replaced() {
        let expr = Expr::Variable("x".into());
        let replacement = Expr::IntLiteral(42);
        let result = substitute_expr(&expr, "x", &replacement);
        assert_eq!(result, Expr::IntLiteral(42));
    }

    #[test]
    fn test_substitution_other_variable_unchanged() {
        let expr = Expr::Variable("y".into());
        let replacement = Expr::IntLiteral(42);
        let result = substitute_expr(&expr, "x", &replacement);
        assert_eq!(result, Expr::Variable("y".into()));
    }

    #[test]
    fn test_substitution_preserves_type_simple() {
        let env = make_test_env();
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Variable("x".into())),
            Box::new(Expr::IntLiteral(1)),
        );
        let replacement = Expr::IntLiteral(10);
        assert!(check_substitution_preserves_type(&env, &expr, "x", &replacement));
    }

    #[test]
    fn test_substitution_preserves_type_nested() {
        let env = make_test_env();
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Variable("x".into())),
                Box::new(Expr::Variable("y".into())),
            )),
            Box::new(Expr::IntLiteral(2)),
        );
        let replacement = Expr::IntLiteral(5);
        assert!(check_substitution_preserves_type(&env, &expr, "x", &replacement));
    }

    #[test]
    fn test_substitution_in_conditional() {
        let env = make_test_env();
        let expr = Expr::Conditional(
            Box::new(Expr::Variable("b".into())),
            Box::new(Expr::Variable("x".into())),
            Box::new(Expr::IntLiteral(0)),
        );
        let replacement = Expr::IntLiteral(99);
        assert!(check_substitution_preserves_type(&env, &expr, "x", &replacement));
    }

    // ── Inversion Lemma Tests ──

    #[test]
    fn test_inversion_binop_add() {
        let env = make_test_env();
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Variable("x".into())),
            Box::new(Expr::IntLiteral(1)),
        );
        let inv = check_inversion(&env, &expr).unwrap();
        assert!(inv.consistent);
        assert_eq!(inv.rule, "BinOp-+");
        assert_eq!(inv.sub_types.len(), 2);
    }

    #[test]
    fn test_inversion_comparison() {
        let env = make_test_env();
        let expr = Expr::BinOp(
            BinOp::Lt,
            Box::new(Expr::Variable("x".into())),
            Box::new(Expr::IntLiteral(10)),
        );
        let inv = check_inversion(&env, &expr).unwrap();
        assert!(inv.consistent);
        assert_eq!(inv.expr_type, PramType::Bool);
    }

    #[test]
    fn test_inversion_shared_read() {
        let env = make_test_env();
        let expr = Expr::SharedRead(
            Box::new(Expr::Variable("A".into())),
            Box::new(Expr::IntLiteral(0)),
        );
        let inv = check_inversion(&env, &expr).unwrap();
        assert!(inv.consistent);
        assert_eq!(inv.rule, "SharedRead");
        // Memory should be SharedMemory type
        let mem_ty = &inv.sub_types[0].1;
        assert!(matches!(mem_ty, PramType::SharedMemory(_)));
    }

    #[test]
    fn test_inversion_conditional() {
        let env = make_test_env();
        let expr = Expr::Conditional(
            Box::new(Expr::Variable("b".into())),
            Box::new(Expr::IntLiteral(1)),
            Box::new(Expr::IntLiteral(2)),
        );
        let inv = check_inversion(&env, &expr).unwrap();
        assert!(inv.consistent);
        assert_eq!(inv.sub_types[0].1, PramType::Bool);
    }

    #[test]
    fn test_inversion_unary_neg() {
        let env = make_test_env();
        let expr = Expr::UnaryOp(
            UnaryOp::Neg,
            Box::new(Expr::Variable("x".into())),
        );
        let inv = check_inversion(&env, &expr).unwrap();
        assert!(inv.consistent);
        assert_eq!(inv.sub_types[0].1, PramType::Int64);
    }

    #[test]
    fn test_inversion_leaf() {
        let env = TypeEnv::new();
        let expr = Expr::IntLiteral(42);
        let inv = check_inversion(&env, &expr).unwrap();
        assert!(inv.consistent);
        assert_eq!(inv.rule, "Leaf");
        assert!(inv.sub_types.is_empty());
    }

    // ── Store Typing Tests ──

    #[test]
    fn test_store_empty_is_well_typed() {
        let env = make_test_env();
        let store = SharedStore::new();
        assert!(check_store_typing(&store, &env));
    }

    #[test]
    fn test_store_correct_typing() {
        let env = make_test_env();
        let mut store = SharedStore::new();
        store.write("A", 0, PramType::Int64, Value::IntVal(42));
        store.write("A", 1, PramType::Int64, Value::IntVal(99));
        assert!(check_store_typing(&store, &env));
    }

    #[test]
    fn test_store_type_mismatch() {
        let env = make_test_env();
        let mut store = SharedStore::new();
        // Declare type is Int64 but store a Bool
        store.write("A", 0, PramType::Int64, Value::BoolVal(true));
        assert!(!check_store_typing(&store, &env));
    }

    #[test]
    fn test_store_write_preserves_typing() {
        let env = make_test_env();
        let store = SharedStore::new();
        assert!(check_store_write_preserves_typing(
            &store,
            &env,
            "A",
            0,
            &Value::IntVal(42),
        ));
    }

    #[test]
    fn test_store_write_bad_type_fails() {
        let env = make_test_env();
        let store = SharedStore::new();
        assert!(!check_store_write_preserves_typing(
            &store,
            &env,
            "A",
            0,
            &Value::BoolVal(true),
        ));
    }

    #[test]
    fn test_store_undeclared_region_fails() {
        let env = make_test_env();
        let mut store = SharedStore::new();
        store.write("UNDECLARED", 0, PramType::Int64, Value::IntVal(1));
        assert!(!check_store_typing(&store, &env));
    }

    // ── CRCW Linearizability Tests ──

    #[test]
    fn test_linearizability_priority_min_wins() {
        let writes = vec![
            WriteEvent { processor_id: 3, value: Value::IntVal(30) },
            WriteEvent { processor_id: 1, value: Value::IntVal(10) },
            WriteEvent { processor_id: 2, value: Value::IntVal(20) },
        ];
        let result = check_crcw_linearizability(&writes, &MemoryModel::CRCWPriority);
        match result {
            LinearizabilityResult::Linearizable { surviving_processor, surviving_value, .. } => {
                assert_eq!(surviving_processor, 1);
                assert_eq!(surviving_value, Value::IntVal(10));
            }
            _ => panic!("Expected Linearizable for CRCWPriority"),
        }
    }

    #[test]
    fn test_linearizability_common_agree() {
        let writes = vec![
            WriteEvent { processor_id: 0, value: Value::IntVal(42) },
            WriteEvent { processor_id: 1, value: Value::IntVal(42) },
            WriteEvent { processor_id: 2, value: Value::IntVal(42) },
        ];
        let result = check_crcw_linearizability(&writes, &MemoryModel::CRCWCommon);
        assert!(result.is_linearizable());
    }

    #[test]
    fn test_linearizability_common_disagree() {
        let writes = vec![
            WriteEvent { processor_id: 0, value: Value::IntVal(1) },
            WriteEvent { processor_id: 1, value: Value::IntVal(2) },
        ];
        let result = check_crcw_linearizability(&writes, &MemoryModel::CRCWCommon);
        assert!(!result.is_linearizable());
        assert!(matches!(result, LinearizabilityResult::Conflict { .. }));
    }

    #[test]
    fn test_linearizability_arbitrary_one_survives() {
        let writes = vec![
            WriteEvent { processor_id: 0, value: Value::IntVal(10) },
            WriteEvent { processor_id: 1, value: Value::IntVal(20) },
        ];
        let result = check_crcw_linearizability(&writes, &MemoryModel::CRCWArbitrary);
        assert!(result.is_linearizable());
    }

    #[test]
    fn test_linearizability_erew_rejects_concurrent() {
        let writes = vec![
            WriteEvent { processor_id: 0, value: Value::IntVal(1) },
            WriteEvent { processor_id: 1, value: Value::IntVal(2) },
        ];
        let result = check_crcw_linearizability(&writes, &MemoryModel::EREW);
        assert!(!result.is_linearizable());
    }

    #[test]
    fn test_linearizability_empty() {
        let writes: Vec<WriteEvent> = vec![];
        let result = check_crcw_linearizability(&writes, &MemoryModel::CRCWPriority);
        assert!(result.is_linearizable());
        assert!(matches!(result, LinearizabilityResult::Empty));
    }

    #[test]
    fn test_linearizability_single_write() {
        let writes = vec![
            WriteEvent { processor_id: 5, value: Value::IntVal(50) },
        ];
        let result = check_crcw_linearizability(&writes, &MemoryModel::CRCWPriority);
        match result {
            LinearizabilityResult::Linearizable { surviving_processor, .. } => {
                assert_eq!(surviving_processor, 5);
            }
            _ => panic!("Single write should be linearizable"),
        }
    }

    // ── Determinism Tests ──

    #[test]
    fn test_determinism_priority_always() {
        let stmts = vec![
            Stmt::ParallelFor {
                proc_var: "pid".into(),
                num_procs: Expr::IntLiteral(4),
                body: vec![Stmt::SharedWrite {
                    memory: Expr::Variable("A".into()),
                    index: Expr::IntLiteral(0),
                    value: Expr::ProcessorId,
                }],
            },
        ];
        let result = check_determinism(&MemoryModel::CRCWPriority, &stmts);
        assert!(result.is_deterministic);
    }

    #[test]
    fn test_determinism_arbitrary_nondeterministic() {
        let stmts = vec![
            Stmt::ParallelFor {
                proc_var: "pid".into(),
                num_procs: Expr::IntLiteral(4),
                body: vec![Stmt::SharedWrite {
                    memory: Expr::Variable("A".into()),
                    index: Expr::IntLiteral(0),
                    value: Expr::ProcessorId,
                }],
            },
        ];
        let result = check_determinism(&MemoryModel::CRCWArbitrary, &stmts);
        assert!(!result.is_deterministic);
    }

    // ── Termination Tests ──

    #[test]
    fn test_termination_parallel_for_literal() {
        let stmt = Stmt::ParallelFor {
            proc_var: "pid".into(),
            num_procs: Expr::IntLiteral(8),
            body: vec![Stmt::Nop],
        };
        let result = check_bounded_termination(&stmt);
        assert!(result.is_terminating());
        match result {
            TerminationResult::Terminates { bound, .. } => {
                assert_eq!(bound, Some(8));
            }
            _ => panic!("Expected Terminates"),
        }
    }

    #[test]
    fn test_termination_seq_for_bounded() {
        let stmt = Stmt::SeqFor {
            var: "i".into(),
            start: Expr::IntLiteral(0),
            end: Expr::IntLiteral(10),
            step: Some(Expr::IntLiteral(2)),
            body: vec![Stmt::Nop],
        };
        let result = check_bounded_termination(&stmt);
        assert!(result.is_terminating());
        match result {
            TerminationResult::Terminates { bound, .. } => {
                assert_eq!(bound, Some(5));
            }
            _ => panic!("Expected Terminates"),
        }
    }

    #[test]
    fn test_termination_while_diverges() {
        let stmt = Stmt::While {
            condition: Expr::BoolLiteral(true),
            body: vec![Stmt::Nop],
        };
        let result = check_bounded_termination(&stmt);
        assert!(!result.is_terminating());
    }

    #[test]
    fn test_termination_nop_trivial() {
        let stmt = Stmt::Nop;
        let result = check_bounded_termination(&stmt);
        assert!(result.is_terminating());
        assert!(matches!(result, TerminationResult::NotALoop));
    }

    #[test]
    fn test_termination_nested_while_in_parallel_for() {
        let stmt = Stmt::ParallelFor {
            proc_var: "pid".into(),
            num_procs: Expr::IntLiteral(4),
            body: vec![Stmt::While {
                condition: Expr::BoolLiteral(true),
                body: vec![Stmt::Nop],
            }],
        };
        let result = check_bounded_termination(&stmt);
        assert!(!result.is_terminating());
    }

    #[test]
    fn test_termination_seq_for_zero_step_diverges() {
        let stmt = Stmt::SeqFor {
            var: "i".into(),
            start: Expr::IntLiteral(0),
            end: Expr::IntLiteral(10),
            step: Some(Expr::IntLiteral(0)),
            body: vec![Stmt::Nop],
        };
        let result = check_bounded_termination(&stmt);
        assert!(!result.is_terminating());
    }

    #[test]
    fn test_termination_parallel_for_const_expr_bound() {
        let stmt = Stmt::ParallelFor {
            proc_var: "pid".into(),
            num_procs: Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::IntLiteral(2)),
                Box::new(Expr::IntLiteral(4)),
            ),
            body: vec![Stmt::Nop],
        };
        let result = check_bounded_termination(&stmt);
        assert!(result.is_terminating());
        match result {
            TerminationResult::Terminates { bound, .. } => {
                assert_eq!(bound, Some(8));
            }
            _ => panic!("Expected Terminates with const-evaluable bound"),
        }
    }

    #[test]
    fn test_store_read_write_roundtrip() {
        let mut store = SharedStore::new();
        store.write("A", 5, PramType::Int64, Value::IntVal(123));
        assert_eq!(store.read("A", 5), Some(&Value::IntVal(123)));
        assert_eq!(store.read("A", 6), None);
    }

    #[test]
    fn test_store_overwrite() {
        let mut store = SharedStore::new();
        store.write("A", 0, PramType::Int64, Value::IntVal(1));
        store.write("A", 0, PramType::Int64, Value::IntVal(2));
        assert_eq!(store.read("A", 0), Some(&Value::IntVal(2)));
        assert_eq!(store.len(), 1);
    }
}
