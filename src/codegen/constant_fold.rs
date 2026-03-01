//! Constant folding pass for the PRAM IR.
//!
//! Evaluates constant sub-expressions at compile time, applies algebraic
//! simplifications (x+0=x, x*1=x, x*0=0, …), and eliminates dead branches
//! whose conditions are statically known.

use crate::pram_ir::ast::*;
use crate::pram_ir::types::*;

// ---------------------------------------------------------------------------
// ConstantFolder
// ---------------------------------------------------------------------------

/// Performs constant folding and algebraic simplification on the PRAM IR.
#[derive(Debug, Default)]
pub struct ConstantFolder {
    /// Number of folds applied (for diagnostics).
    pub folds_applied: usize,
}

impl ConstantFolder {
    pub fn new() -> Self {
        Self { folds_applied: 0 }
    }

    /// Fold constants in an expression, returning the simplified expression.
    pub fn fold_expr(&mut self, expr: &Expr) -> Expr {
        match expr {
            // Leaves are already in normal form.
            Expr::IntLiteral(_)
            | Expr::FloatLiteral(_)
            | Expr::BoolLiteral(_)
            | Expr::Variable(_)
            | Expr::ProcessorId
            | Expr::NumProcessors => expr.clone(),

            Expr::UnaryOp(op, inner) => {
                let folded = self.fold_expr(inner);
                self.fold_unary(*op, folded)
            }

            Expr::BinOp(op, lhs, rhs) => {
                let l = self.fold_expr(lhs);
                let r = self.fold_expr(rhs);
                self.fold_binop(*op, l, r)
            }

            Expr::SharedRead(mem, idx) => {
                Expr::SharedRead(Box::new(self.fold_expr(mem)), Box::new(self.fold_expr(idx)))
            }

            Expr::ArrayIndex(arr, idx) => {
                Expr::ArrayIndex(Box::new(self.fold_expr(arr)), Box::new(self.fold_expr(idx)))
            }

            Expr::FunctionCall(name, args) => {
                let folded_args: Vec<Expr> = args.iter().map(|a| self.fold_expr(a)).collect();
                Expr::FunctionCall(name.clone(), folded_args)
            }

            Expr::Cast(inner, ty) => {
                let folded = self.fold_expr(inner);
                // Fold cast of literal
                match (&folded, ty) {
                    (Expr::IntLiteral(v), PramType::Float64) => {
                        self.folds_applied += 1;
                        Expr::FloatLiteral(*v as f64)
                    }
                    (Expr::IntLiteral(v), PramType::Float32) => {
                        self.folds_applied += 1;
                        Expr::FloatLiteral(*v as f64)
                    }
                    (Expr::IntLiteral(v), PramType::Int32) => {
                        self.folds_applied += 1;
                        Expr::IntLiteral(*v as i32 as i64)
                    }
                    (Expr::IntLiteral(v), PramType::Bool) => {
                        self.folds_applied += 1;
                        Expr::BoolLiteral(*v != 0)
                    }
                    _ => Expr::Cast(Box::new(folded), ty.clone()),
                }
            }

            Expr::Conditional(cond, then_e, else_e) => {
                let c = self.fold_expr(cond);
                let t = self.fold_expr(then_e);
                let e = self.fold_expr(else_e);
                // Dead-branch elimination at expression level
                match &c {
                    Expr::BoolLiteral(true) => {
                        self.folds_applied += 1;
                        t
                    }
                    Expr::BoolLiteral(false) => {
                        self.folds_applied += 1;
                        e
                    }
                    _ => Expr::Conditional(Box::new(c), Box::new(t), Box::new(e)),
                }
            }
        }
    }

    /// Fold a unary operation on a (possibly constant) operand.
    fn fold_unary(&mut self, op: UnaryOp, operand: Expr) -> Expr {
        match (op, &operand) {
            // -(-x) => x
            (UnaryOp::Neg, Expr::UnaryOp(UnaryOp::Neg, inner)) => {
                self.folds_applied += 1;
                *inner.clone()
            }
            // !(!x) => x
            (UnaryOp::Not, Expr::UnaryOp(UnaryOp::Not, inner)) => {
                self.folds_applied += 1;
                *inner.clone()
            }
            // ~(~x) => x
            (UnaryOp::BitNot, Expr::UnaryOp(UnaryOp::BitNot, inner)) => {
                self.folds_applied += 1;
                *inner.clone()
            }
            // Fold constant
            (UnaryOp::Neg, Expr::IntLiteral(v)) => {
                self.folds_applied += 1;
                Expr::IntLiteral(-v)
            }
            (UnaryOp::Neg, Expr::FloatLiteral(v)) => {
                self.folds_applied += 1;
                Expr::FloatLiteral(-v)
            }
            (UnaryOp::Not, Expr::BoolLiteral(b)) => {
                self.folds_applied += 1;
                Expr::BoolLiteral(!b)
            }
            (UnaryOp::BitNot, Expr::IntLiteral(v)) => {
                self.folds_applied += 1;
                Expr::IntLiteral(!v)
            }
            _ => Expr::UnaryOp(op, Box::new(operand)),
        }
    }

    /// Fold a binary operation, applying constant evaluation and algebraic
    /// identities.
    fn fold_binop(&mut self, op: BinOp, lhs: Expr, rhs: Expr) -> Expr {
        // Full constant evaluation
        if let (Some(lv), Some(rv)) = (lhs.eval_const_int(), rhs.eval_const_int()) {
            let result = Expr::BinOp(op, Box::new(lhs.clone()), Box::new(rhs.clone()));
            if let Some(val) = result.eval_const_int() {
                self.folds_applied += 1;
                return if op.is_comparison() || op.is_logical() {
                    Expr::BoolLiteral(val != 0)
                } else {
                    Expr::IntLiteral(val)
                };
            }
            // Division by zero – leave it unreduced
            let _ = (lv, rv);
        }

        // Float constant evaluation
        if let (Expr::FloatLiteral(l), Expr::FloatLiteral(r)) = (&lhs, &rhs) {
            let result = match op {
                BinOp::Add => Some(l + r),
                BinOp::Sub => Some(l - r),
                BinOp::Mul => Some(l * r),
                BinOp::Div if *r != 0.0 => Some(l / r),
                _ => None,
            };
            if let Some(val) = result {
                self.folds_applied += 1;
                return Expr::FloatLiteral(val);
            }
        }

        // Algebraic simplifications
        match op {
            // x + 0 => x
            BinOp::Add if is_zero(&rhs) => {
                self.folds_applied += 1;
                return lhs;
            }
            // 0 + x => x
            BinOp::Add if is_zero(&lhs) => {
                self.folds_applied += 1;
                return rhs;
            }
            // x - 0 => x
            BinOp::Sub if is_zero(&rhs) => {
                self.folds_applied += 1;
                return lhs;
            }
            // x - x => 0
            BinOp::Sub if lhs == rhs => {
                self.folds_applied += 1;
                return Expr::IntLiteral(0);
            }
            // x * 0 => 0
            BinOp::Mul if is_zero(&rhs) => {
                self.folds_applied += 1;
                return Expr::IntLiteral(0);
            }
            // 0 * x => 0
            BinOp::Mul if is_zero(&lhs) => {
                self.folds_applied += 1;
                return Expr::IntLiteral(0);
            }
            // x * 1 => x
            BinOp::Mul if is_one(&rhs) => {
                self.folds_applied += 1;
                return lhs;
            }
            // 1 * x => x
            BinOp::Mul if is_one(&lhs) => {
                self.folds_applied += 1;
                return rhs;
            }
            // x / 1 => x
            BinOp::Div if is_one(&rhs) => {
                self.folds_applied += 1;
                return lhs;
            }
            // Boolean: true && x => x
            BinOp::And if matches!(&lhs, Expr::BoolLiteral(true)) => {
                self.folds_applied += 1;
                return rhs;
            }
            // Boolean: x && true => x
            BinOp::And if matches!(&rhs, Expr::BoolLiteral(true)) => {
                self.folds_applied += 1;
                return lhs;
            }
            // Boolean: false && x => false
            BinOp::And if matches!(&lhs, Expr::BoolLiteral(false)) => {
                self.folds_applied += 1;
                return Expr::BoolLiteral(false);
            }
            // Boolean: x && false => false
            BinOp::And if matches!(&rhs, Expr::BoolLiteral(false)) => {
                self.folds_applied += 1;
                return Expr::BoolLiteral(false);
            }
            // Boolean: false || x => x
            BinOp::Or if matches!(&lhs, Expr::BoolLiteral(false)) => {
                self.folds_applied += 1;
                return rhs;
            }
            // Boolean: x || false => x
            BinOp::Or if matches!(&rhs, Expr::BoolLiteral(false)) => {
                self.folds_applied += 1;
                return lhs;
            }
            // Boolean: true || x => true
            BinOp::Or if matches!(&lhs, Expr::BoolLiteral(true)) => {
                self.folds_applied += 1;
                return Expr::BoolLiteral(true);
            }
            // Boolean: x || true => true
            BinOp::Or if matches!(&rhs, Expr::BoolLiteral(true)) => {
                self.folds_applied += 1;
                return Expr::BoolLiteral(true);
            }
            // x ^ 0 => x
            BinOp::BitXor if is_zero(&rhs) => {
                self.folds_applied += 1;
                return lhs;
            }
            // x ^ x => 0
            BinOp::BitXor if lhs == rhs => {
                self.folds_applied += 1;
                return Expr::IntLiteral(0);
            }
            // x & 0 => 0
            BinOp::BitAnd if is_zero(&rhs) || is_zero(&lhs) => {
                self.folds_applied += 1;
                return Expr::IntLiteral(0);
            }
            // x | 0 => x
            BinOp::BitOr if is_zero(&rhs) => {
                self.folds_applied += 1;
                return lhs;
            }
            // 0 | x => x
            BinOp::BitOr if is_zero(&lhs) => {
                self.folds_applied += 1;
                return rhs;
            }
            _ => {}
        }

        Expr::BinOp(op, Box::new(lhs), Box::new(rhs))
    }

    /// Fold constants in a statement, returning the simplified statement.
    pub fn fold_stmt(&mut self, stmt: &Stmt) -> Stmt {
        match stmt {
            Stmt::LocalDecl(name, ty, init) => {
                Stmt::LocalDecl(
                    name.clone(),
                    ty.clone(),
                    init.as_ref().map(|e| self.fold_expr(e)),
                )
            }

            Stmt::Assign(name, expr) => {
                Stmt::Assign(name.clone(), self.fold_expr(expr))
            }

            Stmt::SharedWrite { memory, index, value } => {
                Stmt::SharedWrite {
                    memory: self.fold_expr(memory),
                    index: self.fold_expr(index),
                    value: self.fold_expr(value),
                }
            }

            Stmt::ParallelFor { proc_var, num_procs, body } => {
                let folded_np = self.fold_expr(num_procs);
                // If num_procs folds to 0, the loop is dead
                if matches!(&folded_np, Expr::IntLiteral(0)) {
                    self.folds_applied += 1;
                    return Stmt::Nop;
                }
                Stmt::ParallelFor {
                    proc_var: proc_var.clone(),
                    num_procs: folded_np,
                    body: body.iter().map(|s| self.fold_stmt(s)).collect(),
                }
            }

            Stmt::SeqFor { var, start, end, step, body } => {
                let fs = self.fold_expr(start);
                let fe = self.fold_expr(end);
                let fstep = step.as_ref().map(|s| self.fold_expr(s));
                // If start >= end (both constant), loop is dead
                if let (Some(sv), Some(ev)) = (fs.eval_const_int(), fe.eval_const_int()) {
                    if sv >= ev {
                        self.folds_applied += 1;
                        return Stmt::Nop;
                    }
                }
                Stmt::SeqFor {
                    var: var.clone(),
                    start: fs,
                    end: fe,
                    step: fstep,
                    body: body.iter().map(|s| self.fold_stmt(s)).collect(),
                }
            }

            Stmt::While { condition, body } => {
                let c = self.fold_expr(condition);
                // while(false) => nop
                if matches!(&c, Expr::BoolLiteral(false)) {
                    self.folds_applied += 1;
                    return Stmt::Nop;
                }
                Stmt::While {
                    condition: c,
                    body: body.iter().map(|s| self.fold_stmt(s)).collect(),
                }
            }

            Stmt::If { condition, then_body, else_body } => {
                let c = self.fold_expr(condition);
                // Dead branch elimination
                match &c {
                    Expr::BoolLiteral(true) => {
                        self.folds_applied += 1;
                        let folded: Vec<Stmt> = then_body.iter().map(|s| self.fold_stmt(s)).collect();
                        Stmt::Block(folded)
                    }
                    Expr::BoolLiteral(false) => {
                        self.folds_applied += 1;
                        if else_body.is_empty() {
                            Stmt::Nop
                        } else {
                            let folded: Vec<Stmt> = else_body.iter().map(|s| self.fold_stmt(s)).collect();
                            Stmt::Block(folded)
                        }
                    }
                    _ => {
                        Stmt::If {
                            condition: c,
                            then_body: then_body.iter().map(|s| self.fold_stmt(s)).collect(),
                            else_body: else_body.iter().map(|s| self.fold_stmt(s)).collect(),
                        }
                    }
                }
            }

            Stmt::Block(stmts) => {
                let folded: Vec<Stmt> = stmts
                    .iter()
                    .map(|s| self.fold_stmt(s))
                    .filter(|s| !matches!(s, Stmt::Nop))
                    .collect();
                if folded.is_empty() {
                    Stmt::Nop
                } else {
                    Stmt::Block(folded)
                }
            }

            Stmt::ExprStmt(expr) => Stmt::ExprStmt(self.fold_expr(expr)),

            Stmt::Return(opt) => Stmt::Return(opt.as_ref().map(|e| self.fold_expr(e))),

            Stmt::AllocShared { name, elem_type, size } => {
                Stmt::AllocShared {
                    name: name.clone(),
                    elem_type: elem_type.clone(),
                    size: self.fold_expr(size),
                }
            }

            Stmt::Assert(expr, msg) => Stmt::Assert(self.fold_expr(expr), msg.clone()),

            Stmt::PrefixSum { input, output, size, op } => {
                Stmt::PrefixSum {
                    input: input.clone(),
                    output: output.clone(),
                    size: self.fold_expr(size),
                    op: *op,
                }
            }

            Stmt::AtomicCAS { memory, index, expected, desired, result_var } => {
                Stmt::AtomicCAS {
                    memory: self.fold_expr(memory),
                    index: self.fold_expr(index),
                    expected: self.fold_expr(expected),
                    desired: self.fold_expr(desired),
                    result_var: result_var.clone(),
                }
            }

            Stmt::FetchAdd { memory, index, value, result_var } => {
                Stmt::FetchAdd {
                    memory: self.fold_expr(memory),
                    index: self.fold_expr(index),
                    value: self.fold_expr(value),
                    result_var: result_var.clone(),
                }
            }

            // Pass-through for statements that don't contain foldable expressions
            Stmt::Barrier | Stmt::Nop | Stmt::FreeShared(_) | Stmt::Comment(_) => stmt.clone(),
        }
    }

    /// Fold all statements in a program body.
    pub fn fold_body(&mut self, body: &[Stmt]) -> Vec<Stmt> {
        body.iter()
            .map(|s| self.fold_stmt(s))
            .filter(|s| !matches!(s, Stmt::Nop))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_zero(expr: &Expr) -> bool {
    match expr {
        Expr::IntLiteral(0) => true,
        Expr::FloatLiteral(v) => *v == 0.0,
        _ => false,
    }
}

fn is_one(expr: &Expr) -> bool {
    match expr {
        Expr::IntLiteral(1) => true,
        Expr::FloatLiteral(v) => *v == 1.0,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Program-level folding
// ---------------------------------------------------------------------------

/// Fold all constant expressions in an entire PRAM program.
pub fn fold_program(program: &PramProgram) -> PramProgram {
    let mut folder = ConstantFolder::new();
    let folded_body = folder.fold_body(&program.body);
    let folded_num_procs = folder.fold_expr(&program.num_processors);
    let folded_shared: Vec<SharedMemoryDecl> = program
        .shared_memory
        .iter()
        .map(|decl| SharedMemoryDecl {
            name: decl.name.clone(),
            elem_type: decl.elem_type.clone(),
            size: folder.fold_expr(&decl.size),
        })
        .collect();
    PramProgram {
        name: program.name.clone(),
        memory_model: program.memory_model,
        parameters: program.parameters.clone(),
        shared_memory: folded_shared,
        body: folded_body,
        num_processors: folded_num_procs,
        work_bound: program.work_bound.clone(),
        time_bound: program.time_bound.clone(),
        description: program.description.clone(),
    }
}

/// Count how many nodes differ between two expression trees.
///
/// Walks both trees simultaneously; each node where the variant or leaf value
/// differs counts as 1.
pub fn count_folded(before: &Expr, after: &Expr) -> usize {
    if before == after {
        return 0;
    }
    match (before, after) {
        (Expr::BinOp(op1, l1, r1), Expr::BinOp(op2, l2, r2)) => {
            let root_diff = if op1 != op2 { 1 } else { 0 };
            root_diff + count_folded(l1, l2) + count_folded(r1, r2)
        }
        (Expr::UnaryOp(op1, inner1), Expr::UnaryOp(op2, inner2)) => {
            let root_diff = if op1 != op2 { 1 } else { 0 };
            root_diff + count_folded(inner1, inner2)
        }
        (Expr::Conditional(c1, t1, e1), Expr::Conditional(c2, t2, e2)) => {
            count_folded(c1, c2) + count_folded(t1, t2) + count_folded(e1, e2)
        }
        (Expr::SharedRead(m1, i1), Expr::SharedRead(m2, i2)) => {
            count_folded(m1, m2) + count_folded(i1, i2)
        }
        (Expr::ArrayIndex(a1, i1), Expr::ArrayIndex(a2, i2)) => {
            count_folded(a1, a2) + count_folded(i1, i2)
        }
        (Expr::Cast(inner1, ty1), Expr::Cast(inner2, ty2)) => {
            let root_diff = if ty1 != ty2 { 1 } else { 0 };
            root_diff + count_folded(inner1, inner2)
        }
        // Structurally different trees — the entire `before` tree was replaced.
        _ => 1,
    }
}

/// Iteratively fold an expression until a fixpoint is reached.
///
/// Re-applies constant folding until the result stops changing, with a cap of
/// 100 iterations to prevent infinite loops.
pub fn deep_fold(expr: &Expr) -> Expr {
    let mut current = expr.clone();
    for _ in 0..100 {
        let mut folder = ConstantFolder::new();
        let next = folder.fold_expr(&current);
        if next == current {
            break;
        }
        current = next;
    }
    current
}

// ---------------------------------------------------------------------------
// FoldingStats
// ---------------------------------------------------------------------------

/// Aggregate statistics about a constant-folding pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoldingStats {
    pub constant_folds: usize,
    pub algebraic_simplifications: usize,
    pub dead_branch_eliminations: usize,
    pub total: usize,
}

impl FoldingStats {
    /// Estimate folding statistics by comparing statement lists before and
    /// after a constant-folding pass.
    ///
    /// Uses simple heuristics: statement count reduction indicates dead-branch
    /// elimination; remaining folds are split between constant folds and
    /// algebraic simplifications based on the `ConstantFolder` diagnostics.
    pub fn collect(before: &[Stmt], after: &[Stmt]) -> FoldingStats {
        let mut folder = ConstantFolder::new();
        let re_folded = folder.fold_body(before);

        let before_count = Self::count_stmts(before);
        let after_count = Self::count_stmts(after);

        let dead_branch_eliminations = before_count.saturating_sub(after_count);
        let total_folds = folder.folds_applied;
        let algebraic_simplifications = total_folds / 3;
        let constant_folds = total_folds.saturating_sub(algebraic_simplifications)
            .saturating_sub(dead_branch_eliminations);

        let total = constant_folds + algebraic_simplifications + dead_branch_eliminations;
        let _ = re_folded; // used only to drive the folder

        FoldingStats {
            constant_folds,
            algebraic_simplifications,
            dead_branch_eliminations,
            total,
        }
    }

    /// Recursively count statements (including nested bodies).
    fn count_stmts(stmts: &[Stmt]) -> usize {
        stmts.iter().map(|s| Self::count_stmt(s)).sum()
    }

    fn count_stmt(stmt: &Stmt) -> usize {
        match stmt {
            Stmt::ParallelFor { body, .. }
            | Stmt::SeqFor { body, .. }
            | Stmt::While { body, .. } => 1 + Self::count_stmts(body),
            Stmt::If { then_body, else_body, .. } => {
                1 + Self::count_stmts(then_body) + Self::count_stmts(else_body)
            }
            Stmt::Block(inner) => Self::count_stmts(inner),
            Stmt::Nop => 0,
            _ => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fold_constant_add() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::Add, Expr::int(3), Expr::int(4));
        let result = f.fold_expr(&e);
        assert_eq!(result, Expr::IntLiteral(7));
        assert!(f.folds_applied > 0);
    }

    #[test]
    fn test_fold_constant_mul() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::Mul, Expr::int(6), Expr::int(7));
        assert_eq!(f.fold_expr(&e), Expr::IntLiteral(42));
    }

    #[test]
    fn test_fold_nested_constants() {
        let mut f = ConstantFolder::new();
        // (2 + 3) * (4 + 1) => 25
        let e = Expr::binop(
            BinOp::Mul,
            Expr::binop(BinOp::Add, Expr::int(2), Expr::int(3)),
            Expr::binop(BinOp::Add, Expr::int(4), Expr::int(1)),
        );
        assert_eq!(f.fold_expr(&e), Expr::IntLiteral(25));
    }

    #[test]
    fn test_fold_add_zero() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::Add, Expr::var("x"), Expr::int(0));
        assert_eq!(f.fold_expr(&e), Expr::var("x"));
    }

    #[test]
    fn test_fold_zero_add() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::Add, Expr::int(0), Expr::var("x"));
        assert_eq!(f.fold_expr(&e), Expr::var("x"));
    }

    #[test]
    fn test_fold_sub_zero() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::Sub, Expr::var("x"), Expr::int(0));
        assert_eq!(f.fold_expr(&e), Expr::var("x"));
    }

    #[test]
    fn test_fold_sub_self() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::Sub, Expr::var("x"), Expr::var("x"));
        assert_eq!(f.fold_expr(&e), Expr::IntLiteral(0));
    }

    #[test]
    fn test_fold_mul_zero() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::Mul, Expr::var("x"), Expr::int(0));
        assert_eq!(f.fold_expr(&e), Expr::IntLiteral(0));
    }

    #[test]
    fn test_fold_zero_mul() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::Mul, Expr::int(0), Expr::var("x"));
        assert_eq!(f.fold_expr(&e), Expr::IntLiteral(0));
    }

    #[test]
    fn test_fold_mul_one() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::Mul, Expr::var("x"), Expr::int(1));
        assert_eq!(f.fold_expr(&e), Expr::var("x"));
    }

    #[test]
    fn test_fold_one_mul() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::Mul, Expr::int(1), Expr::var("x"));
        assert_eq!(f.fold_expr(&e), Expr::var("x"));
    }

    #[test]
    fn test_fold_div_one() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::Div, Expr::var("x"), Expr::int(1));
        assert_eq!(f.fold_expr(&e), Expr::var("x"));
    }

    #[test]
    fn test_fold_true_and_x() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::And, Expr::bool_lit(true), Expr::var("x"));
        assert_eq!(f.fold_expr(&e), Expr::var("x"));
    }

    #[test]
    fn test_fold_false_and_x() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::And, Expr::bool_lit(false), Expr::var("x"));
        assert_eq!(f.fold_expr(&e), Expr::BoolLiteral(false));
    }

    #[test]
    fn test_fold_false_or_x() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::Or, Expr::bool_lit(false), Expr::var("x"));
        assert_eq!(f.fold_expr(&e), Expr::var("x"));
    }

    #[test]
    fn test_fold_true_or_x() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::Or, Expr::bool_lit(true), Expr::var("x"));
        assert_eq!(f.fold_expr(&e), Expr::BoolLiteral(true));
    }

    #[test]
    fn test_fold_double_neg() {
        let mut f = ConstantFolder::new();
        let e = Expr::unop(UnaryOp::Neg, Expr::unop(UnaryOp::Neg, Expr::var("x")));
        assert_eq!(f.fold_expr(&e), Expr::var("x"));
    }

    #[test]
    fn test_fold_double_not() {
        let mut f = ConstantFolder::new();
        let e = Expr::unop(UnaryOp::Not, Expr::unop(UnaryOp::Not, Expr::var("b")));
        assert_eq!(f.fold_expr(&e), Expr::var("b"));
    }

    #[test]
    fn test_fold_neg_literal() {
        let mut f = ConstantFolder::new();
        let e = Expr::unop(UnaryOp::Neg, Expr::int(5));
        assert_eq!(f.fold_expr(&e), Expr::IntLiteral(-5));
    }

    #[test]
    fn test_fold_not_literal() {
        let mut f = ConstantFolder::new();
        assert_eq!(
            f.fold_expr(&Expr::unop(UnaryOp::Not, Expr::bool_lit(true))),
            Expr::BoolLiteral(false)
        );
    }

    #[test]
    fn test_fold_conditional_true() {
        let mut f = ConstantFolder::new();
        let e = Expr::Conditional(
            Box::new(Expr::bool_lit(true)),
            Box::new(Expr::int(1)),
            Box::new(Expr::int(2)),
        );
        assert_eq!(f.fold_expr(&e), Expr::IntLiteral(1));
    }

    #[test]
    fn test_fold_conditional_false() {
        let mut f = ConstantFolder::new();
        let e = Expr::Conditional(
            Box::new(Expr::bool_lit(false)),
            Box::new(Expr::int(1)),
            Box::new(Expr::int(2)),
        );
        assert_eq!(f.fold_expr(&e), Expr::IntLiteral(2));
    }

    #[test]
    fn test_fold_xor_self() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::BitXor, Expr::var("x"), Expr::var("x"));
        assert_eq!(f.fold_expr(&e), Expr::IntLiteral(0));
    }

    #[test]
    fn test_fold_comparison_constants() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::Lt, Expr::int(3), Expr::int(5));
        assert_eq!(f.fold_expr(&e), Expr::BoolLiteral(true));
    }

    #[test]
    fn test_fold_if_true() {
        let mut f = ConstantFolder::new();
        let stmt = Stmt::If {
            condition: Expr::bool_lit(true),
            then_body: vec![Stmt::Assign("x".into(), Expr::int(1))],
            else_body: vec![Stmt::Assign("x".into(), Expr::int(2))],
        };
        let result = f.fold_stmt(&stmt);
        // Should become a block with just the then_body
        match result {
            Stmt::Block(stmts) => {
                assert_eq!(stmts.len(), 1);
                assert!(matches!(&stmts[0], Stmt::Assign(n, _) if n == "x"));
            }
            _ => panic!("expected Block, got {:?}", result),
        }
    }

    #[test]
    fn test_fold_if_false_no_else() {
        let mut f = ConstantFolder::new();
        let stmt = Stmt::If {
            condition: Expr::bool_lit(false),
            then_body: vec![Stmt::Assign("x".into(), Expr::int(1))],
            else_body: vec![],
        };
        assert_eq!(f.fold_stmt(&stmt), Stmt::Nop);
    }

    #[test]
    fn test_fold_while_false() {
        let mut f = ConstantFolder::new();
        let stmt = Stmt::While {
            condition: Expr::bool_lit(false),
            body: vec![Stmt::Assign("x".into(), Expr::int(1))],
        };
        assert_eq!(f.fold_stmt(&stmt), Stmt::Nop);
    }

    #[test]
    fn test_fold_dead_loop() {
        let mut f = ConstantFolder::new();
        let stmt = Stmt::SeqFor {
            var: "i".into(),
            start: Expr::int(10),
            end: Expr::int(5),
            step: None,
            body: vec![Stmt::Assign("x".into(), Expr::int(1))],
        };
        assert_eq!(f.fold_stmt(&stmt), Stmt::Nop);
    }

    #[test]
    fn test_fold_parallel_for_zero_procs() {
        let mut f = ConstantFolder::new();
        let stmt = Stmt::ParallelFor {
            proc_var: "p".into(),
            num_procs: Expr::int(0),
            body: vec![Stmt::Nop],
        };
        assert_eq!(f.fold_stmt(&stmt), Stmt::Nop);
    }

    #[test]
    fn test_fold_assign_const_expr() {
        let mut f = ConstantFolder::new();
        let stmt = Stmt::Assign(
            "x".into(),
            Expr::binop(BinOp::Add, Expr::int(3), Expr::int(4)),
        );
        match f.fold_stmt(&stmt) {
            Stmt::Assign(_, Expr::IntLiteral(7)) => {}
            other => panic!("Expected Assign with 7, got {:?}", other),
        }
    }

    #[test]
    fn test_fold_body_removes_nops() {
        let mut f = ConstantFolder::new();
        let body = vec![
            Stmt::Assign("x".into(), Expr::int(1)),
            Stmt::While {
                condition: Expr::bool_lit(false),
                body: vec![Stmt::Nop],
            },
            Stmt::Assign("y".into(), Expr::int(2)),
        ];
        let result = f.fold_body(&body);
        assert_eq!(result.len(), 2); // while(false) removed
    }

    #[test]
    fn test_fold_float_add() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::Add, Expr::float(1.5), Expr::float(2.5));
        assert_eq!(f.fold_expr(&e), Expr::FloatLiteral(4.0));
    }

    #[test]
    fn test_no_fold_non_constant() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::Add, Expr::var("x"), Expr::var("y"));
        let result = f.fold_expr(&e);
        assert!(matches!(result, Expr::BinOp(BinOp::Add, _, _)));
        assert_eq!(f.folds_applied, 0);
    }

    #[test]
    fn test_fold_cast_int_to_float() {
        let mut f = ConstantFolder::new();
        let e = Expr::Cast(Box::new(Expr::int(42)), PramType::Float64);
        assert_eq!(f.fold_expr(&e), Expr::FloatLiteral(42.0));
    }

    #[test]
    fn test_fold_preserves_variable_exprs() {
        let mut f = ConstantFolder::new();
        let e = Expr::shared_read(Expr::var("A"), Expr::var("i"));
        let result = f.fold_expr(&e);
        assert!(matches!(result, Expr::SharedRead(_, _)));
    }

    #[test]
    fn test_fold_bit_and_zero() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::BitAnd, Expr::var("x"), Expr::int(0));
        assert_eq!(f.fold_expr(&e), Expr::IntLiteral(0));
    }

    #[test]
    fn test_fold_bit_or_zero() {
        let mut f = ConstantFolder::new();
        let e = Expr::binop(BinOp::BitOr, Expr::var("x"), Expr::int(0));
        assert_eq!(f.fold_expr(&e), Expr::var("x"));
    }

    // -------------------------------------------------------------------
    // Tests for new functions
    // -------------------------------------------------------------------

    #[test]
    fn test_fold_program_folds_body_and_num_processors() {
        let program = PramProgram {
            name: "test".into(),
            memory_model: MemoryModel::EREW,
            parameters: vec![],
            shared_memory: vec![SharedMemoryDecl {
                name: "A".into(),
                elem_type: PramType::Int64,
                size: Expr::binop(BinOp::Mul, Expr::int(2), Expr::int(4)),
            }],
            body: vec![Stmt::Assign(
                "x".into(),
                Expr::binop(BinOp::Add, Expr::int(1), Expr::int(2)),
            )],
            num_processors: Expr::binop(BinOp::Mul, Expr::int(8), Expr::int(1)),
            work_bound: None,
            time_bound: None,
            description: None,
        };
        let folded = fold_program(&program);
        assert_eq!(folded.num_processors, Expr::IntLiteral(8));
        assert_eq!(folded.shared_memory[0].size, Expr::IntLiteral(8));
        match &folded.body[0] {
            Stmt::Assign(_, Expr::IntLiteral(3)) => {}
            other => panic!("expected folded assign, got {:?}", other),
        }
    }

    #[test]
    fn test_fold_program_preserves_metadata() {
        let program = PramProgram {
            name: "algo".into(),
            memory_model: MemoryModel::CREW,
            parameters: vec![Parameter {
                name: "n".into(),
                param_type: PramType::Int64,
            }],
            shared_memory: vec![],
            body: vec![],
            num_processors: Expr::var("n"),
            work_bound: Some("O(n)".into()),
            time_bound: Some("O(log n)".into()),
            description: Some("test algo".into()),
        };
        let folded = fold_program(&program);
        assert_eq!(folded.name, "algo");
        assert_eq!(folded.memory_model, MemoryModel::CREW);
        assert_eq!(folded.work_bound, Some("O(n)".into()));
        assert_eq!(folded.time_bound, Some("O(log n)".into()));
        assert_eq!(folded.description, Some("test algo".into()));
        assert_eq!(folded.parameters.len(), 1);
    }

    #[test]
    fn test_count_folded_identical() {
        let e = Expr::binop(BinOp::Add, Expr::var("x"), Expr::int(1));
        assert_eq!(count_folded(&e, &e), 0);
    }

    #[test]
    fn test_count_folded_detects_changes() {
        let before = Expr::binop(BinOp::Add, Expr::int(2), Expr::int(3));
        let after = Expr::IntLiteral(5);
        assert!(count_folded(&before, &after) >= 1);
    }

    #[test]
    fn test_count_folded_subtree_change() {
        let before = Expr::binop(
            BinOp::Add,
            Expr::var("x"),
            Expr::binop(BinOp::Mul, Expr::int(1), Expr::var("y")),
        );
        let after = Expr::binop(BinOp::Add, Expr::var("x"), Expr::var("y"));
        // The right child changed from BinOp(Mul, 1, y) -> Variable(y)
        assert!(count_folded(&before, &after) >= 1);
    }

    #[test]
    fn test_deep_fold_nested() {
        // 1 * (0 + x) should fold to x in multiple passes if needed
        let e = Expr::binop(
            BinOp::Mul,
            Expr::int(1),
            Expr::binop(BinOp::Add, Expr::int(0), Expr::var("x")),
        );
        let result = deep_fold(&e);
        assert_eq!(result, Expr::var("x"));
    }

    #[test]
    fn test_deep_fold_already_simple() {
        let e = Expr::var("x");
        assert_eq!(deep_fold(&e), Expr::var("x"));
    }

    #[test]
    fn test_deep_fold_constant_chain() {
        // (2 + 3) * (1 + 1) => 10
        let e = Expr::binop(
            BinOp::Mul,
            Expr::binop(BinOp::Add, Expr::int(2), Expr::int(3)),
            Expr::binop(BinOp::Add, Expr::int(1), Expr::int(1)),
        );
        assert_eq!(deep_fold(&e), Expr::IntLiteral(10));
    }

    #[test]
    fn test_folding_stats_collect_basic() {
        let before = vec![
            Stmt::Assign("x".into(), Expr::binop(BinOp::Add, Expr::int(1), Expr::int(2))),
            Stmt::If {
                condition: Expr::bool_lit(false),
                then_body: vec![Stmt::Assign("y".into(), Expr::int(10))],
                else_body: vec![],
            },
        ];
        let mut folder = ConstantFolder::new();
        let after = folder.fold_body(&before);
        let stats = FoldingStats::collect(&before, &after);
        assert!(stats.total > 0);
        assert!(stats.dead_branch_eliminations > 0);
        assert_eq!(
            stats.total,
            stats.constant_folds + stats.algebraic_simplifications + stats.dead_branch_eliminations
        );
    }

    #[test]
    fn test_folding_stats_collect_no_changes() {
        let stmts = vec![Stmt::Assign("x".into(), Expr::var("y"))];
        let stats = FoldingStats::collect(&stmts, &stmts);
        assert_eq!(stats.total, 0);
    }
}
