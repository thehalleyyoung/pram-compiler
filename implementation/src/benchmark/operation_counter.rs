//! Instrumented operation counting for PRAM IR programs.

use std::fmt;

use crate::pram_ir::ast::*;

/// Accumulated operation counts for a piece of PRAM IR.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpCounts {
    pub arithmetic_ops: u64,
    pub comparison_ops: u64,
    pub memory_reads: u64,
    pub memory_writes: u64,
    pub branch_ops: u64,
    pub total_ops: u64,
}

impl OpCounts {
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge another `OpCounts` into this one (element-wise addition).
    pub fn merge(&mut self, other: &OpCounts) {
        self.arithmetic_ops += other.arithmetic_ops;
        self.comparison_ops += other.comparison_ops;
        self.memory_reads += other.memory_reads;
        self.memory_writes += other.memory_writes;
        self.branch_ops += other.branch_ops;
        self.total_ops += other.total_ops;
    }

    /// Return the sum of two `OpCounts`.
    pub fn combined(&self, other: &OpCounts) -> OpCounts {
        let mut result = self.clone();
        result.merge(other);
        result
    }

    /// Scale all counters by a constant factor (e.g., for loop unrolling estimates).
    pub fn scale(&self, factor: u64) -> OpCounts {
        OpCounts {
            arithmetic_ops: self.arithmetic_ops * factor,
            comparison_ops: self.comparison_ops * factor,
            memory_reads: self.memory_reads * factor,
            memory_writes: self.memory_writes * factor,
            branch_ops: self.branch_ops * factor,
            total_ops: self.total_ops * factor,
        }
    }
}

impl fmt::Display for OpCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "total={}, arith={}, cmp={}, mem_r={}, mem_w={}, branch={}",
            self.total_ops,
            self.arithmetic_ops,
            self.comparison_ops,
            self.memory_reads,
            self.memory_writes,
            self.branch_ops,
        )
    }
}

// ---------------------------------------------------------------------------
// Expression counting
// ---------------------------------------------------------------------------

/// Count the operations in a single expression.
pub fn count_expr(expr: &Expr) -> OpCounts {
    match expr {
        Expr::IntLiteral(_)
        | Expr::FloatLiteral(_)
        | Expr::BoolLiteral(_)
        | Expr::Variable(_)
        | Expr::ProcessorId
        | Expr::NumProcessors => OpCounts::new(),

        Expr::BinOp(op, lhs, rhs) => {
            let mut counts = count_expr(lhs).combined(&count_expr(rhs));
            if op.is_arithmetic() {
                counts.arithmetic_ops += 1;
            } else if op.is_comparison() {
                counts.comparison_ops += 1;
            } else {
                // logical / bitwise – count as arithmetic
                counts.arithmetic_ops += 1;
            }
            counts.total_ops += 1;
            counts
        }

        Expr::UnaryOp(_op, inner) => {
            let mut counts = count_expr(inner);
            counts.arithmetic_ops += 1;
            counts.total_ops += 1;
            counts
        }

        Expr::SharedRead(mem, idx) => {
            let mut counts = count_expr(mem).combined(&count_expr(idx));
            counts.memory_reads += 1;
            counts.total_ops += 1;
            counts
        }

        Expr::ArrayIndex(arr, idx) => {
            let mut counts = count_expr(arr).combined(&count_expr(idx));
            counts.memory_reads += 1;
            counts.total_ops += 1;
            counts
        }

        Expr::FunctionCall(_name, args) => {
            let mut counts = OpCounts::new();
            for arg in args {
                counts.merge(&count_expr(arg));
            }
            // The call itself counts as one arithmetic op
            counts.arithmetic_ops += 1;
            counts.total_ops += 1;
            counts
        }

        Expr::Cast(inner, _ty) => {
            let mut counts = count_expr(inner);
            counts.arithmetic_ops += 1;
            counts.total_ops += 1;
            counts
        }

        Expr::Conditional(cond, then_expr, else_expr) => {
            let mut counts = count_expr(cond);
            let then_counts = count_expr(then_expr);
            let else_counts = count_expr(else_expr);
            // Conservatively count both branches
            counts.merge(&then_counts);
            counts.merge(&else_counts);
            counts.branch_ops += 1;
            counts.total_ops += 1;
            counts
        }
    }
}

// ---------------------------------------------------------------------------
// Statement counting
// ---------------------------------------------------------------------------

/// Count the operations in a single statement.
pub fn count_stmt(stmt: &Stmt) -> OpCounts {
    match stmt {
        Stmt::LocalDecl(_name, _ty, init) => {
            let mut counts = OpCounts::new();
            if let Some(expr) = init {
                counts.merge(&count_expr(expr));
            }
            counts
        }

        Stmt::Assign(_name, expr) => count_expr(expr),

        Stmt::SharedWrite { memory, index, value } => {
            let mut counts = count_expr(memory);
            counts.merge(&count_expr(index));
            counts.merge(&count_expr(value));
            counts.memory_writes += 1;
            counts.total_ops += 1;
            counts
        }

        Stmt::ParallelFor { num_procs, body, .. } => {
            let mut body_counts = OpCounts::new();
            for s in body {
                body_counts.merge(&count_stmt(s));
            }
            // Estimate: multiply body costs by the number of processors (if constant)
            let factor = num_procs.eval_const_int().unwrap_or(1).max(1) as u64;
            let mut counts = count_expr(num_procs);
            counts.merge(&body_counts.scale(factor));
            counts
        }

        Stmt::SeqFor { start, end, step, body, .. } => {
            let mut counts = count_expr(start);
            counts.merge(&count_expr(end));
            if let Some(s) = step {
                counts.merge(&count_expr(s));
            }
            // Estimate iterations
            let iters = match (start.eval_const_int(), end.eval_const_int()) {
                (Some(s), Some(e)) => {
                    let step_val = step
                        .as_ref()
                        .and_then(|sv| sv.eval_const_int())
                        .unwrap_or(1)
                        .max(1);
                    ((e - s) / step_val).max(0) as u64
                }
                _ => 1,
            };

            let mut body_counts = OpCounts::new();
            for s in body {
                body_counts.merge(&count_stmt(s));
            }
            // Per-iteration overhead: one comparison + one arithmetic (increment)
            let iter_overhead = OpCounts {
                comparison_ops: 1,
                arithmetic_ops: 1,
                total_ops: 2,
                ..Default::default()
            };
            body_counts.merge(&iter_overhead);
            counts.merge(&body_counts.scale(iters));
            counts
        }

        Stmt::While { condition, body } => {
            // Cannot statically determine iteration count; count one iteration
            let mut counts = count_expr(condition);
            counts.branch_ops += 1;
            counts.total_ops += 1;
            for s in body {
                counts.merge(&count_stmt(s));
            }
            counts
        }

        Stmt::If { condition, then_body, else_body } => {
            let mut counts = count_expr(condition);
            counts.branch_ops += 1;
            counts.total_ops += 1;
            // Conservatively count both branches
            for s in then_body {
                counts.merge(&count_stmt(s));
            }
            for s in else_body {
                counts.merge(&count_stmt(s));
            }
            counts
        }

        Stmt::Barrier => OpCounts::new(),

        Stmt::Block(stmts) => {
            let mut counts = OpCounts::new();
            for s in stmts {
                counts.merge(&count_stmt(s));
            }
            counts
        }

        Stmt::ExprStmt(expr) => count_expr(expr),

        Stmt::Return(opt_expr) => {
            if let Some(expr) = opt_expr {
                count_expr(expr)
            } else {
                OpCounts::new()
            }
        }

        Stmt::AllocShared { size, .. } => {
            let mut counts = count_expr(size);
            counts.memory_writes += 1;
            counts.total_ops += 1;
            counts
        }

        Stmt::FreeShared(_) => {
            OpCounts {
                total_ops: 1,
                ..Default::default()
            }
        }

        Stmt::Nop => OpCounts::new(),

        Stmt::AtomicCAS { memory, index, expected, desired, .. } => {
            let mut counts = count_expr(memory);
            counts.merge(&count_expr(index));
            counts.merge(&count_expr(expected));
            counts.merge(&count_expr(desired));
            counts.memory_reads += 1;
            counts.memory_writes += 1;
            counts.comparison_ops += 1;
            counts.total_ops += 3;
            counts
        }

        Stmt::FetchAdd { memory, index, value, .. } => {
            let mut counts = count_expr(memory);
            counts.merge(&count_expr(index));
            counts.merge(&count_expr(value));
            counts.memory_reads += 1;
            counts.memory_writes += 1;
            counts.arithmetic_ops += 1;
            counts.total_ops += 3;
            counts
        }

        Stmt::Assert(expr, _msg) => {
            let mut counts = count_expr(expr);
            counts.comparison_ops += 1;
            counts.total_ops += 1;
            counts
        }

        Stmt::Comment(_) => OpCounts::new(),

        Stmt::PrefixSum { size, .. } => {
            // Prefix sum on n elements: ~2n operations (up-sweep + down-sweep)
            let n = size.eval_const_int().unwrap_or(1).max(1) as u64;
            OpCounts {
                arithmetic_ops: 2 * n,
                memory_reads: 2 * n,
                memory_writes: 2 * n,
                total_ops: 6 * n,
                ..Default::default()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Program counting
// ---------------------------------------------------------------------------

/// Count the total operations in an entire PRAM program.
pub fn count_program(program: &PramProgram) -> OpCounts {
    let mut counts = OpCounts::new();
    for stmt in &program.body {
        counts.merge(&count_stmt(stmt));
    }
    counts
}

// ---------------------------------------------------------------------------
// Category-level counting
// ---------------------------------------------------------------------------

/// Operation counts bucketed into high-level categories.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CategoryCounts {
    pub arithmetic: u64,
    pub comparison: u64,
    pub memory: u64,
    pub branch: u64,
    pub barrier: u64,
    pub total: u64,
}

fn count_barriers_in_stmts(stmts: &[Stmt]) -> u64 {
    let mut count = 0;
    for stmt in stmts {
        match stmt {
            Stmt::Barrier => count += 1,
            Stmt::ParallelFor { body, .. } => count += count_barriers_in_stmts(body),
            Stmt::SeqFor { body, .. } => count += count_barriers_in_stmts(body),
            Stmt::While { body, .. } => count += count_barriers_in_stmts(body),
            Stmt::If { then_body, else_body, .. } => {
                count += count_barriers_in_stmts(then_body)
                    + count_barriers_in_stmts(else_body);
            }
            Stmt::Block(inner) => count += count_barriers_in_stmts(inner),
            _ => {}
        }
    }
    count
}

/// Walk the program body and count operations grouped by category.
///
/// Barriers are counted separately since `count_program` treats them as zero-cost.
pub fn count_by_category(program: &PramProgram) -> CategoryCounts {
    let ops = count_program(program);
    let barriers = count_barriers_in_stmts(&program.body);
    CategoryCounts {
        arithmetic: ops.arithmetic_ops,
        comparison: ops.comparison_ops,
        memory: ops.memory_reads + ops.memory_writes,
        branch: ops.branch_ops,
        barrier: barriers,
        total: ops.total_ops + barriers,
    }
}

// ---------------------------------------------------------------------------
// Density & distribution helpers
// ---------------------------------------------------------------------------

fn count_stmts_recursive(stmts: &[Stmt]) -> u64 {
    let mut count = 0;
    for stmt in stmts {
        count += 1;
        match stmt {
            Stmt::ParallelFor { body, .. } => count += count_stmts_recursive(body),
            Stmt::SeqFor { body, .. } => count += count_stmts_recursive(body),
            Stmt::While { body, .. } => count += count_stmts_recursive(body),
            Stmt::If { then_body, else_body, .. } => {
                count += count_stmts_recursive(then_body)
                    + count_stmts_recursive(else_body);
            }
            Stmt::Block(inner) => count += count_stmts_recursive(inner),
            _ => {}
        }
    }
    count
}

/// Compute total operations divided by the total number of statements
/// (counted recursively). Returns `0.0` if there are no statements.
pub fn operation_density(program: &PramProgram) -> f64 {
    let total_stmts = count_stmts_recursive(&program.body);
    if total_stmts == 0 {
        return 0.0;
    }
    let ops = count_program(program);
    ops.total_ops as f64 / total_stmts as f64
}

fn describe_stmt(stmt: &Stmt) -> String {
    match stmt {
        Stmt::ParallelFor { proc_var, num_procs, .. } => {
            if let Some(n) = num_procs.eval_const_int() {
                format!("ParallelFor({}, n={})", proc_var, n)
            } else {
                format!("ParallelFor({})", proc_var)
            }
        }
        Stmt::SeqFor { var, .. } => format!("SeqFor({})", var),
        Stmt::If { .. } => "If".to_string(),
        Stmt::While { .. } => "While".to_string(),
        Stmt::Barrier => "Barrier".to_string(),
        Stmt::Assign(name, _) => format!("Assign({})", name),
        Stmt::SharedWrite { .. } => "SharedWrite".to_string(),
        Stmt::LocalDecl(name, _, _) => format!("LocalDecl({})", name),
        Stmt::Block(_) => "Block".to_string(),
        Stmt::Return(_) => "Return".to_string(),
        Stmt::AtomicCAS { .. } => "AtomicCAS".to_string(),
        Stmt::FetchAdd { .. } => "FetchAdd".to_string(),
        Stmt::AllocShared { name, .. } => format!("AllocShared({})", name),
        Stmt::FreeShared(name) => format!("FreeShared({})", name),
        Stmt::ExprStmt(_) => "ExprStmt".to_string(),
        Stmt::Assert(_, _) => "Assert".to_string(),
        Stmt::Comment(_) => "Comment".to_string(),
        Stmt::PrefixSum { .. } => "PrefixSum".to_string(),
        Stmt::Nop => "Nop".to_string(),
    }
}

/// For each top-level statement in the program body, compute its total ops and
/// return a vector of `(description, op_count)` pairs.
pub fn work_distribution(program: &PramProgram) -> Vec<(String, u64)> {
    program
        .body
        .iter()
        .map(|stmt| {
            let desc = describe_stmt(stmt);
            let ops = count_stmt(stmt).total_ops;
            (desc, ops)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_literal() {
        let counts = count_expr(&Expr::int(42));
        assert_eq!(counts.total_ops, 0);
    }

    #[test]
    fn test_count_binop_arithmetic() {
        let expr = Expr::binop(BinOp::Add, Expr::int(1), Expr::int(2));
        let counts = count_expr(&expr);
        assert_eq!(counts.arithmetic_ops, 1);
        assert_eq!(counts.total_ops, 1);
    }

    #[test]
    fn test_count_binop_comparison() {
        let expr = Expr::binop(BinOp::Lt, Expr::var("x"), Expr::int(10));
        let counts = count_expr(&expr);
        assert_eq!(counts.comparison_ops, 1);
        assert_eq!(counts.total_ops, 1);
    }

    #[test]
    fn test_count_shared_read() {
        let expr = Expr::shared_read(Expr::var("A"), Expr::var("i"));
        let counts = count_expr(&expr);
        assert_eq!(counts.memory_reads, 1);
        assert_eq!(counts.total_ops, 1);
    }

    #[test]
    fn test_count_nested_expr() {
        // (x + y) < (a * b)
        let expr = Expr::binop(
            BinOp::Lt,
            Expr::binop(BinOp::Add, Expr::var("x"), Expr::var("y")),
            Expr::binop(BinOp::Mul, Expr::var("a"), Expr::var("b")),
        );
        let counts = count_expr(&expr);
        assert_eq!(counts.arithmetic_ops, 2); // Add + Mul
        assert_eq!(counts.comparison_ops, 1); // Lt
        assert_eq!(counts.total_ops, 3);
    }

    #[test]
    fn test_count_conditional_expr() {
        let expr = Expr::Conditional(
            Box::new(Expr::binop(BinOp::Lt, Expr::var("x"), Expr::int(0))),
            Box::new(Expr::int(1)),
            Box::new(Expr::int(0)),
        );
        let counts = count_expr(&expr);
        assert_eq!(counts.branch_ops, 1);
        assert_eq!(counts.comparison_ops, 1);
    }

    #[test]
    fn test_count_assign_stmt() {
        let stmt = Stmt::Assign("x".to_string(), Expr::binop(BinOp::Add, Expr::var("a"), Expr::int(1)));
        let counts = count_stmt(&stmt);
        assert_eq!(counts.arithmetic_ops, 1);
        assert_eq!(counts.total_ops, 1);
    }

    #[test]
    fn test_count_shared_write_stmt() {
        let stmt = Stmt::SharedWrite {
            memory: Expr::var("A"),
            index: Expr::var("i"),
            value: Expr::int(42),
        };
        let counts = count_stmt(&stmt);
        assert_eq!(counts.memory_writes, 1);
        assert_eq!(counts.total_ops, 1);
    }

    #[test]
    fn test_count_if_stmt() {
        let stmt = Stmt::If {
            condition: Expr::binop(BinOp::Lt, Expr::var("x"), Expr::int(10)),
            then_body: vec![Stmt::Assign("y".to_string(), Expr::int(1))],
            else_body: vec![Stmt::Assign("y".to_string(), Expr::int(0))],
        };
        let counts = count_stmt(&stmt);
        assert_eq!(counts.branch_ops, 1);
        assert_eq!(counts.comparison_ops, 1);
    }

    #[test]
    fn test_count_seq_for() {
        // for i in 0..10: x = x + 1
        let stmt = Stmt::SeqFor {
            var: "i".to_string(),
            start: Expr::int(0),
            end: Expr::int(10),
            step: None,
            body: vec![Stmt::Assign(
                "x".to_string(),
                Expr::binop(BinOp::Add, Expr::var("x"), Expr::int(1)),
            )],
        };
        let counts = count_stmt(&stmt);
        // 10 iterations * (1 add + 1 loop comparison + 1 loop increment) + expr evals
        assert!(counts.arithmetic_ops >= 10);
        assert!(counts.total_ops >= 10);
    }

    #[test]
    fn test_count_parallel_for() {
        let stmt = Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::int(4),
            body: vec![Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::ProcessorId,
                value: Expr::int(1),
            }],
        };
        let counts = count_stmt(&stmt);
        // 4 procs * 1 write each
        assert_eq!(counts.memory_writes, 4);
    }

    #[test]
    fn test_count_atomic_cas() {
        let stmt = Stmt::AtomicCAS {
            memory: Expr::var("A"),
            index: Expr::int(0),
            expected: Expr::int(0),
            desired: Expr::int(1),
            result_var: "success".to_string(),
        };
        let counts = count_stmt(&stmt);
        assert_eq!(counts.memory_reads, 1);
        assert_eq!(counts.memory_writes, 1);
        assert_eq!(counts.comparison_ops, 1);
    }

    #[test]
    fn test_count_fetch_add() {
        let stmt = Stmt::FetchAdd {
            memory: Expr::var("counter"),
            index: Expr::int(0),
            value: Expr::int(1),
            result_var: "old".to_string(),
        };
        let counts = count_stmt(&stmt);
        assert_eq!(counts.memory_reads, 1);
        assert_eq!(counts.memory_writes, 1);
        assert_eq!(counts.arithmetic_ops, 1);
    }

    #[test]
    fn test_count_prefix_sum() {
        let stmt = Stmt::PrefixSum {
            input: "A".to_string(),
            output: "B".to_string(),
            size: Expr::int(100),
            op: BinOp::Add,
        };
        let counts = count_stmt(&stmt);
        assert_eq!(counts.arithmetic_ops, 200);
        assert_eq!(counts.memory_reads, 200);
        assert_eq!(counts.memory_writes, 200);
    }

    #[test]
    fn test_count_program_simple() {
        let program = PramProgram {
            name: "test".to_string(),
            memory_model: MemoryModel::CREW,
            parameters: vec![],
            shared_memory: vec![],
            body: vec![
                Stmt::Assign("x".to_string(), Expr::binop(BinOp::Add, Expr::int(1), Expr::int(2))),
                Stmt::Assign("y".to_string(), Expr::binop(BinOp::Mul, Expr::var("x"), Expr::int(3))),
            ],
            num_processors: Expr::int(1),
            work_bound: None,
            time_bound: None,
            description: None,
        };
        let counts = count_program(&program);
        assert_eq!(counts.arithmetic_ops, 2);
        assert_eq!(counts.total_ops, 2);
    }

    #[test]
    fn test_opcounts_display() {
        let counts = OpCounts {
            arithmetic_ops: 10,
            comparison_ops: 5,
            memory_reads: 3,
            memory_writes: 2,
            branch_ops: 1,
            total_ops: 21,
        };
        let s = format!("{}", counts);
        assert!(s.contains("total=21"));
        assert!(s.contains("arith=10"));
    }

    #[test]
    fn test_opcounts_merge() {
        let mut a = OpCounts { arithmetic_ops: 5, total_ops: 5, ..Default::default() };
        let b = OpCounts { comparison_ops: 3, total_ops: 3, ..Default::default() };
        a.merge(&b);
        assert_eq!(a.arithmetic_ops, 5);
        assert_eq!(a.comparison_ops, 3);
        assert_eq!(a.total_ops, 8);
    }

    #[test]
    fn test_opcounts_scale() {
        let c = OpCounts { arithmetic_ops: 2, memory_reads: 1, total_ops: 3, ..Default::default() };
        let scaled = c.scale(10);
        assert_eq!(scaled.arithmetic_ops, 20);
        assert_eq!(scaled.memory_reads, 10);
        assert_eq!(scaled.total_ops, 30);
    }

    #[test]
    fn test_count_nop() {
        let counts = count_stmt(&Stmt::Nop);
        assert_eq!(counts.total_ops, 0);
    }

    #[test]
    fn test_count_barrier() {
        let counts = count_stmt(&Stmt::Barrier);
        assert_eq!(counts.total_ops, 0);
    }

    #[test]
    fn test_count_comment() {
        let counts = count_stmt(&Stmt::Comment("hello".to_string()));
        assert_eq!(counts.total_ops, 0);
    }

    #[test]
    fn test_count_block() {
        let stmt = Stmt::Block(vec![
            Stmt::Assign("x".to_string(), Expr::int(1)),
            Stmt::Assign("y".to_string(), Expr::binop(BinOp::Add, Expr::var("x"), Expr::int(2))),
        ]);
        let counts = count_stmt(&stmt);
        assert_eq!(counts.arithmetic_ops, 1);
    }

    // -------------------------------------------------------------------
    // Tests for CategoryCounts, operation_density, work_distribution
    // -------------------------------------------------------------------

    fn make_program(body: Vec<Stmt>) -> PramProgram {
        PramProgram {
            name: "test".to_string(),
            memory_model: MemoryModel::CREW,
            parameters: vec![],
            shared_memory: vec![],
            body,
            num_processors: Expr::int(1),
            work_bound: None,
            time_bound: None,
            description: None,
        }
    }

    #[test]
    fn test_category_counts_arithmetic() {
        let prog = make_program(vec![
            Stmt::Assign("x".to_string(), Expr::binop(BinOp::Add, Expr::int(1), Expr::int(2))),
            Stmt::Assign("y".to_string(), Expr::binop(BinOp::Mul, Expr::var("x"), Expr::int(3))),
        ]);
        let cat = count_by_category(&prog);
        assert_eq!(cat.arithmetic, 2);
        assert_eq!(cat.comparison, 0);
        assert_eq!(cat.memory, 0);
        assert_eq!(cat.branch, 0);
        assert_eq!(cat.barrier, 0);
        assert_eq!(cat.total, 2);
    }

    #[test]
    fn test_category_counts_with_barriers() {
        let prog = make_program(vec![
            Stmt::Assign("x".to_string(), Expr::int(1)),
            Stmt::Barrier,
            Stmt::Assign("y".to_string(), Expr::binop(BinOp::Add, Expr::var("x"), Expr::int(2))),
            Stmt::Barrier,
        ]);
        let cat = count_by_category(&prog);
        assert_eq!(cat.barrier, 2);
        assert_eq!(cat.arithmetic, 1);
        assert_eq!(cat.total, cat.arithmetic + cat.comparison + cat.memory + cat.branch + cat.barrier);
    }

    #[test]
    fn test_category_counts_memory() {
        let prog = make_program(vec![
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::int(42),
            },
            Stmt::Assign(
                "v".to_string(),
                Expr::shared_read(Expr::var("A"), Expr::int(0)),
            ),
        ]);
        let cat = count_by_category(&prog);
        assert_eq!(cat.memory, 2); // 1 write + 1 read
    }

    #[test]
    fn test_operation_density_simple() {
        // Two top-level assigns, each with 1 arithmetic op => 2 ops / 2 stmts = 1.0
        let prog = make_program(vec![
            Stmt::Assign("x".to_string(), Expr::binop(BinOp::Add, Expr::int(1), Expr::int(2))),
            Stmt::Assign("y".to_string(), Expr::binop(BinOp::Mul, Expr::int(3), Expr::int(4))),
        ]);
        let density = operation_density(&prog);
        assert!((density - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_operation_density_empty() {
        let prog = make_program(vec![]);
        assert_eq!(operation_density(&prog), 0.0);
    }

    #[test]
    fn test_work_distribution_descriptions() {
        let prog = make_program(vec![
            Stmt::ParallelFor {
                proc_var: "pid".to_string(),
                num_procs: Expr::int(100),
                body: vec![Stmt::Nop],
            },
            Stmt::SeqFor {
                var: "i".to_string(),
                start: Expr::int(0),
                end: Expr::int(10),
                step: None,
                body: vec![Stmt::Nop],
            },
            Stmt::If {
                condition: Expr::BoolLiteral(true),
                then_body: vec![],
                else_body: vec![],
            },
            Stmt::Barrier,
        ]);
        let dist = work_distribution(&prog);
        assert_eq!(dist.len(), 4);
        assert_eq!(dist[0].0, "ParallelFor(pid, n=100)");
        assert_eq!(dist[1].0, "SeqFor(i)");
        assert_eq!(dist[2].0, "If");
        assert_eq!(dist[3].0, "Barrier");
    }

    #[test]
    fn test_work_distribution_ops() {
        let prog = make_program(vec![
            Stmt::Assign("x".to_string(), Expr::binop(BinOp::Add, Expr::int(1), Expr::int(2))),
            Stmt::Nop,
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::int(5),
            },
        ]);
        let dist = work_distribution(&prog);
        assert_eq!(dist.len(), 3);
        assert_eq!(dist[0], ("Assign(x)".to_string(), 1)); // 1 Add
        assert_eq!(dist[1], ("Nop".to_string(), 0));
        assert_eq!(dist[2], ("SharedWrite".to_string(), 1)); // 1 write
    }
}
