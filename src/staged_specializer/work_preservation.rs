//! Work-preservation verification.
//!
//! Asserts that the specializer preserves the O(pT) work bound by
//! counting operations before and after specialization, and verifying
//! post_ops <= c1 * pre_ops + c2.

use crate::pram_ir::ast::{Expr, Stmt};

/// Counts for different operation types in a program.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WorkCount {
    /// Total number of arithmetic/logic operations
    pub arith_ops: usize,
    /// Total number of memory reads
    pub mem_reads: usize,
    /// Total number of memory writes
    pub mem_writes: usize,
    /// Total number of comparisons/branches
    pub branches: usize,
    /// Total number of function calls
    pub calls: usize,
    /// Total number of assignments (local)
    pub assigns: usize,
    /// Total number of loop iterations (estimated)
    pub loop_iters: usize,
    /// Number of shared memory regions
    pub shared_regions: usize,
}

impl WorkCount {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total operation count.
    pub fn total(&self) -> usize {
        self.arith_ops
            + self.mem_reads
            + self.mem_writes
            + self.branches
            + self.calls
            + self.assigns
    }

    /// Merge (add) another count into this one.
    pub fn merge(&mut self, other: &WorkCount) {
        self.arith_ops += other.arith_ops;
        self.mem_reads += other.mem_reads;
        self.mem_writes += other.mem_writes;
        self.branches += other.branches;
        self.calls += other.calls;
        self.assigns += other.assigns;
        self.loop_iters += other.loop_iters;
        self.shared_regions += other.shared_regions;
    }

    /// Scale all counts by a factor.
    pub fn scale(&mut self, factor: usize) {
        self.arith_ops *= factor;
        self.mem_reads *= factor;
        self.mem_writes *= factor;
        self.branches *= factor;
        self.calls *= factor;
        self.assigns *= factor;
    }
}

/// Traverses IR and counts operations.
pub struct WorkCounter;

impl WorkCounter {
    /// Count operations in a list of statements.
    pub fn count(stmts: &[Stmt]) -> WorkCount {
        let mut wc = WorkCount::new();
        for stmt in stmts {
            let sc = Self::count_stmt(stmt);
            wc.merge(&sc);
        }
        wc
    }

    fn count_stmt(stmt: &Stmt) -> WorkCount {
        let mut wc = WorkCount::new();
        match stmt {
            Stmt::Assign(_, expr) => {
                wc.assigns += 1;
                let ec = Self::count_expr(expr);
                wc.merge(&ec);
            }
            Stmt::LocalDecl(_, _, Some(expr)) => {
                wc.assigns += 1;
                let ec = Self::count_expr(expr);
                wc.merge(&ec);
            }
            Stmt::LocalDecl(_, _, None) => {
                wc.assigns += 1;
            }
            Stmt::SharedWrite {
                memory,
                index,
                value,
            } => {
                wc.mem_writes += 1;
                wc.merge(&Self::count_expr(memory));
                wc.merge(&Self::count_expr(index));
                wc.merge(&Self::count_expr(value));
            }
            Stmt::If {
                condition,
                then_body,
                else_body,
            } => {
                wc.branches += 1;
                wc.merge(&Self::count_expr(condition));
                let tc = Self::count(then_body);
                let ec = Self::count(else_body);
                // Conservatively count the max of both branches
                let branch_ops = if tc.total() > ec.total() { tc } else { ec };
                wc.merge(&branch_ops);
            }
            Stmt::SeqFor {
                start,
                end,
                body,
                ..
            } => {
                wc.merge(&Self::count_expr(start));
                wc.merge(&Self::count_expr(end));
                let body_count = Self::count(body);
                // Estimate iterations if bounds are constant
                let iterations = match (start.eval_const_int(), end.eval_const_int()) {
                    (Some(s), Some(e)) if e > s => (e - s) as usize,
                    _ => 1, // conservative: at least one iteration
                };
                wc.loop_iters += iterations;
                let mut scaled = body_count;
                scaled.scale(iterations);
                wc.merge(&scaled);
            }
            Stmt::While { condition, body } => {
                wc.branches += 1;
                wc.merge(&Self::count_expr(condition));
                // While loops: count body once (conservative lower bound)
                let body_count = Self::count(body);
                wc.loop_iters += 1;
                wc.merge(&body_count);
            }
            Stmt::ParallelFor {
                num_procs, body, ..
            } => {
                wc.merge(&Self::count_expr(num_procs));
                let body_count = Self::count(body);
                let num_p = match num_procs.eval_const_int() {
                    Some(p) if p > 0 => p as usize,
                    _ => 1,
                };
                wc.loop_iters += num_p;
                let mut scaled = body_count;
                scaled.scale(num_p);
                wc.merge(&scaled);
            }
            Stmt::Block(stmts) => {
                let bc = Self::count(stmts);
                wc.merge(&bc);
            }
            Stmt::ExprStmt(expr) => {
                let ec = Self::count_expr(expr);
                wc.merge(&ec);
            }
            Stmt::Return(Some(expr)) => {
                let ec = Self::count_expr(expr);
                wc.merge(&ec);
            }
            Stmt::AllocShared { size, .. } => {
                wc.shared_regions += 1;
                wc.merge(&Self::count_expr(size));
            }
            Stmt::Assert(expr, _) => {
                wc.branches += 1;
                wc.merge(&Self::count_expr(expr));
            }
            Stmt::AtomicCAS {
                memory,
                index,
                expected,
                desired,
                ..
            } => {
                wc.mem_reads += 1;
                wc.mem_writes += 1;
                wc.merge(&Self::count_expr(memory));
                wc.merge(&Self::count_expr(index));
                wc.merge(&Self::count_expr(expected));
                wc.merge(&Self::count_expr(desired));
            }
            Stmt::FetchAdd {
                memory,
                index,
                value,
                ..
            } => {
                wc.mem_reads += 1;
                wc.mem_writes += 1;
                wc.merge(&Self::count_expr(memory));
                wc.merge(&Self::count_expr(index));
                wc.merge(&Self::count_expr(value));
            }
            Stmt::PrefixSum { size, .. } => {
                // Prefix sum has O(n) work
                let n = match size.eval_const_int() {
                    Some(s) if s > 0 => s as usize,
                    _ => 1,
                };
                wc.arith_ops += n;
                wc.mem_reads += n;
                wc.mem_writes += n;
            }
            // Nop, Barrier, Comment, FreeShared, Return(None) — no work
            _ => {}
        }
        wc
    }

    fn count_expr(expr: &Expr) -> WorkCount {
        let mut wc = WorkCount::new();
        match expr {
            Expr::BinOp(_, a, b) => {
                wc.arith_ops += 1;
                wc.merge(&Self::count_expr(a));
                wc.merge(&Self::count_expr(b));
            }
            Expr::UnaryOp(_, e) => {
                wc.arith_ops += 1;
                wc.merge(&Self::count_expr(e));
            }
            Expr::SharedRead(mem, idx) => {
                wc.mem_reads += 1;
                wc.merge(&Self::count_expr(mem));
                wc.merge(&Self::count_expr(idx));
            }
            Expr::ArrayIndex(arr, idx) => {
                wc.mem_reads += 1;
                wc.merge(&Self::count_expr(arr));
                wc.merge(&Self::count_expr(idx));
            }
            Expr::FunctionCall(_, args) => {
                wc.calls += 1;
                for arg in args {
                    wc.merge(&Self::count_expr(arg));
                }
            }
            Expr::Cast(e, _) => {
                wc.arith_ops += 1;
                wc.merge(&Self::count_expr(e));
            }
            Expr::Conditional(c, t, e) => {
                wc.branches += 1;
                wc.merge(&Self::count_expr(c));
                wc.merge(&Self::count_expr(t));
                wc.merge(&Self::count_expr(e));
            }
            // Literals, variables, ProcessorId, NumProcessors — no work
            _ => {}
        }
        wc
    }
}

/// Verifies that post-specialization work is bounded by the pre-specialization work.
///
/// Checks: post_ops <= c1 * pre_ops + c2
/// where c1 is the multiplicative overhead constant and c2 is the additive
/// overhead proportional to the number of shared memory regions.
pub struct WorkBoundChecker {
    /// Multiplicative overhead constant (default: 4)
    pub c1: usize,
    /// Additive overhead per shared memory region (default: scales with S)
    pub c2_per_region: usize,
}

impl Default for WorkBoundChecker {
    fn default() -> Self {
        Self {
            c1: 4,
            c2_per_region: 10,
        }
    }
}

impl WorkBoundChecker {
    pub fn new(c1: usize, c2_per_region: usize) -> Self {
        Self { c1, c2_per_region }
    }

    /// Check if the work bound is preserved.
    ///
    /// Returns Ok(()) if post_total <= c1 * pre_total + c2,
    /// or Err with a diagnostic message.
    pub fn check(&self, pre: &WorkCount, post: &WorkCount) -> Result<(), WorkBoundViolation> {
        let pre_total = pre.total();
        let post_total = post.total();
        let c2 = self.c2_per_region * pre.shared_regions.max(post.shared_regions).max(1);
        let bound = self.c1 * pre_total + c2;

        if post_total <= bound {
            Ok(())
        } else {
            Err(WorkBoundViolation {
                pre_ops: pre_total,
                post_ops: post_total,
                bound,
                c1: self.c1,
                c2,
                ratio: if pre_total > 0 {
                    post_total as f64 / pre_total as f64
                } else {
                    f64::INFINITY
                },
            })
        }
    }

    /// Check work preservation for statements before and after transformation.
    pub fn check_stmts(
        &self,
        pre_stmts: &[Stmt],
        post_stmts: &[Stmt],
    ) -> Result<(), WorkBoundViolation> {
        let pre = WorkCounter::count(pre_stmts);
        let post = WorkCounter::count(post_stmts);
        self.check(&pre, &post)
    }
}

/// Describes a work bound violation.
#[derive(Debug, Clone)]
pub struct WorkBoundViolation {
    pub pre_ops: usize,
    pub post_ops: usize,
    pub bound: usize,
    pub c1: usize,
    pub c2: usize,
    pub ratio: f64,
}

impl std::fmt::Display for WorkBoundViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Work bound violation: post_ops={} > {} = {}*{} + {} (ratio={:.2}x)",
            self.post_ops, self.bound, self.c1, self.pre_ops, self.c2, self.ratio
        )
    }
}

/// Per-phase work breakdown.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PhaseWork {
    /// Phase identifier (e.g., "parallel_for_0", "seq_for_1")
    pub phase_name: String,
    /// Work count for this phase
    pub work: WorkCount,
    /// Estimated iterations
    pub iterations: usize,
}

/// Detailed work report with per-phase breakdown.
#[derive(Debug, Clone, Default)]
pub struct DetailedWorkReport {
    /// Per-phase work breakdown
    pub phases: Vec<PhaseWork>,
    /// Aggregate work count
    pub total: WorkCount,
}

impl DetailedWorkReport {
    /// Total operations across all phases.
    pub fn total_ops(&self) -> usize {
        self.total.total()
    }

    /// Number of phases analyzed.
    pub fn phase_count(&self) -> usize {
        self.phases.len()
    }
}

/// Analyze work per phase in a program body.
///
/// Each top-level parallel_for, seq_for, or while loop is treated as a separate phase.
pub fn analyze_work_per_phase(body: &[Stmt]) -> Vec<PhaseWork> {
    let mut phases = Vec::new();
    let mut phase_idx = 0;
    for stmt in body {
        match stmt {
            Stmt::ParallelFor { proc_var, num_procs, body: inner, .. } => {
                let work = WorkCounter::count(inner);
                let iters = num_procs.eval_const_int().unwrap_or(1) as usize;
                phases.push(PhaseWork {
                    phase_name: format!("parallel_for_{}_{}", proc_var, phase_idx),
                    work,
                    iterations: iters,
                });
                phase_idx += 1;
            }
            Stmt::SeqFor { var, start, end, body: inner, .. } => {
                let work = WorkCounter::count(inner);
                let iters = match (start.eval_const_int(), end.eval_const_int()) {
                    (Some(s), Some(e)) if e > s => (e - s) as usize,
                    _ => 1,
                };
                phases.push(PhaseWork {
                    phase_name: format!("seq_for_{}_{}", var, phase_idx),
                    work,
                    iterations: iters,
                });
                phase_idx += 1;
            }
            Stmt::While { body: inner, .. } => {
                let work = WorkCounter::count(inner);
                phases.push(PhaseWork {
                    phase_name: format!("while_{}", phase_idx),
                    work,
                    iterations: 1,
                });
                phase_idx += 1;
            }
            _ => {
                // Non-loop statements: count as a scalar phase
                let work = WorkCounter::count(&[stmt.clone()]);
                if work.total() > 0 {
                    phases.push(PhaseWork {
                        phase_name: format!("scalar_{}", phase_idx),
                        work,
                        iterations: 1,
                    });
                    phase_idx += 1;
                }
            }
        }
    }
    phases
}

/// Estimate cache-related work for a body of statements.
///
/// Counts memory accesses and estimates cache line crossings based on
/// the given cache line size.
pub fn estimate_cache_work(body: &[Stmt], cache_line_size: usize) -> u64 {
    let wc = WorkCounter::count(body);
    let total_mem_ops = (wc.mem_reads + wc.mem_writes) as u64;
    // Estimate: each memory access might cross a cache line boundary
    // Heuristic: total_mem_ops * (1 + 1/cache_line_size) for conflict misses
    let line_size = cache_line_size.max(1) as u64;
    total_mem_ops + total_mem_ops / line_size
}

/// Work inflation analysis result.
#[derive(Debug, Clone)]
pub struct WorkInflation {
    /// Ratio of post-specialization to pre-specialization work.
    pub inflation_factor: f64,
    /// Which operation categories inflated the most.
    pub contributing_factors: Vec<(String, f64)>,
}

/// Analyze work inflation between pre- and post-specialization work counts.
pub fn work_inflation_analysis(pre: &WorkCount, post: &WorkCount) -> WorkInflation {
    let mut factors = Vec::new();

    let mut ratio = |name: &str, a: usize, b: usize| {
        if a > 0 {
            factors.push((name.to_string(), b as f64 / a as f64));
        }
    };

    ratio("arith_ops", pre.arith_ops, post.arith_ops);
    ratio("mem_reads", pre.mem_reads, post.mem_reads);
    ratio("mem_writes", pre.mem_writes, post.mem_writes);
    ratio("branches", pre.branches, post.branches);
    ratio("calls", pre.calls, post.calls);
    ratio("assigns", pre.assigns, post.assigns);

    // Sort by inflation factor (descending)
    factors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let pre_total = pre.total();
    let post_total = post.total();
    let overall = if pre_total > 0 {
        post_total as f64 / pre_total as f64
    } else {
        if post_total > 0 { f64::INFINITY } else { 1.0 }
    };

    WorkInflation {
        inflation_factor: overall,
        contributing_factors: factors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pram_ir::ast::BinOp;

    #[test]
    fn test_count_simple_assign() {
        let stmts = vec![Stmt::Assign("x".to_string(), Expr::int(42))];
        let wc = WorkCounter::count(&stmts);
        assert_eq!(wc.assigns, 1);
        assert_eq!(wc.arith_ops, 0);
        assert_eq!(wc.total(), 1);
    }

    #[test]
    fn test_count_assign_with_binop() {
        let stmts = vec![Stmt::Assign(
            "x".to_string(),
            Expr::binop(BinOp::Add, Expr::var("a"), Expr::var("b")),
        )];
        let wc = WorkCounter::count(&stmts);
        assert_eq!(wc.assigns, 1);
        assert_eq!(wc.arith_ops, 1);
        assert_eq!(wc.total(), 2);
    }

    #[test]
    fn test_count_shared_write() {
        let stmts = vec![Stmt::SharedWrite {
            memory: Expr::var("A"),
            index: Expr::int(0),
            value: Expr::int(42),
        }];
        let wc = WorkCounter::count(&stmts);
        assert_eq!(wc.mem_writes, 1);
        assert_eq!(wc.total(), 1);
    }

    #[test]
    fn test_count_shared_read() {
        let stmts = vec![Stmt::Assign(
            "x".to_string(),
            Expr::shared_read(Expr::var("A"), Expr::int(0)),
        )];
        let wc = WorkCounter::count(&stmts);
        assert_eq!(wc.assigns, 1);
        assert_eq!(wc.mem_reads, 1);
        assert_eq!(wc.total(), 2);
    }

    #[test]
    fn test_count_if_branch() {
        let stmts = vec![Stmt::If {
            condition: Expr::BoolLiteral(true),
            then_body: vec![
                Stmt::Assign("x".to_string(), Expr::int(1)),
                Stmt::Assign("y".to_string(), Expr::int(2)),
            ],
            else_body: vec![Stmt::Assign("z".to_string(), Expr::int(3))],
        }];
        let wc = WorkCounter::count(&stmts);
        assert_eq!(wc.branches, 1);
        // Should count max branch (then has 2 assigns)
        assert_eq!(wc.assigns, 2);
    }

    #[test]
    fn test_count_seq_for() {
        let stmts = vec![Stmt::SeqFor {
            var: "i".to_string(),
            start: Expr::int(0),
            end: Expr::int(10),
            step: None,
            body: vec![Stmt::Assign("x".to_string(), Expr::int(1))],
        }];
        let wc = WorkCounter::count(&stmts);
        assert_eq!(wc.loop_iters, 10);
        // Body has 1 assign, executed 10 times
        assert_eq!(wc.assigns, 10);
    }

    #[test]
    fn test_count_parallel_for() {
        let stmts = vec![Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::int(4),
            body: vec![Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::ProcessorId,
                value: Expr::int(1),
            }],
        }];
        let wc = WorkCounter::count(&stmts);
        assert_eq!(wc.loop_iters, 4);
        assert_eq!(wc.mem_writes, 4); // 4 processors * 1 write each
    }

    #[test]
    fn test_count_prefix_sum() {
        let stmts = vec![Stmt::PrefixSum {
            input: "A".to_string(),
            output: "B".to_string(),
            size: Expr::int(100),
            op: BinOp::Add,
        }];
        let wc = WorkCounter::count(&stmts);
        assert_eq!(wc.arith_ops, 100);
        assert_eq!(wc.mem_reads, 100);
        assert_eq!(wc.mem_writes, 100);
    }

    #[test]
    fn test_work_bound_check_pass() {
        let pre = WorkCount {
            arith_ops: 10,
            mem_reads: 5,
            mem_writes: 5,
            branches: 2,
            calls: 0,
            assigns: 8,
            loop_iters: 0,
            shared_regions: 1,
        };
        let post = WorkCount {
            arith_ops: 20,
            mem_reads: 10,
            mem_writes: 10,
            branches: 4,
            calls: 0,
            assigns: 16,
            loop_iters: 0,
            shared_regions: 1,
        };
        let checker = WorkBoundChecker::default();
        assert!(checker.check(&pre, &post).is_ok());
    }

    #[test]
    fn test_work_bound_check_fail() {
        let pre = WorkCount {
            arith_ops: 1,
            assigns: 1,
            ..WorkCount::default()
        };
        let post = WorkCount {
            arith_ops: 100,
            assigns: 100,
            ..WorkCount::default()
        };
        let checker = WorkBoundChecker::new(2, 5);
        let result = checker.check(&pre, &post);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.post_ops > err.bound);
    }

    #[test]
    fn test_work_bound_check_stmts() {
        let pre_stmts = vec![
            Stmt::Assign("x".to_string(), Expr::int(1)),
            Stmt::Assign("y".to_string(), Expr::int(2)),
        ];
        let post_stmts = vec![
            Stmt::Assign("x".to_string(), Expr::int(1)),
            Stmt::Assign("y".to_string(), Expr::int(2)),
            Stmt::Assign("z".to_string(), Expr::int(3)),
        ];
        let checker = WorkBoundChecker::default();
        assert!(checker.check_stmts(&pre_stmts, &post_stmts).is_ok());
    }

    #[test]
    fn test_work_count_merge() {
        let mut a = WorkCount {
            arith_ops: 5,
            assigns: 3,
            ..WorkCount::default()
        };
        let b = WorkCount {
            arith_ops: 2,
            mem_reads: 1,
            ..WorkCount::default()
        };
        a.merge(&b);
        assert_eq!(a.arith_ops, 7);
        assert_eq!(a.assigns, 3);
        assert_eq!(a.mem_reads, 1);
    }

    #[test]
    fn test_work_count_scale() {
        let mut wc = WorkCount {
            arith_ops: 3,
            mem_reads: 2,
            assigns: 1,
            ..WorkCount::default()
        };
        wc.scale(4);
        assert_eq!(wc.arith_ops, 12);
        assert_eq!(wc.mem_reads, 8);
        assert_eq!(wc.assigns, 4);
    }

    #[test]
    fn test_work_count_total() {
        let wc = WorkCount {
            arith_ops: 1,
            mem_reads: 2,
            mem_writes: 3,
            branches: 4,
            calls: 5,
            assigns: 6,
            loop_iters: 100, // loop_iters not counted in total
            shared_regions: 2,
        };
        assert_eq!(wc.total(), 1 + 2 + 3 + 4 + 5 + 6);
    }

    #[test]
    fn test_count_nop_and_comment() {
        let stmts = vec![Stmt::Nop, Stmt::Comment("hello".to_string()), Stmt::Barrier];
        let wc = WorkCounter::count(&stmts);
        assert_eq!(wc.total(), 0);
    }

    #[test]
    fn test_count_atomic_cas() {
        let stmts = vec![Stmt::AtomicCAS {
            memory: Expr::var("A"),
            index: Expr::int(0),
            expected: Expr::int(0),
            desired: Expr::int(1),
            result_var: "ok".to_string(),
        }];
        let wc = WorkCounter::count(&stmts);
        assert_eq!(wc.mem_reads, 1);
        assert_eq!(wc.mem_writes, 1);
    }

    #[test]
    fn test_count_function_call() {
        let stmts = vec![Stmt::ExprStmt(Expr::FunctionCall(
            "foo".to_string(),
            vec![Expr::int(1), Expr::int(2)],
        ))];
        let wc = WorkCounter::count(&stmts);
        assert_eq!(wc.calls, 1);
    }

    #[test]
    fn test_violation_display() {
        let v = WorkBoundViolation {
            pre_ops: 10,
            post_ops: 100,
            bound: 50,
            c1: 4,
            c2: 10,
            ratio: 10.0,
        };
        let s = format!("{}", v);
        assert!(s.contains("100"));
        assert!(s.contains("50"));
    }

    #[test]
    fn test_count_while_loop() {
        let stmts = vec![Stmt::While {
            condition: Expr::var("flag"),
            body: vec![Stmt::Assign("x".to_string(), Expr::int(1))],
        }];
        let wc = WorkCounter::count(&stmts);
        assert_eq!(wc.branches, 1);
        assert_eq!(wc.assigns, 1);
        assert_eq!(wc.loop_iters, 1);
    }

    #[test]
    fn test_empty_program() {
        let wc = WorkCounter::count(&[]);
        assert_eq!(wc.total(), 0);
        let checker = WorkBoundChecker::default();
        assert!(checker.check(&wc, &wc).is_ok());
    }

    #[test]
    fn test_analyze_work_per_phase() {
        let body = vec![
            Stmt::Assign("x".to_string(), Expr::int(1)),
            Stmt::ParallelFor {
                proc_var: "pid".to_string(),
                num_procs: Expr::int(4),
                body: vec![Stmt::SharedWrite {
                    memory: Expr::var("A"),
                    index: Expr::ProcessorId,
                    value: Expr::int(1),
                }],
            },
            Stmt::SeqFor {
                var: "i".to_string(),
                start: Expr::int(0),
                end: Expr::int(10),
                step: None,
                body: vec![Stmt::Assign("y".to_string(), Expr::int(2))],
            },
        ];
        let phases = analyze_work_per_phase(&body);
        assert_eq!(phases.len(), 3);
        assert!(phases[0].phase_name.starts_with("scalar_"));
        assert!(phases[1].phase_name.starts_with("parallel_for_pid_"));
        assert_eq!(phases[1].iterations, 4);
        assert!(phases[2].phase_name.starts_with("seq_for_i_"));
        assert_eq!(phases[2].iterations, 10);
    }

    #[test]
    fn test_estimate_cache_work() {
        let body = vec![
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::int(1),
            },
            Stmt::Assign(
                "x".to_string(),
                Expr::shared_read(Expr::var("A"), Expr::int(0)),
            ),
        ];
        let cache_work = estimate_cache_work(&body, 64);
        // At least 2 memory ops (1 write + 1 read)
        assert!(cache_work >= 2);
    }

    #[test]
    fn test_work_inflation_analysis() {
        let pre = WorkCount {
            arith_ops: 10,
            mem_reads: 5,
            assigns: 5,
            ..WorkCount::default()
        };
        let post = WorkCount {
            arith_ops: 20,
            mem_reads: 10,
            assigns: 10,
            ..WorkCount::default()
        };
        let inflation = work_inflation_analysis(&pre, &post);
        assert!((inflation.inflation_factor - 2.0).abs() < 0.01);
        assert!(!inflation.contributing_factors.is_empty());
    }

    #[test]
    fn test_work_inflation_no_pre_work() {
        let pre = WorkCount::default();
        let post = WorkCount {
            arith_ops: 5,
            ..WorkCount::default()
        };
        let inflation = work_inflation_analysis(&pre, &post);
        assert!(inflation.inflation_factor.is_infinite());
    }

    #[test]
    fn test_work_inflation_no_change() {
        let wc = WorkCount {
            arith_ops: 10,
            assigns: 5,
            ..WorkCount::default()
        };
        let inflation = work_inflation_analysis(&wc, &wc);
        assert!((inflation.inflation_factor - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_detailed_work_report() {
        let phases = vec![
            PhaseWork {
                phase_name: "phase_0".to_string(),
                work: WorkCount { arith_ops: 5, ..WorkCount::default() },
                iterations: 1,
            },
            PhaseWork {
                phase_name: "phase_1".to_string(),
                work: WorkCount { assigns: 3, ..WorkCount::default() },
                iterations: 10,
            },
        ];
        let report = DetailedWorkReport {
            phases,
            total: WorkCount { arith_ops: 5, assigns: 3, ..WorkCount::default() },
        };
        assert_eq!(report.phase_count(), 2);
        assert_eq!(report.total_ops(), 8);
    }

    #[test]
    fn test_estimate_cache_work_zero_line() {
        let body = vec![Stmt::Assign(
            "x".to_string(),
            Expr::shared_read(Expr::var("A"), Expr::int(0)),
        )];
        // cache_line_size of 0 should be treated as 1
        let cache_work = estimate_cache_work(&body, 0);
        assert!(cache_work >= 1);
    }
}
