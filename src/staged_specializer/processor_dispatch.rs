//! Processor dispatch: inline processor-ID dispatch by replacing parallel_for
//! with sequential iteration or full unrolling.

use crate::pram_ir::ast::{BinOp, Expr, Stmt};
use crate::pram_ir::types::PramType;

/// Configuration for the processor dispatch pass.
#[derive(Debug, Clone)]
pub struct DispatchConfig {
    /// Threshold for full unrolling vs. emitting a sequential loop.
    /// If num_procs <= threshold, fully unroll.
    pub unroll_threshold: usize,
}

impl Default for DispatchConfig {
    fn default() -> Self {
        Self {
            unroll_threshold: 32,
        }
    }
}

impl DispatchConfig {
    pub fn new(threshold: usize) -> Self {
        Self {
            unroll_threshold: threshold,
        }
    }
}

/// The processor dispatch pass.
pub struct ProcessorDispatch {
    config: DispatchConfig,
}

impl ProcessorDispatch {
    pub fn new(config: DispatchConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self {
            config: DispatchConfig::default(),
        }
    }

    /// Transform a list of statements, replacing ParallelFor with sequential code.
    pub fn transform(&self, stmts: &[Stmt]) -> Vec<Stmt> {
        let mut result = Vec::new();
        for stmt in stmts {
            let mut transformed = self.transform_stmt(stmt);
            result.append(&mut transformed);
        }
        result
    }

    fn transform_stmt(&self, stmt: &Stmt) -> Vec<Stmt> {
        match stmt {
            Stmt::ParallelFor {
                proc_var,
                num_procs,
                body,
            } => {
                // First, recursively transform any nested parallel_for in the body
                let body = self.transform(body);
                self.dispatch_parallel_for(proc_var, num_procs, &body)
            }
            Stmt::If {
                condition,
                then_body,
                else_body,
            } => {
                vec![Stmt::If {
                    condition: condition.clone(),
                    then_body: self.transform(then_body),
                    else_body: self.transform(else_body),
                }]
            }
            Stmt::SeqFor {
                var,
                start,
                end,
                step,
                body,
            } => {
                vec![Stmt::SeqFor {
                    var: var.clone(),
                    start: start.clone(),
                    end: end.clone(),
                    step: step.clone(),
                    body: self.transform(body),
                }]
            }
            Stmt::While { condition, body } => {
                vec![Stmt::While {
                    condition: condition.clone(),
                    body: self.transform(body),
                }]
            }
            Stmt::Block(stmts) => {
                vec![Stmt::Block(self.transform(stmts))]
            }
            other => vec![other.clone()],
        }
    }

    /// Replace a parallel_for with either unrolled code or a sequential loop.
    fn dispatch_parallel_for(
        &self,
        proc_var: &str,
        num_procs: &Expr,
        body: &[Stmt],
    ) -> Vec<Stmt> {
        // Try to evaluate num_procs as a constant
        if let Some(p) = num_procs.eval_const_int() {
            let p = p as usize;
            if p == 0 {
                return vec![Stmt::Comment(format!(
                    "parallel_for {} in 0..0: empty",
                    proc_var
                ))];
            }
            if p <= self.config.unroll_threshold {
                return self.fully_unroll(proc_var, p, body);
            }
        }

        // Emit a sequential for loop
        self.emit_sequential_loop(proc_var, num_procs, body)
    }

    /// Fully unroll a parallel_for: emit separate code for each processor.
    fn fully_unroll(&self, proc_var: &str, num_procs: usize, body: &[Stmt]) -> Vec<Stmt> {
        let mut result = Vec::new();
        result.push(Stmt::Comment(format!(
            "BEGIN unrolled parallel_for {} in 0..{}", proc_var, num_procs
        )));

        for pid in 0..num_procs {
            result.push(Stmt::Comment(format!("processor {} = {}", proc_var, pid)));
            let replacement = Expr::IntLiteral(pid as i64);
            for stmt in body {
                result.push(substitute_in_stmt(stmt, proc_var, &replacement));
            }
        }

        result.push(Stmt::Comment(format!(
            "END unrolled parallel_for {}", proc_var
        )));
        result
    }

    /// Emit a sequential for loop replacing the parallel_for.
    fn emit_sequential_loop(&self, proc_var: &str, num_procs: &Expr, body: &[Stmt]) -> Vec<Stmt> {
        vec![
            Stmt::Comment(format!(
                "sequential dispatch for parallel_for {} in 0..{}",
                proc_var,
                format_expr_brief(num_procs)
            )),
            Stmt::SeqFor {
                var: proc_var.to_string(),
                start: Expr::IntLiteral(0),
                end: num_procs.clone(),
                step: None,
                body: body.to_vec(),
            },
        ]
    }
}

/// Substitute all occurrences of `proc_var` (as a Variable or ProcessorId)
/// in a statement with `replacement`.
fn substitute_in_stmt(stmt: &Stmt, var: &str, replacement: &Expr) -> Stmt {
    match stmt {
        Stmt::Assign(name, expr) => {
            Stmt::Assign(name.clone(), substitute_in_expr(expr, var, replacement))
        }
        Stmt::LocalDecl(name, ty, init) => Stmt::LocalDecl(
            name.clone(),
            ty.clone(),
            init.as_ref()
                .map(|e| substitute_in_expr(e, var, replacement)),
        ),
        Stmt::SharedWrite {
            memory,
            index,
            value,
        } => Stmt::SharedWrite {
            memory: substitute_in_expr(memory, var, replacement),
            index: substitute_in_expr(index, var, replacement),
            value: substitute_in_expr(value, var, replacement),
        },
        Stmt::If {
            condition,
            then_body,
            else_body,
        } => Stmt::If {
            condition: substitute_in_expr(condition, var, replacement),
            then_body: then_body
                .iter()
                .map(|s| substitute_in_stmt(s, var, replacement))
                .collect(),
            else_body: else_body
                .iter()
                .map(|s| substitute_in_stmt(s, var, replacement))
                .collect(),
        },
        Stmt::SeqFor {
            var: loop_var,
            start,
            end,
            step,
            body,
        } => {
            // Don't substitute the loop variable itself
            if loop_var == var {
                return stmt.clone();
            }
            Stmt::SeqFor {
                var: loop_var.clone(),
                start: substitute_in_expr(start, var, replacement),
                end: substitute_in_expr(end, var, replacement),
                step: step
                    .as_ref()
                    .map(|s| substitute_in_expr(s, var, replacement)),
                body: body
                    .iter()
                    .map(|s| substitute_in_stmt(s, var, replacement))
                    .collect(),
            }
        }
        Stmt::While { condition, body } => Stmt::While {
            condition: substitute_in_expr(condition, var, replacement),
            body: body
                .iter()
                .map(|s| substitute_in_stmt(s, var, replacement))
                .collect(),
        },
        Stmt::ParallelFor {
            proc_var,
            num_procs,
            body,
        } => {
            // Don't substitute if the inner parallel_for rebinds the same variable
            if proc_var == var {
                return stmt.clone();
            }
            Stmt::ParallelFor {
                proc_var: proc_var.clone(),
                num_procs: substitute_in_expr(num_procs, var, replacement),
                body: body
                    .iter()
                    .map(|s| substitute_in_stmt(s, var, replacement))
                    .collect(),
            }
        }
        Stmt::Block(stmts) => Stmt::Block(
            stmts
                .iter()
                .map(|s| substitute_in_stmt(s, var, replacement))
                .collect(),
        ),
        Stmt::ExprStmt(e) => Stmt::ExprStmt(substitute_in_expr(e, var, replacement)),
        Stmt::Return(Some(e)) => Stmt::Return(Some(substitute_in_expr(e, var, replacement))),
        Stmt::Assert(e, msg) => Stmt::Assert(substitute_in_expr(e, var, replacement), msg.clone()),
        Stmt::AllocShared {
            name,
            elem_type,
            size,
        } => Stmt::AllocShared {
            name: name.clone(),
            elem_type: elem_type.clone(),
            size: substitute_in_expr(size, var, replacement),
        },
        Stmt::AtomicCAS {
            memory,
            index,
            expected,
            desired,
            result_var,
        } => Stmt::AtomicCAS {
            memory: substitute_in_expr(memory, var, replacement),
            index: substitute_in_expr(index, var, replacement),
            expected: substitute_in_expr(expected, var, replacement),
            desired: substitute_in_expr(desired, var, replacement),
            result_var: result_var.clone(),
        },
        Stmt::FetchAdd {
            memory,
            index,
            value,
            result_var,
        } => Stmt::FetchAdd {
            memory: substitute_in_expr(memory, var, replacement),
            index: substitute_in_expr(index, var, replacement),
            value: substitute_in_expr(value, var, replacement),
            result_var: result_var.clone(),
        },
        Stmt::PrefixSum {
            input,
            output,
            size,
            op,
        } => Stmt::PrefixSum {
            input: input.clone(),
            output: output.clone(),
            size: substitute_in_expr(size, var, replacement),
            op: *op,
        },
        other => other.clone(),
    }
}

/// Substitute a variable in an expression.
fn substitute_in_expr(expr: &Expr, var: &str, replacement: &Expr) -> Expr {
    expr.substitute(var, replacement)
}

/// Brief formatting of an expression for comments.
fn format_expr_brief(expr: &Expr) -> String {
    match expr {
        Expr::IntLiteral(v) => v.to_string(),
        Expr::Variable(name) => name.clone(),
        Expr::NumProcessors => "P".to_string(),
        _ => "expr".to_string(),
    }
}

/// Statistics about processor dispatch transformations.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DispatchStats {
    /// Total number of parallel_for dispatched
    pub total_dispatched: usize,
    /// Number fully unrolled
    pub unrolled_count: usize,
    /// Number converted to sequential loops
    pub loop_count: usize,
    /// Number with zero processors (eliminated)
    pub empty_count: usize,
}

/// Analyze dispatch statistics for a list of statements.
pub fn analyze_dispatch(stmts: &[Stmt]) -> DispatchStats {
    let mut stats = DispatchStats::default();
    for stmt in stmts {
        analyze_dispatch_stmt(stmt, &mut stats);
    }
    stats
}

fn analyze_dispatch_stmt(stmt: &Stmt, stats: &mut DispatchStats) {
    match stmt {
        Stmt::ParallelFor { num_procs, body, .. } => {
            stats.total_dispatched += 1;
            if let Some(p) = num_procs.eval_const_int() {
                if p == 0 {
                    stats.empty_count += 1;
                } else if p <= 32 {
                    stats.unrolled_count += 1;
                } else {
                    stats.loop_count += 1;
                }
            } else {
                stats.loop_count += 1;
            }
            for s in body {
                analyze_dispatch_stmt(s, stats);
            }
        }
        Stmt::If { then_body, else_body, .. } => {
            for s in then_body { analyze_dispatch_stmt(s, stats); }
            for s in else_body { analyze_dispatch_stmt(s, stats); }
        }
        Stmt::SeqFor { body, .. } | Stmt::While { body, .. } => {
            for s in body { analyze_dispatch_stmt(s, stats); }
        }
        Stmt::Block(inner) => {
            for s in inner { analyze_dispatch_stmt(s, stats); }
        }
        _ => {}
    }
}

/// Partially unroll a parallel_for by a given factor.
///
/// Instead of full unrolling, generates a sequential loop with `factor` copies
/// of the body in each iteration. The loop runs over `num_procs / factor`
/// iterations, and a remainder loop handles the leftover.
pub fn partial_unroll(
    proc_var: &str,
    num_procs: &Expr,
    body: &[Stmt],
    factor: usize,
) -> Vec<Stmt> {
    if factor <= 1 {
        return vec![Stmt::SeqFor {
            var: proc_var.to_string(),
            start: Expr::IntLiteral(0),
            end: num_procs.clone(),
            step: None,
            body: body.to_vec(),
        }];
    }

    let factor_expr = Expr::IntLiteral(factor as i64);
    // Main loop: iterate in steps of `factor`
    let main_end = Expr::binop(
        BinOp::Mul,
        Expr::binop(BinOp::Div, num_procs.clone(), factor_expr.clone()),
        factor_expr.clone(),
    );

    let base_var = format!("{}_base", proc_var);
    let mut unrolled_body = Vec::new();
    for k in 0..factor {
        let pid_expr = if k == 0 {
            Expr::var(&base_var)
        } else {
            Expr::binop(BinOp::Add, Expr::var(&base_var), Expr::IntLiteral(k as i64))
        };
        for stmt in body {
            unrolled_body.push(substitute_in_stmt(stmt, proc_var, &pid_expr));
        }
    }

    let main_loop = Stmt::SeqFor {
        var: base_var.clone(),
        start: Expr::IntLiteral(0),
        end: main_end.clone(),
        step: Some(factor_expr),
        body: unrolled_body,
    };

    // Remainder loop
    let remainder_loop = Stmt::SeqFor {
        var: proc_var.to_string(),
        start: main_end,
        end: num_procs.clone(),
        step: None,
        body: body.to_vec(),
    };

    vec![
        Stmt::Comment(format!(
            "partial unroll parallel_for {} by factor {}",
            proc_var, factor
        )),
        main_loop,
        Stmt::Comment("remainder loop".to_string()),
        remainder_loop,
    ]
}

/// Strip mine a parallel_for: partition into strips of a given size.
///
/// Generates a two-level loop nest: outer loop over strips, inner loop
/// within each strip. This can improve cache locality.
pub fn strip_mine(
    proc_var: &str,
    num_procs: &Expr,
    body: &[Stmt],
    strip_size: usize,
) -> Vec<Stmt> {
    let strip_var = format!("{}_strip", proc_var);
    let strip_expr = Expr::IntLiteral(strip_size as i64);

    let inner_end = Expr::FunctionCall(
        "min".to_string(),
        vec![
            Expr::binop(BinOp::Add, Expr::var(&strip_var), strip_expr.clone()),
            num_procs.clone(),
        ],
    );

    let inner_loop = Stmt::SeqFor {
        var: proc_var.to_string(),
        start: Expr::var(&strip_var),
        end: inner_end,
        step: None,
        body: body.to_vec(),
    };

    vec![
        Stmt::Comment(format!(
            "strip-mined parallel_for {} with strip_size={}",
            proc_var, strip_size
        )),
        Stmt::SeqFor {
            var: strip_var,
            start: Expr::IntLiteral(0),
            end: num_procs.clone(),
            step: Some(strip_expr),
            body: vec![inner_loop],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dispatch() -> ProcessorDispatch {
        ProcessorDispatch::with_default_config()
    }

    fn dispatch_threshold(t: usize) -> ProcessorDispatch {
        ProcessorDispatch::new(DispatchConfig::new(t))
    }

    #[test]
    fn test_unroll_small() {
        let stmts = vec![Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::int(3),
            body: vec![Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::var("pid"),
                value: Expr::int(1),
            }],
        }];
        let result = dispatch().transform(&stmts);
        // Should produce: comment + (comment + write)*3 + comment = 3*2+2 = 8
        // Count SharedWrite statements
        let writes: Vec<_> = result.iter().filter(|s| matches!(s, Stmt::SharedWrite { .. })).collect();
        assert_eq!(writes.len(), 3);

        // Verify processor ID was substituted
        match &writes[0] {
            Stmt::SharedWrite { index, .. } => assert_eq!(*index, Expr::IntLiteral(0)),
            _ => unreachable!(),
        }
        match &writes[1] {
            Stmt::SharedWrite { index, .. } => assert_eq!(*index, Expr::IntLiteral(1)),
            _ => unreachable!(),
        }
        match &writes[2] {
            Stmt::SharedWrite { index, .. } => assert_eq!(*index, Expr::IntLiteral(2)),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_sequential_loop_large() {
        let stmts = vec![Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::int(100),
            body: vec![Stmt::Assign(
                "x".to_string(),
                Expr::var("pid"),
            )],
        }];
        let result = dispatch().transform(&stmts);
        // Should emit a sequential for loop (100 > 32 threshold)
        let has_seq_for = result.iter().any(|s| matches!(s, Stmt::SeqFor { .. }));
        assert!(has_seq_for);
    }

    #[test]
    fn test_threshold_boundary() {
        let body = vec![Stmt::Assign("x".to_string(), Expr::var("pid"))];

        // At threshold: should unroll
        let stmts = vec![Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::int(4),
            body: body.clone(),
        }];
        let result = dispatch_threshold(4).transform(&stmts);
        assert!(!result.iter().any(|s| matches!(s, Stmt::SeqFor { .. })));
        let assigns: Vec<_> = result.iter().filter(|s| matches!(s, Stmt::Assign(..))).collect();
        assert_eq!(assigns.len(), 4);

        // Above threshold: should emit loop
        let stmts = vec![Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::int(5),
            body: body,
        }];
        let result = dispatch_threshold(4).transform(&stmts);
        assert!(result.iter().any(|s| matches!(s, Stmt::SeqFor { .. })));
    }

    #[test]
    fn test_zero_processors() {
        let stmts = vec![Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::int(0),
            body: vec![Stmt::Assign("x".to_string(), Expr::int(1))],
        }];
        let result = dispatch().transform(&stmts);
        // Should just produce a comment
        assert!(result.iter().all(|s| matches!(s, Stmt::Comment(_))));
    }

    #[test]
    fn test_non_constant_procs() {
        let stmts = vec![Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::var("P"),
            body: vec![Stmt::Assign("x".to_string(), Expr::var("pid"))],
        }];
        let result = dispatch().transform(&stmts);
        // Non-constant num_procs: should emit a sequential loop
        assert!(result.iter().any(|s| matches!(s, Stmt::SeqFor { .. })));
    }

    #[test]
    fn test_nested_parallel_for() {
        let stmts = vec![Stmt::ParallelFor {
            proc_var: "i".to_string(),
            num_procs: Expr::int(2),
            body: vec![Stmt::ParallelFor {
                proc_var: "j".to_string(),
                num_procs: Expr::int(2),
                body: vec![Stmt::SharedWrite {
                    memory: Expr::var("A"),
                    index: Expr::binop(
                        BinOp::Add,
                        Expr::binop(BinOp::Mul, Expr::var("i"), Expr::int(2)),
                        Expr::var("j"),
                    ),
                    value: Expr::int(1),
                }],
            }],
        }];
        let result = dispatch().transform(&stmts);
        // Both should be unrolled: 2*2=4 writes
        let writes: Vec<_> = result.iter().filter(|s| matches!(s, Stmt::SharedWrite { .. })).collect();
        assert_eq!(writes.len(), 4);
    }

    #[test]
    fn test_substitution_in_if() {
        let stmts = vec![Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::int(2),
            body: vec![Stmt::If {
                condition: Expr::binop(BinOp::Lt, Expr::var("pid"), Expr::int(1)),
                then_body: vec![Stmt::Assign("x".to_string(), Expr::var("pid"))],
                else_body: vec![],
            }],
        }];
        let result = dispatch().transform(&stmts);
        // The condition should have pid substituted
        let ifs: Vec<_> = result.iter().filter(|s| matches!(s, Stmt::If { .. })).collect();
        assert_eq!(ifs.len(), 2);
    }

    #[test]
    fn test_preserves_non_parallel_stmts() {
        let stmts = vec![
            Stmt::Assign("x".to_string(), Expr::int(1)),
            Stmt::ParallelFor {
                proc_var: "pid".to_string(),
                num_procs: Expr::int(2),
                body: vec![Stmt::Assign("y".to_string(), Expr::var("pid"))],
            },
            Stmt::Assign("z".to_string(), Expr::int(3)),
        ];
        let result = dispatch().transform(&stmts);
        // First and last assigns should be preserved
        match &result[0] {
            Stmt::Assign(name, _) => assert_eq!(name, "x"),
            _ => panic!("Expected assign"),
        }
        match result.last().unwrap() {
            Stmt::Assign(name, _) => assert_eq!(name, "z"),
            _ => panic!("Expected assign"),
        }
    }

    #[test]
    fn test_dispatch_config_default() {
        let config = DispatchConfig::default();
        assert_eq!(config.unroll_threshold, 32);
    }

    #[test]
    fn test_single_processor_unroll() {
        let stmts = vec![Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::int(1),
            body: vec![Stmt::Assign(
                "x".to_string(),
                Expr::var("pid"),
            )],
        }];
        let result = dispatch().transform(&stmts);
        // Single processor: should produce one assignment with pid=0
        let assigns: Vec<_> = result.iter().filter(|s| matches!(s, Stmt::Assign(..))).collect();
        assert_eq!(assigns.len(), 1);
        match &assigns[0] {
            Stmt::Assign(_, Expr::IntLiteral(0)) => {}
            other => panic!("Expected x = 0, got {:?}", other),
        }
    }

    #[test]
    fn test_substitution_respects_shadowing() {
        // Inner parallel_for rebinds the same variable
        let stmts = vec![Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::int(2),
            body: vec![Stmt::ParallelFor {
                proc_var: "pid".to_string(),
                num_procs: Expr::int(3),
                body: vec![Stmt::Assign("x".to_string(), Expr::var("pid"))],
            }],
        }];
        let result = dispatch().transform(&stmts);
        // The inner parallel_for should be independently unrolled
        let assigns: Vec<_> = result.iter().filter(|s| matches!(s, Stmt::Assign(..))).collect();
        // Outer has 2 procs, inner has 3, but inner rebinds pid
        // So we get 2 * 3 = 6 assigns
        assert_eq!(assigns.len(), 6);
    }

    #[test]
    fn test_analyze_dispatch() {
        let stmts = vec![
            Stmt::ParallelFor {
                proc_var: "i".to_string(),
                num_procs: Expr::int(4),
                body: vec![Stmt::Assign("x".to_string(), Expr::int(1))],
            },
            Stmt::ParallelFor {
                proc_var: "j".to_string(),
                num_procs: Expr::int(100),
                body: vec![Stmt::Assign("y".to_string(), Expr::int(2))],
            },
            Stmt::ParallelFor {
                proc_var: "k".to_string(),
                num_procs: Expr::int(0),
                body: vec![Stmt::Assign("z".to_string(), Expr::int(3))],
            },
        ];
        let stats = analyze_dispatch(&stmts);
        assert_eq!(stats.total_dispatched, 3);
        assert_eq!(stats.unrolled_count, 1); // 4 <= 32
        assert_eq!(stats.loop_count, 1); // 100 > 32
        assert_eq!(stats.empty_count, 1); // 0
    }

    #[test]
    fn test_partial_unroll() {
        let body = vec![Stmt::SharedWrite {
            memory: Expr::var("A"),
            index: Expr::var("pid"),
            value: Expr::int(1),
        }];
        let result = partial_unroll("pid", &Expr::int(8), &body, 2);
        // Should have comment, main loop, comment, remainder loop
        assert!(result.len() >= 3);
        // Main loop should have step of 2
        let has_main_loop = result.iter().any(|s| matches!(s, Stmt::SeqFor { step: Some(_), .. }));
        assert!(has_main_loop);
    }

    #[test]
    fn test_partial_unroll_factor_1() {
        let body = vec![Stmt::Assign("x".to_string(), Expr::var("pid"))];
        let result = partial_unroll("pid", &Expr::int(10), &body, 1);
        assert_eq!(result.len(), 1);
        assert!(matches!(&result[0], Stmt::SeqFor { .. }));
    }

    #[test]
    fn test_strip_mine() {
        let body = vec![Stmt::SharedWrite {
            memory: Expr::var("A"),
            index: Expr::var("pid"),
            value: Expr::int(1),
        }];
        let result = strip_mine("pid", &Expr::int(16), &body, 4);
        // Should have comment + outer SeqFor
        assert!(result.len() >= 2);
        let has_outer = result.iter().any(|s| matches!(s, Stmt::SeqFor { .. }));
        assert!(has_outer);
        // Outer loop should have inner loop
        if let Some(Stmt::SeqFor { body: outer_body, step: Some(step), .. }) = result.iter().find(|s| matches!(s, Stmt::SeqFor { step: Some(_), .. })) {
            assert_eq!(*step, Expr::IntLiteral(4));
            assert!(outer_body.iter().any(|s| matches!(s, Stmt::SeqFor { .. })));
        }
    }

    #[test]
    fn test_dispatch_stats_default() {
        let stats = DispatchStats::default();
        assert_eq!(stats.total_dispatched, 0);
        assert_eq!(stats.unrolled_count, 0);
        assert_eq!(stats.loop_count, 0);
        assert_eq!(stats.empty_count, 0);
    }

    #[test]
    fn test_analyze_dispatch_nested() {
        let stmts = vec![Stmt::ParallelFor {
            proc_var: "i".to_string(),
            num_procs: Expr::int(2),
            body: vec![Stmt::ParallelFor {
                proc_var: "j".to_string(),
                num_procs: Expr::int(3),
                body: vec![Stmt::Assign("x".to_string(), Expr::int(1))],
            }],
        }];
        let stats = analyze_dispatch(&stmts);
        assert_eq!(stats.total_dispatched, 2);
        assert_eq!(stats.unrolled_count, 2);
    }
}
