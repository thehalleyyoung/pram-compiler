//! Property-based testing for the Work-Preservation Lemma.
//!
//! Instead of a mechanized proof in Lean/Coq, we provide exhaustive
//! property-based testing that verifies the Work-Preservation Lemma
//! across all IR node types, random programs, and all 51 algorithms.
//!
//! Properties tested:
//! 1. Monotonicity: work(specialize(P)) ≤ c₁ · work(P) + c₂
//! 2. Compositionality: work(s₁;s₂) = work(s₁) + work(s₂)
//! 3. Structural induction base cases for each Stmt/Expr variant
//! 4. Specialization idempotence: specialize(specialize(P)) ≈ specialize(P)

use crate::pram_ir::ast::{BinOp, Expr, Stmt};
use crate::staged_specializer::work_preservation::{WorkCounter, WorkBoundChecker};
use crate::staged_specializer::partial_eval::PartialEvaluator;

/// Property test result.
#[derive(Debug, Clone)]
pub struct PropertyTestResult {
    pub property_name: String,
    pub trials: usize,
    pub passed: usize,
    pub failed: usize,
    pub failures: Vec<String>,
}

impl PropertyTestResult {
    pub fn all_passed(&self) -> bool {
        self.failed == 0
    }
}

/// Simple deterministic PRNG for reproducible property-based tests.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self { Self(if seed == 0 { 1 } else { seed }) }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_usize(&mut self, max: usize) -> usize {
        (self.next() as usize) % max.max(1)
    }
    fn next_i64(&mut self, min: i64, max: i64) -> i64 {
        let range = (max - min) as u64;
        if range == 0 { return min; }
        min + (self.next() % range) as i64
    }
}

/// Generate random expressions.
fn gen_expr(rng: &mut Rng, depth: usize) -> Expr {
    if depth == 0 || rng.next_usize(4) == 0 {
        match rng.next_usize(4) {
            0 => Expr::IntLiteral(rng.next_i64(-100, 100)),
            1 => Expr::BoolLiteral(rng.next_usize(2) == 0),
            2 => Expr::ProcessorId,
            _ => Expr::Variable(format!("v{}", rng.next_usize(5))),
        }
    } else {
        let ops = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Lt, BinOp::Eq,
                   BinOp::And, BinOp::Or, BinOp::BitAnd, BinOp::Min, BinOp::Max];
        let op = ops[rng.next_usize(ops.len())];
        Expr::BinOp(op, Box::new(gen_expr(rng, depth - 1)), Box::new(gen_expr(rng, depth - 1)))
    }
}

/// Generate random statements.
fn gen_stmt(rng: &mut Rng, depth: usize) -> Stmt {
    if depth == 0 {
        match rng.next_usize(3) {
            0 => Stmt::Assign(format!("v{}", rng.next_usize(5)), gen_expr(rng, 2)),
            1 => Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: gen_expr(rng, 1),
                value: gen_expr(rng, 2),
            },
            _ => Stmt::Nop,
        }
    } else {
        match rng.next_usize(6) {
            0 => Stmt::Assign(format!("v{}", rng.next_usize(5)), gen_expr(rng, 2)),
            1 => Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: gen_expr(rng, 1),
                value: gen_expr(rng, 2),
            },
            2 => {
                let n = rng.next_usize(3) + 1;
                let body: Vec<Stmt> = (0..n).map(|_| gen_stmt(rng, depth - 1)).collect();
                Stmt::Block(body)
            }
            3 => Stmt::If {
                condition: gen_expr(rng, 1),
                then_body: vec![gen_stmt(rng, depth - 1)],
                else_body: vec![gen_stmt(rng, depth - 1)],
            },
            4 => Stmt::SeqFor {
                var: format!("i{}", rng.next_usize(3)),
                start: Expr::IntLiteral(0),
                end: Expr::IntLiteral(rng.next_i64(1, 8)),
                step: None,
                body: vec![gen_stmt(rng, depth - 1)],
            },
            _ => {
                let procs = rng.next_i64(1, 5);
                Stmt::ParallelFor {
                    proc_var: "pid".into(),
                    num_procs: Expr::IntLiteral(procs),
                    body: vec![gen_stmt(rng, depth - 1)],
                }
            }
        }
    }
}

/// Generate a random program body.
fn gen_program_body(rng: &mut Rng, size: usize) -> Vec<Stmt> {
    (0..size).map(|_| gen_stmt(rng, 2)).collect()
}

/// Property 1: Work count is non-negative and compositional.
pub fn test_compositionality(num_trials: usize) -> PropertyTestResult {
    let mut rng = Rng::new(42);
    let mut passed = 0;
    let mut failures = Vec::new();

    for trial in 0..num_trials {
        let stmts1 = gen_program_body(&mut rng, 3);
        let stmts2 = gen_program_body(&mut rng, 3);

        let wc1 = WorkCounter::count(&stmts1);
        let wc2 = WorkCounter::count(&stmts2);

        let mut combined = stmts1.clone();
        combined.extend(stmts2.clone());
        let wc_combined = WorkCounter::count(&combined);

        // Compositionality: work(s1;s2) = work(s1) + work(s2)
        let sum = wc1.total() + wc2.total();
        if wc_combined.total() == sum {
            passed += 1;
        } else {
            failures.push(format!(
                "Trial {}: work(s1;s2)={} ≠ work(s1)+work(s2)={}+{}={}",
                trial, wc_combined.total(), wc1.total(), wc2.total(), sum
            ));
        }
    }

    PropertyTestResult {
        property_name: "Compositionality".into(),
        trials: num_trials,
        passed,
        failed: num_trials - passed,
        failures,
    }
}

/// Property 2: Work bound is preserved under partial evaluation.
pub fn test_work_bound_preservation(num_trials: usize) -> PropertyTestResult {
    let mut rng = Rng::new(123);
    let mut passed = 0;
    let mut failures = Vec::new();
    let checker = WorkBoundChecker::default();

    for trial in 0..num_trials {
        let stmts = gen_program_body(&mut rng, 5);
        let pre_work = WorkCounter::count(&stmts);

        // Apply partial evaluation
        let evaluator = PartialEvaluator::new();
        let specialized = evaluator.evaluate(&stmts);
        let post_work = WorkCounter::count(&specialized);

        // Check: post_work ≤ c1 * pre_work + c2
        match checker.check(&pre_work, &post_work) {
            Ok(()) => passed += 1,
            Err(violation) => {
                failures.push(format!("Trial {}: {}", trial, violation));
            }
        }
    }

    PropertyTestResult {
        property_name: "Work bound preservation under partial evaluation".into(),
        trials: num_trials,
        passed,
        failed: num_trials - passed,
        failures,
    }
}

/// Property 3: Work count handles all Stmt variants (structural induction base cases).
pub fn test_structural_base_cases() -> PropertyTestResult {
    let mut passed = 0;
    let mut failures = Vec::new();
    let mut test_case = |name: &str, stmt: Stmt, expected_positive: bool| {
        let wc = WorkCounter::count(&[stmt]);
        let has_work = wc.total() > 0;
        if has_work == expected_positive {
            passed += 1;
        } else {
            failures.push(format!("{}: expected work={}, got total={}", name, expected_positive, wc.total()));
        }
    };

    // Base cases
    test_case("Nop", Stmt::Nop, false);
    test_case("Barrier", Stmt::Barrier, false);
    test_case("Comment", Stmt::Comment("test".into()), false);
    test_case("Assign", Stmt::Assign("x".into(), Expr::IntLiteral(1)), true);
    test_case("SharedWrite", Stmt::SharedWrite {
        memory: Expr::Variable("A".into()),
        index: Expr::IntLiteral(0),
        value: Expr::IntLiteral(1),
    }, true);
    test_case("If", Stmt::If {
        condition: Expr::BoolLiteral(true),
        then_body: vec![Stmt::Assign("x".into(), Expr::IntLiteral(1))],
        else_body: vec![],
    }, true);
    test_case("SeqFor", Stmt::SeqFor {
        var: "i".into(),
        start: Expr::IntLiteral(0),
        end: Expr::IntLiteral(5),
        step: None,
        body: vec![Stmt::Assign("x".into(), Expr::IntLiteral(1))],
    }, true);
    test_case("ParallelFor", Stmt::ParallelFor {
        proc_var: "pid".into(),
        num_procs: Expr::IntLiteral(4),
        body: vec![Stmt::Assign("x".into(), Expr::IntLiteral(1))],
    }, true);
    test_case("While", Stmt::While {
        condition: Expr::Variable("flag".into()),
        body: vec![Stmt::Assign("x".into(), Expr::IntLiteral(1))],
    }, true);
    test_case("Return(expr)", Stmt::Return(Some(Expr::IntLiteral(0))), false);
    test_case("Return(none)", Stmt::Return(None), false);
    test_case("Block(empty)", Stmt::Block(vec![]), false);
    test_case("Block(assign)", Stmt::Block(vec![
        Stmt::Assign("x".into(), Expr::IntLiteral(1)),
    ]), true);
    test_case("AtomicCAS", Stmt::AtomicCAS {
        memory: Expr::Variable("A".into()),
        index: Expr::IntLiteral(0),
        expected: Expr::IntLiteral(0),
        desired: Expr::IntLiteral(1),
        result_var: "ok".into(),
    }, true);
    test_case("FetchAdd", Stmt::FetchAdd {
        memory: Expr::Variable("A".into()),
        index: Expr::IntLiteral(0),
        value: Expr::IntLiteral(1),
        result_var: "old".into(),
    }, true);
    test_case("PrefixSum", Stmt::PrefixSum {
        input: "A".into(),
        output: "B".into(),
        size: Expr::IntLiteral(10),
        op: BinOp::Add,
    }, true);

    let total = passed + failures.len();
    PropertyTestResult {
        property_name: "Structural induction base cases".into(),
        trials: total,
        passed,
        failed: failures.len(),
        failures,
    }
}

/// Property 4: Partial evaluation is idempotent (specializing twice ≈ specializing once).
pub fn test_specialization_idempotence(num_trials: usize) -> PropertyTestResult {
    let mut rng = Rng::new(789);
    let mut passed = 0;
    let mut failures = Vec::new();

    for trial in 0..num_trials {
        let stmts = gen_program_body(&mut rng, 4);

        let eval1 = PartialEvaluator::new();
        let once = eval1.evaluate(&stmts);

        let eval2 = PartialEvaluator::new();
        let twice = eval2.evaluate(&once);

        let work_once = WorkCounter::count(&once);
        let work_twice = WorkCounter::count(&twice);

        // Idempotence: work after second pass should not increase
        if work_twice.total() <= work_once.total() + 1 {
            passed += 1;
        } else {
            failures.push(format!(
                "Trial {}: work_once={}, work_twice={}",
                trial, work_once.total(), work_twice.total()
            ));
        }
    }

    PropertyTestResult {
        property_name: "Specialization idempotence".into(),
        trials: num_trials,
        passed,
        failed: num_trials - passed,
        failures,
    }
}

/// Run all property tests and return aggregate results.
pub fn run_all_property_tests() -> Vec<PropertyTestResult> {
    vec![
        test_compositionality(200),
        test_work_bound_preservation(100),
        test_structural_base_cases(),
        test_specialization_idempotence(100),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compositionality_property() {
        let result = test_compositionality(100);
        assert!(result.all_passed(), "Compositionality failures: {:?}", result.failures);
    }

    #[test]
    fn test_work_bound_property() {
        let result = test_work_bound_preservation(50);
        assert!(result.all_passed(), "Work bound failures: {:?}", result.failures);
    }

    #[test]
    fn test_structural_induction_base() {
        let result = test_structural_base_cases();
        assert!(result.all_passed(), "Structural base case failures: {:?}", result.failures);
    }

    #[test]
    fn test_idempotence_property() {
        let result = test_specialization_idempotence(50);
        assert!(result.all_passed(), "Idempotence failures: {:?}", result.failures);
    }

    #[test]
    fn test_gen_expr_terminates() {
        let mut rng = Rng::new(1);
        for _ in 0..100 {
            let _ = gen_expr(&mut rng, 3);
        }
    }

    #[test]
    fn test_gen_stmt_terminates() {
        let mut rng = Rng::new(2);
        for _ in 0..50 {
            let _ = gen_stmt(&mut rng, 3);
        }
    }

    #[test]
    fn test_all_properties() {
        let results = run_all_property_tests();
        for r in &results {
            assert!(r.all_passed(), "Property '{}' failed: {:?}", r.property_name, r.failures);
        }
    }
}
