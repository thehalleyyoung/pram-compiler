//! Compositional verification: verify that sequential composition of
//! transformation passes preserves end-to-end correctness.
//!
//! Addresses the critique: "individual transformation passes are tested but
//! their sequential composition is not verified for preserving end-to-end
//! correctness."

use crate::pram_ir::ast::{Expr, MemoryModel, Stmt, BinOp};
use super::partial_eval::PartialEvaluator;
use super::model_arbitration::ModelArbitrationPass;
use super::processor_dispatch::{DispatchConfig, ProcessorDispatch};
use super::work_preservation::{WorkCounter, WorkCount};
use crate::failure_analysis::semantic_preservation::verify_semantic_preservation;

/// Result of compositional verification.
#[derive(Debug, Clone)]
pub struct CompositionVerification {
    /// Whether all compositions preserved semantics.
    pub all_preserved: bool,
    /// Number of compositions tested.
    pub compositions_tested: usize,
    /// Results for each composition.
    pub results: Vec<CompositionResult>,
}

/// Result of a single pass-composition test.
#[derive(Debug, Clone)]
pub struct CompositionResult {
    /// Description of the composition (e.g., "dispatch → arbitration → partial_eval").
    pub composition: String,
    /// Whether semantics were preserved.
    pub preserved: bool,
    /// Number of test inputs checked.
    pub inputs_checked: usize,
    /// Number of mismatches found.
    pub mismatches: usize,
}

/// Verify that composing passes in sequence preserves semantics.
///
/// Tests multiple orderings and subsets of passes against a reference
/// program, checking that the final shared memory state matches.
pub fn verify_pass_composition(
    original: &[Stmt],
    model: MemoryModel,
    regions: &[(String, usize)],
    num_trials: usize,
) -> CompositionVerification {
    let mut results = Vec::new();

    // Composition 1: dispatch → partial_eval
    {
        let dispatch = ProcessorDispatch::new(DispatchConfig::new(32));
        let pe = PartialEvaluator::new();
        let after_dispatch = dispatch.transform(original);
        let after_pe = pe.evaluate(&after_dispatch);
        let sem = verify_semantic_preservation(
            original, &after_pe, model, regions, num_trials,
        );
        results.push(CompositionResult {
            composition: "dispatch → partial_eval".into(),
            preserved: sem.preserved,
            inputs_checked: sem.inputs_checked,
            mismatches: sem.mismatches.len(),
        });
    }

    // Composition 2: arbitration → partial_eval
    {
        let arb = ModelArbitrationPass::new(model);
        let pe = PartialEvaluator::new();
        let after_arb = arb.transform(original);
        let after_pe = pe.evaluate(&after_arb);
        let sem = verify_semantic_preservation(
            original, &after_pe, model, regions, num_trials,
        );
        results.push(CompositionResult {
            composition: "arbitration → partial_eval".into(),
            preserved: sem.preserved,
            inputs_checked: sem.inputs_checked,
            mismatches: sem.mismatches.len(),
        });
    }

    // Composition 3: dispatch → arbitration → partial_eval (full pipeline)
    {
        let dispatch = ProcessorDispatch::new(DispatchConfig::new(32));
        let arb = ModelArbitrationPass::new(model);
        let pe = PartialEvaluator::new();
        let step1 = dispatch.transform(original);
        let step2 = arb.transform(&step1);
        let step3 = pe.evaluate(&step2);
        let sem = verify_semantic_preservation(
            original, &step3, model, regions, num_trials,
        );
        results.push(CompositionResult {
            composition: "dispatch → arbitration → partial_eval".into(),
            preserved: sem.preserved,
            inputs_checked: sem.inputs_checked,
            mismatches: sem.mismatches.len(),
        });
    }

    // Composition 4: partial_eval → arbitration (reversed order)
    {
        let arb = ModelArbitrationPass::new(model);
        let pe = PartialEvaluator::new();
        let step1 = pe.evaluate(original);
        let step2 = arb.transform(&step1);
        let sem = verify_semantic_preservation(
            original, &step2, model, regions, num_trials,
        );
        results.push(CompositionResult {
            composition: "partial_eval → arbitration".into(),
            preserved: sem.preserved,
            inputs_checked: sem.inputs_checked,
            mismatches: sem.mismatches.len(),
        });
    }

    let all_preserved = results.iter().all(|r| r.preserved);
    CompositionVerification {
        all_preserved,
        compositions_tested: results.len(),
        results,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pram_ir::ast::{Expr, Stmt, BinOp, MemoryModel};

    fn make_simple_program() -> Vec<Stmt> {
        vec![Stmt::SharedWrite {
            memory: Expr::Variable("A".into()),
            index: Expr::IntLiteral(0),
            value: Expr::IntLiteral(42),
        }]
    }

    fn make_multi_write_program() -> Vec<Stmt> {
        vec![
            Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(0),
                value: Expr::IntLiteral(10),
            },
            Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(1),
                value: Expr::BinOp(
                    BinOp::Add,
                    Box::new(Expr::IntLiteral(20)),
                    Box::new(Expr::IntLiteral(5)),
                ),
            },
        ]
    }

    fn make_loop_program() -> Vec<Stmt> {
        vec![Stmt::SeqFor {
            var: "i".into(),
            start: Expr::IntLiteral(0),
            end: Expr::IntLiteral(4),
            step: None,
            body: vec![Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::Variable("i".into()),
                value: Expr::Variable("i".into()),
            }],
        }]
    }

    #[test]
    fn test_composition_simple_erew() {
        let program = make_simple_program();
        let result = verify_pass_composition(
            &program, MemoryModel::EREW,
            &[("A".into(), 4)], 10,
        );
        assert_eq!(result.compositions_tested, 4);
        for r in &result.results {
            assert!(r.preserved, "Failed: {} ({} mismatches)", r.composition, r.mismatches);
        }
    }

    #[test]
    fn test_composition_multi_write() {
        let program = make_multi_write_program();
        let result = verify_pass_composition(
            &program, MemoryModel::EREW,
            &[("A".into(), 4)], 10,
        );
        assert!(result.compositions_tested >= 4);
        for r in &result.results {
            assert!(r.preserved, "Failed: {} ({} mismatches)", r.composition, r.mismatches);
        }
    }

    #[test]
    fn test_composition_loop() {
        let program = make_loop_program();
        let result = verify_pass_composition(
            &program, MemoryModel::EREW,
            &[("A".into(), 8)], 10,
        );
        for r in &result.results {
            assert!(r.preserved, "Failed: {} ({} mismatches)", r.composition, r.mismatches);
        }
    }

    #[test]
    fn test_composition_crew() {
        let program = make_simple_program();
        let result = verify_pass_composition(
            &program, MemoryModel::CREW,
            &[("A".into(), 4)], 10,
        );
        for r in &result.results {
            assert!(r.preserved, "Failed: {} ({} mismatches)", r.composition, r.mismatches);
        }
    }

    #[test]
    fn test_composition_crcw_priority() {
        let program = make_simple_program();
        let result = verify_pass_composition(
            &program, MemoryModel::CRCWPriority,
            &[("A".into(), 4)], 10,
        );
        for r in &result.results {
            assert!(r.preserved, "Failed: {} ({} mismatches)", r.composition, r.mismatches);
        }
    }
}
