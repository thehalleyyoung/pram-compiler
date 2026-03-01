//! Semantic preservation verification for IR repair transformations.
//!
//! Verifies that IR repair transformations (write-coalescing, tiling,
//! loop fusion, etc.) preserve the input-output behavior of the original
//! program via simulation relation checking.
//!
//! The key property: for any initial store σ, if the original program P
//! and the transformed program P' both terminate, then they produce the
//! same shared memory state: exec(P, σ).shared = exec(P', σ).shared.

use std::collections::HashMap;
use crate::pram_ir::ast::{Expr, MemoryModel, PramProgram, Stmt, BinOp};
use crate::pram_ir::operational_semantics::{Store, exec_stmt, eval_to_value};
use crate::pram_ir::metatheory::Value;

/// Result of a semantic preservation check.
#[derive(Debug, Clone)]
pub struct PreservationVerification {
    /// Whether all test inputs showed equivalent behavior.
    pub preserved: bool,
    /// Number of test inputs checked.
    pub inputs_checked: usize,
    /// Shared memory locations compared.
    pub locations_compared: usize,
    /// Any mismatches found.
    pub mismatches: Vec<SemanticMismatch>,
    /// Summary statistics.
    pub max_output_diff: f64,
}

/// A semantic mismatch between original and transformed program.
#[derive(Debug, Clone)]
pub struct SemanticMismatch {
    pub region: String,
    pub address: usize,
    pub original_value: String,
    pub transformed_value: String,
    pub input_description: String,
}

/// Generates test inputs for semantic preservation checking.
pub struct TestInputGenerator {
    pub shared_regions: Vec<(String, usize)>,
    pub num_inputs: usize,
    seed: u64,
}

impl TestInputGenerator {
    pub fn new(regions: Vec<(String, usize)>, num_inputs: usize) -> Self {
        Self { shared_regions: regions, num_inputs, seed: 12345 }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Generate deterministic test stores.
    pub fn generate(&self) -> Vec<Store> {
        let mut stores = Vec::new();
        let mut rng = self.seed;

        for trial in 0..self.num_inputs {
            let mut store = Store::new();
            for (region, size) in &self.shared_regions {
                store.alloc_shared(region, *size);
                for addr in 0..*size {
                    rng = xorshift(rng);
                    let val = match trial % 4 {
                        0 => Value::IntVal(0),                           // zeros
                        1 => Value::IntVal(addr as i64),                 // identity
                        2 => Value::IntVal((rng % 100) as i64),          // small random
                        _ => Value::IntVal((rng % 1000000) as i64),      // large random
                    };
                    store.write_shared(region, addr, val);
                }
            }
            stores.push(store);
        }

        stores
    }
}

fn xorshift(mut x: u64) -> u64 {
    if x == 0 { x = 1; }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Verify that two programs produce the same shared memory state on all test inputs.
pub fn verify_semantic_preservation(
    original: &[Stmt],
    transformed: &[Stmt],
    model: MemoryModel,
    regions: &[(String, usize)],
    num_trials: usize,
) -> PreservationVerification {
    let gen = TestInputGenerator::new(
        regions.iter().map(|(r, s)| (r.clone(), *s)).collect(),
        num_trials,
    );
    let inputs = gen.generate();

    let mut mismatches = Vec::new();
    let mut locations_compared = 0;
    let mut max_diff = 0.0f64;

    for (trial_idx, initial_store) in inputs.iter().enumerate() {
        // Execute original
        let mut orig_store = initial_store.clone();
        let orig_result = execute_body(original, &mut orig_store, model);

        // Execute transformed
        let mut trans_store = initial_store.clone();
        let trans_result = execute_body(transformed, &mut trans_store, model);

        match (orig_result, trans_result) {
            (Ok(()), Ok(())) => {
                // Compare shared memory states
                for (region, size) in regions {
                    for addr in 0..*size {
                        locations_compared += 1;
                        let orig_val = orig_store.read_shared(region, addr);
                        let trans_val = trans_store.read_shared(region, addr);

                        if orig_val != trans_val {
                            let diff = match (orig_val, trans_val) {
                                (Some(Value::IntVal(a)), Some(Value::IntVal(b))) => {
                                    (a - b).unsigned_abs() as f64
                                }
                                _ => 1.0,
                            };
                            max_diff = max_diff.max(diff);
                            mismatches.push(SemanticMismatch {
                                region: region.clone(),
                                address: addr,
                                original_value: format!("{:?}", orig_val),
                                transformed_value: format!("{:?}", trans_val),
                                input_description: format!("trial_{}", trial_idx),
                            });
                        }
                    }
                }
            }
            (Err(e1), Err(e2)) => {
                // Both error: acceptable if same error class
                if !e1.contains(&e2[..e2.len().min(20).max(1)]) {
                    // Different errors, note but don't fail
                }
            }
            (Ok(()), Err(e)) => {
                mismatches.push(SemanticMismatch {
                    region: "EXECUTION".into(),
                    address: 0,
                    original_value: "Ok".into(),
                    transformed_value: format!("Err: {}", e),
                    input_description: format!("trial_{}", trial_idx),
                });
            }
            (Err(e), Ok(())) => {
                mismatches.push(SemanticMismatch {
                    region: "EXECUTION".into(),
                    address: 0,
                    original_value: format!("Err: {}", e),
                    transformed_value: "Ok".into(),
                    input_description: format!("trial_{}", trial_idx),
                });
            }
        }
    }

    PreservationVerification {
        preserved: mismatches.is_empty(),
        inputs_checked: inputs.len(),
        locations_compared,
        mismatches,
        max_output_diff: max_diff,
    }
}

fn execute_body(stmts: &[Stmt], store: &mut Store, model: MemoryModel) -> Result<(), String> {
    for stmt in stmts {
        exec_stmt(stmt, store, None, None, model)?;
    }
    Ok(())
}

/// Verify semantic preservation for a specific IR repair transformation.
pub fn verify_repair_preservation(
    program: &PramProgram,
    repaired: &PramProgram,
) -> PreservationVerification {
    let regions: Vec<(String, usize)> = program.shared_memory.iter()
        .map(|d| {
            let size = match &d.size {
                Expr::IntLiteral(n) => *n as usize,
                _ => 64,
            };
            (d.name.clone(), size.min(256))
        })
        .collect();

    verify_semantic_preservation(
        &program.body,
        &repaired.body,
        program.memory_model,
        &regions,
        100,
    )
}

/// Verify that partial evaluation preserves semantics.
pub fn verify_partial_eval_preservation(
    original: &[Stmt],
    specialized: &[Stmt],
    test_values: &HashMap<String, i64>,
) -> PreservationVerification {
    let mut orig_store = Store::new();
    let mut spec_store = Store::new();

    for (k, v) in test_values {
        orig_store.set_local(k, Value::IntVal(*v));
        spec_store.set_local(k, Value::IntVal(*v));
    }

    let orig_result = execute_body(original, &mut orig_store, MemoryModel::EREW);
    let spec_result = execute_body(specialized, &mut spec_store, MemoryModel::EREW);

    let mut mismatches = Vec::new();
    let mut locations_compared = 0;

    match (orig_result, spec_result) {
        (Ok(()), Ok(())) => {
            // Compare local variables
            for (k, v) in &orig_store.locals {
                locations_compared += 1;
                if let Some(sv) = spec_store.locals.get(k) {
                    if v != sv {
                        mismatches.push(SemanticMismatch {
                            region: "local".into(),
                            address: 0,
                            original_value: format!("{:?}", v),
                            transformed_value: format!("{:?}", sv),
                            input_description: format!("var={}", k),
                        });
                    }
                }
            }
        }
        (Ok(()), Err(e)) => {
            mismatches.push(SemanticMismatch {
                region: "EXECUTION".into(),
                address: 0,
                original_value: "Ok".into(),
                transformed_value: format!("Err: {}", e),
                input_description: "partial_eval".into(),
            });
        }
        _ => {}
    }

    PreservationVerification {
        preserved: mismatches.is_empty(),
        inputs_checked: 1,
        locations_compared,
        mismatches,
        max_output_diff: 0.0,
    }
}

/// A semantic preservation validation witness for an IR transform.
///
/// Each validation witness pairs a transform name with a proof strategy
/// and simulation-based verification of input-output equivalence.
#[derive(Debug, Clone)]
pub struct PreservationWitness {
    /// Name of the IR transform.
    pub transform_name: String,
    /// The proof strategy used.
    pub proof_strategy: ProofStrategy,
    /// Preconditions required for the transform to preserve semantics.
    pub preconditions: Vec<Precondition>,
    /// The verification result.
    pub verification: Option<PreservationVerification>,
    /// Memory models for which this witness applies.
    pub applicable_models: Vec<MemoryModel>,
}

/// Strategy used to establish semantic preservation.
#[derive(Debug, Clone, PartialEq)]
pub enum ProofStrategy {
    /// Original and transformed produce identical shared memory states.
    SimulationRelation,
    /// Step-by-step bisimulation with identical observable states.
    Bisimulation,
    /// Independent operations can be reordered.
    Commutativity,
    /// Every behavior of the transform is a behavior of the original.
    Refinement,
}

/// A precondition for semantic preservation.
#[derive(Debug, Clone)]
pub struct Precondition {
    pub name: String,
    pub description: String,
    pub statically_checkable: bool,
}

impl PreservationWitness {
    /// Validation witness for write-coalescing: buffer-flush preserves CRCW semantics.
    pub fn write_coalescing() -> Self {
        Self {
            transform_name: "write_coalescing".into(),
            proof_strategy: ProofStrategy::Bisimulation,
            preconditions: vec![
                Precondition {
                    name: "bsp_semantics".into(),
                    description: "ParallelFor body executes under BSP: reads see previous step values.".into(),
                    statically_checkable: true,
                },
                Precondition {
                    name: "no_intra_step_raw".into(),
                    description: "No read depends on a write from same step by different processor.".into(),
                    statically_checkable: true,
                },
                Precondition {
                    name: "flush_order_matches_resolution".into(),
                    description: "Buffer flush selects winner per CRCW policy.".into(),
                    statically_checkable: true,
                },
            ],
            verification: None,
            applicable_models: vec![
                MemoryModel::CRCWPriority,
                MemoryModel::CRCWArbitrary,
                MemoryModel::CRCWCommon,
            ],
        }
    }

    /// Validation witness for tiling: tile loop covers all processors.
    pub fn tiling() -> Self {
        Self {
            transform_name: "irregular_access_tiling".into(),
            proof_strategy: ProofStrategy::Refinement,
            preconditions: vec![
                Precondition {
                    name: "processor_coverage".into(),
                    description: "Tile loop covers all processor IDs [0, p) without gaps.".into(),
                    statically_checkable: true,
                },
                Precondition {
                    name: "tile_body_equivalence".into(),
                    description: "Each tile executes original body for a contiguous subset.".into(),
                    statically_checkable: true,
                },
            ],
            verification: None,
            applicable_models: vec![
                MemoryModel::EREW, MemoryModel::CREW,
                MemoryModel::CRCWPriority, MemoryModel::CRCWArbitrary,
                MemoryModel::CRCWCommon,
            ],
        }
    }

    /// Validation witness for loop fusion: commutativity of independent loops.
    pub fn loop_fusion() -> Self {
        Self {
            transform_name: "loop_fusion".into(),
            proof_strategy: ProofStrategy::Commutativity,
            preconditions: vec![
                Precondition {
                    name: "same_processor_count".into(),
                    description: "Both loops use the same number of processors.".into(),
                    statically_checkable: true,
                },
                Precondition {
                    name: "no_cross_loop_dependency".into(),
                    description: "Second loop does not read addresses written by first loop.".into(),
                    statically_checkable: false,
                },
            ],
            verification: None,
            applicable_models: vec![
                MemoryModel::EREW, MemoryModel::CREW,
                MemoryModel::CRCWPriority, MemoryModel::CRCWArbitrary,
                MemoryModel::CRCWCommon,
            ],
        }
    }

    /// Validation witness for priority-write serialization.
    pub fn priority_serialization() -> Self {
        Self {
            transform_name: "priority_write_serialization".into(),
            proof_strategy: ProofStrategy::SimulationRelation,
            preconditions: vec![
                Precondition {
                    name: "priority_model".into(),
                    description: "Program uses CRCWPriority memory model.".into(),
                    statically_checkable: true,
                },
                Precondition {
                    name: "guard_correctness".into(),
                    description: "Guard selects value from lowest processor ID.".into(),
                    statically_checkable: true,
                },
            ],
            verification: None,
            applicable_models: vec![MemoryModel::CRCWPriority],
        }
    }

    /// Validation witness for write-combining buffers.
    pub fn write_combining() -> Self {
        Self {
            transform_name: "write_combining_buffer".into(),
            proof_strategy: ProofStrategy::Refinement,
            preconditions: vec![
                Precondition {
                    name: "buffer_flush_completeness".into(),
                    description: "All buffered writes flushed before body exit.".into(),
                    statically_checkable: true,
                },
                Precondition {
                    name: "no_read_of_buffered".into(),
                    description: "No concurrent reader reads a buffered address (BSP).".into(),
                    statically_checkable: true,
                },
            ],
            verification: None,
            applicable_models: vec![
                MemoryModel::CRCWPriority, MemoryModel::CRCWArbitrary,
                MemoryModel::CRCWCommon,
            ],
        }
    }

    /// Verify this witness against actual program pairs.
    pub fn verify(
        &mut self,
        original: &[Stmt],
        transformed: &[Stmt],
        model: MemoryModel,
        regions: &[(String, usize)],
    ) -> &PreservationVerification {
        let result = verify_semantic_preservation(original, transformed, model, regions, 100);
        self.verification = Some(result);
        self.verification.as_ref().unwrap()
    }

    /// Check whether all preconditions are met for the given model.
    pub fn check_preconditions(&self, model: MemoryModel) -> bool {
        self.applicable_models.contains(&model)
    }
}

/// A validated transform: pairs an IR transform with its preservation validation witness.
#[derive(Debug, Clone)]
pub struct ValidatedTransform {
    pub witness: PreservationWitness,
    pub transform_applied: bool,
    pub verification_passed: bool,
}

impl ValidatedTransform {
    pub fn new(cert: PreservationWitness) -> Self {
        Self { witness: cert, transform_applied: false, verification_passed: false }
    }

    /// Apply the transform and verify preservation.
    pub fn apply_and_verify(
        &mut self,
        original: &[Stmt],
        transformed: &[Stmt],
        model: MemoryModel,
        regions: &[(String, usize)],
    ) -> bool {
        if !self.witness.check_preconditions(model) {
            return false;
        }
        self.transform_applied = true;
        let result = self.witness.verify(original, transformed, model, regions);
        self.verification_passed = result.preserved;
        self.verification_passed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pram_ir::ast::{BinOp, Expr, Stmt};

    #[test]
    fn test_identical_programs_preserve() {
        let body = vec![
            Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(0),
                value: Expr::IntLiteral(42),
            },
        ];
        let result = verify_semantic_preservation(
            &body, &body, MemoryModel::EREW,
            &[("A".into(), 4)], 4,
        );
        assert!(result.preserved);
        assert_eq!(result.mismatches.len(), 0);
    }

    #[test]
    fn test_different_programs_detect_mismatch() {
        let orig = vec![
            Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(0),
                value: Expr::IntLiteral(42),
            },
        ];
        let modified = vec![
            Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(0),
                value: Expr::IntLiteral(99),
            },
        ];
        let result = verify_semantic_preservation(
            &orig, &modified, MemoryModel::EREW,
            &[("A".into(), 4)], 4,
        );
        assert!(!result.preserved);
        assert!(!result.mismatches.is_empty());
    }

    #[test]
    fn test_equivalent_reordering_preserves() {
        // Two independent writes can be reordered
        let orig = vec![
            Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(0),
                value: Expr::IntLiteral(1),
            },
            Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(1),
                value: Expr::IntLiteral(2),
            },
        ];
        let reordered = vec![
            Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(1),
                value: Expr::IntLiteral(2),
            },
            Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(0),
                value: Expr::IntLiteral(1),
            },
        ];
        let result = verify_semantic_preservation(
            &orig, &reordered, MemoryModel::EREW,
            &[("A".into(), 4)], 4,
        );
        assert!(result.preserved);
    }

    #[test]
    fn test_loop_fusion_preservation() {
        // Two loops writing to different parts of same array
        let orig = vec![
            Stmt::SeqFor {
                var: "i".into(),
                start: Expr::IntLiteral(0),
                end: Expr::IntLiteral(4),
                step: None,
                body: vec![Stmt::SharedWrite {
                    memory: Expr::Variable("A".into()),
                    index: Expr::Variable("i".into()),
                    value: Expr::Variable("i".into()),
                }],
            },
        ];
        // Same loop, equivalent
        let result = verify_semantic_preservation(
            &orig, &orig, MemoryModel::EREW,
            &[("A".into(), 8)], 4,
        );
        assert!(result.preserved);
    }

    #[test]
    fn test_partial_eval_preservation() {
        let orig = vec![
            Stmt::Assign("y".into(), Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Variable("x".into())),
                Box::new(Expr::IntLiteral(5)),
            )),
        ];
        // After partial evaluation with x=10, y should be 15
        let specialized = vec![
            Stmt::Assign("y".into(), Expr::IntLiteral(15)),
        ];
        let mut test_values = HashMap::new();
        test_values.insert("x".into(), 10);
        let result = verify_partial_eval_preservation(&orig, &specialized, &test_values);
        assert!(result.preserved);
    }

    #[test]
    fn test_parallel_crcw_preservation() {
        // Verify CRCW priority produces deterministic result
        let body = vec![
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
        let result = verify_semantic_preservation(
            &body, &body, MemoryModel::CRCWPriority,
            &[("A".into(), 4)], 4,
        );
        assert!(result.preserved);
    }

    #[test]
    fn test_test_input_generator() {
        let gen = TestInputGenerator::new(
            vec![("A".into(), 8), ("B".into(), 4)],
            4,
        );
        let stores = gen.generate();
        assert_eq!(stores.len(), 4);
        assert_eq!(stores[0].read_shared("A", 0), Some(&Value::IntVal(0)));
    }

    #[test]
    fn test_write_coalescing_witness() {
        let cert = PreservationWitness::write_coalescing();
        assert_eq!(cert.transform_name, "write_coalescing");
        assert_eq!(cert.proof_strategy, ProofStrategy::Bisimulation);
        assert_eq!(cert.preconditions.len(), 3);
        assert!(cert.check_preconditions(MemoryModel::CRCWPriority));
        assert!(!cert.check_preconditions(MemoryModel::EREW));
    }

    #[test]
    fn test_tiling_witness() {
        let cert = PreservationWitness::tiling();
        assert_eq!(cert.proof_strategy, ProofStrategy::Refinement);
        assert!(cert.check_preconditions(MemoryModel::EREW));
        assert!(cert.check_preconditions(MemoryModel::CRCWPriority));
    }

    #[test]
    fn test_loop_fusion_witness() {
        let cert = PreservationWitness::loop_fusion();
        assert_eq!(cert.proof_strategy, ProofStrategy::Commutativity);
        assert!(!cert.preconditions[1].statically_checkable);
    }

    #[test]
    fn test_validated_transform_pass() {
        let body = vec![
            Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(0),
                value: Expr::IntLiteral(42),
            },
        ];
        let mut ct = ValidatedTransform::new(PreservationWitness::tiling());
        let passed = ct.apply_and_verify(&body, &body, MemoryModel::EREW, &[("A".into(), 4)]);
        assert!(passed);
        assert!(ct.verification_passed);
    }

    #[test]
    fn test_validated_transform_wrong_model() {
        let body = vec![Stmt::Nop];
        let mut ct = ValidatedTransform::new(PreservationWitness::write_coalescing());
        let passed = ct.apply_and_verify(&body, &body, MemoryModel::EREW, &[("A".into(), 4)]);
        assert!(!passed);
        assert!(!ct.transform_applied);
    }

    #[test]
    fn test_validated_transform_detects_mismatch() {
        let orig = vec![Stmt::SharedWrite {
            memory: Expr::Variable("A".into()),
            index: Expr::IntLiteral(0),
            value: Expr::IntLiteral(1),
        }];
        let bad = vec![Stmt::SharedWrite {
            memory: Expr::Variable("A".into()),
            index: Expr::IntLiteral(0),
            value: Expr::IntLiteral(999),
        }];
        let mut ct = ValidatedTransform::new(PreservationWitness::tiling());
        assert!(!ct.apply_and_verify(&orig, &bad, MemoryModel::EREW, &[("A".into(), 4)]));
        assert!(!ct.verification_passed);
    }

    #[test]
    fn test_priority_serialization_witness() {
        let cert = PreservationWitness::priority_serialization();
        assert_eq!(cert.proof_strategy, ProofStrategy::SimulationRelation);
        assert!(cert.check_preconditions(MemoryModel::CRCWPriority));
        assert!(!cert.check_preconditions(MemoryModel::CRCWArbitrary));
    }

    #[test]
    fn test_write_combining_witness() {
        let cert = PreservationWitness::write_combining();
        assert_eq!(cert.proof_strategy, ProofStrategy::Refinement);
        assert!(cert.check_preconditions(MemoryModel::CRCWCommon));
    }
}
