//! Main specialization pipeline.
//!
//! Orchestrates all passes: processor dispatch → model arbitration →
//! hash residualize → partial eval → work preservation check.

use std::collections::HashMap;

use crate::pram_ir::ast::{Expr, MemoryModel, PramProgram, Stmt};

use super::hash_residualize::{BlockAssignment, ResidualizePass};
use super::model_arbitration::ModelArbitrationPass;
use super::partial_eval::PartialEvaluator;
use super::processor_dispatch::{DispatchConfig, ProcessorDispatch};
use super::work_preservation::{WorkBoundChecker, WorkCounter};

/// Configuration for the specialization pipeline.
#[derive(Debug, Clone)]
pub struct SpecializerConfig {
    /// Whether to run processor dispatch (parallel_for → sequential)
    pub enable_dispatch: bool,
    /// Whether to run model arbitration (memory model → direct ops)
    pub enable_arbitration: bool,
    /// Whether to run hash residualization (hash lookups → direct access)
    pub enable_residualize: bool,
    /// Whether to run partial evaluation (constant prop + DCE + strength reduction)
    pub enable_partial_eval: bool,
    /// Whether to check work preservation bounds
    pub enable_work_check: bool,
    /// Unroll threshold for processor dispatch
    pub unroll_threshold: usize,
    /// Work bound multiplicative constant
    pub work_c1: usize,
    /// Work bound additive constant per region
    pub work_c2_per_region: usize,
}

impl Default for SpecializerConfig {
    fn default() -> Self {
        Self {
            enable_dispatch: true,
            enable_arbitration: true,
            enable_residualize: true,
            enable_partial_eval: true,
            enable_work_check: true,
            unroll_threshold: 32,
            work_c1: 4,
            work_c2_per_region: 10,
        }
    }
}

impl SpecializerConfig {
    pub fn all_disabled() -> Self {
        Self {
            enable_dispatch: false,
            enable_arbitration: false,
            enable_residualize: false,
            enable_partial_eval: false,
            enable_work_check: false,
            ..Self::default()
        }
    }
}

/// Result of specialization.
#[derive(Debug, Clone)]
pub struct SpecializationResult {
    /// The specialized program body
    pub body: Vec<Stmt>,
    /// Pre-specialization work count
    pub pre_work: usize,
    /// Post-specialization work count
    pub post_work: usize,
    /// Whether work preservation check passed
    pub work_preserved: bool,
    /// Passes that were applied
    pub passes_applied: Vec<String>,
    /// Any warnings generated during specialization
    pub warnings: Vec<String>,
}

/// The main specializer, orchestrating all transformation passes.
pub struct Specializer {
    config: SpecializerConfig,
}

impl Specializer {
    pub fn new(config: SpecializerConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self {
            config: SpecializerConfig::default(),
        }
    }

    /// Run the full specialization pipeline on a PRAM program.
    ///
    /// Pipeline order:
    ///   1. processor_dispatch — replace parallel_for with sequential code
    ///   2. model_arbitration — eliminate memory model overhead
    ///   3. hash_residualize — replace hash lookups with direct access
    ///   4. partial_eval — constant prop + DCE + strength reduction
    ///   5. work_preservation — verify work bounds
    ///   6. translation_validation — post-compilation correctness check
    pub fn specialize(
        &self,
        program: &PramProgram,
        block_assignments: Option<&BlockAssignment>,
    ) -> SpecializationResult {
        let pre_count = WorkCounter::count(&program.body);
        let pre_work = pre_count.total();

        let mut body = program.body.clone();
        let mut passes_applied = Vec::new();
        let mut warnings = Vec::new();

        // Phase 1: Processor dispatch
        if self.config.enable_dispatch {
            let dispatch = ProcessorDispatch::new(DispatchConfig::new(self.config.unroll_threshold));
            body = dispatch.transform(&body);
            passes_applied.push("processor_dispatch".to_string());
        }

        // Phase 2: Model arbitration
        if self.config.enable_arbitration {
            let arbitration = ModelArbitrationPass::new(program.memory_model);
            body = arbitration.transform(&body);
            passes_applied.push("model_arbitration".to_string());
        }

        // Phase 3: Hash residualization
        if self.config.enable_residualize {
            if let Some(assignments) = block_assignments {
                let residualize = ResidualizePass::new(assignments.clone());
                body = residualize.transform(&body);
                passes_applied.push("hash_residualize".to_string());
            } else {
                warnings.push("hash_residualize skipped: no block assignments provided".to_string());
            }
        }

        // Phase 4: Partial evaluation
        if self.config.enable_partial_eval {
            let mut pe = PartialEvaluator::new();
            if let Some(p) = program.num_processors.eval_const_int() {
                pe = pe.with_num_procs(p);
            }
            body = pe.evaluate(&body);
            passes_applied.push("partial_eval".to_string());
        }

        // Phase 5: Work preservation check
        let post_count = WorkCounter::count(&body);
        let post_work = post_count.total();
        let mut work_preserved = true;

        if self.config.enable_work_check {
            let checker = WorkBoundChecker::new(self.config.work_c1, self.config.work_c2_per_region);
            let mut adjusted_pre = pre_count.clone();
            adjusted_pre.shared_regions = program.shared_memory.len();
            let mut adjusted_post = post_count.clone();
            adjusted_post.shared_regions = program.shared_memory.len();

            match checker.check(&adjusted_pre, &adjusted_post) {
                Ok(()) => {
                    passes_applied.push("work_preservation_ok".to_string());
                }
                Err(violation) => {
                    work_preserved = false;
                    warnings.push(format!("Work preservation: {}", violation));
                    passes_applied.push("work_preservation_warn".to_string());
                }
            }
        }

        // Phase 6: Translation validation (post-compilation check)
        if self.config.enable_work_check {
            let validator = super::translation_validation::TranslationValidator::new();
            let pre_wc = WorkCounter::count(&program.body);
            let post_wc = WorkCounter::count(&body);
            let tv_result = validator.validate_structural(
                &program.body, &body, &pre_wc, &post_wc,
            );
            if tv_result.valid {
                passes_applied.push("translation_validation_ok".to_string());
            } else {
                warnings.push(format!(
                    "Translation validation: {} checks failed: {:?}",
                    tv_result.failed_checks.len(),
                    tv_result.failed_checks
                ));
                passes_applied.push("translation_validation_warn".to_string());
            }
        }

        SpecializationResult {
            body,
            pre_work,
            post_work,
            work_preserved,
            passes_applied,
            warnings,
        }
    }

    /// Specialize with a simple contiguous block assignment.
    pub fn specialize_with_blocks(
        &self,
        program: &PramProgram,
        block_size: usize,
    ) -> SpecializationResult {
        // Build block assignments from the program's shared memory declarations
        let mut ba = BlockAssignment::contiguous("__default", 1, 1);
        for decl in &program.shared_memory {
            if let Some(size) = decl.size.eval_const_int() {
                ba.add_contiguous_region(&decl.name, size as usize, block_size);
            }
        }
        self.specialize(program, Some(&ba))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pram_ir::ast::{BinOp, SharedMemoryDecl};
    use crate::pram_ir::types::PramType;

    fn simple_program() -> PramProgram {
        let mut prog = PramProgram::new("test_sum", MemoryModel::CREW);
        prog.num_processors = Expr::int(4);
        prog.shared_memory.push(SharedMemoryDecl {
            name: "A".to_string(),
            elem_type: PramType::Int64,
            size: Expr::int(100),
        });
        prog.body = vec![Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::int(4),
            body: vec![Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::var("pid"),
                value: Expr::binop(BinOp::Mul, Expr::var("pid"), Expr::int(2)),
            }],
        }];
        prog
    }

    #[test]
    fn test_full_pipeline() {
        let prog = simple_program();
        let spec = Specializer::with_default_config();
        let result = spec.specialize(&prog, None);

        assert!(result.passes_applied.contains(&"processor_dispatch".to_string()));
        assert!(result.passes_applied.contains(&"model_arbitration".to_string()));
        assert!(result.passes_applied.contains(&"partial_eval".to_string()));
        assert!(!result.body.is_empty());
    }

    #[test]
    fn test_pipeline_with_block_assignments() {
        let prog = simple_program();
        let spec = Specializer::with_default_config();
        let ba = BlockAssignment::contiguous("A", 100, 10);
        let result = spec.specialize(&prog, Some(&ba));

        assert!(result.passes_applied.contains(&"hash_residualize".to_string()));
        assert!(!result.body.is_empty());
    }

    #[test]
    fn test_pipeline_dispatch_only() {
        let prog = simple_program();
        let mut config = SpecializerConfig::all_disabled();
        config.enable_dispatch = true;
        let spec = Specializer::new(config);
        let result = spec.specialize(&prog, None);

        assert_eq!(result.passes_applied, vec!["processor_dispatch"]);
        // Should have unrolled 4 processors
        let writes: Vec<_> = result
            .body
            .iter()
            .filter(|s| matches!(s, Stmt::SharedWrite { .. }))
            .collect();
        assert_eq!(writes.len(), 4);
    }

    #[test]
    fn test_pipeline_partial_eval_only() {
        let prog = simple_program();
        let mut config = SpecializerConfig::all_disabled();
        config.enable_partial_eval = true;
        let spec = Specializer::new(config);
        let result = spec.specialize(&prog, None);

        assert_eq!(result.passes_applied, vec!["partial_eval"]);
    }

    #[test]
    fn test_pipeline_all_disabled() {
        let prog = simple_program();
        let config = SpecializerConfig::all_disabled();
        let spec = Specializer::new(config);
        let result = spec.specialize(&prog, None);

        assert!(result.passes_applied.is_empty());
        assert_eq!(result.body, prog.body);
    }

    #[test]
    fn test_constant_folding_after_dispatch() {
        let mut prog = PramProgram::new("fold_test", MemoryModel::EREW);
        prog.num_processors = Expr::int(2);
        prog.body = vec![Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::int(2),
            body: vec![Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::binop(BinOp::Add, Expr::var("pid"), Expr::int(0)),
                value: Expr::binop(BinOp::Mul, Expr::var("pid"), Expr::int(1)),
            }],
        }];
        prog.shared_memory.push(SharedMemoryDecl {
            name: "A".to_string(),
            elem_type: PramType::Int64,
            size: Expr::int(10),
        });

        let spec = Specializer::with_default_config();
        let result = spec.specialize(&prog, None);

        // After dispatch + partial eval, pid+0 → pid, pid*1 → pid
        // pid is substituted with 0,1 and then folded
        let writes: Vec<_> = result
            .body
            .iter()
            .filter(|s| matches!(s, Stmt::SharedWrite { .. }))
            .collect();
        assert_eq!(writes.len(), 2);

        // Check that values are folded
        match &writes[0] {
            Stmt::SharedWrite { value, index, .. } => {
                // pid=0: 0*1 = 0
                assert_eq!(*value, Expr::IntLiteral(0));
                assert_eq!(*index, Expr::IntLiteral(0));
            }
            _ => unreachable!(),
        }
        match &writes[1] {
            Stmt::SharedWrite { value, index, .. } => {
                // pid=1: 1*1 = 1
                assert_eq!(*value, Expr::IntLiteral(1));
                assert_eq!(*index, Expr::IntLiteral(1));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_work_preservation_check() {
        let prog = simple_program();
        let spec = Specializer::with_default_config();
        let result = spec.specialize(&prog, None);

        // For a small program, work should be preserved
        assert!(result.work_preserved);
    }

    #[test]
    fn test_specialize_with_blocks() {
        let prog = simple_program();
        let spec = Specializer::with_default_config();
        let result = spec.specialize_with_blocks(&prog, 10);

        assert!(result.passes_applied.contains(&"hash_residualize".to_string()));
        assert!(!result.body.is_empty());
    }

    #[test]
    fn test_empty_program() {
        let prog = PramProgram::new("empty", MemoryModel::EREW);
        let spec = Specializer::with_default_config();
        let result = spec.specialize(&prog, None);

        assert!(result.body.is_empty());
        assert_eq!(result.pre_work, 0);
        assert!(result.work_preserved);
    }

    #[test]
    fn test_result_fields() {
        let prog = simple_program();
        let spec = Specializer::with_default_config();
        let result = spec.specialize(&prog, None);

        assert!(result.pre_work > 0);
        assert!(result.post_work > 0);
        assert!(!result.passes_applied.is_empty());
    }

    #[test]
    fn test_erew_no_arbitration_overhead() {
        let mut prog = PramProgram::new("erew_test", MemoryModel::EREW);
        prog.num_processors = Expr::int(2);
        prog.body = vec![Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::int(2),
            body: vec![Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::var("pid"),
                value: Expr::int(1),
            }],
        }];
        prog.shared_memory.push(SharedMemoryDecl {
            name: "A".to_string(),
            elem_type: PramType::Int64,
            size: Expr::int(10),
        });

        let spec = Specializer::with_default_config();
        let result = spec.specialize(&prog, None);

        // Should still have 2 writes
        let writes: Vec<_> = result
            .body
            .iter()
            .filter(|s| matches!(s, Stmt::SharedWrite { .. }))
            .collect();
        assert_eq!(writes.len(), 2);
    }

    #[test]
    fn test_large_proc_count_uses_loop() {
        let mut prog = PramProgram::new("large_test", MemoryModel::EREW);
        prog.num_processors = Expr::int(100);
        prog.body = vec![Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::int(100),
            body: vec![Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::var("pid"),
                value: Expr::int(1),
            }],
        }];

        let spec = Specializer::with_default_config();
        let result = spec.specialize(&prog, None);

        // Should use a sequential loop instead of unrolling
        let has_loop = result.body.iter().any(|s| matches!(s, Stmt::SeqFor { .. }));
        assert!(has_loop, "Expected sequential loop for 100 processors");
    }

    #[test]
    fn test_custom_unroll_threshold() {
        let mut prog = PramProgram::new("thresh_test", MemoryModel::EREW);
        prog.num_processors = Expr::int(5);
        prog.body = vec![Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::int(5),
            body: vec![Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::var("pid"),
                value: Expr::var("pid"),
            }],
        }];
        prog.shared_memory.push(crate::pram_ir::ast::SharedMemoryDecl {
            name: "A".to_string(),
            elem_type: crate::pram_ir::types::PramType::Int64,
            size: Expr::int(5),
        });

        // Threshold of 3: 5 > 3, so should use loop
        let mut config = SpecializerConfig::default();
        config.unroll_threshold = 3;
        config.enable_partial_eval = false;
        let spec = Specializer::new(config);
        let result = spec.specialize(&prog, None);

        let has_loop = result.body.iter().any(|s| matches!(s, Stmt::SeqFor { .. }));
        assert!(has_loop, "Expected loop when num_procs > threshold");

        // Threshold of 5: 5 <= 5, so should unroll
        let mut config2 = SpecializerConfig::default();
        config2.unroll_threshold = 5;
        config2.enable_partial_eval = false;
        let spec2 = Specializer::new(config2);
        let result2 = spec2.specialize(&prog, None);

        let has_loop2 = result2.body.iter().any(|s| matches!(s, Stmt::SeqFor { .. }));
        assert!(!has_loop2, "Expected unrolling when num_procs <= threshold");
    }
}
