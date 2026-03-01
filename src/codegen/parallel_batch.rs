//! Parallel batch compilation using rayon.
//!
//! Compiles multiple PRAM algorithms concurrently, exploiting
//! the independence of each algorithm's code generation pipeline.

use rayon::prelude::*;
use crate::pram_ir::ast::PramProgram;
use super::adaptive::{AdaptiveCompiler, CompilationTarget, ProfiledCompilation};
use super::generator::{CodeGenerator, GeneratorConfig};

/// Result of compiling a single algorithm in a batch.
#[derive(Debug, Clone)]
pub struct BatchCompilationResult {
    pub algorithm_name: String,
    pub target: String,
    pub code: String,
    pub success: bool,
    pub error: Option<String>,
}

/// Result of compiling a batch of algorithms in parallel.
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub results: Vec<BatchCompilationResult>,
    pub total: usize,
    pub succeeded: usize,
    pub failed: usize,
}

/// Compile a batch of PRAM programs in parallel using rayon.
pub fn compile_batch(
    programs: &[PramProgram],
    target: &CompilationTarget,
) -> BatchResult {
    let results: Vec<BatchCompilationResult> = programs
        .par_iter()
        .map(|program| {
            let compiler = AdaptiveCompiler::new();
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                compiler.compile(program, target)
            })) {
                Ok(code) => BatchCompilationResult {
                    algorithm_name: program.name.clone(),
                    target: format!("{:?}", target),
                    code,
                    success: true,
                    error: None,
                },
                Err(_) => BatchCompilationResult {
                    algorithm_name: program.name.clone(),
                    target: format!("{:?}", target),
                    code: String::new(),
                    success: false,
                    error: Some("Compilation panicked".to_string()),
                },
            }
        })
        .collect();

    let succeeded = results.iter().filter(|r| r.success).count();
    let total = results.len();
    BatchResult {
        results,
        total,
        succeeded,
        failed: total - succeeded,
    }
}

/// Compile a batch of programs with profile-guided optimization in parallel.
pub fn compile_batch_profiled(
    programs: &[PramProgram],
    target: &CompilationTarget,
) -> Vec<ProfiledCompilation> {
    use super::adaptive::ProfileGuidedCompiler;

    programs
        .par_iter()
        .map(|program| {
            let compiler = ProfileGuidedCompiler::new();
            compiler.compile_with_profiling(program, target)
        })
        .collect()
}

/// Compile the same program for multiple targets in parallel.
pub fn compile_multi_target(
    program: &PramProgram,
    targets: &[CompilationTarget],
) -> Vec<BatchCompilationResult> {
    targets
        .par_iter()
        .map(|target| {
            let compiler = AdaptiveCompiler::new();
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                compiler.compile(program, target)
            })) {
                Ok(code) => BatchCompilationResult {
                    algorithm_name: program.name.clone(),
                    target: format!("{:?}", target),
                    code,
                    success: true,
                    error: None,
                },
                Err(_) => BatchCompilationResult {
                    algorithm_name: program.name.clone(),
                    target: format!("{:?}", target),
                    code: String::new(),
                    success: false,
                    error: Some("Compilation panicked".to_string()),
                },
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pram_ir::ast::*;
    use crate::pram_ir::types::*;

    fn test_program(name: &str) -> PramProgram {
        PramProgram {
            name: name.into(),
            memory_model: MemoryModel::CREW,
            parameters: vec![Parameter {
                name: "n".into(),
                param_type: PramType::Int64,
            }],
            shared_memory: vec![SharedMemoryDecl {
                name: "A".into(),
                elem_type: PramType::Int64,
                size: Expr::int(100),
            }],
            body: vec![Stmt::ParallelFor {
                proc_var: "pid".into(),
                num_procs: Expr::int(100),
                body: vec![Stmt::SharedWrite {
                    memory: Expr::var("A"),
                    index: Expr::ProcessorId,
                    value: Expr::ProcessorId,
                }],
            }],
            num_processors: Expr::int(100),
            work_bound: Some("O(n)".into()),
            time_bound: Some("O(1)".into()),
            description: Some(format!("Test algorithm {}", name)),
        }
    }

    #[test]
    fn test_batch_compile_sequential() {
        let programs: Vec<PramProgram> = (0..5)
            .map(|i| test_program(&format!("alg_{}", i)))
            .collect();
        let result = compile_batch(&programs, &CompilationTarget::Sequential);
        assert_eq!(result.total, 5);
        assert_eq!(result.succeeded, 5);
        assert_eq!(result.failed, 0);
        for r in &result.results {
            assert!(r.success);
            assert!(r.code.contains("int main("));
        }
    }

    #[test]
    fn test_batch_compile_parallel() {
        let programs: Vec<PramProgram> = (0..3)
            .map(|i| test_program(&format!("par_alg_{}", i)))
            .collect();
        let target = CompilationTarget::Parallel { num_threads: 4 };
        let result = compile_batch(&programs, &target);
        assert_eq!(result.succeeded, 3);
        for r in &result.results {
            assert!(r.code.contains("#pragma omp"));
        }
    }

    #[test]
    fn test_multi_target_compile() {
        let program = test_program("multi_target");
        let targets = vec![
            CompilationTarget::Sequential,
            CompilationTarget::Parallel { num_threads: 4 },
            CompilationTarget::Adaptive { crossover_n: 1024 },
        ];
        let results = compile_multi_target(&program, &targets);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.success));
    }

    #[test]
    fn test_batch_profiled_compile() {
        let programs: Vec<PramProgram> = (0..3)
            .map(|i| test_program(&format!("prof_alg_{}", i)))
            .collect();
        let target = CompilationTarget::Sequential;
        let results = compile_batch_profiled(&programs, &target);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.meets_2x);
            assert!(!r.code.is_empty());
        }
    }
}
