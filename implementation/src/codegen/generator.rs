//! Main code generation pipeline.
//!
//! Combines constant folding, loop restructuring, memory layout computation,
//! and C emission into a single `CodeGenerator` that takes a `PramProgram`
//! and produces a complete C99 source file.

use crate::pram_ir::ast::*;
use crate::pram_ir::types::*;

use super::c_emitter::CEmitter;
use super::constant_fold::ConstantFolder;
use super::loop_restructure::LoopRestructurer;
use super::memory_layout::MemoryLayout;
use super::template::CTemplate;

// ---------------------------------------------------------------------------
// GeneratorConfig
// ---------------------------------------------------------------------------

/// Configuration for the code generation pipeline.
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Optimization level: 0 = none, 1 = constant fold, 2 = fold + loop opt.
    pub opt_level: u8,
    /// Include timing instrumentation in generated code.
    pub include_timing: bool,
    /// Include runtime assertion checks.
    pub include_assertions: bool,
    /// Tile size for loop tiling (0 = disable tiling).
    pub tile_size: usize,
    /// Minimum trip count for loop tiling to kick in.
    pub min_trip_for_tiling: usize,
    /// Default shared memory region size for non-constant sizes.
    pub default_region_size: usize,
}

impl GeneratorConfig {
    pub fn new() -> Self {
        Self {
            opt_level: 2,
            include_timing: false,
            include_assertions: true,
            tile_size: 64,
            min_trip_for_tiling: 128,
            default_region_size: 1024,
        }
    }

    /// No optimizations; useful for debugging.
    pub fn debug() -> Self {
        Self {
            opt_level: 0,
            include_timing: false,
            include_assertions: true,
            tile_size: 0,
            min_trip_for_tiling: usize::MAX,
            default_region_size: 1024,
        }
    }

    /// Maximum optimizations.
    pub fn release() -> Self {
        Self {
            opt_level: 2,
            include_timing: false,
            include_assertions: false,
            tile_size: 64,
            min_trip_for_tiling: 128,
            default_region_size: 1024,
        }
    }
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CodeGenerator
// ---------------------------------------------------------------------------

/// Orchestrates the full code-generation pipeline.
///
/// Pipeline order:
/// 1. Constant folding (opt_level ≥ 1)
/// 2. Loop restructuring (opt_level ≥ 2)
/// 3. Memory layout computation
/// 4. C code emission
pub struct CodeGenerator {
    pub config: GeneratorConfig,
}

impl CodeGenerator {
    pub fn new(config: GeneratorConfig) -> Self {
        Self { config }
    }

    /// Generate a complete C99 source file from a `PramProgram`.
    pub fn generate(&self, program: &PramProgram) -> String {
        // 1. Optionally constant-fold
        let body = if self.config.opt_level >= 1 {
            let mut folder = ConstantFolder::new();
            folder.fold_body(&program.body)
        } else {
            program.body.clone()
        };

        // 2. Optionally restructure loops
        let body = if self.config.opt_level >= 2 && self.config.tile_size > 0 {
            let mut restructurer = LoopRestructurer::new()
                .with_tile_size(self.config.tile_size)
                .with_min_trip(self.config.min_trip_for_tiling);
            restructurer.restructure(&body)
        } else {
            body
        };

        // 3. Compute memory layout (used for documentation/comments; the
        //    emitter handles actual declarations).
        let _layout = MemoryLayout::compute(
            &program.shared_memory,
            self.config.default_region_size,
        );

        // 4. Build template + emit
        let mut template = CTemplate::with_standard_preamble();

        if self.config.include_timing {
            template.add_timing_helpers();
        }

        // Emit program comment header
        template.add_header_line(&format!(
            "/* PRAM program: {} | model: {} */",
            program.name,
            program.memory_model.name(),
        ));
        if let Some(ref desc) = program.description {
            template.add_header_line(&format!("/* {} */", desc));
        }
        if let Some(ref wb) = program.work_bound {
            template.add_header_line(&format!("/* Work bound: {} */", wb));
        }
        if let Some(ref tb) = program.time_bound {
            template.add_header_line(&format!("/* Time bound: {} */", tb));
        }
        template.add_header_line("");

        // Build a modified program with the optimized body
        let opt_program = PramProgram {
            name: program.name.clone(),
            memory_model: program.memory_model,
            parameters: program.parameters.clone(),
            shared_memory: program.shared_memory.clone(),
            body,
            num_processors: program.num_processors.clone(),
            work_bound: program.work_bound.clone(),
            time_bound: program.time_bound.clone(),
            description: program.description.clone(),
        };

        let mut emitter = CEmitter::new();
        let program_code = emitter.emit_program(&opt_program);

        template.add_body(&program_code);

        template.render()
    }
}

impl Default for CodeGenerator {
    fn default() -> Self {
        Self::new(GeneratorConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Multi-variant generation
// ---------------------------------------------------------------------------

/// Generate multiple C code variants from different configs.
/// Returns a vec of (config, generated_code) pairs.
pub fn generate_with_variants(
    program: &PramProgram,
    configs: &[GeneratorConfig],
) -> Vec<(GeneratorConfig, String)> {
    configs
        .iter()
        .map(|cfg| {
            let gen = CodeGenerator::new(cfg.clone());
            let code = gen.generate(program);
            (cfg.clone(), code)
        })
        .collect()
}

/// Estimate the generated C file size in bytes.
///
/// Heuristic: ~50 bytes per statement + ~500 bytes header overhead
/// + ~80 bytes per shared memory declaration.
pub fn estimate_output_size(program: &PramProgram) -> usize {
    let stmt_count = count_stmts(&program.body);
    let region_count = program.shared_memory.len();
    stmt_count * 50 + 500 + region_count * 80
}

// ---------------------------------------------------------------------------
// GenerationReport
// ---------------------------------------------------------------------------

/// Summary report for a generated C file.
#[derive(Debug, Clone)]
pub struct GenerationReport {
    pub code_size: usize,
    pub region_count: usize,
    pub loop_count: usize,
    pub stmt_count: usize,
    pub has_barriers: bool,
    pub memory_model_name: String,
}

impl GenerationReport {
    pub fn from_program(program: &PramProgram, code: &str) -> GenerationReport {
        GenerationReport {
            code_size: code.len(),
            region_count: program.shared_memory.len(),
            loop_count: count_loops(&program.body),
            stmt_count: count_stmts(&program.body),
            has_barriers: has_barrier(&program.body),
            memory_model_name: program.memory_model.name().to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Basic C syntax checks on generated code.
///
/// Returns a list of warning/error strings. Empty means no issues found.
pub fn validate_generated_code(c_code: &str) -> Vec<String> {
    let mut issues = Vec::new();

    let open_braces = c_code.chars().filter(|&c| c == '{').count();
    let close_braces = c_code.chars().filter(|&c| c == '}').count();
    if open_braces != close_braces {
        issues.push(format!(
            "Mismatched braces: {} opening vs {} closing",
            open_braces, close_braces
        ));
    }

    let open_parens = c_code.chars().filter(|&c| c == '(').count();
    let close_parens = c_code.chars().filter(|&c| c == ')').count();
    if open_parens != close_parens {
        issues.push(format!(
            "Mismatched parentheses: {} opening vs {} closing",
            open_parens, close_parens
        ));
    }

    if !c_code.contains("int main(") {
        issues.push("Missing main function".to_string());
    }

    if !c_code.contains("return") {
        issues.push("Missing return statement".to_string());
    }

    issues
}

// ---------------------------------------------------------------------------
// Internal helpers for statement traversal
// ---------------------------------------------------------------------------

fn count_stmts(stmts: &[Stmt]) -> usize {
    let mut count = 0;
    for stmt in stmts {
        count += 1;
        match stmt {
            Stmt::ParallelFor { body, .. } => count += count_stmts(body),
            Stmt::SeqFor { body, .. } => count += count_stmts(body),
            Stmt::While { body, .. } => count += count_stmts(body),
            Stmt::If { then_body, else_body, .. } => {
                count += count_stmts(then_body) + count_stmts(else_body);
            }
            Stmt::Block(inner) => count += count_stmts(inner),
            _ => {}
        }
    }
    count
}

fn count_loops(stmts: &[Stmt]) -> usize {
    let mut count = 0;
    for stmt in stmts {
        match stmt {
            Stmt::ParallelFor { body, .. } => {
                count += 1 + count_loops(body);
            }
            Stmt::SeqFor { body, .. } => {
                count += 1 + count_loops(body);
            }
            Stmt::While { body, .. } => {
                count += 1 + count_loops(body);
            }
            Stmt::If { then_body, else_body, .. } => {
                count += count_loops(then_body) + count_loops(else_body);
            }
            Stmt::Block(inner) => count += count_loops(inner),
            _ => {}
        }
    }
    count
}

fn has_barrier(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|stmt| match stmt {
        Stmt::Barrier => true,
        Stmt::ParallelFor { body, .. } => has_barrier(body),
        Stmt::SeqFor { body, .. } => has_barrier(body),
        Stmt::While { body, .. } => has_barrier(body),
        Stmt::If { then_body, else_body, .. } => {
            has_barrier(then_body) || has_barrier(else_body)
        }
        Stmt::Block(inner) => has_barrier(inner),
        _ => false,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_program() -> PramProgram {
        PramProgram {
            name: "array_init".into(),
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
                    value: Expr::binop(BinOp::Add, Expr::ProcessorId, Expr::int(0)),
                }],
            }],
            num_processors: Expr::int(100),
            work_bound: Some("O(n)".into()),
            time_bound: Some("O(1)".into()),
            description: Some("Initialize array with processor IDs".into()),
        }
    }

    #[test]
    fn test_generate_produces_valid_c_structure() {
        let gen = CodeGenerator::new(GeneratorConfig::new());
        let code = gen.generate(&simple_program());

        // Must have includes
        assert!(code.contains("#include <stdio.h>"));
        assert!(code.contains("#include <stdlib.h>"));
        assert!(code.contains("#include <stdint.h>"));

        // Must have main
        assert!(code.contains("int main("));
        assert!(code.contains("return 0;"));

        // Must have our program's shared memory
        assert!(code.contains("int64_t* A"));

        // Must have the for loop
        assert!(code.contains("for (int64_t pid"));

        // Must free memory
        assert!(code.contains("pram_free(A)"));
    }

    #[test]
    fn test_generate_includes_macros() {
        let gen = CodeGenerator::new(GeneratorConfig::new());
        let code = gen.generate(&simple_program());
        assert!(code.contains("PRAM_MIN"));
        assert!(code.contains("PRAM_MAX"));
        assert!(code.contains("PRAM_SWAP"));
    }

    #[test]
    fn test_generate_includes_wrappers() {
        let gen = CodeGenerator::new(GeneratorConfig::new());
        let code = gen.generate(&simple_program());
        assert!(code.contains("pram_malloc"));
        assert!(code.contains("pram_calloc"));
    }

    #[test]
    fn test_generate_with_timing() {
        let mut config = GeneratorConfig::new();
        config.include_timing = true;
        let gen = CodeGenerator::new(config);
        let code = gen.generate(&simple_program());
        assert!(code.contains("pram_timer_start"));
        assert!(code.contains("PRAM_TIMING"));
    }

    #[test]
    fn test_generate_debug_mode() {
        let gen = CodeGenerator::new(GeneratorConfig::debug());
        let code = gen.generate(&simple_program());
        // Should still produce valid C
        assert!(code.contains("int main("));
        assert!(code.contains("return 0;"));
    }

    #[test]
    fn test_generate_release_mode() {
        let gen = CodeGenerator::new(GeneratorConfig::release());
        let code = gen.generate(&simple_program());
        assert!(code.contains("int main("));
    }

    #[test]
    fn test_constant_folding_applied() {
        // The body has `pid + 0` which should fold to just `pid`
        let gen = CodeGenerator::new(GeneratorConfig::new());
        let code = gen.generate(&simple_program());
        // After folding, A[pid] = pid (not pid + 0)
        assert!(code.contains("A[pid] = pid;"));
    }

    #[test]
    fn test_generate_crcw_priority() {
        let program = PramProgram {
            name: "crcw_test".into(),
            memory_model: MemoryModel::CRCWPriority,
            parameters: vec![],
            shared_memory: vec![SharedMemoryDecl {
                name: "M".into(),
                elem_type: PramType::Int64,
                size: Expr::int(10),
            }],
            body: vec![Stmt::ParallelFor {
                proc_var: "p".into(),
                num_procs: Expr::int(10),
                body: vec![Stmt::SharedWrite {
                    memory: Expr::var("M"),
                    index: Expr::int(0),
                    value: Expr::ProcessorId,
                }],
            }],
            num_processors: Expr::int(10),
            work_bound: None,
            time_bound: None,
            description: None,
        };

        let gen = CodeGenerator::new(GeneratorConfig::new());
        let code = gen.generate(&program);

        assert!(code.contains("CRCW-Priority"));
        assert!(code.contains("_wpid_M"));
        assert!(code.contains("_stg_M"));
    }

    #[test]
    fn test_generate_empty_program() {
        let program = PramProgram::new("empty", MemoryModel::EREW);
        let gen = CodeGenerator::new(GeneratorConfig::new());
        let code = gen.generate(&program);
        assert!(code.contains("int main("));
        assert!(code.contains("return 0;"));
    }

    #[test]
    fn test_generate_with_description() {
        let gen = CodeGenerator::new(GeneratorConfig::new());
        let code = gen.generate(&simple_program());
        assert!(code.contains("Initialize array with processor IDs"));
        assert!(code.contains("O(n)"));  // work bound
        assert!(code.contains("O(1)"));  // time bound
    }

    #[test]
    fn test_dead_code_eliminated() {
        let program = PramProgram {
            name: "dead_code".into(),
            memory_model: MemoryModel::CREW,
            parameters: vec![],
            shared_memory: vec![],
            body: vec![
                Stmt::If {
                    condition: Expr::bool_lit(false),
                    then_body: vec![Stmt::Assign("x".into(), Expr::int(42))],
                    else_body: vec![],
                },
                Stmt::Assign("y".into(), Expr::int(1)),
            ],
            num_processors: Expr::int(1),
            work_bound: None,
            time_bound: None,
            description: None,
        };

        let gen = CodeGenerator::new(GeneratorConfig::new());
        let code = gen.generate(&program);
        // The dead if(false) branch should be eliminated
        assert!(!code.contains("x = 42"));
        assert!(code.contains("y = 1LL;"));
    }

    #[test]
    fn test_generator_config_defaults() {
        let config = GeneratorConfig::default();
        assert_eq!(config.opt_level, 2);
        assert!(!config.include_timing);
        assert!(config.include_assertions);
        assert_eq!(config.tile_size, 64);
    }

    #[test]
    fn test_default_generator() {
        let gen = CodeGenerator::default();
        assert_eq!(gen.config.opt_level, 2);
    }

    #[test]
    fn test_generate_with_multiple_shared_regions() {
        let program = PramProgram {
            name: "multi_region".into(),
            memory_model: MemoryModel::CREW,
            parameters: vec![],
            shared_memory: vec![
                SharedMemoryDecl {
                    name: "A".into(),
                    elem_type: PramType::Int64,
                    size: Expr::int(100),
                },
                SharedMemoryDecl {
                    name: "B".into(),
                    elem_type: PramType::Float64,
                    size: Expr::int(50),
                },
            ],
            body: vec![Stmt::ParallelFor {
                proc_var: "pid".into(),
                num_procs: Expr::int(50),
                body: vec![
                    Stmt::SharedWrite {
                        memory: Expr::var("A"),
                        index: Expr::ProcessorId,
                        value: Expr::int(0),
                    },
                    Stmt::SharedWrite {
                        memory: Expr::var("B"),
                        index: Expr::ProcessorId,
                        value: Expr::float(0.0),
                    },
                ],
            }],
            num_processors: Expr::int(50),
            work_bound: None,
            time_bound: None,
            description: None,
        };

        let gen = CodeGenerator::new(GeneratorConfig::new());
        let code = gen.generate(&program);
        assert!(code.contains("int64_t* A"));
        assert!(code.contains("double* B"));
        assert!(code.contains("pram_free(A)"));
        assert!(code.contains("pram_free(B)"));
    }

    #[test]
    fn test_generate_with_variants_produces_all_configs() {
        let program = simple_program();
        let configs = vec![
            GeneratorConfig::debug(),
            GeneratorConfig::new(),
            GeneratorConfig::release(),
        ];
        let results = generate_with_variants(&program, &configs);
        assert_eq!(results.len(), 3);
        for (_, code) in &results {
            assert!(code.contains("int main("));
        }
    }

    #[test]
    fn test_generate_with_variants_empty_configs() {
        let program = simple_program();
        let results = generate_with_variants(&program, &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_estimate_output_size() {
        let program = simple_program();
        let estimate = estimate_output_size(&program);
        assert!(estimate > 0);
        assert!(estimate >= 500); // at least header overhead
    }

    #[test]
    fn test_generation_report_from_program() {
        let program = simple_program();
        let gen = CodeGenerator::new(GeneratorConfig::new());
        let code = gen.generate(&program);
        let report = GenerationReport::from_program(&program, &code);
        assert_eq!(report.region_count, 1);
        assert_eq!(report.memory_model_name, "CREW");
        assert!(report.code_size > 0);
        assert!(report.loop_count >= 1);
        assert!(report.stmt_count >= 2);
    }

    #[test]
    fn test_generation_report_no_barriers() {
        let program = simple_program();
        let gen = CodeGenerator::new(GeneratorConfig::new());
        let code = gen.generate(&program);
        let report = GenerationReport::from_program(&program, &code);
        assert!(!report.has_barriers);
    }

    #[test]
    fn test_validate_generated_code_valid() {
        let gen = CodeGenerator::new(GeneratorConfig::new());
        let code = gen.generate(&simple_program());
        let issues = validate_generated_code(&code);
        assert!(issues.is_empty(), "Expected no issues but got: {:?}", issues);
    }

    #[test]
    fn test_validate_generated_code_missing_main() {
        let issues = validate_generated_code("void foo() { return; }");
        assert!(issues.iter().any(|s| s.contains("Missing main")));
    }

    #[test]
    fn test_validate_generated_code_mismatched_braces() {
        let issues = validate_generated_code("int main() { { return 0; }");
        assert!(issues.iter().any(|s| s.contains("braces")));
    }
}
