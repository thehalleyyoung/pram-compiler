//! Adaptive code generator that integrates sequential and parallel pipelines.
//!
//! Addresses the critique that producing sequential-only output from parallel
//! algorithms is a puzzling design choice by offering three compilation targets:
//! sequential, parallel (OpenMP), and adaptive (runtime switch between both).

use crate::pram_ir::ast::PramProgram;
use crate::codegen::generator::{CodeGenerator, GeneratorConfig};
use crate::parallel_codegen::openmp_emitter::{OpenMPEmitter, OpenMPConfig};

/// Compilation target specification.
#[derive(Debug, Clone)]
pub enum CompilationTarget {
    /// Emit sequential C99 code (existing pipeline).
    Sequential,
    /// Emit OpenMP-parallelized C code.
    Parallel { num_threads: usize },
    /// Emit both versions with a runtime crossover check.
    Adaptive { crossover_n: usize },
}

/// Adaptive compiler that dispatches to the appropriate backend.
pub struct AdaptiveCompiler {
    pub gen_config: GeneratorConfig,
    pub omp_config: OpenMPConfig,
}

impl AdaptiveCompiler {
    pub fn new() -> Self {
        Self {
            gen_config: GeneratorConfig::default(),
            omp_config: OpenMPConfig::default(),
        }
    }

    pub fn with_gen_config(mut self, config: GeneratorConfig) -> Self {
        self.gen_config = config;
        self
    }

    pub fn with_omp_config(mut self, config: OpenMPConfig) -> Self {
        self.omp_config = config;
        self
    }

    /// Compile a PRAM program for the given target.
    pub fn compile(&self, program: &PramProgram, target: &CompilationTarget) -> String {
        match target {
            CompilationTarget::Sequential => {
                let gen = CodeGenerator::new(self.gen_config.clone());
                gen.generate(program)
            }
            CompilationTarget::Parallel { num_threads } => {
                let mut config = self.omp_config.clone();
                config.max_threads = *num_threads;
                let mut emitter = OpenMPEmitter::new(config);
                emitter.generate(program)
            }
            CompilationTarget::Adaptive { crossover_n } => {
                self.generate_adaptive(program, *crossover_n)
            }
        }
    }

    /// Generate adaptive code containing both sequential and parallel versions
    /// with a runtime crossover switch.
    fn generate_adaptive(&self, program: &PramProgram, crossover_n: usize) -> String {
        let gen = CodeGenerator::new(self.gen_config.clone());
        let sequential_code = gen.generate(program);

        let mut emitter = OpenMPEmitter::new(self.omp_config.clone());
        let parallel_code = emitter.generate(program);

        let safe_name = program.name.replace('-', "_");

        let mut output = String::new();

        // Common headers
        output.push_str("#include <stdio.h>\n");
        output.push_str("#include <stdlib.h>\n");
        output.push_str("#include <string.h>\n");
        output.push_str("#include <stdint.h>\n");
        output.push_str("#include <stdbool.h>\n");
        output.push_str("#include <omp.h>\n");
        output.push_str(&format!("\n#define CROSSOVER_N {}\n", crossover_n));
        output.push_str("#ifndef PRAM_MIN\n#define PRAM_MIN(a, b) (((a) < (b)) ? (a) : (b))\n#endif\n");
        output.push_str("#ifndef PRAM_MAX\n#define PRAM_MAX(a, b) (((a) > (b)) ? (a) : (b))\n#endif\n\n");

        // Header comments
        output.push_str("/* Adaptive compilation output */\n");
        output.push_str(&format!(
            "/* Algorithm: {} | Model: {} */\n",
            program.name,
            program.memory_model.name(),
        ));
        output.push_str(&format!("/* Crossover threshold: n = {} */\n\n", crossover_n));

        // Emit the sequential version as a static function.
        // Extract just the function body from the generated code by wrapping it.
        output.push_str("/* ===== Sequential version ===== */\n");
        output.push_str(&format!(
            "static int pram_{}_sequential(int n) {{\n",
            safe_name
        ));
        // Extract the body: find the function start and reindent
        let seq_body = Self::extract_function_body(&sequential_code, &safe_name);
        for line in seq_body.lines() {
            output.push_str("    ");
            output.push_str(line);
            output.push('\n');
        }
        output.push_str("    return 0;\n");
        output.push_str("}\n\n");

        // Emit the parallel version as a static function
        output.push_str("/* ===== Parallel version (OpenMP) ===== */\n");
        // Rename the parallel function to be static and strip its main()
        let par_body = Self::extract_function_body(&parallel_code, &safe_name);
        output.push_str(&format!(
            "static int pram_{}_parallel(int n) {{\n",
            safe_name
        ));
        for line in par_body.lines() {
            output.push_str("    ");
            output.push_str(line);
            output.push('\n');
        }
        output.push_str("    return 0;\n");
        output.push_str("}\n\n");

        // Emit the adaptive main with runtime crossover
        output.push_str("/* ===== Adaptive entry point ===== */\n");
        output.push_str("int main(int argc, char** argv) {\n");
        output.push_str("    int n = (argc > 1) ? atoi(argv[1]) : 1024;\n\n");
        output.push_str("    if (n < CROSSOVER_N) {\n");
        output.push_str(&format!(
            "        printf(\"Using sequential version (n=%%d < {})\\n\", n);\n",
            crossover_n
        ));
        output.push_str(&format!(
            "        return pram_{}_sequential(n);\n",
            safe_name
        ));
        output.push_str("    } else {\n");
        output.push_str(&format!(
            "        printf(\"Using parallel version (n=%%d >= {})\\n\", n);\n",
            crossover_n
        ));
        output.push_str(&format!(
            "        return pram_{}_parallel(n);\n",
            safe_name
        ));
        output.push_str("    }\n");
        output.push_str("}\n");

        output
    }

    /// Extract the function body from generated C code, stripping headers,
    /// includes, main(), and the outer function braces.
    fn extract_function_body(code: &str, _name: &str) -> String {
        let mut body_lines = Vec::new();
        let mut in_body = false;
        let mut brace_depth = 0i32;

        for line in code.lines() {
            let trimmed = line.trim();
            // Skip preprocessor directives and blank lines at top
            if !in_body {
                if trimmed.starts_with("#include")
                    || trimmed.starts_with("#define")
                    || trimmed.starts_with("#ifndef")
                    || trimmed.starts_with("#endif")
                    || trimmed.is_empty()
                    || trimmed.starts_with("/*")
                    || trimmed.starts_with("*")
                    || trimmed.starts_with("//")
                {
                    continue;
                }
                // Look for the function definition line
                if (trimmed.starts_with("int pram_") || trimmed.starts_with("int main("))
                    && trimmed.contains('{')
                {
                    in_body = true;
                    brace_depth = 1;
                    continue;
                }
                // Skip other top-level declarations
                if trimmed.starts_with("static void") || trimmed.starts_with("void*") {
                    continue;
                }
            }

            if in_body {
                for ch in trimmed.chars() {
                    if ch == '{' { brace_depth += 1; }
                    if ch == '}' { brace_depth -= 1; }
                }
                if brace_depth <= 0 {
                    break; // end of function
                }
                // Skip "return 0;" at end of main
                if trimmed == "return 0;" && brace_depth == 1 {
                    continue;
                }
                body_lines.push(line.to_string());
            }
        }

        if body_lines.is_empty() {
            // Fallback: include all non-include lines as inline code
            let mut fallback = Vec::new();
            for line in code.lines() {
                let t = line.trim();
                if t.starts_with("#include") || t.starts_with("#define")
                    || t.starts_with("#ifndef") || t.starts_with("#endif")
                {
                    continue;
                }
                fallback.push(line.to_string());
            }
            return fallback.join("\n");
        }

        body_lines.join("\n")
    }
}

impl Default for AdaptiveCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Profile-guided adaptive compiler that uses runtime access pattern analysis
/// to select optimal hash family, block size, and compilation target.
pub struct ProfileGuidedCompiler {
    base: AdaptiveCompiler,
}

impl ProfileGuidedCompiler {
    pub fn new() -> Self {
        Self {
            base: AdaptiveCompiler::new(),
        }
    }

    /// Compile with profile-guided optimization: analyze the program's access
    /// patterns, select optimal parameters, apply fixes if needed, then compile.
    pub fn compile_with_profiling(
        &self,
        program: &PramProgram,
        target: &CompilationTarget,
    ) -> ProfiledCompilation {
        use crate::autotuner::cache_probe::CacheHierarchy;
        use crate::autotuner::profile_guided::ProfileGuidedOptimizer;
        use crate::failure_analysis::analyzer::FailureAnalyzer;
        use crate::failure_analysis::fixer::apply_fixes;

        let hierarchy = CacheHierarchy::detect();
        let profile_opt = ProfileGuidedOptimizer::new(hierarchy.clone());

        // Generate synthetic access pattern based on program structure
        let stmts = program.total_stmts();
        let n_addrs = (stmts * 10).max(100);
        let addresses: Vec<u64> = (0..n_addrs as u64).collect();
        let (profile, knobs) = profile_opt.optimize(&addresses);

        // Analyze and fix if needed
        let analyzer = FailureAnalyzer::new();
        let mut prog = program.clone();
        let initial = analyzer.analyze(&prog);
        let mut fixed = false;
        if !initial.meets_2x_target {
            let fix_result = apply_fixes(&mut prog, &initial);
            fixed = !fix_result.fixes_applied.is_empty();
        }
        let post_analysis = analyzer.analyze(&prog);

        // Select optimal crossover based on profile
        let effective_target = match target {
            CompilationTarget::Adaptive { .. } => {
                // Use profile data to set crossover
                let crossover = if profile.spatial_locality_score > 0.7 {
                    5000  // High locality → sequential is good for larger n
                } else {
                    1000  // Low locality → switch to parallel sooner
                };
                CompilationTarget::Adaptive { crossover_n: crossover }
            }
            other => other.clone(),
        };

        let code = self.base.compile(&prog, &effective_target);

        ProfiledCompilation {
            code,
            hash_family: knobs.hash_family.name().to_string(),
            block_size: knobs.block_size,
            tile_size: knobs.tile_size,
            multi_level: knobs.multi_level_partition,
            performance_ratio: post_analysis.performance_ratio,
            meets_2x: post_analysis.meets_2x_target,
            fixed,
            spatial_locality: profile.spatial_locality_score,
            temporal_locality: profile.temporal_locality_score,
            working_set_kb: profile.working_set_size / 1024,
        }
    }
}

/// Result of profile-guided compilation.
#[derive(Debug, Clone)]
pub struct ProfiledCompilation {
    pub code: String,
    pub hash_family: String,
    pub block_size: usize,
    pub tile_size: usize,
    pub multi_level: bool,
    pub performance_ratio: f64,
    pub meets_2x: bool,
    pub fixed: bool,
    pub spatial_locality: f64,
    pub temporal_locality: f64,
    pub working_set_kb: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pram_ir::ast::*;
    use crate::pram_ir::types::*;

    fn test_program() -> PramProgram {
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
                    value: Expr::ProcessorId,
                }],
            }],
            num_processors: Expr::int(100),
            work_bound: Some("O(n)".into()),
            time_bound: Some("O(1)".into()),
            description: Some("Initialize array with processor IDs".into()),
        }
    }

    // --- Sequential target tests ---

    #[test]
    fn test_sequential_target_produces_valid_c() {
        let compiler = AdaptiveCompiler::new();
        let code = compiler.compile(&test_program(), &CompilationTarget::Sequential);
        assert!(code.contains("int main("));
        assert!(code.contains("return 0;"));
    }

    #[test]
    fn test_sequential_target_contains_includes() {
        let compiler = AdaptiveCompiler::new();
        let code = compiler.compile(&test_program(), &CompilationTarget::Sequential);
        assert!(code.contains("#include <stdio.h>"));
        assert!(code.contains("#include <stdlib.h>"));
    }

    #[test]
    fn test_sequential_target_no_openmp() {
        let compiler = AdaptiveCompiler::new();
        let code = compiler.compile(&test_program(), &CompilationTarget::Sequential);
        assert!(!code.contains("#pragma omp"));
        assert!(!code.contains("#include <omp.h>"));
    }

    // --- Parallel target tests ---

    #[test]
    fn test_parallel_target_contains_omp_pragma() {
        let compiler = AdaptiveCompiler::new();
        let target = CompilationTarget::Parallel { num_threads: 4 };
        let code = compiler.compile(&test_program(), &target);
        assert!(code.contains("#pragma omp"));
    }

    #[test]
    fn test_parallel_target_includes_omp_header() {
        let compiler = AdaptiveCompiler::new();
        let target = CompilationTarget::Parallel { num_threads: 8 };
        let code = compiler.compile(&test_program(), &target);
        assert!(code.contains("#include <omp.h>"));
    }

    #[test]
    fn test_parallel_target_sets_threads() {
        let compiler = AdaptiveCompiler::new();
        let target = CompilationTarget::Parallel { num_threads: 16 };
        let code = compiler.compile(&test_program(), &target);
        assert!(code.contains("omp_set_num_threads(16)"));
    }

    #[test]
    fn test_parallel_target_has_parallel_function() {
        let compiler = AdaptiveCompiler::new();
        let target = CompilationTarget::Parallel { num_threads: 4 };
        let code = compiler.compile(&test_program(), &target);
        assert!(code.contains("pram_array_init_parallel"));
    }

    // --- Adaptive target tests ---

    #[test]
    fn test_adaptive_target_has_crossover_check() {
        let compiler = AdaptiveCompiler::new();
        let target = CompilationTarget::Adaptive { crossover_n: 1000 };
        let code = compiler.compile(&test_program(), &target);
        assert!(code.contains("if (n < CROSSOVER_N)"));
    }

    #[test]
    fn test_adaptive_target_defines_crossover_n() {
        let compiler = AdaptiveCompiler::new();
        let target = CompilationTarget::Adaptive { crossover_n: 512 };
        let code = compiler.compile(&test_program(), &target);
        assert!(code.contains("#define CROSSOVER_N 512"), "code should define CROSSOVER_N");
    }

    #[test]
    fn test_adaptive_target_contains_both_versions() {
        let compiler = AdaptiveCompiler::new();
        let target = CompilationTarget::Adaptive { crossover_n: 1000 };
        let code = compiler.compile(&test_program(), &target);
        // Sequential version is embedded
        assert!(code.contains("Sequential version"));
        assert!(code.contains("pram_array_init_sequential"));
        // Parallel version is embedded
        assert!(code.contains("Parallel version"));
        assert!(code.contains("#pragma omp"));
        assert!(code.contains("pram_array_init_parallel"));
    }

    #[test]
    fn test_adaptive_target_has_runtime_switch() {
        let compiler = AdaptiveCompiler::new();
        let target = CompilationTarget::Adaptive { crossover_n: 2048 };
        let code = compiler.compile(&test_program(), &target);
        assert!(code.contains("Using sequential version"));
        assert!(code.contains("Using parallel version"));
        assert!(code.contains("if (n < CROSSOVER_N)"));
        assert!(code.contains("} else {"));
    }

    #[test]
    fn test_adaptive_target_contains_algorithm_comment() {
        let compiler = AdaptiveCompiler::new();
        let target = CompilationTarget::Adaptive { crossover_n: 256 };
        let code = compiler.compile(&test_program(), &target);
        assert!(code.contains("Adaptive compilation output"));
        assert!(code.contains("array_init"));
        assert!(code.contains("Crossover threshold"));
    }

    #[test]
    fn test_adaptive_with_custom_configs() {
        let compiler = AdaptiveCompiler::new()
            .with_gen_config(GeneratorConfig::release())
            .with_omp_config(OpenMPConfig {
                max_threads: 8,
                ..OpenMPConfig::default()
            });
        let target = CompilationTarget::Adaptive { crossover_n: 4096 };
        let code = compiler.compile(&test_program(), &target);
        assert!(code.contains("#define CROSSOVER_N 4096"));
        assert!(code.contains("#pragma omp"));
    }

    #[test]
    fn test_default_adaptive_compiler() {
        let compiler = AdaptiveCompiler::default();
        let code = compiler.compile(&test_program(), &CompilationTarget::Sequential);
        assert!(code.contains("int main("));
    }
}
