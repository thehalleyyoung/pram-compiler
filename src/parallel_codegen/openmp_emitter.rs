//! OpenMP code generation for multi-core execution.

use crate::pram_ir::ast::{PramProgram, Stmt, Expr, MemoryModel, BinOp, UnaryOp};

/// OpenMP code generation configuration.
#[derive(Debug, Clone)]
pub struct OpenMPConfig {
    pub max_threads: usize,
    pub schedule_policy: SchedulePolicy,
    pub chunk_size: Option<usize>,
    pub enable_simd: bool,
    pub enable_task_parallelism: bool,
    pub collapse_depth: usize,
    pub cache_line_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SchedulePolicy {
    Static,
    Dynamic,
    Guided,
    Auto,
}

impl SchedulePolicy {
    pub fn to_omp_str(&self) -> &'static str {
        match self {
            SchedulePolicy::Static => "static",
            SchedulePolicy::Dynamic => "dynamic",
            SchedulePolicy::Guided => "guided",
            SchedulePolicy::Auto => "auto",
        }
    }
}

impl Default for OpenMPConfig {
    fn default() -> Self {
        Self {
            max_threads: 0,
            schedule_policy: SchedulePolicy::Guided,
            chunk_size: None,
            enable_simd: true,
            enable_task_parallelism: true,
            collapse_depth: 1,
            cache_line_size: 64,
        }
    }
}

/// OpenMP code emitter.
pub struct OpenMPEmitter {
    config: OpenMPConfig,
    indent: usize,
    output: String,
}

impl OpenMPEmitter {
    pub fn new(config: OpenMPConfig) -> Self {
        Self { config, indent: 0, output: String::new() }
    }

    /// Generate OpenMP-annotated C code from a PRAM program.
    pub fn generate(&mut self, program: &PramProgram) -> String {
        self.output.clear();

        self.emit_line("#include <stdio.h>");
        self.emit_line("#include <stdlib.h>");
        self.emit_line("#include <string.h>");
        self.emit_line("#include <time.h>");
        self.emit_line("#include <omp.h>");
        self.emit_line("");
        self.emit_line("#define CACHE_LINE_SIZE 64");
        self.emit_line("#define ALIGNED_ALLOC(size) aligned_alloc(CACHE_LINE_SIZE, ((size) + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1))");
        self.emit_line("#define MIN(a,b) ((a) < (b) ? (a) : (b))");
        self.emit_line("#define MAX(a,b) ((a) > (b) ? (a) : (b))");
        self.emit_line("");

        if matches!(program.memory_model,
            MemoryModel::CRCWPriority | MemoryModel::CRCWArbitrary | MemoryModel::CRCWCommon) {
            self.emit_crcw_helpers(&program.memory_model);
        }

        self.emit_line(&format!("/* PRAM Algorithm: {} */", program.name));
        self.emit_line(&format!("/* Memory Model: {} */", program.memory_model.name()));
        if let Some(ref work) = program.work_bound {
            self.emit_line(&format!("/* Work: {} */", work));
        }
        self.emit_line("");

        self.emit_main(program);
        self.output.clone()
    }

    fn emit_main(&mut self, program: &PramProgram) {
        self.emit_line(&format!(
            "int pram_{}_parallel(int n) {{",
            program.name.replace('-', "_")
        ));
        self.indent += 1;

        if self.config.max_threads > 0 {
            self.emit_line(&format!("omp_set_num_threads({});", self.config.max_threads));
        }
        self.emit_line("double start_time = omp_get_wtime();");
        self.emit_line("");

        for region in &program.shared_memory {
            self.emit_line(&format!(
                "int* {} = (int*)ALIGNED_ALLOC(sizeof(int) * n);", region.name
            ));
            self.emit_line(&format!("memset({}, 0, sizeof(int) * n);", region.name));
        }
        self.emit_line("");

        self.emit_stmts(&program.body);

        self.emit_line("");
        self.emit_line("double end_time = omp_get_wtime();");
        self.emit_line("printf(\"Parallel time: %.6f s\\n\", end_time - start_time);");

        for region in &program.shared_memory {
            self.emit_line(&format!("free({});", region.name));
        }

        self.emit_line("return 0;");
        self.indent -= 1;
        self.emit_line("}");
    }

    fn emit_stmts(&mut self, stmts: &[Stmt]) {
        for stmt in stmts { self.emit_stmt(stmt); }
    }

    fn emit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::ParallelFor { proc_var, num_procs, body } => {
                self.emit_parallel_for(proc_var, num_procs, body);
            }
            Stmt::SeqFor { var, start, end, body, .. } => {
                let start_s = self.emit_expr(start);
                let end_s = self.emit_expr(end);
                self.emit_line(&format!("for (int {} = {}; {} < {}; {}++) {{",
                    var, start_s, var, end_s, var));
                self.indent += 1;
                self.emit_stmts(body);
                self.indent -= 1;
                self.emit_line("}");
            }
            Stmt::If { condition, then_body, else_body } => {
                let cond_s = self.emit_expr(condition);
                self.emit_line(&format!("if ({}) {{", cond_s));
                self.indent += 1;
                self.emit_stmts(then_body);
                self.indent -= 1;
                if !else_body.is_empty() {
                    self.emit_line("} else {");
                    self.indent += 1;
                    self.emit_stmts(else_body);
                    self.indent -= 1;
                }
                self.emit_line("}");
            }
            Stmt::Assign(var, value) => {
                let val_s = self.emit_expr(value);
                self.emit_line(&format!("{} = {};", var, val_s));
            }
            Stmt::SharedWrite { memory, index, value } => {
                let mem_s = self.emit_expr(memory);
                let idx_s = self.emit_expr(index);
                let val_s = self.emit_expr(value);
                self.emit_line(&format!("{}[{}] = {};", mem_s, idx_s, val_s));
            }
            Stmt::LocalDecl(name, _typ, init) => {
                if let Some(init_expr) = init {
                    let init_s = self.emit_expr(init_expr);
                    self.emit_line(&format!("int {} = {};", name, init_s));
                } else {
                    self.emit_line(&format!("int {};", name));
                }
            }
            Stmt::Barrier => {
                self.emit_line("#pragma omp barrier");
            }
            Stmt::Block(inner) => {
                self.emit_line("{");
                self.indent += 1;
                self.emit_stmts(inner);
                self.indent -= 1;
                self.emit_line("}");
            }
            Stmt::While { condition, body } => {
                let cond_s = self.emit_expr(condition);
                self.emit_line(&format!("while ({}) {{", cond_s));
                self.indent += 1;
                self.emit_stmts(body);
                self.indent -= 1;
                self.emit_line("}");
            }
            _ => {
                self.emit_line("/* unhandled statement */");
            }
        }
    }

    fn emit_parallel_for(&mut self, proc_var: &str, num_procs: &Expr, body: &[Stmt]) {
        let upper_s = self.emit_expr(num_procs);

        let schedule = if let Some(chunk) = self.config.chunk_size {
            format!("schedule({}, {})", self.config.schedule_policy.to_omp_str(), chunk)
        } else {
            format!("schedule({})", self.config.schedule_policy.to_omp_str())
        };

        let simd_clause = if self.config.enable_simd && body.len() <= 3 { " simd" } else { "" };

        self.emit_line(&format!(
            "#pragma omp parallel for {}{} default(none) shared(n)",
            schedule, simd_clause
        ));
        self.emit_line(&format!("for (int {} = 0; {} < {}; {}++) {{",
            proc_var, proc_var, upper_s, proc_var));
        self.indent += 1;
        self.emit_stmts(body);
        self.indent -= 1;
        self.emit_line("}");
    }

    fn emit_crcw_helpers(&mut self, model: &MemoryModel) {
        self.emit_line("/* CRCW conflict resolution helpers */");
        match model {
            MemoryModel::CRCWPriority => {
                self.emit_line("static inline void crcw_priority_write(int* arr, int idx, int val, int pid) {");
                self.indent += 1;
                self.emit_line("#pragma omp critical");
                self.emit_line("{ arr[idx] = val; }");
                self.indent -= 1;
                self.emit_line("}");
            }
            MemoryModel::CRCWCommon => {
                self.emit_line("static inline void crcw_common_write(int* arr, int idx, int val) {");
                self.indent += 1;
                self.emit_line("arr[idx] = val;");
                self.indent -= 1;
                self.emit_line("}");
            }
            _ => {
                self.emit_line("static inline void crcw_arbitrary_write(int* arr, int idx, int val) {");
                self.indent += 1;
                self.emit_line("#pragma omp atomic write");
                self.emit_line("arr[idx] = val;");
                self.indent -= 1;
                self.emit_line("}");
            }
        }
        self.emit_line("");
    }

    fn emit_expr(&self, expr: &Expr) -> String {
        match expr {
            Expr::IntLiteral(n) => n.to_string(),
            Expr::FloatLiteral(f) => format!("{:.6}", f),
            Expr::BoolLiteral(b) => if *b { "1".to_string() } else { "0".to_string() },
            Expr::Variable(name) => name.clone(),
            Expr::BinOp(op, left, right) => {
                let l = self.emit_expr(left);
                let r = self.emit_expr(right);
                let op_str = match op {
                    BinOp::Add => "+", BinOp::Sub => "-", BinOp::Mul => "*",
                    BinOp::Div => "/", BinOp::Mod => "%", BinOp::Lt => "<",
                    BinOp::Le => "<=", BinOp::Gt => ">", BinOp::Ge => ">=",
                    BinOp::Eq => "==", BinOp::Ne => "!=", BinOp::And => "&&",
                    BinOp::Or => "||", BinOp::BitAnd => "&", BinOp::BitOr => "|",
                    BinOp::BitXor => "^", BinOp::Shl => "<<", BinOp::Shr => ">>",
                    BinOp::Min => return format!("MIN({}, {})", l, r),
                    BinOp::Max => return format!("MAX({}, {})", l, r),
                };
                format!("({} {} {})", l, op_str, r)
            }
            Expr::SharedRead(mem, index) => {
                format!("{}[{}]", self.emit_expr(mem), self.emit_expr(index))
            }
            Expr::ArrayIndex(array, index) => {
                format!("{}[{}]", self.emit_expr(array), self.emit_expr(index))
            }
            Expr::ProcessorId => "omp_get_thread_num()".to_string(),
            Expr::NumProcessors => "omp_get_num_threads()".to_string(),
            Expr::FunctionCall(name, args) => {
                let args_s: Vec<String> = args.iter().map(|a| self.emit_expr(a)).collect();
                format!("{}({})", name, args_s.join(", "))
            }
            Expr::UnaryOp(op, operand) => {
                let o = self.emit_expr(operand);
                match op {
                    UnaryOp::Neg => format!("(-{})", o),
                    UnaryOp::Not => format!("(!{})", o),
                    UnaryOp::BitNot => format!("(~{})", o),
                }
            }
            Expr::Cast(e, _) => self.emit_expr(e),
            Expr::Conditional(c, t, f) => {
                format!("({} ? {} : {})", self.emit_expr(c), self.emit_expr(t), self.emit_expr(f))
            }
        }
    }

    fn emit_line(&mut self, line: &str) {
        let indent_str = "    ".repeat(self.indent);
        self.output.push_str(&indent_str);
        self.output.push_str(line);
        self.output.push('\n');
    }
}

/// Generate both sequential and parallel versions.
pub fn generate_dual_target(program: &PramProgram) -> (String, String) {
    use crate::codegen::generator::{CodeGenerator, GeneratorConfig};

    let seq_gen = CodeGenerator::new(GeneratorConfig::default());
    let sequential = seq_gen.generate(program);

    let mut par_gen = OpenMPEmitter::new(OpenMPConfig::default());
    let parallel = par_gen.generate(program);

    (sequential, parallel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openmp_generation() {
        let mut emitter = OpenMPEmitter::new(OpenMPConfig::default());
        let prog = crate::algorithm_library::sorting::bitonic_sort();
        let code = emitter.generate(&prog);
        assert!(code.contains("#include <omp.h>"));
        assert!(code.contains("#pragma omp"));
    }

    #[test]
    fn test_openmp_prefix_sum() {
        let mut emitter = OpenMPEmitter::new(OpenMPConfig::default());
        let prog = crate::algorithm_library::list::prefix_sum();
        let code = emitter.generate(&prog);
        assert!(code.contains("omp"));
    }

    #[test]
    fn test_dual_target() {
        let prog = crate::algorithm_library::list::prefix_sum();
        let (seq, par) = generate_dual_target(&prog);
        assert!(!seq.is_empty());
        assert!(!par.is_empty());
        assert!(par.contains("omp"));
    }

    #[test]
    fn test_schedule_policies() {
        for policy in [SchedulePolicy::Static, SchedulePolicy::Dynamic, SchedulePolicy::Guided] {
            let config = OpenMPConfig { schedule_policy: policy, ..Default::default() };
            let mut emitter = OpenMPEmitter::new(config);
            let prog = crate::algorithm_library::list::prefix_sum();
            let code = emitter.generate(&prog);
            assert!(code.contains(policy.to_omp_str()));
        }
    }
}
