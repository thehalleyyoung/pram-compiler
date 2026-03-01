//! C99 code emitter – converts PRAM IR to syntactically valid C99 source.
//!
//! The `CEmitter` walks a `PramProgram` AST and produces a complete C source
//! string.  Parallel constructs are lowered to sequential for-loops; shared
//! reads/writes become plain array accesses (with optional CRCW-Priority
//! tracking logic).

use std::fmt::Write;

use crate::pram_ir::ast::*;
use crate::pram_ir::types::*;

use super::memory_layout::pram_type_to_c;

// ---------------------------------------------------------------------------
// CEmitter
// ---------------------------------------------------------------------------

/// Emits C99 source code from a PRAM IR AST.
#[derive(Debug)]
pub struct CEmitter {
    /// Accumulated output buffer.
    buf: String,
    /// Current indentation depth (in units of 4 spaces).
    indent: usize,
    /// Whether the program uses CRCW-Priority semantics.
    crcw_priority: bool,
    /// Variable that holds the current processor id inside a parallel_for.
    pid_var: Option<String>,
    /// Number of barrier phases encountered (used for comments).
    barrier_count: usize,
}

impl CEmitter {
    pub fn new() -> Self {
        Self {
            buf: String::with_capacity(4096),
            indent: 0,
            crcw_priority: false,
            pid_var: None,
            barrier_count: 0,
        }
    }

    // -- indentation helpers ------------------------------------------------

    fn indent_str(&self) -> String {
        "    ".repeat(self.indent)
    }

    fn push_indent(&mut self) {
        self.indent += 1;
    }

    fn pop_indent(&mut self) {
        if self.indent > 0 {
            self.indent -= 1;
        }
    }

    fn write_indented(&mut self, line: &str) {
        let indent = self.indent_str();
        writeln!(self.buf, "{}{}", indent, line).unwrap();
    }

    fn write_blank_line(&mut self) {
        self.buf.push('\n');
    }

    // -- expression emission ------------------------------------------------

    /// Convert a PRAM `Expr` to a C expression string.
    pub fn emit_expr(&self, expr: &Expr) -> String {
        match expr {
            Expr::IntLiteral(v) => {
                if *v == i64::MIN {
                    "(-9223372036854775807LL - 1)".to_string()
                } else if *v < 0 {
                    format!("({}LL)", v)
                } else {
                    format!("{}LL", v)
                }
            }
            Expr::FloatLiteral(v) => {
                if v.fract() == 0.0 && v.is_finite() {
                    format!("{:.1}", v)
                } else {
                    format!("{}", v)
                }
            }
            Expr::BoolLiteral(b) => {
                if *b { "true".to_string() } else { "false".to_string() }
            }
            Expr::Variable(name) => name.clone(),
            Expr::ProcessorId => {
                self.pid_var.clone().unwrap_or_else(|| "_pid".to_string())
            }
            Expr::NumProcessors => "_num_procs".to_string(),
            Expr::BinOp(op, lhs, rhs) => {
                let l = self.emit_expr(lhs);
                let r = self.emit_expr(rhs);
                match op {
                    BinOp::Min => format!("PRAM_MIN({}, {})", l, r),
                    BinOp::Max => format!("PRAM_MAX({}, {})", l, r),
                    _ => format!("({} {} {})", l, op.symbol(), r),
                }
            }
            Expr::UnaryOp(op, operand) => {
                let inner = self.emit_expr(operand);
                format!("({}{})", op.symbol(), inner)
            }
            Expr::SharedRead(mem, idx) => {
                let m = self.emit_expr(mem);
                let i = self.emit_expr(idx);
                format!("{}[{}]", m, i)
            }
            Expr::ArrayIndex(arr, idx) => {
                let a = self.emit_expr(arr);
                let i = self.emit_expr(idx);
                format!("{}[{}]", a, i)
            }
            Expr::FunctionCall(name, args) => {
                let arg_strs: Vec<String> = args.iter().map(|a| self.emit_expr(a)).collect();
                format!("{}({})", name, arg_strs.join(", "))
            }
            Expr::Cast(inner, target) => {
                let e = self.emit_expr(inner);
                let ty = pram_type_to_c(target);
                format!("(({})({}))", ty, e)
            }
            Expr::Conditional(cond, then_e, else_e) => {
                let c = self.emit_expr(cond);
                let t = self.emit_expr(then_e);
                let e = self.emit_expr(else_e);
                format!("({} ? {} : {})", c, t, e)
            }
        }
    }

    // -- statement emission -------------------------------------------------

    /// Emit a single PRAM `Stmt` as one or more C statements.
    pub fn emit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::LocalDecl(name, ty, init) => {
                let cty = pram_type_to_c(ty);
                match init {
                    Some(expr) => {
                        let val = self.emit_expr(expr);
                        self.write_indented(&format!("{} {} = {};", cty, name, val));
                    }
                    None => {
                        let def = super::memory_layout::c_default_value(ty);
                        self.write_indented(&format!("{} {} = {};", cty, name, def));
                    }
                }
            }

            Stmt::Assign(name, expr) => {
                let val = self.emit_expr(expr);
                self.write_indented(&format!("{} = {};", name, val));
            }

            Stmt::SharedWrite { memory, index, value } => {
                let m = self.emit_expr(memory);
                let i = self.emit_expr(index);
                let v = self.emit_expr(value);

                if self.crcw_priority {
                    let pid = self.pid_var.clone().unwrap_or_else(|| "_pid".to_string());
                    self.write_indented(&format!(
                        "if ({pid} < _wpid_{m}[{i}]) {{"
                    ));
                    self.push_indent();
                    self.write_indented(&format!("_wpid_{m}[{i}] = {pid};"));
                    self.write_indented(&format!("_stg_{m}[{i}] = {v};"));
                    self.pop_indent();
                    self.write_indented("}");
                } else {
                    self.write_indented(&format!("{}[{}] = {};", m, i, v));
                }
            }

            Stmt::ParallelFor { proc_var, num_procs, body } => {
                let n = self.emit_expr(num_procs);
                let old_pid = self.pid_var.take();
                self.pid_var = Some(proc_var.clone());

                self.write_indented(&format!(
                    "for (int64_t {} = 0; {} < {}; {}++) {{",
                    proc_var, proc_var, n, proc_var
                ));
                self.push_indent();
                for s in body {
                    self.emit_stmt(s);
                }
                self.pop_indent();
                self.write_indented("}");

                self.pid_var = old_pid;
            }

            Stmt::SeqFor { var, start, end, step, body } => {
                let s = self.emit_expr(start);
                let e = self.emit_expr(end);
                let st = step.as_ref().map(|x| self.emit_expr(x)).unwrap_or_else(|| "1LL".to_string());
                self.write_indented(&format!(
                    "for (int64_t {} = {}; {} < {}; {} += {}) {{",
                    var, s, var, e, var, st
                ));
                self.push_indent();
                for stmt in body {
                    self.emit_stmt(stmt);
                }
                self.pop_indent();
                self.write_indented("}");
            }

            Stmt::While { condition, body } => {
                let c = self.emit_expr(condition);
                self.write_indented(&format!("while ({}) {{", c));
                self.push_indent();
                for s in body {
                    self.emit_stmt(s);
                }
                self.pop_indent();
                self.write_indented("}");
            }

            Stmt::If { condition, then_body, else_body } => {
                let c = self.emit_expr(condition);
                self.write_indented(&format!("if ({}) {{", c));
                self.push_indent();
                for s in then_body {
                    self.emit_stmt(s);
                }
                self.pop_indent();

                if !else_body.is_empty() {
                    self.write_indented("} else {");
                    self.push_indent();
                    for s in else_body {
                        self.emit_stmt(s);
                    }
                    self.pop_indent();
                }
                self.write_indented("}");
            }

            Stmt::Barrier => {
                self.write_indented(&format!(
                    "/* ---- barrier (end of phase {}) ---- */",
                    self.barrier_count
                ));
                self.barrier_count += 1;
            }

            Stmt::Block(stmts) => {
                self.write_indented("{");
                self.push_indent();
                for s in stmts {
                    self.emit_stmt(s);
                }
                self.pop_indent();
                self.write_indented("}");
            }

            Stmt::ExprStmt(expr) => {
                let e = self.emit_expr(expr);
                self.write_indented(&format!("{};", e));
            }

            Stmt::Return(opt_expr) => {
                match opt_expr {
                    Some(expr) => {
                        let e = self.emit_expr(expr);
                        self.write_indented(&format!("return {};", e));
                    }
                    None => {
                        self.write_indented("return;");
                    }
                }
            }

            Stmt::AllocShared { name, elem_type, size } => {
                let cty = pram_type_to_c(elem_type);
                let sz = self.emit_expr(size);
                self.write_indented(&format!(
                    "{}* {} = ({}*)pram_calloc({}, sizeof({}));",
                    cty, name, cty, sz, cty
                ));
            }

            Stmt::FreeShared(name) => {
                self.write_indented(&format!("pram_free({});", name));
            }

            Stmt::Nop => {
                self.write_indented("/* nop */");
            }

            Stmt::AtomicCAS { memory, index, expected, desired, result_var } => {
                let m = self.emit_expr(memory);
                let i = self.emit_expr(index);
                let exp = self.emit_expr(expected);
                let des = self.emit_expr(desired);
                self.write_indented(&format!(
                    "if ({}[{}] == {}) {{ {} = 1; {}[{}] = {}; }} else {{ {} = 0; }}",
                    m, i, exp, result_var, m, i, des, result_var
                ));
            }

            Stmt::FetchAdd { memory, index, value, result_var } => {
                let m = self.emit_expr(memory);
                let i = self.emit_expr(index);
                let v = self.emit_expr(value);
                self.write_indented(&format!(
                    "{} = {}[{}]; {}[{}] += {};",
                    result_var, m, i, m, i, v
                ));
            }

            Stmt::Assert(expr, msg) => {
                let e = self.emit_expr(expr);
                self.write_indented(&format!(
                    "if (!({e})) {{ fprintf(stderr, \"Assertion failed: %s\\n\", \"{msg}\"); exit(1); }}"
                ));
            }

            Stmt::Comment(text) => {
                self.write_indented(&format!("/* {} */", text));
            }

            Stmt::PrefixSum { input, output, size, op } => {
                let sz = self.emit_expr(size);
                self.write_indented(&format!("/* Prefix sum ({}) {} -> {} */", op.symbol(), input, output));
                self.write_indented(&format!("{}[0] = {}[0];", output, input));
                self.write_indented(&format!(
                    "for (int64_t _ps_i = 1; _ps_i < {}; _ps_i++) {{",
                    sz
                ));
                self.push_indent();
                match op {
                    BinOp::Add => {
                        self.write_indented(&format!(
                            "{}[_ps_i] = {}[_ps_i - 1] + {}[_ps_i];",
                            output, output, input
                        ));
                    }
                    BinOp::Min => {
                        self.write_indented(&format!(
                            "{}[_ps_i] = PRAM_MIN({}[_ps_i - 1], {}[_ps_i]);",
                            output, output, input
                        ));
                    }
                    BinOp::Max => {
                        self.write_indented(&format!(
                            "{}[_ps_i] = PRAM_MAX({}[_ps_i - 1], {}[_ps_i]);",
                            output, output, input
                        ));
                    }
                    BinOp::Mul => {
                        self.write_indented(&format!(
                            "{}[_ps_i] = {}[_ps_i - 1] * {}[_ps_i];",
                            output, output, input
                        ));
                    }
                    _ => {
                        self.write_indented(&format!(
                            "{}[_ps_i] = {}[_ps_i - 1] {} {}[_ps_i];",
                            output, output, op.symbol(), input
                        ));
                    }
                }
                self.pop_indent();
                self.write_indented("}");
            }
        }
    }

    // -- program emission ---------------------------------------------------

    /// Emit a complete C99 program from a `PramProgram`.
    pub fn emit_program(&mut self, program: &PramProgram) -> String {
        self.buf.clear();
        self.indent = 0;
        self.barrier_count = 0;
        self.crcw_priority = program.memory_model == MemoryModel::CRCWPriority;

        self.write_indented(&format!(
            "/* Generated C99 code for PRAM program: {} */",
            program.name
        ));
        self.write_indented(&format!(
            "/* Memory model: {} */",
            program.memory_model.name()
        ));
        if let Some(ref desc) = program.description {
            self.write_indented(&format!("/* {} */", desc));
        }
        self.write_blank_line();

        self.write_indented("int main(int argc, char* argv[]) {");
        self.push_indent();

        for param in &program.parameters {
            let cty = pram_type_to_c(&param.param_type);
            let def = super::memory_layout::c_default_value(&param.param_type);
            self.write_indented(&format!("{} {} = {};", cty, param.name, def));
        }

        let np = self.emit_expr(&program.num_processors);
        self.write_indented(&format!("int64_t _num_procs = {};", np));
        self.write_blank_line();

        for decl in &program.shared_memory {
            let cty = pram_type_to_c(&decl.elem_type);
            let sz = self.emit_expr(&decl.size);
            self.write_indented(&format!(
                "{}* {} = ({}*)pram_calloc({}, sizeof({}));",
                cty, decl.name, cty, sz, cty
            ));

            if self.crcw_priority {
                self.write_indented(&format!(
                    "int64_t* _wpid_{name} = (int64_t*)pram_calloc({sz}, sizeof(int64_t));",
                    name = decl.name, sz = sz
                ));
                self.write_indented(&format!(
                    "{}* _stg_{name} = ({}*)pram_calloc({sz}, sizeof({}));",
                    cty, cty, cty, name = decl.name, sz = sz
                ));
                self.write_indented(&format!(
                    "for (int64_t _i = 0; _i < {sz}; _i++) {{ _wpid_{name}[_i] = INT64_MAX; }}",
                    name = decl.name, sz = sz
                ));
            }
        }
        self.write_blank_line();

        for stmt in &program.body {
            self.emit_stmt(stmt);
        }
        self.write_blank_line();

        for decl in &program.shared_memory {
            self.write_indented(&format!("pram_free({});", decl.name));
            if self.crcw_priority {
                self.write_indented(&format!("pram_free(_wpid_{});", decl.name));
                self.write_indented(&format!("pram_free(_stg_{});", decl.name));
            }
        }

        self.write_indented("return 0;");
        self.pop_indent();
        self.write_indented("}");

        self.buf.clone()
    }

    /// Return the accumulated output.
    pub fn output(&self) -> &str {
        &self.buf
    }

    // -- debug / profiling / utility helpers --------------------------------

    /// Generate printf-based debug statements for a given PRAM `Stmt`.
    pub fn emit_debug_code(&self, stmt: &Stmt) -> String {
        match stmt {
            Stmt::Assign(name, expr) => {
                let val = self.emit_expr(expr);
                format!(
                    "printf(\"DEBUG assign: {} = %lld\\n\", (long long)({}));\n",
                    name, val
                )
            }
            Stmt::SharedWrite { memory, index, value } => {
                let m = self.emit_expr(memory);
                let i = self.emit_expr(index);
                let v = self.emit_expr(value);
                format!(
                    "printf(\"DEBUG shared_write: {}[%lld] = %lld\\n\", (long long)({}), (long long)({}));\n",
                    m, i, v
                )
            }
            Stmt::ParallelFor { num_procs, .. } => {
                let n = self.emit_expr(num_procs);
                format!(
                    "printf(\"DEBUG entering parallel_for with %lld processors\\n\", (long long)({}));\n",
                    n
                )
            }
            Stmt::If { condition, .. } => {
                let c = self.emit_expr(condition);
                format!(
                    "printf(\"DEBUG branch condition = %d\\n\", (int)({}));\n",
                    c
                )
            }
            _ => {
                format!("/* DEBUG: {:?} */\n", std::mem::discriminant(stmt))
            }
        }
    }

    /// Return C code for timing instrumentation using `clock_gettime`.
    pub fn emit_profiling_hooks() -> String {
        let mut s = String::new();
        s.push_str("#include <time.h>\n\n");
        s.push_str("#define PRAM_TIMER_START(name) \\\n");
        s.push_str("    struct timespec _ts_start_##name, _ts_end_##name; \\\n");
        s.push_str("    clock_gettime(CLOCK_MONOTONIC, &_ts_start_##name)\n\n");
        s.push_str("#define PRAM_TIMER_STOP(name) \\\n");
        s.push_str("    clock_gettime(CLOCK_MONOTONIC, &_ts_end_##name)\n\n");
        s.push_str("static inline double pram_elapsed_ms(struct timespec start, struct timespec end) {\n");
        s.push_str("    double sec  = (double)(end.tv_sec  - start.tv_sec);\n");
        s.push_str("    double nsec = (double)(end.tv_nsec - start.tv_nsec);\n");
        s.push_str("    return sec * 1000.0 + nsec / 1e6;\n");
        s.push_str("}\n\n");
        s.push_str("#define PRAM_TIMER_PRINT(name) \\\n");
        s.push_str("    printf(\"%s: %.3f ms\\n\", #name, \\\n");
        s.push_str("           pram_elapsed_ms(_ts_start_##name, _ts_end_##name))\n");
        s
    }

    /// Return a C if-statement that aborts when `index` is out of `[0, size)`.
    pub fn emit_bounds_check(arr_name: &str, index: &str, size: &str) -> String {
        format!(
            "if (({index}) < 0 || ({index}) >= ({size})) {{\n\
             \x20   fprintf(stderr, \"Bounds check failed: %s[%lld] out of range [0, %lld)\\n\",\n\
             \x20           \"{arr_name}\", (long long)({index}), (long long)({size}));\n\
             \x20   abort();\n\
             }}\n"
        )
    }

    /// Generate a detailed C comment header for a `PramProgram`.
    pub fn emit_comment_header(program: &PramProgram) -> String {
        let mut s = String::new();
        s.push_str("/*\n");
        s.push_str(&format!(" * Program     : {}\n", program.name));
        s.push_str(&format!(" * Memory Model: {}\n", program.memory_model.name()));
        s.push_str(&format!(" * Parameters  : {}\n", program.parameters.len()));
        if !program.shared_memory.is_empty() {
            s.push_str(" * Shared Memory Regions:\n");
            for decl in &program.shared_memory {
                let cty = pram_type_to_c(&decl.elem_type);
                s.push_str(&format!(" *   {} ({})\n", decl.name, cty));
            }
        }
        if let Some(ref w) = program.work_bound {
            s.push_str(&format!(" * Work Bound  : {}\n", w));
        }
        if let Some(ref t) = program.time_bound {
            s.push_str(&format!(" * Time Bound  : {}\n", t));
        }
        if let Some(ref d) = program.description {
            s.push_str(&format!(" * Description : {}\n", d));
        }
        s.push_str(" */\n");
        s
    }

    /// Generate a test `main()` that allocates shared memory, runs the
    /// algorithm, and prints results.
    pub fn emit_test_driver(program: &PramProgram) -> String {
        let mut s = String::new();
        s.push_str("#include <stdio.h>\n");
        s.push_str("#include <stdlib.h>\n");
        s.push_str("#include <stdint.h>\n");
        s.push_str("#include <stdbool.h>\n\n");
        s.push_str(&format!("/* Test driver for {} */\n", program.name));
        s.push_str("int main(void) {\n");

        // Allocate shared memory with sample data
        for decl in &program.shared_memory {
            let cty = pram_type_to_c(&decl.elem_type);
            s.push_str(&format!(
                "    {cty}* {name} = ({cty}*)calloc(64, sizeof({cty}));\n",
                cty = cty,
                name = decl.name,
            ));
            s.push_str(&format!(
                "    for (int i = 0; i < 64; i++) {{ {name}[i] = ({cty})i; }}\n",
                name = decl.name,
                cty = cty,
            ));
        }

        s.push_str(&format!(
            "    printf(\"Running test for {}...\\n\");\n",
            program.name
        ));
        s.push_str(&format!(
            "    /* TODO: invoke {} algorithm here */\n",
            program.name
        ));

        // Print results
        for decl in &program.shared_memory {
            s.push_str(&format!(
                "    printf(\"{}[0..3] = %lld %lld %lld %lld\\n\",\n",
                decl.name
            ));
            s.push_str(&format!(
                "           (long long){n}[0], (long long){n}[1], (long long){n}[2], (long long){n}[3]);\n",
                n = decl.name
            ));
        }

        // Free memory
        for decl in &program.shared_memory {
            s.push_str(&format!("    free({});\n", decl.name));
        }

        s.push_str("    printf(\"Test passed.\\n\");\n");
        s.push_str("    return 0;\n");
        s.push_str("}\n");
        s
    }
}

impl Default for CEmitter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_emitter() -> CEmitter {
        CEmitter::new()
    }

    #[test]
    fn test_emit_int_literal() {
        let e = make_emitter();
        assert_eq!(e.emit_expr(&Expr::IntLiteral(42)), "42LL");
        assert_eq!(e.emit_expr(&Expr::IntLiteral(-1)), "(-1LL)");
        assert_eq!(e.emit_expr(&Expr::IntLiteral(0)), "0LL");
    }

    #[test]
    fn test_emit_float_literal() {
        let e = make_emitter();
        assert_eq!(e.emit_expr(&Expr::FloatLiteral(3.0)), "3.0");
        let s = e.emit_expr(&Expr::FloatLiteral(3.14));
        assert!(s.starts_with("3.14"));
    }

    #[test]
    fn test_emit_bool_literal() {
        let e = make_emitter();
        assert_eq!(e.emit_expr(&Expr::BoolLiteral(true)), "true");
        assert_eq!(e.emit_expr(&Expr::BoolLiteral(false)), "false");
    }

    #[test]
    fn test_emit_variable() {
        let e = make_emitter();
        assert_eq!(e.emit_expr(&Expr::Variable("x".into())), "x");
    }

    #[test]
    fn test_emit_processor_id_default() {
        let e = make_emitter();
        assert_eq!(e.emit_expr(&Expr::ProcessorId), "_pid");
    }

    #[test]
    fn test_emit_num_processors() {
        let e = make_emitter();
        assert_eq!(e.emit_expr(&Expr::NumProcessors), "_num_procs");
    }

    #[test]
    fn test_emit_binop() {
        let e = make_emitter();
        let expr = Expr::binop(BinOp::Add, Expr::var("x"), Expr::int(1));
        assert_eq!(e.emit_expr(&expr), "(x + 1LL)");
    }

    #[test]
    fn test_emit_binop_min_max() {
        let e = make_emitter();
        let min_expr = Expr::binop(BinOp::Min, Expr::var("a"), Expr::var("b"));
        assert_eq!(e.emit_expr(&min_expr), "PRAM_MIN(a, b)");
        let max_expr = Expr::binop(BinOp::Max, Expr::var("a"), Expr::var("b"));
        assert_eq!(e.emit_expr(&max_expr), "PRAM_MAX(a, b)");
    }

    #[test]
    fn test_emit_unary() {
        let e = make_emitter();
        let neg = Expr::unop(UnaryOp::Neg, Expr::var("x"));
        assert_eq!(e.emit_expr(&neg), "(-x)");
        let not = Expr::unop(UnaryOp::Not, Expr::var("flag"));
        assert_eq!(e.emit_expr(&not), "(!flag)");
    }

    #[test]
    fn test_emit_shared_read() {
        let e = make_emitter();
        let sr = Expr::shared_read(Expr::var("A"), Expr::var("i"));
        assert_eq!(e.emit_expr(&sr), "A[i]");
    }

    #[test]
    fn test_emit_array_index() {
        let e = make_emitter();
        let ai = Expr::array_index(Expr::var("arr"), Expr::int(5));
        assert_eq!(e.emit_expr(&ai), "arr[5LL]");
    }

    #[test]
    fn test_emit_function_call() {
        let e = make_emitter();
        let call = Expr::FunctionCall("printf".into(), vec![Expr::var("fmt")]);
        assert_eq!(e.emit_expr(&call), "printf(fmt)");
    }

    #[test]
    fn test_emit_conditional() {
        let e = make_emitter();
        let cond = Expr::Conditional(
            Box::new(Expr::var("flag")),
            Box::new(Expr::int(1)),
            Box::new(Expr::int(0)),
        );
        assert_eq!(e.emit_expr(&cond), "(flag ? 1LL : 0LL)");
    }

    #[test]
    fn test_emit_assign_stmt() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::Assign("x".into(), Expr::int(42)));
        assert!(e.output().contains("x = 42LL;"));
    }

    #[test]
    fn test_emit_local_decl() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::LocalDecl("x".into(), PramType::Int64, Some(Expr::int(10))));
        assert!(e.output().contains("int64_t x = 10LL;"));
    }

    #[test]
    fn test_emit_local_decl_no_init() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::LocalDecl("y".into(), PramType::Float64, None));
        assert!(e.output().contains("double y = 0.0;"));
    }

    #[test]
    fn test_emit_shared_write() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::SharedWrite {
            memory: Expr::var("A"),
            index: Expr::var("i"),
            value: Expr::int(1),
        });
        assert!(e.output().contains("A[i] = 1LL;"));
    }

    #[test]
    fn test_emit_parallel_for() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::ParallelFor {
            proc_var: "pid".into(),
            num_procs: Expr::int(8),
            body: vec![Stmt::Assign("x".into(), Expr::ProcessorId)],
        });
        let out = e.output();
        assert!(out.contains("for (int64_t pid = 0; pid < 8LL; pid++)"));
        assert!(out.contains("x = pid;"));
    }

    #[test]
    fn test_emit_seq_for() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::SeqFor {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::int(10),
            step: Some(Expr::int(2)),
            body: vec![Stmt::Nop],
        });
        let out = e.output();
        assert!(out.contains("for (int64_t i = 0LL; i < 10LL; i += 2LL)"));
    }

    #[test]
    fn test_emit_while() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::While {
            condition: Expr::var("running"),
            body: vec![Stmt::Assign("running".into(), Expr::BoolLiteral(false))],
        });
        let out = e.output();
        assert!(out.contains("while (running)"));
        assert!(out.contains("running = false;"));
    }

    #[test]
    fn test_emit_if_else() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::If {
            condition: Expr::var("flag"),
            then_body: vec![Stmt::Assign("x".into(), Expr::int(1))],
            else_body: vec![Stmt::Assign("x".into(), Expr::int(0))],
        });
        let out = e.output();
        assert!(out.contains("if (flag)"));
        assert!(out.contains("} else {"));
    }

    #[test]
    fn test_emit_if_no_else() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::If {
            condition: Expr::var("flag"),
            then_body: vec![Stmt::Assign("x".into(), Expr::int(1))],
            else_body: vec![],
        });
        let out = e.output();
        assert!(out.contains("if (flag)"));
        assert!(!out.contains("else"));
    }

    #[test]
    fn test_emit_barrier() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::Barrier);
        assert!(e.output().contains("barrier (end of phase 0)"));
        e.emit_stmt(&Stmt::Barrier);
        assert!(e.output().contains("barrier (end of phase 1)"));
    }

    #[test]
    fn test_emit_alloc_shared() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::AllocShared {
            name: "buf".into(),
            elem_type: PramType::Int32,
            size: Expr::int(100),
        });
        let out = e.output();
        assert!(out.contains("int32_t* buf"));
        assert!(out.contains("pram_calloc(100LL"));
    }

    #[test]
    fn test_emit_free_shared() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::FreeShared("buf".into()));
        assert!(e.output().contains("pram_free(buf);"));
    }

    #[test]
    fn test_emit_assert() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::Assert(
            Expr::binop(BinOp::Gt, Expr::var("n"), Expr::int(0)),
            "n must be positive".into(),
        ));
        let out = e.output();
        assert!(out.contains("Assertion failed"));
        assert!(out.contains("n must be positive"));
    }

    #[test]
    fn test_emit_comment() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::Comment("Phase 1: scatter".into()));
        assert!(e.output().contains("/* Phase 1: scatter */"));
    }

    #[test]
    fn test_emit_return() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::Return(Some(Expr::int(0))));
        assert!(e.output().contains("return 0LL;"));
    }

    #[test]
    fn test_emit_return_void() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::Return(None));
        assert!(e.output().contains("return;"));
    }

    #[test]
    fn test_emit_nop() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::Nop);
        assert!(e.output().contains("/* nop */"));
    }

    #[test]
    fn test_emit_atomic_cas() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::AtomicCAS {
            memory: Expr::var("A"),
            index: Expr::int(0),
            expected: Expr::int(1),
            desired: Expr::int(2),
            result_var: "ok".into(),
        });
        let out = e.output();
        assert!(out.contains("A[0LL] == 1LL"));
        assert!(out.contains("ok = 1"));
    }

    #[test]
    fn test_emit_fetch_add() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::FetchAdd {
            memory: Expr::var("cnt"),
            index: Expr::int(0),
            value: Expr::int(1),
            result_var: "old".into(),
        });
        let out = e.output();
        assert!(out.contains("old = cnt[0LL]"));
        assert!(out.contains("cnt[0LL] += 1LL"));
    }

    #[test]
    fn test_emit_prefix_sum() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::PrefixSum {
            input: "in".into(),
            output: "out".into(),
            size: Expr::int(10),
            op: BinOp::Add,
        });
        let out = e.output();
        assert!(out.contains("out[0] = in[0]"));
        assert!(out.contains("out[_ps_i] = out[_ps_i - 1] + in[_ps_i]"));
    }

    #[test]
    fn test_emit_simple_program() {
        let program = PramProgram {
            name: "test_prog".into(),
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
            body: vec![
                Stmt::ParallelFor {
                    proc_var: "pid".into(),
                    num_procs: Expr::int(100),
                    body: vec![Stmt::SharedWrite {
                        memory: Expr::var("A"),
                        index: Expr::ProcessorId,
                        value: Expr::ProcessorId,
                    }],
                },
            ],
            num_processors: Expr::int(100),
            work_bound: None,
            time_bound: None,
            description: Some("Test program".into()),
        };

        let mut emitter = CEmitter::new();
        let code = emitter.emit_program(&program);

        assert!(code.contains("int main("));
        assert!(code.contains("int64_t* A"));
        assert!(code.contains("for (int64_t pid = 0;"));
        assert!(code.contains("A[pid] = pid;"));
        assert!(code.contains("pram_free(A)"));
        assert!(code.contains("return 0;"));
    }

    #[test]
    fn test_emit_crcw_priority_program() {
        let program = PramProgram {
            name: "crcw_test".into(),
            memory_model: MemoryModel::CRCWPriority,
            parameters: vec![],
            shared_memory: vec![SharedMemoryDecl {
                name: "M".into(),
                elem_type: PramType::Int64,
                size: Expr::int(10),
            }],
            body: vec![
                Stmt::ParallelFor {
                    proc_var: "p".into(),
                    num_procs: Expr::int(10),
                    body: vec![Stmt::SharedWrite {
                        memory: Expr::var("M"),
                        index: Expr::int(0),
                        value: Expr::ProcessorId,
                    }],
                },
            ],
            num_processors: Expr::int(10),
            work_bound: None,
            time_bound: None,
            description: None,
        };

        let mut emitter = CEmitter::new();
        let code = emitter.emit_program(&program);

        assert!(code.contains("_wpid_M"));
        assert!(code.contains("_stg_M"));
        assert!(code.contains("INT64_MAX"));
        assert!(code.contains("if (p < _wpid_M[0LL])"));
    }

    #[test]
    fn test_emit_nested_loops() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::SeqFor {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::int(4),
            step: None,
            body: vec![Stmt::SeqFor {
                var: "j".into(),
                start: Expr::int(0),
                end: Expr::int(4),
                step: None,
                body: vec![Stmt::Assign("x".into(), Expr::binop(
                    BinOp::Add,
                    Expr::var("i"),
                    Expr::var("j"),
                ))],
            }],
        });
        let out = e.output();
        assert!(out.contains("for (int64_t i"));
        assert!(out.contains("for (int64_t j"));
        assert!(out.contains("x = (i + j);"));
    }

    #[test]
    fn test_emit_block() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::Block(vec![
            Stmt::LocalDecl("a".into(), PramType::Int32, Some(Expr::int(1))),
            Stmt::LocalDecl("b".into(), PramType::Int32, Some(Expr::int(2))),
        ]));
        let out = e.output();
        assert!(out.contains("{"));
        assert!(out.contains("int32_t a = 1LL;"));
        assert!(out.contains("int32_t b = 2LL;"));
        assert!(out.contains("}"));
    }

    #[test]
    fn test_emit_expr_stmt() {
        let mut e = make_emitter();
        e.emit_stmt(&Stmt::ExprStmt(Expr::FunctionCall(
            "printf".into(),
            vec![Expr::var("msg")],
        )));
        assert!(e.output().contains("printf(msg);"));
    }

    #[test]
    fn test_emit_cast() {
        let e = make_emitter();
        let cast = Expr::Cast(Box::new(Expr::var("x")), PramType::Float64);
        let result = e.emit_expr(&cast);
        assert!(result.contains("double"));
        assert!(result.contains("x"));
    }

    // -- new tests for added methods ----------------------------------------

    fn make_test_program() -> PramProgram {
        PramProgram {
            name: "TestAlgo".into(),
            memory_model: MemoryModel::CREW,
            parameters: vec![
                Parameter { name: "n".into(), param_type: PramType::Int64 },
            ],
            shared_memory: vec![
                SharedMemoryDecl {
                    name: "A".into(),
                    elem_type: PramType::Int64,
                    size: Expr::var("n"),
                },
            ],
            body: vec![Stmt::Nop],
            num_processors: Expr::var("n"),
            work_bound: Some("O(n)".into()),
            time_bound: Some("O(log n)".into()),
            description: Some("A test algorithm".into()),
        }
    }

    #[test]
    fn test_emit_debug_code_assign() {
        let e = make_emitter();
        let stmt = Stmt::Assign("x".into(), Expr::int(42));
        let dbg = e.emit_debug_code(&stmt);
        assert!(dbg.contains("DEBUG assign"));
        assert!(dbg.contains("x"));
        assert!(dbg.contains("42LL"));
    }

    #[test]
    fn test_emit_debug_code_shared_write() {
        let e = make_emitter();
        let stmt = Stmt::SharedWrite {
            memory: Expr::var("A"),
            index: Expr::var("i"),
            value: Expr::int(7),
        };
        let dbg = e.emit_debug_code(&stmt);
        assert!(dbg.contains("DEBUG shared_write"));
        assert!(dbg.contains("A["));
    }

    #[test]
    fn test_emit_debug_code_parallel_for() {
        let e = make_emitter();
        let stmt = Stmt::ParallelFor {
            proc_var: "pid".into(),
            num_procs: Expr::int(16),
            body: vec![Stmt::Nop],
        };
        let dbg = e.emit_debug_code(&stmt);
        assert!(dbg.contains("parallel_for"));
        assert!(dbg.contains("16LL"));
    }

    #[test]
    fn test_emit_debug_code_if() {
        let e = make_emitter();
        let stmt = Stmt::If {
            condition: Expr::var("flag"),
            then_body: vec![],
            else_body: vec![],
        };
        let dbg = e.emit_debug_code(&stmt);
        assert!(dbg.contains("branch condition"));
        assert!(dbg.contains("flag"));
    }

    #[test]
    fn test_emit_debug_code_other() {
        let e = make_emitter();
        let dbg = e.emit_debug_code(&Stmt::Nop);
        assert!(dbg.contains("DEBUG"));
    }

    #[test]
    fn test_emit_profiling_hooks() {
        let hooks = CEmitter::emit_profiling_hooks();
        assert!(hooks.contains("PRAM_TIMER_START"));
        assert!(hooks.contains("PRAM_TIMER_STOP"));
        assert!(hooks.contains("PRAM_TIMER_PRINT"));
        assert!(hooks.contains("pram_elapsed_ms"));
        assert!(hooks.contains("clock_gettime"));
    }

    #[test]
    fn test_emit_bounds_check() {
        let bc = CEmitter::emit_bounds_check("arr", "idx", "sz");
        assert!(bc.contains("idx"));
        assert!(bc.contains("sz"));
        assert!(bc.contains("abort()"));
        assert!(bc.contains("Bounds check failed"));
        assert!(bc.contains("arr"));
    }

    #[test]
    fn test_emit_comment_header() {
        let prog = make_test_program();
        let header = CEmitter::emit_comment_header(&prog);
        assert!(header.contains("TestAlgo"));
        assert!(header.contains("CREW"));
        assert!(header.contains("Parameters  : 1"));
        assert!(header.contains("A (int64_t)"));
        assert!(header.contains("O(n)"));
        assert!(header.contains("O(log n)"));
        assert!(header.contains("A test algorithm"));
    }

    #[test]
    fn test_emit_comment_header_no_optionals() {
        let prog = PramProgram {
            name: "Minimal".into(),
            memory_model: MemoryModel::EREW,
            parameters: vec![],
            shared_memory: vec![],
            body: vec![],
            num_processors: Expr::int(1),
            work_bound: None,
            time_bound: None,
            description: None,
        };
        let header = CEmitter::emit_comment_header(&prog);
        assert!(header.contains("Minimal"));
        assert!(header.contains("EREW"));
        assert!(!header.contains("Work Bound"));
        assert!(!header.contains("Time Bound"));
        assert!(!header.contains("Description"));
    }

    #[test]
    fn test_emit_test_driver() {
        let prog = make_test_program();
        let driver = CEmitter::emit_test_driver(&prog);
        assert!(driver.contains("int main(void)"));
        assert!(driver.contains("calloc"));
        assert!(driver.contains("free(A)"));
        assert!(driver.contains("TestAlgo"));
        assert!(driver.contains("int64_t"));
        assert!(driver.contains("Test passed"));
    }

    #[test]
    fn test_emit_test_driver_multiple_shared() {
        let prog = PramProgram {
            name: "Multi".into(),
            memory_model: MemoryModel::CRCWCommon,
            parameters: vec![],
            shared_memory: vec![
                SharedMemoryDecl { name: "X".into(), elem_type: PramType::Int32, size: Expr::int(10) },
                SharedMemoryDecl { name: "Y".into(), elem_type: PramType::Float64, size: Expr::int(20) },
            ],
            body: vec![],
            num_processors: Expr::int(4),
            work_bound: None,
            time_bound: None,
            description: None,
        };
        let driver = CEmitter::emit_test_driver(&prog);
        assert!(driver.contains("int32_t* X"));
        assert!(driver.contains("double* Y"));
        assert!(driver.contains("free(X)"));
        assert!(driver.contains("free(Y)"));
    }
}
