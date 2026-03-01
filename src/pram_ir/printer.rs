//! Pretty-printer for PRAM IR programs.
//!
//! Converts a [`PramProgram`] AST back to the textual DSL format.
//! The printer emits minimal parentheses based on operator precedence.

use super::ast::*;
use super::types::PramType;
use std::fmt::Write;

// ---------------------------------------------------------------------------
// PrettyPrinterConfig
// ---------------------------------------------------------------------------

/// Configuration for the pretty printer.
#[derive(Debug, Clone)]
pub struct PrettyPrinterConfig {
    /// Number of spaces per indentation level.
    pub indent_size: usize,
    /// Maximum line width (advisory, not enforced for all constructs).
    pub max_line_width: usize,
    /// Whether to show type annotations in output.
    pub show_types: bool,
}

impl Default for PrettyPrinterConfig {
    fn default() -> Self {
        Self {
            indent_size: 4,
            max_line_width: 100,
            show_types: true,
        }
    }
}

// ---------------------------------------------------------------------------
// PramPrinter
// ---------------------------------------------------------------------------

/// Pretty-prints a PRAM IR AST to the textual DSL format.
pub struct PramPrinter {
    indent: usize,
    indent_str: String,
    buf: String,
}

impl PramPrinter {
    /// Create a new printer that uses `indent_width` spaces per level.
    pub fn new(indent_width: usize) -> Self {
        Self {
            indent: 0,
            indent_str: " ".repeat(indent_width),
            buf: String::new(),
        }
    }

    /// Print a complete program, returning the resulting string.
    pub fn print_program(program: &PramProgram) -> String {
        let mut p = Self::new(4);
        p.emit_program(program);
        p.buf
    }

    // -- internals ---------------------------------------------------------

    fn push_indent(&mut self) {
        for _ in 0..self.indent {
            self.buf.push_str(&self.indent_str);
        }
    }

    fn emit_program(&mut self, prog: &PramProgram) {
        write!(self.buf, "algorithm {}(", prog.name).unwrap();
        for (i, p) in prog.parameters.iter().enumerate() {
            if i > 0 {
                self.buf.push_str(", ");
            }
            write!(self.buf, "{}: {}", p.name, Self::type_str(&p.param_type)).unwrap();
        }
        write!(self.buf, ") model {} {{\n", Self::model_str(prog.memory_model)).unwrap();

        self.indent += 1;

        // Shared memory declarations.
        for s in &prog.shared_memory {
            self.push_indent();
            write!(
                self.buf,
                "shared {}: {}[{}];\n",
                s.name,
                Self::type_str(&s.elem_type),
                Self::expr_str(&s.size),
            )
            .unwrap();
        }

        // Processors declaration.
        self.push_indent();
        write!(self.buf, "processors = {};\n", Self::expr_str(&prog.num_processors)).unwrap();

        // Body.
        for stmt in &prog.body {
            self.emit_stmt(stmt);
        }

        self.indent -= 1;
        self.buf.push_str("}\n");
    }

    pub(crate) fn emit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::LocalDecl(name, ty, init) => {
                self.push_indent();
                write!(self.buf, "let {}: {}", name, Self::type_str(ty)).unwrap();
                if let Some(e) = init {
                    write!(self.buf, " = {}", Self::expr_str(e)).unwrap();
                }
                self.buf.push_str(";\n");
            }
            Stmt::Assign(name, expr) => {
                self.push_indent();
                write!(self.buf, "{} = {};\n", name, Self::expr_str(expr)).unwrap();
            }
            Stmt::SharedWrite { memory, index, value } => {
                self.push_indent();
                write!(
                    self.buf,
                    "shared_write({}, {}, {});\n",
                    Self::expr_str(memory),
                    Self::expr_str(index),
                    Self::expr_str(value),
                )
                .unwrap();
            }
            Stmt::ParallelFor { proc_var, num_procs, body } => {
                self.push_indent();
                write!(
                    self.buf,
                    "parallel_for {} in 0..{} {{\n",
                    proc_var,
                    Self::expr_str(num_procs),
                )
                .unwrap();
                self.indent += 1;
                for s in body {
                    self.emit_stmt(s);
                }
                self.indent -= 1;
                self.push_indent();
                self.buf.push_str("}\n");
            }
            Stmt::SeqFor { var, start, end, step, body } => {
                self.push_indent();
                write!(
                    self.buf,
                    "for {} in {}..{}",
                    var,
                    Self::expr_str(start),
                    Self::expr_str(end),
                )
                .unwrap();
                if let Some(s) = step {
                    write!(self.buf, " step {}", Self::expr_str(s)).unwrap();
                }
                self.buf.push_str(" {\n");
                self.indent += 1;
                for s in body {
                    self.emit_stmt(s);
                }
                self.indent -= 1;
                self.push_indent();
                self.buf.push_str("}\n");
            }
            Stmt::While { condition, body } => {
                self.push_indent();
                write!(self.buf, "while {} {{\n", Self::expr_str(condition)).unwrap();
                self.indent += 1;
                for s in body {
                    self.emit_stmt(s);
                }
                self.indent -= 1;
                self.push_indent();
                self.buf.push_str("}\n");
            }
            Stmt::If { condition, then_body, else_body } => {
                self.push_indent();
                write!(self.buf, "if {} {{\n", Self::expr_str(condition)).unwrap();
                self.indent += 1;
                for s in then_body {
                    self.emit_stmt(s);
                }
                self.indent -= 1;
                if !else_body.is_empty() {
                    self.push_indent();
                    self.buf.push_str("} else {\n");
                    self.indent += 1;
                    for s in else_body {
                        self.emit_stmt(s);
                    }
                    self.indent -= 1;
                }
                self.push_indent();
                self.buf.push_str("}\n");
            }
            Stmt::Barrier => {
                self.push_indent();
                self.buf.push_str("barrier;\n");
            }
            Stmt::Block(stmts) => {
                self.push_indent();
                self.buf.push_str("{\n");
                self.indent += 1;
                for s in stmts {
                    self.emit_stmt(s);
                }
                self.indent -= 1;
                self.push_indent();
                self.buf.push_str("}\n");
            }
            Stmt::ExprStmt(expr) => {
                self.push_indent();
                write!(self.buf, "{};\n", Self::expr_str(expr)).unwrap();
            }
            Stmt::Return(val) => {
                self.push_indent();
                if let Some(e) = val {
                    write!(self.buf, "return {};\n", Self::expr_str(e)).unwrap();
                } else {
                    self.buf.push_str("return;\n");
                }
            }
            Stmt::AllocShared { name, elem_type, size } => {
                self.push_indent();
                write!(
                    self.buf,
                    "shared {}: {}[{}];\n",
                    name,
                    Self::type_str(elem_type),
                    Self::expr_str(size),
                )
                .unwrap();
            }
            Stmt::FreeShared(name) => {
                self.push_indent();
                write!(self.buf, "free_shared {};\n", name).unwrap();
            }
            Stmt::Nop => {
                self.push_indent();
                self.buf.push_str("nop;\n");
            }
            Stmt::AtomicCAS { memory, index, expected, desired, result_var } => {
                self.push_indent();
                write!(
                    self.buf,
                    "{} = atomic_cas({}, {}, {}, {});\n",
                    result_var,
                    Self::expr_str(memory),
                    Self::expr_str(index),
                    Self::expr_str(expected),
                    Self::expr_str(desired),
                )
                .unwrap();
            }
            Stmt::FetchAdd { memory, index, value, result_var } => {
                self.push_indent();
                write!(
                    self.buf,
                    "{} = fetch_add({}, {}, {});\n",
                    result_var,
                    Self::expr_str(memory),
                    Self::expr_str(index),
                    Self::expr_str(value),
                )
                .unwrap();
            }
            Stmt::Assert(expr, msg) => {
                self.push_indent();
                write!(self.buf, "assert({}, \"{}\");\n", Self::expr_str(expr), msg).unwrap();
            }
            Stmt::Comment(text) => {
                self.push_indent();
                write!(self.buf, "// {}\n", text).unwrap();
            }
            Stmt::PrefixSum { input, output, size, op } => {
                self.push_indent();
                write!(
                    self.buf,
                    "prefix_sum({}, {}, {}, {});\n",
                    input,
                    output,
                    Self::expr_str(size),
                    op.symbol(),
                )
                .unwrap();
            }
        }
    }

    // -- expression printing -----------------------------------------------

    /// Convert an expression to a string with minimal parentheses.
    pub fn expr_str(expr: &Expr) -> String {
        Self::expr_to_string(expr, 0)
    }

    /// Recursive helper.  `parent_prec` is the precedence of the enclosing
    /// binary operator (0 = top-level, no parens needed).
    fn expr_to_string(expr: &Expr, parent_prec: u8) -> String {
        match expr {
            Expr::IntLiteral(v) => v.to_string(),
            Expr::FloatLiteral(v) => {
                let s = format!("{}", v);
                if s.contains('.') { s } else { format!("{}.0", s) }
            }
            Expr::BoolLiteral(b) => b.to_string(),
            Expr::Variable(name) => name.clone(),
            Expr::ProcessorId => "pid".to_string(),
            Expr::NumProcessors => "num_processors".to_string(),

            Expr::BinOp(op, left, right) => {
                // min / max are printed as function calls
                if matches!(op, BinOp::Min | BinOp::Max) {
                    return format!(
                        "{}({}, {})",
                        op.symbol(),
                        Self::expr_to_string(left, 0),
                        Self::expr_to_string(right, 0),
                    );
                }

                let prec = op.precedence();
                let left_s = Self::expr_to_string(left, prec);
                // For the right operand of left-assoc ops, use prec+1 to
                // force parens when precedence is equal (e.g. a - (b - c)).
                let right_s = Self::expr_to_string(right, prec + 1);

                let inner = format!("{} {} {}", left_s, op.symbol(), right_s);
                if prec < parent_prec {
                    format!("({})", inner)
                } else {
                    inner
                }
            }

            Expr::UnaryOp(op, operand) => {
                let s = Self::expr_to_string(operand, u8::MAX);
                format!("{}{}", op.symbol(), s)
            }

            Expr::SharedRead(mem, idx) => {
                format!(
                    "shared_read({}, {})",
                    Self::expr_to_string(mem, 0),
                    Self::expr_to_string(idx, 0),
                )
            }

            Expr::ArrayIndex(arr, idx) => {
                format!(
                    "{}[{}]",
                    Self::expr_to_string(arr, 0),
                    Self::expr_to_string(idx, 0),
                )
            }

            Expr::FunctionCall(name, args) => {
                let arg_strs: Vec<_> =
                    args.iter().map(|a| Self::expr_to_string(a, 0)).collect();
                format!("{}({})", name, arg_strs.join(", "))
            }

            Expr::Cast(inner, ty) => {
                format!(
                    "({} as {})",
                    Self::expr_to_string(inner, 0),
                    Self::type_str(ty),
                )
            }

            Expr::Conditional(c, t, e) => {
                format!(
                    "({} ? {} : {})",
                    Self::expr_to_string(c, 0),
                    Self::expr_to_string(t, 0),
                    Self::expr_to_string(e, 0),
                )
            }
        }
    }

    // -- helpers -----------------------------------------------------------

    fn model_str(m: MemoryModel) -> &'static str {
        match m {
            MemoryModel::EREW => "EREW",
            MemoryModel::CREW => "CREW",
            MemoryModel::CRCWPriority => "CRCW_Priority",
            MemoryModel::CRCWArbitrary => "CRCW_Arbitrary",
            MemoryModel::CRCWCommon => "CRCW_Common",
        }
    }

    fn type_str(ty: &PramType) -> String {
        match ty {
            PramType::Int64 => "i64".into(),
            PramType::Int32 => "i32".into(),
            PramType::Float64 => "f64".into(),
            PramType::Float32 => "f32".into(),
            PramType::Bool => "bool".into(),
            PramType::SharedMemory(inner) => format!("shared<{}>", Self::type_str(inner)),
            PramType::Array(inner, size) => format!("[{}; {}]", Self::type_str(inner), size),
            PramType::ProcessorId => "pid".into(),
            PramType::SharedRef(inner) => format!("&shared {}", Self::type_str(inner)),
            PramType::Tuple(types) => {
                let parts: Vec<_> = types.iter().map(|t| Self::type_str(t)).collect();
                format!("({})", parts.join(", "))
            }
            PramType::Unit => "()".into(),
            PramType::Struct(name, _) => format!("struct {}", name),
        }
    }
}

// ---------------------------------------------------------------------------
// Additional print functions
// ---------------------------------------------------------------------------

/// Print a program in a compact single-line format (no newlines, minimal spacing).
pub fn print_compact(program: &PramProgram) -> String {
    let mut buf = String::new();
    write!(buf, "algorithm {}(", program.name).unwrap();
    for (i, p) in program.parameters.iter().enumerate() {
        if i > 0 {
            buf.push_str(", ");
        }
        write!(buf, "{}: {}", p.name, p.param_type).unwrap();
    }
    write!(buf, ") model {} {{ ", PramPrinter::model_str(program.memory_model)).unwrap();
    for s in &program.shared_memory {
        write!(buf, "shared {}: {}[{}]; ", s.name, s.elem_type, PramPrinter::expr_str(&s.size)).unwrap();
    }
    write!(buf, "processors = {}; ", PramPrinter::expr_str(&program.num_processors)).unwrap();
    for stmt in &program.body {
        compact_stmt(&mut buf, stmt);
    }
    buf.push('}');
    buf
}

fn compact_stmt(buf: &mut String, stmt: &Stmt) {
    match stmt {
        Stmt::Assign(name, expr) => {
            write!(buf, "{} = {}; ", name, PramPrinter::expr_str(expr)).unwrap();
        }
        Stmt::LocalDecl(name, ty, init) => {
            write!(buf, "let {}: {}", name, ty).unwrap();
            if let Some(e) = init {
                write!(buf, " = {}", PramPrinter::expr_str(e)).unwrap();
            }
            buf.push_str("; ");
        }
        Stmt::SharedWrite { memory, index, value } => {
            write!(
                buf, "shared_write({}, {}, {}); ",
                PramPrinter::expr_str(memory),
                PramPrinter::expr_str(index),
                PramPrinter::expr_str(value),
            ).unwrap();
        }
        Stmt::ParallelFor { proc_var, num_procs, body } => {
            write!(buf, "parallel_for {} in 0..{} {{ ", proc_var, PramPrinter::expr_str(num_procs)).unwrap();
            for s in body {
                compact_stmt(buf, s);
            }
            buf.push_str("} ");
        }
        Stmt::Barrier => buf.push_str("barrier; "),
        Stmt::Nop => buf.push_str("nop; "),
        Stmt::Return(Some(e)) => {
            write!(buf, "return {}; ", PramPrinter::expr_str(e)).unwrap();
        }
        Stmt::Return(None) => buf.push_str("return; "),
        Stmt::If { condition, then_body, else_body } => {
            write!(buf, "if {} {{ ", PramPrinter::expr_str(condition)).unwrap();
            for s in then_body {
                compact_stmt(buf, s);
            }
            if !else_body.is_empty() {
                buf.push_str("} else { ");
                for s in else_body {
                    compact_stmt(buf, s);
                }
            }
            buf.push_str("} ");
        }
        Stmt::While { condition, body } => {
            write!(buf, "while {} {{ ", PramPrinter::expr_str(condition)).unwrap();
            for s in body {
                compact_stmt(buf, s);
            }
            buf.push_str("} ");
        }
        Stmt::Block(stmts) => {
            buf.push_str("{ ");
            for s in stmts {
                compact_stmt(buf, s);
            }
            buf.push_str("} ");
        }
        Stmt::Comment(text) => {
            write!(buf, "/* {} */ ", text).unwrap();
        }
        other => {
            // Fallback: use the standard printer for remaining statement types
            let mut p = PramPrinter::new(0);
            p.emit_stmt(other);
            buf.push_str(p.buf.trim());
            buf.push(' ');
        }
    }
}

/// Print a program with complexity annotations as comments.
pub fn print_annotated(program: &PramProgram) -> String {
    let mut buf = PramPrinter::print_program(program);

    // Add summary annotations at the end.
    let total_stmts = program.total_stmts();
    let parallel_steps = program.parallel_step_count();
    let shared_regions = program.shared_region_names();
    let uses_cw = program.uses_concurrent_writes();
    let uses_cr = program.uses_concurrent_reads();

    write!(buf, "\n// --- Annotations ---\n").unwrap();
    write!(buf, "// Total statements: {}\n", total_stmts).unwrap();
    write!(buf, "// Parallel steps: {}\n", parallel_steps).unwrap();
    write!(buf, "// Shared regions: {}\n", shared_regions.join(", ")).unwrap();
    write!(buf, "// Uses concurrent writes: {}\n", uses_cw).unwrap();
    write!(buf, "// Uses concurrent reads: {}\n", uses_cr).unwrap();
    if let Some(ref wb) = program.work_bound {
        write!(buf, "// Work bound: {}\n", wb).unwrap();
    }
    if let Some(ref tb) = program.time_bound {
        write!(buf, "// Time bound: {}\n", tb).unwrap();
    }
    buf
}

/// Print an expression with full precedence-aware parenthesization.
/// This is the same as `PramPrinter::expr_str` but exposed as a standalone function.
pub fn print_expr_precedence(expr: &Expr) -> String {
    PramPrinter::expr_str(expr)
}

/// Print a program to a file.
pub fn print_to_file(program: &PramProgram, path: &str) -> std::io::Result<()> {
    let content = PramPrinter::print_program(program);
    std::fs::write(path, content)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pram_ir::parser::parse_program;

    #[test]
    fn test_print_minimal() {
        let prog = PramProgram::new("noop", MemoryModel::EREW);
        let s = PramPrinter::print_program(&prog);
        assert!(s.contains("algorithm noop()"));
        assert!(s.contains("model EREW"));
    }

    #[test]
    fn test_print_shared_decl() {
        let mut prog = PramProgram::new("sd", MemoryModel::CREW);
        prog.shared_memory.push(SharedMemoryDecl {
            name: "A".into(),
            elem_type: PramType::Int64,
            size: Expr::var("n"),
        });
        let s = PramPrinter::print_program(&prog);
        assert!(s.contains("shared A: i64[n]"));
    }

    #[test]
    fn test_print_parallel_for() {
        let mut prog = PramProgram::new("pf", MemoryModel::EREW);
        prog.body.push(Stmt::ParallelFor {
            proc_var: "p".into(),
            num_procs: Expr::var("n"),
            body: vec![
                Stmt::SharedWrite {
                    memory: Expr::var("A"),
                    index: Expr::ProcessorId,
                    value: Expr::int(0),
                },
            ],
        });
        let s = PramPrinter::print_program(&prog);
        assert!(s.contains("parallel_for p in 0..n"));
        assert!(s.contains("shared_write(A, pid, 0)"));
    }

    #[test]
    fn test_expr_precedence() {
        // 1 + 2 * 3  →  no parens needed around 2*3
        let e = Expr::binop(
            BinOp::Add,
            Expr::int(1),
            Expr::binop(BinOp::Mul, Expr::int(2), Expr::int(3)),
        );
        assert_eq!(PramPrinter::expr_str(&e), "1 + 2 * 3");

        // (1 + 2) * 3  →  needs parens
        let e2 = Expr::binop(
            BinOp::Mul,
            Expr::binop(BinOp::Add, Expr::int(1), Expr::int(2)),
            Expr::int(3),
        );
        assert_eq!(PramPrinter::expr_str(&e2), "(1 + 2) * 3");
    }

    #[test]
    fn test_expr_right_assoc_parens() {
        // a - (b - c)
        let e = Expr::binop(
            BinOp::Sub,
            Expr::var("a"),
            Expr::binop(BinOp::Sub, Expr::var("b"), Expr::var("c")),
        );
        assert_eq!(PramPrinter::expr_str(&e), "a - (b - c)");
    }

    #[test]
    fn test_expr_min_max_function() {
        let e = Expr::binop(BinOp::Min, Expr::var("a"), Expr::var("b"));
        assert_eq!(PramPrinter::expr_str(&e), "min(a, b)");
    }

    #[test]
    fn test_expr_unary() {
        let e = Expr::unop(UnaryOp::Neg, Expr::var("x"));
        assert_eq!(PramPrinter::expr_str(&e), "-x");

        let e2 = Expr::unop(UnaryOp::Not, Expr::BoolLiteral(true));
        assert_eq!(PramPrinter::expr_str(&e2), "!true");
    }

    #[test]
    fn test_print_if_else() {
        let mut prog = PramProgram::new("ie", MemoryModel::EREW);
        prog.body.push(Stmt::If {
            condition: Expr::binop(BinOp::Lt, Expr::var("x"), Expr::int(0)),
            then_body: vec![Stmt::Assign("x".into(), Expr::int(0))],
            else_body: vec![Stmt::Assign("x".into(), Expr::int(1))],
        });
        let s = PramPrinter::print_program(&prog);
        assert!(s.contains("if x < 0 {"));
        assert!(s.contains("} else {"));
    }

    #[test]
    fn test_print_barrier() {
        let mut prog = PramProgram::new("b", MemoryModel::EREW);
        prog.body.push(Stmt::Barrier);
        let s = PramPrinter::print_program(&prog);
        assert!(s.contains("barrier;"));
    }

    #[test]
    fn test_print_return() {
        let mut prog = PramProgram::new("r", MemoryModel::EREW);
        prog.body.push(Stmt::Return(Some(Expr::int(42))));
        prog.body.push(Stmt::Return(None));
        let s = PramPrinter::print_program(&prog);
        assert!(s.contains("return 42;"));
        assert!(s.contains("return;"));
    }

    #[test]
    fn test_print_shared_read_expr() {
        let e = Expr::shared_read(Expr::var("M"), Expr::ProcessorId);
        assert_eq!(PramPrinter::expr_str(&e), "shared_read(M, pid)");
    }

    #[test]
    fn test_round_trip_simple() {
        let src = r#"algorithm rt(n: i64) model CREW {
    shared A: i64[n];
    processors = n;
    parallel_for p in 0..n {
        shared_write(A, pid, 0);
    }
}
"#;
        let prog1 = parse_program(src).unwrap();
        let printed = PramPrinter::print_program(&prog1);
        let prog2 = parse_program(&printed).unwrap();
        assert_eq!(prog1.name, prog2.name);
        assert_eq!(prog1.memory_model, prog2.memory_model);
        assert_eq!(prog1.parameters.len(), prog2.parameters.len());
        assert_eq!(prog1.shared_memory.len(), prog2.shared_memory.len());
        assert_eq!(prog1.body.len(), prog2.body.len());
    }

    #[test]
    fn test_round_trip_expressions() {
        let src = r#"algorithm rte() model EREW {
    processors = 1;
    let x: i64 = 1 + 2 * 3;
    let y: i64 = (1 + 2) * 3;
}
"#;
        let prog1 = parse_program(src).unwrap();
        let printed = PramPrinter::print_program(&prog1);
        let prog2 = parse_program(&printed).unwrap();
        // Check that the expressions evaluate to the same constants.
        if let (Stmt::LocalDecl(_, _, Some(e1)), Stmt::LocalDecl(_, _, Some(e2))) =
            (&prog1.body[0], &prog2.body[0])
        {
            assert_eq!(e1.eval_const_int(), e2.eval_const_int());
        }
        if let (Stmt::LocalDecl(_, _, Some(e1)), Stmt::LocalDecl(_, _, Some(e2))) =
            (&prog1.body[1], &prog2.body[1])
        {
            assert_eq!(e1.eval_const_int(), e2.eval_const_int());
        }
    }

    #[test]
    fn test_print_seq_for() {
        let mut prog = PramProgram::new("sf", MemoryModel::EREW);
        prog.body.push(Stmt::SeqFor {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::var("n"),
            step: None,
            body: vec![Stmt::Nop],
        });
        let s = PramPrinter::print_program(&prog);
        assert!(s.contains("for i in 0..n {"));
    }

    #[test]
    fn test_print_while() {
        let mut prog = PramProgram::new("w", MemoryModel::EREW);
        prog.body.push(Stmt::While {
            condition: Expr::BoolLiteral(true),
            body: vec![Stmt::Barrier],
        });
        let s = PramPrinter::print_program(&prog);
        assert!(s.contains("while true {"));
    }

    #[test]
    fn test_print_cast() {
        let e = Expr::Cast(Box::new(Expr::var("x")), PramType::Float64);
        assert_eq!(PramPrinter::expr_str(&e), "(x as f64)");
    }

    #[test]
    fn test_print_conditional() {
        let e = Expr::Conditional(
            Box::new(Expr::BoolLiteral(true)),
            Box::new(Expr::int(1)),
            Box::new(Expr::int(2)),
        );
        assert_eq!(PramPrinter::expr_str(&e), "(true ? 1 : 2)");
    }

    #[test]
    fn test_print_compact() {
        let mut prog = PramProgram::new("compact_test", MemoryModel::EREW);
        prog.body.push(Stmt::Assign("x".into(), Expr::int(1)));
        prog.body.push(Stmt::Barrier);
        let s = print_compact(&prog);
        assert!(!s.contains('\n'));
        assert!(s.contains("x = 1;"));
        assert!(s.contains("barrier;"));
    }

    #[test]
    fn test_print_annotated() {
        let mut prog = PramProgram::new("annotated_test", MemoryModel::CREW);
        prog.shared_memory.push(SharedMemoryDecl {
            name: "A".into(),
            elem_type: PramType::Int64,
            size: Expr::int(10),
        });
        prog.body.push(Stmt::ParallelFor {
            proc_var: "p".into(),
            num_procs: Expr::int(10),
            body: vec![Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::ProcessorId,
                value: Expr::int(0),
            }],
        });
        let s = print_annotated(&prog);
        assert!(s.contains("Annotations"));
        assert!(s.contains("Total statements:"));
        assert!(s.contains("Parallel steps:"));
    }

    #[test]
    fn test_print_expr_precedence() {
        let e = Expr::binop(
            BinOp::Mul,
            Expr::binop(BinOp::Add, Expr::int(1), Expr::int(2)),
            Expr::int(3),
        );
        assert_eq!(print_expr_precedence(&e), "(1 + 2) * 3");
    }

    #[test]
    fn test_print_to_file() {
        let prog = PramProgram::new("file_test", MemoryModel::EREW);
        let path = "/tmp/pram_printer_test_output.pram";
        print_to_file(&prog, path).unwrap();
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("algorithm file_test"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_pretty_printer_config_default() {
        let config = PrettyPrinterConfig::default();
        assert_eq!(config.indent_size, 4);
        assert_eq!(config.max_line_width, 100);
        assert!(config.show_types);
    }

    #[test]
    fn test_compact_with_if() {
        let mut prog = PramProgram::new("cif", MemoryModel::EREW);
        prog.body.push(Stmt::If {
            condition: Expr::BoolLiteral(true),
            then_body: vec![Stmt::Assign("x".into(), Expr::int(1))],
            else_body: vec![Stmt::Assign("x".into(), Expr::int(2))],
        });
        let s = print_compact(&prog);
        assert!(s.contains("if true"));
        assert!(s.contains("else"));
    }

    #[test]
    fn test_compact_with_while() {
        let mut prog = PramProgram::new("cw", MemoryModel::EREW);
        prog.body.push(Stmt::While {
            condition: Expr::var("running"),
            body: vec![Stmt::Nop],
        });
        let s = print_compact(&prog);
        assert!(s.contains("while running"));
    }
}
