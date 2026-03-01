//! Programmatic IR builder API for constructing PRAM programs.
//!
//! Provides a fluent interface so that the algorithm library (or tests) can
//! build a [`PramProgram`] without hand-constructing every AST node.

use super::ast::*;
use super::types::PramType;

// ---------------------------------------------------------------------------
// PramBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for [`PramProgram`] values.
pub struct PramBuilder {
    name: String,
    model: MemoryModel,
    parameters: Vec<Parameter>,
    shared_memory: Vec<SharedMemoryDecl>,
    num_processors: Expr,
    body: Vec<Stmt>,
    work_bound: Option<String>,
    time_bound: Option<String>,
    description: Option<String>,
}

impl PramBuilder {
    /// Start building a new program with the given name and memory model.
    pub fn new(name: &str, model: MemoryModel) -> Self {
        Self {
            name: name.to_string(),
            model,
            parameters: Vec::new(),
            shared_memory: Vec::new(),
            num_processors: Expr::IntLiteral(1),
            body: Vec::new(),
            work_bound: None,
            time_bound: None,
            description: None,
        }
    }

    // -- metadata ----------------------------------------------------------

    /// Add a parameter to the algorithm.
    pub fn add_param(&mut self, name: &str, ty: PramType) -> &mut Self {
        self.parameters.push(Parameter {
            name: name.to_string(),
            param_type: ty,
        });
        self
    }

    /// Declare a shared-memory region.
    pub fn add_shared_memory(
        &mut self,
        name: &str,
        elem_ty: PramType,
        size: Expr,
    ) -> &mut Self {
        self.shared_memory.push(SharedMemoryDecl {
            name: name.to_string(),
            elem_type: elem_ty,
            size,
        });
        self
    }

    /// Set the number-of-processors expression.
    pub fn set_num_processors(&mut self, expr: Expr) -> &mut Self {
        self.num_processors = expr;
        self
    }

    /// Set the work-bound annotation (e.g. `"O(n log n)"`).
    pub fn set_work_bound(&mut self, bound: &str) -> &mut Self {
        self.work_bound = Some(bound.to_string());
        self
    }

    /// Set the time-bound annotation (e.g. `"O(log n)"`).
    pub fn set_time_bound(&mut self, bound: &str) -> &mut Self {
        self.time_bound = Some(bound.to_string());
        self
    }

    /// Set a human-readable description.
    pub fn set_description(&mut self, desc: &str) -> &mut Self {
        self.description = Some(desc.to_string());
        self
    }

    // -- statement helpers (push into body) --------------------------------

    /// Append a raw statement.
    pub fn add_stmt(&mut self, stmt: Stmt) -> &mut Self {
        self.body.push(stmt);
        self
    }

    /// Append a `parallel_for` statement.
    pub fn parallel_for(
        &mut self,
        proc_var: &str,
        num_procs: Expr,
        body: Vec<Stmt>,
    ) -> &mut Self {
        self.body.push(Stmt::ParallelFor {
            proc_var: proc_var.to_string(),
            num_procs,
            body,
        });
        self
    }

    /// Append a sequential `for` loop.
    pub fn seq_for(
        &mut self,
        var: &str,
        start: Expr,
        end: Expr,
        step: Option<Expr>,
        body: Vec<Stmt>,
    ) -> &mut Self {
        self.body.push(Stmt::SeqFor {
            var: var.to_string(),
            start,
            end,
            step,
            body,
        });
        self
    }

    /// Append a `while` loop.
    pub fn while_loop(&mut self, condition: Expr, body: Vec<Stmt>) -> &mut Self {
        self.body.push(Stmt::While { condition, body });
        self
    }

    /// Append an `if`/`else`.
    pub fn if_then_else(
        &mut self,
        condition: Expr,
        then_body: Vec<Stmt>,
        else_body: Vec<Stmt>,
    ) -> &mut Self {
        self.body.push(Stmt::If {
            condition,
            then_body,
            else_body,
        });
        self
    }

    /// Append a variable assignment.
    pub fn assign(&mut self, var: &str, expr: Expr) -> &mut Self {
        self.body
            .push(Stmt::Assign(var.to_string(), expr));
        self
    }

    /// Append a local-variable declaration.
    pub fn local_decl(
        &mut self,
        name: &str,
        ty: PramType,
        init: Option<Expr>,
    ) -> &mut Self {
        self.body
            .push(Stmt::LocalDecl(name.to_string(), ty, init));
        self
    }

    /// Append a barrier.
    pub fn barrier(&mut self) -> &mut Self {
        self.body.push(Stmt::Barrier);
        self
    }

    /// Append an `alloc_shared` statement.
    pub fn alloc_shared(
        &mut self,
        name: &str,
        elem_ty: PramType,
        size: Expr,
    ) -> &mut Self {
        self.body.push(Stmt::AllocShared {
            name: name.to_string(),
            elem_type: elem_ty,
            size,
        });
        self
    }

    /// Append a `free_shared` statement.
    pub fn free_shared(&mut self, name: &str) -> &mut Self {
        self.body.push(Stmt::FreeShared(name.to_string()));
        self
    }

    /// Append a return statement.
    pub fn return_value(&mut self, expr: Option<Expr>) -> &mut Self {
        self.body.push(Stmt::Return(expr));
        self
    }

    /// Append an assertion.
    pub fn assert_stmt(&mut self, condition: Expr, msg: &str) -> &mut Self {
        self.body
            .push(Stmt::Assert(condition, msg.to_string()));
        self
    }

    /// Append a comment.
    pub fn comment(&mut self, text: &str) -> &mut Self {
        self.body.push(Stmt::Comment(text.to_string()));
        self
    }

    /// Append a nop.
    pub fn nop(&mut self) -> &mut Self {
        self.body.push(Stmt::Nop);
        self
    }

    // -- expression helpers (static) ---------------------------------------

    /// Build a `shared_read` expression.
    pub fn shared_read(mem: &str, index: Expr) -> Expr {
        Expr::SharedRead(
            Box::new(Expr::Variable(mem.to_string())),
            Box::new(index),
        )
    }

    /// Build a `shared_write` statement.
    pub fn shared_write(mem: &str, index: Expr, value: Expr) -> Stmt {
        Stmt::SharedWrite {
            memory: Expr::Variable(mem.to_string()),
            index,
            value,
        }
    }

    /// Variable reference.
    pub fn var(name: &str) -> Expr {
        Expr::Variable(name.to_string())
    }

    /// Integer literal.
    pub fn int(val: i64) -> Expr {
        Expr::IntLiteral(val)
    }

    /// Float literal.
    pub fn float(val: f64) -> Expr {
        Expr::FloatLiteral(val)
    }

    /// Boolean literal.
    pub fn bool_val(val: bool) -> Expr {
        Expr::BoolLiteral(val)
    }

    /// Processor-ID expression.
    pub fn pid() -> Expr {
        Expr::ProcessorId
    }

    /// Number-of-processors expression.
    pub fn num_procs() -> Expr {
        Expr::NumProcessors
    }

    /// Generic binary-op expression.
    pub fn binop(op: BinOp, left: Expr, right: Expr) -> Expr {
        Expr::binop(op, left, right)
    }

    /// Shorthand: addition.
    pub fn add(left: Expr, right: Expr) -> Expr {
        Expr::binop(BinOp::Add, left, right)
    }

    /// Shorthand: subtraction.
    pub fn sub(left: Expr, right: Expr) -> Expr {
        Expr::binop(BinOp::Sub, left, right)
    }

    /// Shorthand: multiplication.
    pub fn mul(left: Expr, right: Expr) -> Expr {
        Expr::binop(BinOp::Mul, left, right)
    }

    /// Shorthand: division.
    pub fn div(left: Expr, right: Expr) -> Expr {
        Expr::binop(BinOp::Div, left, right)
    }

    /// Shorthand: modulo.
    pub fn modulo(left: Expr, right: Expr) -> Expr {
        Expr::binop(BinOp::Mod, left, right)
    }

    /// Shorthand: less-than comparison.
    pub fn lt(left: Expr, right: Expr) -> Expr {
        Expr::binop(BinOp::Lt, left, right)
    }

    /// Shorthand: less-than-or-equal comparison.
    pub fn le(left: Expr, right: Expr) -> Expr {
        Expr::binop(BinOp::Le, left, right)
    }

    /// Shorthand: equality comparison.
    pub fn eq(left: Expr, right: Expr) -> Expr {
        Expr::binop(BinOp::Eq, left, right)
    }

    /// Shorthand: not-equal comparison.
    pub fn ne(left: Expr, right: Expr) -> Expr {
        Expr::binop(BinOp::Ne, left, right)
    }

    /// Shorthand: logical AND.
    pub fn and(left: Expr, right: Expr) -> Expr {
        Expr::binop(BinOp::And, left, right)
    }

    /// Shorthand: logical OR.
    pub fn or(left: Expr, right: Expr) -> Expr {
        Expr::binop(BinOp::Or, left, right)
    }

    /// Shorthand: logical NOT.
    pub fn not(expr: Expr) -> Expr {
        Expr::unop(UnaryOp::Not, expr)
    }

    /// Shorthand: negation.
    pub fn neg(expr: Expr) -> Expr {
        Expr::unop(UnaryOp::Neg, expr)
    }

    /// Function call expression.
    pub fn call(name: &str, args: Vec<Expr>) -> Expr {
        Expr::FunctionCall(name.to_string(), args)
    }

    /// Conditional (ternary) expression.
    pub fn cond(c: Expr, t: Expr, e: Expr) -> Expr {
        Expr::Conditional(Box::new(c), Box::new(t), Box::new(e))
    }

    // -- build -------------------------------------------------------------

    /// Consume the builder and produce the finished [`PramProgram`].
    pub fn build(self) -> PramProgram {
        PramProgram {
            name: self.name,
            memory_model: self.model,
            parameters: self.parameters,
            shared_memory: self.shared_memory,
            body: self.body,
            num_processors: self.num_processors,
            work_bound: self.work_bound,
            time_bound: self.time_bound,
            description: self.description,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_build() {
        let mut b = PramBuilder::new("test", MemoryModel::CREW);
        b.add_param("n", PramType::Int64);
        b.set_num_processors(PramBuilder::var("n"));
        let prog = b.build();

        assert_eq!(prog.name, "test");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert_eq!(prog.parameters.len(), 1);
        assert_eq!(prog.parameters[0].name, "n");
    }

    #[test]
    fn test_shared_memory_decl() {
        let mut b = PramBuilder::new("sort", MemoryModel::EREW);
        b.add_shared_memory("A", PramType::Int64, Expr::var("n"));
        b.add_shared_memory("B", PramType::Int64, Expr::var("n"));
        let prog = b.build();

        assert_eq!(prog.shared_memory.len(), 2);
        assert_eq!(prog.shared_memory[0].name, "A");
        assert_eq!(prog.shared_memory[1].name, "B");
    }

    #[test]
    fn test_parallel_for_builder() {
        let body = vec![
            PramBuilder::shared_write("A", PramBuilder::pid(), PramBuilder::int(1)),
        ];
        let mut b = PramBuilder::new("init", MemoryModel::EREW);
        b.add_shared_memory("A", PramType::Int64, Expr::int(8));
        b.set_num_processors(Expr::int(8));
        b.parallel_for("pid", Expr::int(8), body);
        let prog = b.build();

        assert_eq!(prog.body.len(), 1);
        match &prog.body[0] {
            Stmt::ParallelFor { proc_var, body, .. } => {
                assert_eq!(proc_var, "pid");
                assert_eq!(body.len(), 1);
            }
            _ => panic!("expected ParallelFor"),
        }
    }

    #[test]
    fn test_expr_helpers() {
        let e = PramBuilder::add(PramBuilder::var("x"), PramBuilder::int(1));
        assert_eq!(e, Expr::binop(BinOp::Add, Expr::var("x"), Expr::int(1)));

        let e2 = PramBuilder::lt(PramBuilder::pid(), PramBuilder::var("n"));
        match &e2 {
            Expr::BinOp(BinOp::Lt, _, _) => {}
            _ => panic!("expected Lt"),
        }
    }

    #[test]
    fn test_control_flow() {
        let mut b = PramBuilder::new("cf", MemoryModel::CREW);
        b.if_then_else(
            PramBuilder::lt(PramBuilder::var("x"), PramBuilder::int(0)),
            vec![Stmt::Assign("x".into(), Expr::int(0))],
            vec![],
        );
        b.while_loop(
            PramBuilder::lt(PramBuilder::var("i"), PramBuilder::var("n")),
            vec![Stmt::Assign(
                "i".into(),
                PramBuilder::add(PramBuilder::var("i"), PramBuilder::int(1)),
            )],
        );
        b.seq_for(
            "j",
            Expr::int(0),
            Expr::var("m"),
            None,
            vec![Stmt::Nop],
        );
        let prog = b.build();
        assert_eq!(prog.body.len(), 3);
    }

    #[test]
    fn test_barrier_and_nop() {
        let mut b = PramBuilder::new("b", MemoryModel::EREW);
        b.nop();
        b.barrier();
        b.nop();
        let prog = b.build();
        assert_eq!(prog.body.len(), 3);
        assert_eq!(prog.body[1], Stmt::Barrier);
    }

    #[test]
    fn test_metadata() {
        let mut b = PramBuilder::new("m", MemoryModel::CREW);
        b.set_work_bound("O(n)");
        b.set_time_bound("O(log n)");
        b.set_description("A test program");
        let prog = b.build();
        assert_eq!(prog.work_bound.as_deref(), Some("O(n)"));
        assert_eq!(prog.time_bound.as_deref(), Some("O(log n)"));
        assert_eq!(prog.description.as_deref(), Some("A test program"));
    }

    #[test]
    fn test_shared_read_expr() {
        let r = PramBuilder::shared_read("A", PramBuilder::pid());
        match &r {
            Expr::SharedRead(mem, _idx) => {
                assert_eq!(**mem, Expr::Variable("A".into()));
            }
            _ => panic!("expected SharedRead"),
        }
    }

    #[test]
    fn test_shared_write_stmt() {
        let s = PramBuilder::shared_write("B", Expr::int(0), Expr::int(42));
        match &s {
            Stmt::SharedWrite { memory, index, value } => {
                assert_eq!(*memory, Expr::Variable("B".into()));
                assert_eq!(*index, Expr::int(0));
                assert_eq!(*value, Expr::int(42));
            }
            _ => panic!("expected SharedWrite"),
        }
    }

    #[test]
    fn test_local_decl_and_assign() {
        let mut b = PramBuilder::new("d", MemoryModel::EREW);
        b.local_decl("x", PramType::Int64, Some(Expr::int(0)));
        b.assign("x", PramBuilder::add(PramBuilder::var("x"), PramBuilder::int(1)));
        let prog = b.build();
        assert_eq!(prog.body.len(), 2);
        match &prog.body[0] {
            Stmt::LocalDecl(name, ty, init) => {
                assert_eq!(name, "x");
                assert_eq!(*ty, PramType::Int64);
                assert_eq!(*init, Some(Expr::int(0)));
            }
            _ => panic!("expected LocalDecl"),
        }
    }

    #[test]
    fn test_complex_program() {
        let inner_body = vec![
            Stmt::LocalDecl(
                "val".into(),
                PramType::Int64,
                Some(PramBuilder::shared_read("A", PramBuilder::pid())),
            ),
            PramBuilder::shared_write(
                "B",
                PramBuilder::pid(),
                PramBuilder::add(PramBuilder::var("val"), PramBuilder::int(1)),
            ),
            Stmt::Barrier,
            PramBuilder::shared_write(
                "A",
                PramBuilder::pid(),
                PramBuilder::shared_read("B", PramBuilder::pid()),
            ),
        ];

        let mut b = PramBuilder::new("copy_inc", MemoryModel::CREW);
        b.add_param("n", PramType::Int64);
        b.add_shared_memory("A", PramType::Int64, Expr::var("n"));
        b.add_shared_memory("B", PramType::Int64, Expr::var("n"));
        b.set_num_processors(Expr::var("n"));
        b.parallel_for("pid", Expr::var("n"), inner_body);
        let prog = b.build();

        assert_eq!(prog.name, "copy_inc");
        assert_eq!(prog.shared_memory.len(), 2);
        assert_eq!(prog.parallel_step_count(), 1);
    }

    #[test]
    fn test_comment_and_return() {
        let mut b = PramBuilder::new("r", MemoryModel::EREW);
        b.comment("compute result");
        b.return_value(Some(Expr::int(0)));
        let prog = b.build();
        assert_eq!(prog.body.len(), 2);
        match &prog.body[0] {
            Stmt::Comment(s) => assert_eq!(s, "compute result"),
            _ => panic!("expected Comment"),
        }
    }

    #[test]
    fn test_logic_ops() {
        let e = PramBuilder::and(
            PramBuilder::not(PramBuilder::bool_val(false)),
            PramBuilder::or(PramBuilder::bool_val(true), PramBuilder::bool_val(false)),
        );
        match &e {
            Expr::BinOp(BinOp::And, _, _) => {}
            _ => panic!("expected And"),
        }
    }
}
