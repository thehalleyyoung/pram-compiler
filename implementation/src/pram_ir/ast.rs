use serde::{Deserialize, Serialize};
use std::fmt;

use super::types::PramType;

/// A unique identifier for AST nodes.
pub type NodeId = usize;

/// Memory model for a PRAM program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryModel {
    /// Exclusive Read Exclusive Write
    EREW,
    /// Concurrent Read Exclusive Write
    CREW,
    /// Concurrent Read Concurrent Write with priority resolution
    CRCWPriority,
    /// Concurrent Read Concurrent Write with arbitrary resolution
    CRCWArbitrary,
    /// Concurrent Read Concurrent Write with common resolution
    CRCWCommon,
}

impl MemoryModel {
    pub fn allows_concurrent_read(&self) -> bool {
        !matches!(self, MemoryModel::EREW)
    }

    pub fn allows_concurrent_write(&self) -> bool {
        matches!(
            self,
            MemoryModel::CRCWPriority | MemoryModel::CRCWArbitrary | MemoryModel::CRCWCommon
        )
    }

    pub fn name(&self) -> &'static str {
        match self {
            MemoryModel::EREW => "EREW",
            MemoryModel::CREW => "CREW",
            MemoryModel::CRCWPriority => "CRCW-Priority",
            MemoryModel::CRCWArbitrary => "CRCW-Arbitrary",
            MemoryModel::CRCWCommon => "CRCW-Common",
        }
    }
}

impl fmt::Display for MemoryModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// CRCW write resolution policy for concurrent writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WriteResolution {
    /// Lowest processor-ID wins
    Priority,
    /// Arbitrary processor wins (nondeterministic)
    Arbitrary,
    /// All writers must agree on the same value
    Common,
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Min,
    Max,
}

impl BinOp {
    pub fn symbol(&self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::And => "&&",
            BinOp::Or => "||",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::Min => "min",
            BinOp::Max => "max",
        }
    }

    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne
        )
    }

    pub fn is_arithmetic(&self) -> bool {
        matches!(
            self,
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod
        )
    }

    pub fn is_logical(&self) -> bool {
        matches!(self, BinOp::And | BinOp::Or)
    }

    pub fn precedence(&self) -> u8 {
        match self {
            BinOp::Or => 1,
            BinOp::And => 2,
            BinOp::BitOr => 3,
            BinOp::BitXor => 4,
            BinOp::BitAnd => 5,
            BinOp::Eq | BinOp::Ne => 6,
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => 7,
            BinOp::Shl | BinOp::Shr => 8,
            BinOp::Add | BinOp::Sub => 9,
            BinOp::Mul | BinOp::Div | BinOp::Mod => 10,
            BinOp::Min | BinOp::Max => 11,
        }
    }
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
}

impl UnaryOp {
    pub fn symbol(&self) -> &'static str {
        match self {
            UnaryOp::Neg => "-",
            UnaryOp::Not => "!",
            UnaryOp::BitNot => "~",
        }
    }
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

/// Expressions in the PRAM IR.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// Integer literal
    IntLiteral(i64),
    /// Float literal
    FloatLiteral(f64),
    /// Boolean literal
    BoolLiteral(bool),
    /// Variable reference
    Variable(String),
    /// Current processor ID (implicit in parallel_for body)
    ProcessorId,
    /// Number of processors
    NumProcessors,
    /// Binary operation
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    /// Unary operation
    UnaryOp(UnaryOp, Box<Expr>),
    /// Read from shared memory: shared_read(memory, index)
    SharedRead(Box<Expr>, Box<Expr>),
    /// Array indexing (local arrays)
    ArrayIndex(Box<Expr>, Box<Expr>),
    /// Function call
    FunctionCall(String, Vec<Expr>),
    /// Type cast
    Cast(Box<Expr>, PramType),
    /// Conditional expression (ternary)
    Conditional(Box<Expr>, Box<Expr>, Box<Expr>),
}

impl Expr {
    /// Create a binary operation expression.
    pub fn binop(op: BinOp, left: Expr, right: Expr) -> Self {
        Expr::BinOp(op, Box::new(left), Box::new(right))
    }

    /// Create a unary operation expression.
    pub fn unop(op: UnaryOp, operand: Expr) -> Self {
        Expr::UnaryOp(op, Box::new(operand))
    }

    /// Create a shared memory read expression.
    pub fn shared_read(mem: Expr, index: Expr) -> Self {
        Expr::SharedRead(Box::new(mem), Box::new(index))
    }

    /// Create an array index expression.
    pub fn array_index(arr: Expr, index: Expr) -> Self {
        Expr::ArrayIndex(Box::new(arr), Box::new(index))
    }

    /// Create a variable reference.
    pub fn var(name: &str) -> Self {
        Expr::Variable(name.to_string())
    }

    /// Create an integer literal.
    pub fn int(value: i64) -> Self {
        Expr::IntLiteral(value)
    }

    /// Create a float literal.
    pub fn float(value: f64) -> Self {
        Expr::FloatLiteral(value)
    }

    /// Create a boolean literal.
    pub fn bool_lit(value: bool) -> Self {
        Expr::BoolLiteral(value)
    }

    /// Test if this expression is a constant.
    pub fn is_constant(&self) -> bool {
        match self {
            Expr::IntLiteral(_) | Expr::FloatLiteral(_) | Expr::BoolLiteral(_) => true,
            Expr::UnaryOp(_, e) => e.is_constant(),
            Expr::BinOp(_, a, b) => a.is_constant() && b.is_constant(),
            _ => false,
        }
    }

    /// Compute the depth of the expression tree.
    pub fn depth(&self) -> usize {
        match self {
            Expr::IntLiteral(_)
            | Expr::FloatLiteral(_)
            | Expr::BoolLiteral(_)
            | Expr::Variable(_)
            | Expr::ProcessorId
            | Expr::NumProcessors => 1,
            Expr::UnaryOp(_, e) | Expr::Cast(e, _) => 1 + e.depth(),
            Expr::BinOp(_, a, b) | Expr::SharedRead(a, b) | Expr::ArrayIndex(a, b) => {
                1 + a.depth().max(b.depth())
            }
            Expr::FunctionCall(_, args) => {
                1 + args.iter().map(|a| a.depth()).max().unwrap_or(0)
            }
            Expr::Conditional(c, t, e) => {
                1 + c.depth().max(t.depth()).max(e.depth())
            }
        }
    }

    /// Map a function over the immediate children of this expression node,
    /// returning a new expression with the same top-level constructor.
    pub fn map_children(&self, f: impl Fn(&Expr) -> Expr) -> Expr {
        match self {
            Expr::IntLiteral(_)
            | Expr::FloatLiteral(_)
            | Expr::BoolLiteral(_)
            | Expr::Variable(_)
            | Expr::ProcessorId
            | Expr::NumProcessors => self.clone(),
            Expr::UnaryOp(op, e) => Expr::UnaryOp(*op, Box::new(f(e))),
            Expr::BinOp(op, a, b) => {
                Expr::BinOp(*op, Box::new(f(a)), Box::new(f(b)))
            }
            Expr::SharedRead(m, i) => {
                Expr::SharedRead(Box::new(f(m)), Box::new(f(i)))
            }
            Expr::ArrayIndex(a, i) => {
                Expr::ArrayIndex(Box::new(f(a)), Box::new(f(i)))
            }
            Expr::FunctionCall(name, args) => {
                Expr::FunctionCall(name.clone(), args.iter().map(&f).collect())
            }
            Expr::Cast(e, ty) => Expr::Cast(Box::new(f(e)), ty.clone()),
            Expr::Conditional(c, t, e) => {
                Expr::Conditional(Box::new(f(c)), Box::new(f(t)), Box::new(f(e)))
            }
        }
    }

    /// Evaluate a constant expression to an i64.
    pub fn eval_const_int(&self) -> Option<i64> {
        match self {
            Expr::IntLiteral(v) => Some(*v),
            Expr::BoolLiteral(b) => Some(if *b { 1 } else { 0 }),
            Expr::UnaryOp(UnaryOp::Neg, e) => e.eval_const_int().map(|v| -v),
            Expr::BinOp(op, a, b) => {
                let av = a.eval_const_int()?;
                let bv = b.eval_const_int()?;
                Some(match op {
                    BinOp::Add => av.wrapping_add(bv),
                    BinOp::Sub => av.wrapping_sub(bv),
                    BinOp::Mul => av.wrapping_mul(bv),
                    BinOp::Div => {
                        if bv == 0 {
                            return None;
                        }
                        av / bv
                    }
                    BinOp::Mod => {
                        if bv == 0 {
                            return None;
                        }
                        av % bv
                    }
                    BinOp::Lt => (av < bv) as i64,
                    BinOp::Le => (av <= bv) as i64,
                    BinOp::Gt => (av > bv) as i64,
                    BinOp::Ge => (av >= bv) as i64,
                    BinOp::Eq => (av == bv) as i64,
                    BinOp::Ne => (av != bv) as i64,
                    BinOp::BitAnd => av & bv,
                    BinOp::BitOr => av | bv,
                    BinOp::BitXor => av ^ bv,
                    BinOp::Shl => av.wrapping_shl(bv as u32),
                    BinOp::Shr => av.wrapping_shr(bv as u32),
                    BinOp::And => ((av != 0) && (bv != 0)) as i64,
                    BinOp::Or => ((av != 0) || (bv != 0)) as i64,
                    BinOp::Min => av.min(bv),
                    BinOp::Max => av.max(bv),
                })
            }
            _ => None,
        }
    }

    /// Count the number of sub-expressions.
    pub fn node_count(&self) -> usize {
        match self {
            Expr::IntLiteral(_)
            | Expr::FloatLiteral(_)
            | Expr::BoolLiteral(_)
            | Expr::Variable(_)
            | Expr::ProcessorId
            | Expr::NumProcessors => 1,
            Expr::UnaryOp(_, e) => 1 + e.node_count(),
            Expr::BinOp(_, a, b) => 1 + a.node_count() + b.node_count(),
            Expr::SharedRead(m, i) => 1 + m.node_count() + i.node_count(),
            Expr::ArrayIndex(a, i) => 1 + a.node_count() + i.node_count(),
            Expr::FunctionCall(_, args) => {
                1 + args.iter().map(|a| a.node_count()).sum::<usize>()
            }
            Expr::Cast(e, _) => 1 + e.node_count(),
            Expr::Conditional(c, t, e) => {
                1 + c.node_count() + t.node_count() + e.node_count()
            }
        }
    }

    /// Collect all variable names referenced.
    pub fn collect_variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_variables_into(&mut vars);
        vars
    }

    fn collect_variables_into(&self, vars: &mut Vec<String>) {
        match self {
            Expr::Variable(name) => {
                if !vars.contains(name) {
                    vars.push(name.clone());
                }
            }
            Expr::BinOp(_, a, b) | Expr::SharedRead(a, b) | Expr::ArrayIndex(a, b) => {
                a.collect_variables_into(vars);
                b.collect_variables_into(vars);
            }
            Expr::UnaryOp(_, e) | Expr::Cast(e, _) => e.collect_variables_into(vars),
            Expr::Conditional(c, t, e) => {
                c.collect_variables_into(vars);
                t.collect_variables_into(vars);
                e.collect_variables_into(vars);
            }
            Expr::FunctionCall(_, args) => {
                for arg in args {
                    arg.collect_variables_into(vars);
                }
            }
            _ => {}
        }
    }

    /// Check if this expression contains any shared memory access (read or write reference).
    pub fn has_shared_access(&self) -> bool {
        match self {
            Expr::SharedRead(_, _) => true,
            Expr::BinOp(_, a, b) | Expr::ArrayIndex(a, b) => {
                a.has_shared_access() || b.has_shared_access()
            }
            Expr::UnaryOp(_, e) | Expr::Cast(e, _) => e.has_shared_access(),
            Expr::Conditional(c, t, e) => {
                c.has_shared_access() || t.has_shared_access() || e.has_shared_access()
            }
            Expr::FunctionCall(_, args) => args.iter().any(|a| a.has_shared_access()),
            _ => false,
        }
    }

    /// Count total number of operations (arithmetic, comparison, logical, bitwise).
    pub fn operation_count(&self) -> usize {
        match self {
            Expr::BinOp(_, a, b) => 1 + a.operation_count() + b.operation_count(),
            Expr::UnaryOp(_, e) => 1 + e.operation_count(),
            Expr::SharedRead(m, i) => m.operation_count() + i.operation_count(),
            Expr::ArrayIndex(a, i) => a.operation_count() + i.operation_count(),
            Expr::Cast(e, _) => e.operation_count(),
            Expr::Conditional(c, t, e) => {
                1 + c.operation_count() + t.operation_count() + e.operation_count()
            }
            Expr::FunctionCall(_, args) => {
                1 + args.iter().map(|a| a.operation_count()).sum::<usize>()
            }
            _ => 0,
        }
    }

    /// Substitute a variable with an expression.
    pub fn substitute(&self, var: &str, replacement: &Expr) -> Expr {
        match self {
            Expr::Variable(name) if name == var => replacement.clone(),
            Expr::BinOp(op, a, b) => Expr::BinOp(
                *op,
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            Expr::UnaryOp(op, e) => {
                Expr::UnaryOp(*op, Box::new(e.substitute(var, replacement)))
            }
            Expr::SharedRead(m, i) => Expr::SharedRead(
                Box::new(m.substitute(var, replacement)),
                Box::new(i.substitute(var, replacement)),
            ),
            Expr::ArrayIndex(a, i) => Expr::ArrayIndex(
                Box::new(a.substitute(var, replacement)),
                Box::new(i.substitute(var, replacement)),
            ),
            Expr::FunctionCall(name, args) => Expr::FunctionCall(
                name.clone(),
                args.iter().map(|a| a.substitute(var, replacement)).collect(),
            ),
            Expr::Cast(e, t) => {
                Expr::Cast(Box::new(e.substitute(var, replacement)), t.clone())
            }
            Expr::Conditional(c, t, e) => Expr::Conditional(
                Box::new(c.substitute(var, replacement)),
                Box::new(t.substitute(var, replacement)),
                Box::new(e.substitute(var, replacement)),
            ),
            other => other.clone(),
        }
    }
}

/// Statements in the PRAM IR.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Stmt {
    /// Declare a local variable with optional initializer
    LocalDecl(String, PramType, Option<Expr>),

    /// Assign to a local variable
    Assign(String, Expr),

    /// Write to shared memory: shared_write(memory, index, value)
    SharedWrite {
        memory: Expr,
        index: Expr,
        value: Expr,
    },

    /// Parallel for loop (the core PRAM construct)
    ParallelFor {
        /// Variable name for processor ID (bound in body)
        proc_var: String,
        /// Number of processors (expression)
        num_procs: Expr,
        /// Loop body (executed by each processor)
        body: Vec<Stmt>,
    },

    /// Sequential for loop (within a single processor's work)
    SeqFor {
        var: String,
        start: Expr,
        end: Expr,
        step: Option<Expr>,
        body: Vec<Stmt>,
    },

    /// While loop
    While {
        condition: Expr,
        body: Vec<Stmt>,
    },

    /// If-then-else
    If {
        condition: Expr,
        then_body: Vec<Stmt>,
        else_body: Vec<Stmt>,
    },

    /// Synchronization barrier
    Barrier,

    /// Block of statements
    Block(Vec<Stmt>),

    /// Expression statement (for side-effecting function calls)
    ExprStmt(Expr),

    /// Return statement (for algorithms that produce a result)
    Return(Option<Expr>),

    /// Allocate shared memory
    AllocShared {
        name: String,
        elem_type: PramType,
        size: Expr,
    },

    /// Free shared memory
    FreeShared(String),

    /// No-op
    Nop,

    /// Atomic compare-and-swap on shared memory
    AtomicCAS {
        memory: Expr,
        index: Expr,
        expected: Expr,
        desired: Expr,
        result_var: String,
    },

    /// Fetch-and-add on shared memory
    FetchAdd {
        memory: Expr,
        index: Expr,
        value: Expr,
        result_var: String,
    },

    /// Assertion (for invariant checking)
    Assert(Expr, String),

    /// Comment (preserved in output)
    Comment(String),

    /// Parallel prefix / scan
    PrefixSum {
        input: String,
        output: String,
        size: Expr,
        op: BinOp,
    },
}

impl Stmt {
    /// Count the total number of statements (recursive).
    pub fn stmt_count(&self) -> usize {
        match self {
            Stmt::ParallelFor { body, .. } => 1 + body.iter().map(|s| s.stmt_count()).sum::<usize>(),
            Stmt::SeqFor { body, .. } => 1 + body.iter().map(|s| s.stmt_count()).sum::<usize>(),
            Stmt::While { body, .. } => 1 + body.iter().map(|s| s.stmt_count()).sum::<usize>(),
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                1 + then_body.iter().map(|s| s.stmt_count()).sum::<usize>()
                    + else_body.iter().map(|s| s.stmt_count()).sum::<usize>()
            }
            Stmt::Block(stmts) => 1 + stmts.iter().map(|s| s.stmt_count()).sum::<usize>(),
            _ => 1,
        }
    }

    /// Check if this statement contains a barrier.
    pub fn contains_barrier(&self) -> bool {
        match self {
            Stmt::Barrier => true,
            Stmt::ParallelFor { body, .. }
            | Stmt::SeqFor { body, .. }
            | Stmt::While { body, .. }
            | Stmt::Block(body) => body.iter().any(|s| s.contains_barrier()),
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                then_body.iter().any(|s| s.contains_barrier())
                    || else_body.iter().any(|s| s.contains_barrier())
            }
            _ => false,
        }
    }

    /// Check if this statement writes to shared memory.
    pub fn writes_shared(&self) -> bool {
        match self {
            Stmt::SharedWrite { .. } => true,
            Stmt::AtomicCAS { .. } | Stmt::FetchAdd { .. } => true,
            Stmt::ParallelFor { body, .. }
            | Stmt::SeqFor { body, .. }
            | Stmt::While { body, .. }
            | Stmt::Block(body) => body.iter().any(|s| s.writes_shared()),
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                then_body.iter().any(|s| s.writes_shared())
                    || else_body.iter().any(|s| s.writes_shared())
            }
            _ => false,
        }
    }

    /// Apply a transformation function to all expressions within this statement,
    /// returning a new statement with transformed expressions.
    pub fn transform_exprs(&self, f: &impl Fn(&Expr) -> Expr) -> Stmt {
        match self {
            Stmt::LocalDecl(name, ty, init) => {
                Stmt::LocalDecl(name.clone(), ty.clone(), init.as_ref().map(f))
            }
            Stmt::Assign(name, expr) => Stmt::Assign(name.clone(), f(expr)),
            Stmt::SharedWrite { memory, index, value } => Stmt::SharedWrite {
                memory: f(memory),
                index: f(index),
                value: f(value),
            },
            Stmt::ParallelFor { proc_var, num_procs, body } => Stmt::ParallelFor {
                proc_var: proc_var.clone(),
                num_procs: f(num_procs),
                body: body.iter().map(|s| s.transform_exprs(f)).collect(),
            },
            Stmt::SeqFor { var, start, end, step, body } => Stmt::SeqFor {
                var: var.clone(),
                start: f(start),
                end: f(end),
                step: step.as_ref().map(f),
                body: body.iter().map(|s| s.transform_exprs(f)).collect(),
            },
            Stmt::While { condition, body } => Stmt::While {
                condition: f(condition),
                body: body.iter().map(|s| s.transform_exprs(f)).collect(),
            },
            Stmt::If { condition, then_body, else_body } => Stmt::If {
                condition: f(condition),
                then_body: then_body.iter().map(|s| s.transform_exprs(f)).collect(),
                else_body: else_body.iter().map(|s| s.transform_exprs(f)).collect(),
            },
            Stmt::Block(stmts) => {
                Stmt::Block(stmts.iter().map(|s| s.transform_exprs(f)).collect())
            }
            Stmt::ExprStmt(expr) => Stmt::ExprStmt(f(expr)),
            Stmt::Return(val) => Stmt::Return(val.as_ref().map(f)),
            Stmt::AllocShared { name, elem_type, size } => Stmt::AllocShared {
                name: name.clone(),
                elem_type: elem_type.clone(),
                size: f(size),
            },
            Stmt::AtomicCAS { memory, index, expected, desired, result_var } => {
                Stmt::AtomicCAS {
                    memory: f(memory),
                    index: f(index),
                    expected: f(expected),
                    desired: f(desired),
                    result_var: result_var.clone(),
                }
            }
            Stmt::FetchAdd { memory, index, value, result_var } => Stmt::FetchAdd {
                memory: f(memory),
                index: f(index),
                value: f(value),
                result_var: result_var.clone(),
            },
            Stmt::Assert(expr, msg) => Stmt::Assert(f(expr), msg.clone()),
            Stmt::PrefixSum { input, output, size, op } => Stmt::PrefixSum {
                input: input.clone(),
                output: output.clone(),
                size: f(size),
                op: *op,
            },
            Stmt::Barrier | Stmt::Nop | Stmt::FreeShared(_) | Stmt::Comment(_) => self.clone(),
        }
    }

    /// Flatten nested Block statements into a flat list of statements.
    pub fn flatten_blocks(&self) -> Vec<Stmt> {
        match self {
            Stmt::Block(stmts) => {
                let mut result = Vec::new();
                for s in stmts {
                    match s {
                        Stmt::Block(_) => result.extend(s.flatten_blocks()),
                        other => result.push(other.clone()),
                    }
                }
                result
            }
            other => vec![other.clone()],
        }
    }

    /// Check if this statement reads from shared memory.
    pub fn reads_shared(&self) -> bool {
        match self {
            Stmt::Assign(_, expr) => expr_reads_shared(expr),
            Stmt::SharedWrite { value, .. } => expr_reads_shared(value),
            Stmt::ParallelFor { body, .. }
            | Stmt::SeqFor { body, .. }
            | Stmt::While { body, .. }
            | Stmt::Block(body) => body.iter().any(|s| s.reads_shared()),
            Stmt::If {
                condition,
                then_body,
                else_body,
            } => {
                expr_reads_shared(condition)
                    || then_body.iter().any(|s| s.reads_shared())
                    || else_body.iter().any(|s| s.reads_shared())
            }
            _ => false,
        }
    }

    /// Collect all shared memory accesses (reads and writes).
    pub fn collect_shared_accesses(&self) -> Vec<SharedAccess> {
        let mut accesses = Vec::new();
        self.collect_shared_accesses_into(&mut accesses);
        accesses
    }

    fn collect_shared_accesses_into(&self, accesses: &mut Vec<SharedAccess>) {
        match self {
            Stmt::SharedWrite { memory, index, value } => {
                collect_reads_from_expr(memory, accesses);
                collect_reads_from_expr(index, accesses);
                collect_reads_from_expr(value, accesses);
                if let Expr::Variable(name) = memory {
                    accesses.push(SharedAccess {
                        memory: name.clone(),
                        index: index.clone(),
                        is_write: true,
                    });
                }
            }
            Stmt::Assign(_, expr) => {
                collect_reads_from_expr(expr, accesses);
            }
            Stmt::ParallelFor { body, .. }
            | Stmt::SeqFor { body, .. }
            | Stmt::While { body, .. }
            | Stmt::Block(body) => {
                for s in body {
                    s.collect_shared_accesses_into(accesses);
                }
            }
            Stmt::If {
                condition,
                then_body,
                else_body,
            } => {
                collect_reads_from_expr(condition, accesses);
                for s in then_body {
                    s.collect_shared_accesses_into(accesses);
                }
                for s in else_body {
                    s.collect_shared_accesses_into(accesses);
                }
            }
            _ => {}
        }
    }
}

/// Represents an access to shared memory.
#[derive(Debug, Clone)]
pub struct SharedAccess {
    pub memory: String,
    pub index: Expr,
    pub is_write: bool,
}

impl SharedAccess {
    /// Check if two accesses conflict. Two accesses conflict when they target
    /// the same memory region and at least one is a write, and their indices
    /// are either equal or cannot be proven disjoint statically.
    pub fn conflicts_with(&self, other: &SharedAccess) -> bool {
        if self.memory != other.memory {
            return false;
        }
        if !self.is_write && !other.is_write {
            return false;
        }
        // If both indices are constant and different, no conflict.
        if let (Some(a), Some(b)) = (self.index.eval_const_int(), other.index.eval_const_int()) {
            return a == b;
        }
        // If both are the same expression (structurally), they might conflict.
        // Conservative: assume conflict.
        true
    }
}

fn expr_reads_shared(expr: &Expr) -> bool {
    match expr {
        Expr::SharedRead(_, _) => true,
        Expr::BinOp(_, a, b) => expr_reads_shared(a) || expr_reads_shared(b),
        Expr::UnaryOp(_, e) | Expr::Cast(e, _) => expr_reads_shared(e),
        Expr::ArrayIndex(a, i) => expr_reads_shared(a) || expr_reads_shared(i),
        Expr::FunctionCall(_, args) => args.iter().any(|a| expr_reads_shared(a)),
        Expr::Conditional(c, t, e) => {
            expr_reads_shared(c) || expr_reads_shared(t) || expr_reads_shared(e)
        }
        _ => false,
    }
}

fn collect_reads_from_expr(expr: &Expr, accesses: &mut Vec<SharedAccess>) {
    match expr {
        Expr::SharedRead(mem, idx) => {
            if let Expr::Variable(name) = mem.as_ref() {
                accesses.push(SharedAccess {
                    memory: name.clone(),
                    index: *idx.clone(),
                    is_write: false,
                });
            }
            collect_reads_from_expr(mem, accesses);
            collect_reads_from_expr(idx, accesses);
        }
        Expr::BinOp(_, a, b) | Expr::ArrayIndex(a, b) => {
            collect_reads_from_expr(a, accesses);
            collect_reads_from_expr(b, accesses);
        }
        Expr::UnaryOp(_, e) | Expr::Cast(e, _) => collect_reads_from_expr(e, accesses),
        Expr::FunctionCall(_, args) => {
            for arg in args {
                collect_reads_from_expr(arg, accesses);
            }
        }
        Expr::Conditional(c, t, e) => {
            collect_reads_from_expr(c, accesses);
            collect_reads_from_expr(t, accesses);
            collect_reads_from_expr(e, accesses);
        }
        _ => {}
    }
}

/// Declaration of a shared memory region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedMemoryDecl {
    pub name: String,
    pub elem_type: PramType,
    pub size: Expr,
}

/// A parameter to a PRAM algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub param_type: PramType,
}

/// A complete PRAM program / algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PramProgram {
    pub name: String,
    pub memory_model: MemoryModel,
    pub parameters: Vec<Parameter>,
    pub shared_memory: Vec<SharedMemoryDecl>,
    pub body: Vec<Stmt>,
    pub num_processors: Expr,
    pub work_bound: Option<String>,
    pub time_bound: Option<String>,
    pub description: Option<String>,
}

impl PramProgram {
    pub fn new(name: &str, model: MemoryModel) -> Self {
        Self {
            name: name.to_string(),
            memory_model: model,
            parameters: Vec::new(),
            shared_memory: Vec::new(),
            body: Vec::new(),
            num_processors: Expr::IntLiteral(1),
            work_bound: None,
            time_bound: None,
            description: None,
        }
    }

    /// Total statement count in the program.
    pub fn total_stmts(&self) -> usize {
        self.body.iter().map(|s| s.stmt_count()).sum()
    }

    /// Whether the program uses concurrent writes.
    pub fn uses_concurrent_writes(&self) -> bool {
        self.memory_model.allows_concurrent_write()
            && self.body.iter().any(|s| s.writes_shared())
    }

    /// Whether the program uses concurrent reads.
    pub fn uses_concurrent_reads(&self) -> bool {
        self.memory_model.allows_concurrent_read()
            && self.body.iter().any(|s| s.reads_shared())
    }

    /// Count the number of parallel steps (parallel_for statements at top level).
    pub fn parallel_step_count(&self) -> usize {
        count_parallel_steps(&self.body)
    }

    /// Estimate the number of processors used by the program.
    pub fn processor_count(&self) -> Option<usize> {
        match &self.num_processors {
            Expr::IntLiteral(n) => Some(*n as usize),
            _ => None,
        }
    }

    /// Collect all shared memory region names.
    pub fn shared_region_names(&self) -> Vec<String> {
        self.shared_memory.iter().map(|s| s.name.clone()).collect()
    }

    /// Add a parameter to the program.
    pub fn add_parameter(&mut self, name: &str, ty: PramType) {
        self.parameters.push(Parameter {
            name: name.to_string(),
            param_type: ty,
        });
    }

    /// Add a shared memory region to the program.
    pub fn add_shared_memory(&mut self, name: &str, elem_type: PramType, size: Expr) {
        self.shared_memory.push(SharedMemoryDecl {
            name: name.to_string(),
            elem_type,
            size,
        });
    }

    /// Perform basic validation of the program structure, returning a list
    /// of human-readable issue descriptions.
    pub fn validate_basic(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if self.name.is_empty() {
            issues.push("Program name is empty".to_string());
        }
        if self.body.is_empty() {
            issues.push("Program body is empty".to_string());
        }
        if let Some(n) = self.num_processors.eval_const_int() {
            if n <= 0 {
                issues.push(format!("Number of processors must be positive, got {}", n));
            }
        }
        // Check for duplicate shared memory names.
        let mut seen = std::collections::HashSet::new();
        for s in &self.shared_memory {
            if !seen.insert(&s.name) {
                issues.push(format!("Duplicate shared memory region: {}", s.name));
            }
        }
        // Check for duplicate parameter names.
        let mut param_seen = std::collections::HashSet::new();
        for p in &self.parameters {
            if !param_seen.insert(&p.name) {
                issues.push(format!("Duplicate parameter: {}", p.name));
            }
        }
        issues
    }

    /// Collect all variable names referenced across the entire program.
    pub fn collect_all_variables(&self) -> std::collections::HashSet<String> {
        let mut vars = std::collections::HashSet::new();
        for p in &self.parameters {
            vars.insert(p.name.clone());
        }
        for s in &self.shared_memory {
            vars.insert(s.name.clone());
            for v in s.size.collect_variables() {
                vars.insert(v);
            }
        }
        for v in self.num_processors.collect_variables() {
            vars.insert(v);
        }
        for stmt in &self.body {
            collect_vars_from_stmt(stmt, &mut vars);
        }
        vars
    }
}

fn collect_vars_from_stmt(stmt: &Stmt, vars: &mut std::collections::HashSet<String>) {
    match stmt {
        Stmt::LocalDecl(name, _, init) => {
            vars.insert(name.clone());
            if let Some(e) = init {
                for v in e.collect_variables() {
                    vars.insert(v);
                }
            }
        }
        Stmt::Assign(name, expr) => {
            vars.insert(name.clone());
            for v in expr.collect_variables() {
                vars.insert(v);
            }
        }
        Stmt::SharedWrite { memory, index, value } => {
            for e in [memory, index, value] {
                for v in e.collect_variables() {
                    vars.insert(v);
                }
            }
        }
        Stmt::ParallelFor { proc_var, num_procs, body } => {
            vars.insert(proc_var.clone());
            for v in num_procs.collect_variables() {
                vars.insert(v);
            }
            for s in body {
                collect_vars_from_stmt(s, vars);
            }
        }
        Stmt::SeqFor { var, start, end, step, body } => {
            vars.insert(var.clone());
            for v in start.collect_variables() {
                vars.insert(v);
            }
            for v in end.collect_variables() {
                vars.insert(v);
            }
            if let Some(s) = step {
                for v in s.collect_variables() {
                    vars.insert(v);
                }
            }
            for s in body {
                collect_vars_from_stmt(s, vars);
            }
        }
        Stmt::While { condition, body } => {
            for v in condition.collect_variables() {
                vars.insert(v);
            }
            for s in body {
                collect_vars_from_stmt(s, vars);
            }
        }
        Stmt::If { condition, then_body, else_body } => {
            for v in condition.collect_variables() {
                vars.insert(v);
            }
            for s in then_body {
                collect_vars_from_stmt(s, vars);
            }
            for s in else_body {
                collect_vars_from_stmt(s, vars);
            }
        }
        Stmt::Block(stmts) => {
            for s in stmts {
                collect_vars_from_stmt(s, vars);
            }
        }
        Stmt::ExprStmt(expr) | Stmt::Assert(expr, _) => {
            for v in expr.collect_variables() {
                vars.insert(v);
            }
        }
        Stmt::Return(Some(expr)) => {
            for v in expr.collect_variables() {
                vars.insert(v);
            }
        }
        Stmt::AtomicCAS { memory, index, expected, desired, result_var } => {
            vars.insert(result_var.clone());
            for e in [memory, index, expected, desired] {
                for v in e.collect_variables() {
                    vars.insert(v);
                }
            }
        }
        Stmt::FetchAdd { memory, index, value, result_var } => {
            vars.insert(result_var.clone());
            for e in [memory, index, value] {
                for v in e.collect_variables() {
                    vars.insert(v);
                }
            }
        }
        Stmt::AllocShared { name, size, .. } => {
            vars.insert(name.clone());
            for v in size.collect_variables() {
                vars.insert(v);
            }
        }
        Stmt::FreeShared(name) => {
            vars.insert(name.clone());
        }
        Stmt::PrefixSum { input, output, size, .. } => {
            vars.insert(input.clone());
            vars.insert(output.clone());
            for v in size.collect_variables() {
                vars.insert(v);
            }
        }
        _ => {}
    }
}

fn count_parallel_steps(stmts: &[Stmt]) -> usize {
    let mut count = 0;
    for stmt in stmts {
        match stmt {
            Stmt::ParallelFor { body, .. } => {
                count += 1;
                count += count_parallel_steps(body);
            }
            Stmt::SeqFor { body, .. } | Stmt::While { body, .. } | Stmt::Block(body) => {
                count += count_parallel_steps(body);
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                count += count_parallel_steps(then_body);
                count += count_parallel_steps(else_body);
            }
            _ => {}
        }
    }
    count
}

/// Represents a phase of a PRAM algorithm (between barriers).
#[derive(Debug, Clone)]
pub struct ParallelPhase {
    pub phase_id: usize,
    pub statements: Vec<Stmt>,
    pub num_processors: Expr,
}

/// Split a PRAM program body into phases separated by barriers.
pub fn split_into_phases(body: &[Stmt], num_procs: &Expr) -> Vec<ParallelPhase> {
    let mut phases = Vec::new();
    let mut current = Vec::new();
    let mut phase_id = 0;

    for stmt in body {
        match stmt {
            Stmt::Barrier => {
                if !current.is_empty() {
                    phases.push(ParallelPhase {
                        phase_id,
                        statements: std::mem::take(&mut current),
                        num_processors: num_procs.clone(),
                    });
                    phase_id += 1;
                }
            }
            _ => {
                current.push(stmt.clone());
            }
        }
    }

    if !current.is_empty() {
        phases.push(ParallelPhase {
            phase_id,
            statements: current,
            num_processors: num_procs.clone(),
        });
    }

    phases
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_int_literal() {
        let e = Expr::int(42);
        assert_eq!(e.eval_const_int(), Some(42));
        assert!(e.is_constant());
    }

    #[test]
    fn test_expr_binop() {
        let e = Expr::binop(BinOp::Add, Expr::int(3), Expr::int(4));
        assert_eq!(e.eval_const_int(), Some(7));
        assert!(e.is_constant());
    }

    #[test]
    fn test_expr_variables() {
        let e = Expr::binop(
            BinOp::Add,
            Expr::var("x"),
            Expr::binop(BinOp::Mul, Expr::var("y"), Expr::int(2)),
        );
        let vars = e.collect_variables();
        assert!(vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_expr_substitute() {
        let e = Expr::binop(BinOp::Add, Expr::var("x"), Expr::int(1));
        let result = e.substitute("x", &Expr::int(42));
        assert_eq!(result.eval_const_int(), Some(43));
    }

    #[test]
    fn test_stmt_count() {
        let stmt = Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::int(8),
            body: vec![
                Stmt::Assign("x".to_string(), Expr::int(0)),
                Stmt::SharedWrite {
                    memory: Expr::var("A"),
                    index: Expr::ProcessorId,
                    value: Expr::int(1),
                },
            ],
        };
        assert_eq!(stmt.stmt_count(), 3);
    }

    #[test]
    fn test_binop_precedence() {
        assert!(BinOp::Mul.precedence() > BinOp::Add.precedence());
        assert!(BinOp::Add.precedence() > BinOp::Lt.precedence());
        assert!(BinOp::Lt.precedence() > BinOp::And.precedence());
    }

    #[test]
    fn test_memory_model() {
        assert!(!MemoryModel::EREW.allows_concurrent_read());
        assert!(!MemoryModel::EREW.allows_concurrent_write());
        assert!(MemoryModel::CREW.allows_concurrent_read());
        assert!(!MemoryModel::CREW.allows_concurrent_write());
        assert!(MemoryModel::CRCWPriority.allows_concurrent_read());
        assert!(MemoryModel::CRCWPriority.allows_concurrent_write());
    }

    #[test]
    fn test_pram_program_new() {
        let prog = PramProgram::new("test", MemoryModel::CREW);
        assert_eq!(prog.name, "test");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.parameters.is_empty());
    }

    #[test]
    fn test_split_into_phases() {
        let body = vec![
            Stmt::Assign("x".to_string(), Expr::int(0)),
            Stmt::Barrier,
            Stmt::Assign("y".to_string(), Expr::int(1)),
            Stmt::Barrier,
            Stmt::Assign("z".to_string(), Expr::int(2)),
        ];
        let phases = split_into_phases(&body, &Expr::int(8));
        assert_eq!(phases.len(), 3);
        assert_eq!(phases[0].phase_id, 0);
        assert_eq!(phases[1].phase_id, 1);
        assert_eq!(phases[2].phase_id, 2);
    }

    #[test]
    fn test_expr_node_count() {
        let e = Expr::binop(
            BinOp::Add,
            Expr::var("x"),
            Expr::binop(BinOp::Mul, Expr::int(2), Expr::int(3)),
        );
        assert_eq!(e.node_count(), 5);
    }

    #[test]
    fn test_contains_barrier() {
        let s = Stmt::Block(vec![
            Stmt::Assign("x".to_string(), Expr::int(0)),
            Stmt::Barrier,
        ]);
        assert!(s.contains_barrier());

        let s2 = Stmt::Assign("x".to_string(), Expr::int(0));
        assert!(!s2.contains_barrier());
    }

    #[test]
    fn test_const_eval_division() {
        let e = Expr::binop(BinOp::Div, Expr::int(10), Expr::int(3));
        assert_eq!(e.eval_const_int(), Some(3));
        let e2 = Expr::binop(BinOp::Div, Expr::int(10), Expr::int(0));
        assert_eq!(e2.eval_const_int(), None);
    }

    #[test]
    fn test_expr_depth() {
        assert_eq!(Expr::int(1).depth(), 1);
        assert_eq!(Expr::var("x").depth(), 1);
        let e = Expr::binop(BinOp::Add, Expr::int(1), Expr::int(2));
        assert_eq!(e.depth(), 2);
        let deep = Expr::binop(
            BinOp::Add,
            Expr::binop(BinOp::Mul, Expr::var("x"), Expr::int(2)),
            Expr::int(3),
        );
        assert_eq!(deep.depth(), 3);
    }

    #[test]
    fn test_expr_map_children() {
        let e = Expr::binop(BinOp::Add, Expr::int(1), Expr::int(2));
        let mapped = e.map_children(|child| {
            if let Expr::IntLiteral(v) = child {
                Expr::IntLiteral(v * 10)
            } else {
                child.clone()
            }
        });
        assert_eq!(mapped.eval_const_int(), Some(30));
    }

    #[test]
    fn test_stmt_transform_exprs() {
        let stmt = Stmt::Assign("x".to_string(), Expr::int(5));
        let transformed = stmt.transform_exprs(&|e| {
            if let Expr::IntLiteral(v) = e {
                Expr::IntLiteral(v + 1)
            } else {
                e.clone()
            }
        });
        match transformed {
            Stmt::Assign(_, Expr::IntLiteral(6)) => {}
            other => panic!("expected Assign with 6, got {:?}", other),
        }
    }

    #[test]
    fn test_stmt_flatten_blocks() {
        let stmt = Stmt::Block(vec![
            Stmt::Assign("x".to_string(), Expr::int(1)),
            Stmt::Block(vec![
                Stmt::Assign("y".to_string(), Expr::int(2)),
                Stmt::Assign("z".to_string(), Expr::int(3)),
            ]),
        ]);
        let flat = stmt.flatten_blocks();
        assert_eq!(flat.len(), 3);
    }

    #[test]
    fn test_shared_access_conflicts_with() {
        let a = SharedAccess { memory: "A".to_string(), index: Expr::int(0), is_write: true };
        let b = SharedAccess { memory: "A".to_string(), index: Expr::int(0), is_write: false };
        assert!(a.conflicts_with(&b));

        let c = SharedAccess { memory: "A".to_string(), index: Expr::int(1), is_write: true };
        assert!(!a.conflicts_with(&c));

        let d = SharedAccess { memory: "B".to_string(), index: Expr::int(0), is_write: true };
        assert!(!a.conflicts_with(&d));

        // Two reads don't conflict
        let e = SharedAccess { memory: "A".to_string(), index: Expr::int(0), is_write: false };
        assert!(!b.conflicts_with(&e));
    }

    #[test]
    fn test_program_validate_basic() {
        let prog = PramProgram::new("test", MemoryModel::EREW);
        let issues = prog.validate_basic();
        assert!(issues.iter().any(|i| i.contains("body is empty")));

        let mut prog2 = PramProgram::new("", MemoryModel::EREW);
        prog2.body.push(Stmt::Nop);
        let issues2 = prog2.validate_basic();
        assert!(issues2.iter().any(|i| i.contains("name is empty")));
    }

    #[test]
    fn test_program_collect_all_variables() {
        let mut prog = PramProgram::new("test", MemoryModel::EREW);
        prog.add_parameter("n", PramType::Int64);
        prog.add_shared_memory("A", PramType::Int64, Expr::var("n"));
        prog.body.push(Stmt::Assign("x".to_string(), Expr::var("n")));
        let vars = prog.collect_all_variables();
        assert!(vars.contains("n"));
        assert!(vars.contains("A"));
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_program_add_parameter_and_shared() {
        let mut prog = PramProgram::new("test", MemoryModel::CREW);
        prog.add_parameter("n", PramType::Int64);
        prog.add_shared_memory("A", PramType::Int64, Expr::var("n"));
        assert_eq!(prog.parameters.len(), 1);
        assert_eq!(prog.parameters[0].name, "n");
        assert_eq!(prog.shared_memory.len(), 1);
        assert_eq!(prog.shared_memory[0].name, "A");
    }

    #[test]
    fn test_validate_basic_duplicate_params() {
        let mut prog = PramProgram::new("dup", MemoryModel::EREW);
        prog.add_parameter("n", PramType::Int64);
        prog.add_parameter("n", PramType::Int64);
        prog.body.push(Stmt::Nop);
        let issues = prog.validate_basic();
        assert!(issues.iter().any(|i| i.contains("Duplicate parameter")));
    }
}
