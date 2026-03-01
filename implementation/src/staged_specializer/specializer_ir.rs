//! Specializer IR: a simplified intermediate representation after specialization.
//!
//! This IR strips away PRAM-specific constructs (parallel_for, shared memory models)
//! and represents the program as basic blocks with sequential operations.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::pram_ir::ast::BinOp;

/// Simplified type system for the specializer IR.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SType {
    I64,
    F64,
    Bool,
    Ptr,
}

impl fmt::Display for SType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SType::I64 => write!(f, "i64"),
            SType::F64 => write!(f, "f64"),
            SType::Bool => write!(f, "bool"),
            SType::Ptr => write!(f, "ptr"),
        }
    }
}

/// Unique identifier for basic blocks.
pub type BlockId = usize;

/// Simplified expressions in the specializer IR.
#[derive(Debug, Clone, PartialEq)]
pub enum SExpr {
    /// Integer constant
    IntConst(i64),
    /// Float constant
    FloatConst(f64),
    /// Boolean constant
    BoolConst(bool),
    /// Variable reference
    Var(String),
    /// Binary operation
    BinOp(BinOp, Box<SExpr>, Box<SExpr>),
    /// Direct memory load at a resolved offset
    Load {
        base: String,
        offset: Box<SExpr>,
    },
    /// Function call
    Call(String, Vec<SExpr>),
}

impl SExpr {
    pub fn int(v: i64) -> Self {
        SExpr::IntConst(v)
    }

    pub fn var(name: &str) -> Self {
        SExpr::Var(name.to_string())
    }

    pub fn binop(op: BinOp, left: SExpr, right: SExpr) -> Self {
        SExpr::BinOp(op, Box::new(left), Box::new(right))
    }

    pub fn load(base: &str, offset: SExpr) -> Self {
        SExpr::Load {
            base: base.to_string(),
            offset: Box::new(offset),
        }
    }

    /// Returns true if this expression is a compile-time constant.
    pub fn is_constant(&self) -> bool {
        match self {
            SExpr::IntConst(_) | SExpr::FloatConst(_) | SExpr::BoolConst(_) => true,
            SExpr::BinOp(_, a, b) => a.is_constant() && b.is_constant(),
            _ => false,
        }
    }

    /// Evaluate a constant integer expression.
    pub fn eval_const_int(&self) -> Option<i64> {
        match self {
            SExpr::IntConst(v) => Some(*v),
            SExpr::BoolConst(b) => Some(if *b { 1 } else { 0 }),
            SExpr::BinOp(op, a, b) => {
                let av = a.eval_const_int()?;
                let bv = b.eval_const_int()?;
                Some(match op {
                    BinOp::Add => av.wrapping_add(bv),
                    BinOp::Sub => av.wrapping_sub(bv),
                    BinOp::Mul => av.wrapping_mul(bv),
                    BinOp::Div if bv != 0 => av / bv,
                    BinOp::Mod if bv != 0 => av % bv,
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
                    _ => return None,
                })
            }
            _ => None,
        }
    }

    /// Count the number of sub-expressions.
    pub fn node_count(&self) -> usize {
        match self {
            SExpr::IntConst(_) | SExpr::FloatConst(_) | SExpr::BoolConst(_) | SExpr::Var(_) => 1,
            SExpr::BinOp(_, a, b) => 1 + a.node_count() + b.node_count(),
            SExpr::Load { offset, .. } => 1 + offset.node_count(),
            SExpr::Call(_, args) => 1 + args.iter().map(|a| a.node_count()).sum::<usize>(),
        }
    }

    /// Collect all variable names referenced.
    pub fn collect_vars(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_vars_into(&mut vars);
        vars
    }

    fn collect_vars_into(&self, vars: &mut Vec<String>) {
        match self {
            SExpr::Var(name) => {
                if !vars.contains(name) {
                    vars.push(name.clone());
                }
            }
            SExpr::BinOp(_, a, b) => {
                a.collect_vars_into(vars);
                b.collect_vars_into(vars);
            }
            SExpr::Load { offset, .. } => offset.collect_vars_into(vars),
            SExpr::Call(_, args) => {
                for arg in args {
                    arg.collect_vars_into(vars);
                }
            }
            _ => {}
        }
    }

    /// Compute the complexity of this expression.
    /// Weights: constants=0, vars=1, binops=2, loads=3, calls=4.
    pub fn complexity(&self) -> usize {
        match self {
            SExpr::IntConst(_) | SExpr::FloatConst(_) | SExpr::BoolConst(_) => 0,
            SExpr::Var(_) => 1,
            SExpr::BinOp(_, a, b) => 2 + a.complexity() + b.complexity(),
            SExpr::Load { offset, .. } => 3 + offset.complexity(),
            SExpr::Call(_, args) => 4 + args.iter().map(|a| a.complexity()).sum::<usize>(),
        }
    }

    /// Substitute a variable with another expression.
    pub fn substitute(&self, var: &str, replacement: &SExpr) -> SExpr {
        match self {
            SExpr::Var(name) if name == var => replacement.clone(),
            SExpr::BinOp(op, a, b) => SExpr::BinOp(
                *op,
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            SExpr::Load { base, offset } => SExpr::Load {
                base: base.clone(),
                offset: Box::new(offset.substitute(var, replacement)),
            },
            SExpr::Call(name, args) => SExpr::Call(
                name.clone(),
                args.iter().map(|a| a.substitute(var, replacement)).collect(),
            ),
            other => other.clone(),
        }
    }
}

impl fmt::Display for SExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SExpr::IntConst(v) => write!(f, "{}", v),
            SExpr::FloatConst(v) => write!(f, "{:.6}", v),
            SExpr::BoolConst(b) => write!(f, "{}", b),
            SExpr::Var(name) => write!(f, "{}", name),
            SExpr::BinOp(op, a, b) => write!(f, "({} {} {})", a, op, b),
            SExpr::Load { base, offset } => write!(f, "{}[{}]", base, offset),
            SExpr::Call(name, args) => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
        }
    }
}

/// Memory access descriptor with resolved block information.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryAccessDescriptor {
    /// Name of the shared memory region
    pub region: String,
    /// Resolved block ID (if known at compile time)
    pub resolved_block_id: Option<usize>,
    /// Offset within the block
    pub offset: SExpr,
    /// Whether this is a write access
    pub is_write: bool,
}

impl MemoryAccessDescriptor {
    pub fn new_read(region: &str, offset: SExpr) -> Self {
        Self {
            region: region.to_string(),
            resolved_block_id: None,
            offset,
            is_write: false,
        }
    }

    pub fn new_write(region: &str, offset: SExpr) -> Self {
        Self {
            region: region.to_string(),
            resolved_block_id: None,
            offset,
            is_write: true,
        }
    }

    pub fn with_block_id(mut self, block_id: usize) -> Self {
        self.resolved_block_id = Some(block_id);
        self
    }
}

/// Instructions in the specializer IR.
#[derive(Debug, Clone, PartialEq)]
pub enum SInstr {
    /// Assign expression result to a variable
    Assign {
        dst: String,
        src: SExpr,
        ty: SType,
    },
    /// Load from memory with known offset
    Load {
        dst: String,
        descriptor: MemoryAccessDescriptor,
    },
    /// Store to memory
    Store {
        descriptor: MemoryAccessDescriptor,
        value: SExpr,
    },
    /// Sequential loop
    Loop {
        var: String,
        start: SExpr,
        end: SExpr,
        body: Vec<SInstr>,
    },
    /// Conditional branch
    Conditional {
        condition: SExpr,
        then_body: Vec<SInstr>,
        else_body: Vec<SInstr>,
    },
    /// Function call (side-effecting)
    Call {
        result: Option<String>,
        function: String,
        args: Vec<SExpr>,
    },
    /// Return value
    Return(Option<SExpr>),
    /// Debug/documentation comment
    Comment(String),
    /// No-op (placeholder)
    Nop,
    /// Assertion with message
    Assert(SExpr, String),
}

impl SInstr {
    /// Count instructions recursively.
    pub fn instr_count(&self) -> usize {
        match self {
            SInstr::Loop { body, .. } => 1 + body.iter().map(|i| i.instr_count()).sum::<usize>(),
            SInstr::Conditional {
                then_body,
                else_body,
                ..
            } => {
                1 + then_body.iter().map(|i| i.instr_count()).sum::<usize>()
                    + else_body.iter().map(|i| i.instr_count()).sum::<usize>()
            }
            SInstr::Nop => 0,
            SInstr::Comment(_) => 0,
            _ => 1,
        }
    }

    /// Check if this instruction is a no-op or comment (non-operational).
    pub fn is_non_operational(&self) -> bool {
        matches!(self, SInstr::Nop | SInstr::Comment(_))
    }

    /// Check if this instruction has side effects (memory writes, calls, assertions).
    pub fn has_side_effects(&self) -> bool {
        match self {
            SInstr::Store { .. } => true,
            SInstr::Call { .. } => true,
            SInstr::Assert(_, _) => true,
            SInstr::Return(_) => true,
            SInstr::Loop { body, .. } => body.iter().any(|i| i.has_side_effects()),
            SInstr::Conditional {
                then_body,
                else_body,
                ..
            } => {
                then_body.iter().any(|i| i.has_side_effects())
                    || else_body.iter().any(|i| i.has_side_effects())
            }
            SInstr::Assign { .. } | SInstr::Load { .. } => false,
            SInstr::Nop | SInstr::Comment(_) => false,
        }
    }
}

impl fmt::Display for SInstr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, 0)
    }
}

impl SInstr {
    fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = "  ".repeat(indent);
        match self {
            SInstr::Assign { dst, src, ty } => {
                writeln!(f, "{}let {}: {} = {};", pad, dst, ty, src)
            }
            SInstr::Load { dst, descriptor } => {
                if let Some(bid) = descriptor.resolved_block_id {
                    writeln!(
                        f,
                        "{}{} = {}[block={}, offset={}];",
                        pad, dst, descriptor.region, bid, descriptor.offset
                    )
                } else {
                    writeln!(
                        f,
                        "{}{} = {}[{}];",
                        pad, dst, descriptor.region, descriptor.offset
                    )
                }
            }
            SInstr::Store { descriptor, value } => {
                if let Some(bid) = descriptor.resolved_block_id {
                    writeln!(
                        f,
                        "{}{}[block={}, offset={}] = {};",
                        pad, descriptor.region, bid, descriptor.offset, value
                    )
                } else {
                    writeln!(
                        f,
                        "{}{}[{}] = {};",
                        pad, descriptor.region, descriptor.offset, value
                    )
                }
            }
            SInstr::Loop {
                var,
                start,
                end,
                body,
            } => {
                writeln!(f, "{}for {} in {}..{} {{", pad, var, start, end)?;
                for instr in body {
                    instr.fmt_indent(f, indent + 1)?;
                }
                writeln!(f, "{}}}", pad)
            }
            SInstr::Conditional {
                condition,
                then_body,
                else_body,
            } => {
                writeln!(f, "{}if {} {{", pad, condition)?;
                for instr in then_body {
                    instr.fmt_indent(f, indent + 1)?;
                }
                if !else_body.is_empty() {
                    writeln!(f, "{}}} else {{", pad)?;
                    for instr in else_body {
                        instr.fmt_indent(f, indent + 1)?;
                    }
                }
                writeln!(f, "{}}}", pad)
            }
            SInstr::Call {
                result,
                function,
                args,
            } => {
                let args_str: Vec<String> = args.iter().map(|a| format!("{}", a)).collect();
                if let Some(r) = result {
                    writeln!(f, "{}{} = {}({});", pad, r, function, args_str.join(", "))
                } else {
                    writeln!(f, "{}{}({});", pad, function, args_str.join(", "))
                }
            }
            SInstr::Return(Some(expr)) => writeln!(f, "{}return {};", pad, expr),
            SInstr::Return(None) => writeln!(f, "{}return;", pad),
            SInstr::Comment(text) => writeln!(f, "{}// {}", pad, text),
            SInstr::Nop => Ok(()),
            SInstr::Assert(expr, msg) => writeln!(f, "{}assert({}, \"{}\");", pad, expr, msg),
        }
    }
}

/// A basic block in the specializer IR.
#[derive(Debug, Clone)]
pub struct SBlock {
    pub id: BlockId,
    pub label: String,
    pub instructions: Vec<SInstr>,
    /// Successor block IDs
    pub successors: Vec<BlockId>,
}

impl SBlock {
    pub fn new(id: BlockId, label: &str) -> Self {
        Self {
            id,
            label: label.to_string(),
            instructions: Vec::new(),
            successors: Vec::new(),
        }
    }

    pub fn push(&mut self, instr: SInstr) {
        self.instructions.push(instr);
    }

    pub fn instr_count(&self) -> usize {
        self.instructions.iter().map(|i| i.instr_count()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.instructions.iter().all(|i| i.is_non_operational())
    }

    /// Compute the set of live variables at entry to this block.
    /// A variable is live if it is used before being defined.
    pub fn liveness_analysis(&self) -> HashSet<String> {
        let mut live = HashSet::new();
        // Walk instructions in reverse
        for instr in self.instructions.iter().rev() {
            Self::liveness_instr(instr, &mut live);
        }
        live
    }

    fn liveness_instr(instr: &SInstr, live: &mut HashSet<String>) {
        match instr {
            SInstr::Assign { dst, src, .. } => {
                live.remove(dst);
                for v in src.collect_vars() {
                    live.insert(v);
                }
            }
            SInstr::Load { dst, descriptor } => {
                live.remove(dst);
                for v in descriptor.offset.collect_vars() {
                    live.insert(v);
                }
            }
            SInstr::Store { descriptor, value } => {
                for v in descriptor.offset.collect_vars() {
                    live.insert(v);
                }
                for v in value.collect_vars() {
                    live.insert(v);
                }
            }
            SInstr::Loop { var, start, end, body } => {
                live.remove(var);
                for v in start.collect_vars() {
                    live.insert(v);
                }
                for v in end.collect_vars() {
                    live.insert(v);
                }
                for i in body.iter().rev() {
                    Self::liveness_instr(i, live);
                }
            }
            SInstr::Conditional { condition, then_body, else_body } => {
                for v in condition.collect_vars() {
                    live.insert(v);
                }
                for i in then_body.iter().rev() {
                    Self::liveness_instr(i, live);
                }
                for i in else_body.iter().rev() {
                    Self::liveness_instr(i, live);
                }
            }
            SInstr::Call { result, args, .. } => {
                if let Some(r) = result {
                    live.remove(r);
                }
                for arg in args {
                    for v in arg.collect_vars() {
                        live.insert(v);
                    }
                }
            }
            SInstr::Return(Some(e)) | SInstr::Assert(e, _) => {
                for v in e.collect_vars() {
                    live.insert(v);
                }
            }
            _ => {}
        }
    }
}

impl fmt::Display for SBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "block {} ({}):", self.id, self.label)?;
        for instr in &self.instructions {
            write!(f, "  {}", instr)?;
        }
        if !self.successors.is_empty() {
            writeln!(
                f,
                "  -> [{}]",
                self.successors
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;
        }
        Ok(())
    }
}

/// A complete specialized program.
#[derive(Debug, Clone)]
pub struct SProgram {
    pub name: String,
    pub blocks: HashMap<BlockId, SBlock>,
    pub entry: BlockId,
    /// Local variable declarations: (name, type)
    pub locals: Vec<(String, SType)>,
    /// Memory region declarations: (name, element_size, count)
    pub memory_regions: Vec<(String, usize, usize)>,
    next_block_id: BlockId,
}

impl SProgram {
    pub fn new(name: &str) -> Self {
        let entry_block = SBlock::new(0, "entry");
        let mut blocks = HashMap::new();
        blocks.insert(0, entry_block);
        Self {
            name: name.to_string(),
            blocks,
            entry: 0,
            locals: Vec::new(),
            memory_regions: Vec::new(),
            next_block_id: 1,
        }
    }

    /// Create a new basic block and return its ID.
    pub fn new_block(&mut self, label: &str) -> BlockId {
        let id = self.next_block_id;
        self.next_block_id += 1;
        let block = SBlock::new(id, label);
        self.blocks.insert(id, block);
        id
    }

    /// Get a mutable reference to a block.
    pub fn block_mut(&mut self, id: BlockId) -> Option<&mut SBlock> {
        self.blocks.get_mut(&id)
    }

    /// Get a reference to a block.
    pub fn block(&self, id: BlockId) -> Option<&SBlock> {
        self.blocks.get(&id)
    }

    /// Add a local variable declaration.
    pub fn add_local(&mut self, name: &str, ty: SType) {
        if !self.locals.iter().any(|(n, _)| n == name) {
            self.locals.push((name.to_string(), ty));
        }
    }

    /// Add a memory region.
    pub fn add_memory_region(&mut self, name: &str, elem_size: usize, count: usize) {
        self.memory_regions
            .push((name.to_string(), elem_size, count));
    }

    /// Total instruction count across all blocks.
    pub fn total_instr_count(&self) -> usize {
        self.blocks.values().map(|b| b.instr_count()).sum()
    }

    /// Number of blocks.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Collect all instructions in program order starting from entry.
    pub fn collect_instructions(&self) -> Vec<&SInstr> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut worklist = vec![self.entry];
        while let Some(bid) = worklist.pop() {
            if !visited.insert(bid) {
                continue;
            }
            if let Some(block) = self.blocks.get(&bid) {
                for instr in &block.instructions {
                    result.push(instr);
                }
                for &succ in block.successors.iter().rev() {
                    worklist.push(succ);
                }
            }
        }
        result
    }

    /// Basic optimization pass: fold constants in all instructions and remove nops.
    pub fn optimize(&self) -> SProgram {
        let mut result = self.clone();
        let block_ids: Vec<BlockId> = result.blocks.keys().cloned().collect();
        for bid in block_ids {
            if let Some(block) = result.blocks.get_mut(&bid) {
                block.instructions = block
                    .instructions
                    .iter()
                    .filter(|i| !i.is_non_operational())
                    .map(|i| Self::optimize_instr(i))
                    .collect();
            }
        }
        result
    }

    fn optimize_instr(instr: &SInstr) -> SInstr {
        match instr {
            SInstr::Assign { dst, src, ty } => SInstr::Assign {
                dst: dst.clone(),
                src: Self::fold_expr(src),
                ty: ty.clone(),
            },
            SInstr::Store { descriptor, value } => SInstr::Store {
                descriptor: MemoryAccessDescriptor {
                    region: descriptor.region.clone(),
                    resolved_block_id: descriptor.resolved_block_id,
                    offset: Self::fold_expr(&descriptor.offset),
                    is_write: descriptor.is_write,
                },
                value: Self::fold_expr(value),
            },
            SInstr::Loop { var, start, end, body } => SInstr::Loop {
                var: var.clone(),
                start: Self::fold_expr(start),
                end: Self::fold_expr(end),
                body: body.iter().map(|i| Self::optimize_instr(i)).collect(),
            },
            SInstr::Conditional { condition, then_body, else_body } => {
                let cond = Self::fold_expr(condition);
                // If condition is constant true, keep only then_body
                if let Some(1) = cond.eval_const_int() {
                    if then_body.len() == 1 {
                        return Self::optimize_instr(&then_body[0]);
                    }
                }
                SInstr::Conditional {
                    condition: cond,
                    then_body: then_body.iter().map(|i| Self::optimize_instr(i)).collect(),
                    else_body: else_body.iter().map(|i| Self::optimize_instr(i)).collect(),
                }
            }
            SInstr::Assert(e, msg) => SInstr::Assert(Self::fold_expr(e), msg.clone()),
            SInstr::Return(Some(e)) => SInstr::Return(Some(Self::fold_expr(e))),
            other => other.clone(),
        }
    }

    fn fold_expr(expr: &SExpr) -> SExpr {
        match expr {
            SExpr::BinOp(op, a, b) => {
                let fa = Self::fold_expr(a);
                let fb = Self::fold_expr(b);
                let folded = SExpr::BinOp(*op, Box::new(fa.clone()), Box::new(fb.clone()));
                if let Some(v) = folded.eval_const_int() {
                    SExpr::IntConst(v)
                } else {
                    // Identity reductions
                    match (op, &fa, &fb) {
                        (BinOp::Add, _, SExpr::IntConst(0)) => fa,
                        (BinOp::Add, SExpr::IntConst(0), _) => fb,
                        (BinOp::Mul, _, SExpr::IntConst(1)) => fa,
                        (BinOp::Mul, SExpr::IntConst(1), _) => fb,
                        (BinOp::Mul, _, SExpr::IntConst(0)) | (BinOp::Mul, SExpr::IntConst(0), _) => {
                            SExpr::IntConst(0)
                        }
                        (BinOp::Sub, _, SExpr::IntConst(0)) => fa,
                        _ => SExpr::BinOp(*op, Box::new(fa), Box::new(fb)),
                    }
                }
            }
            SExpr::Load { base, offset } => SExpr::Load {
                base: base.clone(),
                offset: Box::new(Self::fold_expr(offset)),
            },
            SExpr::Call(name, args) => SExpr::Call(
                name.clone(),
                args.iter().map(|a| Self::fold_expr(a)).collect(),
            ),
            other => other.clone(),
        }
    }

    /// Eliminate dead (unreachable) blocks from the program.
    pub fn dead_block_elimination(&self) -> SProgram {
        let mut reachable = HashSet::new();
        let mut worklist = vec![self.entry];
        while let Some(bid) = worklist.pop() {
            if !reachable.insert(bid) {
                continue;
            }
            if let Some(block) = self.blocks.get(&bid) {
                for &succ in &block.successors {
                    worklist.push(succ);
                }
            }
        }
        let mut result = self.clone();
        result.blocks.retain(|id, _| reachable.contains(id));
        result
    }

    /// Generate a DOT graph representation for visualization.
    pub fn to_dot(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("digraph {} {{\n", self.name));
        out.push_str("  node [shape=record];\n");
        let mut ids: Vec<BlockId> = self.blocks.keys().cloned().collect();
        ids.sort();
        for id in &ids {
            if let Some(block) = self.blocks.get(id) {
                let instr_count = block.instr_count();
                let label = format!(
                    "B{}: {} | {} instrs",
                    id, block.label, instr_count
                );
                out.push_str(&format!("  B{} [label=\"{}\"];\n", id, label));
            }
        }
        for id in &ids {
            if let Some(block) = self.blocks.get(id) {
                for succ in &block.successors {
                    out.push_str(&format!("  B{} -> B{};\n", id, succ));
                }
            }
        }
        out.push_str("}\n");
        out
    }
}

impl fmt::Display for SProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "program {} {{", self.name)?;
        if !self.locals.is_empty() {
            writeln!(f, "  locals:")?;
            for (name, ty) in &self.locals {
                writeln!(f, "    {}: {}", name, ty)?;
            }
        }
        if !self.memory_regions.is_empty() {
            writeln!(f, "  memory:")?;
            for (name, elem_size, count) in &self.memory_regions {
                writeln!(f, "    {}[{}] (elem={}B)", name, count, elem_size)?;
            }
        }
        // Print blocks in order
        let mut ids: Vec<BlockId> = self.blocks.keys().cloned().collect();
        ids.sort();
        for id in ids {
            if let Some(block) = self.blocks.get(&id) {
                write!(f, "  {}", block)?;
            }
        }
        writeln!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pram_ir::ast::BinOp;

    #[test]
    fn test_sexpr_constant() {
        let e = SExpr::int(42);
        assert!(e.is_constant());
        assert_eq!(e.eval_const_int(), Some(42));
    }

    #[test]
    fn test_sexpr_binop_eval() {
        let e = SExpr::binop(BinOp::Add, SExpr::int(3), SExpr::int(7));
        assert!(e.is_constant());
        assert_eq!(e.eval_const_int(), Some(10));
    }

    #[test]
    fn test_sexpr_binop_mul() {
        let e = SExpr::binop(BinOp::Mul, SExpr::int(6), SExpr::int(7));
        assert_eq!(e.eval_const_int(), Some(42));
    }

    #[test]
    fn test_sexpr_non_constant() {
        let e = SExpr::binop(BinOp::Add, SExpr::var("x"), SExpr::int(1));
        assert!(!e.is_constant());
        assert_eq!(e.eval_const_int(), None);
    }

    #[test]
    fn test_sexpr_substitute() {
        let e = SExpr::binop(BinOp::Add, SExpr::var("x"), SExpr::int(1));
        let result = e.substitute("x", &SExpr::int(41));
        assert_eq!(result.eval_const_int(), Some(42));
    }

    #[test]
    fn test_sexpr_collect_vars() {
        let e = SExpr::binop(
            BinOp::Add,
            SExpr::var("a"),
            SExpr::binop(BinOp::Mul, SExpr::var("b"), SExpr::var("a")),
        );
        let vars = e.collect_vars();
        assert!(vars.contains(&"a".to_string()));
        assert!(vars.contains(&"b".to_string()));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_sexpr_node_count() {
        let e = SExpr::binop(BinOp::Add, SExpr::int(1), SExpr::int(2));
        assert_eq!(e.node_count(), 3);
    }

    #[test]
    fn test_sexpr_load() {
        let e = SExpr::load("mem", SExpr::int(5));
        assert!(!e.is_constant());
        let vars = e.collect_vars();
        assert!(vars.is_empty());
    }

    #[test]
    fn test_stype_display() {
        assert_eq!(format!("{}", SType::I64), "i64");
        assert_eq!(format!("{}", SType::F64), "f64");
        assert_eq!(format!("{}", SType::Bool), "bool");
        assert_eq!(format!("{}", SType::Ptr), "ptr");
    }

    #[test]
    fn test_sblock_creation() {
        let mut block = SBlock::new(0, "entry");
        assert!(block.is_empty());
        block.push(SInstr::Assign {
            dst: "x".to_string(),
            src: SExpr::int(42),
            ty: SType::I64,
        });
        assert!(!block.is_empty());
        assert_eq!(block.instr_count(), 1);
    }

    #[test]
    fn test_sblock_comments_are_non_operational() {
        let mut block = SBlock::new(0, "test");
        block.push(SInstr::Comment("hello".to_string()));
        block.push(SInstr::Nop);
        assert!(block.is_empty());
        assert_eq!(block.instr_count(), 0);
    }

    #[test]
    fn test_sprogram_creation() {
        let mut prog = SProgram::new("test");
        assert_eq!(prog.block_count(), 1);
        assert_eq!(prog.entry, 0);

        let b1 = prog.new_block("second");
        assert_eq!(b1, 1);
        assert_eq!(prog.block_count(), 2);
    }

    #[test]
    fn test_sprogram_locals() {
        let mut prog = SProgram::new("test");
        prog.add_local("x", SType::I64);
        prog.add_local("y", SType::F64);
        prog.add_local("x", SType::I64); // duplicate, should not add
        assert_eq!(prog.locals.len(), 2);
    }

    #[test]
    fn test_sprogram_instr_count() {
        let mut prog = SProgram::new("test");
        prog.block_mut(0).unwrap().push(SInstr::Assign {
            dst: "a".to_string(),
            src: SExpr::int(1),
            ty: SType::I64,
        });
        prog.block_mut(0).unwrap().push(SInstr::Assign {
            dst: "b".to_string(),
            src: SExpr::int(2),
            ty: SType::I64,
        });
        assert_eq!(prog.total_instr_count(), 2);
    }

    #[test]
    fn test_sinstr_loop_count() {
        let instr = SInstr::Loop {
            var: "i".to_string(),
            start: SExpr::int(0),
            end: SExpr::int(10),
            body: vec![
                SInstr::Assign {
                    dst: "x".to_string(),
                    src: SExpr::int(0),
                    ty: SType::I64,
                },
                SInstr::Assign {
                    dst: "y".to_string(),
                    src: SExpr::int(1),
                    ty: SType::I64,
                },
            ],
        };
        assert_eq!(instr.instr_count(), 3);
    }

    #[test]
    fn test_memory_access_descriptor() {
        let desc = MemoryAccessDescriptor::new_read("A", SExpr::int(5)).with_block_id(3);
        assert_eq!(desc.region, "A");
        assert_eq!(desc.resolved_block_id, Some(3));
        assert!(!desc.is_write);
    }

    #[test]
    fn test_sprogram_display() {
        let mut prog = SProgram::new("demo");
        prog.add_local("x", SType::I64);
        prog.block_mut(0).unwrap().push(SInstr::Assign {
            dst: "x".to_string(),
            src: SExpr::int(42),
            ty: SType::I64,
        });
        let output = format!("{}", prog);
        assert!(output.contains("program demo"));
        assert!(output.contains("x: i64"));
        assert!(output.contains("42"));
    }

    #[test]
    fn test_sexpr_display() {
        let e = SExpr::binop(BinOp::Add, SExpr::var("x"), SExpr::int(1));
        let s = format!("{}", e);
        assert_eq!(s, "(x + 1)");
    }

    #[test]
    fn test_sexpr_div_by_zero() {
        let e = SExpr::binop(BinOp::Div, SExpr::int(10), SExpr::int(0));
        assert_eq!(e.eval_const_int(), None);
    }

    #[test]
    fn test_sinstr_conditional_count() {
        let instr = SInstr::Conditional {
            condition: SExpr::BoolConst(true),
            then_body: vec![SInstr::Assign {
                dst: "x".to_string(),
                src: SExpr::int(1),
                ty: SType::I64,
            }],
            else_body: vec![SInstr::Assign {
                dst: "x".to_string(),
                src: SExpr::int(2),
                ty: SType::I64,
            }],
        };
        assert_eq!(instr.instr_count(), 3);
    }

    #[test]
    fn test_collect_instructions() {
        let mut prog = SProgram::new("test");
        prog.block_mut(0).unwrap().push(SInstr::Assign {
            dst: "a".to_string(),
            src: SExpr::int(1),
            ty: SType::I64,
        });
        let b1 = prog.new_block("next");
        prog.block_mut(b1).unwrap().push(SInstr::Assign {
            dst: "b".to_string(),
            src: SExpr::int(2),
            ty: SType::I64,
        });
        prog.block_mut(0).unwrap().successors.push(b1);
        let instrs = prog.collect_instructions();
        assert_eq!(instrs.len(), 2);
    }

    #[test]
    fn test_sexpr_complexity() {
        assert_eq!(SExpr::int(42).complexity(), 0);
        assert_eq!(SExpr::var("x").complexity(), 1);
        let binop = SExpr::binop(BinOp::Add, SExpr::var("x"), SExpr::int(1));
        assert_eq!(binop.complexity(), 3); // 2 + 1(var) + 0(const)
        let load = SExpr::load("mem", SExpr::var("i"));
        assert_eq!(load.complexity(), 4); // 3 + 1(var)
        let call = SExpr::Call("foo".to_string(), vec![SExpr::int(1)]);
        assert_eq!(call.complexity(), 4); // 4 + 0(const)
    }

    #[test]
    fn test_sinstr_has_side_effects() {
        let assign = SInstr::Assign {
            dst: "x".to_string(),
            src: SExpr::int(1),
            ty: SType::I64,
        };
        assert!(!assign.has_side_effects());

        let store = SInstr::Store {
            descriptor: MemoryAccessDescriptor::new_write("A", SExpr::int(0)),
            value: SExpr::int(42),
        };
        assert!(store.has_side_effects());

        let call = SInstr::Call {
            result: None,
            function: "print".to_string(),
            args: vec![],
        };
        assert!(call.has_side_effects());

        assert!(!SInstr::Nop.has_side_effects());
        assert!(!SInstr::Comment("test".to_string()).has_side_effects());

        let assert_instr = SInstr::Assert(SExpr::BoolConst(true), "ok".to_string());
        assert!(assert_instr.has_side_effects());
    }

    #[test]
    fn test_sblock_liveness() {
        let mut block = SBlock::new(0, "test");
        // x = 1; y = x + 2
        block.push(SInstr::Assign {
            dst: "x".to_string(),
            src: SExpr::int(1),
            ty: SType::I64,
        });
        block.push(SInstr::Assign {
            dst: "y".to_string(),
            src: SExpr::binop(BinOp::Add, SExpr::var("x"), SExpr::int(2)),
            ty: SType::I64,
        });
        let live = block.liveness_analysis();
        // x is defined before use, so not live at entry
        assert!(!live.contains("x"));
        // y is only defined, never used from outside, but x is used internally
        assert!(!live.contains("y"));
    }

    #[test]
    fn test_sblock_liveness_external_ref() {
        let mut block = SBlock::new(0, "test");
        // y = a + b; (a and b are not defined here, so they're live at entry)
        block.push(SInstr::Assign {
            dst: "y".to_string(),
            src: SExpr::binop(BinOp::Add, SExpr::var("a"), SExpr::var("b")),
            ty: SType::I64,
        });
        let live = block.liveness_analysis();
        assert!(live.contains("a"));
        assert!(live.contains("b"));
        assert!(!live.contains("y"));
    }

    #[test]
    fn test_sprogram_optimize() {
        let mut prog = SProgram::new("test");
        prog.block_mut(0).unwrap().push(SInstr::Assign {
            dst: "x".to_string(),
            src: SExpr::binop(BinOp::Add, SExpr::int(3), SExpr::int(4)),
            ty: SType::I64,
        });
        prog.block_mut(0).unwrap().push(SInstr::Comment("remove me".to_string()));
        prog.block_mut(0).unwrap().push(SInstr::Nop);
        let optimized = prog.optimize();
        let block = optimized.block(0).unwrap();
        // Comments and Nops should be removed
        assert_eq!(block.instructions.len(), 1);
        // 3+4 should be folded to 7
        match &block.instructions[0] {
            SInstr::Assign { src: SExpr::IntConst(7), .. } => {}
            other => panic!("Expected folded constant 7, got {:?}", other),
        }
    }

    #[test]
    fn test_sprogram_dead_block_elimination() {
        let mut prog = SProgram::new("test");
        prog.block_mut(0).unwrap().push(SInstr::Assign {
            dst: "x".to_string(),
            src: SExpr::int(1),
            ty: SType::I64,
        });
        let b1 = prog.new_block("reachable");
        let _b2 = prog.new_block("unreachable");
        prog.block_mut(0).unwrap().successors.push(b1);
        // b2 is not reachable from entry
        assert_eq!(prog.block_count(), 3);
        let cleaned = prog.dead_block_elimination();
        assert_eq!(cleaned.block_count(), 2);
        assert!(cleaned.block(b1).is_some());
        assert!(cleaned.block(_b2).is_none());
    }

    #[test]
    fn test_sprogram_to_dot() {
        let mut prog = SProgram::new("test");
        let b1 = prog.new_block("loop");
        prog.block_mut(0).unwrap().successors.push(b1);
        prog.block_mut(b1).unwrap().successors.push(0);
        let dot = prog.to_dot();
        assert!(dot.contains("digraph test"));
        assert!(dot.contains("B0 -> B1"));
        assert!(dot.contains("B1 -> B0"));
        assert!(dot.contains("node [shape=record]"));
    }

    #[test]
    fn test_sprogram_optimize_identity_mul() {
        let mut prog = SProgram::new("test");
        prog.block_mut(0).unwrap().push(SInstr::Assign {
            dst: "y".to_string(),
            src: SExpr::binop(BinOp::Mul, SExpr::var("x"), SExpr::int(1)),
            ty: SType::I64,
        });
        let optimized = prog.optimize();
        let block = optimized.block(0).unwrap();
        match &block.instructions[0] {
            SInstr::Assign { src: SExpr::Var(name), .. } => assert_eq!(name, "x"),
            other => panic!("Expected x, got {:?}", other),
        }
    }
}
