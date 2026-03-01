//! Three-phase partial evaluator for the PRAM IR.
//!
//! Phase 1: Constant propagation — propagate known constants through expressions
//! Phase 2: Dead code elimination — remove unreachable branches, unused variables
//! Phase 3: Strength reduction — replace expensive ops with cheaper equivalents

use std::collections::{HashMap, HashSet};

use crate::pram_ir::ast::{BinOp, Expr, Stmt, UnaryOp};

/// A known value in the partial evaluation environment.
#[derive(Debug, Clone, PartialEq)]
pub enum KnownValue {
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl KnownValue {
    pub fn to_expr(&self) -> Expr {
        match self {
            KnownValue::Int(v) => Expr::IntLiteral(*v),
            KnownValue::Float(v) => Expr::FloatLiteral(*v),
            KnownValue::Bool(b) => Expr::BoolLiteral(*b),
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            KnownValue::Int(v) => Some(*v),
            KnownValue::Bool(b) => Some(if *b { 1 } else { 0 }),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            KnownValue::Bool(b) => Some(*b),
            KnownValue::Int(v) => Some(*v != 0),
            _ => None,
        }
    }
}

/// Environment mapping variables to known constant values.
#[derive(Debug, Clone, Default)]
pub struct PartialEnv {
    bindings: HashMap<String, KnownValue>,
}

impl PartialEnv {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    pub fn bind(&mut self, name: &str, value: KnownValue) {
        self.bindings.insert(name.to_string(), value);
    }

    pub fn lookup(&self, name: &str) -> Option<&KnownValue> {
        self.bindings.get(name)
    }

    pub fn remove(&mut self, name: &str) {
        self.bindings.remove(name);
    }

    /// Create a child scope inheriting all current bindings.
    pub fn child_scope(&self) -> Self {
        self.clone()
    }
}

/// The three-phase partial evaluator.
pub struct PartialEvaluator {
    /// Processor ID substitution value, if known.
    proc_id_value: Option<i64>,
    /// Number of processors, if known.
    num_procs_value: Option<i64>,
    /// Whether to perform strength reduction.
    enable_strength_reduction: bool,
    /// Whether to perform dead code elimination.
    enable_dce: bool,
    /// Preset variable bindings.
    initial_bindings: HashMap<String, i64>,
}

impl PartialEvaluator {
    pub fn new() -> Self {
        Self {
            proc_id_value: None,
            num_procs_value: None,
            enable_strength_reduction: true,
            enable_dce: true,
            initial_bindings: HashMap::new(),
        }
    }

    /// Set a known processor ID for substitution (used when inlining parallel_for).
    pub fn with_proc_id(mut self, pid: i64) -> Self {
        self.proc_id_value = Some(pid);
        self
    }

    /// Set the known number of processors.
    pub fn with_num_procs(mut self, p: i64) -> Self {
        self.num_procs_value = Some(p);
        self
    }

    pub fn with_strength_reduction(mut self, enabled: bool) -> Self {
        self.enable_strength_reduction = enabled;
        self
    }

    pub fn with_dce(mut self, enabled: bool) -> Self {
        self.enable_dce = enabled;
        self
    }

    /// Create a partial evaluator with preset variable bindings.
    pub fn with_bindings(mut self, bindings: HashMap<String, i64>) -> Self {
        self.initial_bindings = bindings;
        self
    }

    /// Run all three phases on a list of statements.
    pub fn evaluate(&self, stmts: &[Stmt]) -> Vec<Stmt> {
        let mut env = PartialEnv::new();
        // Apply initial bindings
        for (name, val) in &self.initial_bindings {
            env.bind(name, KnownValue::Int(*val));
        }
        // Phase 1: Constant propagation
        let stmts = self.propagate_constants(&mut env, stmts);
        // Phase 2: Dead code elimination
        let stmts = if self.enable_dce {
            self.eliminate_dead_code(&stmts)
        } else {
            stmts
        };
        // Phase 3: Strength reduction
        if self.enable_strength_reduction {
            self.strength_reduce(&stmts)
        } else {
            stmts
        }
    }

    // ── Phase 1: Constant Propagation ──────────────────────────────────

    /// Propagate constants through a list of statements.
    fn propagate_constants(&self, env: &mut PartialEnv, stmts: &[Stmt]) -> Vec<Stmt> {
        let mut result = Vec::new();
        for stmt in stmts {
            let mut simplified = self.propagate_stmt(env, stmt);
            result.append(&mut simplified);
        }
        result
    }

    /// Partially evaluate a single statement.
    fn propagate_stmt(&self, env: &mut PartialEnv, stmt: &Stmt) -> Vec<Stmt> {
        match stmt {
            Stmt::LocalDecl(name, ty, init) => {
                let init = init.as_ref().map(|e| self.eval_expr(env, e));
                if let Some(ref e) = init {
                    if let Some(val) = self.extract_known_value(e) {
                        env.bind(name, val);
                    } else {
                        env.remove(name);
                    }
                }
                vec![Stmt::LocalDecl(name.clone(), ty.clone(), init)]
            }
            Stmt::Assign(name, expr) => {
                let simplified = self.eval_expr(env, expr);
                if let Some(val) = self.extract_known_value(&simplified) {
                    env.bind(name, val);
                } else {
                    env.remove(name);
                }
                vec![Stmt::Assign(name.clone(), simplified)]
            }
            Stmt::SharedWrite {
                memory,
                index,
                value,
            } => {
                vec![Stmt::SharedWrite {
                    memory: self.eval_expr(env, memory),
                    index: self.eval_expr(env, index),
                    value: self.eval_expr(env, value),
                }]
            }
            Stmt::If {
                condition,
                then_body,
                else_body,
            } => {
                let cond = self.eval_expr(env, condition);
                // If condition is known, eliminate dead branch
                match self.extract_known_value(&cond) {
                    Some(KnownValue::Bool(true)) => {
                        let mut child_env = env.child_scope();
                        self.propagate_constants(&mut child_env, then_body)
                    }
                    Some(KnownValue::Int(v)) if v != 0 => {
                        let mut child_env = env.child_scope();
                        self.propagate_constants(&mut child_env, then_body)
                    }
                    Some(KnownValue::Bool(false)) | Some(KnownValue::Int(0)) => {
                        let mut child_env = env.child_scope();
                        self.propagate_constants(&mut child_env, else_body)
                    }
                    _ => {
                        let mut then_env = env.child_scope();
                        let mut else_env = env.child_scope();
                        vec![Stmt::If {
                            condition: cond,
                            then_body: self.propagate_constants(&mut then_env, then_body),
                            else_body: self.propagate_constants(&mut else_env, else_body),
                        }]
                    }
                }
            }
            Stmt::SeqFor {
                var,
                start,
                end,
                step,
                body,
            } => {
                let start_e = self.eval_expr(env, start);
                let end_e = self.eval_expr(env, end);
                let step_e = step.as_ref().map(|s| self.eval_expr(env, s));
                // Loop variable is not constant inside the loop
                let mut body_env = env.child_scope();
                body_env.remove(var);
                vec![Stmt::SeqFor {
                    var: var.clone(),
                    start: start_e,
                    end: end_e,
                    step: step_e,
                    body: self.propagate_constants(&mut body_env, body),
                }]
            }
            Stmt::While { condition, body } => {
                let cond = self.eval_expr(env, condition);
                // Check for while(false)
                if let Some(KnownValue::Bool(false)) = self.extract_known_value(&cond) {
                    return vec![];
                }
                let mut body_env = env.child_scope();
                vec![Stmt::While {
                    condition: cond,
                    body: self.propagate_constants(&mut body_env, body),
                }]
            }
            Stmt::ParallelFor {
                proc_var,
                num_procs,
                body,
            } => {
                let num = self.eval_expr(env, num_procs);
                let mut body_env = env.child_scope();
                // proc_var is unknown in the body (unless we're specializing for a specific pid)
                if let Some(pid) = self.proc_id_value {
                    body_env.bind(proc_var, KnownValue::Int(pid));
                } else {
                    body_env.remove(proc_var);
                }
                vec![Stmt::ParallelFor {
                    proc_var: proc_var.clone(),
                    num_procs: num,
                    body: self.propagate_constants(&mut body_env, body),
                }]
            }
            Stmt::Block(stmts) => {
                let mut child_env = env.child_scope();
                let result = self.propagate_constants(&mut child_env, stmts);
                vec![Stmt::Block(result)]
            }
            Stmt::ExprStmt(expr) => {
                vec![Stmt::ExprStmt(self.eval_expr(env, expr))]
            }
            Stmt::Return(Some(expr)) => {
                vec![Stmt::Return(Some(self.eval_expr(env, expr)))]
            }
            Stmt::Assert(expr, msg) => {
                let e = self.eval_expr(env, expr);
                // If assertion is provably true, remove it
                if let Some(KnownValue::Bool(true)) = self.extract_known_value(&e) {
                    vec![]
                } else {
                    vec![Stmt::Assert(e, msg.clone())]
                }
            }
            Stmt::AllocShared {
                name,
                elem_type,
                size,
            } => {
                vec![Stmt::AllocShared {
                    name: name.clone(),
                    elem_type: elem_type.clone(),
                    size: self.eval_expr(env, size),
                }]
            }
            Stmt::PrefixSum {
                input,
                output,
                size,
                op,
            } => {
                vec![Stmt::PrefixSum {
                    input: input.clone(),
                    output: output.clone(),
                    size: self.eval_expr(env, size),
                    op: *op,
                }]
            }
            // Pass-through statements
            other => vec![other.clone()],
        }
    }

    /// Partially evaluate an expression, substituting known values.
    pub fn eval_expr(&self, env: &PartialEnv, expr: &Expr) -> Expr {
        match expr {
            Expr::Variable(name) => {
                if let Some(val) = env.lookup(name) {
                    val.to_expr()
                } else {
                    expr.clone()
                }
            }
            Expr::ProcessorId => {
                if let Some(pid) = self.proc_id_value {
                    Expr::IntLiteral(pid)
                } else {
                    Expr::ProcessorId
                }
            }
            Expr::NumProcessors => {
                if let Some(p) = self.num_procs_value {
                    Expr::IntLiteral(p)
                } else {
                    Expr::NumProcessors
                }
            }
            Expr::BinOp(op, left, right) => {
                let l = self.eval_expr(env, left);
                let r = self.eval_expr(env, right);
                // Try to fold constants
                let folded = Expr::BinOp(*op, Box::new(l.clone()), Box::new(r.clone()));
                if let Some(v) = folded.eval_const_int() {
                    if op.is_comparison() || op.is_logical() {
                        Expr::BoolLiteral(v != 0)
                    } else {
                        Expr::IntLiteral(v)
                    }
                } else if self.enable_strength_reduction {
                    self.reduce_binop(*op, l, r)
                } else {
                    Expr::BinOp(*op, Box::new(l), Box::new(r))
                }
            }
            Expr::UnaryOp(op, operand) => {
                let inner = self.eval_expr(env, operand);
                match (op, &inner) {
                    (UnaryOp::Neg, Expr::IntLiteral(v)) => Expr::IntLiteral(-v),
                    (UnaryOp::Neg, Expr::FloatLiteral(v)) => Expr::FloatLiteral(-v),
                    (UnaryOp::Not, Expr::BoolLiteral(b)) => Expr::BoolLiteral(!b),
                    (UnaryOp::BitNot, Expr::IntLiteral(v)) => Expr::IntLiteral(!v),
                    // Double negation elimination
                    (UnaryOp::Neg, Expr::UnaryOp(UnaryOp::Neg, inner)) => *inner.clone(),
                    (UnaryOp::Not, Expr::UnaryOp(UnaryOp::Not, inner)) => *inner.clone(),
                    _ => Expr::UnaryOp(*op, Box::new(inner)),
                }
            }
            Expr::SharedRead(mem, idx) => Expr::SharedRead(
                Box::new(self.eval_expr(env, mem)),
                Box::new(self.eval_expr(env, idx)),
            ),
            Expr::ArrayIndex(arr, idx) => Expr::ArrayIndex(
                Box::new(self.eval_expr(env, arr)),
                Box::new(self.eval_expr(env, idx)),
            ),
            Expr::FunctionCall(name, args) => {
                let args: Vec<Expr> = args.iter().map(|a| self.eval_expr(env, a)).collect();
                // Try to evaluate known functions on constant args
                match name.as_str() {
                    "min" if args.len() == 2 => {
                        if let (Some(a), Some(b)) = (args[0].eval_const_int(), args[1].eval_const_int()) {
                            return Expr::IntLiteral(a.min(b));
                        }
                    }
                    "max" if args.len() == 2 => {
                        if let (Some(a), Some(b)) = (args[0].eval_const_int(), args[1].eval_const_int()) {
                            return Expr::IntLiteral(a.max(b));
                        }
                    }
                    "abs" if args.len() == 1 => {
                        if let Some(a) = args[0].eval_const_int() {
                            return Expr::IntLiteral(a.abs());
                        }
                    }
                    _ => {}
                }
                Expr::FunctionCall(name.clone(), args)
            }
            Expr::Cast(inner, ty) => {
                let e = self.eval_expr(env, inner);
                Expr::Cast(Box::new(e), ty.clone())
            }
            Expr::Conditional(cond, then_e, else_e) => {
                let c = self.eval_expr(env, cond);
                let t = self.eval_expr(env, then_e);
                let e = self.eval_expr(env, else_e);
                match &c {
                    Expr::BoolLiteral(true) => t,
                    Expr::BoolLiteral(false) => e,
                    Expr::IntLiteral(v) if *v != 0 => t,
                    Expr::IntLiteral(0) => e,
                    _ => Expr::Conditional(Box::new(c), Box::new(t), Box::new(e)),
                }
            }
            other => other.clone(),
        }
    }

    /// Extract a KnownValue from a constant expression.
    fn extract_known_value(&self, expr: &Expr) -> Option<KnownValue> {
        match expr {
            Expr::IntLiteral(v) => Some(KnownValue::Int(*v)),
            Expr::FloatLiteral(v) => Some(KnownValue::Float(*v)),
            Expr::BoolLiteral(b) => Some(KnownValue::Bool(*b)),
            _ => None,
        }
    }

    // ── Phase 2: Dead Code Elimination ─────────────────────────────────

    /// Remove dead code: unreachable branches, unused assignments, nops.
    fn eliminate_dead_code(&self, stmts: &[Stmt]) -> Vec<Stmt> {
        // First pass: collect used variables
        let used = self.collect_used_vars(stmts);
        // Second pass: eliminate dead assignments
        self.dce_stmts(stmts, &used)
    }

    /// Collect all variables that are actually used (read).
    fn collect_used_vars(&self, stmts: &[Stmt]) -> HashSet<String> {
        let mut used = HashSet::new();
        for stmt in stmts {
            self.collect_used_in_stmt(stmt, &mut used);
        }
        used
    }

    fn collect_used_in_stmt(&self, stmt: &Stmt, used: &mut HashSet<String>) {
        match stmt {
            Stmt::Assign(_, expr) => {
                for v in expr.collect_variables() {
                    used.insert(v);
                }
            }
            Stmt::SharedWrite {
                memory,
                index,
                value,
            } => {
                for e in [memory, index, value] {
                    for v in e.collect_variables() {
                        used.insert(v);
                    }
                }
            }
            Stmt::If {
                condition,
                then_body,
                else_body,
            } => {
                for v in condition.collect_variables() {
                    used.insert(v);
                }
                for s in then_body {
                    self.collect_used_in_stmt(s, used);
                }
                for s in else_body {
                    self.collect_used_in_stmt(s, used);
                }
            }
            Stmt::ParallelFor { num_procs, body, .. } => {
                for v in num_procs.collect_variables() {
                    used.insert(v);
                }
                for s in body {
                    self.collect_used_in_stmt(s, used);
                }
            }
            Stmt::SeqFor {
                start,
                end,
                step,
                body,
                ..
            } => {
                for v in start.collect_variables() {
                    used.insert(v);
                }
                for v in end.collect_variables() {
                    used.insert(v);
                }
                if let Some(s) = step {
                    for v in s.collect_variables() {
                        used.insert(v);
                    }
                }
                for s in body {
                    self.collect_used_in_stmt(s, used);
                }
            }
            Stmt::While { condition, body } => {
                for v in condition.collect_variables() {
                    used.insert(v);
                }
                for s in body {
                    self.collect_used_in_stmt(s, used);
                }
            }
            Stmt::Return(Some(e)) => {
                for v in e.collect_variables() {
                    used.insert(v);
                }
            }
            Stmt::ExprStmt(e) => {
                for v in e.collect_variables() {
                    used.insert(v);
                }
            }
            Stmt::Assert(e, _) => {
                for v in e.collect_variables() {
                    used.insert(v);
                }
            }
            Stmt::Block(stmts) => {
                for s in stmts {
                    self.collect_used_in_stmt(s, used);
                }
            }
            Stmt::AllocShared { size, .. } => {
                for v in size.collect_variables() {
                    used.insert(v);
                }
            }
            Stmt::PrefixSum {
                input,
                output,
                size,
                ..
            } => {
                used.insert(input.clone());
                used.insert(output.clone());
                for v in size.collect_variables() {
                    used.insert(v);
                }
            }
            Stmt::AtomicCAS {
                memory,
                index,
                expected,
                desired,
                ..
            } => {
                for e in [memory, index, expected, desired] {
                    for v in e.collect_variables() {
                        used.insert(v);
                    }
                }
            }
            Stmt::FetchAdd {
                memory,
                index,
                value,
                ..
            } => {
                for e in [memory, index, value] {
                    for v in e.collect_variables() {
                        used.insert(v);
                    }
                }
            }
            _ => {}
        }
    }

    fn dce_stmts(&self, stmts: &[Stmt], used: &HashSet<String>) -> Vec<Stmt> {
        let mut result = Vec::new();
        for stmt in stmts {
            match stmt {
                Stmt::Nop => continue,
                // Remove assignments to unused variables (if they have no side effects)
                Stmt::Assign(name, expr) if !used.contains(name) && !expr_has_side_effects(expr) => {
                    continue;
                }
                Stmt::LocalDecl(name, _, Some(expr))
                    if !used.contains(name) && !expr_has_side_effects(expr) =>
                {
                    continue;
                }
                Stmt::If {
                    condition,
                    then_body,
                    else_body,
                } => {
                    let then_r = self.dce_stmts(then_body, used);
                    let else_r = self.dce_stmts(else_body, used);
                    // If both branches are empty, drop the if entirely
                    if then_r.is_empty() && else_r.is_empty() && !expr_has_side_effects(condition) {
                        continue;
                    }
                    result.push(Stmt::If {
                        condition: condition.clone(),
                        then_body: then_r,
                        else_body: else_r,
                    });
                }
                Stmt::SeqFor {
                    var,
                    start,
                    end,
                    step,
                    body,
                } => {
                    let body_r = self.dce_stmts(body, used);
                    if body_r.is_empty() {
                        continue;
                    }
                    result.push(Stmt::SeqFor {
                        var: var.clone(),
                        start: start.clone(),
                        end: end.clone(),
                        step: step.clone(),
                        body: body_r,
                    });
                }
                Stmt::While { condition, body } => {
                    let body_r = self.dce_stmts(body, used);
                    result.push(Stmt::While {
                        condition: condition.clone(),
                        body: body_r,
                    });
                }
                Stmt::Block(stmts) => {
                    let inner = self.dce_stmts(stmts, used);
                    if !inner.is_empty() {
                        result.push(Stmt::Block(inner));
                    }
                }
                Stmt::ParallelFor {
                    proc_var,
                    num_procs,
                    body,
                } => {
                    let body_r = self.dce_stmts(body, used);
                    if !body_r.is_empty() {
                        result.push(Stmt::ParallelFor {
                            proc_var: proc_var.clone(),
                            num_procs: num_procs.clone(),
                            body: body_r,
                        });
                    }
                }
                other => result.push(other.clone()),
            }
        }
        result
    }

    // ── Phase 3: Strength Reduction ────────────────────────────────────

    /// Apply strength reduction to all expressions in statements.
    fn strength_reduce(&self, stmts: &[Stmt]) -> Vec<Stmt> {
        stmts.iter().map(|s| self.sr_stmt(s)).collect()
    }

    fn sr_stmt(&self, stmt: &Stmt) -> Stmt {
        match stmt {
            Stmt::Assign(name, expr) => Stmt::Assign(name.clone(), self.sr_expr(expr)),
            Stmt::SharedWrite {
                memory,
                index,
                value,
            } => Stmt::SharedWrite {
                memory: self.sr_expr(memory),
                index: self.sr_expr(index),
                value: self.sr_expr(value),
            },
            Stmt::If {
                condition,
                then_body,
                else_body,
            } => Stmt::If {
                condition: self.sr_expr(condition),
                then_body: self.strength_reduce(then_body),
                else_body: self.strength_reduce(else_body),
            },
            Stmt::SeqFor {
                var,
                start,
                end,
                step,
                body,
            } => Stmt::SeqFor {
                var: var.clone(),
                start: self.sr_expr(start),
                end: self.sr_expr(end),
                step: step.as_ref().map(|s| self.sr_expr(s)),
                body: self.strength_reduce(body),
            },
            Stmt::While { condition, body } => Stmt::While {
                condition: self.sr_expr(condition),
                body: self.strength_reduce(body),
            },
            Stmt::ParallelFor {
                proc_var,
                num_procs,
                body,
            } => Stmt::ParallelFor {
                proc_var: proc_var.clone(),
                num_procs: self.sr_expr(num_procs),
                body: self.strength_reduce(body),
            },
            Stmt::Block(stmts) => Stmt::Block(self.strength_reduce(stmts)),
            Stmt::LocalDecl(name, ty, init) => {
                Stmt::LocalDecl(name.clone(), ty.clone(), init.as_ref().map(|e| self.sr_expr(e)))
            }
            Stmt::Return(Some(e)) => Stmt::Return(Some(self.sr_expr(e))),
            Stmt::ExprStmt(e) => Stmt::ExprStmt(self.sr_expr(e)),
            Stmt::Assert(e, msg) => Stmt::Assert(self.sr_expr(e), msg.clone()),
            other => other.clone(),
        }
    }

    /// Strength-reduce a binary operation.
    fn reduce_binop(&self, op: BinOp, left: Expr, right: Expr) -> Expr {
        match (op, &left, &right) {
            // x + 0 = x, 0 + x = x
            (BinOp::Add, _, Expr::IntLiteral(0)) => left,
            (BinOp::Add, Expr::IntLiteral(0), _) => right,
            // x - 0 = x
            (BinOp::Sub, _, Expr::IntLiteral(0)) => left,
            // x - x = 0
            (BinOp::Sub, l, r) if l == r => Expr::IntLiteral(0),
            // x * 0 = 0, 0 * x = 0
            (BinOp::Mul, _, Expr::IntLiteral(0)) | (BinOp::Mul, Expr::IntLiteral(0), _) => {
                Expr::IntLiteral(0)
            }
            // x * 1 = x, 1 * x = x
            (BinOp::Mul, _, Expr::IntLiteral(1)) => left,
            (BinOp::Mul, Expr::IntLiteral(1), _) => right,
            // x * 2 = x + x (cheaper on some architectures)
            (BinOp::Mul, _, Expr::IntLiteral(2)) => {
                Expr::BinOp(BinOp::Add, Box::new(left.clone()), Box::new(left))
            }
            // x * power_of_2 = x << log2(power_of_2)
            (BinOp::Mul, _, Expr::IntLiteral(v)) if *v > 0 && (*v & (*v - 1)) == 0 => {
                let shift = (*v as u64).trailing_zeros() as i64;
                Expr::BinOp(BinOp::Shl, Box::new(left), Box::new(Expr::IntLiteral(shift)))
            }
            // x / 1 = x
            (BinOp::Div, _, Expr::IntLiteral(1)) => left,
            // x / power_of_2 = x >> log2(power_of_2) (for positive values)
            (BinOp::Div, _, Expr::IntLiteral(v)) if *v > 0 && (*v & (*v - 1)) == 0 => {
                let shift = (*v as u64).trailing_zeros() as i64;
                Expr::BinOp(BinOp::Shr, Box::new(left), Box::new(Expr::IntLiteral(shift)))
            }
            // x % power_of_2 = x & (power_of_2 - 1)
            (BinOp::Mod, _, Expr::IntLiteral(v)) if *v > 0 && (*v & (*v - 1)) == 0 => {
                Expr::BinOp(
                    BinOp::BitAnd,
                    Box::new(left),
                    Box::new(Expr::IntLiteral(*v - 1)),
                )
            }
            // x & 0 = 0
            (BinOp::BitAnd, _, Expr::IntLiteral(0)) | (BinOp::BitAnd, Expr::IntLiteral(0), _) => {
                Expr::IntLiteral(0)
            }
            // x | 0 = x, 0 | x = x
            (BinOp::BitOr, _, Expr::IntLiteral(0)) => left,
            (BinOp::BitOr, Expr::IntLiteral(0), _) => right,
            // x ^ 0 = x
            (BinOp::BitXor, _, Expr::IntLiteral(0)) => left,
            (BinOp::BitXor, Expr::IntLiteral(0), _) => right,
            // x ^ x = 0
            (BinOp::BitXor, l, r) if l == r => Expr::IntLiteral(0),
            // x && true = x, true && x = x
            (BinOp::And, _, Expr::BoolLiteral(true)) => left,
            (BinOp::And, Expr::BoolLiteral(true), _) => right,
            // x && false = false
            (BinOp::And, _, Expr::BoolLiteral(false)) | (BinOp::And, Expr::BoolLiteral(false), _) => {
                Expr::BoolLiteral(false)
            }
            // x || false = x, false || x = x
            (BinOp::Or, _, Expr::BoolLiteral(false)) => left,
            (BinOp::Or, Expr::BoolLiteral(false), _) => right,
            // x || true = true
            (BinOp::Or, _, Expr::BoolLiteral(true)) | (BinOp::Or, Expr::BoolLiteral(true), _) => {
                Expr::BoolLiteral(true)
            }
            // x == x = true, x != x = false
            (BinOp::Eq, l, r) if l == r => Expr::BoolLiteral(true),
            (BinOp::Ne, l, r) if l == r => Expr::BoolLiteral(false),
            // x <= x = true, x >= x = true
            (BinOp::Le, l, r) if l == r => Expr::BoolLiteral(true),
            (BinOp::Ge, l, r) if l == r => Expr::BoolLiteral(true),
            // x < x = false, x > x = false
            (BinOp::Lt, l, r) if l == r => Expr::BoolLiteral(false),
            (BinOp::Gt, l, r) if l == r => Expr::BoolLiteral(false),
            // Default: no reduction
            _ => Expr::BinOp(op, Box::new(left), Box::new(right)),
        }
    }

    fn sr_expr(&self, expr: &Expr) -> Expr {
        match expr {
            Expr::BinOp(op, left, right) => {
                let l = self.sr_expr(left);
                let r = self.sr_expr(right);
                self.reduce_binop(*op, l, r)
            }
            Expr::UnaryOp(op, inner) => {
                let e = self.sr_expr(inner);
                Expr::UnaryOp(*op, Box::new(e))
            }
            Expr::SharedRead(mem, idx) => {
                Expr::SharedRead(Box::new(self.sr_expr(mem)), Box::new(self.sr_expr(idx)))
            }
            Expr::ArrayIndex(arr, idx) => {
                Expr::ArrayIndex(Box::new(self.sr_expr(arr)), Box::new(self.sr_expr(idx)))
            }
            Expr::FunctionCall(name, args) => {
                Expr::FunctionCall(name.clone(), args.iter().map(|a| self.sr_expr(a)).collect())
            }
            Expr::Cast(e, ty) => Expr::Cast(Box::new(self.sr_expr(e)), ty.clone()),
            Expr::Conditional(c, t, e) => Expr::Conditional(
                Box::new(self.sr_expr(c)),
                Box::new(self.sr_expr(t)),
                Box::new(self.sr_expr(e)),
            ),
            other => other.clone(),
        }
    }
}

/// Check if an expression has side effects (function calls, shared memory reads).
fn expr_has_side_effects(expr: &Expr) -> bool {
    match expr {
        Expr::FunctionCall(_, _) => true,
        Expr::SharedRead(_, _) => true,
        Expr::BinOp(_, a, b) => expr_has_side_effects(a) || expr_has_side_effects(b),
        Expr::UnaryOp(_, e) => expr_has_side_effects(e),
        Expr::ArrayIndex(a, i) => expr_has_side_effects(a) || expr_has_side_effects(i),
        Expr::Cast(e, _) => expr_has_side_effects(e),
        Expr::Conditional(c, t, e) => {
            expr_has_side_effects(c) || expr_has_side_effects(t) || expr_has_side_effects(e)
        }
        _ => false,
    }
}

/// Attempt to unroll a sequential for-loop with small constant bounds.
///
/// If the loop has constant start and end, and (end - start) <= max_iters,
/// produces a flattened list of body statements with the loop variable substituted.
pub fn loop_unrolling(stmt: &Stmt, max_iters: usize) -> Vec<Stmt> {
    match stmt {
        Stmt::SeqFor {
            var,
            start,
            end,
            step,
            body,
        } => {
            let s = match start.eval_const_int() {
                Some(v) => v,
                None => return vec![stmt.clone()],
            };
            let e = match end.eval_const_int() {
                Some(v) => v,
                None => return vec![stmt.clone()],
            };
            let step_val = match step {
                Some(se) => match se.eval_const_int() {
                    Some(v) if v > 0 => v,
                    _ => return vec![stmt.clone()],
                },
                None => 1,
            };
            if e <= s {
                return vec![];
            }
            let iters = ((e - s + step_val - 1) / step_val) as usize;
            if iters > max_iters {
                return vec![stmt.clone()];
            }
            let mut result = Vec::new();
            let mut i = s;
            while i < e {
                let replacement = Expr::IntLiteral(i);
                for b in body {
                    result.push(substitute_stmt(b, var, &replacement));
                }
                i += step_val;
            }
            result
        }
        _ => vec![stmt.clone()],
    }
}

/// Substitute a variable in a statement (used by loop unrolling).
fn substitute_stmt(stmt: &Stmt, var: &str, replacement: &Expr) -> Stmt {
    match stmt {
        Stmt::Assign(name, expr) => {
            Stmt::Assign(name.clone(), expr.substitute(var, replacement))
        }
        Stmt::SharedWrite { memory, index, value } => Stmt::SharedWrite {
            memory: memory.substitute(var, replacement),
            index: index.substitute(var, replacement),
            value: value.substitute(var, replacement),
        },
        Stmt::If { condition, then_body, else_body } => Stmt::If {
            condition: condition.substitute(var, replacement),
            then_body: then_body.iter().map(|s| substitute_stmt(s, var, replacement)).collect(),
            else_body: else_body.iter().map(|s| substitute_stmt(s, var, replacement)).collect(),
        },
        Stmt::ExprStmt(e) => Stmt::ExprStmt(e.substitute(var, replacement)),
        Stmt::Return(Some(e)) => Stmt::Return(Some(e.substitute(var, replacement))),
        Stmt::Assert(e, msg) => Stmt::Assert(e.substitute(var, replacement), msg.clone()),
        Stmt::LocalDecl(name, ty, init) => Stmt::LocalDecl(
            name.clone(),
            ty.clone(),
            init.as_ref().map(|e| e.substitute(var, replacement)),
        ),
        Stmt::Block(stmts) => Stmt::Block(
            stmts.iter().map(|s| substitute_stmt(s, var, replacement)).collect(),
        ),
        Stmt::SeqFor { var: v, start, end, step, body } => {
            if v == var {
                return stmt.clone();
            }
            Stmt::SeqFor {
                var: v.clone(),
                start: start.substitute(var, replacement),
                end: end.substitute(var, replacement),
                step: step.as_ref().map(|s| s.substitute(var, replacement)),
                body: body.iter().map(|s| substitute_stmt(s, var, replacement)).collect(),
            }
        }
        other => other.clone(),
    }
}

/// Inline function calls where definitions are known.
///
/// For each `ExprStmt(FunctionCall(name, args))` where `name` is in `defs`,
/// replaces the call with the definition body.
pub fn inline_function_calls(stmts: &[Stmt], defs: &HashMap<String, Vec<Stmt>>) -> Vec<Stmt> {
    let mut result = Vec::new();
    for stmt in stmts {
        match stmt {
            Stmt::ExprStmt(Expr::FunctionCall(name, _args)) => {
                if let Some(body) = defs.get(name) {
                    result.extend(body.clone());
                } else {
                    result.push(stmt.clone());
                }
            }
            Stmt::If { condition, then_body, else_body } => {
                result.push(Stmt::If {
                    condition: condition.clone(),
                    then_body: inline_function_calls(then_body, defs),
                    else_body: inline_function_calls(else_body, defs),
                });
            }
            Stmt::SeqFor { var, start, end, step, body } => {
                result.push(Stmt::SeqFor {
                    var: var.clone(),
                    start: start.clone(),
                    end: end.clone(),
                    step: step.clone(),
                    body: inline_function_calls(body, defs),
                });
            }
            Stmt::Block(inner) => {
                result.push(Stmt::Block(inline_function_calls(inner, defs)));
            }
            other => result.push(other.clone()),
        }
    }
    result
}

/// Apply additional algebraic simplifications to an expression.
///
/// Simplifications include:
/// - x * 2 → x + x
/// - x + x → x * 2 (inverse, kept as x + x for strength reduction)
/// - x * (-1) → -x
/// - --x → x
/// - x & x → x
/// - x | x → x
/// - x ^ 0 → x
pub fn algebraic_simplify(expr: &Expr) -> Expr {
    match expr {
        Expr::BinOp(op, a, b) => {
            let sa = algebraic_simplify(a);
            let sb = algebraic_simplify(b);
            match (op, &sa, &sb) {
                // x * 2 -> x + x
                (BinOp::Mul, _, Expr::IntLiteral(2)) => {
                    Expr::BinOp(BinOp::Add, Box::new(sa.clone()), Box::new(sa))
                }
                (BinOp::Mul, Expr::IntLiteral(2), _) => {
                    Expr::BinOp(BinOp::Add, Box::new(sb.clone()), Box::new(sb))
                }
                // x * (-1) -> -x
                (BinOp::Mul, _, Expr::IntLiteral(-1)) => {
                    Expr::UnaryOp(UnaryOp::Neg, Box::new(sa))
                }
                (BinOp::Mul, Expr::IntLiteral(-1), _) => {
                    Expr::UnaryOp(UnaryOp::Neg, Box::new(sb))
                }
                // x & x -> x
                (BinOp::BitAnd, l, r) if l == r => sa,
                // x | x -> x
                (BinOp::BitOr, l, r) if l == r => sa,
                // x + 0 -> x
                (BinOp::Add, _, Expr::IntLiteral(0)) => sa,
                (BinOp::Add, Expr::IntLiteral(0), _) => sb,
                // x - 0 -> x
                (BinOp::Sub, _, Expr::IntLiteral(0)) => sa,
                // 0 - x -> -x
                (BinOp::Sub, Expr::IntLiteral(0), _) => {
                    Expr::UnaryOp(UnaryOp::Neg, Box::new(sb))
                }
                _ => Expr::BinOp(*op, Box::new(sa), Box::new(sb)),
            }
        }
        Expr::UnaryOp(op, inner) => {
            let si = algebraic_simplify(inner);
            match (op, &si) {
                // --x -> x
                (UnaryOp::Neg, Expr::UnaryOp(UnaryOp::Neg, x)) => *x.clone(),
                // !!x -> x
                (UnaryOp::Not, Expr::UnaryOp(UnaryOp::Not, x)) => *x.clone(),
                _ => Expr::UnaryOp(*op, Box::new(si)),
            }
        }
        Expr::SharedRead(m, i) => Expr::SharedRead(
            Box::new(algebraic_simplify(m)),
            Box::new(algebraic_simplify(i)),
        ),
        Expr::ArrayIndex(a, i) => Expr::ArrayIndex(
            Box::new(algebraic_simplify(a)),
            Box::new(algebraic_simplify(i)),
        ),
        Expr::FunctionCall(name, args) => Expr::FunctionCall(
            name.clone(),
            args.iter().map(algebraic_simplify).collect(),
        ),
        Expr::Cast(e, ty) => Expr::Cast(Box::new(algebraic_simplify(e)), ty.clone()),
        Expr::Conditional(c, t, e) => Expr::Conditional(
            Box::new(algebraic_simplify(c)),
            Box::new(algebraic_simplify(t)),
            Box::new(algebraic_simplify(e)),
        ),
        other => other.clone(),
    }
}

/// Perform common subexpression elimination on a statement list.
///
/// Identifies identical non-trivial expressions computed multiple times
/// and replaces duplicates with references to a temporary variable.
pub fn common_subexpression_elimination(stmts: &[Stmt]) -> Vec<Stmt> {
    // Collect expressions assigned to variables
    let mut expr_to_var: HashMap<String, String> = HashMap::new();
    let mut result = Vec::new();

    for stmt in stmts {
        match stmt {
            Stmt::Assign(name, expr) => {
                let key = format!("{:?}", expr);
                if let Some(existing_var) = expr_to_var.get(&key) {
                    // Replace with reference to existing variable
                    result.push(Stmt::Assign(
                        name.clone(),
                        Expr::Variable(existing_var.clone()),
                    ));
                } else {
                    if !matches!(expr, Expr::IntLiteral(_) | Expr::BoolLiteral(_) | Expr::FloatLiteral(_) | Expr::Variable(_)) {
                        expr_to_var.insert(key, name.clone());
                    }
                    result.push(stmt.clone());
                }
            }
            _ => result.push(stmt.clone()),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pram_ir::types::PramType;

    fn pe() -> PartialEvaluator {
        PartialEvaluator::new()
    }

    #[test]
    fn test_constant_folding_add() {
        let env = PartialEnv::new();
        let expr = Expr::binop(BinOp::Add, Expr::int(3), Expr::int(4));
        let result = pe().eval_expr(&env, &expr);
        assert_eq!(result, Expr::IntLiteral(7));
    }

    #[test]
    fn test_constant_folding_mul() {
        let env = PartialEnv::new();
        let expr = Expr::binop(BinOp::Mul, Expr::int(6), Expr::int(7));
        let result = pe().eval_expr(&env, &expr);
        assert_eq!(result, Expr::IntLiteral(42));
    }

    #[test]
    fn test_constant_propagation_variable() {
        let mut env = PartialEnv::new();
        env.bind("x", KnownValue::Int(10));
        let expr = Expr::binop(BinOp::Add, Expr::var("x"), Expr::int(5));
        let result = pe().eval_expr(&env, &expr);
        assert_eq!(result, Expr::IntLiteral(15));
    }

    #[test]
    fn test_constant_propagation_through_assign() {
        let stmts = vec![
            Stmt::Assign("x".to_string(), Expr::int(10)),
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::binop(BinOp::Add, Expr::var("x"), Expr::int(5)),
            },
        ];
        let result = pe().evaluate(&stmts);
        // The SharedWrite value should have x folded to 10, yielding 15
        match &result.last().unwrap() {
            Stmt::SharedWrite { value: Expr::IntLiteral(15), .. } => {},
            other => panic!("Expected value = 15, got {:?}", other),
        }
    }

    #[test]
    fn test_dead_branch_elimination_true() {
        let stmts = vec![Stmt::If {
            condition: Expr::BoolLiteral(true),
            then_body: vec![Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::int(1),
            }],
            else_body: vec![Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::int(2),
            }],
        }];
        let result = pe().evaluate(&stmts);
        // Should inline the then branch
        assert_eq!(result.len(), 1);
        match &result[0] {
            Stmt::SharedWrite { value: Expr::IntLiteral(1), .. } => {},
            other => panic!("Expected SharedWrite with value 1, got {:?}", other),
        }
    }

    #[test]
    fn test_dead_branch_elimination_false() {
        let stmts = vec![Stmt::If {
            condition: Expr::BoolLiteral(false),
            then_body: vec![Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::int(1),
            }],
            else_body: vec![Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::int(2),
            }],
        }];
        let result = pe().evaluate(&stmts);
        assert_eq!(result.len(), 1);
        match &result[0] {
            Stmt::SharedWrite { value: Expr::IntLiteral(2), .. } => {},
            other => panic!("Expected SharedWrite with value 2, got {:?}", other),
        }
    }

    #[test]
    fn test_strength_reduction_mul_power_of_2() {
        let env = PartialEnv::new();
        let expr = Expr::binop(BinOp::Mul, Expr::var("x"), Expr::int(8));
        let result = pe().eval_expr(&env, &expr);
        match result {
            Expr::BinOp(BinOp::Shl, _, r) => assert_eq!(*r, Expr::IntLiteral(3)),
            other => panic!("Expected shift, got {:?}", other),
        }
    }

    #[test]
    fn test_strength_reduction_div_power_of_2() {
        let env = PartialEnv::new();
        let expr = Expr::binop(BinOp::Div, Expr::var("x"), Expr::int(4));
        let result = pe().eval_expr(&env, &expr);
        match result {
            Expr::BinOp(BinOp::Shr, _, r) => assert_eq!(*r, Expr::IntLiteral(2)),
            other => panic!("Expected shift, got {:?}", other),
        }
    }

    #[test]
    fn test_strength_reduction_mod_power_of_2() {
        let env = PartialEnv::new();
        let expr = Expr::binop(BinOp::Mod, Expr::var("x"), Expr::int(16));
        let result = pe().eval_expr(&env, &expr);
        match result {
            Expr::BinOp(BinOp::BitAnd, _, r) => assert_eq!(*r, Expr::IntLiteral(15)),
            other => panic!("Expected bitand, got {:?}", other),
        }
    }

    #[test]
    fn test_strength_reduction_add_zero() {
        let env = PartialEnv::new();
        let expr = Expr::binop(BinOp::Add, Expr::var("x"), Expr::int(0));
        let result = pe().eval_expr(&env, &expr);
        assert_eq!(result, Expr::var("x"));
    }

    #[test]
    fn test_strength_reduction_mul_zero() {
        let env = PartialEnv::new();
        let expr = Expr::binop(BinOp::Mul, Expr::var("x"), Expr::int(0));
        let result = pe().eval_expr(&env, &expr);
        assert_eq!(result, Expr::IntLiteral(0));
    }

    #[test]
    fn test_strength_reduction_mul_one() {
        let env = PartialEnv::new();
        let expr = Expr::binop(BinOp::Mul, Expr::var("x"), Expr::int(1));
        let result = pe().eval_expr(&env, &expr);
        assert_eq!(result, Expr::var("x"));
    }

    #[test]
    fn test_strength_reduction_sub_self() {
        let env = PartialEnv::new();
        let expr = Expr::binop(BinOp::Sub, Expr::var("x"), Expr::var("x"));
        let result = pe().eval_expr(&env, &expr);
        assert_eq!(result, Expr::IntLiteral(0));
    }

    #[test]
    fn test_processor_id_substitution() {
        let env = PartialEnv::new();
        let expr = Expr::binop(BinOp::Add, Expr::ProcessorId, Expr::int(1));
        let result = pe().with_proc_id(5).eval_expr(&env, &expr);
        assert_eq!(result, Expr::IntLiteral(6));
    }

    #[test]
    fn test_num_processors_substitution() {
        let env = PartialEnv::new();
        let expr = Expr::NumProcessors;
        let result = pe().with_num_procs(8).eval_expr(&env, &expr);
        assert_eq!(result, Expr::IntLiteral(8));
    }

    #[test]
    fn test_double_negation_elimination() {
        let env = PartialEnv::new();
        let expr = Expr::unop(UnaryOp::Neg, Expr::unop(UnaryOp::Neg, Expr::var("x")));
        let result = pe().eval_expr(&env, &expr);
        assert_eq!(result, Expr::var("x"));
    }

    #[test]
    fn test_dead_code_unused_variable() {
        // y is never used, but x is used in a SharedWrite
        let stmts = vec![
            Stmt::Assign("y".to_string(), Expr::int(42)),
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::var("x"),
            },
        ];
        let result = pe().evaluate(&stmts);
        // y assignment should be removed
        assert_eq!(result.len(), 1);
        assert!(matches!(&result[0], Stmt::SharedWrite { .. }));
    }

    #[test]
    fn test_while_false_elimination() {
        let stmts = vec![Stmt::While {
            condition: Expr::BoolLiteral(false),
            body: vec![Stmt::Assign("x".to_string(), Expr::int(1))],
        }];
        let result = pe().evaluate(&stmts);
        assert!(result.is_empty());
    }

    #[test]
    fn test_assertion_true_eliminated() {
        let stmts = vec![
            Stmt::Assert(Expr::BoolLiteral(true), "always true".to_string()),
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::int(1),
            },
        ];
        let result = pe().evaluate(&stmts);
        // The assertion should be eliminated, only the SharedWrite remains
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_conditional_expr_true() {
        let env = PartialEnv::new();
        let expr = Expr::Conditional(
            Box::new(Expr::BoolLiteral(true)),
            Box::new(Expr::int(10)),
            Box::new(Expr::int(20)),
        );
        let result = pe().eval_expr(&env, &expr);
        assert_eq!(result, Expr::IntLiteral(10));
    }

    #[test]
    fn test_known_value_conversions() {
        let kv = KnownValue::Int(42);
        assert_eq!(kv.as_int(), Some(42));
        assert_eq!(kv.as_bool(), Some(true));
        assert_eq!(kv.to_expr(), Expr::IntLiteral(42));

        let kb = KnownValue::Bool(false);
        assert_eq!(kb.as_int(), Some(0));
        assert_eq!(kb.as_bool(), Some(false));

        let kf = KnownValue::Float(3.14);
        assert_eq!(kf.as_int(), None);
        assert_eq!(kf.as_bool(), None);
    }

    #[test]
    fn test_partial_env_child_scope() {
        let mut env = PartialEnv::new();
        env.bind("x", KnownValue::Int(10));
        let mut child = env.child_scope();
        child.bind("y", KnownValue::Int(20));
        // Parent should not have y
        assert!(env.lookup("y").is_none());
        // Child should have both
        assert!(child.lookup("x").is_some());
        assert!(child.lookup("y").is_some());
    }

    #[test]
    fn test_function_call_constant_eval() {
        let env = PartialEnv::new();
        let expr = Expr::FunctionCall("min".to_string(), vec![Expr::int(3), Expr::int(7)]);
        let result = pe().eval_expr(&env, &expr);
        assert_eq!(result, Expr::IntLiteral(3));
    }

    #[test]
    fn test_boolean_strength_reduction() {
        let env = PartialEnv::new();
        let expr = Expr::binop(BinOp::And, Expr::var("x"), Expr::BoolLiteral(true));
        let result = pe().eval_expr(&env, &expr);
        assert_eq!(result, Expr::var("x"));

        let expr2 = Expr::binop(BinOp::Or, Expr::var("x"), Expr::BoolLiteral(true));
        let result2 = pe().eval_expr(&env, &expr2);
        assert_eq!(result2, Expr::BoolLiteral(true));
    }

    #[test]
    fn test_eq_self_elimination() {
        let env = PartialEnv::new();
        let expr = Expr::binop(BinOp::Eq, Expr::var("x"), Expr::var("x"));
        let result = pe().eval_expr(&env, &expr);
        assert_eq!(result, Expr::BoolLiteral(true));
    }

    #[test]
    fn test_xor_self_elimination() {
        let env = PartialEnv::new();
        let expr = Expr::binop(BinOp::BitXor, Expr::var("x"), Expr::var("x"));
        let result = pe().eval_expr(&env, &expr);
        assert_eq!(result, Expr::IntLiteral(0));
    }

    #[test]
    fn test_with_bindings() {
        let mut bindings = HashMap::new();
        bindings.insert("N".to_string(), 100);
        let stmts = vec![Stmt::SharedWrite {
            memory: Expr::var("A"),
            index: Expr::int(0),
            value: Expr::binop(BinOp::Add, Expr::var("N"), Expr::int(1)),
        }];
        let result = pe().with_bindings(bindings).evaluate(&stmts);
        match &result[0] {
            Stmt::SharedWrite { value: Expr::IntLiteral(101), .. } => {}
            other => panic!("Expected SharedWrite with value 101, got {:?}", other),
        }
    }

    #[test]
    fn test_loop_unrolling_small() {
        let stmt = Stmt::SeqFor {
            var: "i".to_string(),
            start: Expr::int(0),
            end: Expr::int(3),
            step: None,
            body: vec![Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::var("i"),
                value: Expr::int(1),
            }],
        };
        let result = loop_unrolling(&stmt, 10);
        // Should produce 3 writes with i substituted
        let writes: Vec<_> = result.iter().filter(|s| matches!(s, Stmt::SharedWrite { .. })).collect();
        assert_eq!(writes.len(), 3);
        match &writes[0] {
            Stmt::SharedWrite { index: Expr::IntLiteral(0), .. } => {}
            other => panic!("Expected index=0, got {:?}", other),
        }
        match &writes[2] {
            Stmt::SharedWrite { index: Expr::IntLiteral(2), .. } => {}
            other => panic!("Expected index=2, got {:?}", other),
        }
    }

    #[test]
    fn test_loop_unrolling_too_large() {
        let stmt = Stmt::SeqFor {
            var: "i".to_string(),
            start: Expr::int(0),
            end: Expr::int(100),
            step: None,
            body: vec![Stmt::Assign("x".to_string(), Expr::var("i"))],
        };
        let result = loop_unrolling(&stmt, 10);
        // Should not unroll, return original
        assert_eq!(result.len(), 1);
        assert!(matches!(&result[0], Stmt::SeqFor { .. }));
    }

    #[test]
    fn test_loop_unrolling_with_step() {
        let stmt = Stmt::SeqFor {
            var: "i".to_string(),
            start: Expr::int(0),
            end: Expr::int(6),
            step: Some(Expr::int(2)),
            body: vec![Stmt::Assign("x".to_string(), Expr::var("i"))],
        };
        let result = loop_unrolling(&stmt, 10);
        // 0, 2, 4 => 3 iterations
        let assigns: Vec<_> = result.iter().filter(|s| matches!(s, Stmt::Assign(..))).collect();
        assert_eq!(assigns.len(), 3);
    }

    #[test]
    fn test_inline_function_calls() {
        let mut defs = HashMap::new();
        defs.insert(
            "init".to_string(),
            vec![Stmt::Assign("x".to_string(), Expr::int(0))],
        );
        let stmts = vec![
            Stmt::ExprStmt(Expr::FunctionCall("init".to_string(), vec![])),
            Stmt::Assign("y".to_string(), Expr::int(1)),
        ];
        let result = inline_function_calls(&stmts, &defs);
        assert_eq!(result.len(), 2);
        match &result[0] {
            Stmt::Assign(name, Expr::IntLiteral(0)) => assert_eq!(name, "x"),
            other => panic!("Expected inlined assign, got {:?}", other),
        }
    }

    #[test]
    fn test_algebraic_simplify_mul2() {
        let expr = Expr::binop(BinOp::Mul, Expr::var("x"), Expr::int(2));
        let result = algebraic_simplify(&expr);
        match result {
            Expr::BinOp(BinOp::Add, l, r) => {
                assert_eq!(*l, Expr::var("x"));
                assert_eq!(*r, Expr::var("x"));
            }
            other => panic!("Expected x+x, got {:?}", other),
        }
    }

    #[test]
    fn test_algebraic_simplify_neg_neg() {
        let expr = Expr::unop(UnaryOp::Neg, Expr::unop(UnaryOp::Neg, Expr::var("x")));
        let result = algebraic_simplify(&expr);
        assert_eq!(result, Expr::var("x"));
    }

    #[test]
    fn test_algebraic_simplify_mul_neg1() {
        let expr = Expr::binop(BinOp::Mul, Expr::var("x"), Expr::int(-1));
        let result = algebraic_simplify(&expr);
        match result {
            Expr::UnaryOp(UnaryOp::Neg, inner) => assert_eq!(*inner, Expr::var("x")),
            other => panic!("Expected -x, got {:?}", other),
        }
    }

    #[test]
    fn test_common_subexpression_elimination() {
        let expr = Expr::binop(BinOp::Add, Expr::var("a"), Expr::var("b"));
        let stmts = vec![
            Stmt::Assign("x".to_string(), expr.clone()),
            Stmt::Assign("y".to_string(), expr),
        ];
        let result = common_subexpression_elimination(&stmts);
        assert_eq!(result.len(), 2);
        // Second should reference x instead of recomputing
        match &result[1] {
            Stmt::Assign(name, Expr::Variable(ref_name)) => {
                assert_eq!(name, "y");
                assert_eq!(ref_name, "x");
            }
            other => panic!("Expected variable reference, got {:?}", other),
        }
    }

    #[test]
    fn test_loop_unrolling_empty() {
        let stmt = Stmt::SeqFor {
            var: "i".to_string(),
            start: Expr::int(5),
            end: Expr::int(5),
            step: None,
            body: vec![Stmt::Assign("x".to_string(), Expr::int(1))],
        };
        let result = loop_unrolling(&stmt, 10);
        assert!(result.is_empty());
    }
}
