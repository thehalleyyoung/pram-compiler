//! Formal small-step operational semantics for the PRAM IR.
//!
//! Defines evaluation rules for expressions and statements as
//! a stepping relation `⟨e, σ⟩ → ⟨e', σ'⟩` where σ is the shared store.
//!
//! # Evaluation Rules
//!
//! ## Expressions
//! E-CONST:  ⟨n, σ⟩ → ⟨n, σ⟩  (values are stuck in the good sense)
//! E-VAR:    ⟨x, σ⟩ → ⟨σ(x), σ⟩
//! E-BINOP:  ⟨v₁ ⊕ v₂, σ⟩ → ⟨δ(⊕, v₁, v₂), σ⟩
//! E-LEFT:   ⟨e₁ ⊕ e₂, σ⟩ → ⟨e₁' ⊕ e₂, σ'⟩  if ⟨e₁, σ⟩ → ⟨e₁', σ'⟩
//! E-RIGHT:  ⟨v₁ ⊕ e₂, σ⟩ → ⟨v₁ ⊕ e₂', σ'⟩  if ⟨e₂, σ⟩ → ⟨e₂', σ'⟩
//! E-READ:   ⟨SharedRead(m, v), σ⟩ → ⟨σ.mem[m][v], σ⟩
//! E-COND-T: ⟨true ? e₁ : e₂, σ⟩ → ⟨e₁, σ⟩
//! E-COND-F: ⟨false ? e₁ : e₂, σ⟩ → ⟨e₂, σ⟩
//!
//! ## Statements
//! S-ASSIGN:  ⟨x := v, σ⟩ → ⟨nop, σ[x↦v]⟩
//! S-WRITE:   ⟨SharedWrite(m, i, v), σ⟩ → ⟨nop, σ.mem[m][i↦v]⟩
//! S-SEQ:     ⟨s₁; s₂, σ⟩ → ⟨s₁'; s₂, σ'⟩  if ⟨s₁, σ⟩ → ⟨s₁', σ'⟩
//! S-PAR:     ⟨ParallelFor(p,n,body), σ⟩ → ⟨body[0/p]; body[1/p]; ...; body[n-1/p], σ⟩
//!
//! ## CRCW Conflict Resolution
//! S-CRCW-PRIORITY: When multiple writes target the same address in the same
//!   parallel phase, the write from the processor with the lowest ID survives.
//! S-CRCW-COMMON: All concurrent writers must produce the same value;
//!   if not, the step is undefined (runtime error).
//! S-CRCW-ARBITRARY: One arbitrary write survives (first encountered).

use std::collections::HashMap;
use super::ast::{BinOp, Expr, MemoryModel, Stmt, UnaryOp, WriteResolution};
use super::metatheory::Value;

/// The shared memory store: maps (region_name, index) → Value.
#[derive(Debug, Clone, Default)]
pub struct Store {
    pub locals: HashMap<String, Value>,
    pub shared: HashMap<String, HashMap<usize, Value>>,
}

impl Store {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_local(&self, name: &str) -> Option<&Value> {
        self.locals.get(name)
    }

    pub fn set_local(&mut self, name: &str, val: Value) {
        self.locals.insert(name.to_string(), val);
    }

    pub fn read_shared(&self, region: &str, index: usize) -> Option<&Value> {
        self.shared.get(region).and_then(|m| m.get(&index))
    }

    pub fn write_shared(&mut self, region: &str, index: usize, val: Value) {
        self.shared
            .entry(region.to_string())
            .or_default()
            .insert(index, val);
    }

    pub fn alloc_shared(&mut self, region: &str, size: usize) {
        let mem = self.shared.entry(region.to_string()).or_default();
        for i in 0..size {
            mem.entry(i).or_insert(Value::IntVal(0));
        }
    }
}

/// Result of a single evaluation step.
#[derive(Debug, Clone)]
pub enum StepResult<T> {
    /// The term stepped to a new form.
    Stepped(T, Store),
    /// The term is a value (no further steps).
    Value(Value),
    /// The term is stuck (error).
    Stuck(String),
}

/// Evaluate an expression by one step under the given store.
pub fn step_expr(expr: &Expr, store: &Store, proc_id: Option<usize>, num_procs: Option<usize>) -> StepResult<Expr> {
    match expr {
        // E-CONST: literals are values
        Expr::IntLiteral(n) => StepResult::Value(Value::IntVal(*n)),
        Expr::FloatLiteral(f) => StepResult::Value(Value::FloatVal(*f)),
        Expr::BoolLiteral(b) => StepResult::Value(Value::BoolVal(*b)),

        // E-PROCID
        Expr::ProcessorId => {
            match proc_id {
                Some(id) => StepResult::Value(Value::IntVal(id as i64)),
                None => StepResult::Stuck("ProcessorId outside parallel context".into()),
            }
        }
        Expr::NumProcessors => {
            match num_procs {
                Some(n) => StepResult::Value(Value::IntVal(n as i64)),
                None => StepResult::Stuck("NumProcessors outside parallel context".into()),
            }
        }

        // E-VAR
        Expr::Variable(name) => {
            match store.get_local(name) {
                Some(v) => StepResult::Value(v.clone()),
                None => StepResult::Stuck(format!("Undefined variable: {}", name)),
            }
        }

        // E-BINOP: evaluate left, then right, then apply δ
        Expr::BinOp(op, left, right) => {
            let lv = eval_to_value(left, store, proc_id, num_procs);
            let rv = eval_to_value(right, store, proc_id, num_procs);
            match (lv, rv) {
                (Ok(l), Ok(r)) => {
                    match apply_delta(op, &l, &r) {
                        Some(result) => StepResult::Value(result),
                        None => StepResult::Stuck(format!("Cannot apply {:?} to {:?} and {:?}", op, l, r)),
                    }
                }
                (Err(e), _) | (_, Err(e)) => StepResult::Stuck(e),
            }
        }

        // E-UNARY
        Expr::UnaryOp(op, inner) => {
            let v = eval_to_value(inner, store, proc_id, num_procs);
            match v {
                Ok(val) => {
                    match apply_unary(op, &val) {
                        Some(result) => StepResult::Value(result),
                        None => StepResult::Stuck(format!("Cannot apply {:?} to {:?}", op, val)),
                    }
                }
                Err(e) => StepResult::Stuck(e),
            }
        }

        // E-READ
        Expr::SharedRead(mem_expr, idx_expr) => {
            if let Expr::Variable(region) = mem_expr.as_ref() {
                let idx = eval_to_value(idx_expr, store, proc_id, num_procs);
                match idx {
                    Ok(Value::IntVal(i)) => {
                        match store.read_shared(region, i as usize) {
                            Some(v) => StepResult::Value(v.clone()),
                            None => StepResult::Value(Value::IntVal(0)), // default
                        }
                    }
                    Ok(_) => StepResult::Stuck("Non-integer index".into()),
                    Err(e) => StepResult::Stuck(e),
                }
            } else {
                StepResult::Stuck("SharedRead on non-variable memory".into())
            }
        }

        // E-ARRAY-INDEX
        Expr::ArrayIndex(arr, idx) => {
            let av = eval_to_value(arr, store, proc_id, num_procs);
            let iv = eval_to_value(idx, store, proc_id, num_procs);
            match (av, iv) {
                (Ok(Value::ArrayVal(elems)), Ok(Value::IntVal(i))) => {
                    let i = i as usize;
                    if i < elems.len() {
                        StepResult::Value(elems[i].clone())
                    } else {
                        StepResult::Stuck(format!("Array index {} out of bounds (len {})", i, elems.len()))
                    }
                }
                _ => StepResult::Value(Value::IntVal(0)),
            }
        }

        // E-COND
        Expr::Conditional(cond, then_e, else_e) => {
            let cv = eval_to_value(cond, store, proc_id, num_procs);
            match cv {
                Ok(Value::BoolVal(true)) => step_expr(then_e, store, proc_id, num_procs),
                Ok(Value::BoolVal(false)) => step_expr(else_e, store, proc_id, num_procs),
                Ok(Value::IntVal(n)) => {
                    if n != 0 {
                        step_expr(then_e, store, proc_id, num_procs)
                    } else {
                        step_expr(else_e, store, proc_id, num_procs)
                    }
                }
                Ok(_) => StepResult::Stuck("Non-boolean condition".into()),
                Err(e) => StepResult::Stuck(e),
            }
        }

        // E-CAST
        Expr::Cast(inner, _ty) => step_expr(inner, store, proc_id, num_procs),

        // E-CALL (uninterpreted)
        Expr::FunctionCall(name, _args) => {
            StepResult::Stuck(format!("Cannot step function call: {}", name))
        }
    }
}

/// Fully evaluate an expression to a value.
pub fn eval_to_value(expr: &Expr, store: &Store, proc_id: Option<usize>, num_procs: Option<usize>) -> Result<Value, String> {
    match step_expr(expr, store, proc_id, num_procs) {
        StepResult::Value(v) => Ok(v),
        StepResult::Stepped(e, new_store) => eval_to_value(&e, &new_store, proc_id, num_procs),
        StepResult::Stuck(msg) => Err(msg),
    }
}

/// Apply the δ function for binary operators.
fn apply_delta(op: &BinOp, left: &Value, right: &Value) -> Option<Value> {
    match (left, right) {
        (Value::IntVal(a), Value::IntVal(b)) => {
            Some(match op {
                BinOp::Add => Value::IntVal(a.wrapping_add(*b)),
                BinOp::Sub => Value::IntVal(a.wrapping_sub(*b)),
                BinOp::Mul => Value::IntVal(a.wrapping_mul(*b)),
                BinOp::Div => if *b != 0 { Value::IntVal(a / b) } else { return None },
                BinOp::Mod => if *b != 0 { Value::IntVal(a % b) } else { return None },
                BinOp::Lt => Value::BoolVal(a < b),
                BinOp::Le => Value::BoolVal(a <= b),
                BinOp::Gt => Value::BoolVal(a > b),
                BinOp::Ge => Value::BoolVal(a >= b),
                BinOp::Eq => Value::BoolVal(a == b),
                BinOp::Ne => Value::BoolVal(a != b),
                BinOp::And => Value::BoolVal(*a != 0 && *b != 0),
                BinOp::Or => Value::BoolVal(*a != 0 || *b != 0),
                BinOp::BitAnd => Value::IntVal(a & b),
                BinOp::BitOr => Value::IntVal(a | b),
                BinOp::BitXor => Value::IntVal(a ^ b),
                BinOp::Shl => Value::IntVal(a.wrapping_shl(*b as u32)),
                BinOp::Shr => Value::IntVal(a.wrapping_shr(*b as u32)),
                BinOp::Min => Value::IntVal(*a.min(b)),
                BinOp::Max => Value::IntVal(*a.max(b)),
            })
        }
        (Value::FloatVal(a), Value::FloatVal(b)) => {
            Some(match op {
                BinOp::Add => Value::FloatVal(a + b),
                BinOp::Sub => Value::FloatVal(a - b),
                BinOp::Mul => Value::FloatVal(a * b),
                BinOp::Div => Value::FloatVal(a / b),
                BinOp::Lt => Value::BoolVal(a < b),
                BinOp::Le => Value::BoolVal(a <= b),
                BinOp::Gt => Value::BoolVal(a > b),
                BinOp::Ge => Value::BoolVal(a >= b),
                BinOp::Eq => Value::BoolVal((a - b).abs() < f64::EPSILON),
                BinOp::Ne => Value::BoolVal((a - b).abs() >= f64::EPSILON),
                BinOp::Min => Value::FloatVal(a.min(*b)),
                BinOp::Max => Value::FloatVal(a.max(*b)),
                _ => return None,
            })
        }
        (Value::IntVal(a), Value::FloatVal(b)) => {
            apply_delta(op, &Value::FloatVal(*a as f64), &Value::FloatVal(*b))
        }
        (Value::FloatVal(a), Value::IntVal(b)) => {
            apply_delta(op, &Value::FloatVal(*a), &Value::FloatVal(*b as f64))
        }
        (Value::BoolVal(a), Value::BoolVal(b)) => {
            Some(match op {
                BinOp::And => Value::BoolVal(*a && *b),
                BinOp::Or => Value::BoolVal(*a || *b),
                BinOp::Eq => Value::BoolVal(a == b),
                BinOp::Ne => Value::BoolVal(a != b),
                _ => return None,
            })
        }
        _ => None,
    }
}

/// Apply unary operator.
fn apply_unary(op: &UnaryOp, val: &Value) -> Option<Value> {
    match (op, val) {
        (UnaryOp::Neg, Value::IntVal(n)) => Some(Value::IntVal(-n)),
        (UnaryOp::Neg, Value::FloatVal(f)) => Some(Value::FloatVal(-f)),
        (UnaryOp::Not, Value::BoolVal(b)) => Some(Value::BoolVal(!b)),
        (UnaryOp::Not, Value::IntVal(n)) => Some(Value::IntVal(!n)),
        (UnaryOp::BitNot, Value::IntVal(n)) => Some(Value::IntVal(!n)),
        _ => None,
    }
}

/// Execute a statement under the given store, returning the new store.
pub fn exec_stmt(stmt: &Stmt, store: &mut Store, proc_id: Option<usize>, num_procs: Option<usize>, model: MemoryModel) -> Result<(), String> {
    match stmt {
        Stmt::Nop | Stmt::Comment(_) | Stmt::Barrier => Ok(()),

        Stmt::Assign(name, expr) => {
            let v = eval_to_value(expr, store, proc_id, num_procs)?;
            store.set_local(name, v);
            Ok(())
        }

        Stmt::LocalDecl(name, _ty, init) => {
            let v = if let Some(expr) = init {
                eval_to_value(expr, store, proc_id, num_procs)?
            } else {
                Value::IntVal(0)
            };
            store.set_local(name, v);
            Ok(())
        }

        Stmt::SharedWrite { memory, index, value } => {
            if let Expr::Variable(region) = memory {
                let idx = eval_to_value(index, store, proc_id, num_procs)?;
                let val = eval_to_value(value, store, proc_id, num_procs)?;
                if let Value::IntVal(i) = idx {
                    store.write_shared(region, i as usize, val);
                    Ok(())
                } else {
                    Err("Non-integer shared write index".into())
                }
            } else {
                Err("SharedWrite on non-variable memory".into())
            }
        }

        Stmt::Block(stmts) => {
            for s in stmts {
                exec_stmt(s, store, proc_id, num_procs, model)?;
            }
            Ok(())
        }

        Stmt::If { condition, then_body, else_body } => {
            let cv = eval_to_value(condition, store, proc_id, num_procs)?;
            let branch = match cv {
                Value::BoolVal(true) => then_body,
                Value::BoolVal(false) => else_body,
                Value::IntVal(n) => if n != 0 { then_body } else { else_body },
                _ => return Err("Non-boolean if condition".into()),
            };
            for s in branch {
                exec_stmt(s, store, proc_id, num_procs, model)?;
            }
            Ok(())
        }

        Stmt::SeqFor { var, start, end, step, body } => {
            let s = eval_to_value(start, store, proc_id, num_procs)?;
            let e = eval_to_value(end, store, proc_id, num_procs)?;
            let st = if let Some(step_e) = step {
                eval_to_value(step_e, store, proc_id, num_procs)?
            } else {
                Value::IntVal(1)
            };

            if let (Value::IntVal(mut i), Value::IntVal(end_val), Value::IntVal(step_val)) = (s, e, st) {
                if step_val <= 0 { return Err("Non-positive step".into()); }
                while i < end_val {
                    store.set_local(var, Value::IntVal(i));
                    for s in body {
                        exec_stmt(s, store, proc_id, num_procs, model)?;
                    }
                    i += step_val;
                }
                Ok(())
            } else {
                Err("Non-integer loop bounds".into())
            }
        }

        // S-PAR: parallel for is sequentialized under Brent scheduling
        Stmt::ParallelFor { proc_var, num_procs: np_expr, body } => {
            let np = eval_to_value(np_expr, store, proc_id, num_procs)?;
            if let Value::IntVal(n) = np {
                // Collect writes per-processor for CRCW resolution
                let mut pending_writes: Vec<(usize, String, usize, Value)> = Vec::new();
                let n = n as usize;

                for pid in 0..n {
                    let mut proc_store = store.clone();
                    proc_store.set_local(proc_var, Value::IntVal(pid as i64));

                    for s in body {
                        exec_stmt_with_write_collection(
                            s, &mut proc_store, Some(pid), Some(n),
                            model, &mut pending_writes, pid,
                        )?;
                    }

                    // Copy local variable updates back
                    for (k, v) in &proc_store.locals {
                        if k != proc_var {
                            store.set_local(k, v.clone());
                        }
                    }
                }

                // Resolve CRCW conflicts
                resolve_pending_writes(store, &pending_writes, model)?;
                Ok(())
            } else {
                Err("Non-integer processor count".into())
            }
        }

        Stmt::Return(_) => Ok(()),

        Stmt::AllocShared { name, size, .. } => {
            let sz = eval_to_value(size, store, proc_id, num_procs)?;
            if let Value::IntVal(n) = sz {
                store.alloc_shared(name, n as usize);
                Ok(())
            } else {
                Err("Non-integer alloc size".into())
            }
        }

        Stmt::While { condition, body } => {
            let max_iters = 10000;
            for _ in 0..max_iters {
                let cv = eval_to_value(condition, store, proc_id, num_procs)?;
                match cv {
                    Value::BoolVal(false) | Value::IntVal(0) => return Ok(()),
                    _ => {
                        for s in body {
                            exec_stmt(s, store, proc_id, num_procs, model)?;
                        }
                    }
                }
            }
            Err("While loop exceeded max iterations".into())
        }

        _ => Ok(()), // Other statements are no-ops in this semantics
    }
}

/// Execute a statement, collecting shared writes for later CRCW resolution.
fn exec_stmt_with_write_collection(
    stmt: &Stmt,
    store: &mut Store,
    proc_id: Option<usize>,
    num_procs: Option<usize>,
    model: MemoryModel,
    pending_writes: &mut Vec<(usize, String, usize, Value)>,
    current_proc: usize,
) -> Result<(), String> {
    match stmt {
        Stmt::SharedWrite { memory, index, value } => {
            if let Expr::Variable(region) = memory {
                let idx = eval_to_value(index, store, proc_id, num_procs)?;
                let val = eval_to_value(value, store, proc_id, num_procs)?;
                if let Value::IntVal(i) = idx {
                    pending_writes.push((current_proc, region.clone(), i as usize, val));
                    Ok(())
                } else {
                    Err("Non-integer write index".into())
                }
            } else {
                Err("SharedWrite on non-variable memory".into())
            }
        }
        Stmt::Block(stmts) => {
            for s in stmts {
                exec_stmt_with_write_collection(s, store, proc_id, num_procs, model, pending_writes, current_proc)?;
            }
            Ok(())
        }
        Stmt::If { condition, then_body, else_body } => {
            let cv = eval_to_value(condition, store, proc_id, num_procs)?;
            let branch = match cv {
                Value::BoolVal(true) | Value::IntVal(1..) => then_body,
                _ => else_body,
            };
            for s in branch {
                exec_stmt_with_write_collection(s, store, proc_id, num_procs, model, pending_writes, current_proc)?;
            }
            Ok(())
        }
        _ => exec_stmt(stmt, store, proc_id, num_procs, model),
    }
}

/// Resolve pending CRCW writes according to the memory model.
fn resolve_pending_writes(
    store: &mut Store,
    writes: &[(usize, String, usize, Value)],
    model: MemoryModel,
) -> Result<(), String> {
    // Group writes by (region, address)
    let mut groups: HashMap<(String, usize), Vec<(usize, Value)>> = HashMap::new();
    for (proc_id, region, addr, val) in writes {
        groups.entry((region.clone(), *addr)).or_default().push((*proc_id, val.clone()));
    }

    for ((region, addr), writers) in &groups {
        if writers.len() <= 1 {
            if let Some((_, val)) = writers.first() {
                store.write_shared(region, *addr, val.clone());
            }
            continue;
        }

        // Multiple writers: resolve according to model
        match model {
            MemoryModel::CRCWPriority => {
                // Lowest processor ID wins
                let winner = writers.iter().min_by_key(|(pid, _)| *pid).unwrap();
                store.write_shared(region, *addr, winner.1.clone());
            }
            MemoryModel::CRCWArbitrary => {
                // First writer wins
                store.write_shared(region, *addr, writers[0].1.clone());
            }
            MemoryModel::CRCWCommon => {
                // All must agree
                let first_val = &writers[0].1;
                if writers.iter().all(|(_, v)| v == first_val) {
                    store.write_shared(region, *addr, first_val.clone());
                } else {
                    return Err(format!(
                        "CRCW-Common conflict: {} writers to {}[{}] disagree",
                        writers.len(), region, addr
                    ));
                }
            }
            MemoryModel::EREW | MemoryModel::CREW => {
                return Err(format!(
                    "EREW/CREW violation: {} concurrent writes to {}[{}]",
                    writers.len(), region, addr
                ));
            }
        }
    }

    Ok(())
}

/// Verify that a program produces the expected output under given semantics.
pub fn verify_execution(
    body: &[Stmt],
    initial_store: &Store,
    model: MemoryModel,
    expected_shared: &HashMap<String, HashMap<usize, Value>>,
) -> Result<bool, String> {
    let mut store = initial_store.clone();
    for stmt in body {
        exec_stmt(stmt, &mut store, None, None, model)?;
    }

    for (region, expected_mem) in expected_shared {
        for (addr, expected_val) in expected_mem {
            match store.read_shared(region, *addr) {
                Some(actual) if actual == expected_val => {}
                Some(actual) => {
                    return Err(format!(
                        "Mismatch at {}[{}]: expected {:?}, got {:?}",
                        region, addr, expected_val, actual
                    ));
                }
                None => {
                    return Err(format!("Missing value at {}[{}]", region, addr));
                }
            }
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pram_ir::ast::{BinOp, Expr, Stmt};

    #[test]
    fn test_eval_int_literal() {
        let store = Store::new();
        let result = eval_to_value(&Expr::IntLiteral(42), &store, None, None);
        assert_eq!(result, Ok(Value::IntVal(42)));
    }

    #[test]
    fn test_eval_binop_add() {
        let store = Store::new();
        let expr = Expr::BinOp(BinOp::Add, Box::new(Expr::IntLiteral(3)), Box::new(Expr::IntLiteral(4)));
        let result = eval_to_value(&expr, &store, None, None);
        assert_eq!(result, Ok(Value::IntVal(7)));
    }

    #[test]
    fn test_eval_binop_comparison() {
        let store = Store::new();
        let expr = Expr::BinOp(BinOp::Lt, Box::new(Expr::IntLiteral(3)), Box::new(Expr::IntLiteral(5)));
        let result = eval_to_value(&expr, &store, None, None);
        assert_eq!(result, Ok(Value::BoolVal(true)));
    }

    #[test]
    fn test_eval_variable() {
        let mut store = Store::new();
        store.set_local("x", Value::IntVal(10));
        let result = eval_to_value(&Expr::Variable("x".into()), &store, None, None);
        assert_eq!(result, Ok(Value::IntVal(10)));
    }

    #[test]
    fn test_eval_processor_id() {
        let store = Store::new();
        let result = eval_to_value(&Expr::ProcessorId, &store, Some(3), Some(8));
        assert_eq!(result, Ok(Value::IntVal(3)));
    }

    #[test]
    fn test_eval_conditional() {
        let store = Store::new();
        let expr = Expr::Conditional(
            Box::new(Expr::BoolLiteral(true)),
            Box::new(Expr::IntLiteral(1)),
            Box::new(Expr::IntLiteral(2)),
        );
        let result = eval_to_value(&expr, &store, None, None);
        assert_eq!(result, Ok(Value::IntVal(1)));
    }

    #[test]
    fn test_exec_assign() {
        let mut store = Store::new();
        let stmt = Stmt::Assign("x".into(), Expr::IntLiteral(42));
        exec_stmt(&stmt, &mut store, None, None, MemoryModel::EREW).unwrap();
        assert_eq!(store.get_local("x"), Some(&Value::IntVal(42)));
    }

    #[test]
    fn test_exec_shared_write_read() {
        let mut store = Store::new();
        store.alloc_shared("A", 10);
        let write = Stmt::SharedWrite {
            memory: Expr::Variable("A".into()),
            index: Expr::IntLiteral(5),
            value: Expr::IntLiteral(99),
        };
        exec_stmt(&write, &mut store, None, None, MemoryModel::EREW).unwrap();
        assert_eq!(store.read_shared("A", 5), Some(&Value::IntVal(99)));
    }

    #[test]
    fn test_exec_if_true_branch() {
        let mut store = Store::new();
        let stmt = Stmt::If {
            condition: Expr::BoolLiteral(true),
            then_body: vec![Stmt::Assign("x".into(), Expr::IntLiteral(1))],
            else_body: vec![Stmt::Assign("x".into(), Expr::IntLiteral(2))],
        };
        exec_stmt(&stmt, &mut store, None, None, MemoryModel::EREW).unwrap();
        assert_eq!(store.get_local("x"), Some(&Value::IntVal(1)));
    }

    #[test]
    fn test_exec_seq_for() {
        let mut store = Store::new();
        store.alloc_shared("A", 5);
        let stmt = Stmt::SeqFor {
            var: "i".into(),
            start: Expr::IntLiteral(0),
            end: Expr::IntLiteral(5),
            step: None,
            body: vec![Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::Variable("i".into()),
                value: Expr::BinOp(
                    BinOp::Mul,
                    Box::new(Expr::Variable("i".into())),
                    Box::new(Expr::IntLiteral(2)),
                ),
            }],
        };
        exec_stmt(&stmt, &mut store, None, None, MemoryModel::EREW).unwrap();
        for i in 0..5 {
            assert_eq!(store.read_shared("A", i), Some(&Value::IntVal(i as i64 * 2)));
        }
    }

    #[test]
    fn test_exec_parallel_for_erew() {
        let mut store = Store::new();
        store.alloc_shared("A", 4);
        let stmt = Stmt::ParallelFor {
            proc_var: "pid".into(),
            num_procs: Expr::IntLiteral(4),
            body: vec![Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::ProcessorId,
                value: Expr::BinOp(
                    BinOp::Add,
                    Box::new(Expr::ProcessorId),
                    Box::new(Expr::IntLiteral(10)),
                ),
            }],
        };
        exec_stmt(&stmt, &mut store, None, None, MemoryModel::CRCWPriority).unwrap();
        for i in 0..4 {
            assert_eq!(store.read_shared("A", i), Some(&Value::IntVal(i as i64 + 10)));
        }
    }

    #[test]
    fn test_crcw_priority_resolution() {
        let mut store = Store::new();
        store.alloc_shared("A", 1);
        // All 4 processors write to A[0] with their PID
        let stmt = Stmt::ParallelFor {
            proc_var: "pid".into(),
            num_procs: Expr::IntLiteral(4),
            body: vec![Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(0),
                value: Expr::ProcessorId,
            }],
        };
        exec_stmt(&stmt, &mut store, None, None, MemoryModel::CRCWPriority).unwrap();
        // Lowest PID (0) should win
        assert_eq!(store.read_shared("A", 0), Some(&Value::IntVal(0)));
    }

    #[test]
    fn test_crcw_common_agreement() {
        let mut store = Store::new();
        store.alloc_shared("A", 1);
        // All processors write the same value (42) to A[0]
        let stmt = Stmt::ParallelFor {
            proc_var: "pid".into(),
            num_procs: Expr::IntLiteral(4),
            body: vec![Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(0),
                value: Expr::IntLiteral(42),
            }],
        };
        exec_stmt(&stmt, &mut store, None, None, MemoryModel::CRCWCommon).unwrap();
        assert_eq!(store.read_shared("A", 0), Some(&Value::IntVal(42)));
    }

    #[test]
    fn test_crcw_common_conflict_error() {
        let mut store = Store::new();
        store.alloc_shared("A", 1);
        // Processors write different values to A[0]
        let stmt = Stmt::ParallelFor {
            proc_var: "pid".into(),
            num_procs: Expr::IntLiteral(4),
            body: vec![Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(0),
                value: Expr::ProcessorId,
            }],
        };
        let result = exec_stmt(&stmt, &mut store, None, None, MemoryModel::CRCWCommon);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("CRCW-Common conflict"));
    }

    #[test]
    fn test_verify_execution() {
        let mut store = Store::new();
        store.alloc_shared("A", 3);
        let body = vec![
            Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(0),
                value: Expr::IntLiteral(10),
            },
            Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(1),
                value: Expr::IntLiteral(20),
            },
        ];
        let mut expected = HashMap::new();
        let mut region = HashMap::new();
        region.insert(0, Value::IntVal(10));
        region.insert(1, Value::IntVal(20));
        expected.insert("A".to_string(), region);

        let result = verify_execution(&body, &store, MemoryModel::EREW, &expected);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_eval_unary_neg() {
        let store = Store::new();
        let expr = Expr::UnaryOp(UnaryOp::Neg, Box::new(Expr::IntLiteral(5)));
        let result = eval_to_value(&expr, &store, None, None);
        assert_eq!(result, Ok(Value::IntVal(-5)));
    }

    #[test]
    fn test_eval_unary_not() {
        let store = Store::new();
        let expr = Expr::UnaryOp(UnaryOp::Not, Box::new(Expr::BoolLiteral(true)));
        let result = eval_to_value(&expr, &store, None, None);
        assert_eq!(result, Ok(Value::BoolVal(false)));
    }

    #[test]
    fn test_eval_shared_read() {
        let mut store = Store::new();
        store.write_shared("B", 3, Value::IntVal(77));
        let expr = Expr::SharedRead(
            Box::new(Expr::Variable("B".into())),
            Box::new(Expr::IntLiteral(3)),
        );
        let result = eval_to_value(&expr, &store, None, None);
        assert_eq!(result, Ok(Value::IntVal(77)));
    }

    #[test]
    fn test_exec_while_loop() {
        let mut store = Store::new();
        store.set_local("count", Value::IntVal(0));
        let stmt = Stmt::While {
            condition: Expr::BinOp(
                BinOp::Lt,
                Box::new(Expr::Variable("count".into())),
                Box::new(Expr::IntLiteral(5)),
            ),
            body: vec![Stmt::Assign(
                "count".into(),
                Expr::BinOp(
                    BinOp::Add,
                    Box::new(Expr::Variable("count".into())),
                    Box::new(Expr::IntLiteral(1)),
                ),
            )],
        };
        exec_stmt(&stmt, &mut store, None, None, MemoryModel::EREW).unwrap();
        assert_eq!(store.get_local("count"), Some(&Value::IntVal(5)));
    }

    #[test]
    fn test_alloc_shared() {
        let mut store = Store::new();
        let stmt = Stmt::AllocShared {
            name: "C".into(),
            size: Expr::IntLiteral(10),
            elem_type: crate::pram_ir::types::PramType::Int64,
        };
        exec_stmt(&stmt, &mut store, None, None, MemoryModel::EREW).unwrap();
        assert_eq!(store.read_shared("C", 0), Some(&Value::IntVal(0)));
        assert_eq!(store.read_shared("C", 9), Some(&Value::IntVal(0)));
    }

    #[test]
    fn test_exec_block() {
        let mut store = Store::new();
        let stmt = Stmt::Block(vec![
            Stmt::Assign("a".into(), Expr::IntLiteral(1)),
            Stmt::Assign("b".into(), Expr::IntLiteral(2)),
        ]);
        exec_stmt(&stmt, &mut store, None, None, MemoryModel::EREW).unwrap();
        assert_eq!(store.get_local("a"), Some(&Value::IntVal(1)));
        assert_eq!(store.get_local("b"), Some(&Value::IntVal(2)));
    }

    #[test]
    fn test_erew_concurrent_write_error() {
        let mut store = Store::new();
        store.alloc_shared("A", 1);
        let stmt = Stmt::ParallelFor {
            proc_var: "pid".into(),
            num_procs: Expr::IntLiteral(2),
            body: vec![Stmt::SharedWrite {
                memory: Expr::Variable("A".into()),
                index: Expr::IntLiteral(0),
                value: Expr::ProcessorId,
            }],
        };
        let result = exec_stmt(&stmt, &mut store, None, None, MemoryModel::EREW);
        assert!(result.is_err());
    }

    #[test]
    fn test_delta_div_by_zero() {
        let result = apply_delta(&BinOp::Div, &Value::IntVal(10), &Value::IntVal(0));
        assert!(result.is_none());
    }

    #[test]
    fn test_delta_bitwise_ops() {
        assert_eq!(apply_delta(&BinOp::BitAnd, &Value::IntVal(0xFF), &Value::IntVal(0x0F)),
                   Some(Value::IntVal(0x0F)));
        assert_eq!(apply_delta(&BinOp::BitOr, &Value::IntVal(0xF0), &Value::IntVal(0x0F)),
                   Some(Value::IntVal(0xFF)));
        assert_eq!(apply_delta(&BinOp::BitXor, &Value::IntVal(0xFF), &Value::IntVal(0xFF)),
                   Some(Value::IntVal(0)));
    }

    #[test]
    fn test_delta_min_max() {
        assert_eq!(apply_delta(&BinOp::Min, &Value::IntVal(3), &Value::IntVal(7)),
                   Some(Value::IntVal(3)));
        assert_eq!(apply_delta(&BinOp::Max, &Value::IntVal(3), &Value::IntVal(7)),
                   Some(Value::IntVal(7)));
    }
}
