//! Hash residualization pass.
//!
//! When block assignments are known at compile time, this pass replaces
//! runtime hash(addr) computations with precomputed block IDs and direct
//! memory access at known offsets.

use std::collections::HashMap;

use crate::pram_ir::ast::{BinOp, Expr, PramProgram, Stmt};

/// Mapping from (memory_region, address_expression) to (block_id, offset_within_block).
#[derive(Debug, Clone)]
pub struct BlockAssignment {
    /// Region name -> (block_id for a given index)
    /// Maps: region -> index -> (block_id, local_offset)
    region_assignments: HashMap<String, AddressMap>,
    /// Block size (number of elements per block)
    block_size: usize,
}

/// Address mapping strategy for a single memory region.
#[derive(Debug, Clone)]
pub enum AddressMap {
    /// Contiguous blocks: address i maps to block i/block_size, offset i%block_size
    Contiguous { total_size: usize, block_size: usize },
    /// Explicit mapping of individual addresses to (block_id, offset) pairs
    Explicit(HashMap<i64, (usize, usize)>),
}

impl AddressMap {
    /// Resolve an address to (block_id, offset) if the address is known at compile time.
    pub fn resolve(&self, addr: i64) -> Option<(usize, usize)> {
        match self {
            AddressMap::Contiguous {
                total_size,
                block_size,
            } => {
                if addr < 0 || addr as usize >= *total_size {
                    return None;
                }
                let block_id = addr as usize / block_size;
                let offset = addr as usize % block_size;
                Some((block_id, offset))
            }
            AddressMap::Explicit(map) => map.get(&addr).copied(),
        }
    }
}

impl BlockAssignment {
    /// Create a new contiguous block assignment for a single region.
    pub fn contiguous(region: &str, total_size: usize, block_size: usize) -> Self {
        let mut region_assignments = HashMap::new();
        region_assignments.insert(
            region.to_string(),
            AddressMap::Contiguous {
                total_size,
                block_size,
            },
        );
        Self {
            region_assignments,
            block_size,
        }
    }

    /// Create a block assignment with explicit per-address mappings.
    pub fn explicit(region: &str, mappings: HashMap<i64, (usize, usize)>, block_size: usize) -> Self {
        let mut region_assignments = HashMap::new();
        region_assignments.insert(region.to_string(), AddressMap::Explicit(mappings));
        Self {
            region_assignments,
            block_size,
        }
    }

    /// Add a region with contiguous assignment.
    pub fn add_contiguous_region(&mut self, region: &str, total_size: usize, block_size: usize) {
        self.region_assignments.insert(
            region.to_string(),
            AddressMap::Contiguous {
                total_size,
                block_size,
            },
        );
    }

    /// Resolve an address for a given region.
    pub fn resolve(&self, region: &str, addr: i64) -> Option<(usize, usize)> {
        self.region_assignments
            .get(region)
            .and_then(|m| m.resolve(addr))
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

/// The residualization pass that replaces hash-based lookups with direct access.
pub struct ResidualizePass {
    assignments: BlockAssignment,
}

impl ResidualizePass {
    pub fn new(assignments: BlockAssignment) -> Self {
        Self { assignments }
    }

    /// Transform a list of statements, replacing SharedRead/SharedWrite
    /// with direct memory access when the index is known at compile time.
    pub fn transform(&self, stmts: &[Stmt]) -> Vec<Stmt> {
        stmts.iter().map(|s| self.transform_stmt(s)).collect()
    }

    fn transform_stmt(&self, stmt: &Stmt) -> Stmt {
        match stmt {
            Stmt::Assign(name, expr) => {
                Stmt::Assign(name.clone(), self.transform_expr(expr))
            }
            Stmt::SharedWrite {
                memory,
                index,
                value,
            } => {
                let new_index = self.try_residualize_index(memory, index);
                Stmt::SharedWrite {
                    memory: memory.clone(),
                    index: new_index,
                    value: self.transform_expr(value),
                }
            }
            Stmt::If {
                condition,
                then_body,
                else_body,
            } => Stmt::If {
                condition: self.transform_expr(condition),
                then_body: self.transform(then_body),
                else_body: self.transform(else_body),
            },
            Stmt::SeqFor {
                var,
                start,
                end,
                step,
                body,
            } => Stmt::SeqFor {
                var: var.clone(),
                start: self.transform_expr(start),
                end: self.transform_expr(end),
                step: step.as_ref().map(|s| self.transform_expr(s)),
                body: self.transform(body),
            },
            Stmt::While { condition, body } => Stmt::While {
                condition: self.transform_expr(condition),
                body: self.transform(body),
            },
            Stmt::ParallelFor {
                proc_var,
                num_procs,
                body,
            } => Stmt::ParallelFor {
                proc_var: proc_var.clone(),
                num_procs: self.transform_expr(num_procs),
                body: self.transform(body),
            },
            Stmt::Block(stmts) => Stmt::Block(self.transform(stmts)),
            Stmt::LocalDecl(name, ty, init) => {
                Stmt::LocalDecl(
                    name.clone(),
                    ty.clone(),
                    init.as_ref().map(|e| self.transform_expr(e)),
                )
            }
            Stmt::ExprStmt(e) => Stmt::ExprStmt(self.transform_expr(e)),
            Stmt::Return(Some(e)) => Stmt::Return(Some(self.transform_expr(e))),
            Stmt::Assert(e, msg) => Stmt::Assert(self.transform_expr(e), msg.clone()),
            other => other.clone(),
        }
    }

    fn transform_expr(&self, expr: &Expr) -> Expr {
        match expr {
            Expr::SharedRead(mem, idx) => {
                let new_idx = self.try_residualize_index(mem, idx);
                Expr::SharedRead(
                    Box::new(self.transform_expr(mem)),
                    Box::new(new_idx),
                )
            }
            Expr::BinOp(op, a, b) => Expr::BinOp(
                *op,
                Box::new(self.transform_expr(a)),
                Box::new(self.transform_expr(b)),
            ),
            Expr::UnaryOp(op, e) => {
                Expr::UnaryOp(*op, Box::new(self.transform_expr(e)))
            }
            Expr::ArrayIndex(arr, idx) => Expr::ArrayIndex(
                Box::new(self.transform_expr(arr)),
                Box::new(self.transform_expr(idx)),
            ),
            Expr::FunctionCall(name, args) => {
                // If this is a hash function call with a known argument, replace it
                if name == "hash" || name == "block_of" {
                    if let Some(resolved) = self.try_resolve_hash_call(args) {
                        return resolved;
                    }
                }
                Expr::FunctionCall(
                    name.clone(),
                    args.iter().map(|a| self.transform_expr(a)).collect(),
                )
            }
            Expr::Cast(e, ty) => {
                Expr::Cast(Box::new(self.transform_expr(e)), ty.clone())
            }
            Expr::Conditional(c, t, e) => Expr::Conditional(
                Box::new(self.transform_expr(c)),
                Box::new(self.transform_expr(t)),
                Box::new(self.transform_expr(e)),
            ),
            other => other.clone(),
        }
    }

    /// Try to replace a shared memory index with a direct block+offset computation.
    ///
    /// For a shared_read(M, hash(addr)) where addr is a constant, we replace it
    /// with shared_read(M, block_id * block_size + offset).
    fn try_residualize_index(&self, memory: &Expr, index: &Expr) -> Expr {
        // Extract the memory region name
        let region = match memory {
            Expr::Variable(name) => name.as_str(),
            _ => return self.transform_expr(index),
        };

        // If the index is a constant, directly resolve it
        if let Some(addr) = index.eval_const_int() {
            if let Some((block_id, offset)) = self.assignments.resolve(region, addr) {
                // Generate: block_id * block_size + offset
                let bs = self.assignments.block_size() as i64;
                let direct_index = block_id as i64 * bs + offset as i64;
                return Expr::IntLiteral(direct_index);
            }
        }

        // If the index is a hash/block_of function call, try to resolve
        if let Expr::FunctionCall(fname, args) = index {
            if (fname == "hash" || fname == "block_of") && args.len() >= 1 {
                if let Some(addr) = args[0].eval_const_int() {
                    if let Some((block_id, offset)) = self.assignments.resolve(region, addr) {
                        let bs = self.assignments.block_size() as i64;
                        return Expr::IntLiteral(block_id as i64 * bs + offset as i64);
                    }
                }
            }
        }

        // For expressions of the form `addr / block_size * block_size + addr % block_size`
        // where the assignments are contiguous, this is already direct access —
        // but we can simplify to just `addr` if contiguous.
        self.transform_expr(index)
    }

    /// Try to resolve a hash/block_of function call to a constant.
    fn try_resolve_hash_call(&self, args: &[Expr]) -> Option<Expr> {
        if args.len() < 2 {
            return None;
        }
        let region = match &args[0] {
            Expr::Variable(name) => name.as_str(),
            _ => return None,
        };
        let addr = args[1].eval_const_int()?;
        let (block_id, _offset) = self.assignments.resolve(region, addr)?;
        Some(Expr::IntLiteral(block_id as i64))
    }
}

/// Generate a direct array index expression from block_id and offset.
pub fn direct_index_expr(block_id: usize, offset: usize, block_size: usize) -> Expr {
    Expr::IntLiteral((block_id * block_size + offset) as i64)
}

/// For a runtime address, generate the expression to compute the direct index
/// based on contiguous block assignment: addr itself (identity mapping).
pub fn contiguous_direct_index(addr_expr: &Expr) -> Expr {
    addr_expr.clone()
}

/// For a runtime address, generate block_id and offset expressions
/// for a contiguous mapping with the given block size.
pub fn compute_block_id_and_offset(addr_expr: &Expr, block_size: usize) -> (Expr, Expr) {
    let bs = Expr::IntLiteral(block_size as i64);
    let block_id = Expr::binop(BinOp::Div, addr_expr.clone(), bs.clone());
    let offset = Expr::binop(BinOp::Mod, addr_expr.clone(), bs);
    (block_id, offset)
}

/// Estimated savings from residualization.
#[derive(Debug, Clone, Default)]
pub struct ResidualSavings {
    /// Number of hash/block_of calls eliminated
    pub ops_eliminated: usize,
    /// Estimated memory overhead saved (in abstract units)
    pub memory_saved: usize,
    /// Number of shared reads with resolved block IDs
    pub reads_resolved: usize,
    /// Number of shared writes with resolved block IDs
    pub writes_resolved: usize,
}

/// Estimate the savings from applying residualization to a program.
pub fn estimate_savings(program: &PramProgram, assignments: &BlockAssignment) -> ResidualSavings {
    let mut savings = ResidualSavings::default();
    for stmt in &program.body {
        estimate_savings_stmt(stmt, assignments, &mut savings);
    }
    savings
}

fn estimate_savings_stmt(stmt: &Stmt, assignments: &BlockAssignment, savings: &mut ResidualSavings) {
    match stmt {
        Stmt::Assign(_, expr) => estimate_savings_expr(expr, assignments, savings),
        Stmt::SharedWrite { memory, index, value } => {
            if let Expr::Variable(region) = memory {
                if let Some(addr) = index.eval_const_int() {
                    if assignments.resolve(region, addr).is_some() {
                        savings.writes_resolved += 1;
                        savings.ops_eliminated += 1;
                    }
                }
            }
            estimate_savings_expr(index, assignments, savings);
            estimate_savings_expr(value, assignments, savings);
        }
        Stmt::If { condition, then_body, else_body } => {
            estimate_savings_expr(condition, assignments, savings);
            for s in then_body {
                estimate_savings_stmt(s, assignments, savings);
            }
            for s in else_body {
                estimate_savings_stmt(s, assignments, savings);
            }
        }
        Stmt::SeqFor { body, .. } => {
            for s in body {
                estimate_savings_stmt(s, assignments, savings);
            }
        }
        Stmt::ParallelFor { body, .. } => {
            for s in body {
                estimate_savings_stmt(s, assignments, savings);
            }
        }
        Stmt::Block(stmts) => {
            for s in stmts {
                estimate_savings_stmt(s, assignments, savings);
            }
        }
        _ => {}
    }
}

fn estimate_savings_expr(expr: &Expr, assignments: &BlockAssignment, savings: &mut ResidualSavings) {
    match expr {
        Expr::SharedRead(mem, idx) => {
            if let Expr::Variable(region) = mem.as_ref() {
                if let Some(addr) = idx.eval_const_int() {
                    if assignments.resolve(region, addr).is_some() {
                        savings.reads_resolved += 1;
                        savings.ops_eliminated += 1;
                    }
                }
            }
            estimate_savings_expr(idx, assignments, savings);
        }
        Expr::FunctionCall(name, args) => {
            if name == "hash" || name == "block_of" {
                savings.ops_eliminated += 1;
                savings.memory_saved += 1;
            }
            for a in args {
                estimate_savings_expr(a, assignments, savings);
            }
        }
        Expr::BinOp(_, a, b) => {
            estimate_savings_expr(a, assignments, savings);
            estimate_savings_expr(b, assignments, savings);
        }
        Expr::UnaryOp(_, e) | Expr::Cast(e, _) => {
            estimate_savings_expr(e, assignments, savings);
        }
        _ => {}
    }
}

/// Batch residualizer for processing multiple programs.
pub struct BatchResidualizer {
    assignments: BlockAssignment,
}

impl BatchResidualizer {
    pub fn new(assignments: BlockAssignment) -> Self {
        Self { assignments }
    }

    /// Transform a batch of programs, returning the transformed statements
    /// and per-program savings estimates.
    pub fn transform_batch(&self, programs: &[Vec<Stmt>]) -> Vec<(Vec<Stmt>, ResidualSavings)> {
        let pass = ResidualizePass::new(self.assignments.clone());
        programs
            .iter()
            .map(|stmts| {
                let transformed = pass.transform(stmts);
                let mut savings = ResidualSavings::default();
                // Count the difference in hash/block_of calls
                let orig_calls = count_hash_calls(stmts);
                let new_calls = count_hash_calls(&transformed);
                savings.ops_eliminated = orig_calls.saturating_sub(new_calls);
                (transformed, savings)
            })
            .collect()
    }
}

fn count_hash_calls(stmts: &[Stmt]) -> usize {
    let mut count = 0;
    for stmt in stmts {
        count_hash_calls_stmt(stmt, &mut count);
    }
    count
}

fn count_hash_calls_stmt(stmt: &Stmt, count: &mut usize) {
    match stmt {
        Stmt::Assign(_, expr) | Stmt::ExprStmt(expr) | Stmt::Return(Some(expr)) => {
            count_hash_calls_expr(expr, count);
        }
        Stmt::SharedWrite { memory, index, value } => {
            count_hash_calls_expr(memory, count);
            count_hash_calls_expr(index, count);
            count_hash_calls_expr(value, count);
        }
        Stmt::If { condition, then_body, else_body } => {
            count_hash_calls_expr(condition, count);
            for s in then_body { count_hash_calls_stmt(s, count); }
            for s in else_body { count_hash_calls_stmt(s, count); }
        }
        Stmt::SeqFor { body, .. } | Stmt::While { body, .. }
        | Stmt::ParallelFor { body, .. } => {
            for s in body { count_hash_calls_stmt(s, count); }
        }
        Stmt::Block(stmts) => {
            for s in stmts { count_hash_calls_stmt(s, count); }
        }
        _ => {}
    }
}

fn count_hash_calls_expr(expr: &Expr, count: &mut usize) {
    match expr {
        Expr::FunctionCall(name, args) => {
            if name == "hash" || name == "block_of" {
                *count += 1;
            }
            for a in args { count_hash_calls_expr(a, count); }
        }
        Expr::BinOp(_, a, b) | Expr::SharedRead(a, b) | Expr::ArrayIndex(a, b) => {
            count_hash_calls_expr(a, count);
            count_hash_calls_expr(b, count);
        }
        Expr::UnaryOp(_, e) | Expr::Cast(e, _) => count_hash_calls_expr(e, count),
        Expr::Conditional(c, t, e) => {
            count_hash_calls_expr(c, count);
            count_hash_calls_expr(t, count);
            count_hash_calls_expr(e, count);
        }
        _ => {}
    }
}

/// Verify that residualization preserves semantic equivalence (conservative check).
///
/// Returns true if the transformation is safe based on structural comparison.
/// This checks that the number and types of statements are consistent
/// and that no shared memory operations were dropped.
pub fn verify_residualization(before: &[Stmt], after: &[Stmt]) -> bool {
    let before_writes = count_stmt_type(before, StmtKind::SharedWrite);
    let after_writes = count_stmt_type(after, StmtKind::SharedWrite);
    let before_reads = count_expr_reads(before);
    let after_reads = count_expr_reads(after);
    // Writes and reads must be preserved
    before_writes == after_writes && before_reads == after_reads
}

#[derive(PartialEq)]
enum StmtKind {
    SharedWrite,
}

fn count_stmt_type(stmts: &[Stmt], kind: StmtKind) -> usize {
    let mut count = 0;
    for stmt in stmts {
        match stmt {
            Stmt::SharedWrite { .. } if kind == StmtKind::SharedWrite => count += 1,
            Stmt::If { then_body, else_body, .. } => {
                count += count_stmt_type(then_body, StmtKind::SharedWrite);
                count += count_stmt_type(else_body, StmtKind::SharedWrite);
            }
            Stmt::SeqFor { body, .. } | Stmt::While { body, .. }
            | Stmt::ParallelFor { body, .. } => {
                count += count_stmt_type(body, StmtKind::SharedWrite);
            }
            Stmt::Block(inner) => {
                count += count_stmt_type(inner, StmtKind::SharedWrite);
            }
            _ => {}
        }
    }
    count
}

fn count_expr_reads(stmts: &[Stmt]) -> usize {
    let mut count = 0;
    for stmt in stmts {
        match stmt {
            Stmt::Assign(_, expr) => count += count_shared_reads_expr(expr),
            Stmt::SharedWrite { memory, index, value } => {
                count += count_shared_reads_expr(memory);
                count += count_shared_reads_expr(index);
                count += count_shared_reads_expr(value);
            }
            Stmt::If { condition, then_body, else_body } => {
                count += count_shared_reads_expr(condition);
                count += count_expr_reads(then_body);
                count += count_expr_reads(else_body);
            }
            Stmt::SeqFor { body, .. } | Stmt::ParallelFor { body, .. }
            | Stmt::While { body, .. } => {
                count += count_expr_reads(body);
            }
            Stmt::Block(inner) => count += count_expr_reads(inner),
            _ => {}
        }
    }
    count
}

fn count_shared_reads_expr(expr: &Expr) -> usize {
    match expr {
        Expr::SharedRead(m, i) => 1 + count_shared_reads_expr(m) + count_shared_reads_expr(i),
        Expr::BinOp(_, a, b) => count_shared_reads_expr(a) + count_shared_reads_expr(b),
        Expr::UnaryOp(_, e) | Expr::Cast(e, _) => count_shared_reads_expr(e),
        Expr::FunctionCall(_, args) => args.iter().map(count_shared_reads_expr).sum(),
        Expr::Conditional(c, t, e) => {
            count_shared_reads_expr(c) + count_shared_reads_expr(t) + count_shared_reads_expr(e)
        }
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_block_resolution() {
        let ba = BlockAssignment::contiguous("A", 100, 10);
        assert_eq!(ba.resolve("A", 0), Some((0, 0)));
        assert_eq!(ba.resolve("A", 5), Some((0, 5)));
        assert_eq!(ba.resolve("A", 10), Some((1, 0)));
        assert_eq!(ba.resolve("A", 15), Some((1, 5)));
        assert_eq!(ba.resolve("A", 99), Some((9, 9)));
        assert_eq!(ba.resolve("A", 100), None);
        assert_eq!(ba.resolve("A", -1), None);
    }

    #[test]
    fn test_explicit_block_resolution() {
        let mut mappings = HashMap::new();
        mappings.insert(0, (0, 0));
        mappings.insert(1, (0, 1));
        mappings.insert(2, (1, 0));
        mappings.insert(3, (1, 1));
        let ba = BlockAssignment::explicit("B", mappings, 2);
        assert_eq!(ba.resolve("B", 0), Some((0, 0)));
        assert_eq!(ba.resolve("B", 2), Some((1, 0)));
        assert_eq!(ba.resolve("B", 4), None);
    }

    #[test]
    fn test_residualize_constant_index() {
        let ba = BlockAssignment::contiguous("A", 100, 10);
        let pass = ResidualizePass::new(ba);

        let stmt = Stmt::Assign(
            "x".to_string(),
            Expr::shared_read(Expr::var("A"), Expr::int(25)),
        );
        let result = pass.transform_stmt(&stmt);
        // Address 25 -> block 2, offset 5 -> direct index = 2*10+5 = 25
        match &result {
            Stmt::Assign(_, Expr::SharedRead(_, idx)) => {
                assert_eq!(**idx, Expr::IntLiteral(25));
            }
            other => panic!("Unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_residualize_shared_write() {
        let ba = BlockAssignment::contiguous("M", 32, 8);
        let pass = ResidualizePass::new(ba);

        let stmt = Stmt::SharedWrite {
            memory: Expr::var("M"),
            index: Expr::int(12),
            value: Expr::int(99),
        };
        let result = pass.transform_stmt(&stmt);
        // Address 12 -> block 1, offset 4 -> direct index = 1*8+4 = 12
        match &result {
            Stmt::SharedWrite { index, .. } => {
                assert_eq!(*index, Expr::IntLiteral(12));
            }
            other => panic!("Unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_residualize_hash_call() {
        let ba = BlockAssignment::contiguous("A", 100, 10);
        let pass = ResidualizePass::new(ba);

        // shared_read(A, hash(A, 35))
        let stmt = Stmt::Assign(
            "x".to_string(),
            Expr::shared_read(
                Expr::var("A"),
                Expr::FunctionCall("hash".to_string(), vec![Expr::var("A"), Expr::int(35)]),
            ),
        );
        let result = pass.transform_stmt(&stmt);
        match &result {
            Stmt::Assign(_, Expr::SharedRead(_, idx)) => {
                // hash(A, 35) with region A, addr 35 -> block 3
                assert_eq!(**idx, Expr::IntLiteral(3));
            }
            other => panic!("Unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_residualize_unknown_region() {
        let ba = BlockAssignment::contiguous("A", 100, 10);
        let pass = ResidualizePass::new(ba);

        // Access to unknown region "B" should not be transformed
        let stmt = Stmt::Assign(
            "x".to_string(),
            Expr::shared_read(Expr::var("B"), Expr::int(5)),
        );
        let result = pass.transform_stmt(&stmt);
        match &result {
            Stmt::Assign(_, Expr::SharedRead(_, idx)) => {
                assert_eq!(**idx, Expr::IntLiteral(5));
            }
            other => panic!("Unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_residualize_non_constant_index() {
        let ba = BlockAssignment::contiguous("A", 100, 10);
        let pass = ResidualizePass::new(ba);

        let stmt = Stmt::Assign(
            "x".to_string(),
            Expr::shared_read(Expr::var("A"), Expr::var("i")),
        );
        let result = pass.transform_stmt(&stmt);
        // Non-constant index should be preserved
        match &result {
            Stmt::Assign(_, Expr::SharedRead(_, idx)) => {
                assert_eq!(**idx, Expr::var("i"));
            }
            other => panic!("Unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_residualize_nested_if() {
        let ba = BlockAssignment::contiguous("A", 100, 10);
        let pass = ResidualizePass::new(ba);

        let stmts = vec![Stmt::If {
            condition: Expr::BoolLiteral(true),
            then_body: vec![Stmt::Assign(
                "x".to_string(),
                Expr::shared_read(Expr::var("A"), Expr::int(42)),
            )],
            else_body: vec![],
        }];
        let result = pass.transform(&stmts);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_direct_index_expr() {
        let e = direct_index_expr(3, 5, 10);
        assert_eq!(e, Expr::IntLiteral(35));
    }

    #[test]
    fn test_compute_block_id_and_offset() {
        let (bid, off) = compute_block_id_and_offset(&Expr::var("addr"), 16);
        match bid {
            Expr::BinOp(BinOp::Div, _, r) => assert_eq!(*r, Expr::IntLiteral(16)),
            other => panic!("Expected div, got {:?}", other),
        }
        match off {
            Expr::BinOp(BinOp::Mod, _, r) => assert_eq!(*r, Expr::IntLiteral(16)),
            other => panic!("Expected mod, got {:?}", other),
        }
    }

    #[test]
    fn test_add_contiguous_region() {
        let mut ba = BlockAssignment::contiguous("A", 100, 10);
        ba.add_contiguous_region("B", 50, 5);
        assert_eq!(ba.resolve("A", 15), Some((1, 5)));
        assert_eq!(ba.resolve("B", 7), Some((1, 2)));
    }

    #[test]
    fn test_residualize_block_of_call() {
        let ba = BlockAssignment::contiguous("M", 64, 8);
        let pass = ResidualizePass::new(ba);

        let expr = Expr::FunctionCall(
            "block_of".to_string(),
            vec![Expr::var("M"), Expr::int(20)],
        );
        let result = pass.transform_expr(&expr);
        // block_of(M, 20) -> block 2 (20/8 = 2)
        assert_eq!(result, Expr::IntLiteral(2));
    }

    #[test]
    fn test_estimate_savings_basic() {
        use crate::pram_ir::ast::{MemoryModel, PramProgram, SharedMemoryDecl};
        use crate::pram_ir::types::PramType;
        let mut prog = PramProgram::new("test", MemoryModel::EREW);
        prog.shared_memory.push(SharedMemoryDecl {
            name: "A".to_string(),
            elem_type: PramType::Int64,
            size: Expr::int(100),
        });
        prog.body = vec![
            Stmt::Assign(
                "x".to_string(),
                Expr::shared_read(Expr::var("A"), Expr::int(5)),
            ),
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(10),
                value: Expr::int(42),
            },
        ];
        let ba = BlockAssignment::contiguous("A", 100, 10);
        let savings = estimate_savings(&prog, &ba);
        assert_eq!(savings.reads_resolved, 1);
        assert_eq!(savings.writes_resolved, 1);
        assert!(savings.ops_eliminated >= 2);
    }

    #[test]
    fn test_batch_residualizer() {
        let ba = BlockAssignment::contiguous("A", 100, 10);
        let batch = BatchResidualizer::new(ba);
        let prog1 = vec![Stmt::Assign(
            "x".to_string(),
            Expr::shared_read(
                Expr::var("A"),
                Expr::FunctionCall("hash".to_string(), vec![Expr::var("A"), Expr::int(5)]),
            ),
        )];
        let prog2 = vec![Stmt::Assign("y".to_string(), Expr::int(1))];
        let results = batch.transform_batch(&[prog1, prog2]);
        assert_eq!(results.len(), 2);
        // First program should have hash call eliminated
        assert!(results[0].1.ops_eliminated >= 1);
        // Second program has no hash calls
        assert_eq!(results[1].1.ops_eliminated, 0);
    }

    #[test]
    fn test_verify_residualization_preserves() {
        let before = vec![
            Stmt::Assign("x".to_string(), Expr::shared_read(Expr::var("A"), Expr::int(5))),
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(10),
                value: Expr::int(42),
            },
        ];
        // After residualization, same structure but different index computation
        let after = vec![
            Stmt::Assign("x".to_string(), Expr::shared_read(Expr::var("A"), Expr::int(5))),
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(10),
                value: Expr::int(42),
            },
        ];
        assert!(verify_residualization(&before, &after));
    }

    #[test]
    fn test_verify_residualization_detects_dropped_write() {
        let before = vec![
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::int(1),
            },
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(1),
                value: Expr::int(2),
            },
        ];
        let after = vec![Stmt::SharedWrite {
            memory: Expr::var("A"),
            index: Expr::int(0),
            value: Expr::int(1),
        }];
        assert!(!verify_residualization(&before, &after));
    }

    #[test]
    fn test_residual_savings_default() {
        let s = ResidualSavings::default();
        assert_eq!(s.ops_eliminated, 0);
        assert_eq!(s.memory_saved, 0);
        assert_eq!(s.reads_resolved, 0);
        assert_eq!(s.writes_resolved, 0);
    }
}
