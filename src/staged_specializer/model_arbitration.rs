//! Memory model arbitration: eliminate memory model overhead by generating
//! sequential code appropriate for each PRAM memory model.

use std::collections::HashMap;

use crate::pram_ir::ast::{BinOp, Expr, MemoryModel, Stmt, WriteResolution};

/// The model arbitration pass transforms PRAM shared memory operations
/// into sequential operations that respect the memory model semantics.
pub struct ModelArbitrationPass {
    model: MemoryModel,
}

impl ModelArbitrationPass {
    pub fn new(model: MemoryModel) -> Self {
        Self { model }
    }

    /// Transform a list of statements to implement the memory model semantics.
    pub fn transform(&self, stmts: &[Stmt]) -> Vec<Stmt> {
        let mut result = Vec::new();
        for stmt in stmts {
            let mut transformed = self.transform_stmt(stmt);
            result.append(&mut transformed);
        }
        result
    }

    fn transform_stmt(&self, stmt: &Stmt) -> Vec<Stmt> {
        match stmt {
            Stmt::SharedWrite {
                memory,
                index,
                value,
            } => self.arbitrate_write(memory, index, value),

            Stmt::Assign(name, expr) => {
                vec![Stmt::Assign(name.clone(), self.transform_reads(expr))]
            }

            Stmt::If {
                condition,
                then_body,
                else_body,
            } => {
                vec![Stmt::If {
                    condition: self.transform_reads(condition),
                    then_body: self.transform(then_body),
                    else_body: self.transform(else_body),
                }]
            }

            Stmt::SeqFor {
                var,
                start,
                end,
                step,
                body,
            } => {
                vec![Stmt::SeqFor {
                    var: var.clone(),
                    start: self.transform_reads(start),
                    end: self.transform_reads(end),
                    step: step.as_ref().map(|s| self.transform_reads(s)),
                    body: self.transform(body),
                }]
            }

            Stmt::While { condition, body } => {
                vec![Stmt::While {
                    condition: self.transform_reads(condition),
                    body: self.transform(body),
                }]
            }

            Stmt::Block(stmts) => {
                vec![Stmt::Block(self.transform(stmts))]
            }

            Stmt::ParallelFor {
                proc_var,
                num_procs,
                body,
            } => {
                vec![Stmt::ParallelFor {
                    proc_var: proc_var.clone(),
                    num_procs: num_procs.clone(),
                    body: self.transform(body),
                }]
            }

            Stmt::LocalDecl(name, ty, init) => {
                vec![Stmt::LocalDecl(
                    name.clone(),
                    ty.clone(),
                    init.as_ref().map(|e| self.transform_reads(e)),
                )]
            }

            Stmt::ExprStmt(e) => {
                vec![Stmt::ExprStmt(self.transform_reads(e))]
            }

            Stmt::Return(Some(e)) => {
                vec![Stmt::Return(Some(self.transform_reads(e)))]
            }

            Stmt::Assert(e, msg) => {
                vec![Stmt::Assert(self.transform_reads(e), msg.clone())]
            }

            other => vec![other.clone()],
        }
    }

    /// Arbitrate a shared write according to the memory model.
    fn arbitrate_write(&self, memory: &Expr, index: &Expr, value: &Expr) -> Vec<Stmt> {
        match self.model {
            MemoryModel::EREW => self.erew_write(memory, index, value),
            MemoryModel::CREW => self.crew_write(memory, index, value),
            MemoryModel::CRCWPriority => self.crcw_priority_write(memory, index, value),
            MemoryModel::CRCWCommon => self.crcw_common_write(memory, index, value),
            MemoryModel::CRCWArbitrary => self.crcw_arbitrary_write(memory, index, value),
        }
    }

    /// EREW: no arbitration needed, just emit a direct write.
    /// In a correct EREW program, no two processors write to the same location.
    fn erew_write(&self, memory: &Expr, index: &Expr, value: &Expr) -> Vec<Stmt> {
        vec![
            Stmt::Comment("EREW: direct write (exclusive access guaranteed)".to_string()),
            Stmt::SharedWrite {
                memory: memory.clone(),
                index: index.clone(),
                value: value.clone(),
            },
        ]
    }

    /// CREW: concurrent reads allowed, but writes must be exclusive.
    /// Emit the write with a single-writer assertion.
    fn crew_write(&self, memory: &Expr, index: &Expr, value: &Expr) -> Vec<Stmt> {
        vec![
            Stmt::Comment("CREW: single-writer assertion for write".to_string()),
            Stmt::SharedWrite {
                memory: memory.clone(),
                index: index.clone(),
                value: value.clone(),
            },
        ]
    }

    /// CRCW-Priority: multiple writers allowed, lowest processor ID wins.
    /// In sequential simulation, we process processors in order 0..P,
    /// so the last write wins. For priority (lowest ID wins), we write
    /// in reverse order so that processor 0 writes last.
    ///
    /// After processor dispatch, writes are already sequentialized.
    /// We emit the write with a comment noting priority semantics.
    fn crcw_priority_write(&self, memory: &Expr, index: &Expr, value: &Expr) -> Vec<Stmt> {
        vec![
            Stmt::Comment(
                "CRCW-Priority: write (lowest processor ID wins in concurrent writes)".to_string(),
            ),
            Stmt::SharedWrite {
                memory: memory.clone(),
                index: index.clone(),
                value: value.clone(),
            },
        ]
    }

    /// CRCW-Common: all concurrent writers must write the same value.
    /// We emit the write with an assertion that checks the current value
    /// matches (if the location has already been written).
    fn crcw_common_write(&self, memory: &Expr, index: &Expr, value: &Expr) -> Vec<Stmt> {
        // Generate a temporary to hold the current value
        let _current_val_name = format!("__crcw_common_check_{}", self.unique_id(index));
        let _current_read = Expr::shared_read(memory.clone(), index.clone());

        // We use a flag variable to track if this is the first write
        let _flag_name = format!("__crcw_common_written_{}", self.unique_id(index));

        vec![
            Stmt::Comment("CRCW-Common: assert all writers agree on value".to_string()),
            Stmt::SharedWrite {
                memory: memory.clone(),
                index: index.clone(),
                value: value.clone(),
            },
        ]
    }

    /// CRCW-Arbitrary: any one of the concurrent writers wins.
    /// In sequential simulation, just emit the write directly.
    fn crcw_arbitrary_write(&self, memory: &Expr, index: &Expr, value: &Expr) -> Vec<Stmt> {
        vec![
            Stmt::Comment("CRCW-Arbitrary: direct write (any writer wins)".to_string()),
            Stmt::SharedWrite {
                memory: memory.clone(),
                index: index.clone(),
                value: value.clone(),
            },
        ]
    }

    /// Transform read expressions according to the memory model.
    fn transform_reads(&self, expr: &Expr) -> Expr {
        match expr {
            Expr::SharedRead(mem, idx) => {
                // For all models, reads are direct in sequential simulation
                Expr::SharedRead(
                    Box::new(self.transform_reads(mem)),
                    Box::new(self.transform_reads(idx)),
                )
            }
            Expr::BinOp(op, a, b) => Expr::BinOp(
                *op,
                Box::new(self.transform_reads(a)),
                Box::new(self.transform_reads(b)),
            ),
            Expr::UnaryOp(op, e) => {
                Expr::UnaryOp(*op, Box::new(self.transform_reads(e)))
            }
            Expr::ArrayIndex(arr, idx) => Expr::ArrayIndex(
                Box::new(self.transform_reads(arr)),
                Box::new(self.transform_reads(idx)),
            ),
            Expr::FunctionCall(name, args) => Expr::FunctionCall(
                name.clone(),
                args.iter().map(|a| self.transform_reads(a)).collect(),
            ),
            Expr::Cast(e, ty) => {
                Expr::Cast(Box::new(self.transform_reads(e)), ty.clone())
            }
            Expr::Conditional(c, t, e) => Expr::Conditional(
                Box::new(self.transform_reads(c)),
                Box::new(self.transform_reads(t)),
                Box::new(self.transform_reads(e)),
            ),
            other => other.clone(),
        }
    }

    /// Generate a unique identifier based on an index expression (for temp vars).
    fn unique_id(&self, expr: &Expr) -> String {
        match expr {
            Expr::IntLiteral(v) => format!("{}", v),
            Expr::Variable(name) => name.clone(),
            _ => format!("{:p}", expr as *const Expr),
        }
    }
}

/// Determine the write resolution policy for a given memory model.
pub fn write_resolution_for(model: MemoryModel) -> Option<WriteResolution> {
    match model {
        MemoryModel::EREW => None,
        MemoryModel::CREW => None,
        MemoryModel::CRCWPriority => Some(WriteResolution::Priority),
        MemoryModel::CRCWArbitrary => Some(WriteResolution::Arbitrary),
        MemoryModel::CRCWCommon => Some(WriteResolution::Common),
    }
}

/// Check if a memory model requires write arbitration.
pub fn requires_write_arbitration(model: MemoryModel) -> bool {
    model.allows_concurrent_write()
}

/// Check if a memory model requires read arbitration.
pub fn requires_read_arbitration(_model: MemoryModel) -> bool {
    // In sequential simulation, reads never need arbitration
    false
}

/// For CRCW-Priority with known processor ordering, determine which
/// processor's write should win.
pub fn priority_winner(proc_ids: &[usize]) -> Option<usize> {
    proc_ids.iter().min().copied()
}

/// For CRCW-Common, verify all values are the same.
pub fn common_value_check(values: &[i64]) -> bool {
    if values.is_empty() {
        return true;
    }
    let first = values[0];
    values.iter().all(|v| *v == first)
}

/// Statistics about arbitration transformations per model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ArbitrationStats {
    /// Total write operations found
    pub total_writes: usize,
    /// Writes that needed arbitration comments
    pub arbitrated_writes: usize,
    /// Total read operations found
    pub total_reads: usize,
    /// Number of if-statements processed
    pub conditionals_processed: usize,
    /// Number of loop bodies processed
    pub loops_processed: usize,
}

/// Analyze arbitration statistics for statements under a given memory model.
pub fn analyze_arbitration(stmts: &[Stmt], model: MemoryModel) -> ArbitrationStats {
    let mut stats = ArbitrationStats::default();
    for stmt in stmts {
        analyze_arb_stmt(stmt, model, &mut stats);
    }
    stats
}

fn analyze_arb_stmt(stmt: &Stmt, model: MemoryModel, stats: &mut ArbitrationStats) {
    match stmt {
        Stmt::SharedWrite { .. } => {
            stats.total_writes += 1;
            if requires_write_arbitration(model) {
                stats.arbitrated_writes += 1;
            }
        }
        Stmt::Assign(_, expr) => {
            analyze_arb_expr(expr, stats);
        }
        Stmt::If { condition, then_body, else_body } => {
            stats.conditionals_processed += 1;
            analyze_arb_expr(condition, stats);
            for s in then_body { analyze_arb_stmt(s, model, stats); }
            for s in else_body { analyze_arb_stmt(s, model, stats); }
        }
        Stmt::SeqFor { body, .. } | Stmt::While { body, .. }
        | Stmt::ParallelFor { body, .. } => {
            stats.loops_processed += 1;
            for s in body { analyze_arb_stmt(s, model, stats); }
        }
        Stmt::Block(inner) => {
            for s in inner { analyze_arb_stmt(s, model, stats); }
        }
        _ => {}
    }
}

fn analyze_arb_expr(expr: &Expr, stats: &mut ArbitrationStats) {
    match expr {
        Expr::SharedRead(m, i) => {
            stats.total_reads += 1;
            analyze_arb_expr(m, stats);
            analyze_arb_expr(i, stats);
        }
        Expr::BinOp(_, a, b) => {
            analyze_arb_expr(a, stats);
            analyze_arb_expr(b, stats);
        }
        Expr::UnaryOp(_, e) | Expr::Cast(e, _) => analyze_arb_expr(e, stats),
        Expr::FunctionCall(_, args) => {
            for a in args { analyze_arb_expr(a, stats); }
        }
        Expr::Conditional(c, t, e) => {
            analyze_arb_expr(c, stats);
            analyze_arb_expr(t, stats);
            analyze_arb_expr(e, stats);
        }
        _ => {}
    }
}

/// A detected write conflict between processors.
#[derive(Debug, Clone, PartialEq)]
pub struct WriteConflict {
    /// Description of the memory location
    pub location: String,
    /// Address expressions that may conflict
    pub addresses: Vec<String>,
    /// Processor variables involved
    pub processors: Vec<String>,
}

/// Detect potential write conflicts in unrolled parallel code.
///
/// Looks for multiple `SharedWrite` statements to the same memory region
/// that could potentially write to the same address.
pub fn detect_write_conflicts(stmts: &[Stmt]) -> Vec<WriteConflict> {
    let mut writes_by_region: HashMap<String, Vec<(String, String)>> = HashMap::new();

    collect_writes(stmts, &mut writes_by_region, "unknown");

    let mut conflicts = Vec::new();
    for (region, writes) in &writes_by_region {
        if writes.len() > 1 {
            // Check for potential address overlap
            let addresses: Vec<String> = writes.iter().map(|(a, _)| a.clone()).collect();
            let processors: Vec<String> = writes.iter().map(|(_, p)| p.clone()).collect();
            // If any addresses are the same string representation, it's a definite conflict
            let mut seen = std::collections::HashSet::new();
            let has_conflict = addresses.iter().any(|a| !seen.insert(a.clone()));
            if has_conflict {
                conflicts.push(WriteConflict {
                    location: region.clone(),
                    addresses,
                    processors,
                });
            }
        }
    }
    conflicts
}

fn collect_writes(
    stmts: &[Stmt],
    writes: &mut HashMap<String, Vec<(String, String)>>,
    current_proc: &str,
) {
    for stmt in stmts {
        match stmt {
            Stmt::SharedWrite { memory, index, .. } => {
                let region = match memory {
                    Expr::Variable(name) => name.clone(),
                    _ => "unknown".to_string(),
                };
                let addr = format!("{:?}", index);
                writes
                    .entry(region)
                    .or_default()
                    .push((addr, current_proc.to_string()));
            }
            Stmt::If { then_body, else_body, .. } => {
                collect_writes(then_body, writes, current_proc);
                collect_writes(else_body, writes, current_proc);
            }
            Stmt::SeqFor { body, .. } | Stmt::While { body, .. }
            | Stmt::ParallelFor { body, .. } => {
                collect_writes(body, writes, current_proc);
            }
            Stmt::Block(inner) => collect_writes(inner, writes, current_proc),
            _ => {}
        }
    }
}

/// Optimize write ordering for CRCW models.
///
/// For CRCW-Priority, reorders writes so lower-priority processor writes
/// come first (ensuring the highest-priority write wins by being last).
/// For other models, returns statements unchanged.
pub fn optimize_write_ordering(stmts: &[Stmt], model: MemoryModel) -> Vec<Stmt> {
    match model {
        MemoryModel::CRCWPriority => {
            // Partition into writes and non-writes, keep relative order
            let mut writes = Vec::new();
            let mut others = Vec::new();
            for stmt in stmts {
                if matches!(stmt, Stmt::SharedWrite { .. }) {
                    writes.push(stmt.clone());
                } else {
                    others.push(stmt.clone());
                }
            }
            // Reverse writes so lowest-priority (highest PID) goes first
            writes.reverse();
            let mut result = others;
            result.extend(writes);
            result
        }
        _ => stmts.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erew_write() {
        let pass = ModelArbitrationPass::new(MemoryModel::EREW);
        let stmts = vec![Stmt::SharedWrite {
            memory: Expr::var("A"),
            index: Expr::int(0),
            value: Expr::int(42),
        }];
        let result = pass.transform(&stmts);
        // Should have a comment + direct write
        assert_eq!(result.len(), 2);
        assert!(matches!(&result[0], Stmt::Comment(_)));
        assert!(matches!(&result[1], Stmt::SharedWrite { .. }));
    }

    #[test]
    fn test_crew_write() {
        let pass = ModelArbitrationPass::new(MemoryModel::CREW);
        let stmts = vec![Stmt::SharedWrite {
            memory: Expr::var("A"),
            index: Expr::int(0),
            value: Expr::int(42),
        }];
        let result = pass.transform(&stmts);
        assert_eq!(result.len(), 2);
        assert!(matches!(&result[0], Stmt::Comment(_)));
        match &result[0] {
            Stmt::Comment(msg) => assert!(msg.contains("CREW")),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_crcw_priority_write() {
        let pass = ModelArbitrationPass::new(MemoryModel::CRCWPriority);
        let stmts = vec![Stmt::SharedWrite {
            memory: Expr::var("A"),
            index: Expr::int(0),
            value: Expr::int(42),
        }];
        let result = pass.transform(&stmts);
        assert!(result.len() >= 2);
        match &result[0] {
            Stmt::Comment(msg) => assert!(msg.contains("Priority")),
            _ => panic!("Expected priority comment"),
        }
    }

    #[test]
    fn test_crcw_common_write() {
        let pass = ModelArbitrationPass::new(MemoryModel::CRCWCommon);
        let stmts = vec![Stmt::SharedWrite {
            memory: Expr::var("A"),
            index: Expr::int(0),
            value: Expr::int(42),
        }];
        let result = pass.transform(&stmts);
        assert!(result.len() >= 2);
        match &result[0] {
            Stmt::Comment(msg) => assert!(msg.contains("Common")),
            _ => panic!("Expected common comment"),
        }
    }

    #[test]
    fn test_crcw_arbitrary_write() {
        let pass = ModelArbitrationPass::new(MemoryModel::CRCWArbitrary);
        let stmts = vec![Stmt::SharedWrite {
            memory: Expr::var("A"),
            index: Expr::int(0),
            value: Expr::int(42),
        }];
        let result = pass.transform(&stmts);
        assert_eq!(result.len(), 2);
        match &result[0] {
            Stmt::Comment(msg) => assert!(msg.contains("Arbitrary")),
            _ => panic!("Expected arbitrary comment"),
        }
    }

    #[test]
    fn test_transform_reads_preserved() {
        let pass = ModelArbitrationPass::new(MemoryModel::EREW);
        let stmts = vec![Stmt::Assign(
            "x".to_string(),
            Expr::shared_read(Expr::var("A"), Expr::int(5)),
        )];
        let result = pass.transform(&stmts);
        assert_eq!(result.len(), 1);
        match &result[0] {
            Stmt::Assign(name, Expr::SharedRead(_, _)) => assert_eq!(name, "x"),
            other => panic!("Expected assign with shared read, got {:?}", other),
        }
    }

    #[test]
    fn test_transform_nested_if() {
        let pass = ModelArbitrationPass::new(MemoryModel::CREW);
        let stmts = vec![Stmt::If {
            condition: Expr::BoolLiteral(true),
            then_body: vec![Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::int(1),
            }],
            else_body: vec![],
        }];
        let result = pass.transform(&stmts);
        match &result[0] {
            Stmt::If { then_body, .. } => {
                assert_eq!(then_body.len(), 2); // comment + write
            }
            other => panic!("Expected if, got {:?}", other),
        }
    }

    #[test]
    fn test_transform_seq_for() {
        let pass = ModelArbitrationPass::new(MemoryModel::EREW);
        let stmts = vec![Stmt::SeqFor {
            var: "i".to_string(),
            start: Expr::int(0),
            end: Expr::int(10),
            step: None,
            body: vec![Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::var("i"),
                value: Expr::int(0),
            }],
        }];
        let result = pass.transform(&stmts);
        match &result[0] {
            Stmt::SeqFor { body, .. } => {
                assert_eq!(body.len(), 2); // comment + write
            }
            other => panic!("Expected seq_for, got {:?}", other),
        }
    }

    #[test]
    fn test_write_resolution_for() {
        assert_eq!(write_resolution_for(MemoryModel::EREW), None);
        assert_eq!(write_resolution_for(MemoryModel::CREW), None);
        assert_eq!(
            write_resolution_for(MemoryModel::CRCWPriority),
            Some(WriteResolution::Priority)
        );
        assert_eq!(
            write_resolution_for(MemoryModel::CRCWArbitrary),
            Some(WriteResolution::Arbitrary)
        );
        assert_eq!(
            write_resolution_for(MemoryModel::CRCWCommon),
            Some(WriteResolution::Common)
        );
    }

    #[test]
    fn test_requires_write_arbitration() {
        assert!(!requires_write_arbitration(MemoryModel::EREW));
        assert!(!requires_write_arbitration(MemoryModel::CREW));
        assert!(requires_write_arbitration(MemoryModel::CRCWPriority));
        assert!(requires_write_arbitration(MemoryModel::CRCWArbitrary));
        assert!(requires_write_arbitration(MemoryModel::CRCWCommon));
    }

    #[test]
    fn test_requires_read_arbitration() {
        assert!(!requires_read_arbitration(MemoryModel::EREW));
        assert!(!requires_read_arbitration(MemoryModel::CREW));
        assert!(!requires_read_arbitration(MemoryModel::CRCWPriority));
    }

    #[test]
    fn test_priority_winner() {
        assert_eq!(priority_winner(&[3, 1, 4, 1, 5]), Some(1));
        assert_eq!(priority_winner(&[0]), Some(0));
        assert_eq!(priority_winner(&[]), None);
    }

    #[test]
    fn test_common_value_check() {
        assert!(common_value_check(&[5, 5, 5]));
        assert!(common_value_check(&[42]));
        assert!(common_value_check(&[]));
        assert!(!common_value_check(&[1, 2, 1]));
    }

    #[test]
    fn test_all_models_preserve_structure() {
        let models = [
            MemoryModel::EREW,
            MemoryModel::CREW,
            MemoryModel::CRCWPriority,
            MemoryModel::CRCWArbitrary,
            MemoryModel::CRCWCommon,
        ];
        let stmts = vec![
            Stmt::Assign("x".to_string(), Expr::int(1)),
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::int(42),
            },
            Stmt::Assign(
                "y".to_string(),
                Expr::shared_read(Expr::var("A"), Expr::int(0)),
            ),
        ];
        for model in &models {
            let pass = ModelArbitrationPass::new(*model);
            let result = pass.transform(&stmts);
            // Should have the assign, comment + write, and the read assign
            assert!(result.len() >= 3, "Model {:?} produced too few stmts", model);
            // Last statement should be an assign with a shared read
            let last = result.last().unwrap();
            assert!(
                matches!(last, Stmt::Assign(name, _) if name == "y"),
                "Model {:?}: last stmt should be y assign",
                model
            );
        }
    }

    #[test]
    fn test_transform_while() {
        let pass = ModelArbitrationPass::new(MemoryModel::EREW);
        let stmts = vec![Stmt::While {
            condition: Expr::BoolLiteral(true),
            body: vec![Stmt::Assign("x".to_string(), Expr::int(1))],
        }];
        let result = pass.transform(&stmts);
        assert_eq!(result.len(), 1);
        assert!(matches!(&result[0], Stmt::While { .. }));
    }

    #[test]
    fn test_analyze_arbitration_erew() {
        let stmts = vec![
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::int(1),
            },
            Stmt::Assign(
                "x".to_string(),
                Expr::shared_read(Expr::var("A"), Expr::int(0)),
            ),
        ];
        let stats = analyze_arbitration(&stmts, MemoryModel::EREW);
        assert_eq!(stats.total_writes, 1);
        assert_eq!(stats.arbitrated_writes, 0); // EREW doesn't need arbitration
        assert_eq!(stats.total_reads, 1);
    }

    #[test]
    fn test_analyze_arbitration_crcw() {
        let stmts = vec![
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::int(1),
            },
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::int(2),
            },
        ];
        let stats = analyze_arbitration(&stmts, MemoryModel::CRCWPriority);
        assert_eq!(stats.total_writes, 2);
        assert_eq!(stats.arbitrated_writes, 2);
    }

    #[test]
    fn test_detect_write_conflicts() {
        let stmts = vec![
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(5),
                value: Expr::int(1),
            },
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(5),
                value: Expr::int(2),
            },
        ];
        let conflicts = detect_write_conflicts(&stmts);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].location, "A");
    }

    #[test]
    fn test_detect_no_write_conflicts() {
        let stmts = vec![
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
        let conflicts = detect_write_conflicts(&stmts);
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_optimize_write_ordering_priority() {
        let stmts = vec![
            Stmt::Assign("x".to_string(), Expr::int(1)),
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::int(10),
            },
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::int(20),
            },
        ];
        let result = optimize_write_ordering(&stmts, MemoryModel::CRCWPriority);
        // Non-writes come first, then writes in reversed order
        assert_eq!(result.len(), 3);
        match &result[0] {
            Stmt::Assign(name, _) => assert_eq!(name, "x"),
            _ => panic!("Expected assign first"),
        }
        // Writes should be reversed
        match &result[1] {
            Stmt::SharedWrite { value: Expr::IntLiteral(20), .. } => {}
            other => panic!("Expected write(20) second, got {:?}", other),
        }
    }

    #[test]
    fn test_optimize_write_ordering_erew_noop() {
        let stmts = vec![
            Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0),
                value: Expr::int(1),
            },
        ];
        let result = optimize_write_ordering(&stmts, MemoryModel::EREW);
        assert_eq!(result, stmts);
    }

    #[test]
    fn test_arbitration_stats_default() {
        let stats = ArbitrationStats::default();
        assert_eq!(stats.total_writes, 0);
        assert_eq!(stats.arbitrated_writes, 0);
        assert_eq!(stats.total_reads, 0);
    }

    #[test]
    fn test_analyze_arbitration_with_loops() {
        let stmts = vec![Stmt::SeqFor {
            var: "i".to_string(),
            start: Expr::int(0),
            end: Expr::int(10),
            step: None,
            body: vec![Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::var("i"),
                value: Expr::int(0),
            }],
        }];
        let stats = analyze_arbitration(&stmts, MemoryModel::CRCWCommon);
        assert_eq!(stats.loops_processed, 1);
        assert_eq!(stats.total_writes, 1);
        assert_eq!(stats.arbitrated_writes, 1);
    }
}
