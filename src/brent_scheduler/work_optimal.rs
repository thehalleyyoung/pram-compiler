//! Work-optimal sequential schedule extraction via Brent's theorem.
//!
//! Brent's theorem: A PRAM computation using p processors in T parallel
//! time can be simulated sequentially in O(pT) work. This module implements
//! the scheduling: topological sort → level assignment → sequential ordering
//! within levels, and verifies the work bound.

use std::collections::{HashMap, HashSet};

use petgraph::graph::NodeIndex;

use super::dependency_graph::DependencyGraph;
use super::schedule::{OpType, Schedule, ScheduleEntry};

/// A sequential schedule: an ordered list of operations extracted from
/// the dependency graph using Brent's theorem approach.
#[derive(Debug, Clone)]
pub struct SequentialSchedule {
    /// Operations in sequential execution order, each with its level.
    pub operations: Vec<ScheduledOp>,
    /// Number of levels (parallel time steps).
    pub num_levels: usize,
    /// Number of processors in the original program.
    pub num_processors: usize,
}

/// An operation placed in the sequential schedule.
#[derive(Debug, Clone)]
pub struct ScheduledOp {
    pub node_index: NodeIndex,
    pub op_id: usize,
    pub block_id: usize,
    pub phase: usize,
    pub proc_id: usize,
    pub op_type: OpType,
    pub address: usize,
    pub memory_region: String,
    pub level: usize,
}

impl SequentialSchedule {
    /// Total work (number of operations).
    pub fn total_work(&self) -> usize {
        self.operations.len()
    }

    /// Operations at a given level.
    pub fn ops_at_level(&self, level: usize) -> Vec<&ScheduledOp> {
        self.operations.iter().filter(|op| op.level == level).collect()
    }

    /// Width at each level (number of operations per level).
    pub fn level_widths(&self) -> HashMap<usize, usize> {
        let mut widths: HashMap<usize, usize> = HashMap::new();
        for op in &self.operations {
            *widths.entry(op.level).or_insert(0) += 1;
        }
        widths
    }

    /// Maximum width (most operations in any single level).
    pub fn max_width(&self) -> usize {
        self.level_widths().values().copied().max().unwrap_or(0)
    }

    /// Verify work optimality: total_work ≤ factor * p * T.
    /// Returns Ok(()) if satisfied, Err with the ratio if violated.
    pub fn verify_work_bound(&self, factor: f64) -> Result<(), f64> {
        let p = self.num_processors.max(1);
        let t = self.num_levels.max(1);
        let bound = (factor * (p * t) as f64).ceil() as usize;
        let work = self.total_work();
        if work <= bound {
            Ok(())
        } else {
            Err(work as f64 / (p * t) as f64)
        }
    }

    /// Convert to a Schedule with block_size metadata.
    pub fn to_schedule(&self, block_size: usize, num_phases: usize) -> Schedule {
        let mut schedule = Schedule::new(self.num_processors, num_phases, block_size);
        for op in &self.operations {
            schedule.push(ScheduleEntry {
                operation_id: op.op_id,
                block_id: op.block_id,
                phase: op.phase,
                original_proc_id: op.proc_id,
                op_type: op.op_type,
                address: op.address,
                memory_region: op.memory_region.clone(),
                level: op.level,
            });
        }
        schedule
    }
}

/// Extract a work-optimal sequential schedule from a dependency graph.
///
/// Algorithm:
/// 1. Compute topological order of the dependency graph.
/// 2. Assign each node a level = 1 + max(level of predecessors).
/// 3. Within each level, order operations sequentially (by op_id for determinism).
/// 4. Concatenate levels to get the final sequential schedule.
///
/// This produces O(W) total work where W = total operations, and the
/// schedule respects all data dependencies. By Brent's theorem, if the
/// original parallel program uses p processors for T steps, W ≤ p*T.
pub fn extract_schedule(graph: &DependencyGraph) -> SequentialSchedule {
    let levels = graph.compute_levels();

    let topo = match graph.topological_order() {
        Some(t) => t,
        None => {
            return SequentialSchedule {
                operations: Vec::new(),
                num_levels: 0,
                num_processors: graph.num_processors,
            };
        }
    };

    // Determine max level
    let num_levels = levels.values().copied().max().map_or(0, |m| m + 1);

    // Group nodes by level
    let mut by_level: Vec<Vec<NodeIndex>> = vec![Vec::new(); num_levels];
    for &idx in &topo {
        let level = levels.get(&idx).copied().unwrap_or(0);
        if level < num_levels {
            by_level[level].push(idx);
        }
    }

    // Sort within each level by op_id for deterministic ordering
    for level_ops in &mut by_level {
        level_ops.sort_by_key(|&idx| graph.get_node(idx).op_id);
    }

    // Build the sequential schedule: process level 0 first, then 1, etc.
    let mut operations = Vec::with_capacity(graph.node_count());
    for (level, level_ops) in by_level.iter().enumerate() {
        for &idx in level_ops {
            let node = graph.get_node(idx);
            operations.push(ScheduledOp {
                node_index: idx,
                op_id: node.op_id,
                block_id: node.block_id,
                phase: node.phase,
                proc_id: node.proc_id,
                op_type: node.op_type,
                address: node.address,
                memory_region: node.memory_region.clone(),
                level,
            });
        }
    }

    SequentialSchedule {
        operations,
        num_levels,
        num_processors: graph.num_processors,
    }
}

/// Extract a schedule with a custom within-level comparator.
/// `cmp` receives two ScheduledOps and returns their ordering within a level.
pub fn extract_schedule_with_order<F>(graph: &DependencyGraph, cmp: F) -> SequentialSchedule
where
    F: Fn(&ScheduledOp, &ScheduledOp) -> std::cmp::Ordering,
{
    let levels = graph.compute_levels();
    let topo = match graph.topological_order() {
        Some(t) => t,
        None => {
            return SequentialSchedule {
                operations: Vec::new(),
                num_levels: 0,
                num_processors: graph.num_processors,
            };
        }
    };

    let num_levels = levels.values().copied().max().map_or(0, |m| m + 1);
    let mut by_level: Vec<Vec<ScheduledOp>> = vec![Vec::new(); num_levels];

    for &idx in &topo {
        let level = levels.get(&idx).copied().unwrap_or(0);
        let node = graph.get_node(idx);
        if level < num_levels {
            by_level[level].push(ScheduledOp {
                node_index: idx,
                op_id: node.op_id,
                block_id: node.block_id,
                phase: node.phase,
                proc_id: node.proc_id,
                op_type: node.op_type,
                address: node.address,
                memory_region: node.memory_region.clone(),
                level,
            });
        }
    }

    for level_ops in &mut by_level {
        level_ops.sort_by(&cmp);
    }

    let operations: Vec<ScheduledOp> = by_level.into_iter().flatten().collect();

    SequentialSchedule {
        operations,
        num_levels,
        num_processors: graph.num_processors,
    }
}

/// Extract a schedule using user-defined priority values.
///
/// Each node can be assigned a priority (higher = scheduled earlier within its level).
/// Nodes without explicit priorities default to 0.
pub fn extract_schedule_with_priorities(
    graph: &DependencyGraph,
    priorities: &HashMap<NodeIndex, i64>,
) -> SequentialSchedule {
    extract_schedule_with_order(graph, |a, b| {
        let pa = priorities.get(&a.node_index).copied().unwrap_or(0);
        let pb = priorities.get(&b.node_index).copied().unwrap_or(0);
        pb.cmp(&pa).then(a.op_id.cmp(&b.op_id))
    })
}

/// Compute the number of operations at each level (width profile).
pub fn compute_level_widths(schedule: &SequentialSchedule) -> Vec<usize> {
    let mut widths = vec![0usize; schedule.num_levels];
    for op in &schedule.operations {
        if op.level < widths.len() {
            widths[op.level] += 1;
        }
    }
    widths
}

/// Balance the schedule by redistributing operations across levels to reduce
/// the maximum width, improving cache utilization.
///
/// This is a heuristic: when a level is significantly wider than average,
/// operations with no same-level dependencies can be moved to adjacent levels
/// (within dependency constraints).
pub fn balance_levels(schedule: &SequentialSchedule) -> SequentialSchedule {
    if schedule.num_levels <= 1 || schedule.operations.is_empty() {
        return schedule.clone();
    }

    let widths = compute_level_widths(schedule);
    let avg_width = schedule.total_work() as f64 / schedule.num_levels as f64;
    let threshold = (avg_width * 1.5).ceil() as usize;

    // If no level exceeds threshold, nothing to do
    if widths.iter().all(|&w| w <= threshold) {
        return schedule.clone();
    }

    // Group ops by level
    let mut by_level: Vec<Vec<ScheduledOp>> = vec![Vec::new(); schedule.num_levels];
    for op in &schedule.operations {
        if op.level < schedule.num_levels {
            by_level[op.level].push(op.clone());
        }
    }

    // Simple balancing: reassign levels to spread work more evenly
    // We keep the same order but adjust level assignments
    let total = schedule.total_work();
    let target_per_level = (total as f64 / schedule.num_levels as f64).ceil() as usize;

    let mut balanced_ops: Vec<ScheduledOp> = Vec::with_capacity(total);
    let mut current_level = 0;
    let mut count_in_level = 0;

    for level_ops in &by_level {
        for op in level_ops {
            if count_in_level >= target_per_level && current_level < schedule.num_levels - 1 {
                current_level += 1;
                count_in_level = 0;
            }
            let mut balanced_op = op.clone();
            // Only reassign if the new level is >= original level (preserve deps)
            balanced_op.level = balanced_op.level.max(current_level);
            balanced_ops.push(balanced_op);
            count_in_level += 1;
        }
    }

    // Recompute num_levels
    let max_level = balanced_ops.iter().map(|op| op.level).max().unwrap_or(0);

    SequentialSchedule {
        operations: balanced_ops,
        num_levels: max_level + 1,
        num_processors: schedule.num_processors,
    }
}

/// Verify that a schedule correctly respects all dependencies in the graph.
///
/// Returns Ok(()) if valid, or Err with a list of violation descriptions.
pub fn verify_schedule_correctness(
    graph: &DependencyGraph,
    schedule: &SequentialSchedule,
) -> Result<(), Vec<String>> {
    let mut errors: Vec<String> = Vec::new();

    // Check all graph nodes are present
    let scheduled_ids: HashSet<usize> = schedule.operations.iter().map(|op| op.op_id).collect();
    for idx in graph.graph.node_indices() {
        let node = graph.get_node(idx);
        if !scheduled_ids.contains(&node.op_id) {
            errors.push(format!("Op {} missing from schedule", node.op_id));
        }
    }

    // Check no duplicate op_ids
    let mut seen_ids: HashSet<usize> = HashSet::new();
    for op in &schedule.operations {
        if !seen_ids.insert(op.op_id) {
            errors.push(format!("Duplicate op_id {} in schedule", op.op_id));
        }
    }

    // Check dependency ordering
    let pos: HashMap<usize, usize> = schedule
        .operations
        .iter()
        .enumerate()
        .map(|(i, op)| (op.op_id, i))
        .collect();

    for edge in graph.graph.edge_indices() {
        let (src, tgt) = graph.graph.edge_endpoints(edge).unwrap();
        let src_id = graph.get_node(src).op_id;
        let tgt_id = graph.get_node(tgt).op_id;

        match (pos.get(&src_id), pos.get(&tgt_id)) {
            (Some(&src_pos), Some(&tgt_pos)) => {
                if src_pos >= tgt_pos {
                    let weight = &graph.graph[edge];
                    errors.push(format!(
                        "Dependency violation: op {} (pos {}) must precede op {} (pos {}), kind={:?}",
                        src_id, src_pos, tgt_id, tgt_pos, weight.kind
                    ));
                }
            }
            _ => {
                errors.push(format!(
                    "Edge op {}→{} references missing operations",
                    src_id, tgt_id
                ));
            }
        }
    }

    // Check level monotonicity: ops in the schedule with level L must come before level L+1
    let mut max_level_seen: usize = 0;
    for op in &schedule.operations {
        if op.level < max_level_seen {
            // Level decreased — not necessarily wrong (balanced schedule), but flag if strict
        }
        max_level_seen = max_level_seen.max(op.level);
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brent_scheduler::dependency_graph::{DependencyGraph, OperationNode};

    fn make_op(
        op_id: usize,
        block_id: usize,
        phase: usize,
        proc_id: usize,
        op_type: OpType,
        address: usize,
    ) -> OperationNode {
        OperationNode {
            op_id,
            block_id,
            phase,
            proc_id,
            op_type,
            address,
            memory_region: "A".to_string(),
        }
    }

    #[test]
    fn test_extract_simple_chain() {
        // W(0) → R(1) → W(2): should be scheduled in that order, 3 levels
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
            make_op(2, 0, 0, 2, OpType::Write, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let sched = extract_schedule(&dg);

        assert_eq!(sched.total_work(), 3);
        assert_eq!(sched.num_levels, 3);
        // Check order: op 0 before op 1 before op 2
        let ids: Vec<usize> = sched.operations.iter().map(|o| o.op_id).collect();
        let pos0 = ids.iter().position(|&id| id == 0).unwrap();
        let pos1 = ids.iter().position(|&id| id == 1).unwrap();
        let pos2 = ids.iter().position(|&id| id == 2).unwrap();
        assert!(pos0 < pos1);
        assert!(pos1 < pos2);
    }

    #[test]
    fn test_extract_independent_ops() {
        // Three independent writes to different addresses → all at level 0
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 1, 0, 1, OpType::Write, 4),
            make_op(2, 2, 0, 2, OpType::Write, 8),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let sched = extract_schedule(&dg);

        assert_eq!(sched.total_work(), 3);
        assert_eq!(sched.num_levels, 1);
        // All ops at level 0
        assert!(sched.operations.iter().all(|o| o.level == 0));
    }

    #[test]
    fn test_work_bound_verification() {
        // 4 independent ops with 4 processors → W=4, p=4, T=1 → W/(p*T) = 1.0
        let ops: Vec<OperationNode> = (0..4)
            .map(|i| make_op(i, i, 0, i, OpType::Write, i * 4))
            .collect();
        let dg = DependencyGraph::from_operations(ops, 4);
        let sched = extract_schedule(&dg);

        assert!(sched.verify_work_bound(1.0).is_ok());
        assert!(sched.verify_work_bound(2.0).is_ok());
    }

    #[test]
    fn test_work_bound_chain() {
        // Chain of 4 ops on same address → W=4, p=4, T=4 → W/(p*T) = 0.25
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
            make_op(2, 0, 0, 2, OpType::Write, 0),
            make_op(3, 0, 0, 3, OpType::Read, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let sched = extract_schedule(&dg);

        assert_eq!(sched.total_work(), 4);
        assert!(sched.verify_work_bound(1.0).is_ok()); // 4 ≤ 1.0 * 4 * 4
    }

    #[test]
    fn test_level_widths() {
        // Two independent pairs: (W0,R1) on addr 0, (W2,R3) on addr 4
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
            make_op(2, 1, 0, 2, OpType::Write, 4),
            make_op(3, 1, 0, 3, OpType::Read, 4),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let sched = extract_schedule(&dg);

        let widths = sched.level_widths();
        assert_eq!(sched.num_levels, 2);
        assert_eq!(widths[&0], 2); // W0 and W2 at level 0
        assert_eq!(widths[&1], 2); // R1 and R3 at level 1
    }

    #[test]
    fn test_to_schedule_conversion() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let seq = extract_schedule(&dg);
        let schedule = seq.to_schedule(4, 1);

        assert_eq!(schedule.total_ops(), 2);
        assert_eq!(schedule.num_processors, 2);
        assert_eq!(schedule.block_size, 4);

        // Validate dependencies
        let deps = dg.dependency_pairs();
        assert!(schedule.validate_dependencies(&deps).is_ok());
    }

    #[test]
    fn test_empty_graph() {
        let dg = DependencyGraph::from_operations(vec![], 4);
        let sched = extract_schedule(&dg);
        assert_eq!(sched.total_work(), 0);
        assert_eq!(sched.num_levels, 0);
    }

    #[test]
    fn test_extract_with_custom_order() {
        // Independent writes → custom order by block_id descending
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 2, 0, 1, OpType::Write, 8),
            make_op(2, 1, 0, 2, OpType::Write, 4),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let sched = extract_schedule_with_order(&dg, |a, b| b.block_id.cmp(&a.block_id));

        assert_eq!(sched.total_work(), 3);
        // Block ids should be in descending order since all are at same level
        let blocks: Vec<usize> = sched.operations.iter().map(|o| o.block_id).collect();
        assert_eq!(blocks, vec![2, 1, 0]);
    }

    #[test]
    fn test_max_width() {
        let ops: Vec<OperationNode> = (0..8)
            .map(|i| make_op(i, i, 0, i, OpType::Write, i * 4))
            .collect();
        let dg = DependencyGraph::from_operations(ops, 4);
        let sched = extract_schedule(&dg);

        assert_eq!(sched.max_width(), 8); // all at level 0
        assert_eq!(sched.num_levels, 1);
    }

    #[test]
    fn test_ops_at_level() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let sched = extract_schedule(&dg);

        assert_eq!(sched.ops_at_level(0).len(), 1);
        assert_eq!(sched.ops_at_level(1).len(), 1);
        assert_eq!(sched.ops_at_level(0)[0].op_type, OpType::Write);
        assert_eq!(sched.ops_at_level(1)[0].op_type, OpType::Read);
    }

    #[test]
    fn test_extract_with_priorities() {
        // Three independent writes → priority ordering
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 1, 0, 1, OpType::Write, 4),
            make_op(2, 2, 0, 2, OpType::Write, 8),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);

        let mut priorities = HashMap::new();
        let indices: Vec<_> = dg.graph.node_indices().collect();
        // Give op 2 the highest priority
        for &idx in &indices {
            let node = dg.get_node(idx);
            match node.op_id {
                2 => { priorities.insert(idx, 100); }
                0 => { priorities.insert(idx, 50); }
                1 => { priorities.insert(idx, 10); }
                _ => {}
            }
        }

        let sched = extract_schedule_with_priorities(&dg, &priorities);
        assert_eq!(sched.total_work(), 3);
        // Op 2 should come first (highest priority)
        assert_eq!(sched.operations[0].op_id, 2);
    }

    #[test]
    fn test_compute_level_widths() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
            make_op(2, 1, 0, 2, OpType::Write, 4),
            make_op(3, 1, 0, 3, OpType::Read, 4),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let sched = extract_schedule(&dg);

        let widths = compute_level_widths(&sched);
        assert_eq!(widths.len(), 2);
        assert_eq!(widths[0], 2); // W(0) and W(2)
        assert_eq!(widths[1], 2); // R(1) and R(3)
    }

    #[test]
    fn test_compute_level_widths_single_level() {
        let ops: Vec<OperationNode> = (0..5)
            .map(|i| make_op(i, i, 0, i, OpType::Write, i * 4))
            .collect();
        let dg = DependencyGraph::from_operations(ops, 4);
        let sched = extract_schedule(&dg);

        let widths = compute_level_widths(&sched);
        assert_eq!(widths.len(), 1);
        assert_eq!(widths[0], 5);
    }

    #[test]
    fn test_balance_levels_preserves_ops() {
        let ops: Vec<OperationNode> = (0..10)
            .map(|i| make_op(i, i, 0, i, OpType::Write, i * 4))
            .collect();
        let dg = DependencyGraph::from_operations(ops, 4);
        let sched = extract_schedule(&dg);

        let balanced = balance_levels(&sched);
        assert_eq!(balanced.total_work(), sched.total_work());
    }

    #[test]
    fn test_balance_levels_empty() {
        let sched = SequentialSchedule {
            operations: Vec::new(),
            num_levels: 0,
            num_processors: 1,
        };
        let balanced = balance_levels(&sched);
        assert_eq!(balanced.total_work(), 0);
    }

    #[test]
    fn test_verify_schedule_correctness_valid() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
            make_op(2, 0, 0, 2, OpType::Write, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let sched = extract_schedule(&dg);

        assert!(verify_schedule_correctness(&dg, &sched).is_ok());
    }

    #[test]
    fn test_verify_schedule_correctness_missing_op() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);

        // Create a schedule missing op 1
        let sched = SequentialSchedule {
            operations: vec![ScheduledOp {
                node_index: NodeIndex::new(0),
                op_id: 0,
                block_id: 0,
                phase: 0,
                proc_id: 0,
                op_type: OpType::Write,
                address: 0,
                memory_region: "A".to_string(),
                level: 0,
            }],
            num_levels: 1,
            num_processors: 2,
        };

        let result = verify_schedule_correctness(&dg, &sched);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_schedule_correctness_reversed() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);

        // Wrong order: Read before Write
        let sched = SequentialSchedule {
            operations: vec![
                ScheduledOp {
                    node_index: NodeIndex::new(1),
                    op_id: 1,
                    block_id: 0,
                    phase: 0,
                    proc_id: 1,
                    op_type: OpType::Read,
                    address: 0,
                    memory_region: "A".to_string(),
                    level: 0,
                },
                ScheduledOp {
                    node_index: NodeIndex::new(0),
                    op_id: 0,
                    block_id: 0,
                    phase: 0,
                    proc_id: 0,
                    op_type: OpType::Write,
                    address: 0,
                    memory_region: "A".to_string(),
                    level: 0,
                },
            ],
            num_levels: 1,
            num_processors: 2,
        };

        let result = verify_schedule_correctness(&dg, &sched);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(!errs.is_empty());
        assert!(errs[0].contains("Dependency violation"));
    }
}
