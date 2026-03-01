//! Locality-aware ordering within schedule levels.
//!
//! Within each level of the Brent schedule, operations are independent and
//! can be executed in any order. This module reorders them to maximize
//! temporal locality: group all operations on the same cache block together
//! before moving to the next block.

use std::collections::{HashMap, HashSet};

use super::dependency_graph::DependencyGraph;
use super::schedule::{Schedule, ScheduleEntry};
use super::work_optimal::{ScheduledOp, SequentialSchedule, extract_schedule};

/// Optimizer that reorders operations within each level to improve cache locality.
pub struct LocalityOptimizer {
    /// Block size for cache simulation.
    pub block_size: usize,
}

impl LocalityOptimizer {
    pub fn new(block_size: usize) -> Self {
        Self { block_size }
    }

    /// Reorder operations within each level by grouping same-block operations.
    ///
    /// Greedy strategy: within a level, process all operations targeting block B
    /// before moving to block B'. Blocks are ordered by their first appearance
    /// (lowest op_id in the level).
    pub fn optimize(&self, schedule: &SequentialSchedule) -> SequentialSchedule {
        let num_levels = schedule.num_levels;
        if num_levels == 0 {
            return schedule.clone();
        }

        // Group operations by level
        let mut by_level: Vec<Vec<ScheduledOp>> = vec![Vec::new(); num_levels];
        for op in &schedule.operations {
            if op.level < num_levels {
                by_level[op.level].push(op.clone());
            }
        }

        // Reorder within each level
        let mut result_ops = Vec::with_capacity(schedule.operations.len());
        for level_ops in &mut by_level {
            let reordered = self.reorder_for_locality(level_ops);
            result_ops.extend(reordered);
        }

        SequentialSchedule {
            operations: result_ops,
            num_levels: schedule.num_levels,
            num_processors: schedule.num_processors,
        }
    }

    /// Reorder a set of independent operations (within a single level)
    /// to group operations on the same block together.
    fn reorder_for_locality(&self, ops: &[ScheduledOp]) -> Vec<ScheduledOp> {
        if ops.len() <= 1 {
            return ops.to_vec();
        }

        // Group by block_id, preserving order within each block
        let mut block_groups: HashMap<usize, Vec<ScheduledOp>> = HashMap::new();
        let mut block_order: Vec<usize> = Vec::new();

        for op in ops {
            if !block_groups.contains_key(&op.block_id) {
                block_order.push(op.block_id);
            }
            block_groups.entry(op.block_id).or_default().push(op.clone());
        }

        // Order blocks by first op_id in each group (deterministic)
        block_order.sort_by_key(|blk| {
            block_groups[blk]
                .iter()
                .map(|op| op.op_id)
                .min()
                .unwrap_or(usize::MAX)
        });

        // Concatenate: all ops for block 0, then block 1, etc.
        let mut result = Vec::with_capacity(ops.len());
        for blk in &block_order {
            if let Some(group) = block_groups.get(blk) {
                result.extend(group.iter().cloned());
            }
        }
        result
    }

    /// Multi-level locality optimization: combines block grouping with
    /// inter-block reuse distance ordering and Hilbert curve fallback.
    pub fn multi_level_optimize(&self, schedule: &SequentialSchedule) -> SequentialSchedule {
        // First do block grouping per level
        let grouped = self.optimize(schedule);

        // Then apply inter-level reuse-aware ordering
        let entries: Vec<ScheduleEntry> = grouped.operations.iter()
            .map(|op| ScheduleEntry {
                operation_id: op.op_id,
                block_id: op.block_id,
                phase: op.phase,
                original_proc_id: op.proc_id,
                op_type: op.op_type,
                address: op.address,
                memory_region: op.memory_region.clone(),
                level: op.level,
            })
            .collect();

        // Compute reuse distances for current ordering
        let distances = reuse_distance_analysis(&entries);
        let avg_reuse = if distances.is_empty() { 0.0 } else {
            let finite: Vec<f64> = distances.iter()
                .filter(|&&d| d != usize::MAX)
                .map(|&d| d as f64)
                .collect();
            if finite.is_empty() { f64::MAX } else {
                finite.iter().sum::<f64>() / finite.len() as f64
            }
        };

        // If average reuse distance is high, try Hilbert curve ordering within levels
        if avg_reuse > 32.0 {
            self.hilbert_optimize(&grouped)
        } else {
            grouped
        }
    }

    fn hilbert_optimize(&self, schedule: &SequentialSchedule) -> SequentialSchedule {
        let num_levels = schedule.num_levels;
        if num_levels == 0 {
            return schedule.clone();
        }

        let mut by_level: Vec<Vec<ScheduledOp>> = vec![Vec::new(); num_levels];
        for op in &schedule.operations {
            if op.level < num_levels {
                by_level[op.level].push(op.clone());
            }
        }

        let mut result_ops = Vec::with_capacity(schedule.operations.len());
        for level_ops in &by_level {
            if level_ops.is_empty() {
                continue;
            }
            let entries: Vec<ScheduleEntry> = level_ops.iter()
                .map(|op| ScheduleEntry {
                    operation_id: op.op_id,
                    block_id: op.block_id,
                    phase: op.phase,
                    original_proc_id: op.proc_id,
                    op_type: op.op_type,
                    address: op.address,
                    memory_region: op.memory_region.clone(),
                    level: op.level,
                })
                .collect();
            let hilbert_order = hilbert_curve_order(&entries);
            for &idx in &hilbert_order {
                result_ops.push(level_ops[idx].clone());
            }
        }

        SequentialSchedule {
            operations: result_ops,
            num_levels: schedule.num_levels,
            num_processors: schedule.num_processors,
        }
    }

    /// Apply locality optimization and produce a Schedule.
    pub fn optimize_to_schedule(
        &self,
        graph: &DependencyGraph,
    ) -> Schedule {
        let seq = extract_schedule(graph);
        let optimized = self.optimize(&seq);
        optimized.to_schedule(self.block_size, graph.num_phases)
    }
}

/// Estimate the number of cache misses for a given ordering of operations,
/// using a simple window-based model.
///
/// Model: maintain a set of "hot" blocks of size `cache_blocks`. When an
/// operation accesses a block not in the hot set, it's a cache miss and the
/// oldest block is evicted (LRU approximation).
pub fn estimate_cache_misses(entries: &[ScheduleEntry], cache_blocks: usize) -> usize {
    if cache_blocks == 0 || entries.is_empty() {
        return entries.len();
    }

    let mut cache: Vec<usize> = Vec::with_capacity(cache_blocks);
    let mut misses: usize = 0;

    for entry in entries {
        let blk = entry.block_id;
        if let Some(pos) = cache.iter().position(|&b| b == blk) {
            // Cache hit: move to end (most recently used)
            cache.remove(pos);
            cache.push(blk);
        } else {
            // Cache miss
            misses += 1;
            if cache.len() >= cache_blocks {
                cache.remove(0); // evict LRU
            }
            cache.push(blk);
        }
    }

    misses
}

/// Estimate cache misses for a SequentialSchedule.
pub fn estimate_misses_seq(schedule: &SequentialSchedule, cache_blocks: usize) -> usize {
    let entries: Vec<ScheduleEntry> = schedule
        .operations
        .iter()
        .map(|op| ScheduleEntry {
            operation_id: op.op_id,
            block_id: op.block_id,
            phase: op.phase,
            original_proc_id: op.proc_id,
            op_type: op.op_type,
            address: op.address,
            memory_region: op.memory_region.clone(),
            level: op.level,
        })
        .collect();
    estimate_cache_misses(&entries, cache_blocks)
}

/// Count block transitions: each time the sequential order moves to
/// a different block, it's a transition. Number of transitions + 1 ≥ distinct blocks.
pub fn count_block_transitions(entries: &[ScheduleEntry]) -> usize {
    if entries.len() <= 1 {
        return 0;
    }
    let mut transitions = 0;
    for i in 1..entries.len() {
        if entries[i].block_id != entries[i - 1].block_id {
            transitions += 1;
        }
    }
    transitions
}

/// Count block transitions in a SequentialSchedule.
pub fn count_transitions_seq(schedule: &SequentialSchedule) -> usize {
    if schedule.operations.len() <= 1 {
        return 0;
    }
    let mut transitions = 0;
    for i in 1..schedule.operations.len() {
        if schedule.operations[i].block_id != schedule.operations[i - 1].block_id {
            transitions += 1;
        }
    }
    transitions
}

/// Compare two orderings: random vs locality-optimized.
/// Returns (random_misses, optimized_misses).
pub fn compare_orderings(
    graph: &DependencyGraph,
    cache_blocks: usize,
) -> (usize, usize) {
    let base = extract_schedule(graph);
    let optimizer = LocalityOptimizer::new(graph.block_size);
    let optimized = optimizer.optimize(&base);

    let base_misses = estimate_misses_seq(&base, cache_blocks);
    let opt_misses = estimate_misses_seq(&optimized, cache_blocks);

    (base_misses, opt_misses)
}

/// Map a (x, y) coordinate to a position on a Hilbert curve of order n.
fn xy_to_hilbert(n: usize, mut x: usize, mut y: usize) -> usize {
    let mut d: usize = 0;
    let mut s = n / 2;
    while s > 0 {
        let rx = if (x & s) > 0 { 1 } else { 0 };
        let ry = if (y & s) > 0 { 1 } else { 0 };
        d += s * s * ((3 * rx) ^ ry);

        // Rotate
        if ry == 0 {
            if rx == 1 {
                x = s.wrapping_mul(2).wrapping_sub(1).wrapping_sub(x);
                y = s.wrapping_mul(2).wrapping_sub(1).wrapping_sub(y);
            }
            std::mem::swap(&mut x, &mut y);
        }

        s /= 2;
    }
    d
}

/// Compute a space-filling Hilbert curve ordering for operations based on
/// 2D block IDs.
///
/// Treats block IDs as (x, y) coordinates in a 2D grid and orders them
/// according to the Hilbert curve to maximize spatial locality.
/// The grid dimension is the smallest power of 2 ≥ sqrt(max_block_id + 1).
pub fn hilbert_curve_order(ops: &[ScheduleEntry]) -> Vec<usize> {
    if ops.is_empty() {
        return Vec::new();
    }

    let max_block = ops.iter().map(|op| op.block_id).max().unwrap_or(0);
    let side = {
        let s = ((max_block + 1) as f64).sqrt().ceil() as usize;
        s.next_power_of_two().max(1)
    };

    let mut indexed: Vec<(usize, usize)> = ops
        .iter()
        .enumerate()
        .map(|(i, op)| {
            let x = op.block_id % side;
            let y = op.block_id / side;
            let h = xy_to_hilbert(side, x, y);
            (i, h)
        })
        .collect();

    indexed.sort_by_key(|&(_, h)| h);
    indexed.into_iter().map(|(i, _)| i).collect()
}

/// Compute reuse distances for a given ordering of operations.
///
/// The reuse distance of an access to block B is the number of distinct
/// blocks accessed since the last access to B. First accesses get
/// distance = usize::MAX (infinity).
pub fn reuse_distance_analysis(entries: &[ScheduleEntry]) -> Vec<usize> {
    let mut distances = Vec::with_capacity(entries.len());
    let mut last_seen: HashMap<usize, usize> = HashMap::new();
    let mut distinct_since: Vec<HashSet<usize>> = Vec::new();

    for (i, entry) in entries.iter().enumerate() {
        let blk = entry.block_id;
        if let Some(&last_pos) = last_seen.get(&blk) {
            // Count distinct blocks between last_pos+1 and i-1
            let mut distinct: HashSet<usize> = HashSet::new();
            for j in (last_pos + 1)..i {
                distinct.insert(entries[j].block_id);
            }
            distances.push(distinct.len());
        } else {
            distances.push(usize::MAX);
        }
        last_seen.insert(blk, i);
    }

    distances
}

/// Greedy approximation for optimal block ordering (weighted TSP-like).
///
/// Given a set of blocks and a transition cost matrix, find an ordering
/// that minimizes total transition cost using nearest-neighbor heuristic.
pub fn optimal_block_ordering_greedy(
    blocks: &[usize],
    transition_costs: &HashMap<(usize, usize), f64>,
) -> Vec<usize> {
    if blocks.is_empty() {
        return Vec::new();
    }
    if blocks.len() == 1 {
        return blocks.to_vec();
    }

    let mut remaining: HashSet<usize> = blocks.iter().copied().collect();
    let mut order: Vec<usize> = Vec::with_capacity(blocks.len());

    // Start with the first block
    let start = blocks[0];
    order.push(start);
    remaining.remove(&start);

    while !remaining.is_empty() {
        let current = *order.last().unwrap();
        // Find nearest unvisited block
        let next = remaining
            .iter()
            .min_by(|&&a, &&b| {
                let cost_a = transition_costs
                    .get(&(current, a))
                    .copied()
                    .unwrap_or(1.0);
                let cost_b = transition_costs
                    .get(&(current, b))
                    .copied()
                    .unwrap_or(1.0);
                cost_a.partial_cmp(&cost_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap();

        order.push(next);
        remaining.remove(&next);
    }

    order
}

/// Comparison report for multiple orderings.
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    /// Name and cache miss count for each ordering.
    pub results: Vec<(String, usize)>,
    /// Name of the best ordering.
    pub best: String,
    /// Name of the worst ordering.
    pub worst: String,
}

impl std::fmt::Display for ComparisonReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Ordering Comparison ===")?;
        for (name, misses) in &self.results {
            writeln!(f, "  {}: {} misses", name, misses)?;
        }
        writeln!(f, "Best:  {}", self.best)?;
        writeln!(f, "Worst: {}", self.worst)?;
        Ok(())
    }
}

/// Compare multiple orderings of the same entries, evaluating cache misses.
///
/// `orders` contains index permutations; `names` gives a label to each.
/// `entries` is the base set of schedule entries.
pub fn compare_multiple_orderings(
    entries: &[ScheduleEntry],
    orders: &[Vec<usize>],
    names: &[&str],
    cache_blocks: usize,
) -> ComparisonReport {
    let mut results: Vec<(String, usize)> = Vec::new();

    for (order, &name) in orders.iter().zip(names.iter()) {
        let reordered: Vec<ScheduleEntry> = order
            .iter()
            .filter(|&&i| i < entries.len())
            .map(|&i| entries[i].clone())
            .collect();
        let misses = estimate_cache_misses(&reordered, cache_blocks);
        results.push((name.to_string(), misses));
    }

    let best = results
        .iter()
        .min_by_key(|(_, m)| *m)
        .map(|(n, _)| n.clone())
        .unwrap_or_default();
    let worst = results
        .iter()
        .max_by_key(|(_, m)| *m)
        .map(|(n, _)| n.clone())
        .unwrap_or_default();

    ComparisonReport {
        results,
        best,
        worst,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brent_scheduler::dependency_graph::{DependencyGraph, OperationNode};
    use crate::brent_scheduler::schedule::{OpType, ScheduleEntry};

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

    fn make_entry(op_id: usize, block_id: usize) -> ScheduleEntry {
        ScheduleEntry {
            operation_id: op_id,
            block_id,
            phase: 0,
            original_proc_id: 0,
            op_type: OpType::Read,
            address: block_id * 4,
            memory_region: "A".to_string(),
            level: 0,
        }
    }

    #[test]
    fn test_locality_grouping() {
        // 6 independent writes to 3 blocks, interleaved: 0,1,2,0,1,2
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 1, 0, 1, OpType::Write, 4),
            make_op(2, 2, 0, 2, OpType::Write, 8),
            make_op(3, 0, 0, 3, OpType::Write, 1),
            make_op(4, 1, 0, 4, OpType::Write, 5),
            make_op(5, 2, 0, 5, OpType::Write, 9),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let base = extract_schedule(&dg);

        let optimizer = LocalityOptimizer::new(4);
        let optimized = optimizer.optimize(&base);

        assert_eq!(optimized.total_work(), 6);

        // After optimization, same-block ops should be adjacent
        let blocks: Vec<usize> = optimized.operations.iter().map(|o| o.block_id).collect();
        // Should be grouped: [0,0,1,1,2,2] in some order of groups
        for i in 0..blocks.len() - 1 {
            if blocks[i] == blocks[i + 1] {
                continue; // same block, good
            }
            // If we switch blocks, the old block should not appear again
            let old_blk = blocks[i];
            for j in (i + 2)..blocks.len() {
                assert_ne!(
                    blocks[j], old_blk,
                    "Block {} reappears after transition at position {} (found at {})",
                    old_blk, i, j
                );
            }
        }
    }

    #[test]
    fn test_cache_misses_all_same_block() {
        let entries: Vec<ScheduleEntry> = (0..5).map(|i| make_entry(i, 0)).collect();
        let misses = estimate_cache_misses(&entries, 4);
        assert_eq!(misses, 1); // one cold miss, then all hits
    }

    #[test]
    fn test_cache_misses_all_different_blocks() {
        let entries: Vec<ScheduleEntry> = (0..5).map(|i| make_entry(i, i)).collect();
        let misses = estimate_cache_misses(&entries, 4);
        assert_eq!(misses, 5); // all cold misses (cache holds 4, 5 distinct)
    }

    #[test]
    fn test_cache_misses_with_reuse() {
        // Blocks: 0,1,0,1,0 with cache size 2
        let entries = vec![
            make_entry(0, 0),
            make_entry(1, 1),
            make_entry(2, 0),
            make_entry(3, 1),
            make_entry(4, 0),
        ];
        let misses = estimate_cache_misses(&entries, 2);
        assert_eq!(misses, 2); // miss on first 0, miss on first 1, then hits
    }

    #[test]
    fn test_block_transitions() {
        let entries = vec![
            make_entry(0, 0),
            make_entry(1, 0),
            make_entry(2, 1),
            make_entry(3, 1),
            make_entry(4, 2),
        ];
        let transitions = count_block_transitions(&entries);
        assert_eq!(transitions, 2); // 0→1, 1→2
    }

    #[test]
    fn test_block_transitions_interleaved() {
        let entries = vec![
            make_entry(0, 0),
            make_entry(1, 1),
            make_entry(2, 0),
            make_entry(3, 1),
        ];
        let transitions = count_block_transitions(&entries);
        assert_eq!(transitions, 3); // 0→1, 1→0, 0→1
    }

    #[test]
    fn test_locality_reduces_transitions() {
        // Interleaved blocks → many transitions
        // After locality optimization → grouped → fewer transitions
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 1, 0, 1, OpType::Write, 4),
            make_op(2, 2, 0, 2, OpType::Write, 8),
            make_op(3, 0, 0, 3, OpType::Write, 1),
            make_op(4, 1, 0, 4, OpType::Write, 5),
            make_op(5, 2, 0, 5, OpType::Write, 9),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let base = extract_schedule(&dg);
        let optimizer = LocalityOptimizer::new(4);
        let optimized = optimizer.optimize(&base);

        let base_trans = count_transitions_seq(&base);
        let opt_trans = count_transitions_seq(&optimized);

        assert!(
            opt_trans <= base_trans,
            "Optimized should have ≤ transitions: {} vs {}",
            opt_trans,
            base_trans
        );
        // After grouping 3 blocks with 2 ops each, transitions = 2 (at most)
        assert!(opt_trans <= 2);
    }

    #[test]
    fn test_compare_orderings() {
        let ops: Vec<OperationNode> = (0..12)
            .map(|i| make_op(i, i % 3, 0, i, OpType::Write, (i % 3) * 4 + i / 3))
            .collect();
        let dg = DependencyGraph::from_operations(ops, 4);
        let (base_misses, opt_misses) = compare_orderings(&dg, 2);
        assert!(opt_misses <= base_misses);
    }

    #[test]
    fn test_empty_schedule_locality() {
        let dg = DependencyGraph::from_operations(vec![], 4);
        let base = extract_schedule(&dg);
        let optimizer = LocalityOptimizer::new(4);
        let optimized = optimizer.optimize(&base);
        assert_eq!(optimized.total_work(), 0);
    }

    #[test]
    fn test_single_op_locality() {
        let ops = vec![make_op(0, 0, 0, 0, OpType::Read, 0)];
        let dg = DependencyGraph::from_operations(ops, 4);
        let base = extract_schedule(&dg);
        let optimizer = LocalityOptimizer::new(4);
        let optimized = optimizer.optimize(&base);
        assert_eq!(optimized.total_work(), 1);
        assert_eq!(count_transitions_seq(&optimized), 0);
    }

    #[test]
    fn test_optimize_to_schedule() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 1, 0, 1, OpType::Write, 4),
            make_op(2, 0, 0, 2, OpType::Write, 1),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let optimizer = LocalityOptimizer::new(4);
        let schedule = optimizer.optimize_to_schedule(&dg);

        assert_eq!(schedule.total_ops(), 3);
        // Block 0 ops should be adjacent
        let blocks: Vec<usize> = schedule.entries.iter().map(|e| e.block_id).collect();
        let first_0 = blocks.iter().position(|&b| b == 0).unwrap();
        let last_0 = blocks.iter().rposition(|&b| b == 0).unwrap();
        // All block-0 entries should be contiguous
        for i in first_0..=last_0 {
            assert_eq!(blocks[i], 0);
        }
    }

    #[test]
    fn test_hilbert_curve_order_basic() {
        let entries = vec![
            make_entry(0, 3),
            make_entry(1, 0),
            make_entry(2, 1),
            make_entry(3, 2),
        ];
        let order = hilbert_curve_order(&entries);
        assert_eq!(order.len(), 4);
        // All indices should be present
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_hilbert_curve_order_empty() {
        let order = hilbert_curve_order(&[]);
        assert!(order.is_empty());
    }

    #[test]
    fn test_hilbert_curve_order_single() {
        let entries = vec![make_entry(0, 5)];
        let order = hilbert_curve_order(&entries);
        assert_eq!(order, vec![0]);
    }

    #[test]
    fn test_reuse_distance_analysis() {
        // Blocks: 0, 1, 0, 1, 2
        let entries = vec![
            make_entry(0, 0),
            make_entry(1, 1),
            make_entry(2, 0),
            make_entry(3, 1),
            make_entry(4, 2),
        ];
        let distances = reuse_distance_analysis(&entries);
        assert_eq!(distances.len(), 5);
        // First access to block 0 → infinity
        assert_eq!(distances[0], usize::MAX);
        // First access to block 1 → infinity
        assert_eq!(distances[1], usize::MAX);
        // Second access to block 0, 1 distinct block between → distance 1
        assert_eq!(distances[2], 1);
        // Second access to block 1, 1 distinct block between → distance 1
        assert_eq!(distances[3], 1);
        // First access to block 2 → infinity
        assert_eq!(distances[4], usize::MAX);
    }

    #[test]
    fn test_reuse_distance_same_block() {
        let entries = vec![
            make_entry(0, 0),
            make_entry(1, 0),
            make_entry(2, 0),
        ];
        let distances = reuse_distance_analysis(&entries);
        assert_eq!(distances[0], usize::MAX);
        assert_eq!(distances[1], 0); // no distinct blocks between
        assert_eq!(distances[2], 0);
    }

    #[test]
    fn test_optimal_block_ordering_greedy_basic() {
        let blocks = vec![0, 1, 2];
        let mut costs = HashMap::new();
        costs.insert((0, 1), 1.0);
        costs.insert((0, 2), 10.0);
        costs.insert((1, 0), 1.0);
        costs.insert((1, 2), 1.0);
        costs.insert((2, 0), 10.0);
        costs.insert((2, 1), 1.0);

        let order = optimal_block_ordering_greedy(&blocks, &costs);
        assert_eq!(order.len(), 3);
        // Starting from 0, nearest is 1, then from 1 nearest is 2
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn test_optimal_block_ordering_greedy_empty() {
        let order = optimal_block_ordering_greedy(&[], &HashMap::new());
        assert!(order.is_empty());
    }

    #[test]
    fn test_optimal_block_ordering_greedy_single() {
        let order = optimal_block_ordering_greedy(&[42], &HashMap::new());
        assert_eq!(order, vec![42]);
    }

    #[test]
    fn test_compare_multiple_orderings() {
        let entries: Vec<ScheduleEntry> = (0..6)
            .map(|i| make_entry(i, i % 3))
            .collect();

        let natural: Vec<usize> = (0..6).collect();
        let reversed: Vec<usize> = (0..6).rev().collect();
        // Group by block
        let grouped = vec![0, 3, 1, 4, 2, 5];

        let report = compare_multiple_orderings(
            &entries,
            &[natural, reversed, grouped],
            &["natural", "reversed", "grouped"],
            2,
        );

        assert_eq!(report.results.len(), 3);
        assert!(!report.best.is_empty());
        assert!(!report.worst.is_empty());
        let s = format!("{}", report);
        assert!(s.contains("Ordering Comparison"));
    }

    #[test]
    fn test_multi_level_optimize() {
        // Create ops spread across many distinct blocks to trigger high reuse distance
        // (Hilbert fallback path) and also a low-reuse case.
        // Low reuse distance case: few blocks, repeated accesses
        let ops_low: Vec<OperationNode> = (0..8)
            .map(|i| make_op(i, i % 2, 0, i, OpType::Write, (i % 2) * 4 + i / 2))
            .collect();
        let dg_low = DependencyGraph::from_operations(ops_low, 4);
        let base_low = extract_schedule(&dg_low);
        let optimizer = LocalityOptimizer::new(4);
        let result_low = optimizer.multi_level_optimize(&base_low);
        // Should preserve all operations
        assert_eq!(result_low.total_work(), base_low.total_work());
        // With low reuse distance, should return the grouped result (not Hilbert)
        let transitions_low = count_transitions_seq(&result_low);
        assert!(transitions_low <= 1, "Expected ≤1 transition for 2 blocks, got {}", transitions_low);

        // High reuse distance case: many distinct blocks, no repeated accesses
        let ops_high: Vec<OperationNode> = (0..64)
            .map(|i| make_op(i, i, 0, i, OpType::Read, i * 4))
            .collect();
        let dg_high = DependencyGraph::from_operations(ops_high, 4);
        let base_high = extract_schedule(&dg_high);
        let result_high = optimizer.multi_level_optimize(&base_high);
        assert_eq!(result_high.total_work(), base_high.total_work());

        // Empty schedule
        let dg_empty = DependencyGraph::from_operations(vec![], 4);
        let base_empty = extract_schedule(&dg_empty);
        let result_empty = optimizer.multi_level_optimize(&base_empty);
        assert_eq!(result_empty.total_work(), 0);
    }
}
