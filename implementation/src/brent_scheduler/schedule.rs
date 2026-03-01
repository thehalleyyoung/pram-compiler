//! Schedule representation for Brent's theorem-based PRAM scheduling.
//!
//! Defines the output format: a sequential ordering of PRAM operations
//! with metadata for cost analysis and validation.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// The type of a PRAM memory operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpType {
    Read,
    Write,
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpType::Read => write!(f, "R"),
            OpType::Write => write!(f, "W"),
        }
    }
}

/// A single entry in the sequential schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleEntry {
    /// Unique operation id (matches the dependency graph node).
    pub operation_id: usize,
    /// Cache block id this operation targets.
    pub block_id: usize,
    /// Phase (between barriers) this operation belongs to.
    pub phase: usize,
    /// Original processor id that issued this operation.
    pub original_proc_id: usize,
    /// Read or write.
    pub op_type: OpType,
    /// Shared memory address (element index).
    pub address: usize,
    /// Name of the shared memory region.
    pub memory_region: String,
    /// The level assigned during work-optimal scheduling.
    pub level: usize,
}

impl fmt::Display for ScheduleEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[op={} blk={} ph={} proc={} {} {}[{}]]",
            self.operation_id,
            self.block_id,
            self.phase,
            self.original_proc_id,
            self.op_type,
            self.memory_region,
            self.address,
        )
    }
}

/// A complete sequential schedule of PRAM operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    /// Ordered list of schedule entries (execution order).
    pub entries: Vec<ScheduleEntry>,
    /// Number of processors in the original parallel program.
    pub num_processors: usize,
    /// Number of parallel phases.
    pub num_phases: usize,
    /// Block size used for cache analysis.
    pub block_size: usize,
}

impl Schedule {
    /// Create a new empty schedule.
    pub fn new(num_processors: usize, num_phases: usize, block_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            num_processors,
            num_phases,
            block_size,
        }
    }

    /// Append an entry to the schedule.
    pub fn push(&mut self, entry: ScheduleEntry) {
        self.entries.push(entry);
    }

    /// Total number of operations.
    pub fn total_ops(&self) -> usize {
        self.entries.len()
    }

    /// Number of operations in each phase.
    pub fn ops_per_phase(&self) -> HashMap<usize, usize> {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for e in &self.entries {
            *counts.entry(e.phase).or_insert(0) += 1;
        }
        counts
    }

    /// Number of operations touching each block.
    pub fn ops_per_block(&self) -> HashMap<usize, usize> {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for e in &self.entries {
            *counts.entry(e.block_id).or_insert(0) += 1;
        }
        counts
    }

    /// Number of read operations.
    pub fn read_count(&self) -> usize {
        self.entries.iter().filter(|e| e.op_type == OpType::Read).count()
    }

    /// Number of write operations.
    pub fn write_count(&self) -> usize {
        self.entries.iter().filter(|e| e.op_type == OpType::Write).count()
    }

    /// Distinct blocks accessed.
    pub fn distinct_blocks(&self) -> usize {
        let blocks: HashSet<usize> = self.entries.iter().map(|e| e.block_id).collect();
        blocks.len()
    }

    /// Distinct phases present.
    pub fn distinct_phases(&self) -> usize {
        let phases: HashSet<usize> = self.entries.iter().map(|e| e.phase).collect();
        phases.len()
    }

    /// Iterator over schedule entries in execution order.
    pub fn iter(&self) -> impl Iterator<Item = &ScheduleEntry> {
        self.entries.iter()
    }

    /// Iterator over entries in a specific phase.
    pub fn iter_phase(&self, phase: usize) -> impl Iterator<Item = &ScheduleEntry> {
        self.entries.iter().filter(move |e| e.phase == phase)
    }

    /// Iterator over entries touching a specific block.
    pub fn iter_block(&self, block_id: usize) -> impl Iterator<Item = &ScheduleEntry> {
        self.entries.iter().filter(move |e| e.block_id == block_id)
    }

    /// Validate that every dependency (from_id, to_id) in `deps` is respected:
    /// the entry with `from_id` appears before `to_id` in the schedule.
    pub fn validate_dependencies(&self, deps: &[(usize, usize)]) -> Result<(), Vec<(usize, usize)>> {
        // Build position map: operation_id -> index in schedule
        let pos: HashMap<usize, usize> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.operation_id, i))
            .collect();

        let mut violations = Vec::new();
        for &(from, to) in deps {
            match (pos.get(&from), pos.get(&to)) {
                (Some(&from_pos), Some(&to_pos)) => {
                    if from_pos >= to_pos {
                        violations.push((from, to));
                    }
                }
                (None, _) | (_, None) => {
                    // Missing operations in schedule — also a violation
                    violations.push((from, to));
                }
            }
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }

    /// Maximum level in the schedule (proxy for critical path length).
    pub fn max_level(&self) -> usize {
        self.entries.iter().map(|e| e.level).max().unwrap_or(0)
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Merge two non-conflicting schedules into one.
    ///
    /// Entries from `other` are appended after `self`'s entries.
    /// The resulting schedule takes the maximum of processors, phases,
    /// and uses the block_size from `self`.
    pub fn merge(a: &Schedule, b: &Schedule) -> Schedule {
        let mut merged = Schedule::new(
            a.num_processors.max(b.num_processors),
            a.num_phases.max(b.num_phases),
            a.block_size,
        );
        for entry in &a.entries {
            merged.push(entry.clone());
        }
        for entry in &b.entries {
            merged.push(entry.clone());
        }
        merged
    }

    /// Split the schedule into separate schedules by phase.
    pub fn split_by_phase(&self) -> Vec<Schedule> {
        let mut phase_map: HashMap<usize, Vec<ScheduleEntry>> = HashMap::new();
        for entry in &self.entries {
            phase_map.entry(entry.phase).or_default().push(entry.clone());
        }

        let mut phases: Vec<usize> = phase_map.keys().copied().collect();
        phases.sort();

        phases
            .into_iter()
            .map(|phase| {
                let mut sched = Schedule::new(self.num_processors, 1, self.block_size);
                if let Some(entries) = phase_map.get(&phase) {
                    for e in entries {
                        sched.push(e.clone());
                    }
                }
                sched
            })
            .collect()
    }

    /// Reorder entries within each level for improved locality.
    ///
    /// Groups entries by block_id within each level, keeping the level
    /// ordering intact.
    pub fn reorder_for_locality(&self) -> Schedule {
        if self.entries.is_empty() {
            return self.clone();
        }

        // Group by level
        let max_level = self.max_level();
        let mut by_level: Vec<Vec<ScheduleEntry>> = vec![Vec::new(); max_level + 1];
        for entry in &self.entries {
            by_level[entry.level].push(entry.clone());
        }

        // Within each level, sort by block_id for locality
        for level_entries in &mut by_level {
            level_entries.sort_by_key(|e| e.block_id);
        }

        let mut result = Schedule::new(self.num_processors, self.num_phases, self.block_size);
        for level_entries in by_level {
            for entry in level_entries {
                result.push(entry);
            }
        }
        result
    }

    /// Compute a detailed analysis of the schedule.
    pub fn analyze(&self) -> ScheduleAnalysis {
        let ops_per_phase = self.ops_per_phase();
        let ops_per_block = self.ops_per_block();

        let phase_count = self.distinct_phases();
        let block_count = self.distinct_blocks();

        let avg_ops_per_phase = if phase_count > 0 {
            self.total_ops() as f64 / phase_count as f64
        } else {
            0.0
        };

        let avg_ops_per_block = if block_count > 0 {
            self.total_ops() as f64 / block_count as f64
        } else {
            0.0
        };

        let max_ops_per_block = ops_per_block.values().copied().max().unwrap_or(0);
        let min_ops_per_block = ops_per_block.values().copied().min().unwrap_or(0);

        // Block utilization: how evenly are operations distributed
        let utilization = if max_ops_per_block > 0 {
            min_ops_per_block as f64 / max_ops_per_block as f64
        } else {
            1.0
        };

        // Reuse count: how many operations access a previously-seen block
        let mut seen_blocks: HashSet<usize> = HashSet::new();
        let mut reuse_count = 0;
        for entry in &self.entries {
            if !seen_blocks.insert(entry.block_id) {
                reuse_count += 1;
            }
        }

        ScheduleAnalysis {
            total_ops: self.total_ops(),
            reads: self.read_count(),
            writes: self.write_count(),
            num_phases: phase_count,
            num_blocks: block_count,
            avg_ops_per_phase,
            avg_ops_per_block,
            max_ops_per_block,
            min_ops_per_block,
            block_utilization: utilization,
            reuse_count,
            max_level: self.max_level(),
        }
    }
}

/// Detailed analysis of a schedule's structure.
#[derive(Debug, Clone)]
pub struct ScheduleAnalysis {
    pub total_ops: usize,
    pub reads: usize,
    pub writes: usize,
    pub num_phases: usize,
    pub num_blocks: usize,
    pub avg_ops_per_phase: f64,
    pub avg_ops_per_block: f64,
    pub max_ops_per_block: usize,
    pub min_ops_per_block: usize,
    pub block_utilization: f64,
    pub reuse_count: usize,
    pub max_level: usize,
}

impl fmt::Display for ScheduleAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Schedule Analysis ===")?;
        writeln!(f, "Total ops:        {}", self.total_ops)?;
        writeln!(f, "Reads/Writes:     {}/{}", self.reads, self.writes)?;
        writeln!(f, "Phases:           {}", self.num_phases)?;
        writeln!(f, "Blocks:           {}", self.num_blocks)?;
        writeln!(f, "Avg ops/phase:    {:.2}", self.avg_ops_per_phase)?;
        writeln!(f, "Avg ops/block:    {:.2}", self.avg_ops_per_block)?;
        writeln!(f, "Max ops/block:    {}", self.max_ops_per_block)?;
        writeln!(f, "Min ops/block:    {}", self.min_ops_per_block)?;
        writeln!(f, "Block utilization:{:.4}", self.block_utilization)?;
        writeln!(f, "Reuse count:      {}", self.reuse_count)?;
        writeln!(f, "Max level:        {}", self.max_level)?;
        Ok(())
    }
}

impl fmt::Display for Schedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Schedule: {} ops, {} procs, {} phases, block_size={}",
            self.total_ops(),
            self.num_processors,
            self.num_phases,
            self.block_size,
        )?;
        for (i, entry) in self.entries.iter().enumerate() {
            writeln!(f, "  {:4}: {}", i, entry)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(op_id: usize, block: usize, phase: usize, proc: usize, op: OpType) -> ScheduleEntry {
        ScheduleEntry {
            operation_id: op_id,
            block_id: block,
            phase,
            original_proc_id: proc,
            op_type: op,
            address: block * 4 + proc % 4,
            memory_region: "A".to_string(),
            level: phase,
        }
    }

    #[test]
    fn test_schedule_basic_stats() {
        let mut sched = Schedule::new(4, 2, 4);
        sched.push(make_entry(0, 0, 0, 0, OpType::Read));
        sched.push(make_entry(1, 0, 0, 1, OpType::Read));
        sched.push(make_entry(2, 1, 0, 2, OpType::Write));
        sched.push(make_entry(3, 1, 1, 3, OpType::Write));

        assert_eq!(sched.total_ops(), 4);
        assert_eq!(sched.read_count(), 2);
        assert_eq!(sched.write_count(), 2);
        assert_eq!(sched.distinct_blocks(), 2);
        assert_eq!(sched.distinct_phases(), 2);
    }

    #[test]
    fn test_ops_per_phase() {
        let mut sched = Schedule::new(4, 2, 4);
        sched.push(make_entry(0, 0, 0, 0, OpType::Read));
        sched.push(make_entry(1, 0, 0, 1, OpType::Read));
        sched.push(make_entry(2, 1, 1, 2, OpType::Write));

        let per_phase = sched.ops_per_phase();
        assert_eq!(per_phase[&0], 2);
        assert_eq!(per_phase[&1], 1);
    }

    #[test]
    fn test_ops_per_block() {
        let mut sched = Schedule::new(4, 1, 4);
        sched.push(make_entry(0, 0, 0, 0, OpType::Read));
        sched.push(make_entry(1, 0, 0, 1, OpType::Write));
        sched.push(make_entry(2, 1, 0, 2, OpType::Read));

        let per_block = sched.ops_per_block();
        assert_eq!(per_block[&0], 2);
        assert_eq!(per_block[&1], 1);
    }

    #[test]
    fn test_validate_dependencies_ok() {
        let mut sched = Schedule::new(2, 1, 4);
        sched.push(make_entry(0, 0, 0, 0, OpType::Read));
        sched.push(make_entry(1, 0, 0, 1, OpType::Write));
        sched.push(make_entry(2, 1, 0, 0, OpType::Read));

        let deps = vec![(0, 1), (1, 2)];
        assert!(sched.validate_dependencies(&deps).is_ok());
    }

    #[test]
    fn test_validate_dependencies_violation() {
        let mut sched = Schedule::new(2, 1, 4);
        sched.push(make_entry(1, 0, 0, 1, OpType::Write));
        sched.push(make_entry(0, 0, 0, 0, OpType::Read));

        // 0 should come before 1, but 1 is first
        let deps = vec![(0, 1)];
        let result = sched.validate_dependencies(&deps);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), vec![(0, 1)]);
    }

    #[test]
    fn test_schedule_serialization() {
        let mut sched = Schedule::new(2, 1, 4);
        sched.push(make_entry(0, 0, 0, 0, OpType::Read));
        sched.push(make_entry(1, 0, 0, 1, OpType::Write));

        let json = sched.to_json().unwrap();
        let restored = Schedule::from_json(&json).unwrap();
        assert_eq!(restored.total_ops(), 2);
        assert_eq!(restored.num_processors, 2);
        assert_eq!(restored.entries[0].operation_id, 0);
        assert_eq!(restored.entries[1].op_type, OpType::Write);
    }

    #[test]
    fn test_schedule_iterators() {
        let mut sched = Schedule::new(4, 2, 4);
        sched.push(make_entry(0, 0, 0, 0, OpType::Read));
        sched.push(make_entry(1, 1, 0, 1, OpType::Read));
        sched.push(make_entry(2, 0, 1, 2, OpType::Write));
        sched.push(make_entry(3, 1, 1, 3, OpType::Write));

        assert_eq!(sched.iter().count(), 4);
        assert_eq!(sched.iter_phase(0).count(), 2);
        assert_eq!(sched.iter_phase(1).count(), 2);
        assert_eq!(sched.iter_block(0).count(), 2);
        assert_eq!(sched.iter_block(1).count(), 2);
    }

    #[test]
    fn test_max_level() {
        let mut sched = Schedule::new(4, 3, 4);
        sched.push(ScheduleEntry {
            operation_id: 0,
            block_id: 0,
            phase: 0,
            original_proc_id: 0,
            op_type: OpType::Read,
            address: 0,
            memory_region: "A".to_string(),
            level: 0,
        });
        sched.push(ScheduleEntry {
            operation_id: 1,
            block_id: 0,
            phase: 0,
            original_proc_id: 1,
            op_type: OpType::Write,
            address: 1,
            memory_region: "A".to_string(),
            level: 3,
        });
        assert_eq!(sched.max_level(), 3);
    }

    #[test]
    fn test_empty_schedule() {
        let sched = Schedule::new(1, 0, 4);
        assert_eq!(sched.total_ops(), 0);
        assert_eq!(sched.read_count(), 0);
        assert_eq!(sched.write_count(), 0);
        assert_eq!(sched.distinct_blocks(), 0);
        assert_eq!(sched.max_level(), 0);
        assert!(sched.validate_dependencies(&[]).is_ok());
    }

    #[test]
    fn test_display() {
        let mut sched = Schedule::new(2, 1, 4);
        sched.push(make_entry(0, 0, 0, 0, OpType::Read));
        let s = format!("{}", sched);
        assert!(s.contains("1 ops"));
        assert!(s.contains("2 procs"));
    }

    #[test]
    fn test_merge_schedules() {
        let mut a = Schedule::new(2, 1, 4);
        a.push(make_entry(0, 0, 0, 0, OpType::Read));
        a.push(make_entry(1, 0, 0, 1, OpType::Write));

        let mut b = Schedule::new(4, 2, 4);
        b.push(make_entry(2, 1, 1, 2, OpType::Read));

        let merged = Schedule::merge(&a, &b);
        assert_eq!(merged.total_ops(), 3);
        assert_eq!(merged.num_processors, 4);
        assert_eq!(merged.num_phases, 2);
        assert_eq!(merged.entries[0].operation_id, 0);
        assert_eq!(merged.entries[2].operation_id, 2);
    }

    #[test]
    fn test_split_by_phase() {
        let mut sched = Schedule::new(4, 3, 4);
        sched.push(make_entry(0, 0, 0, 0, OpType::Read));
        sched.push(make_entry(1, 0, 0, 1, OpType::Write));
        sched.push(make_entry(2, 1, 1, 2, OpType::Read));
        sched.push(make_entry(3, 1, 1, 3, OpType::Write));
        sched.push(make_entry(4, 2, 2, 0, OpType::Read));

        let phases = sched.split_by_phase();
        assert_eq!(phases.len(), 3);
        assert_eq!(phases[0].total_ops(), 2);
        assert_eq!(phases[1].total_ops(), 2);
        assert_eq!(phases[2].total_ops(), 1);
    }

    #[test]
    fn test_split_by_phase_single() {
        let mut sched = Schedule::new(2, 1, 4);
        sched.push(make_entry(0, 0, 0, 0, OpType::Read));
        sched.push(make_entry(1, 0, 0, 1, OpType::Write));

        let phases = sched.split_by_phase();
        assert_eq!(phases.len(), 1);
        assert_eq!(phases[0].total_ops(), 2);
    }

    #[test]
    fn test_reorder_for_locality() {
        let mut sched = Schedule::new(4, 1, 4);
        // Interleaved blocks at same level
        sched.push(ScheduleEntry {
            operation_id: 0, block_id: 1, phase: 0, original_proc_id: 0,
            op_type: OpType::Read, address: 4, memory_region: "A".to_string(), level: 0,
        });
        sched.push(ScheduleEntry {
            operation_id: 1, block_id: 0, phase: 0, original_proc_id: 1,
            op_type: OpType::Read, address: 0, memory_region: "A".to_string(), level: 0,
        });
        sched.push(ScheduleEntry {
            operation_id: 2, block_id: 1, phase: 0, original_proc_id: 2,
            op_type: OpType::Write, address: 5, memory_region: "A".to_string(), level: 0,
        });
        sched.push(ScheduleEntry {
            operation_id: 3, block_id: 0, phase: 0, original_proc_id: 3,
            op_type: OpType::Write, address: 1, memory_region: "A".to_string(), level: 0,
        });

        let reordered = sched.reorder_for_locality();
        assert_eq!(reordered.total_ops(), 4);
        // Block 0 entries should come first (sorted by block_id)
        assert_eq!(reordered.entries[0].block_id, 0);
        assert_eq!(reordered.entries[1].block_id, 0);
        assert_eq!(reordered.entries[2].block_id, 1);
        assert_eq!(reordered.entries[3].block_id, 1);
    }

    #[test]
    fn test_schedule_analysis() {
        let mut sched = Schedule::new(4, 2, 4);
        sched.push(make_entry(0, 0, 0, 0, OpType::Read));
        sched.push(make_entry(1, 0, 0, 1, OpType::Write));
        sched.push(make_entry(2, 1, 1, 2, OpType::Read));
        sched.push(make_entry(3, 1, 1, 3, OpType::Read));

        let analysis = sched.analyze();
        assert_eq!(analysis.total_ops, 4);
        assert_eq!(analysis.reads, 3);
        assert_eq!(analysis.writes, 1);
        assert_eq!(analysis.num_phases, 2);
        assert_eq!(analysis.num_blocks, 2);
        assert!((analysis.avg_ops_per_phase - 2.0).abs() < 1e-6);
        assert!((analysis.avg_ops_per_block - 2.0).abs() < 1e-6);
        assert_eq!(analysis.reuse_count, 2); // block 0 reused once, block 1 reused once
        let s = format!("{}", analysis);
        assert!(s.contains("Schedule Analysis"));
    }

    #[test]
    fn test_schedule_analysis_empty() {
        let sched = Schedule::new(1, 0, 4);
        let analysis = sched.analyze();
        assert_eq!(analysis.total_ops, 0);
        assert_eq!(analysis.reuse_count, 0);
        assert!((analysis.block_utilization - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_merge_empty_schedules() {
        let a = Schedule::new(2, 1, 4);
        let b = Schedule::new(3, 2, 4);
        let merged = Schedule::merge(&a, &b);
        assert_eq!(merged.total_ops(), 0);
        assert_eq!(merged.num_processors, 3);
        assert_eq!(merged.num_phases, 2);
    }

    #[test]
    fn test_reorder_for_locality_empty() {
        let sched = Schedule::new(1, 0, 4);
        let reordered = sched.reorder_for_locality();
        assert_eq!(reordered.total_ops(), 0);
    }
}
