//! Cost analysis for sequential PRAM schedules.
//!
//! Analyzes total work, cache misses, and critical path length, and
//! compares against theoretical bounds from Brent's theorem.

use std::collections::HashMap;
use std::fmt;

use super::dependency_graph::DependencyGraph;
use super::locality_order::estimate_cache_misses;
use super::schedule::Schedule;

/// Complete cost report for a schedule.
#[derive(Debug, Clone)]
pub struct CostReport {
    /// Total number of operations (work).
    pub total_work: usize,
    /// Number of read operations.
    pub reads: usize,
    /// Number of write operations.
    pub writes: usize,
    /// Critical path length (longest dependency chain).
    pub critical_path: usize,
    /// Number of processors in the original parallel program.
    pub num_processors: usize,
    /// Number of parallel phases.
    pub num_phases: usize,
    /// Cache misses under LRU model.
    pub cache_misses: usize,
    /// Number of cache blocks in the model.
    pub cache_blocks: usize,
    /// Block size (elements per block).
    pub block_size: usize,
    /// Number of distinct blocks accessed.
    pub distinct_blocks: usize,
    /// Number of block transitions in the sequential order.
    pub block_transitions: usize,
    /// Work bound: expected ≤ factor * p * T.
    pub work_bound_ratio: f64,
    /// Cache miss bound: expected ≤ O(pT/B + T).
    pub miss_bound_ratio: f64,
}

impl CostReport {
    /// Whether the work bound is satisfied (ratio ≤ given factor).
    pub fn work_bound_ok(&self, factor: f64) -> bool {
        self.work_bound_ratio <= factor
    }

    /// Whether the cache miss bound is satisfied.
    pub fn miss_bound_ok(&self, factor: f64) -> bool {
        self.miss_bound_ratio <= factor
    }

    /// Parallelism: work / critical_path.
    pub fn parallelism(&self) -> f64 {
        if self.critical_path == 0 {
            return 0.0;
        }
        self.total_work as f64 / self.critical_path as f64
    }

    /// Cache miss rate: misses / total_work.
    pub fn miss_rate(&self) -> f64 {
        if self.total_work == 0 {
            return 0.0;
        }
        self.cache_misses as f64 / self.total_work as f64
    }
}

impl fmt::Display for CostReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Cost Report ===")?;
        writeln!(f, "Total work:        {}", self.total_work)?;
        writeln!(f, "  Reads:           {}", self.reads)?;
        writeln!(f, "  Writes:          {}", self.writes)?;
        writeln!(f, "Critical path:     {}", self.critical_path)?;
        writeln!(f, "Parallelism:       {:.2}", self.parallelism())?;
        writeln!(f, "Processors:        {}", self.num_processors)?;
        writeln!(f, "Phases:            {}", self.num_phases)?;
        writeln!(f, "Block size:        {}", self.block_size)?;
        writeln!(f, "Cache blocks:      {}", self.cache_blocks)?;
        writeln!(f, "Cache misses:      {}", self.cache_misses)?;
        writeln!(f, "Miss rate:         {:.4}", self.miss_rate())?;
        writeln!(f, "Distinct blocks:   {}", self.distinct_blocks)?;
        writeln!(f, "Block transitions: {}", self.block_transitions)?;
        writeln!(f, "Work bound ratio:  {:.4} (W / p*T)", self.work_bound_ratio)?;
        writeln!(f, "Miss bound ratio:  {:.4} (M / (pT/B+T))", self.miss_bound_ratio)?;
        Ok(())
    }
}

/// Analyze a schedule and produce a cost report.
///
/// `cache_blocks` is the number of blocks the cache can hold.
/// `critical_path` should come from the dependency graph analysis.
pub fn analyze_schedule(
    schedule: &Schedule,
    critical_path: usize,
    cache_blocks: usize,
) -> CostReport {
    let total_work = schedule.total_ops();
    let reads = schedule.read_count();
    let writes = schedule.write_count();
    let distinct_blocks = schedule.distinct_blocks();
    let num_processors = schedule.num_processors;
    let num_phases = schedule.num_phases;
    let block_size = schedule.block_size;

    let cache_misses = estimate_cache_misses(&schedule.entries, cache_blocks);

    let block_transitions = count_transitions(&schedule.entries);

    // Work bound: W / (p * T)
    let p_t = (num_processors.max(1) * critical_path.max(1)) as f64;
    let work_bound_ratio = total_work as f64 / p_t;

    // Cache miss bound: M / (p*T/B + T)
    let miss_bound_denom = (num_processors.max(1) as f64 * critical_path.max(1) as f64
        / block_size.max(1) as f64)
        + critical_path.max(1) as f64;
    let miss_bound_ratio = if miss_bound_denom > 0.0 {
        cache_misses as f64 / miss_bound_denom
    } else {
        0.0
    };

    CostReport {
        total_work,
        reads,
        writes,
        critical_path,
        num_processors,
        num_phases,
        cache_misses,
        cache_blocks,
        block_size,
        distinct_blocks,
        block_transitions,
        work_bound_ratio,
        miss_bound_ratio,
    }
}

/// Convenience: analyze a schedule using the dependency graph for critical path.
pub fn analyze_with_graph(
    schedule: &Schedule,
    graph: &DependencyGraph,
    cache_blocks: usize,
) -> CostReport {
    let critical_path = graph.critical_path_length();
    analyze_schedule(schedule, critical_path, cache_blocks)
}

/// Count block transitions in a schedule entry slice.
fn count_transitions(entries: &[super::schedule::ScheduleEntry]) -> usize {
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

/// Compare two schedules and return a comparison summary.
pub fn compare_schedules(
    report_a: &CostReport,
    report_b: &CostReport,
) -> ScheduleComparison {
    ScheduleComparison {
        work_diff: report_b.total_work as i64 - report_a.total_work as i64,
        miss_diff: report_b.cache_misses as i64 - report_a.cache_misses as i64,
        transition_diff: report_b.block_transitions as i64 - report_a.block_transitions as i64,
        a_miss_rate: report_a.miss_rate(),
        b_miss_rate: report_b.miss_rate(),
    }
}

/// Comparison between two schedule cost reports.
#[derive(Debug, Clone)]
pub struct ScheduleComparison {
    /// Difference in total work (B - A). Negative means B is better.
    pub work_diff: i64,
    /// Difference in cache misses (B - A). Negative means B is better.
    pub miss_diff: i64,
    /// Difference in block transitions (B - A).
    pub transition_diff: i64,
    /// Miss rate of schedule A.
    pub a_miss_rate: f64,
    /// Miss rate of schedule B.
    pub b_miss_rate: f64,
}

impl fmt::Display for ScheduleComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Schedule Comparison (B vs A) ===")?;
        writeln!(f, "Work diff:       {:+}", self.work_diff)?;
        writeln!(f, "Miss diff:       {:+}", self.miss_diff)?;
        writeln!(f, "Transition diff: {:+}", self.transition_diff)?;
        writeln!(f, "A miss rate:     {:.4}", self.a_miss_rate)?;
        writeln!(f, "B miss rate:     {:.4}", self.b_miss_rate)?;
        Ok(())
    }
}

/// Configuration for cache simulation.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Number of cache blocks.
    pub num_blocks: usize,
    /// Block size (elements per block).
    pub block_size: usize,
    /// Associativity (1 = direct mapped, num_blocks = fully associative).
    pub associativity: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            num_blocks: 64,
            block_size: 4,
            associativity: 64,
        }
    }
}

/// Detailed per-block cache analysis.
#[derive(Debug, Clone)]
pub struct CacheAnalysis {
    /// Total cache misses.
    pub total_misses: usize,
    /// Total cache hits.
    pub total_hits: usize,
    /// Misses per block_id.
    pub per_block_misses: HashMap<usize, usize>,
    /// Hits per block_id.
    pub per_block_hits: HashMap<usize, usize>,
    /// Access count per block_id.
    pub per_block_accesses: HashMap<usize, usize>,
    /// Cold (compulsory) misses.
    pub cold_misses: usize,
    /// Capacity misses (total - cold).
    pub capacity_misses: usize,
    /// Overall hit rate.
    pub hit_rate: f64,
}

impl fmt::Display for CacheAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Cache Analysis ===")?;
        writeln!(f, "Total misses:    {}", self.total_misses)?;
        writeln!(f, "Total hits:      {}", self.total_hits)?;
        writeln!(f, "Hit rate:        {:.4}", self.hit_rate)?;
        writeln!(f, "Cold misses:     {}", self.cold_misses)?;
        writeln!(f, "Capacity misses: {}", self.capacity_misses)?;
        writeln!(f, "Blocks accessed: {}", self.per_block_accesses.len())?;
        Ok(())
    }
}

/// Perform a detailed cache analysis with per-block miss counts.
pub fn detailed_cache_analysis(
    schedule: &Schedule,
    cache_config: &CacheConfig,
) -> CacheAnalysis {
    let cache_blocks = cache_config.num_blocks;
    let mut cache: Vec<usize> = Vec::with_capacity(cache_blocks);
    let mut total_misses: usize = 0;
    let mut total_hits: usize = 0;
    let mut cold_misses: usize = 0;
    let mut per_block_misses: HashMap<usize, usize> = HashMap::new();
    let mut per_block_hits: HashMap<usize, usize> = HashMap::new();
    let mut per_block_accesses: HashMap<usize, usize> = HashMap::new();
    let mut ever_seen: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for entry in &schedule.entries {
        let blk = entry.block_id;
        *per_block_accesses.entry(blk).or_insert(0) += 1;

        if let Some(pos) = cache.iter().position(|&b| b == blk) {
            // Cache hit
            total_hits += 1;
            *per_block_hits.entry(blk).or_insert(0) += 1;
            cache.remove(pos);
            cache.push(blk);
        } else {
            // Cache miss
            total_misses += 1;
            *per_block_misses.entry(blk).or_insert(0) += 1;
            if !ever_seen.contains(&blk) {
                cold_misses += 1;
                ever_seen.insert(blk);
            }
            if cache.len() >= cache_blocks {
                cache.remove(0);
            }
            cache.push(blk);
        }
    }

    let total = total_hits + total_misses;
    let hit_rate = if total > 0 {
        total_hits as f64 / total as f64
    } else {
        0.0
    };

    CacheAnalysis {
        total_misses,
        total_hits,
        per_block_misses,
        per_block_hits,
        per_block_accesses,
        cold_misses,
        capacity_misses: total_misses.saturating_sub(cold_misses),
        hit_rate,
    }
}

/// Predict execution time based on schedule, clock speed, and cache miss penalty.
///
/// Simple model: each operation takes 1 cycle for a hit, and
/// `cache_miss_penalty` extra cycles for a miss.
/// Total time = (hits * 1 + misses * (1 + penalty)) / clock_speed
pub fn predict_execution_time(
    schedule: &Schedule,
    clock_speed_ghz: f64,
    cache_miss_penalty_cycles: usize,
    cache_config: &CacheConfig,
) -> f64 {
    let analysis = detailed_cache_analysis(schedule, cache_config);
    let hit_cycles = analysis.total_hits as f64;
    let miss_cycles = analysis.total_misses as f64 * (1.0 + cache_miss_penalty_cycles as f64);
    let total_cycles = hit_cycles + miss_cycles;
    let clock_hz = clock_speed_ghz * 1e9;
    if clock_hz > 0.0 {
        total_cycles / clock_hz
    } else {
        0.0
    }
}

/// Information about a bottleneck in the schedule.
#[derive(Debug, Clone)]
pub struct BottleneckReport {
    /// Block with the most cache misses.
    pub worst_block: Option<usize>,
    /// Number of misses in the worst block.
    pub worst_block_misses: usize,
    /// Phase with the most operations (potential serialization bottleneck).
    pub busiest_phase: Option<usize>,
    /// Operations in the busiest phase.
    pub busiest_phase_ops: usize,
    /// Level with the most operations.
    pub widest_level: usize,
    /// Operations at the widest level.
    pub widest_level_ops: usize,
    /// Total block transitions.
    pub total_transitions: usize,
}

impl fmt::Display for BottleneckReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Bottleneck Report ===")?;
        if let Some(blk) = self.worst_block {
            writeln!(f, "Worst block:     {} ({} misses)", blk, self.worst_block_misses)?;
        }
        if let Some(ph) = self.busiest_phase {
            writeln!(f, "Busiest phase:   {} ({} ops)", ph, self.busiest_phase_ops)?;
        }
        writeln!(f, "Widest level:    {} ({} ops)", self.widest_level, self.widest_level_ops)?;
        writeln!(f, "Transitions:     {}", self.total_transitions)?;
        Ok(())
    }
}

/// Identify bottlenecks in a schedule.
pub fn bottleneck_analysis(
    schedule: &Schedule,
    cache_config: &CacheConfig,
) -> BottleneckReport {
    let cache_analysis = detailed_cache_analysis(schedule, cache_config);

    let worst_block = cache_analysis
        .per_block_misses
        .iter()
        .max_by_key(|(_, &v)| v)
        .map(|(&k, _)| k);
    let worst_block_misses = worst_block
        .and_then(|b| cache_analysis.per_block_misses.get(&b))
        .copied()
        .unwrap_or(0);

    let ops_per_phase = schedule.ops_per_phase();
    let busiest_phase = ops_per_phase
        .iter()
        .max_by_key(|(_, &v)| v)
        .map(|(&k, _)| k);
    let busiest_phase_ops = busiest_phase
        .and_then(|p| ops_per_phase.get(&p))
        .copied()
        .unwrap_or(0);

    // Level width
    let mut level_ops: HashMap<usize, usize> = HashMap::new();
    for entry in &schedule.entries {
        *level_ops.entry(entry.level).or_insert(0) += 1;
    }
    let (widest_level, widest_level_ops) = level_ops
        .iter()
        .max_by_key(|(_, &v)| v)
        .map(|(&k, &v)| (k, v))
        .unwrap_or((0, 0));

    let total_transitions = count_transitions(&schedule.entries);

    BottleneckReport {
        worst_block,
        worst_block_misses,
        busiest_phase,
        busiest_phase_ops,
        widest_level,
        widest_level_ops,
        total_transitions,
    }
}

/// Run sensitivity analysis by varying cache parameters.
///
/// For each (num_blocks, block_size) pair in the parameter ranges,
/// produce a CostReport.
pub fn sensitivity_analysis(
    schedule: &Schedule,
    critical_path: usize,
    cache_block_range: &[usize],
) -> Vec<CostReport> {
    cache_block_range
        .iter()
        .map(|&cb| analyze_schedule(schedule, critical_path, cb))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brent_scheduler::dependency_graph::{DependencyGraph, OperationNode};
    use crate::brent_scheduler::schedule::{OpType, Schedule, ScheduleEntry};

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

    fn make_entry(op_id: usize, block: usize, op: OpType) -> ScheduleEntry {
        ScheduleEntry {
            operation_id: op_id,
            block_id: block,
            phase: 0,
            original_proc_id: op_id,
            op_type: op,
            address: block * 4 + op_id % 4,
            memory_region: "A".to_string(),
            level: 0,
        }
    }

    #[test]
    fn test_basic_cost_report() {
        let mut sched = Schedule::new(4, 1, 4);
        sched.push(make_entry(0, 0, OpType::Read));
        sched.push(make_entry(1, 0, OpType::Read));
        sched.push(make_entry(2, 1, OpType::Write));
        sched.push(make_entry(3, 1, OpType::Write));

        let report = analyze_schedule(&sched, 2, 4);

        assert_eq!(report.total_work, 4);
        assert_eq!(report.reads, 2);
        assert_eq!(report.writes, 2);
        assert_eq!(report.critical_path, 2);
        assert_eq!(report.distinct_blocks, 2);
    }

    #[test]
    fn test_work_bound_ratio() {
        let mut sched = Schedule::new(4, 1, 4);
        for i in 0..8 {
            sched.push(make_entry(i, i % 2, OpType::Read));
        }

        // W = 8, p = 4, T = 2 → ratio = 8/8 = 1.0
        let report = analyze_schedule(&sched, 2, 4);
        assert!((report.work_bound_ratio - 1.0).abs() < 1e-6);
        assert!(report.work_bound_ok(1.0));
    }

    #[test]
    fn test_cache_misses_in_report() {
        let mut sched = Schedule::new(2, 1, 4);
        // All same block → 1 miss
        sched.push(make_entry(0, 0, OpType::Read));
        sched.push(make_entry(1, 0, OpType::Read));
        sched.push(make_entry(2, 0, OpType::Write));

        let report = analyze_schedule(&sched, 1, 4);
        assert_eq!(report.cache_misses, 1);
        assert!((report.miss_rate() - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_block_transitions_in_report() {
        let mut sched = Schedule::new(4, 1, 4);
        sched.push(make_entry(0, 0, OpType::Read));
        sched.push(make_entry(1, 1, OpType::Read));
        sched.push(make_entry(2, 0, OpType::Write));

        let report = analyze_schedule(&sched, 1, 4);
        assert_eq!(report.block_transitions, 2); // 0→1, 1→0
    }

    #[test]
    fn test_parallelism() {
        let mut sched = Schedule::new(4, 1, 4);
        for i in 0..8 {
            sched.push(make_entry(i, i, OpType::Read));
        }
        let report = analyze_schedule(&sched, 1, 4);
        assert!((report.parallelism() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_schedule_report() {
        let sched = Schedule::new(4, 1, 4);
        let report = analyze_schedule(&sched, 1, 4);
        assert_eq!(report.total_work, 0);
        assert_eq!(report.cache_misses, 0);
        assert!((report.miss_rate() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compare_schedules() {
        let mut sched_a = Schedule::new(4, 1, 4);
        sched_a.push(make_entry(0, 0, OpType::Read));
        sched_a.push(make_entry(1, 1, OpType::Read));
        sched_a.push(make_entry(2, 0, OpType::Read));

        let mut sched_b = Schedule::new(4, 1, 4);
        sched_b.push(make_entry(0, 0, OpType::Read));
        sched_b.push(make_entry(2, 0, OpType::Read));
        sched_b.push(make_entry(1, 1, OpType::Read));

        let report_a = analyze_schedule(&sched_a, 1, 2);
        let report_b = analyze_schedule(&sched_b, 1, 2);

        let cmp = compare_schedules(&report_a, &report_b);
        assert_eq!(cmp.work_diff, 0); // Same total work
        // B should have fewer misses (grouped block 0 ops)
        assert!(cmp.miss_diff <= 0);
    }

    #[test]
    fn test_analyze_with_graph() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
            make_op(2, 1, 0, 2, OpType::Write, 4),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);

        let seq = crate::brent_scheduler::work_optimal::extract_schedule(&dg);
        let schedule = seq.to_schedule(4, 1);

        let report = analyze_with_graph(&schedule, &dg, 4);
        assert_eq!(report.total_work, 3);
        assert_eq!(report.critical_path, 2); // W→R chain
    }

    #[test]
    fn test_display_report() {
        let mut sched = Schedule::new(2, 1, 4);
        sched.push(make_entry(0, 0, OpType::Read));
        let report = analyze_schedule(&sched, 1, 2);
        let s = format!("{}", report);
        assert!(s.contains("Total work"));
        assert!(s.contains("Cache misses"));
    }

    #[test]
    fn test_display_comparison() {
        let report_a = CostReport {
            total_work: 10,
            reads: 5,
            writes: 5,
            critical_path: 3,
            num_processors: 4,
            num_phases: 1,
            cache_misses: 5,
            cache_blocks: 4,
            block_size: 4,
            distinct_blocks: 3,
            block_transitions: 4,
            work_bound_ratio: 0.83,
            miss_bound_ratio: 1.0,
        };
        let report_b = CostReport {
            total_work: 10,
            reads: 5,
            writes: 5,
            critical_path: 3,
            num_processors: 4,
            num_phases: 1,
            cache_misses: 3,
            cache_blocks: 4,
            block_size: 4,
            distinct_blocks: 3,
            block_transitions: 2,
            work_bound_ratio: 0.83,
            miss_bound_ratio: 0.6,
        };
        let cmp = compare_schedules(&report_a, &report_b);
        let s = format!("{}", cmp);
        assert!(s.contains("Miss diff"));
    }

    #[test]
    fn test_detailed_cache_analysis_basic() {
        let mut sched = Schedule::new(2, 1, 4);
        sched.push(make_entry(0, 0, OpType::Read));
        sched.push(make_entry(1, 1, OpType::Read));
        sched.push(make_entry(2, 0, OpType::Write));
        sched.push(make_entry(3, 1, OpType::Write));

        let config = CacheConfig { num_blocks: 4, block_size: 4, associativity: 4 };
        let analysis = detailed_cache_analysis(&sched, &config);

        assert_eq!(analysis.cold_misses, 2); // first access to block 0 and block 1
        assert_eq!(analysis.total_misses, 2);
        assert_eq!(analysis.total_hits, 2);
        assert!((analysis.hit_rate - 0.5).abs() < 1e-6);
        assert_eq!(analysis.per_block_accesses[&0], 2);
        assert_eq!(analysis.per_block_accesses[&1], 2);
        let s = format!("{}", analysis);
        assert!(s.contains("Cache Analysis"));
    }

    #[test]
    fn test_detailed_cache_analysis_all_misses() {
        let mut sched = Schedule::new(1, 1, 4);
        for i in 0..10 {
            sched.push(make_entry(i, i, OpType::Read));
        }

        let config = CacheConfig { num_blocks: 2, block_size: 4, associativity: 2 };
        let analysis = detailed_cache_analysis(&sched, &config);

        assert_eq!(analysis.total_misses, 10);
        assert_eq!(analysis.cold_misses, 10);
        assert_eq!(analysis.total_hits, 0);
    }

    #[test]
    fn test_predict_execution_time() {
        let mut sched = Schedule::new(2, 1, 4);
        sched.push(make_entry(0, 0, OpType::Read));
        sched.push(make_entry(1, 0, OpType::Read));

        let config = CacheConfig::default();
        let time = predict_execution_time(&sched, 1.0, 100, &config);
        assert!(time > 0.0);
    }

    #[test]
    fn test_predict_execution_time_empty() {
        let sched = Schedule::new(1, 1, 4);
        let config = CacheConfig::default();
        let time = predict_execution_time(&sched, 1.0, 100, &config);
        assert!((time - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_bottleneck_analysis() {
        let mut sched = Schedule::new(4, 2, 4);
        // Phase 0: many ops on block 0
        for i in 0..5 {
            sched.push(make_entry(i, 0, OpType::Read));
        }
        // Phase 1: few ops on block 1
        for i in 5..7 {
            sched.push(ScheduleEntry {
                operation_id: i,
                block_id: 1,
                phase: 1,
                original_proc_id: i,
                op_type: OpType::Write,
                address: 4 + i % 4,
                memory_region: "A".to_string(),
                level: 0,
            });
        }

        let config = CacheConfig { num_blocks: 4, block_size: 4, associativity: 4 };
        let report = bottleneck_analysis(&sched, &config);

        assert!(report.busiest_phase.is_some());
        assert_eq!(report.busiest_phase_ops, 5);
        let s = format!("{}", report);
        assert!(s.contains("Bottleneck Report"));
    }

    #[test]
    fn test_bottleneck_analysis_empty() {
        let sched = Schedule::new(1, 1, 4);
        let config = CacheConfig::default();
        let report = bottleneck_analysis(&sched, &config);
        assert!(report.worst_block.is_none());
        assert!(report.busiest_phase.is_none());
    }

    #[test]
    fn test_sensitivity_analysis() {
        let mut sched = Schedule::new(4, 1, 4);
        for i in 0..8 {
            sched.push(make_entry(i, i % 4, OpType::Read));
        }

        let reports = sensitivity_analysis(&sched, 2, &[1, 2, 4, 8, 16]);
        assert_eq!(reports.len(), 5);

        // More cache blocks → fewer misses (monotonic for LRU)
        for i in 1..reports.len() {
            assert!(reports[i].cache_misses <= reports[i - 1].cache_misses);
        }
    }

    #[test]
    fn test_sensitivity_analysis_empty() {
        let sched = Schedule::new(1, 1, 4);
        let reports = sensitivity_analysis(&sched, 1, &[1, 2, 4]);
        assert_eq!(reports.len(), 3);
        for r in &reports {
            assert_eq!(r.total_work, 0);
        }
    }
}
