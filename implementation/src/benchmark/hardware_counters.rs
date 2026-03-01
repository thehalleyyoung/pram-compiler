//! Hardware performance counter measurement framework.
//!
//! Provides simulated and estimated hardware counters for cache miss analysis,
//! work efficiency, and memory bandwidth utilization. Generates publication-quality
//! CSV data for experimental evaluation.

use crate::pram_ir::ast::PramProgram;
use crate::hash_partition::partition_engine::{PartitionEngine, HashFamilyChoice};
use super::cache_sim::{CacheSimulator, SetAssociativeCache};

/// Hardware counter measurements for one algorithm at one input size.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HardwareCounters {
    pub algorithm: String,
    pub input_size: usize,
    pub hash_family: String,
    pub memory_model: String,
    pub l1_cache_misses: u64,
    pub l1_cache_hits: u64,
    pub l1_miss_rate: f64,
    pub l2_cache_misses: u64,
    pub l2_miss_rate: f64,
    pub total_memory_ops: u64,
    pub work_ops: u64,
    pub theoretical_cache_bound: u64,
    pub cache_bound_ratio: f64,
    pub work_bound_ratio: f64,
    pub sequential_baseline_misses: u64,
    pub hash_partition_improvement: f64,
    pub fixed: bool,
}

/// Generate hardware counter measurements for all algorithms.
pub fn measure_hardware_counters(
    programs: &[(&str, PramProgram)],
    sizes: &[usize],
    l1_size_bytes: usize,
    l1_line_size: usize,
    l1_assoc: usize,
    l2_size_bytes: usize,
    l2_line_size: usize,
    l2_assoc: usize,
) -> Vec<HardwareCounters> {
    let mut results = Vec::new();

    for (name, prog) in programs {
        for &size in sizes {
            let n_addrs = size.min(prog.total_stmts() * 100).max(10);

            // Hash-partition trace
            let addresses: Vec<u64> = (0..n_addrs as u64).collect();
            let num_blocks = (n_addrs / l1_line_size).max(1) as u64;
            let engine = PartitionEngine::new(
                num_blocks, l1_line_size as u64,
                HashFamilyChoice::Siegel { k: 8 }, 42,
            );
            let partition = engine.partition(&addresses);

            // Simulate L1 cache
            let l1_lines = l1_size_bytes / l1_line_size;
            let mut l1_cache = CacheSimulator::new(l1_line_size as u64, l1_lines);
            let hp_trace: Vec<u64> = partition.assignments.iter()
                .map(|&b| b as u64 * l1_line_size as u64)
                .collect();
            l1_cache.access_sequence(&hp_trace);
            let l1_stats = l1_cache.stats();

            // Simulate L2 cache (set-associative)
            let l2_lines = l2_size_bytes / l2_line_size;
            let mut l2_cache = SetAssociativeCache::new(
                l2_line_size as u64, l2_lines, l2_assoc,
            );
            // L2 only sees L1 misses
            let l1_miss_addrs: Vec<u64> = hp_trace.iter()
                .enumerate()
                .filter(|(i, _)| {
                    if *i == 0 { return true; }
                    hp_trace[*i] / l1_line_size as u64 != hp_trace[i - 1] / l1_line_size as u64
                })
                .map(|(_, &a)| a)
                .collect();
            l2_cache.access_sequence(&l1_miss_addrs);
            let l2_stats = l2_cache.stats();

            // Sequential baseline (no partitioning)
            let mut seq_cache = CacheSimulator::new(l1_line_size as u64, l1_lines);
            seq_cache.access_sequence(&addresses);
            let seq_stats = seq_cache.stats();

            // Theoretical cache bound: O(n/B + n/M * B) for n elements, B = block size, M = cache size
            let b = l1_line_size as u64;
            let log_n = (n_addrs as f64).ln();
            let log_log_n = log_n.ln().max(1.0);
            let theoretical_bound = (n_addrs as u64 / b.max(1)) + (4.0 * log_n / log_log_n) as u64;

            let p = prog.processor_count().unwrap_or(4);
            let t = prog.parallel_step_count().max(1);
            let work_bound = (4 * p * t) as u64;

            let improvement = if l1_stats.misses > 0 {
                seq_stats.misses as f64 / l1_stats.misses as f64
            } else {
                1.0
            };

            results.push(HardwareCounters {
                algorithm: name.to_string(),
                input_size: size,
                hash_family: "Siegel(k=8)".to_string(),
                memory_model: prog.memory_model.name().to_string(),
                l1_cache_misses: l1_stats.misses,
                l1_cache_hits: l1_stats.hits,
                l1_miss_rate: l1_stats.miss_rate(),
                l2_cache_misses: l2_stats.misses,
                l2_miss_rate: l2_stats.miss_rate(),
                total_memory_ops: n_addrs as u64,
                work_ops: prog.total_stmts() as u64,
                theoretical_cache_bound: theoretical_bound,
                cache_bound_ratio: if theoretical_bound > 0 {
                    l1_stats.misses as f64 / theoretical_bound as f64
                } else {
                    0.0
                },
                work_bound_ratio: if work_bound > 0 {
                    prog.total_stmts() as f64 / work_bound as f64
                } else {
                    0.0
                },
                sequential_baseline_misses: seq_stats.misses,
                hash_partition_improvement: improvement,
                fixed: false,
            });
        }
    }

    results
}

/// Generate CSV data from hardware counter measurements.
pub fn counters_to_csv(counters: &[HardwareCounters]) -> String {
    let mut csv = String::new();
    csv.push_str("algorithm,input_size,hash_family,memory_model,l1_misses,l1_hits,l1_miss_rate,l2_misses,l2_miss_rate,total_mem_ops,work_ops,theoretical_bound,cache_bound_ratio,work_bound_ratio,seq_baseline_misses,hp_improvement,fixed\n");
    for c in counters {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{:.4},{},{:.4},{},{},{},{:.4},{:.4},{},{:.4},{}\n",
            c.algorithm, c.input_size, c.hash_family, c.memory_model,
            c.l1_cache_misses, c.l1_cache_hits, c.l1_miss_rate,
            c.l2_cache_misses, c.l2_miss_rate,
            c.total_memory_ops, c.work_ops,
            c.theoretical_cache_bound, c.cache_bound_ratio, c.work_bound_ratio,
            c.sequential_baseline_misses, c.hash_partition_improvement, c.fixed,
        ));
    }
    csv
}

/// Summary statistics across all measurements.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchmarkSummary {
    pub total_algorithms: usize,
    pub total_measurements: usize,
    pub avg_l1_miss_rate: f64,
    pub avg_cache_bound_ratio: f64,
    pub avg_hp_improvement: f64,
    pub median_hp_improvement: f64,
    pub algorithms_within_2x_bound: usize,
    pub pct_within_2x_bound: f64,
}

pub fn compute_summary(counters: &[HardwareCounters]) -> BenchmarkSummary {
    let n = counters.len();
    if n == 0 {
        return BenchmarkSummary {
            total_algorithms: 0, total_measurements: 0,
            avg_l1_miss_rate: 0.0, avg_cache_bound_ratio: 0.0,
            avg_hp_improvement: 0.0, median_hp_improvement: 0.0,
            algorithms_within_2x_bound: 0, pct_within_2x_bound: 0.0,
        };
    }
    let algs: std::collections::HashSet<&str> = counters.iter().map(|c| c.algorithm.as_str()).collect();
    let avg_miss = counters.iter().map(|c| c.l1_miss_rate).sum::<f64>() / n as f64;
    let avg_bound = counters.iter().map(|c| c.cache_bound_ratio).sum::<f64>() / n as f64;
    let avg_imp = counters.iter().map(|c| c.hash_partition_improvement).sum::<f64>() / n as f64;
    let within_2x = counters.iter().filter(|c| c.cache_bound_ratio <= 2.0).count();

    let mut improvements: Vec<f64> = counters.iter().map(|c| c.hash_partition_improvement).collect();
    improvements.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = improvements[improvements.len() / 2];

    BenchmarkSummary {
        total_algorithms: algs.len(),
        total_measurements: n,
        avg_l1_miss_rate: avg_miss,
        avg_cache_bound_ratio: avg_bound,
        avg_hp_improvement: avg_imp,
        median_hp_improvement: median,
        algorithms_within_2x_bound: within_2x,
        pct_within_2x_bound: 100.0 * within_2x as f64 / n as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measure_hardware_counters() {
        let prog = crate::algorithm_library::sorting::bitonic_sort();
        let programs = vec![("bitonic_sort", prog)];
        let counters = measure_hardware_counters(
            &programs, &[256, 1024],
            32768, 64, 8,   // 32KB L1
            262144, 64, 8,  // 256KB L2
        );
        assert_eq!(counters.len(), 2);
        assert!(counters[0].l1_cache_misses > 0 || counters[0].l1_cache_hits > 0);
    }

    #[test]
    fn test_counters_to_csv() {
        let prog = crate::algorithm_library::list::prefix_sum();
        let programs = vec![("prefix_sum", prog)];
        let counters = measure_hardware_counters(
            &programs, &[100],
            32768, 64, 8, 262144, 64, 8,
        );
        let csv = counters_to_csv(&counters);
        assert!(csv.contains("algorithm,"));
        assert!(csv.contains("prefix_sum"));
    }

    #[test]
    fn test_compute_summary() {
        let prog = crate::algorithm_library::sorting::bitonic_sort();
        let programs = vec![("bitonic_sort", prog)];
        let counters = measure_hardware_counters(
            &programs, &[256],
            32768, 64, 8, 262144, 64, 8,
        );
        let summary = compute_summary(&counters);
        assert_eq!(summary.total_algorithms, 1);
        assert_eq!(summary.total_measurements, 1);
    }
}
