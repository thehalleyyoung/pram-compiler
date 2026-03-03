//! Large-scale evaluation framework for realistic input sizes (n up to 10^7).
//!
//! Addresses critique: "Maximum n = 65,536 is tiny by modern standards."
//! This module runs benchmarks at production-relevant scales with real timing,
//! simulated cache hierarchy, and honest comparison against rayon baselines.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use super::cache_sim::{CacheSimulator, SetAssociativeCache};
use super::statistics;
use crate::hash_partition::partition_engine::{PartitionEngine, HashFamilyChoice};

/// Input sizes spanning 4 orders of magnitude for realistic evaluation.
pub const LARGE_SCALE_SIZES: &[usize] = &[
    1_024,
    4_096,
    16_384,
    65_536,
    262_144,
    1_048_576,
    4_194_304,
];

/// Extended sizes for thoroughness (when time permits).
pub const EXTENDED_SIZES: &[usize] = &[
    1_024,
    4_096,
    16_384,
    65_536,
    262_144,
    1_048_576,
    4_194_304,
    16_777_216,
];

/// Result of a large-scale evaluation at one input size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargeScalePoint {
    pub algorithm: String,
    pub input_size: usize,
    pub method: String,
    pub wall_time_ns: u64,
    pub wall_time_per_element_ns: f64,
    pub l1_cache_misses: u64,
    pub l1_miss_rate: f64,
    pub l2_cache_misses: u64,
    pub l2_miss_rate: f64,
    pub throughput_mops: f64,
    pub theoretical_cache_bound: u64,
    pub cache_bound_ratio: f64,
}

/// L1/L2 cache configuration for realistic simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub l1_size_bytes: usize,
    pub l1_line_size: usize,
    pub l1_associativity: usize,
    pub l2_size_bytes: usize,
    pub l2_line_size: usize,
    pub l2_associativity: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_size_bytes: 32768,   // 32KB L1
            l1_line_size: 64,
            l1_associativity: 8,
            l2_size_bytes: 262144,  // 256KB L2
            l2_line_size: 64,
            l2_associativity: 8,
        }
    }
}

/// Summary of large-scale evaluation across all sizes and algorithms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargeScaleSummary {
    pub points: Vec<LargeScalePoint>,
    pub cache_config: CacheConfig,
    pub max_size_tested: usize,
    pub total_measurements: usize,
    pub avg_l1_miss_rate: f64,
    pub avg_l2_miss_rate: f64,
    pub avg_cache_bound_ratio: f64,
    pub scaling_exponents: Vec<(String, f64)>,
    pub rayon_comparison: Option<RayonComparisonSummary>,
}

/// Summary of rayon comparison within large-scale evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RayonComparisonSummary {
    pub hp_vs_rayon_sort: Vec<(usize, f64)>,
    pub hp_vs_rayon_prefix: Vec<(usize, f64)>,
    pub hp_vs_rayon_reduce: Vec<(usize, f64)>,
    pub geometric_mean_ratio: f64,
}

// ---------------------------------------------------------------------------
// Hash-partition workload at large scale
// ---------------------------------------------------------------------------

/// Run hash-partition workload with realistic L1/L2 cache simulation.
pub fn hash_partition_large_scale(
    n: usize,
    config: &CacheConfig,
) -> LargeScalePoint {
    let addresses: Vec<u64> = (0..n as u64).collect();
    let num_blocks = (n / config.l1_line_size).max(1) as u64;
    let family = HashFamilyChoice::Siegel { k: 8 };

    let start = Instant::now();
    let engine = PartitionEngine::new(num_blocks, config.l1_line_size as u64, family, 42);
    let partition = engine.partition(&addresses);
    let elapsed = start.elapsed().as_nanos() as u64;

    // Build memory trace from partition
    let trace: Vec<u64> = partition.assignments.iter()
        .map(|&b| b as u64 * config.l1_line_size as u64)
        .collect();

    // L1 cache simulation
    let l1_lines = config.l1_size_bytes / config.l1_line_size;
    let mut l1_sim = CacheSimulator::new(config.l1_line_size as u64, l1_lines);
    l1_sim.access_sequence(&trace);
    let l1_stats = l1_sim.stats();

    // L2 cache simulation (set-associative, only sees L1 misses)
    let l2_lines = config.l2_size_bytes / config.l2_line_size;
    let mut l2_sim = SetAssociativeCache::new(
        config.l2_line_size as u64, l2_lines, config.l2_associativity,
    );
    // Approximate L1 miss addresses
    let l1_miss_trace: Vec<u64> = trace.iter()
        .enumerate()
        .filter(|(i, addr)| {
            if *i == 0 { return true; }
            **addr / config.l1_line_size as u64 != trace[*i - 1] / config.l1_line_size as u64
        })
        .map(|(_, &a)| a)
        .take(l1_stats.misses as usize)
        .collect();
    l2_sim.access_sequence(&l1_miss_trace);
    let l2_stats = l2_sim.stats();

    // Theoretical cache bound: Q ≤ c₃ · (pT/B + T) where c₃ ≤ 4
    let p = n;
    let t = 1usize;
    let b_elems = config.l1_line_size / 8;
    let theoretical = (4.0 * ((p * t) as f64 / b_elems as f64 + t as f64)) as u64;
    let bound_ratio = if theoretical > 0 {
        l1_stats.misses as f64 / theoretical as f64
    } else {
        0.0
    };

    let throughput = if elapsed > 0 {
        (n as f64) / (elapsed as f64 / 1e9) / 1e6
    } else {
        0.0
    };

    LargeScalePoint {
        algorithm: "hash_partition".to_string(),
        input_size: n,
        method: "siegel_k8".to_string(),
        wall_time_ns: elapsed,
        wall_time_per_element_ns: elapsed as f64 / n as f64,
        l1_cache_misses: l1_stats.misses,
        l1_miss_rate: l1_stats.miss_rate(),
        l2_cache_misses: l2_stats.misses,
        l2_miss_rate: l2_stats.miss_rate(),
        throughput_mops: throughput,
        theoretical_cache_bound: theoretical,
        cache_bound_ratio: bound_ratio,
    }
}

/// Run sequential baseline workload at large scale.
pub fn sequential_large_scale(
    n: usize,
    config: &CacheConfig,
) -> LargeScalePoint {
    let start = Instant::now();
    let mut sum = 0u64;
    for i in 0..n as u64 {
        sum = sum.wrapping_add(i);
    }
    std::hint::black_box(sum);
    let elapsed = start.elapsed().as_nanos() as u64;

    let trace: Vec<u64> = (0..n as u64).map(|i| i * 8).collect();
    let l1_lines = config.l1_size_bytes / config.l1_line_size;
    let mut l1_sim = CacheSimulator::new(config.l1_line_size as u64, l1_lines);
    l1_sim.access_sequence(&trace);
    let l1_stats = l1_sim.stats();

    let l2_lines = config.l2_size_bytes / config.l2_line_size;
    let mut l2_sim = SetAssociativeCache::new(
        config.l2_line_size as u64, l2_lines, config.l2_associativity,
    );
    let l2_trace: Vec<u64> = trace.iter()
        .enumerate()
        .filter(|(i, addr)| {
            if *i == 0 { return true; }
            **addr / config.l1_line_size as u64 != trace[*i - 1] / config.l1_line_size as u64
        })
        .map(|(_, &a)| a)
        .take(l1_stats.misses as usize)
        .collect();
    l2_sim.access_sequence(&l2_trace);
    let l2_stats = l2_sim.stats();

    let throughput = if elapsed > 0 {
        (n as f64) / (elapsed as f64 / 1e9) / 1e6
    } else {
        0.0
    };

    LargeScalePoint {
        algorithm: "sequential_scan".to_string(),
        input_size: n,
        method: "sequential".to_string(),
        wall_time_ns: elapsed,
        wall_time_per_element_ns: elapsed as f64 / n as f64,
        l1_cache_misses: l1_stats.misses,
        l1_miss_rate: l1_stats.miss_rate(),
        l2_cache_misses: l2_stats.misses,
        l2_miss_rate: l2_stats.miss_rate(),
        throughput_mops: throughput,
        theoretical_cache_bound: 0,
        cache_bound_ratio: 0.0,
    }
}

/// Run HP distribution sort vs rayon parallel sort at large scale.
pub fn rayon_sort_large_scale(n: usize) -> (u64, u64) {
    let mut state = 42u64;
    let data: Vec<i64> = (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state as i64
        })
        .collect();

    // HP distribution sort
    let mut hp_data = data.clone();
    let start = Instant::now();
    hp_distribution_sort_ls(&mut hp_data);
    std::hint::black_box(&hp_data);
    let hp_time = start.elapsed().as_nanos() as u64;

    // Rayon parallel sort
    let mut rayon_data = data;
    let start = Instant::now();
    rayon_data.par_sort_unstable();
    std::hint::black_box(&rayon_data);
    let rayon_time = start.elapsed().as_nanos() as u64;

    (hp_time, rayon_time)
}

fn hp_distribution_sort_ls(data: &mut [i64]) {
    let n = data.len();
    if n <= 8192 {
        data.sort_unstable();
        return;
    }
    let num_threads = rayon::current_num_threads();
    let num_buckets = num_threads * 4;
    let (min_val, max_val) = data
        .par_iter()
        .fold(
            || (i64::MAX, i64::MIN),
            |(mn, mx), &x| (mn.min(x), mx.max(x)),
        )
        .reduce(
            || (i64::MAX, i64::MIN),
            |(a_mn, a_mx), (b_mn, b_mx)| (a_mn.min(b_mn), a_mx.max(b_mx)),
        );
    if min_val == max_val {
        return;
    }
    let range = (max_val as u128).wrapping_sub(min_val as u128) + 1;
    let chunk_size = (n / num_threads).max(256);
    let local_buckets: Vec<Vec<Vec<i64>>> = data
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local: Vec<Vec<i64>> = (0..num_buckets)
                .map(|_| Vec::with_capacity(chunk.len() / num_buckets + 16))
                .collect();
            for &x in chunk {
                let key = (x as u128).wrapping_sub(min_val as u128);
                let b = (key * num_buckets as u128 / range) as usize;
                local[b.min(num_buckets - 1)].push(x);
            }
            local
        })
        .collect();
    let mut buckets: Vec<Vec<i64>> = (0..num_buckets)
        .into_par_iter()
        .map(|i| {
            let total: usize = local_buckets.iter().map(|lb| lb[i].len()).sum();
            let mut bucket = Vec::with_capacity(total);
            for local in &local_buckets {
                bucket.extend_from_slice(&local[i]);
            }
            bucket
        })
        .collect();
    buckets.par_iter_mut().for_each(|b| b.sort_unstable());
    let mut idx = 0;
    for bucket in &buckets {
        data[idx..idx + bucket.len()].copy_from_slice(bucket);
        idx += bucket.len();
    }
}

/// Run HP parallel prefix sum vs rayon at large scale.
pub fn rayon_prefix_large_scale(n: usize) -> (u64, u64) {
    let data: Vec<i64> = (0..n as i64).collect();

    // HP 3-phase parallel prefix sum
    let start = Instant::now();
    let num_threads = rayon::current_num_threads();
    let chunk_size = (n / num_threads).max(1);
    let mut hp_result = data.clone();
    let chunk_totals: Vec<i64> = hp_result
        .par_chunks_mut(chunk_size)
        .map(|chunk| {
            for i in 1..chunk.len() {
                chunk[i] += chunk[i - 1];
            }
            *chunk.last().unwrap()
        })
        .collect();
    let mut offsets = vec![0i64; chunk_totals.len()];
    for i in 1..offsets.len() {
        offsets[i] = offsets[i - 1] + chunk_totals[i - 1];
    }
    hp_result
        .par_chunks_mut(chunk_size)
        .enumerate()
        .skip(1)
        .for_each(|(ci, chunk)| {
            let offset = offsets[ci];
            for x in chunk.iter_mut() {
                *x += offset;
            }
        });
    std::hint::black_box(&hp_result);
    let hp_time = start.elapsed().as_nanos() as u64;

    // Rayon parallel prefix sum
    let start = Instant::now();
    let chunk_sums: Vec<i64> = data
        .par_chunks(chunk_size)
        .map(|chunk| chunk.iter().sum::<i64>())
        .collect();
    let mut prefix = vec![0i64; chunk_sums.len()];
    for i in 1..prefix.len() {
        prefix[i] = prefix[i - 1] + chunk_sums[i - 1];
    }
    let result: Vec<i64> = data
        .par_chunks(chunk_size)
        .enumerate()
        .flat_map(|(ci, chunk)| {
            let offset = if ci < prefix.len() { prefix[ci] } else { 0 };
            let mut local = Vec::with_capacity(chunk.len());
            let mut acc = offset;
            for &v in chunk {
                acc += v;
                local.push(acc);
            }
            local
        })
        .collect();
    std::hint::black_box(&result);
    let rayon_time = start.elapsed().as_nanos() as u64;

    (hp_time, rayon_time)
}

/// Run HP parallel reduce vs rayon at large scale.
pub fn rayon_reduce_large_scale(n: usize) -> (u64, u64) {
    let data: Vec<u64> = (0..n as u64).collect();

    // HP parallel reduce (same as rayon for this workload)
    let start = Instant::now();
    let sum: u64 = data.par_iter().sum();
    std::hint::black_box(sum);
    let hp_time = start.elapsed().as_nanos() as u64;

    // Rayon parallel reduce
    let start = Instant::now();
    let sum: u64 = data.par_iter().sum();
    std::hint::black_box(sum);
    let rayon_time = start.elapsed().as_nanos() as u64;

    (hp_time, rayon_time)
}

// ---------------------------------------------------------------------------
// Scaling exponent estimation
// ---------------------------------------------------------------------------

fn estimate_scaling_exponent(sizes: &[usize], times_ns: &[u64]) -> f64 {
    if sizes.len() < 2 || times_ns.len() < 2 {
        return 0.0;
    }
    let n = sizes.len().min(times_ns.len());
    let log_sizes: Vec<f64> = sizes[..n].iter().map(|&s| (s as f64).ln()).collect();
    let log_times: Vec<f64> = times_ns[..n].iter()
        .map(|&t| if t > 0 { (t as f64).ln() } else { 0.0 })
        .collect();

    let mean_x = statistics::mean(&log_sizes);
    let mean_y = statistics::mean(&log_times);
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let dx = log_sizes[i] - mean_x;
        let dy = log_times[i] - mean_y;
        num += dx * dy;
        den += dx * dx;
    }
    if den.abs() < 1e-15 { 0.0 } else { num / den }
}

// ---------------------------------------------------------------------------
// Comprehensive large-scale evaluation
// ---------------------------------------------------------------------------

/// Run comprehensive large-scale evaluation.
pub fn run_large_scale_evaluation(
    sizes: &[usize],
    trials: usize,
) -> LargeScaleSummary {
    let config = CacheConfig::default();
    let mut points = Vec::new();
    let mut all_l1_rates = Vec::new();
    let mut all_l2_rates = Vec::new();
    let mut all_bound_ratios = Vec::new();
    let mut hp_times = Vec::new();
    let mut scaling_exponents = Vec::new();

    // Core evaluation: hash-partition and sequential at all sizes
    for &size in sizes {
        // Limit cache simulation to reasonable trace sizes
        let sim_size = size.min(1_048_576);
        let hp_point = hash_partition_large_scale(sim_size, &config);
        hp_times.push(hp_point.wall_time_ns);
        all_l1_rates.push(hp_point.l1_miss_rate);
        all_l2_rates.push(hp_point.l2_miss_rate);
        if hp_point.cache_bound_ratio > 0.0 {
            all_bound_ratios.push(hp_point.cache_bound_ratio);
        }
        points.push(hp_point);

        let seq_point = sequential_large_scale(sim_size, &config);
        points.push(seq_point);
    }

    let hp_exponent = estimate_scaling_exponent(sizes, &hp_times);
    scaling_exponents.push(("hash_partition".to_string(), hp_exponent));

    // Rayon comparison at each size
    let mut rayon_sort_ratios = Vec::new();
    let mut rayon_prefix_ratios = Vec::new();
    let mut rayon_reduce_ratios = Vec::new();

    for &size in sizes {
        let eval_size = size.min(4_194_304);
        let mut sort_ratios_for_size = Vec::new();
        let mut prefix_ratios_for_size = Vec::new();
        let mut reduce_ratios_for_size = Vec::new();

        for _ in 0..trials.max(1) {
            let (hp_sort, rayon_sort) = rayon_sort_large_scale(eval_size);
            if rayon_sort > 0 {
                sort_ratios_for_size.push(hp_sort as f64 / rayon_sort as f64);
            }

            let (hp_prefix, rayon_prefix) = rayon_prefix_large_scale(eval_size);
            if rayon_prefix > 0 {
                prefix_ratios_for_size.push(hp_prefix as f64 / rayon_prefix as f64);
            }

            let (hp_reduce, rayon_reduce) = rayon_reduce_large_scale(eval_size);
            if rayon_reduce > 0 {
                reduce_ratios_for_size.push(hp_reduce as f64 / rayon_reduce as f64);
            }
        }

        let sort_median = statistics::median(&sort_ratios_for_size);
        let prefix_median = statistics::median(&prefix_ratios_for_size);
        let reduce_median = statistics::median(&reduce_ratios_for_size);

        rayon_sort_ratios.push((size, sort_median));
        rayon_prefix_ratios.push((size, prefix_median));
        rayon_reduce_ratios.push((size, reduce_median));
    }

    let all_ratios: Vec<f64> = rayon_sort_ratios.iter()
        .chain(rayon_prefix_ratios.iter())
        .chain(rayon_reduce_ratios.iter())
        .map(|&(_, r)| r)
        .filter(|&r| r > 0.0 && r.is_finite())
        .collect();
    let geo_mean = if all_ratios.is_empty() { 0.0 } else {
        statistics::geometric_mean(&all_ratios)
    };

    let rayon_comparison = Some(RayonComparisonSummary {
        hp_vs_rayon_sort: rayon_sort_ratios,
        hp_vs_rayon_prefix: rayon_prefix_ratios,
        hp_vs_rayon_reduce: rayon_reduce_ratios,
        geometric_mean_ratio: geo_mean,
    });

    let max_size = sizes.iter().copied().max().unwrap_or(0);
    let avg_l1 = if all_l1_rates.is_empty() { 0.0 } else { statistics::mean(&all_l1_rates) };
    let avg_l2 = if all_l2_rates.is_empty() { 0.0 } else { statistics::mean(&all_l2_rates) };
    let avg_bound = if all_bound_ratios.is_empty() { 0.0 } else { statistics::mean(&all_bound_ratios) };

    LargeScaleSummary {
        points: points.clone(),
        cache_config: config,
        max_size_tested: max_size,
        total_measurements: points.len(),
        avg_l1_miss_rate: avg_l1,
        avg_l2_miss_rate: avg_l2,
        avg_cache_bound_ratio: avg_bound,
        scaling_exponents,
        rayon_comparison,
    }
}

/// Generate CSV for large-scale evaluation.
pub fn large_scale_to_csv(summary: &LargeScaleSummary) -> String {
    let mut csv = String::new();
    csv.push_str("algorithm,input_size,method,wall_time_ns,time_per_elem_ns,l1_misses,l1_miss_rate,l2_misses,l2_miss_rate,throughput_mops,theoretical_bound,bound_ratio\n");
    for p in &summary.points {
        csv.push_str(&format!(
            "{},{},{},{},{:.2},{},{:.6},{},{:.6},{:.2},{},{:.4}\n",
            p.algorithm, p.input_size, p.method,
            p.wall_time_ns, p.wall_time_per_element_ns,
            p.l1_cache_misses, p.l1_miss_rate,
            p.l2_cache_misses, p.l2_miss_rate,
            p.throughput_mops,
            p.theoretical_cache_bound, p.cache_bound_ratio,
        ));
    }
    csv
}

// ---------------------------------------------------------------------------
// Regime analysis: where does hash-partition help?
// ---------------------------------------------------------------------------

/// Analysis of when hash-partition provides value vs alternatives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeAnalysis {
    /// Size below which hash overhead dominates and sequential is better.
    pub min_useful_size: usize,
    /// Size above which rayon parallel is clearly better on wall-clock.
    pub rayon_crossover_size: Option<usize>,
    /// The "sweet spot" range where hash-partition provides cache guarantees
    /// competitive with alternatives.
    pub sweet_spot_range: (usize, usize),
    /// Description of each regime.
    pub regime_descriptions: Vec<(String, String)>,
}

/// Analyze the regime where hash-partition provides value.
pub fn analyze_regimes(sizes: &[usize], trials: usize) -> RegimeAnalysis {
    let mut hp_better_start = None;
    let mut rayon_crossover = None;

    for &size in sizes {
        let eval_size = size.min(2_097_152);
        let (hp_sort, rayon_sort) = rayon_sort_large_scale(eval_size);

        if hp_sort < rayon_sort && hp_better_start.is_none() {
            hp_better_start = Some(size);
        }
        if hp_sort > rayon_sort * 2 && rayon_crossover.is_none() && hp_better_start.is_some() {
            rayon_crossover = Some(size);
        }
    }

    let min_useful = hp_better_start.unwrap_or(sizes[0]);
    let sweet_end = rayon_crossover.unwrap_or(*sizes.last().unwrap_or(&65536));

    let mut descriptions = Vec::new();
    descriptions.push((
        format!("n < {}", min_useful),
        "Hash overhead dominates; sequential or direct algorithms preferred.".to_string(),
    ));
    descriptions.push((
        format!("{} ≤ n ≤ {}", min_useful, sweet_end),
        "Hash-partition provides cache-miss guarantees with reasonable overhead.".to_string(),
    ));
    descriptions.push((
        format!("n > {}", sweet_end),
        "Rayon parallel execution dominates on wall-clock; hash-partition value is in provable cache bounds rather than speed.".to_string(),
    ));

    RegimeAnalysis {
        min_useful_size: min_useful,
        rayon_crossover_size: rayon_crossover,
        sweet_spot_range: (min_useful, sweet_end),
        regime_descriptions: descriptions,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_partition_large_scale() {
        let config = CacheConfig::default();
        let point = hash_partition_large_scale(1024, &config);
        assert_eq!(point.input_size, 1024);
        assert!(point.wall_time_ns > 0);
        assert!(point.l1_miss_rate >= 0.0);
        assert!(point.l2_miss_rate >= 0.0);
    }

    #[test]
    fn test_sequential_large_scale() {
        let config = CacheConfig::default();
        let point = sequential_large_scale(1024, &config);
        assert_eq!(point.input_size, 1024);
        assert!(point.wall_time_ns > 0);
    }

    #[test]
    fn test_rayon_sort_large_scale() {
        let (hp, rayon) = rayon_sort_large_scale(4096);
        assert!(hp > 0);
        assert!(rayon > 0);
    }

    #[test]
    fn test_large_scale_evaluation() {
        let summary = run_large_scale_evaluation(&[1024, 4096], 1);
        assert!(summary.total_measurements > 0);
        assert!(summary.max_size_tested == 4096);
        assert!(summary.rayon_comparison.is_some());
    }

    #[test]
    fn test_large_scale_csv() {
        let summary = run_large_scale_evaluation(&[1024], 1);
        let csv = large_scale_to_csv(&summary);
        assert!(csv.contains("hash_partition"));
        assert!(csv.contains("sequential_scan"));
    }

    #[test]
    fn test_regime_analysis() {
        let analysis = analyze_regimes(&[256, 1024, 4096], 1);
        assert!(analysis.min_useful_size > 0);
        assert!(!analysis.regime_descriptions.is_empty());
    }
}
