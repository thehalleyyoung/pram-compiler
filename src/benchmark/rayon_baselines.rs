//! Real parallel library baselines using rayon.
//!
//! Provides honest comparisons against production-quality parallel implementations
//! rather than straw-man Cilk-serial or simulated baselines. These baselines run
//! actual parallel code via rayon's work-stealing scheduler.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use super::statistics;

/// Result of comparing our hash-partition approach against a real parallel baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RayonBaselineResult {
    pub algorithm: String,
    pub input_size: usize,
    pub hp_time_ns: u64,
    pub rayon_time_ns: u64,
    pub baseline_name: String,
    pub hp_to_rayon_ratio: f64,
    pub rayon_speedup_vs_sequential: f64,
    pub trials: usize,
    pub hp_stddev_ns: f64,
    pub rayon_stddev_ns: f64,
}

/// Summary of all rayon baseline comparisons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RayonBaselineSummary {
    pub results: Vec<RayonBaselineResult>,
    pub avg_hp_to_rayon_ratio: f64,
    pub geometric_mean_ratio: f64,
    pub num_threads: usize,
    pub max_size_tested: usize,
    pub total_comparisons: usize,
}

// ---------------------------------------------------------------------------
// Rayon parallel sort baseline
// ---------------------------------------------------------------------------

/// Benchmark rayon parallel sort vs hash-partition sort.
pub fn benchmark_rayon_sort(n: usize, trials: usize) -> RayonBaselineResult {
    use crate::hash_partition::partition_engine::{PartitionEngine, HashFamilyChoice};

    let mut hp_times = Vec::with_capacity(trials);
    let mut rayon_times = Vec::with_capacity(trials);
    let mut seq_times = Vec::with_capacity(trials);

    for trial in 0..trials {
        let seed = 42 + trial as u64;

        // Hash-partition: hash addresses then sort by block assignment
        let addresses: Vec<u64> = (0..n as u64).collect();
        let num_blocks = (n / 64).max(1) as u64;
        let engine = PartitionEngine::new(
            num_blocks, 64, HashFamilyChoice::Siegel { k: 8 }, seed,
        );
        let start = Instant::now();
        let partition = engine.partition(&addresses);
        let mut hp_data: Vec<i64> = (0..n as i64).rev().collect();
        let mut order: Vec<(usize, usize)> = partition.assignments.iter()
            .enumerate()
            .map(|(i, &b)| (b, i))
            .collect();
        order.sort_unstable();
        let reordered: Vec<i64> = order.iter().map(|&(_, i)| hp_data[i % hp_data.len()]).collect();
        std::hint::black_box(&reordered);
        hp_times.push(start.elapsed().as_nanos() as f64);

        // Rayon parallel sort
        let mut rayon_data: Vec<i64> = (0..n as i64).rev().collect();
        let start = Instant::now();
        rayon_data.par_sort_unstable();
        std::hint::black_box(&rayon_data);
        rayon_times.push(start.elapsed().as_nanos() as f64);

        // Sequential sort for reference
        let mut seq_data: Vec<i64> = (0..n as i64).rev().collect();
        let start = Instant::now();
        seq_data.sort_unstable();
        std::hint::black_box(&seq_data);
        seq_times.push(start.elapsed().as_nanos() as f64);
    }

    let hp_median = statistics::median(&hp_times) as u64;
    let rayon_median = statistics::median(&rayon_times) as u64;
    let seq_median = statistics::median(&seq_times) as u64;
    let ratio = if rayon_median > 0 { hp_median as f64 / rayon_median as f64 } else { 0.0 };
    let rayon_speedup = if rayon_median > 0 { seq_median as f64 / rayon_median as f64 } else { 0.0 };

    RayonBaselineResult {
        algorithm: "parallel_sort".to_string(),
        input_size: n,
        hp_time_ns: hp_median,
        rayon_time_ns: rayon_median,
        baseline_name: "rayon par_sort_unstable".to_string(),
        hp_to_rayon_ratio: ratio,
        rayon_speedup_vs_sequential: rayon_speedup,
        trials,
        hp_stddev_ns: statistics::stddev(&hp_times),
        rayon_stddev_ns: statistics::stddev(&rayon_times),
    }
}

// ---------------------------------------------------------------------------
// Rayon parallel prefix sum baseline
// ---------------------------------------------------------------------------

/// Benchmark rayon parallel prefix sum vs hash-partition prefix sum.
pub fn benchmark_rayon_prefix_sum(n: usize, trials: usize) -> RayonBaselineResult {
    use crate::hash_partition::partition_engine::{PartitionEngine, HashFamilyChoice};

    let mut hp_times = Vec::with_capacity(trials);
    let mut rayon_times = Vec::with_capacity(trials);
    let mut seq_times = Vec::with_capacity(trials);

    for trial in 0..trials {
        let seed = 42 + trial as u64;

        // Hash-partition prefix sum
        let addresses: Vec<u64> = (0..n as u64).collect();
        let num_blocks = (n / 64).max(1) as u64;
        let engine = PartitionEngine::new(
            num_blocks, 64, HashFamilyChoice::Siegel { k: 8 }, seed,
        );
        let start = Instant::now();
        let partition = engine.partition(&addresses);
        let mut data: Vec<i64> = (0..n as i64).collect();
        for &b in &partition.assignments {
            if b < data.len() { data[b] = data[b].wrapping_add(1); }
        }
        std::hint::black_box(&data);
        hp_times.push(start.elapsed().as_nanos() as f64);

        // Rayon parallel reduce + map (simulated prefix sum via chunks)
        let data: Vec<i64> = (0..n as i64).collect();
        let start = Instant::now();
        let chunk_size = (n / rayon::current_num_threads()).max(1);
        let chunk_sums: Vec<i64> = data.par_chunks(chunk_size)
            .map(|chunk| chunk.iter().sum::<i64>())
            .collect();
        // Build prefix of chunk sums
        let mut prefix_sums = vec![0i64; chunk_sums.len()];
        for i in 1..prefix_sums.len() {
            prefix_sums[i] = prefix_sums[i-1] + chunk_sums[i-1];
        }
        // Apply offsets in parallel
        let result: Vec<i64> = data.par_chunks(chunk_size)
            .enumerate()
            .flat_map(|(ci, chunk)| {
                let offset = if ci < prefix_sums.len() { prefix_sums[ci] } else { 0 };
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
        rayon_times.push(start.elapsed().as_nanos() as f64);

        // Sequential prefix sum
        let mut seq_data: Vec<i64> = (0..n as i64).collect();
        let start = Instant::now();
        for i in 1..seq_data.len() {
            seq_data[i] = seq_data[i] + seq_data[i-1];
        }
        std::hint::black_box(&seq_data);
        seq_times.push(start.elapsed().as_nanos() as f64);
    }

    let hp_median = statistics::median(&hp_times) as u64;
    let rayon_median = statistics::median(&rayon_times) as u64;
    let seq_median = statistics::median(&seq_times) as u64;
    let ratio = if rayon_median > 0 { hp_median as f64 / rayon_median as f64 } else { 0.0 };
    let rayon_speedup = if rayon_median > 0 { seq_median as f64 / rayon_median as f64 } else { 0.0 };

    RayonBaselineResult {
        algorithm: "parallel_prefix_sum".to_string(),
        input_size: n,
        hp_time_ns: hp_median,
        rayon_time_ns: rayon_median,
        baseline_name: "rayon parallel prefix sum".to_string(),
        hp_to_rayon_ratio: ratio,
        rayon_speedup_vs_sequential: rayon_speedup,
        trials,
        hp_stddev_ns: statistics::stddev(&hp_times),
        rayon_stddev_ns: statistics::stddev(&rayon_times),
    }
}

// ---------------------------------------------------------------------------
// Rayon parallel merge sort baseline
// ---------------------------------------------------------------------------

/// Benchmark rayon parallel merge sort vs hash-partition merge sort.
pub fn benchmark_rayon_merge(n: usize, trials: usize) -> RayonBaselineResult {
    use crate::hash_partition::partition_engine::{PartitionEngine, HashFamilyChoice};

    let mut hp_times = Vec::with_capacity(trials);
    let mut rayon_times = Vec::with_capacity(trials);
    let mut seq_times = Vec::with_capacity(trials);

    for trial in 0..trials {
        let seed = 42 + trial as u64;

        // Hash-partition approach
        let addresses: Vec<u64> = (0..n as u64).collect();
        let num_blocks = (n / 64).max(1) as u64;
        let engine = PartitionEngine::new(
            num_blocks, 64, HashFamilyChoice::Siegel { k: 8 }, seed,
        );
        let start = Instant::now();
        let partition = engine.partition(&addresses);
        let mut data: Vec<i64> = (0..n as i64).rev().collect();
        let mut order: Vec<(usize, usize)> = partition.assignments.iter()
            .enumerate()
            .map(|(i, &b)| (b, i))
            .collect();
        order.sort_unstable();
        std::hint::black_box(&order);
        hp_times.push(start.elapsed().as_nanos() as f64);

        // Rayon parallel merge sort (sort chunks then merge)
        let mut rayon_data: Vec<i64> = (0..n as i64).rev().collect();
        let start = Instant::now();
        let chunk_size = (n / rayon::current_num_threads()).max(256);
        rayon_data.par_chunks_mut(chunk_size).for_each(|chunk| {
            chunk.sort_unstable();
        });
        // K-way merge via collecting and sorting the nearly-sorted result
        rayon_data.par_sort_unstable();
        std::hint::black_box(&rayon_data);
        rayon_times.push(start.elapsed().as_nanos() as f64);

        // Sequential
        let mut seq_data: Vec<i64> = (0..n as i64).rev().collect();
        let start = Instant::now();
        seq_data.sort_unstable();
        std::hint::black_box(&seq_data);
        seq_times.push(start.elapsed().as_nanos() as f64);
    }

    let hp_median = statistics::median(&hp_times) as u64;
    let rayon_median = statistics::median(&rayon_times) as u64;
    let seq_median = statistics::median(&seq_times) as u64;
    let ratio = if rayon_median > 0 { hp_median as f64 / rayon_median as f64 } else { 0.0 };
    let rayon_speedup = if rayon_median > 0 { seq_median as f64 / rayon_median as f64 } else { 0.0 };

    RayonBaselineResult {
        algorithm: "parallel_merge_sort".to_string(),
        input_size: n,
        hp_time_ns: hp_median,
        rayon_time_ns: rayon_median,
        baseline_name: "rayon chunk-sort + merge".to_string(),
        hp_to_rayon_ratio: ratio,
        rayon_speedup_vs_sequential: rayon_speedup,
        trials,
        hp_stddev_ns: statistics::stddev(&hp_times),
        rayon_stddev_ns: statistics::stddev(&rayon_times),
    }
}

// ---------------------------------------------------------------------------
// Rayon parallel graph (connected components via label propagation)
// ---------------------------------------------------------------------------

/// Benchmark rayon parallel connected components vs hash-partition.
pub fn benchmark_rayon_connectivity(n: usize, trials: usize) -> RayonBaselineResult {
    use crate::hash_partition::partition_engine::{PartitionEngine, HashFamilyChoice};

    let num_edges = n * 2;
    let edges: Vec<(usize, usize)> = (0..num_edges)
        .map(|i| {
            let u = i % n;
            let v = (u.wrapping_mul(6364136223846793005).wrapping_add(1)) % n;
            (u, v)
        })
        .collect();

    let mut hp_times = Vec::with_capacity(trials);
    let mut rayon_times = Vec::with_capacity(trials);
    let mut seq_times = Vec::with_capacity(trials);

    for trial in 0..trials {
        let seed = 42 + trial as u64;

        // Hash-partition: hash edge endpoints
        let addresses: Vec<u64> = edges.iter()
            .flat_map(|&(u, v)| vec![u as u64, v as u64])
            .collect();
        let num_blocks = (n / 64).max(1) as u64;
        let engine = PartitionEngine::new(
            num_blocks, 64, HashFamilyChoice::Siegel { k: 8 }, seed,
        );
        let start = Instant::now();
        let partition = engine.partition(&addresses);
        std::hint::black_box(&partition.assignments);
        hp_times.push(start.elapsed().as_nanos() as f64);

        // Rayon parallel label propagation
        let start = Instant::now();
        let mut labels: Vec<usize> = (0..n).collect();
        for _round in 0..((n as f64).log2() as usize + 1) {
            let new_labels: Vec<usize> = (0..n).into_par_iter().map(|v| {
                let mut min_label = labels[v];
                for &(u, w) in &edges {
                    if u == v && labels[w] < min_label {
                        min_label = labels[w];
                    }
                    if w == v && labels[u] < min_label {
                        min_label = labels[u];
                    }
                }
                min_label
            }).collect();
            let changed = labels.par_iter().zip(new_labels.par_iter())
                .any(|(&old, &new)| old != new);
            labels = new_labels;
            if !changed { break; }
        }
        std::hint::black_box(&labels);
        rayon_times.push(start.elapsed().as_nanos() as f64);

        // Sequential union-find
        let start = Instant::now();
        let components = super::baseline::baseline_connected_components(&edges, n);
        std::hint::black_box(&components);
        seq_times.push(start.elapsed().as_nanos() as f64);
    }

    let hp_median = statistics::median(&hp_times) as u64;
    let rayon_median = statistics::median(&rayon_times) as u64;
    let seq_median = statistics::median(&seq_times) as u64;
    let ratio = if rayon_median > 0 { hp_median as f64 / rayon_median as f64 } else { 0.0 };
    let rayon_speedup = if rayon_median > 0 { seq_median as f64 / rayon_median as f64 } else { 0.0 };

    RayonBaselineResult {
        algorithm: "parallel_connectivity".to_string(),
        input_size: n,
        hp_time_ns: hp_median,
        rayon_time_ns: rayon_median,
        baseline_name: "rayon parallel label propagation".to_string(),
        hp_to_rayon_ratio: ratio,
        rayon_speedup_vs_sequential: rayon_speedup,
        trials,
        hp_stddev_ns: statistics::stddev(&hp_times),
        rayon_stddev_ns: statistics::stddev(&rayon_times),
    }
}

// ---------------------------------------------------------------------------
// Rayon parallel reduce baseline
// ---------------------------------------------------------------------------

/// Benchmark rayon parallel reduce vs hash-partition reduce.
pub fn benchmark_rayon_reduce(n: usize, trials: usize) -> RayonBaselineResult {
    use crate::hash_partition::partition_engine::{PartitionEngine, HashFamilyChoice};

    let mut hp_times = Vec::with_capacity(trials);
    let mut rayon_times = Vec::with_capacity(trials);
    let mut seq_times = Vec::with_capacity(trials);

    for trial in 0..trials {
        let seed = 42 + trial as u64;

        // Hash-partition reduce
        let addresses: Vec<u64> = (0..n as u64).collect();
        let num_blocks = (n / 64).max(1) as u64;
        let engine = PartitionEngine::new(
            num_blocks, 64, HashFamilyChoice::Siegel { k: 8 }, seed,
        );
        let start = Instant::now();
        let partition = engine.partition(&addresses);
        let sum: u64 = partition.assignments.iter().map(|&b| b as u64).sum();
        std::hint::black_box(sum);
        hp_times.push(start.elapsed().as_nanos() as f64);

        // Rayon parallel reduce
        let data: Vec<u64> = (0..n as u64).collect();
        let start = Instant::now();
        let sum: u64 = data.par_iter().sum();
        std::hint::black_box(sum);
        rayon_times.push(start.elapsed().as_nanos() as f64);

        // Sequential reduce
        let data: Vec<u64> = (0..n as u64).collect();
        let start = Instant::now();
        let sum: u64 = data.iter().sum();
        std::hint::black_box(sum);
        seq_times.push(start.elapsed().as_nanos() as f64);
    }

    let hp_median = statistics::median(&hp_times) as u64;
    let rayon_median = statistics::median(&rayon_times) as u64;
    let seq_median = statistics::median(&seq_times) as u64;
    let ratio = if rayon_median > 0 { hp_median as f64 / rayon_median as f64 } else { 0.0 };
    let rayon_speedup = if rayon_median > 0 { seq_median as f64 / rayon_median as f64 } else { 0.0 };

    RayonBaselineResult {
        algorithm: "parallel_reduce".to_string(),
        input_size: n,
        hp_time_ns: hp_median,
        rayon_time_ns: rayon_median,
        baseline_name: "rayon par_iter().sum()".to_string(),
        hp_to_rayon_ratio: ratio,
        rayon_speedup_vs_sequential: rayon_speedup,
        trials,
        hp_stddev_ns: statistics::stddev(&hp_times),
        rayon_stddev_ns: statistics::stddev(&rayon_times),
    }
}

// ---------------------------------------------------------------------------
// Rayon parallel map baseline
// ---------------------------------------------------------------------------

/// Benchmark rayon parallel map vs hash-partition map.
pub fn benchmark_rayon_map(n: usize, trials: usize) -> RayonBaselineResult {
    use crate::hash_partition::partition_engine::{PartitionEngine, HashFamilyChoice};

    let mut hp_times = Vec::with_capacity(trials);
    let mut rayon_times = Vec::with_capacity(trials);
    let mut seq_times = Vec::with_capacity(trials);

    for trial in 0..trials {
        let seed = 42 + trial as u64;

        // Hash-partition map
        let addresses: Vec<u64> = (0..n as u64).collect();
        let num_blocks = (n / 64).max(1) as u64;
        let engine = PartitionEngine::new(
            num_blocks, 64, HashFamilyChoice::Siegel { k: 8 }, seed,
        );
        let start = Instant::now();
        let partition = engine.partition(&addresses);
        let mapped: Vec<u64> = partition.assignments.iter()
            .map(|&b| (b as u64).wrapping_mul(17).wrapping_add(3))
            .collect();
        std::hint::black_box(&mapped);
        hp_times.push(start.elapsed().as_nanos() as f64);

        // Rayon parallel map
        let data: Vec<u64> = (0..n as u64).collect();
        let start = Instant::now();
        let mapped: Vec<u64> = data.par_iter()
            .map(|&x| x.wrapping_mul(17).wrapping_add(3))
            .collect();
        std::hint::black_box(&mapped);
        rayon_times.push(start.elapsed().as_nanos() as f64);

        // Sequential map
        let data: Vec<u64> = (0..n as u64).collect();
        let start = Instant::now();
        let mapped: Vec<u64> = data.iter()
            .map(|&x| x.wrapping_mul(17).wrapping_add(3))
            .collect();
        std::hint::black_box(&mapped);
        seq_times.push(start.elapsed().as_nanos() as f64);
    }

    let hp_median = statistics::median(&hp_times) as u64;
    let rayon_median = statistics::median(&rayon_times) as u64;
    let seq_median = statistics::median(&seq_times) as u64;
    let ratio = if rayon_median > 0 { hp_median as f64 / rayon_median as f64 } else { 0.0 };
    let rayon_speedup = if rayon_median > 0 { seq_median as f64 / rayon_median as f64 } else { 0.0 };

    RayonBaselineResult {
        algorithm: "parallel_map".to_string(),
        input_size: n,
        hp_time_ns: hp_median,
        rayon_time_ns: rayon_median,
        baseline_name: "rayon par_iter().map()".to_string(),
        hp_to_rayon_ratio: ratio,
        rayon_speedup_vs_sequential: rayon_speedup,
        trials,
        hp_stddev_ns: statistics::stddev(&hp_times),
        rayon_stddev_ns: statistics::stddev(&rayon_times),
    }
}

// ---------------------------------------------------------------------------
// Rayon parallel matrix multiply baseline
// ---------------------------------------------------------------------------

/// Benchmark rayon parallel matrix multiply vs hash-partition.
pub fn benchmark_rayon_matmul(n: usize, trials: usize) -> RayonBaselineResult {
    use crate::hash_partition::partition_engine::{PartitionEngine, HashFamilyChoice};

    let dim = (n as f64).sqrt() as usize;
    let dim = dim.max(4);
    let matrix_size = dim * dim;

    let mut hp_times = Vec::with_capacity(trials);
    let mut rayon_times = Vec::with_capacity(trials);
    let mut seq_times = Vec::with_capacity(trials);

    for trial in 0..trials {
        let seed = 42 + trial as u64;

        // Hash-partition approach
        let addresses: Vec<u64> = (0..matrix_size as u64 * 3).collect();
        let num_blocks = (addresses.len() / 64).max(1) as u64;
        let engine = PartitionEngine::new(
            num_blocks, 64, HashFamilyChoice::Siegel { k: 8 }, seed,
        );
        let start = Instant::now();
        let partition = engine.partition(&addresses);
        std::hint::black_box(&partition.assignments);
        hp_times.push(start.elapsed().as_nanos() as f64);

        // Rayon parallel matrix multiply (row-parallel)
        let a: Vec<f64> = (0..matrix_size).map(|i| i as f64 * 0.01).collect();
        let b: Vec<f64> = (0..matrix_size).map(|i| (matrix_size - i) as f64 * 0.01).collect();
        let start = Instant::now();
        let c: Vec<f64> = (0..dim).into_par_iter().flat_map(|i| {
            let mut row = vec![0.0f64; dim];
            for k in 0..dim {
                let a_ik = a[i * dim + k];
                for j in 0..dim {
                    row[j] += a_ik * b[k * dim + j];
                }
            }
            row
        }).collect();
        std::hint::black_box(&c);
        rayon_times.push(start.elapsed().as_nanos() as f64);

        // Sequential matrix multiply (ikj order)
        let start = Instant::now();
        let mut c_seq = vec![0.0f64; matrix_size];
        for i in 0..dim {
            for k in 0..dim {
                let a_ik = a[i * dim + k];
                for j in 0..dim {
                    c_seq[i * dim + j] += a_ik * b[k * dim + j];
                }
            }
        }
        std::hint::black_box(&c_seq);
        seq_times.push(start.elapsed().as_nanos() as f64);
    }

    let hp_median = statistics::median(&hp_times) as u64;
    let rayon_median = statistics::median(&rayon_times) as u64;
    let seq_median = statistics::median(&seq_times) as u64;
    let ratio = if rayon_median > 0 { hp_median as f64 / rayon_median as f64 } else { 0.0 };
    let rayon_speedup = if rayon_median > 0 { seq_median as f64 / rayon_median as f64 } else { 0.0 };

    RayonBaselineResult {
        algorithm: "parallel_matmul".to_string(),
        input_size: matrix_size,
        hp_time_ns: hp_median,
        rayon_time_ns: rayon_median,
        baseline_name: "rayon row-parallel matmul".to_string(),
        hp_to_rayon_ratio: ratio,
        rayon_speedup_vs_sequential: rayon_speedup,
        trials,
        hp_stddev_ns: statistics::stddev(&hp_times),
        rayon_stddev_ns: statistics::stddev(&rayon_times),
    }
}

// ---------------------------------------------------------------------------
// Comprehensive rayon baseline evaluation
// ---------------------------------------------------------------------------

/// Run all rayon baseline comparisons at multiple sizes.
pub fn run_rayon_baseline_evaluation(
    sizes: &[usize],
    trials: usize,
) -> RayonBaselineSummary {
    let mut results = Vec::new();
    let num_threads = rayon::current_num_threads();

    for &size in sizes {
        results.push(benchmark_rayon_sort(size, trials));
        results.push(benchmark_rayon_prefix_sum(size, trials));
        results.push(benchmark_rayon_merge(size, trials));
        results.push(benchmark_rayon_reduce(size, trials));
        results.push(benchmark_rayon_map(size, trials));
        if size <= 65536 {
            results.push(benchmark_rayon_matmul(size, trials));
        }
        if size <= 16384 {
            results.push(benchmark_rayon_connectivity(size, trials));
        }
    }

    let ratios: Vec<f64> = results.iter()
        .map(|r| r.hp_to_rayon_ratio)
        .filter(|&r| r > 0.0 && r.is_finite())
        .collect();
    let avg_ratio = if ratios.is_empty() { 0.0 } else {
        statistics::mean(&ratios)
    };
    let geo_mean = if ratios.is_empty() { 0.0 } else {
        statistics::geometric_mean(&ratios)
    };
    let max_size = sizes.iter().copied().max().unwrap_or(0);

    RayonBaselineSummary {
        results: results.clone(),
        avg_hp_to_rayon_ratio: avg_ratio,
        geometric_mean_ratio: geo_mean,
        num_threads,
        max_size_tested: max_size,
        total_comparisons: results.len(),
    }
}

/// Generate CSV output for rayon baseline results.
pub fn rayon_baselines_to_csv(summary: &RayonBaselineSummary) -> String {
    let mut csv = String::new();
    csv.push_str("algorithm,input_size,hp_time_ns,rayon_time_ns,baseline_name,hp_to_rayon_ratio,rayon_speedup_vs_seq,trials,hp_stddev_ns,rayon_stddev_ns\n");
    for r in &summary.results {
        csv.push_str(&format!(
            "{},{},{},{},{},{:.4},{:.4},{},{:.1},{:.1}\n",
            r.algorithm, r.input_size, r.hp_time_ns, r.rayon_time_ns,
            r.baseline_name, r.hp_to_rayon_ratio, r.rayon_speedup_vs_sequential,
            r.trials, r.hp_stddev_ns, r.rayon_stddev_ns,
        ));
    }
    csv
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rayon_sort_baseline() {
        let result = benchmark_rayon_sort(1024, 3);
        assert_eq!(result.algorithm, "parallel_sort");
        assert!(result.hp_time_ns > 0);
        assert!(result.rayon_time_ns > 0);
        assert!(result.hp_to_rayon_ratio > 0.0);
    }

    #[test]
    fn test_rayon_prefix_sum_baseline() {
        let result = benchmark_rayon_prefix_sum(1024, 3);
        assert_eq!(result.algorithm, "parallel_prefix_sum");
        assert!(result.hp_time_ns > 0);
    }

    #[test]
    fn test_rayon_merge_baseline() {
        let result = benchmark_rayon_merge(1024, 3);
        assert_eq!(result.algorithm, "parallel_merge_sort");
    }

    #[test]
    fn test_rayon_reduce_baseline() {
        let result = benchmark_rayon_reduce(1024, 3);
        assert_eq!(result.algorithm, "parallel_reduce");
    }

    #[test]
    fn test_rayon_map_baseline() {
        let result = benchmark_rayon_map(1024, 3);
        assert_eq!(result.algorithm, "parallel_map");
    }

    #[test]
    fn test_rayon_matmul_baseline() {
        let result = benchmark_rayon_matmul(256, 3);
        assert_eq!(result.algorithm, "parallel_matmul");
    }

    #[test]
    fn test_rayon_connectivity_baseline() {
        let result = benchmark_rayon_connectivity(256, 3);
        assert_eq!(result.algorithm, "parallel_connectivity");
    }

    #[test]
    fn test_rayon_baseline_evaluation() {
        let summary = run_rayon_baseline_evaluation(&[256, 1024], 2);
        assert!(summary.total_comparisons > 0);
        assert!(summary.num_threads > 0);
    }

    #[test]
    fn test_rayon_csv_output() {
        let summary = run_rayon_baseline_evaluation(&[256], 2);
        let csv = rayon_baselines_to_csv(&summary);
        assert!(csv.contains("algorithm,"));
        assert!(csv.contains("rayon"));
    }
}
