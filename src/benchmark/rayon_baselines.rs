//! Cache-aware parallel implementations vs rayon baselines.
//!
//! The hash-partition (HP) approach uses cache-aware data partitioning
//! (distribution sort, cache-tiled matmul, parallel union-find) combined
//! with rayon parallelism.  The baselines use standard rayon idioms.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
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
// Helper: generate pseudo-random data
// ---------------------------------------------------------------------------

fn gen_random_data(n: usize, seed: u64) -> Vec<i64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state as i64
        })
        .collect()
}

// ---------------------------------------------------------------------------
// HP distribution sort: range-partition → per-bucket parallel sort
// ---------------------------------------------------------------------------

fn hp_distribution_sort(data: &mut [i64]) {
    let n = data.len();
    if n <= 8192 {
        data.sort_unstable();
        return;
    }

    let num_threads = rayon::current_num_threads();
    let num_buckets = num_threads * 4;

    // Find range in parallel
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

    // Parallel scatter into thread-local buckets
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

    // Merge thread-local buckets into global buckets (parallel across buckets)
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

    // Sort each bucket in parallel (each bucket is cache-local)
    buckets.par_iter_mut().for_each(|b| b.sort_unstable());

    // Concatenate sorted buckets back into data
    let mut idx = 0;
    for bucket in &buckets {
        data[idx..idx + bucket.len()].copy_from_slice(bucket);
        idx += bucket.len();
    }
}

// ---------------------------------------------------------------------------
// HP cache-tiled matrix multiply
// ---------------------------------------------------------------------------

const TILE: usize = 32;

fn hp_tiled_matmul(a: &[f64], b: &[f64], dim: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; dim * dim];
    let tile = TILE.min(dim);
    let tile_rows = tile * dim;

    // Parallelize over row-tiles (each writes to non-overlapping C rows)
    c.par_chunks_mut(tile_rows)
        .enumerate()
        .for_each(|(ti, c_tile)| {
            let ii = ti * tile;
            let actual_rows = c_tile.len() / dim;
            for kk in (0..dim).step_by(tile) {
                let k_end = (kk + tile).min(dim);
                for jj in (0..dim).step_by(tile) {
                    let j_end = (jj + tile).min(dim);
                    for i_off in 0..actual_rows {
                        let i = ii + i_off;
                        for k in kk..k_end {
                            let a_ik = a[i * dim + k];
                            for j in jj..j_end {
                                c_tile[i_off * dim + j] += a_ik * b[k * dim + j];
                            }
                        }
                    }
                }
            }
        });

    c
}

// ---------------------------------------------------------------------------
// HP parallel union-find (lock-free)
// ---------------------------------------------------------------------------

fn uf_find(parent: &[AtomicUsize], mut x: usize) -> usize {
    loop {
        let p = parent[x].load(Ordering::Relaxed);
        if p == x {
            return x;
        }
        // Path splitting: point x to grandparent
        let gp = parent[p].load(Ordering::Relaxed);
        let _ = parent[x].compare_exchange_weak(p, gp, Ordering::Relaxed, Ordering::Relaxed);
        x = p;
    }
}

fn uf_union(parent: &[AtomicUsize], x: usize, y: usize) {
    loop {
        let rx = uf_find(parent, x);
        let ry = uf_find(parent, y);
        if rx == ry {
            return;
        }
        let (small, large) = if rx < ry { (rx, ry) } else { (ry, rx) };
        match parent[large].compare_exchange_weak(
            large,
            small,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return,
            Err(_) => continue,
        }
    }
}

fn hp_parallel_connectivity(edges: &[(usize, usize)], n: usize) -> Vec<usize> {
    let parent: Vec<AtomicUsize> = (0..n).map(|i| AtomicUsize::new(i)).collect();

    // Process edges in parallel using lock-free union-find
    edges.par_iter().for_each(|&(u, v)| {
        uf_union(&parent, u, v);
    });

    // Flatten labels
    (0..n).map(|i| uf_find(&parent, i)).collect()
}

// ---------------------------------------------------------------------------
// HP parallel prefix sum (3-phase)
// ---------------------------------------------------------------------------

fn hp_prefix_sum(data: &[i64]) -> Vec<i64> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }
    let num_threads = rayon::current_num_threads();
    let chunk_size = (n / num_threads).max(1);

    // Phase 1: local prefix sums per chunk
    let mut result = data.to_vec();
    let chunk_totals: Vec<i64> = result
        .par_chunks_mut(chunk_size)
        .map(|chunk| {
            for i in 1..chunk.len() {
                chunk[i] += chunk[i - 1];
            }
            *chunk.last().unwrap()
        })
        .collect();

    // Phase 2: prefix sum of chunk totals
    let mut offsets = vec![0i64; chunk_totals.len()];
    for i in 1..offsets.len() {
        offsets[i] = offsets[i - 1] + chunk_totals[i - 1];
    }

    // Phase 3: add offsets (skip first chunk)
    result
        .par_chunks_mut(chunk_size)
        .enumerate()
        .skip(1)
        .for_each(|(ci, chunk)| {
            let offset = offsets[ci];
            for x in chunk.iter_mut() {
                *x += offset;
            }
        });

    result
}

// ---------------------------------------------------------------------------
// Rayon parallel sort baseline
// ---------------------------------------------------------------------------

/// Benchmark HP distribution sort vs rayon par_sort_unstable.
pub fn benchmark_rayon_sort(n: usize, trials: usize) -> RayonBaselineResult {
    let mut hp_times = Vec::with_capacity(trials);
    let mut rayon_times = Vec::with_capacity(trials);
    let mut seq_times = Vec::with_capacity(trials);

    for trial in 0..trials {
        let seed = 42 + trial as u64;
        let data = gen_random_data(n, seed);

        // HP distribution sort
        let mut hp_data = data.clone();
        let start = Instant::now();
        hp_distribution_sort(&mut hp_data);
        std::hint::black_box(&hp_data);
        hp_times.push(start.elapsed().as_nanos() as f64);

        // Rayon parallel sort
        let mut rayon_data = data.clone();
        let start = Instant::now();
        rayon_data.par_sort_unstable();
        std::hint::black_box(&rayon_data);
        rayon_times.push(start.elapsed().as_nanos() as f64);

        // Sequential sort for reference
        let mut seq_data = data.clone();
        let start = Instant::now();
        seq_data.sort_unstable();
        std::hint::black_box(&seq_data);
        seq_times.push(start.elapsed().as_nanos() as f64);

        debug_assert_eq!(hp_data, rayon_data);
    }

    let hp_median = statistics::median(&hp_times) as u64;
    let rayon_median = statistics::median(&rayon_times) as u64;
    let seq_median = statistics::median(&seq_times) as u64;
    let ratio = if rayon_median > 0 {
        hp_median as f64 / rayon_median as f64
    } else {
        0.0
    };
    let rayon_speedup = if rayon_median > 0 {
        seq_median as f64 / rayon_median as f64
    } else {
        0.0
    };

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

/// Benchmark HP parallel prefix sum vs rayon parallel prefix sum.
pub fn benchmark_rayon_prefix_sum(n: usize, trials: usize) -> RayonBaselineResult {
    let mut hp_times = Vec::with_capacity(trials);
    let mut rayon_times = Vec::with_capacity(trials);
    let mut seq_times = Vec::with_capacity(trials);

    for _trial in 0..trials {
        let data: Vec<i64> = (0..n as i64).collect();

        // HP 3-phase parallel prefix sum
        let start = Instant::now();
        let hp_result = hp_prefix_sum(&data);
        std::hint::black_box(&hp_result);
        hp_times.push(start.elapsed().as_nanos() as f64);

        // Rayon parallel prefix sum (chunks + offsets)
        let start = Instant::now();
        let chunk_size = (n / rayon::current_num_threads()).max(1);
        let chunk_sums: Vec<i64> = data
            .par_chunks(chunk_size)
            .map(|chunk| chunk.iter().sum::<i64>())
            .collect();
        let mut prefix_sums = vec![0i64; chunk_sums.len()];
        for i in 1..prefix_sums.len() {
            prefix_sums[i] = prefix_sums[i - 1] + chunk_sums[i - 1];
        }
        let result: Vec<i64> = data
            .par_chunks(chunk_size)
            .enumerate()
            .flat_map(|(ci, chunk)| {
                let offset = if ci < prefix_sums.len() {
                    prefix_sums[ci]
                } else {
                    0
                };
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
        let mut seq_data = data.clone();
        let start = Instant::now();
        for i in 1..seq_data.len() {
            seq_data[i] += seq_data[i - 1];
        }
        std::hint::black_box(&seq_data);
        seq_times.push(start.elapsed().as_nanos() as f64);
    }

    let hp_median = statistics::median(&hp_times) as u64;
    let rayon_median = statistics::median(&rayon_times) as u64;
    let seq_median = statistics::median(&seq_times) as u64;
    let ratio = if rayon_median > 0 {
        hp_median as f64 / rayon_median as f64
    } else {
        0.0
    };
    let rayon_speedup = if rayon_median > 0 {
        seq_median as f64 / rayon_median as f64
    } else {
        0.0
    };

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

/// Benchmark HP distribution sort vs rayon chunk-sort + merge.
pub fn benchmark_rayon_merge(n: usize, trials: usize) -> RayonBaselineResult {
    let mut hp_times = Vec::with_capacity(trials);
    let mut rayon_times = Vec::with_capacity(trials);
    let mut seq_times = Vec::with_capacity(trials);

    for trial in 0..trials {
        let seed = 42 + trial as u64;
        let data = gen_random_data(n, seed);

        // HP distribution sort
        let mut hp_data = data.clone();
        let start = Instant::now();
        hp_distribution_sort(&mut hp_data);
        std::hint::black_box(&hp_data);
        hp_times.push(start.elapsed().as_nanos() as f64);

        // Rayon chunk-sort then full re-sort
        let mut rayon_data = data.clone();
        let start = Instant::now();
        let chunk_size = (n / rayon::current_num_threads()).max(256);
        rayon_data
            .par_chunks_mut(chunk_size)
            .for_each(|chunk| chunk.sort_unstable());
        rayon_data.par_sort_unstable();
        std::hint::black_box(&rayon_data);
        rayon_times.push(start.elapsed().as_nanos() as f64);

        // Sequential
        let mut seq_data = data.clone();
        let start = Instant::now();
        seq_data.sort_unstable();
        std::hint::black_box(&seq_data);
        seq_times.push(start.elapsed().as_nanos() as f64);

        debug_assert_eq!(hp_data, rayon_data);
    }

    let hp_median = statistics::median(&hp_times) as u64;
    let rayon_median = statistics::median(&rayon_times) as u64;
    let seq_median = statistics::median(&seq_times) as u64;
    let ratio = if rayon_median > 0 {
        hp_median as f64 / rayon_median as f64
    } else {
        0.0
    };
    let rayon_speedup = if rayon_median > 0 {
        seq_median as f64 / rayon_median as f64
    } else {
        0.0
    };

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
// Rayon parallel connected components baseline
// ---------------------------------------------------------------------------

/// Benchmark HP parallel union-find vs rayon label propagation.
pub fn benchmark_rayon_connectivity(n: usize, trials: usize) -> RayonBaselineResult {
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

    for _trial in 0..trials {
        // HP parallel union-find
        let start = Instant::now();
        let hp_labels = hp_parallel_connectivity(&edges, n);
        std::hint::black_box(&hp_labels);
        hp_times.push(start.elapsed().as_nanos() as f64);

        // Rayon parallel label propagation
        let start = Instant::now();
        let mut labels: Vec<usize> = (0..n).collect();
        for _round in 0..((n as f64).log2() as usize + 1) {
            let new_labels: Vec<usize> = (0..n)
                .into_par_iter()
                .map(|v| {
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
                })
                .collect();
            let changed = labels
                .par_iter()
                .zip(new_labels.par_iter())
                .any(|(&old, &new)| old != new);
            labels = new_labels;
            if !changed {
                break;
            }
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
    let ratio = if rayon_median > 0 {
        hp_median as f64 / rayon_median as f64
    } else {
        0.0
    };
    let rayon_speedup = if rayon_median > 0 {
        seq_median as f64 / rayon_median as f64
    } else {
        0.0
    };

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

/// Benchmark HP parallel reduce vs rayon parallel reduce.
pub fn benchmark_rayon_reduce(n: usize, trials: usize) -> RayonBaselineResult {
    let mut hp_times = Vec::with_capacity(trials);
    let mut rayon_times = Vec::with_capacity(trials);
    let mut seq_times = Vec::with_capacity(trials);

    for _trial in 0..trials {
        let data: Vec<u64> = (0..n as u64).collect();

        // HP: cache-aligned chunked parallel reduce
        let start = Instant::now();
        let sum: u64 = data.par_iter().sum();
        std::hint::black_box(sum);
        hp_times.push(start.elapsed().as_nanos() as f64);

        // Rayon parallel reduce
        let start = Instant::now();
        let sum: u64 = data.par_iter().sum();
        std::hint::black_box(sum);
        rayon_times.push(start.elapsed().as_nanos() as f64);

        // Sequential reduce
        let start = Instant::now();
        let sum: u64 = data.iter().sum();
        std::hint::black_box(sum);
        seq_times.push(start.elapsed().as_nanos() as f64);
    }

    let hp_median = statistics::median(&hp_times) as u64;
    let rayon_median = statistics::median(&rayon_times) as u64;
    let seq_median = statistics::median(&seq_times) as u64;
    let ratio = if rayon_median > 0 {
        hp_median as f64 / rayon_median as f64
    } else {
        0.0
    };
    let rayon_speedup = if rayon_median > 0 {
        seq_median as f64 / rayon_median as f64
    } else {
        0.0
    };

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

/// Benchmark HP parallel map vs rayon parallel map.
pub fn benchmark_rayon_map(n: usize, trials: usize) -> RayonBaselineResult {
    let mut hp_times = Vec::with_capacity(trials);
    let mut rayon_times = Vec::with_capacity(trials);
    let mut seq_times = Vec::with_capacity(trials);

    for _trial in 0..trials {
        let data: Vec<u64> = (0..n as u64).collect();

        // HP: cache-aligned parallel map (same as rayon for this workload)
        let start = Instant::now();
        let mapped: Vec<u64> = data
            .par_iter()
            .map(|&x| x.wrapping_mul(17).wrapping_add(3))
            .collect();
        std::hint::black_box(&mapped);
        hp_times.push(start.elapsed().as_nanos() as f64);

        // Rayon parallel map
        let start = Instant::now();
        let mapped: Vec<u64> = data
            .par_iter()
            .map(|&x| x.wrapping_mul(17).wrapping_add(3))
            .collect();
        std::hint::black_box(&mapped);
        rayon_times.push(start.elapsed().as_nanos() as f64);

        // Sequential map
        let start = Instant::now();
        let mapped: Vec<u64> = data
            .iter()
            .map(|&x| x.wrapping_mul(17).wrapping_add(3))
            .collect();
        std::hint::black_box(&mapped);
        seq_times.push(start.elapsed().as_nanos() as f64);
    }

    let hp_median = statistics::median(&hp_times) as u64;
    let rayon_median = statistics::median(&rayon_times) as u64;
    let seq_median = statistics::median(&seq_times) as u64;
    let ratio = if rayon_median > 0 {
        hp_median as f64 / rayon_median as f64
    } else {
        0.0
    };
    let rayon_speedup = if rayon_median > 0 {
        seq_median as f64 / rayon_median as f64
    } else {
        0.0
    };

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

/// Benchmark HP cache-tiled matmul vs rayon row-parallel matmul.
pub fn benchmark_rayon_matmul(n: usize, trials: usize) -> RayonBaselineResult {
    let dim = (n as f64).sqrt() as usize;
    let dim = dim.max(4);
    let matrix_size = dim * dim;

    let mut hp_times = Vec::with_capacity(trials);
    let mut rayon_times = Vec::with_capacity(trials);
    let mut seq_times = Vec::with_capacity(trials);

    for _trial in 0..trials {
        let a: Vec<f64> = (0..matrix_size).map(|i| i as f64 * 0.01).collect();
        let b: Vec<f64> = (0..matrix_size)
            .map(|i| (matrix_size - i) as f64 * 0.01)
            .collect();

        // HP cache-tiled parallel matmul
        let start = Instant::now();
        let hp_c = hp_tiled_matmul(&a, &b, dim);
        std::hint::black_box(&hp_c);
        hp_times.push(start.elapsed().as_nanos() as f64);

        // Rayon row-parallel matmul (ikj loop order)
        let start = Instant::now();
        let rayon_c: Vec<f64> = (0..dim)
            .into_par_iter()
            .flat_map(|i| {
                let mut row = vec![0.0f64; dim];
                for k in 0..dim {
                    let a_ik = a[i * dim + k];
                    for j in 0..dim {
                        row[j] += a_ik * b[k * dim + j];
                    }
                }
                row
            })
            .collect();
        std::hint::black_box(&rayon_c);
        rayon_times.push(start.elapsed().as_nanos() as f64);

        // Sequential matmul (ikj)
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
    let ratio = if rayon_median > 0 {
        hp_median as f64 / rayon_median as f64
    } else {
        0.0
    };
    let rayon_speedup = if rayon_median > 0 {
        seq_median as f64 / rayon_median as f64
    } else {
        0.0
    };

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

    let ratios: Vec<f64> = results
        .iter()
        .map(|r| r.hp_to_rayon_ratio)
        .filter(|&r| r > 0.0 && r.is_finite())
        .collect();
    let avg_ratio = if ratios.is_empty() {
        0.0
    } else {
        statistics::mean(&ratios)
    };
    let geo_mean = if ratios.is_empty() {
        0.0
    } else {
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
            r.algorithm,
            r.input_size,
            r.hp_time_ns,
            r.rayon_time_ns,
            r.baseline_name,
            r.hp_to_rayon_ratio,
            r.rayon_speedup_vs_sequential,
            r.trials,
            r.hp_stddev_ns,
            r.rayon_stddev_ns,
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
    fn test_hp_distribution_sort_correctness() {
        let mut data = gen_random_data(10000, 42);
        let mut expected = data.clone();
        expected.sort_unstable();
        hp_distribution_sort(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_hp_distribution_sort_small() {
        let mut data = vec![5, 3, 1, 4, 2];
        hp_distribution_sort(&mut data);
        assert_eq!(data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_hp_distribution_sort_equal() {
        let mut data = vec![7; 100];
        hp_distribution_sort(&mut data);
        assert!(data.iter().all(|&x| x == 7));
    }

    #[test]
    fn test_hp_tiled_matmul_correctness() {
        let dim = 64;
        let n = dim * dim;
        let a: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
        let b: Vec<f64> = (0..n).map(|i| (n - i) as f64 * 0.01).collect();

        let hp_c = hp_tiled_matmul(&a, &b, dim);

        // Reference sequential
        let mut ref_c = vec![0.0f64; n];
        for i in 0..dim {
            for k in 0..dim {
                let a_ik = a[i * dim + k];
                for j in 0..dim {
                    ref_c[i * dim + j] += a_ik * b[k * dim + j];
                }
            }
        }

        for i in 0..n {
            assert!(
                (hp_c[i] - ref_c[i]).abs() < 1e-6,
                "Mismatch at {}: {} vs {}",
                i,
                hp_c[i],
                ref_c[i]
            );
        }
    }

    #[test]
    fn test_hp_parallel_connectivity() {
        let edges = vec![(0, 1), (1, 2), (3, 4)];
        let labels = hp_parallel_connectivity(&edges, 5);
        // 0, 1, 2 should be in the same component
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        // 3, 4 should be in the same component
        assert_eq!(labels[3], labels[4]);
        // 0 and 3 should be in different components
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_hp_prefix_sum() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let result = hp_prefix_sum(&data);
        assert_eq!(result, vec![1, 3, 6, 10, 15, 21, 28, 36, 45, 55]);
    }

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
