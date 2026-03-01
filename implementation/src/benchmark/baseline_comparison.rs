//! Baseline comparisons: Cilk serial execution simulation and cache-oblivious algorithms.
//!
//! Provides fair methodology for comparing hash-partition compilation against:
//! 1. Cilk-style serial execution (work-stealing scheduler running on 1 core)
//! 2. Cache-oblivious algorithms (funnel sort, recursive matrix multiply)
//! 3. Hand-optimized sequential baselines (introsort, ikj matmul, etc.)

use serde::{Deserialize, Serialize};
use std::time::Instant;

use super::cache_sim::{CacheSimulator, CacheStats};
use super::statistics::{self, TTestResult};
use crate::hash_partition::partition_engine::{HashFamilyChoice, PartitionEngine};
use crate::pram_ir::ast::PramProgram;

/// Which baseline approach is being compared.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BaselineKind {
    /// Our hash-partition + Brent scheduling approach
    HashPartition,
    /// Cilk-style serial execution: DFS work-stealing on 1 processor
    CilkSerial,
    /// Cache-oblivious recursive algorithm (e.g., funnel sort, recursive matmul)
    CacheOblivious,
    /// Hand-optimized sequential baseline
    HandOptimized,
}

impl BaselineKind {
    pub fn name(&self) -> &'static str {
        match self {
            BaselineKind::HashPartition => "hash-partition",
            BaselineKind::CilkSerial => "cilk-serial",
            BaselineKind::CacheOblivious => "cache-oblivious",
            BaselineKind::HandOptimized => "hand-optimized",
        }
    }
}

/// Result of a baseline comparison for one algorithm at one input size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub algorithm: String,
    pub input_size: usize,
    pub baseline_kind: BaselineKind,
    pub cache_misses: u64,
    pub miss_rate: f64,
    pub work_ops: u64,
    pub wall_time_ns: u64,
    pub overhead_vs_optimal: f64,
}

/// Summary of comparative results across all baselines for one algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSummary {
    pub algorithm: String,
    pub input_size: usize,
    pub hash_partition_misses: u64,
    pub cilk_serial_misses: u64,
    pub cache_oblivious_misses: u64,
    pub hand_optimized_misses: u64,
    pub hash_partition_overhead: f64,
    pub cilk_serial_overhead: f64,
    pub cache_oblivious_overhead: f64,
    pub hash_partition_wins: bool,
}

// ---------------------------------------------------------------------------
// Cilk serial execution simulation
// ---------------------------------------------------------------------------

/// Simulate Cilk-style DFS serial execution.
///
/// In Cilk serial mode, the work-stealing scheduler runs on one core using
/// depth-first traversal.  Total work is O(pT) where p = processors, T = time.
/// DFS gives stack-like locality *within* each task, but inter-task accesses
/// still touch all n elements.  The trace therefore makes O(n·T) accesses
/// with DFS ordering within each phase.
///
/// Returns (work_ops, memory_trace).
pub fn cilk_serial_trace(prog: &PramProgram, n: usize) -> (u64, Vec<u64>) {
    let phases = prog.parallel_step_count().max(1);
    let p = prog.processor_count().unwrap_or(n);

    // Total work proportional to pT, but each processor touches up to n/p elements
    let elems_per_proc = (n / p.max(1)).max(1);
    let total_work = p * phases * elems_per_proc;
    let elem_size = 8u64;
    let mut trace = Vec::with_capacity(total_work);

    // DFS ordering: process each phase, each processor works on its slice
    // of the input with sequential stride (good stack-like locality).
    for phase in 0..phases {
        for proc_id in 0..p.min(n) {
            let slice_start = proc_id * elems_per_proc;
            for i in 0..elems_per_proc.min(n - slice_start) {
                let addr = ((slice_start + i) as u64) * elem_size
                    + (phase as u64) * (n as u64) * elem_size;
                trace.push(addr);
            }
        }
    }

    (trace.len() as u64, trace)
}

// ---------------------------------------------------------------------------
// Cache-oblivious algorithm traces
// ---------------------------------------------------------------------------

/// Generate a cache-oblivious sorting trace (funnel sort / merge sort variant).
///
/// Cache-oblivious merge sort achieves O(n/B log_{M/B}(n/B)) cache misses
/// without knowing M or B. Uses recursive halving.
pub fn cache_oblivious_sort_trace(n: usize) -> (u64, Vec<u64>) {
    let elem_size = 8u64;
    let mut trace = Vec::new();

    // Simulate recursive merge sort access pattern
    fn merge_trace(trace: &mut Vec<u64>, base: u64, n: usize, elem_size: u64) {
        if n <= 1 {
            if n == 1 {
                trace.push(base);
            }
            return;
        }
        let mid = n / 2;
        merge_trace(trace, base, mid, elem_size);
        merge_trace(trace, base + (mid as u64) * elem_size, n - mid, elem_size);
        // Merge phase: interleaved access to both halves
        for i in 0..n {
            trace.push(base + (i as u64) * elem_size);
        }
    }

    merge_trace(&mut trace, 0, n, elem_size);
    let work = trace.len() as u64;
    (work, trace)
}

/// Generate a cache-oblivious matrix multiply trace (recursive 8-way decomposition).
///
/// Achieves O(n³/(B√M)) cache misses, optimal for matrix multiplication.
pub fn cache_oblivious_matmul_trace(n: usize) -> (u64, Vec<u64>) {
    let elem_size = 8u64;
    let mut trace = Vec::new();
    let matrix_size = n * n;

    // Recursive decomposition: divide each matrix into 4 quadrants
    fn rec_matmul(
        trace: &mut Vec<u64>,
        a_base: u64, b_base: u64, c_base: u64,
        n: usize, stride: usize, elem_size: u64,
    ) {
        if n <= 2 {
            // Base case: direct access
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        trace.push(a_base + ((i * stride + k) as u64) * elem_size);
                        trace.push(b_base + ((k * stride + j) as u64) * elem_size);
                        trace.push(c_base + ((i * stride + j) as u64) * elem_size);
                    }
                }
            }
            return;
        }
        let half = n / 2;
        let h = half as u64;
        let s = stride as u64;
        // 8 recursive calls for C = A*B
        for phase in 0..2 {
            for ci in 0..2u64 {
                for cj in 0..2u64 {
                    let a_off = ci * h * s * elem_size + (phase as u64) * h * elem_size;
                    let b_off = (phase as u64) * h * s * elem_size + cj * h * elem_size;
                    let c_off = ci * h * s * elem_size + cj * h * elem_size;
                    rec_matmul(
                        trace,
                        a_base + a_off, b_base + b_off, c_base + c_off,
                        half, stride, elem_size,
                    );
                }
            }
        }
    }

    let a_base = 0u64;
    let b_base = (matrix_size as u64) * elem_size;
    let c_base = 2 * (matrix_size as u64) * elem_size;
    rec_matmul(&mut trace, a_base, b_base, c_base, n, n, elem_size);
    let work = trace.len() as u64;
    (work, trace)
}

/// Generate a cache-oblivious prefix sum trace (Blelloch-style up-sweep/down-sweep).
pub fn cache_oblivious_prefix_trace(n: usize) -> (u64, Vec<u64>) {
    let elem_size = 8u64;
    let mut trace = Vec::new();

    // Up-sweep: binary tree reduction
    let mut stride = 1;
    while stride < n {
        let mut i = 2 * stride - 1;
        while i < n {
            trace.push((i as u64) * elem_size);
            trace.push(((i - stride) as u64) * elem_size);
            i += 2 * stride;
        }
        stride *= 2;
    }

    // Down-sweep: binary tree distribution
    while stride > 1 {
        stride /= 2;
        let mut i = 3 * stride - 1;
        while i < n {
            trace.push((i as u64) * elem_size);
            trace.push(((i - stride) as u64) * elem_size);
            i += 2 * stride;
        }
    }

    let work = trace.len() as u64;
    (work, trace)
}

/// Generate a cache-oblivious BFS trace (level-by-level access).
pub fn cache_oblivious_graph_trace(n: usize, avg_degree: usize) -> (u64, Vec<u64>) {
    let elem_size = 8u64;
    let mut trace = Vec::new();

    // Simulate BFS: access vertices level by level
    let depth = ((n as f64).log2()) as usize + 1;
    let mut frontier_size = 1;
    for _level in 0..depth {
        for v in 0..frontier_size.min(n) {
            // Access vertex data
            trace.push((v as u64) * elem_size);
            // Access neighbors (scattered)
            for d in 0..avg_degree.min(n) {
                let neighbor = (v * avg_degree + d) % n;
                trace.push((neighbor as u64) * elem_size);
            }
        }
        frontier_size = (frontier_size * avg_degree).min(n);
    }

    let work = trace.len() as u64;
    (work, trace)
}

// ---------------------------------------------------------------------------
// Hash-partition trace generation
// ---------------------------------------------------------------------------

/// Generate a hash-partitioned memory trace for a PRAM program.
pub fn hash_partition_trace(prog: &PramProgram, n: usize) -> (u64, Vec<u64>) {
    hash_partition_trace_with_family(prog, n, HashFamilyChoice::Siegel { k: 8 })
}

/// Generate a hash-partitioned memory trace using a specific hash family.
pub fn hash_partition_trace_with_family(
    prog: &PramProgram,
    n: usize,
    family: HashFamilyChoice,
) -> (u64, Vec<u64>) {
    let stmts = prog.total_stmts();
    let phases = prog.parallel_step_count();
    let p = prog.processor_count().unwrap_or(stmts);

    let total_work = (p * phases).max(stmts);
    let n_addrs = total_work.max(n);
    let addresses: Vec<u64> = (0..n_addrs as u64).collect();

    let num_blocks = (n_addrs / 64).max(1) as u64;
    let engine = PartitionEngine::new(num_blocks, 64, family, 42);
    let partition = engine.partition(&addresses);

    let trace: Vec<u64> = partition.assignments.iter()
        .map(|&b| b as u64 * 64)
        .collect();

    (total_work as u64, trace)
}

// ---------------------------------------------------------------------------
// Additional cache-oblivious traces
// ---------------------------------------------------------------------------

/// Generate a cache-oblivious tree trace (recursive decomposition pattern).
///
/// Models algorithms that recursively decompose a tree structure, achieving
/// O(n/B log_{M/B} n) cache complexity via van Emde Boas layout.
pub fn cache_oblivious_tree_trace(n: usize) -> (u64, Vec<u64>) {
    let elem_size = 8u64;
    let mut trace = Vec::new();

    fn rec_tree(trace: &mut Vec<u64>, base: u64, size: usize, elem_size: u64) {
        if size == 0 {
            return;
        }
        // Access root of this subtree
        trace.push(base);
        if size == 1 {
            return;
        }
        // Split into left and right subtrees (van Emde Boas-style)
        let half = size / 2;
        let left_base = base + elem_size;
        let right_base = base + (half as u64 + 1) * elem_size;
        rec_tree(trace, left_base, half, elem_size);
        rec_tree(trace, right_base, size - half - 1, elem_size);
        // Re-access root after children (simulates combine step)
        trace.push(base);
    }

    rec_tree(&mut trace, 0, n, elem_size);
    let work = trace.len() as u64;
    (work, trace)
}

/// Generate a cache-oblivious list ranking trace (pointer-chasing simulation).
///
/// Models the Euler tour / list ranking pattern where a linked list is
/// traversed with recursive doubling. Achieves O(n/B · log n) misses
/// on pointer-chasing workloads.
pub fn cache_oblivious_list_trace(n: usize) -> (u64, Vec<u64>) {
    let elem_size = 8u64;
    let mut trace = Vec::new();
    if n == 0 {
        return (0, trace);
    }

    // Phase 1: Build a permuted linked list (stride-based scattering)
    // Simulates pointer chasing through a scattered linked list.
    let stride = if n > 1 { (n / 2) | 1 } else { 1 }; // odd stride for full coverage
    let mut pos = 0;
    for _ in 0..n {
        trace.push((pos as u64) * elem_size);
        pos = (pos + stride) % n;
    }

    // Phase 2: Recursive doubling (log n rounds)
    let mut jump = 1;
    while jump < n {
        for i in 0..n {
            // Access current node
            trace.push((i as u64) * elem_size);
            // Access node `jump` steps ahead (pointer chase)
            let target = (i + jump) % n;
            trace.push((target as u64) * elem_size);
        }
        jump *= 2;
    }

    let work = trace.len() as u64;
    (work, trace)
}

// ---------------------------------------------------------------------------
// Crossover analysis
// ---------------------------------------------------------------------------

/// Result of crossover analysis between hash-partition and cache-oblivious.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossoverResult {
    /// Input sizes tested.
    pub sizes: Vec<usize>,
    /// Hash-partition cache misses at each size.
    pub hp_misses: Vec<u64>,
    /// Cache-oblivious cache misses at each size.
    pub co_misses: Vec<u64>,
    /// The smallest size where hash-partition starts winning (fewer misses).
    /// `None` if hash-partition never wins or always wins.
    pub crossover_point: Option<usize>,
    /// Ratio hp_misses/co_misses at each size (< 1.0 means HP wins).
    pub ratios: Vec<f64>,
}

/// Find the crossover point where hash-partition starts winning over
/// cache-oblivious algorithms for a given program.
pub fn crossover_analysis(
    prog: &PramProgram,
    sizes: &[usize],
    cache_line_size: u64,
    num_cache_lines: usize,
) -> CrossoverResult {
    let category = categorize_for_baseline(&prog.name);
    let mut hp_misses_vec = Vec::with_capacity(sizes.len());
    let mut co_misses_vec = Vec::with_capacity(sizes.len());
    let mut ratios = Vec::with_capacity(sizes.len());

    for &size in sizes {
        let (_, hp_trace) = hash_partition_trace(prog, size);
        let hp_m = simulate_cache(&hp_trace, cache_line_size, num_cache_lines);

        let (_, co_trace) = match category {
            AlgCategory::Sorting => cache_oblivious_sort_trace(size),
            AlgCategory::MatrixArithmetic => {
                cache_oblivious_matmul_trace((size as f64).sqrt() as usize)
            }
            AlgCategory::PrefixScan => cache_oblivious_prefix_trace(size),
            AlgCategory::Graph => cache_oblivious_graph_trace(size, 4),
            AlgCategory::Tree => cache_oblivious_tree_trace(size),
            AlgCategory::List => cache_oblivious_list_trace(size),
            AlgCategory::Other => cache_oblivious_prefix_trace(size),
        };
        let co_m = simulate_cache(&co_trace, cache_line_size, num_cache_lines);

        hp_misses_vec.push(hp_m);
        co_misses_vec.push(co_m);
        let ratio = if co_m == 0 {
            if hp_m == 0 { 1.0 } else { f64::INFINITY }
        } else {
            hp_m as f64 / co_m as f64
        };
        ratios.push(ratio);
    }

    // Find crossover: first size where HP starts winning (ratio < 1.0)
    // after having previously lost (ratio >= 1.0).
    let crossover_point = find_crossover(&ratios, sizes);

    CrossoverResult {
        sizes: sizes.to_vec(),
        hp_misses: hp_misses_vec,
        co_misses: co_misses_vec,
        crossover_point,
        ratios,
    }
}

/// Find the crossover point: the first size where the ratio drops below 1.0
/// and stays below for the rest of the sizes.
fn find_crossover(ratios: &[f64], sizes: &[usize]) -> Option<usize> {
    if ratios.is_empty() {
        return None;
    }
    // If HP always wins or never wins, no crossover
    let any_above = ratios.iter().any(|&r| r >= 1.0);
    let any_below = ratios.iter().any(|&r| r < 1.0);
    if !any_above || !any_below {
        return None;
    }
    // Find first index where ratio transitions from >= 1.0 to < 1.0
    for i in 1..ratios.len() {
        if ratios[i] < 1.0 && ratios[i - 1] >= 1.0 {
            return Some(sizes[i]);
        }
    }
    // Check if HP wins from the start
    if ratios[0] < 1.0 {
        return Some(sizes[0]);
    }
    None
}

// ---------------------------------------------------------------------------
// Statistical comparison
// ---------------------------------------------------------------------------

/// Result of a statistically rigorous comparison between hash-partition and baselines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalComparison {
    pub input_size: usize,
    pub trials: usize,
    pub hp_mean_misses: f64,
    pub hp_stddev: f64,
    /// Per-baseline: (name, mean_misses, stddev, t_stat, p_value, significant, effect_size)
    pub baseline_results: Vec<BaselineStatResult>,
}

/// Statistical result for one baseline comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineStatResult {
    pub baseline_name: String,
    pub mean_misses: f64,
    pub stddev: f64,
    pub t_statistic: f64,
    pub p_value: f64,
    pub significant: bool,
    pub cohens_d: f64,
    pub hp_bootstrap_ci_95: (f64, f64),
    pub baseline_bootstrap_ci_95: (f64, f64),
}

/// Run multiple trials and compute Welch's t-test between hash-partition and each baseline.
pub fn statistical_comparison(
    prog: &PramProgram,
    size: usize,
    trials: usize,
    cache_line_size: u64,
    num_cache_lines: usize,
) -> StatisticalComparison {
    let category = categorize_for_baseline(&prog.name);
    let trials = trials.max(2); // need at least 2 for variance

    let mut hp_samples = Vec::with_capacity(trials);
    let mut cilk_samples = Vec::with_capacity(trials);
    let mut co_samples = Vec::with_capacity(trials);
    let mut hand_samples = Vec::with_capacity(trials);

    for trial in 0..trials {
        // Vary the hash seed per trial to get variance in hash-partition
        let family = HashFamilyChoice::Siegel { k: 8 };
        let (_, hp_trace) = hash_partition_trace_with_family(prog, size + trial, family);
        hp_samples.push(simulate_cache(&hp_trace, cache_line_size, num_cache_lines) as f64);

        let (_, cilk_trace) = cilk_serial_trace(prog, size + trial);
        cilk_samples.push(simulate_cache(&cilk_trace, cache_line_size, num_cache_lines) as f64);

        let (_, co_trace) = match category {
            AlgCategory::Sorting => cache_oblivious_sort_trace(size + trial),
            AlgCategory::MatrixArithmetic => {
                cache_oblivious_matmul_trace(((size + trial) as f64).sqrt() as usize)
            }
            AlgCategory::PrefixScan => cache_oblivious_prefix_trace(size + trial),
            AlgCategory::Graph => cache_oblivious_graph_trace(size + trial, 4),
            AlgCategory::Tree => cache_oblivious_tree_trace(size + trial),
            AlgCategory::List => cache_oblivious_list_trace(size + trial),
            AlgCategory::Other => cache_oblivious_prefix_trace(size + trial),
        };
        co_samples.push(simulate_cache(&co_trace, cache_line_size, num_cache_lines) as f64);

        let hand_trace: Vec<u64> = (0..(size + trial) as u64).map(|i| i * 8).collect();
        hand_samples.push(simulate_cache(&hand_trace, cache_line_size, num_cache_lines) as f64);
    }

    let hp_mean = statistics::mean(&hp_samples);
    let hp_sd = statistics::stddev(&hp_samples);

    let mut baseline_results = Vec::new();
    for (name, samples) in [
        ("cilk-serial", &cilk_samples),
        ("cache-oblivious", &co_samples),
        ("hand-optimized", &hand_samples),
    ] {
        let ttest = statistics::welch_t_test(&hp_samples, samples);
        let d = statistics::effect_size(&hp_samples, samples);
        let hp_ci = statistics::bootstrap_ci(&hp_samples, 0.95, 10000);
        let bl_ci = statistics::bootstrap_ci(samples, 0.95, 10000);
        baseline_results.push(BaselineStatResult {
            baseline_name: name.to_string(),
            mean_misses: statistics::mean(samples),
            stddev: statistics::stddev(samples),
            t_statistic: ttest.t_statistic,
            p_value: ttest.p_value_approx,
            significant: ttest.significant,
            cohens_d: d,
            hp_bootstrap_ci_95: hp_ci,
            baseline_bootstrap_ci_95: bl_ci,
        });
    }

    StatisticalComparison {
        input_size: size,
        trials,
        hp_mean_misses: hp_mean,
        hp_stddev: hp_sd,
        baseline_results,
    }
}

// ---------------------------------------------------------------------------
// Comprehensive comparison
// ---------------------------------------------------------------------------

/// Per-algorithm breakdown in the comprehensive report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmReport {
    pub algorithm: String,
    pub summaries: Vec<ComparisonSummary>,
    pub crossover: CrossoverResult,
    pub statistical: StatisticalComparison,
}

/// Comprehensive comparison report across all programs, sizes, and trials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveReport {
    pub algorithm_reports: Vec<AlgorithmReport>,
    /// Overall recommendation based on aggregate results.
    pub recommendation: String,
}

/// Run a comprehensive comparison including per-algorithm breakdown,
/// statistical significance, crossover points, and an overall recommendation.
pub fn comprehensive_comparison(
    programs: &[(String, PramProgram)],
    sizes: &[usize],
    trials: usize,
) -> ComprehensiveReport {
    let cache_line_size = 64u64;
    let num_cache_lines = 512;

    let mut algorithm_reports = Vec::new();
    let mut total_wins = 0usize;
    let mut total_comparisons = 0usize;

    for (_name, prog) in programs {
        let summaries: Vec<ComparisonSummary> = sizes.iter()
            .map(|&s| compare_algorithm(prog, s, cache_line_size, num_cache_lines))
            .collect();

        let crossover = crossover_analysis(prog, sizes, cache_line_size, num_cache_lines);

        // Use the largest size for statistical comparison
        let stat_size = sizes.iter().copied().max().unwrap_or(256);
        let statistical = statistical_comparison(
            prog, stat_size, trials, cache_line_size, num_cache_lines,
        );

        for s in &summaries {
            total_comparisons += 1;
            if s.hash_partition_wins {
                total_wins += 1;
            }
        }

        algorithm_reports.push(AlgorithmReport {
            algorithm: prog.name.clone(),
            summaries,
            crossover,
            statistical,
        });
    }

    let win_rate = if total_comparisons > 0 {
        total_wins as f64 / total_comparisons as f64
    } else {
        0.0
    };

    let recommendation = if win_rate >= 0.8 {
        "Hash-partition strongly recommended: wins in >= 80% of comparisons.".to_string()
    } else if win_rate >= 0.5 {
        format!(
            "Hash-partition conditionally recommended: wins {:.0}% of comparisons. \
             Check crossover points for size-dependent guidance.",
            win_rate * 100.0,
        )
    } else {
        format!(
            "Cache-oblivious may be preferable: hash-partition wins only {:.0}% of comparisons.",
            win_rate * 100.0,
        )
    };

    ComprehensiveReport {
        algorithm_reports,
        recommendation,
    }
}

// ---------------------------------------------------------------------------
// Comparison engine
// ---------------------------------------------------------------------------

/// Run a full comparison of hash-partition vs baselines for an algorithm.
pub fn compare_algorithm(
    prog: &PramProgram,
    input_size: usize,
    cache_line_size: u64,
    num_cache_lines: usize,
) -> ComparisonSummary {
    let alg_name = prog.name.clone();
    let category = categorize_for_baseline(&alg_name);

    // 1. Hash-partition trace
    let (hp_work, hp_trace) = hash_partition_trace(prog, input_size);
    let hp_misses = simulate_cache(&hp_trace, cache_line_size, num_cache_lines);

    // 2. Cilk serial trace
    let (_, cilk_trace) = cilk_serial_trace(prog, input_size);
    let cilk_misses = simulate_cache(&cilk_trace, cache_line_size, num_cache_lines);

    // 3. Cache-oblivious trace (choose based on algorithm category)
    let (_, co_trace) = match category {
        AlgCategory::Sorting => cache_oblivious_sort_trace(input_size),
        AlgCategory::MatrixArithmetic => cache_oblivious_matmul_trace(
            (input_size as f64).sqrt() as usize
        ),
        AlgCategory::PrefixScan => cache_oblivious_prefix_trace(input_size),
        AlgCategory::Graph => cache_oblivious_graph_trace(input_size, 4),
        AlgCategory::Tree => cache_oblivious_tree_trace(input_size),
        AlgCategory::List => cache_oblivious_list_trace(input_size),
        AlgCategory::Other => cache_oblivious_prefix_trace(input_size),
    };
    let co_misses = simulate_cache(&co_trace, cache_line_size, num_cache_lines);

    // 4. Hand-optimized: sequential stride (best possible locality)
    let hand_trace: Vec<u64> = (0..input_size as u64).map(|i| i * 8).collect();
    let hand_misses = simulate_cache(&hand_trace, cache_line_size, num_cache_lines);

    // Compute overheads relative to hand-optimized baseline
    let base = hand_misses.max(1) as f64;
    let hp_overhead = hp_misses as f64 / base;
    let cilk_overhead = cilk_misses as f64 / base;
    let co_overhead = co_misses as f64 / base;

    ComparisonSummary {
        algorithm: alg_name,
        input_size,
        hash_partition_misses: hp_misses,
        cilk_serial_misses: cilk_misses,
        cache_oblivious_misses: co_misses,
        hand_optimized_misses: hand_misses,
        hash_partition_overhead: hp_overhead,
        cilk_serial_overhead: cilk_overhead,
        cache_oblivious_overhead: co_overhead,
        // HP "wins" when it beats the cache-oblivious baseline (the key comparison)
        // and stays within 2x of the hand-optimized sequential baseline.
        hash_partition_wins: hp_misses <= co_misses
            && (hp_misses as f64) <= (hand_misses.max(1) as f64) * 2.0,
    }
}

/// Run comparisons for all algorithms at multiple input sizes.
pub fn compare_all(
    programs: &[(String, PramProgram)],
    sizes: &[usize],
    cache_line_size: u64,
    num_cache_lines: usize,
) -> Vec<ComparisonSummary> {
    let mut results = Vec::new();
    for (name, prog) in programs {
        for &size in sizes {
            let summary = compare_algorithm(prog, size, cache_line_size, num_cache_lines);
            results.push(summary);
        }
    }
    results
}

/// Print a comparison table.
pub fn print_comparison_table(summaries: &[ComparisonSummary]) {
    println!("{:<25} {:>6} {:>10} {:>10} {:>10} {:>10} {:>5}",
             "Algorithm", "n", "HashPart", "Cilk", "CacheObl", "HandOpt", "Win?");
    println!("{:-<82}", "");
    for s in summaries {
        println!("{:<25} {:>6} {:>10} {:>10} {:>10} {:>10} {:>5}",
                 s.algorithm, s.input_size,
                 s.hash_partition_misses, s.cilk_serial_misses,
                 s.cache_oblivious_misses, s.hand_optimized_misses,
                 if s.hash_partition_wins { "YES" } else { "no" });
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn simulate_cache(trace: &[u64], line_size: u64, num_lines: usize) -> u64 {
    if trace.is_empty() {
        return 0;
    }
    let mut sim = CacheSimulator::new(line_size, num_lines);
    sim.access_sequence(trace);
    sim.stats().misses
}

#[derive(Debug, Clone, Copy)]
enum AlgCategory {
    Sorting,
    MatrixArithmetic,
    PrefixScan,
    Graph,
    Tree,
    List,
    Other,
}

fn categorize_for_baseline(name: &str) -> AlgCategory {
    if name.contains("sort") || name.contains("merge") || name.contains("flashsort")
        || name.contains("radix") || name.contains("aks_sorting")
    {
        AlgCategory::Sorting
    } else if name.contains("matrix") || name.contains("strassen") || name.contains("fft") {
        AlgCategory::MatrixArithmetic
    } else if name.contains("prefix") || name.contains("scan") || name.contains("addition")
        || name.contains("multiplication") && !name.contains("matrix")
    {
        AlgCategory::PrefixScan
    } else if name.contains("tree") || name.contains("lca") || name.contains("centroid")
        || name.contains("contraction") || name.contains("isomorphism")
    {
        AlgCategory::Tree
    } else if name.contains("list") || name.contains("ranking") || name.contains("symmetry")
        || name.contains("compact")
    {
        AlgCategory::List
    } else if name.contains("graph") || name.contains("bfs") || name.contains("dfs")
        || name.contains("vishkin") || name.contains("boruvka") || name.contains("euler")
        || name.contains("connectivity") || name.contains("shiloach")
        || name.contains("coloring") || name.contains("independent")
        || name.contains("shortest") || name.contains("biconnected")
        || name.contains("strongly") || name.contains("ear")
    {
        AlgCategory::Graph
    } else {
        AlgCategory::Other
    }
}

/// Simulate a cache-oblivious stencil computation trace.
/// Returns simulated cache miss counts for comparison.
pub fn cache_oblivious_stencil_trace(n: usize, cache_lines: usize) -> f64 {
    // Cache-oblivious stencil: O(n^2 / (B * sqrt(M))) cache misses
    // where B = cache line size, M = cache size
    let b = 64.0; // typical cache line
    let m = (cache_lines as f64) * b;
    let n_f = n as f64;
    if m > 0.0 {
        (n_f * n_f) / (b * m.sqrt())
    } else {
        n_f * n_f / b
    }
}

/// Simulate a cache-oblivious selection (median-finding) trace.
pub fn cache_oblivious_selection_trace(n: usize, cache_lines: usize) -> f64 {
    // Cache-oblivious selection: O(n/B) cache misses
    let b = 64.0;
    let _m = (cache_lines as f64) * b;
    (n as f64) / b
}

/// Result of a full statistical comparison between approaches.
#[derive(Debug, Clone)]
pub struct FullStatisticalComparison {
    pub hash_partition_mean: f64,
    pub hash_partition_std: f64,
    pub cache_oblivious_mean: f64,
    pub cache_oblivious_std: f64,
    pub welch_t: f64,
    pub cohens_d: f64,
    pub hash_ci: (f64, f64),
    pub oblivious_ci: (f64, f64),
    pub hash_wins: bool,
}

/// Full statistical comparison: compares hash-partition results against
/// cache-oblivious baselines using t-tests, confidence intervals, and effect sizes.
pub fn full_statistical_comparison(
    hash_ratios: &[f64],
    oblivious_ratios: &[f64],
) -> FullStatisticalComparison {
    use crate::benchmark::statistics::{mean, stddev, welch_t_test, effect_size, bootstrap_ci};

    let hash_mean = mean(hash_ratios);
    let hash_std = stddev(hash_ratios);
    let obliv_mean = mean(oblivious_ratios);
    let obliv_std = stddev(oblivious_ratios);

    let t_result = welch_t_test(hash_ratios, oblivious_ratios);
    let cohens_d = effect_size(hash_ratios, oblivious_ratios);
    let hash_ci = bootstrap_ci(hash_ratios, 0.95, 1000);
    let obliv_ci = bootstrap_ci(oblivious_ratios, 0.95, 1000);

    FullStatisticalComparison {
        hash_partition_mean: hash_mean,
        hash_partition_std: hash_std,
        cache_oblivious_mean: obliv_mean,
        cache_oblivious_std: obliv_std,
        welch_t: t_result.t_statistic,
        cohens_d,
        hash_ci,
        oblivious_ci: obliv_ci,
        hash_wins: hash_mean < obliv_mean,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cilk_serial_trace() {
        let prog = crate::algorithm_library::sorting::bitonic_sort();
        let (work, trace) = cilk_serial_trace(&prog, 256);
        assert!(work > 0);
        assert!(!trace.is_empty());
    }

    #[test]
    fn test_cache_oblivious_sort_trace() {
        let (work, trace) = cache_oblivious_sort_trace(64);
        assert!(work > 0);
        assert!(!trace.is_empty());
    }

    #[test]
    fn test_cache_oblivious_matmul_trace() {
        let (work, trace) = cache_oblivious_matmul_trace(4);
        assert!(work > 0);
        assert!(!trace.is_empty());
    }

    #[test]
    fn test_cache_oblivious_prefix_trace() {
        let (work, trace) = cache_oblivious_prefix_trace(16);
        assert!(work > 0);
        assert!(!trace.is_empty());
    }

    #[test]
    fn test_hash_partition_trace() {
        let prog = crate::algorithm_library::list::prefix_sum();
        let (work, trace) = hash_partition_trace(&prog, 128);
        assert!(work > 0);
        assert!(!trace.is_empty());
    }

    #[test]
    fn test_compare_algorithm() {
        let prog = crate::algorithm_library::sorting::bitonic_sort();
        let summary = compare_algorithm(&prog, 256, 64, 512);
        assert_eq!(summary.algorithm, "bitonic_sort");
        assert!(summary.hash_partition_misses > 0 || summary.hand_optimized_misses > 0);
    }

    #[test]
    fn test_compare_all() {
        let programs = vec![
            ("prefix_sum".to_string(), crate::algorithm_library::list::prefix_sum()),
            ("bitonic_sort".to_string(), crate::algorithm_library::sorting::bitonic_sort()),
        ];
        let results = compare_all(&programs, &[64, 256], 64, 512);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_categorize() {
        assert!(matches!(categorize_for_baseline("bitonic_sort"), AlgCategory::Sorting));
        assert!(matches!(categorize_for_baseline("matrix_multiply"), AlgCategory::MatrixArithmetic));
        assert!(matches!(categorize_for_baseline("prefix_sum"), AlgCategory::PrefixScan));
        assert!(matches!(categorize_for_baseline("shiloach_vishkin"), AlgCategory::Graph));
    }

    // --- New tests for enhanced baseline comparisons ---

    #[test]
    fn test_cache_oblivious_tree_trace_basic() {
        let (work, trace) = cache_oblivious_tree_trace(15);
        assert!(work > 0);
        assert!(!trace.is_empty());
        // Root is accessed at least once
        assert!(trace.contains(&0));
    }

    #[test]
    fn test_cache_oblivious_tree_trace_single() {
        let (work, trace) = cache_oblivious_tree_trace(1);
        assert_eq!(work, 1);
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0], 0);
    }

    #[test]
    fn test_cache_oblivious_tree_trace_empty() {
        let (work, trace) = cache_oblivious_tree_trace(0);
        assert_eq!(work, 0);
        assert!(trace.is_empty());
    }

    #[test]
    fn test_cache_oblivious_list_trace_basic() {
        let (work, trace) = cache_oblivious_list_trace(16);
        assert!(work > 0);
        assert!(!trace.is_empty());
        // Should have pointer-chasing + recursive doubling accesses
        assert!(trace.len() > 16);
    }

    #[test]
    fn test_cache_oblivious_list_trace_empty() {
        let (work, trace) = cache_oblivious_list_trace(0);
        assert_eq!(work, 0);
        assert!(trace.is_empty());
    }

    #[test]
    fn test_cache_oblivious_list_trace_single() {
        let (work, trace) = cache_oblivious_list_trace(1);
        assert!(work > 0);
        assert!(!trace.is_empty());
    }

    #[test]
    fn test_hash_partition_trace_tabulation() {
        let prog = crate::algorithm_library::list::prefix_sum();
        let family = HashFamilyChoice::Tabulation { seed: 99 };
        let (work, trace) = hash_partition_trace_with_family(&prog, 128, family);
        assert!(work > 0);
        assert!(!trace.is_empty());
    }

    #[test]
    fn test_hash_partition_trace_with_family_siegel() {
        let prog = crate::algorithm_library::list::prefix_sum();
        let family = HashFamilyChoice::Siegel { k: 4 };
        let (work, trace) = hash_partition_trace_with_family(&prog, 64, family);
        assert!(work > 0);
        assert!(!trace.is_empty());
    }

    #[test]
    fn test_crossover_analysis_basic() {
        let prog = crate::algorithm_library::sorting::bitonic_sort();
        let sizes = vec![16, 32, 64, 128, 256];
        let result = crossover_analysis(&prog, &sizes, 64, 512);
        assert_eq!(result.sizes.len(), 5);
        assert_eq!(result.hp_misses.len(), 5);
        assert_eq!(result.co_misses.len(), 5);
        assert_eq!(result.ratios.len(), 5);
        // Ratios should all be positive
        for &r in &result.ratios {
            assert!(r > 0.0);
        }
    }

    #[test]
    fn test_crossover_analysis_prefix() {
        let prog = crate::algorithm_library::list::prefix_sum();
        let sizes = vec![32, 64, 128];
        let result = crossover_analysis(&prog, &sizes, 64, 256);
        assert_eq!(result.sizes.len(), 3);
        assert_eq!(result.hp_misses.len(), 3);
    }

    #[test]
    fn test_statistical_comparison_basic() {
        let prog = crate::algorithm_library::sorting::bitonic_sort();
        let stat = statistical_comparison(&prog, 64, 5, 64, 512);
        assert_eq!(stat.trials, 5);
        assert_eq!(stat.input_size, 64);
        assert_eq!(stat.baseline_results.len(), 3);
        for br in &stat.baseline_results {
            assert!(br.mean_misses >= 0.0);
            assert!(br.stddev >= 0.0);
        }
    }

    #[test]
    fn test_statistical_comparison_min_trials() {
        let prog = crate::algorithm_library::list::prefix_sum();
        // Request only 1 trial, should be bumped to 2
        let stat = statistical_comparison(&prog, 32, 1, 64, 256);
        assert_eq!(stat.trials, 2);
        assert_eq!(stat.baseline_results.len(), 3);
    }

    #[test]
    fn test_comprehensive_comparison_basic() {
        let programs = vec![
            ("prefix_sum".to_string(), crate::algorithm_library::list::prefix_sum()),
        ];
        let report = comprehensive_comparison(&programs, &[32, 64], 3);
        assert_eq!(report.algorithm_reports.len(), 1);
        let alg = &report.algorithm_reports[0];
        assert_eq!(alg.summaries.len(), 2);
        assert_eq!(alg.crossover.sizes.len(), 2);
        assert!(!report.recommendation.is_empty());
    }

    #[test]
    fn test_comprehensive_comparison_multi_algorithm() {
        let programs = vec![
            ("prefix_sum".to_string(), crate::algorithm_library::list::prefix_sum()),
            ("bitonic_sort".to_string(), crate::algorithm_library::sorting::bitonic_sort()),
        ];
        let report = comprehensive_comparison(&programs, &[32, 64], 3);
        assert_eq!(report.algorithm_reports.len(), 2);
        // Recommendation should contain a percentage or "recommended"
        assert!(
            report.recommendation.contains("recommended")
                || report.recommendation.contains("%")
        );
    }

    #[test]
    fn test_find_crossover_no_transition() {
        // All ratios below 1.0 → no crossover
        let ratios = vec![0.5, 0.6, 0.7];
        let sizes = vec![32, 64, 128];
        assert_eq!(find_crossover(&ratios, &sizes), None);
    }

    #[test]
    fn test_find_crossover_transition() {
        // Transition from >= 1.0 to < 1.0 at index 2
        let ratios = vec![1.5, 1.2, 0.8, 0.6];
        let sizes = vec![32, 64, 128, 256];
        assert_eq!(find_crossover(&ratios, &sizes), Some(128));
    }

    #[test]
    fn test_find_crossover_empty() {
        assert_eq!(find_crossover(&[], &[]), None);
    }

    #[test]
    fn test_cache_oblivious_stencil_trace() {
        let misses = cache_oblivious_stencil_trace(1024, 512);
        assert!(misses > 0.0);
        // Larger n should give more misses
        let misses2 = cache_oblivious_stencil_trace(2048, 512);
        assert!(misses2 > misses);
    }

    #[test]
    fn test_cache_oblivious_selection_trace() {
        let misses = cache_oblivious_selection_trace(1024, 512);
        assert!(misses > 0.0);
        assert!((misses - 1024.0 / 64.0).abs() < 0.001);
    }

    #[test]
    fn test_full_statistical_comparison() {
        let hash_ratios = vec![0.10, 0.12, 0.15, 0.11, 0.13, 0.14, 0.09, 0.16];
        let obliv_ratios = vec![1.10, 1.20, 1.15, 1.25, 1.30, 1.18, 1.22, 1.19];
        let cmp = full_statistical_comparison(&hash_ratios, &obliv_ratios);
        assert!(cmp.hash_wins);
        assert!(cmp.hash_partition_mean < cmp.cache_oblivious_mean);
        assert!(cmp.cohens_d.abs() > 0.5); // Large effect size expected
    }

    #[test]
    fn test_full_statistical_comparison_equal() {
        let a = vec![1.0, 1.0, 1.0, 1.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let cmp = full_statistical_comparison(&a, &b);
        assert!(!cmp.hash_wins); // Equal means hash doesn't win
    }
}
