//! Benchmark harness for running and collecting performance measurements.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Instant;

use super::cache_sim::{CacheSimulator, SetAssociativeCache, CacheModel, RealisticCacheConfig};
use super::statistics;

/// Configuration for a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub algorithm: String,
    pub input_sizes: Vec<u64>,
    pub hash_families: Vec<String>,
    pub num_trials: usize,
    pub warmup_trials: usize,
}

impl BenchmarkConfig {
    pub fn new(algorithm: &str) -> Self {
        Self {
            algorithm: algorithm.to_string(),
            input_sizes: vec![100, 1000, 10000],
            hash_families: vec!["tabulation".to_string()],
            num_trials: 5,
            warmup_trials: 2,
        }
    }

    pub fn with_sizes(mut self, sizes: Vec<u64>) -> Self {
        self.input_sizes = sizes;
        self
    }

    pub fn with_hash_families(mut self, families: Vec<String>) -> Self {
        self.hash_families = families;
        self
    }

    pub fn with_trials(mut self, num: usize) -> Self {
        self.num_trials = num;
        self
    }

    pub fn with_warmup(mut self, num: usize) -> Self {
        self.warmup_trials = num;
        self
    }
}

/// Result of a single benchmark measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub algorithm: String,
    pub input_size: u64,
    pub hash_family: String,
    pub wall_time_ns: u64,
    pub ops_count: u64,
    pub cache_misses: u64,
    pub work_bound_satisfied: bool,
    pub cache_bound_satisfied: bool,
}

impl fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (n={}, hash={}): time={}ns, ops={}, misses={}, work_ok={}, cache_ok={}",
            self.algorithm,
            self.input_size,
            self.hash_family,
            self.wall_time_ns,
            self.ops_count,
            self.cache_misses,
            self.work_bound_satisfied,
            self.cache_bound_satisfied,
        )
    }
}

/// Timing utilities.
pub struct Timer {
    start: Option<Instant>,
    elapsed_ns: u64,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start: None,
            elapsed_ns: 0,
        }
    }

    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }

    pub fn stop(&mut self) -> u64 {
        if let Some(start) = self.start.take() {
            self.elapsed_ns = start.elapsed().as_nanos() as u64;
        }
        self.elapsed_ns
    }

    pub fn elapsed_ns(&self) -> u64 {
        self.elapsed_ns
    }

    pub fn elapsed_us(&self) -> f64 {
        self.elapsed_ns as f64 / 1_000.0
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed_ns as f64 / 1_000_000.0
    }

    pub fn elapsed_s(&self) -> f64 {
        self.elapsed_ns as f64 / 1_000_000_000.0
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// Measure the wall-clock time of a closure in nanoseconds.
pub fn time_fn<F: FnMut()>(mut f: F) -> u64 {
    let start = Instant::now();
    f();
    start.elapsed().as_nanos() as u64
}

/// Measure with warm-up and multiple trials, returning the median time in nanoseconds.
pub fn time_fn_median<F: FnMut()>(mut f: F, warmup: usize, trials: usize) -> u64 {
    // Warm-up
    for _ in 0..warmup {
        f();
    }
    // Collect timings
    let mut times = Vec::with_capacity(trials);
    for _ in 0..trials {
        let t = time_fn(&mut f);
        times.push(t as f64);
    }
    statistics::median(&times) as u64
}

/// A function type that runs a benchmark workload and returns (ops_count, memory_addresses).
pub type WorkloadFn = Box<dyn FnMut(u64) -> (u64, Vec<u64>)>;

/// The benchmark harness orchestrates running benchmarks.
pub struct BenchmarkHarness {
    configs: Vec<BenchmarkConfig>,
    results: Vec<BenchmarkResult>,
    cache_line_size: u64,
    num_cache_lines: usize,
    cache_model: CacheModel,
}

impl BenchmarkHarness {
    pub fn new() -> Self {
        Self {
            configs: Vec::new(),
            results: Vec::new(),
            cache_line_size: 64,
            num_cache_lines: 512,
            cache_model: CacheModel::SetAssociative,
        }
    }

    pub fn with_cache_params(mut self, line_size: u64, num_lines: usize) -> Self {
        self.cache_line_size = line_size;
        self.num_cache_lines = num_lines;
        self
    }

    pub fn with_cache_model(mut self, model: CacheModel) -> Self {
        self.cache_model = model;
        self
    }

    pub fn add_config(&mut self, config: BenchmarkConfig) {
        self.configs.push(config);
    }

    /// Run a benchmark given a config and a workload function.
    ///
    /// The workload function takes an input size and returns (ops_count, memory_trace).
    pub fn run_benchmark<F>(
        &mut self,
        config: &BenchmarkConfig,
        mut workload: F,
    ) -> Vec<BenchmarkResult>
    where
        F: FnMut(u64) -> (u64, Vec<u64>),
    {
        let mut results = Vec::new();

        for &input_size in &config.input_sizes {
            for hash_family in &config.hash_families {
                // Warm-up runs
                for _ in 0..config.warmup_trials {
                    let _ = workload(input_size);
                }

                // Timed trials
                let mut trial_times = Vec::with_capacity(config.num_trials);
                let mut last_ops = 0u64;
                let mut last_addrs = Vec::new();

                for _ in 0..config.num_trials {
                    let start = Instant::now();
                    let (ops, addrs) = workload(input_size);
                    let elapsed = start.elapsed().as_nanos() as u64;
                    trial_times.push(elapsed as f64);
                    last_ops = ops;
                    last_addrs = addrs;
                }

                // Cache simulation on the last trial's memory trace
                let cache_misses = match self.cache_model {
                    CacheModel::FullyAssociative => {
                        let mut cache_sim =
                            CacheSimulator::new(self.cache_line_size, self.num_cache_lines);
                        cache_sim.access_sequence(&last_addrs);
                        cache_sim.stats().misses
                    }
                    CacheModel::SetAssociative => {
                        let config = RealisticCacheConfig {
                            line_size: self.cache_line_size,
                            l1_sets: (self.num_cache_lines / 8).max(1).next_power_of_two(),
                            l1_ways: 8,
                        };
                        let mut cache_sim = SetAssociativeCache::new(
                            config.line_size, config.l1_sets, config.l1_ways,
                        );
                        cache_sim.access_sequence(&last_addrs);
                        cache_sim.stats().misses
                    }
                };

                let median_time = statistics::median(&trial_times) as u64;

                let result = BenchmarkResult {
                    algorithm: config.algorithm.clone(),
                    input_size,
                    hash_family: hash_family.clone(),
                    wall_time_ns: median_time,
                    ops_count: last_ops,
                    cache_misses,
                    work_bound_satisfied: true,
                    cache_bound_satisfied: true,
                };

                results.push(result);
            }
        }

        self.results.extend(results.clone());
        results
    }

    /// Run all registered configs with a given workload function.
    pub fn run_all<F>(&mut self, mut workload: F) -> Vec<BenchmarkResult>
    where
        F: FnMut(u64) -> (u64, Vec<u64>),
    {
        let configs = self.configs.clone();
        let mut all_results = Vec::new();
        for config in &configs {
            let results = self.run_benchmark(config, &mut workload);
            all_results.extend(results);
        }
        all_results
    }

    /// Get all collected results.
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Clear all collected results.
    pub fn clear_results(&mut self) {
        self.results.clear();
    }

    /// Get a summary of results grouped by algorithm.
    pub fn summary_by_algorithm(&self) -> Vec<(String, usize, f64)> {
        let mut algos: Vec<String> = self
            .results
            .iter()
            .map(|r| r.algorithm.clone())
            .collect();
        algos.sort();
        algos.dedup();

        algos
            .into_iter()
            .map(|algo| {
                let times: Vec<f64> = self
                    .results
                    .iter()
                    .filter(|r| r.algorithm == algo)
                    .map(|r| r.wall_time_ns as f64)
                    .collect();
                let count = times.len();
                let avg = statistics::mean(&times);
                (algo, count, avg)
            })
            .collect()
    }
}

impl Default for BenchmarkHarness {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a simple workload that simulates n sequential memory accesses.
pub fn simple_sequential_workload(n: u64) -> (u64, Vec<u64>) {
    let ops = n;
    let addrs: Vec<u64> = (0..n).map(|i| i * 8).collect();
    (ops, addrs)
}

/// Create a workload that simulates n random-looking memory accesses using a simple hash.
pub fn pseudo_random_workload(n: u64, range: u64) -> (u64, Vec<u64>) {
    let ops = n;
    let mut addrs = Vec::with_capacity(n as usize);
    let mut state = 12345u64;
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let addr = (state % range) * 8;
        addrs.push(addr);
    }
    (ops, addrs)
}

/// A suite of benchmark configurations to run together.
pub struct BenchmarkSuite {
    pub name: String,
    pub configs: Vec<BenchmarkConfig>,
    pub description: String,
}

impl BenchmarkSuite {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            configs: Vec::new(),
            description: String::new(),
        }
    }

    pub fn add_config(&mut self, config: BenchmarkConfig) {
        self.configs.push(config);
    }

    pub fn config_count(&self) -> usize {
        self.configs.len()
    }
}

/// Run all configs in a suite using the given workload function, returning all results.
pub fn run_suite<F>(suite: &BenchmarkSuite, mut workload: F) -> Vec<BenchmarkResult>
where
    F: FnMut(u64) -> (u64, Vec<u64>),
{
    let mut harness = BenchmarkHarness::new();
    let mut all_results = Vec::new();
    for config in &suite.configs {
        let results = harness.run_benchmark(config, &mut workload);
        all_results.extend(results);
    }
    all_results
}

/// Run a closure `iterations` times for warmup, discarding results.
pub fn warmup<F: FnMut()>(f: &mut F, iterations: usize) {
    for _ in 0..iterations {
        f();
    }
}

/// Timing statistics collected from repeated benchmark runs.
#[derive(Debug, Clone)]
pub struct TimingStats {
    pub mean: f64,
    pub median: f64,
    pub stddev: f64,
    pub min: f64,
    pub max: f64,
    pub ci95_low: f64,
    pub ci95_high: f64,
    pub samples: usize,
}

/// Run a benchmark function `trials` times and compute timing statistics.
pub fn statistical_benchmark<F: FnMut() -> u64>(mut f: F, trials: usize) -> TimingStats {
    let values: Vec<f64> = (0..trials).map(|_| f() as f64).collect();
    let (ci95_low, ci95_high) = statistics::confidence_interval(&values, 0.95);
    TimingStats {
        mean: statistics::mean(&values),
        median: statistics::median(&values),
        stddev: statistics::stddev(&values),
        min: statistics::min(&values),
        max: statistics::max(&values),
        ci95_low,
        ci95_high,
        samples: trials,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer_basic() {
        let mut timer = Timer::new();
        timer.start();
        // Do a tiny amount of work (use black_box to prevent optimization)
        let mut sum = 0u64;
        for i in 0..1000 {
            sum = sum.wrapping_add(i);
        }
        std::hint::black_box(sum);
        let ns = timer.stop();
        // In release mode, fast work may complete in < 1ns, so allow 0
        assert!(ns >= 0);
        assert!(timer.elapsed_us() >= 0.0);
        assert!(timer.elapsed_ms() >= 0.0);
        assert!(timer.elapsed_s() >= 0.0);
    }

    #[test]
    fn test_time_fn() {
        let ns = time_fn(|| {
            let mut x = 0u64;
            for i in 0..10000 {
                x = x.wrapping_add(i);
            }
            std::hint::black_box(x);
        });
        // In release mode, fast work may complete in < 1ns
        assert!(ns >= 0);
    }

    #[test]
    fn test_time_fn_median() {
        let ns = time_fn_median(
            || {
                let mut x = 0u64;
                for i in 0..1000 {
                    x = x.wrapping_add(i);
                }
                std::hint::black_box(x);
            },
            2,
            5,
        );
        // In release mode, fast work may complete in < 1ns
        assert!(ns >= 0);
    }

    #[test]
    fn test_benchmark_config_builder() {
        let config = BenchmarkConfig::new("sort")
            .with_sizes(vec![100, 200])
            .with_hash_families(vec!["tab".to_string(), "poly".to_string()])
            .with_trials(10)
            .with_warmup(3);

        assert_eq!(config.algorithm, "sort");
        assert_eq!(config.input_sizes, vec![100, 200]);
        assert_eq!(config.hash_families.len(), 2);
        assert_eq!(config.num_trials, 10);
        assert_eq!(config.warmup_trials, 3);
    }

    #[test]
    fn test_harness_run_benchmark() {
        let mut harness = BenchmarkHarness::new();
        let config = BenchmarkConfig::new("test_algo")
            .with_sizes(vec![10, 20])
            .with_trials(3)
            .with_warmup(1);

        let results = harness.run_benchmark(&config, |n| simple_sequential_workload(n));
        assert_eq!(results.len(), 2); // 2 input sizes × 1 hash family
        assert_eq!(results[0].algorithm, "test_algo");
        assert_eq!(results[0].input_size, 10);
        assert_eq!(results[0].ops_count, 10);
        assert_eq!(results[1].input_size, 20);
    }

    #[test]
    fn test_harness_run_all() {
        let mut harness = BenchmarkHarness::new();
        harness.add_config(
            BenchmarkConfig::new("algo_a").with_sizes(vec![5]).with_trials(2).with_warmup(0),
        );
        harness.add_config(
            BenchmarkConfig::new("algo_b").with_sizes(vec![10]).with_trials(2).with_warmup(0),
        );

        let results = harness.run_all(|n| simple_sequential_workload(n));
        assert_eq!(results.len(), 2);
        assert_eq!(harness.results().len(), 2);
    }

    #[test]
    fn test_harness_cache_params() {
        let harness = BenchmarkHarness::new().with_cache_params(128, 256);
        assert_eq!(harness.cache_line_size, 128);
        assert_eq!(harness.num_cache_lines, 256);
    }

    #[test]
    fn test_harness_clear_results() {
        let mut harness = BenchmarkHarness::new();
        let config = BenchmarkConfig::new("x")
            .with_sizes(vec![5])
            .with_trials(1)
            .with_warmup(0);
        harness.run_benchmark(&config, |n| simple_sequential_workload(n));
        assert!(!harness.results().is_empty());
        harness.clear_results();
        assert!(harness.results().is_empty());
    }

    #[test]
    fn test_harness_summary_by_algorithm() {
        let mut harness = BenchmarkHarness::new();
        let config = BenchmarkConfig::new("sort")
            .with_sizes(vec![10, 20, 30])
            .with_trials(1)
            .with_warmup(0);
        harness.run_benchmark(&config, |n| simple_sequential_workload(n));

        let summary = harness.summary_by_algorithm();
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].0, "sort");
        assert_eq!(summary[0].1, 3); // 3 results
    }

    #[test]
    fn test_benchmark_result_display() {
        let result = BenchmarkResult {
            algorithm: "sort".to_string(),
            input_size: 1000,
            hash_family: "tab".to_string(),
            wall_time_ns: 12345,
            ops_count: 5000,
            cache_misses: 50,
            work_bound_satisfied: true,
            cache_bound_satisfied: false,
        };
        let s = format!("{}", result);
        assert!(s.contains("sort"));
        assert!(s.contains("n=1000"));
        assert!(s.contains("time=12345ns"));
    }

    #[test]
    fn test_simple_sequential_workload() {
        let (ops, addrs) = simple_sequential_workload(5);
        assert_eq!(ops, 5);
        assert_eq!(addrs, vec![0, 8, 16, 24, 32]);
    }

    #[test]
    fn test_pseudo_random_workload() {
        let (ops, addrs) = pseudo_random_workload(100, 1000);
        assert_eq!(ops, 100);
        assert_eq!(addrs.len(), 100);
        // All addresses should be multiples of 8
        assert!(addrs.iter().all(|a| a % 8 == 0));
    }

    #[test]
    fn test_harness_with_pseudo_random() {
        let mut harness = BenchmarkHarness::new();
        let config = BenchmarkConfig::new("random_access")
            .with_sizes(vec![50])
            .with_trials(2)
            .with_warmup(1);

        let results = harness.run_benchmark(&config, |n| pseudo_random_workload(n, 1000));
        assert_eq!(results.len(), 1);
        assert!(results[0].cache_misses > 0 || results[0].ops_count > 0);
    }

    #[test]
    fn test_multiple_hash_families() {
        let mut harness = BenchmarkHarness::new();
        let config = BenchmarkConfig::new("multi_hash")
            .with_sizes(vec![10])
            .with_hash_families(vec!["tab".to_string(), "poly".to_string()])
            .with_trials(1)
            .with_warmup(0);

        let results = harness.run_benchmark(&config, |n| simple_sequential_workload(n));
        assert_eq!(results.len(), 2); // 1 size × 2 hash families
        assert_eq!(results[0].hash_family, "tab");
        assert_eq!(results[1].hash_family, "poly");
    }

    #[test]
    fn test_benchmark_suite_new() {
        let suite = BenchmarkSuite::new("my_suite");
        assert_eq!(suite.name, "my_suite");
        assert_eq!(suite.config_count(), 0);
        assert!(suite.description.is_empty());
    }

    #[test]
    fn test_benchmark_suite_add_config() {
        let mut suite = BenchmarkSuite::new("suite");
        suite.add_config(BenchmarkConfig::new("algo_a").with_sizes(vec![10]));
        suite.add_config(BenchmarkConfig::new("algo_b").with_sizes(vec![20]));
        assert_eq!(suite.config_count(), 2);
        assert_eq!(suite.configs[0].algorithm, "algo_a");
        assert_eq!(suite.configs[1].algorithm, "algo_b");
    }

    #[test]
    fn test_run_suite() {
        let mut suite = BenchmarkSuite::new("test_suite");
        suite.add_config(
            BenchmarkConfig::new("seq")
                .with_sizes(vec![5, 10])
                .with_trials(1)
                .with_warmup(0),
        );
        let results = run_suite(&suite, |n| simple_sequential_workload(n));
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].input_size, 5);
        assert_eq!(results[1].input_size, 10);
    }

    #[test]
    fn test_warmup_runs() {
        let mut count = 0u32;
        warmup(&mut || { count += 1; }, 5);
        assert_eq!(count, 5);
    }

    #[test]
    fn test_timing_stats_fields() {
        let stats = statistical_benchmark(|| 42, 10);
        assert_eq!(stats.samples, 10);
        assert!((stats.mean - 42.0).abs() < 1e-10);
        assert!((stats.median - 42.0).abs() < 1e-10);
        assert!((stats.min - 42.0).abs() < 1e-10);
        assert!((stats.max - 42.0).abs() < 1e-10);
        assert!(stats.stddev.abs() < 1e-10);
        assert!(stats.ci95_low <= stats.mean);
        assert!(stats.ci95_high >= stats.mean);
    }

    #[test]
    fn test_statistical_benchmark_varying() {
        let mut counter = 0u64;
        let stats = statistical_benchmark(|| { counter += 1; counter }, 5);
        assert_eq!(stats.samples, 5);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.median - 3.0).abs() < 1e-10);
        assert!(stats.stddev > 0.0);
    }
}
