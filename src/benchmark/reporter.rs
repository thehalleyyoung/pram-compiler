//! Results reporting: CSV, JSON, and human-readable table output.

use serde::{Deserialize, Serialize};
use std::fmt;

use super::harness::BenchmarkResult;
use super::statistics;

/// Configuration for the reporter.
#[derive(Debug, Clone)]
pub struct ReporterConfig {
    /// Decimal places for floating-point values in table output.
    pub float_precision: usize,
    /// Whether to include summary statistics at the bottom of tables.
    pub include_summary: bool,
}

impl Default for ReporterConfig {
    fn default() -> Self {
        Self {
            float_precision: 2,
            include_summary: true,
        }
    }
}

/// Reporter for formatting benchmark results.
pub struct Reporter {
    config: ReporterConfig,
}

impl Reporter {
    pub fn new() -> Self {
        Self {
            config: ReporterConfig::default(),
        }
    }

    pub fn with_config(config: ReporterConfig) -> Self {
        Self { config }
    }

    /// Generate CSV output from benchmark results.
    pub fn report_csv(&self, results: &[BenchmarkResult]) -> String {
        report_csv(results)
    }

    /// Generate JSON output from benchmark results.
    pub fn report_json(&self, results: &[BenchmarkResult]) -> String {
        report_json(results)
    }

    /// Generate a human-readable table from benchmark results.
    pub fn report_table(&self, results: &[BenchmarkResult]) -> String {
        report_table_with_config(results, &self.config)
    }
}

impl Default for Reporter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

/// Generate CSV output from benchmark results.
pub fn report_csv(results: &[BenchmarkResult]) -> String {
    let mut out = String::new();
    out.push_str("algorithm,input_size,hash_family,wall_time_ns,ops_count,cache_misses,work_ok,cache_ok\n");
    for r in results {
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            r.algorithm,
            r.input_size,
            r.hash_family,
            r.wall_time_ns,
            r.ops_count,
            r.cache_misses,
            r.work_bound_satisfied,
            r.cache_bound_satisfied,
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

/// Generate JSON output from benchmark results.
pub fn report_json(results: &[BenchmarkResult]) -> String {
    serde_json::to_string_pretty(results).unwrap_or_else(|_| "[]".to_string())
}

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------

/// Generate a human-readable table from benchmark results.
pub fn report_table(results: &[BenchmarkResult]) -> String {
    report_table_with_config(results, &ReporterConfig::default())
}

fn report_table_with_config(results: &[BenchmarkResult], config: &ReporterConfig) -> String {
    if results.is_empty() {
        return "(no results)\n".to_string();
    }

    // Column widths
    let algo_w = results.iter().map(|r| r.algorithm.len()).max().unwrap_or(9).max(9);
    let hash_w = results.iter().map(|r| r.hash_family.len()).max().unwrap_or(6).max(6);

    let mut out = String::new();

    // Header
    out.push_str(&format!(
        "{:<algo_w$}  {:>10}  {:<hash_w$}  {:>14}  {:>10}  {:>12}  {:>7}  {:>8}\n",
        "Algorithm", "InputSize", "Hash", "WallTime(ns)", "Ops", "CacheMisses", "WorkOK", "CacheOK",
        algo_w = algo_w,
        hash_w = hash_w,
    ));

    // Separator
    let total_w = algo_w + 10 + hash_w + 14 + 10 + 12 + 7 + 8 + 14; // approx
    out.push_str(&"-".repeat(total_w));
    out.push('\n');

    // Rows
    for r in results {
        out.push_str(&format!(
            "{:<algo_w$}  {:>10}  {:<hash_w$}  {:>14}  {:>10}  {:>12}  {:>7}  {:>8}\n",
            r.algorithm,
            r.input_size,
            r.hash_family,
            r.wall_time_ns,
            r.ops_count,
            r.cache_misses,
            if r.work_bound_satisfied { "YES" } else { "NO" },
            if r.cache_bound_satisfied { "YES" } else { "NO" },
            algo_w = algo_w,
            hash_w = hash_w,
        ));
    }

    // Summary
    if config.include_summary && !results.is_empty() {
        out.push_str(&"-".repeat(total_w));
        out.push('\n');

        let times: Vec<f64> = results.iter().map(|r| r.wall_time_ns as f64).collect();
        let ops: Vec<f64> = results.iter().map(|r| r.ops_count as f64).collect();
        let misses: Vec<f64> = results.iter().map(|r| r.cache_misses as f64).collect();

        let prec = config.float_precision;

        out.push_str(&format!(
            "Time(ns)  — mean: {:.*}, median: {:.*}, min: {:.*}, max: {:.*}, stddev: {:.*}\n",
            prec, statistics::mean(&times),
            prec, statistics::median(&times),
            prec, statistics::min(&times),
            prec, statistics::max(&times),
            prec, statistics::stddev(&times),
        ));
        out.push_str(&format!(
            "Ops       — mean: {:.*}, median: {:.*}, min: {:.*}, max: {:.*}, stddev: {:.*}\n",
            prec, statistics::mean(&ops),
            prec, statistics::median(&ops),
            prec, statistics::min(&ops),
            prec, statistics::max(&ops),
            prec, statistics::stddev(&ops),
        ));
        out.push_str(&format!(
            "Misses    — mean: {:.*}, median: {:.*}, min: {:.*}, max: {:.*}, stddev: {:.*}\n",
            prec, statistics::mean(&misses),
            prec, statistics::median(&misses),
            prec, statistics::min(&misses),
            prec, statistics::max(&misses),
            prec, statistics::stddev(&misses),
        ));

        let work_pass = results.iter().filter(|r| r.work_bound_satisfied).count();
        let cache_pass = results.iter().filter(|r| r.cache_bound_satisfied).count();
        out.push_str(&format!(
            "Bounds    — work passed: {}/{}, cache passed: {}/{}\n",
            work_pass,
            results.len(),
            cache_pass,
            results.len(),
        ));
    }

    out
}

// ---------------------------------------------------------------------------
// Summary statistics helper
// ---------------------------------------------------------------------------

/// Per-algorithm summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmSummary {
    pub algorithm: String,
    pub num_runs: usize,
    pub mean_time_ns: f64,
    pub median_time_ns: f64,
    pub min_time_ns: f64,
    pub max_time_ns: f64,
    pub stddev_time_ns: f64,
    pub mean_ops: f64,
    pub mean_cache_misses: f64,
}

impl fmt::Display for AlgorithmSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: runs={}, time(ns) mean={:.1} median={:.1} min={:.1} max={:.1}, ops_mean={:.1}, misses_mean={:.1}",
            self.algorithm, self.num_runs, self.mean_time_ns, self.median_time_ns,
            self.min_time_ns, self.max_time_ns, self.mean_ops, self.mean_cache_misses,
        )
    }
}

/// Compute per-algorithm summaries from benchmark results.
pub fn summarize_by_algorithm(results: &[BenchmarkResult]) -> Vec<AlgorithmSummary> {
    let mut algos: Vec<String> = results.iter().map(|r| r.algorithm.clone()).collect();
    algos.sort();
    algos.dedup();

    algos
        .into_iter()
        .map(|algo| {
            let runs: Vec<&BenchmarkResult> =
                results.iter().filter(|r| r.algorithm == algo).collect();
            let times: Vec<f64> = runs.iter().map(|r| r.wall_time_ns as f64).collect();
            let ops: Vec<f64> = runs.iter().map(|r| r.ops_count as f64).collect();
            let misses: Vec<f64> = runs.iter().map(|r| r.cache_misses as f64).collect();

            AlgorithmSummary {
                algorithm: algo,
                num_runs: runs.len(),
                mean_time_ns: statistics::mean(&times),
                median_time_ns: statistics::median(&times),
                min_time_ns: statistics::min(&times),
                max_time_ns: statistics::max(&times),
                stddev_time_ns: statistics::stddev(&times),
                mean_ops: statistics::mean(&ops),
                mean_cache_misses: statistics::mean(&misses),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_results() -> Vec<BenchmarkResult> {
        vec![
            BenchmarkResult {
                algorithm: "sort".to_string(),
                input_size: 100,
                hash_family: "tab".to_string(),
                wall_time_ns: 5000,
                ops_count: 700,
                cache_misses: 20,
                work_bound_satisfied: true,
                cache_bound_satisfied: true,
            },
            BenchmarkResult {
                algorithm: "sort".to_string(),
                input_size: 1000,
                hash_family: "tab".to_string(),
                wall_time_ns: 50000,
                ops_count: 10000,
                cache_misses: 200,
                work_bound_satisfied: true,
                cache_bound_satisfied: false,
            },
            BenchmarkResult {
                algorithm: "prefix_sum".to_string(),
                input_size: 100,
                hash_family: "poly".to_string(),
                wall_time_ns: 1000,
                ops_count: 100,
                cache_misses: 5,
                work_bound_satisfied: true,
                cache_bound_satisfied: true,
            },
        ]
    }

    #[test]
    fn test_csv_header() {
        let csv = report_csv(&sample_results());
        let lines: Vec<&str> = csv.lines().collect();
        assert!(lines[0].contains("algorithm"));
        assert!(lines[0].contains("input_size"));
        assert!(lines[0].contains("wall_time_ns"));
    }

    #[test]
    fn test_csv_rows() {
        let csv = report_csv(&sample_results());
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 4); // header + 3 data rows
        assert!(lines[1].starts_with("sort"));
    }

    #[test]
    fn test_csv_empty() {
        let csv = report_csv(&[]);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 1); // header only
    }

    #[test]
    fn test_json_parses() {
        let json = report_json(&sample_results());
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_array());
        assert_eq!(parsed.as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_json_empty() {
        let json = report_json(&[]);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_table_contains_header() {
        let table = report_table(&sample_results());
        assert!(table.contains("Algorithm"));
        assert!(table.contains("InputSize"));
        assert!(table.contains("WallTime"));
    }

    #[test]
    fn test_table_contains_data() {
        let table = report_table(&sample_results());
        assert!(table.contains("sort"));
        assert!(table.contains("prefix_sum"));
        assert!(table.contains("5000"));
    }

    #[test]
    fn test_table_contains_summary() {
        let table = report_table(&sample_results());
        assert!(table.contains("mean:"));
        assert!(table.contains("median:"));
    }

    #[test]
    fn test_table_empty() {
        let table = report_table(&[]);
        assert!(table.contains("no results"));
    }

    #[test]
    fn test_table_no_summary() {
        let config = ReporterConfig {
            float_precision: 1,
            include_summary: false,
        };
        let table = report_table_with_config(&sample_results(), &config);
        assert!(!table.contains("mean:"));
    }

    #[test]
    fn test_reporter_struct() {
        let reporter = Reporter::new();
        let results = sample_results();
        let csv = reporter.report_csv(&results);
        let json = reporter.report_json(&results);
        let table = reporter.report_table(&results);
        assert!(!csv.is_empty());
        assert!(!json.is_empty());
        assert!(!table.is_empty());
    }

    #[test]
    fn test_summarize_by_algorithm() {
        let summaries = summarize_by_algorithm(&sample_results());
        assert_eq!(summaries.len(), 2); // "sort" and "prefix_sum"
        let sort_summary = summaries.iter().find(|s| s.algorithm == "sort").unwrap();
        assert_eq!(sort_summary.num_runs, 2);
        assert!(sort_summary.mean_time_ns > 0.0);
    }

    #[test]
    fn test_algorithm_summary_display() {
        let summary = AlgorithmSummary {
            algorithm: "sort".to_string(),
            num_runs: 5,
            mean_time_ns: 1000.0,
            median_time_ns: 900.0,
            min_time_ns: 500.0,
            max_time_ns: 1500.0,
            stddev_time_ns: 200.0,
            mean_ops: 5000.0,
            mean_cache_misses: 50.0,
        };
        let s = format!("{}", summary);
        assert!(s.contains("sort"));
        assert!(s.contains("runs=5"));
    }

    #[test]
    fn test_csv_roundtrip_fields() {
        let results = sample_results();
        let csv = report_csv(&results);
        let lines: Vec<&str> = csv.lines().collect();
        let fields: Vec<&str> = lines[1].split(',').collect();
        assert_eq!(fields[0], "sort");
        assert_eq!(fields[1], "100");
        assert_eq!(fields[2], "tab");
    }

    #[test]
    fn test_reporter_with_config() {
        let config = ReporterConfig {
            float_precision: 4,
            include_summary: true,
        };
        let reporter = Reporter::with_config(config);
        let table = reporter.report_table(&sample_results());
        assert!(table.contains("mean:"));
    }
}
