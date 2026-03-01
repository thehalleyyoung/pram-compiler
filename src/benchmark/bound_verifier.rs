//! Runtime bound verification for PRAM work and cache-miss bounds.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Result of verifying a single bound.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundResult {
    /// Whether the actual value satisfied the bound.
    pub satisfied: bool,
    /// The measured value.
    pub actual: u64,
    /// The computed upper bound.
    pub bound: u64,
    /// Ratio actual / bound (< 1.0 means satisfied).
    pub ratio: f64,
}

impl fmt::Display for BoundResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: actual={}, bound={}, ratio={:.4}",
            if self.satisfied { "PASS" } else { "FAIL" },
            self.actual,
            self.bound,
            self.ratio,
        )
    }
}

/// A single verification entry for batch verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationEntry {
    pub algorithm: String,
    pub input_size: u64,
    pub work_result: Option<BoundResult>,
    pub cache_result: Option<BoundResult>,
}

/// Summary report from batch verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    pub entries: Vec<VerificationEntry>,
    pub total_checks: usize,
    pub passed: usize,
    pub failed: usize,
}

impl VerificationReport {
    pub fn pass_rate(&self) -> f64 {
        if self.total_checks == 0 {
            1.0
        } else {
            self.passed as f64 / self.total_checks as f64
        }
    }

    pub fn all_passed(&self) -> bool {
        self.failed == 0
    }
}

impl fmt::Display for VerificationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Verification Report ===")?;
        writeln!(
            f,
            "Total checks: {}, Passed: {}, Failed: {}, Pass rate: {:.2}%",
            self.total_checks,
            self.passed,
            self.failed,
            self.pass_rate() * 100.0,
        )?;
        writeln!(f, "---")?;
        for entry in &self.entries {
            write!(f, "Algorithm: {}, n={}", entry.algorithm, entry.input_size)?;
            if let Some(ref wr) = entry.work_result {
                write!(f, " | Work: {}", wr)?;
            }
            if let Some(ref cr) = entry.cache_result {
                write!(f, " | Cache: {}", cr)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Core verification functions
// ---------------------------------------------------------------------------

/// Verify a work bound.
///
/// The theoretical work bound is: `c1 * p * T`
/// where `p` = number of processors and `T` = parallel time (steps).
///
/// Returns a `BoundResult` indicating whether `actual_ops <= c1 * p * T`.
pub fn verify_work_bound(
    actual_ops: u64,
    p: u64,
    t: u64,
    c1: f64,
) -> BoundResult {
    let bound_f = c1 * (p as f64) * (t as f64);
    let bound = bound_f.ceil() as u64;
    let ratio = if bound == 0 {
        if actual_ops == 0 { 0.0 } else { f64::INFINITY }
    } else {
        actual_ops as f64 / bound as f64
    };

    BoundResult {
        satisfied: actual_ops <= bound,
        actual: actual_ops,
        bound,
        ratio,
    }
}

/// Verify a cache-miss bound.
///
/// The theoretical cache-miss bound is: `c3 * (p * T) / B`
/// where `B` = cache-line size in elements.
///
/// Returns a `BoundResult` indicating whether `actual_misses <= c3 * p * T / B`.
pub fn verify_cache_bound(
    actual_misses: u64,
    p: u64,
    t: u64,
    b: u64,
    c3: f64,
) -> BoundResult {
    let bound_f = if b == 0 {
        f64::INFINITY
    } else {
        c3 * (p as f64) * (t as f64) / (b as f64)
    };
    let bound = if bound_f.is_infinite() {
        u64::MAX
    } else {
        bound_f.ceil() as u64
    };

    let ratio = if bound == 0 {
        if actual_misses == 0 { 0.0 } else { f64::INFINITY }
    } else {
        actual_misses as f64 / bound as f64
    };

    BoundResult {
        satisfied: actual_misses <= bound,
        actual: actual_misses,
        bound,
        ratio,
    }
}

// ---------------------------------------------------------------------------
// Batch verifier
// ---------------------------------------------------------------------------

/// Configuration for a single bound-verification task.
#[derive(Debug, Clone)]
pub struct BoundCheckConfig {
    pub algorithm: String,
    pub input_size: u64,
    pub num_processors: u64,
    pub parallel_time: u64,
    pub cache_line_elements: u64,
    pub actual_ops: Option<u64>,
    pub actual_misses: Option<u64>,
    pub work_constant: f64,
    pub cache_constant: f64,
}

/// Verifier that can batch-check multiple (algorithm, input_size) pairs.
pub struct BoundVerifier {
    configs: Vec<BoundCheckConfig>,
}

impl BoundVerifier {
    pub fn new() -> Self {
        Self { configs: Vec::new() }
    }

    /// Add a verification configuration.
    pub fn add(&mut self, config: BoundCheckConfig) {
        self.configs.push(config);
    }

    /// Add multiple configurations at once.
    pub fn add_batch(&mut self, configs: impl IntoIterator<Item = BoundCheckConfig>) {
        for c in configs {
            self.configs.push(c);
        }
    }

    /// Run all verifications and produce a report.
    pub fn verify_all(&self) -> VerificationReport {
        let mut entries = Vec::new();
        let mut total_checks = 0usize;
        let mut passed = 0usize;
        let mut failed = 0usize;

        for config in &self.configs {
            let work_result = config.actual_ops.map(|ops| {
                verify_work_bound(ops, config.num_processors, config.parallel_time, config.work_constant)
            });
            let cache_result = config.actual_misses.map(|misses| {
                verify_cache_bound(
                    misses,
                    config.num_processors,
                    config.parallel_time,
                    config.cache_line_elements,
                    config.cache_constant,
                )
            });

            if let Some(ref wr) = work_result {
                total_checks += 1;
                if wr.satisfied {
                    passed += 1;
                } else {
                    failed += 1;
                }
            }
            if let Some(ref cr) = cache_result {
                total_checks += 1;
                if cr.satisfied {
                    passed += 1;
                } else {
                    failed += 1;
                }
            }

            entries.push(VerificationEntry {
                algorithm: config.algorithm.clone(),
                input_size: config.input_size,
                work_result,
                cache_result,
            });
        }

        VerificationReport {
            entries,
            total_checks,
            passed,
            failed,
        }
    }

    /// Clear all configurations.
    pub fn clear(&mut self) {
        self.configs.clear();
    }

    /// Number of pending configurations.
    pub fn len(&self) -> usize {
        self.configs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.configs.is_empty()
    }
}

impl Default for BoundVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Convenience helpers
// ---------------------------------------------------------------------------

/// Quick check: does the work satisfy O(p * T)?
pub fn work_satisfies_bound(actual_ops: u64, p: u64, t: u64) -> bool {
    verify_work_bound(actual_ops, p, t, 1.0).satisfied
}

/// Quick check: does the cache usage satisfy O(p * T / B)?
pub fn cache_satisfies_bound(actual_misses: u64, p: u64, t: u64, b: u64) -> bool {
    verify_cache_bound(actual_misses, p, t, b, 1.0).satisfied
}

/// Compute the minimal constant c such that actual <= c * p * T.
pub fn compute_work_constant(actual_ops: u64, p: u64, t: u64) -> f64 {
    let denom = (p as f64) * (t as f64);
    if denom == 0.0 {
        f64::INFINITY
    } else {
        actual_ops as f64 / denom
    }
}

/// Compute the minimal constant c such that actual <= c * p * T / B.
pub fn compute_cache_constant(actual_misses: u64, p: u64, t: u64, b: u64) -> f64 {
    let denom = (p as f64) * (t as f64) / (b as f64);
    if denom == 0.0 {
        f64::INFINITY
    } else {
        actual_misses as f64 / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_bound_satisfied() {
        let result = verify_work_bound(100, 4, 50, 1.0);
        // bound = 1.0 * 4 * 50 = 200
        assert!(result.satisfied);
        assert_eq!(result.actual, 100);
        assert_eq!(result.bound, 200);
        assert!((result.ratio - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_work_bound_violated() {
        let result = verify_work_bound(500, 4, 50, 1.0);
        // bound = 200, actual = 500
        assert!(!result.satisfied);
        assert!((result.ratio - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_work_bound_exact() {
        let result = verify_work_bound(200, 4, 50, 1.0);
        assert!(result.satisfied);
        assert!((result.ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_work_bound_with_constant() {
        let result = verify_work_bound(300, 4, 50, 2.0);
        // bound = 2.0 * 4 * 50 = 400
        assert!(result.satisfied);
        assert_eq!(result.bound, 400);
    }

    #[test]
    fn test_cache_bound_satisfied() {
        let result = verify_cache_bound(10, 4, 50, 8, 1.0);
        // bound = ceil(1.0 * 4 * 50 / 8) = 25
        assert!(result.satisfied);
        assert_eq!(result.bound, 25);
    }

    #[test]
    fn test_cache_bound_violated() {
        let result = verify_cache_bound(100, 4, 50, 8, 1.0);
        // bound = 25, actual = 100
        assert!(!result.satisfied);
    }

    #[test]
    fn test_cache_bound_with_constant() {
        let result = verify_cache_bound(40, 4, 50, 8, 2.0);
        // bound = ceil(2.0 * 200 / 8) = 50
        assert!(result.satisfied);
        assert_eq!(result.bound, 50);
    }

    #[test]
    fn test_work_satisfies_bound_helper() {
        assert!(work_satisfies_bound(100, 10, 20));
        assert!(!work_satisfies_bound(300, 10, 20));
    }

    #[test]
    fn test_cache_satisfies_bound_helper() {
        assert!(cache_satisfies_bound(10, 10, 20, 8));
        assert!(!cache_satisfies_bound(100, 10, 20, 8));
    }

    #[test]
    fn test_compute_work_constant() {
        let c = compute_work_constant(300, 10, 10);
        assert!((c - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_cache_constant() {
        let c = compute_cache_constant(50, 10, 10, 4);
        // denom = 10 * 10 / 4 = 25
        assert!((c - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_bound_result_display() {
        let result = BoundResult {
            satisfied: true,
            actual: 100,
            bound: 200,
            ratio: 0.5,
        };
        let s = format!("{}", result);
        assert!(s.contains("PASS"));
        assert!(s.contains("actual=100"));
    }

    #[test]
    fn test_batch_verifier_empty() {
        let verifier = BoundVerifier::new();
        let report = verifier.verify_all();
        assert_eq!(report.total_checks, 0);
        assert!(report.all_passed());
        assert!((report.pass_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_verifier_single() {
        let mut verifier = BoundVerifier::new();
        verifier.add(BoundCheckConfig {
            algorithm: "sort".to_string(),
            input_size: 1000,
            num_processors: 8,
            parallel_time: 125,
            cache_line_elements: 8,
            actual_ops: Some(900),
            actual_misses: Some(100),
            work_constant: 1.0,
            cache_constant: 1.0,
        });

        let report = verifier.verify_all();
        assert_eq!(report.total_checks, 2);
        assert_eq!(report.entries.len(), 1);
    }

    #[test]
    fn test_batch_verifier_mixed() {
        let mut verifier = BoundVerifier::new();
        verifier.add(BoundCheckConfig {
            algorithm: "sort".to_string(),
            input_size: 1000,
            num_processors: 8,
            parallel_time: 125,
            cache_line_elements: 8,
            actual_ops: Some(500),
            actual_misses: None,
            work_constant: 1.0,
            cache_constant: 1.0,
        });
        verifier.add(BoundCheckConfig {
            algorithm: "prefix_sum".to_string(),
            input_size: 500,
            num_processors: 4,
            parallel_time: 100,
            cache_line_elements: 8,
            actual_ops: Some(10000),
            actual_misses: Some(5),
            work_constant: 1.0,
            cache_constant: 1.0,
        });

        let report = verifier.verify_all();
        assert_eq!(report.entries.len(), 2);
        // sort: work bound = 1000, actual = 500 → pass
        // prefix: work bound = 400, actual = 10000 → fail
        // prefix: cache bound = ceil(400/8) = 50, actual = 5 → pass
        assert_eq!(report.total_checks, 3);
    }

    #[test]
    fn test_batch_verifier_add_batch() {
        let mut verifier = BoundVerifier::new();
        let configs = vec![
            BoundCheckConfig {
                algorithm: "a".to_string(),
                input_size: 100,
                num_processors: 1,
                parallel_time: 100,
                cache_line_elements: 8,
                actual_ops: Some(50),
                actual_misses: None,
                work_constant: 1.0,
                cache_constant: 1.0,
            },
            BoundCheckConfig {
                algorithm: "b".to_string(),
                input_size: 200,
                num_processors: 2,
                parallel_time: 100,
                cache_line_elements: 8,
                actual_ops: Some(150),
                actual_misses: None,
                work_constant: 1.0,
                cache_constant: 1.0,
            },
        ];
        verifier.add_batch(configs);
        assert_eq!(verifier.len(), 2);
    }

    #[test]
    fn test_batch_verifier_clear() {
        let mut verifier = BoundVerifier::new();
        verifier.add(BoundCheckConfig {
            algorithm: "x".to_string(),
            input_size: 10,
            num_processors: 1,
            parallel_time: 10,
            cache_line_elements: 8,
            actual_ops: Some(5),
            actual_misses: None,
            work_constant: 1.0,
            cache_constant: 1.0,
        });
        assert!(!verifier.is_empty());
        verifier.clear();
        assert!(verifier.is_empty());
    }

    #[test]
    fn test_verification_report_display() {
        let report = VerificationReport {
            entries: vec![VerificationEntry {
                algorithm: "sort".to_string(),
                input_size: 1000,
                work_result: Some(BoundResult {
                    satisfied: true,
                    actual: 100,
                    bound: 200,
                    ratio: 0.5,
                }),
                cache_result: None,
            }],
            total_checks: 1,
            passed: 1,
            failed: 0,
        };
        let s = format!("{}", report);
        assert!(s.contains("sort"));
        assert!(s.contains("Passed: 1"));
    }

    #[test]
    fn test_zero_inputs() {
        let result = verify_work_bound(0, 0, 0, 1.0);
        assert!(result.satisfied);
        assert_eq!(result.actual, 0);
    }

    #[test]
    fn test_cache_bound_zero_b() {
        let result = verify_cache_bound(10, 4, 50, 0, 1.0);
        // B=0 → bound = infinity → always satisfied
        assert!(result.satisfied);
    }

    #[test]
    fn test_compute_work_constant_zero() {
        let c = compute_work_constant(100, 0, 0);
        assert!(c.is_infinite());
    }
}
