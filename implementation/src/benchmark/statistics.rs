//! Statistical analysis utilities for benchmark data.

/// Compute the arithmetic mean of a slice of f64 values.
/// Returns 0.0 for an empty slice.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Compute the median of a slice of f64 values.
/// Returns 0.0 for an empty slice.
pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Compute the sample variance (using Bessel's correction) of a slice of f64 values.
/// Returns 0.0 if fewer than 2 elements.
pub fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let sum_sq: f64 = data.iter().map(|x| (x - m) * (x - m)).sum();
    sum_sq / (data.len() - 1) as f64
}

/// Compute the sample standard deviation of a slice of f64 values.
pub fn stddev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Compute the geometric mean of a slice of positive f64 values.
/// Returns 0.0 for an empty slice. Values must be positive.
pub fn geometric_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let log_sum: f64 = data.iter().map(|x| x.ln()).sum();
    (log_sum / data.len() as f64).exp()
}

/// Compute a specific percentile (0.0 to 100.0) of a data set using linear interpolation.
/// Returns 0.0 for an empty slice.
pub fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }

    let p = p.clamp(0.0, 100.0);
    let rank = (p / 100.0) * (n - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f64;

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// Compute a confidence interval for the mean using the t-distribution approximation.
///
/// `confidence_level` is a value like 0.95 for a 95% CI.
/// Returns (lower, upper). For small samples, uses a rough z-approximation.
pub fn confidence_interval(data: &[f64], confidence_level: f64) -> (f64, f64) {
    if data.len() < 2 {
        let m = mean(data);
        return (m, m);
    }

    let n = data.len() as f64;
    let m = mean(data);
    let s = stddev(data);
    let se = s / n.sqrt();

    // Approximate t-critical value using common z-values
    let z = if confidence_level >= 0.99 {
        2.576
    } else if confidence_level >= 0.95 {
        1.960
    } else if confidence_level >= 0.90 {
        1.645
    } else {
        1.282 // ~80%
    };

    // Apply finite-sample correction for small n
    let t_approx = if n < 30.0 {
        z * (1.0 + 1.0 / (4.0 * (n - 1.0)))
    } else {
        z
    };

    let margin = t_approx * se;
    (m - margin, m + margin)
}

/// Detect outliers using the IQR (Interquartile Range) method.
///
/// Returns a vector of (index, value) pairs for values that fall outside
/// [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
pub fn detect_outliers(data: &[f64]) -> Vec<(usize, f64)> {
    if data.len() < 4 {
        return Vec::new();
    }

    let q1 = percentile(data, 25.0);
    let q3 = percentile(data, 75.0);
    let iqr = q3 - q1;
    let lower_fence = q1 - 1.5 * iqr;
    let upper_fence = q3 + 1.5 * iqr;

    data.iter()
        .enumerate()
        .filter(|(_, &v)| v < lower_fence || v > upper_fence)
        .map(|(i, &v)| (i, v))
        .collect()
}

/// Remove outliers from data (returns a new vector without outlier values).
pub fn remove_outliers(data: &[f64]) -> Vec<f64> {
    if data.len() < 4 {
        return data.to_vec();
    }

    let q1 = percentile(data, 25.0);
    let q3 = percentile(data, 75.0);
    let iqr = q3 - q1;
    let lower_fence = q1 - 1.5 * iqr;
    let upper_fence = q3 + 1.5 * iqr;

    data.iter()
        .copied()
        .filter(|&v| v >= lower_fence && v <= upper_fence)
        .collect()
}

/// Compute the minimum of a slice.
pub fn min(data: &[f64]) -> f64 {
    data.iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
}

/// Compute the maximum of a slice.
pub fn max(data: &[f64]) -> f64 {
    data.iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Summary statistics for a data set.
#[derive(Debug, Clone)]
pub struct SummaryStats {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
    pub variance: f64,
    pub p25: f64,
    pub p75: f64,
    pub p95: f64,
    pub p99: f64,
}

impl SummaryStats {
    /// Compute summary statistics from a data slice.
    pub fn from_data(data: &[f64]) -> Self {
        Self {
            count: data.len(),
            mean: mean(data),
            median: median(data),
            min: min(data),
            max: max(data),
            stddev: stddev(data),
            variance: variance(data),
            p25: percentile(data, 25.0),
            p75: percentile(data, 75.0),
            p95: percentile(data, 95.0),
            p99: percentile(data, 99.0),
        }
    }
}

impl std::fmt::Display for SummaryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n={}, mean={:.4}, median={:.4}, min={:.4}, max={:.4}, stddev={:.4}",
            self.count, self.mean, self.median, self.min, self.max, self.stddev
        )
    }
}

/// Result of a Welch's t-test.
#[derive(Debug, Clone)]
pub struct TTestResult {
    pub t_statistic: f64,
    pub degrees_of_freedom: f64,
    pub p_value_approx: f64,
    pub significant: bool,
}

/// Welch's t-test for two samples with possibly unequal variances.
///
/// Returns a `TTestResult` with the t-statistic, approximate degrees of freedom
/// (Welch–Satterthwaite), a rough p-value approximation, and whether the result
/// is significant at the 0.05 level.
pub fn welch_t_test(a: &[f64], b: &[f64]) -> TTestResult {
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;
    let mean_a = mean(a);
    let mean_b = mean(b);
    let var_a = variance(a);
    let var_b = variance(b);

    let se = (var_a / n_a + var_b / n_b).sqrt();
    let t = if se == 0.0 { 0.0 } else { (mean_a - mean_b) / se };

    // Welch–Satterthwaite degrees of freedom
    let num = (var_a / n_a + var_b / n_b).powi(2);
    let denom = (var_a / n_a).powi(2) / (n_a - 1.0) + (var_b / n_b).powi(2) / (n_b - 1.0);
    let df = if denom == 0.0 { 1.0 } else { num / denom };

    // Rough p-value approximation using normal distribution for large samples
    let abs_t = t.abs();
    let p_approx = if abs_t > 3.291 {
        0.001
    } else if abs_t > 2.576 {
        0.01
    } else if abs_t > 1.960 {
        0.05
    } else if abs_t > 1.645 {
        0.10
    } else {
        0.50
    };

    TTestResult {
        t_statistic: t,
        degrees_of_freedom: df,
        p_value_approx: p_approx,
        significant: abs_t > 1.96,
    }
}

/// Mann-Whitney U test. Returns the U statistic for sample `a`.
///
/// Combines both samples, ranks all values, then computes U for the first sample.
pub fn mann_whitney_u(a: &[f64], b: &[f64]) -> f64 {
    let n_a = a.len();
    let n_b = b.len();

    // Build combined list with group labels
    let mut combined: Vec<(f64, usize)> = Vec::with_capacity(n_a + n_b);
    for &v in a {
        combined.push((v, 0)); // group 0 = a
    }
    for &v in b {
        combined.push((v, 1)); // group 1 = b
    }
    combined.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign average ranks for ties
    let n = combined.len();
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (combined[j].0 - combined[i].0).abs() < 1e-12 {
            j += 1;
        }
        let avg_rank = (i + 1..=j).map(|r| r as f64).sum::<f64>() / (j - i) as f64;
        for k in i..j {
            ranks[k] = avg_rank;
        }
        i = j;
    }

    // Sum ranks for group a
    let rank_sum_a: f64 = combined
        .iter()
        .enumerate()
        .filter(|(_, (_, group))| *group == 0)
        .map(|(idx, _)| ranks[idx])
        .sum();

    // U = R_a - n_a*(n_a+1)/2
    rank_sum_a - (n_a as f64) * (n_a as f64 + 1.0) / 2.0
}

/// Bootstrap confidence interval for the mean using pseudo-random resampling.
///
/// Uses a seeded xorshift PRNG for reproducibility while providing genuine
/// resampling variance (unlike cyclic shifts which collapse for small n).
pub fn bootstrap_ci(data: &[f64], confidence: f64, iterations: usize) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0);
    }

    let n = data.len();
    let mut bootstrap_means: Vec<f64> = Vec::with_capacity(iterations);
    let mut rng: u64 = 0xDEAD_BEEF_CAFE_1234;

    for _ in 0..iterations {
        let mut sum = 0.0;
        for _ in 0..n {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let idx = (rng as usize) % n;
            sum += data[idx];
        }
        bootstrap_means.push(sum / n as f64);
    }

    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence;
    let lower_idx = ((alpha / 2.0) * iterations as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * iterations as f64).ceil() as usize;
    let upper_idx = upper_idx.min(iterations - 1);

    (bootstrap_means[lower_idx], bootstrap_means[upper_idx])
}

/// Cohen's d effect size between two samples.
///
/// d = (mean_a - mean_b) / pooled_stddev, where pooled_stddev uses the pooled
/// variance formula: sqrt(((n_a-1)*var_a + (n_b-1)*var_b) / (n_a + n_b - 2)).
pub fn effect_size(a: &[f64], b: &[f64]) -> f64 {
    let mean_a = mean(a);
    let mean_b = mean(b);
    let var_a = variance(a);
    let var_b = variance(b);
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;

    let pooled_var = ((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / (n_a + n_b - 2.0);
    let pooled_sd = pooled_var.sqrt();

    if pooled_sd == 0.0 {
        0.0
    } else {
        (mean_a - mean_b) / pooled_sd
    }
}

/// A point on the roofline model for performance analysis.
#[derive(Debug, Clone)]
pub struct RooflinePoint {
    pub algorithm: String,
    pub operational_intensity: f64,  // flops/byte
    pub achieved_performance: f64,   // flops/sec (normalized)
    pub peak_bandwidth_bound: f64,
    pub peak_compute_bound: f64,
    pub bottleneck: RooflineBottleneck,
}

/// Whether an algorithm is memory-bound or compute-bound.
#[derive(Debug, Clone, PartialEq)]
pub enum RooflineBottleneck {
    MemoryBound,
    ComputeBound,
}

/// Perform roofline analysis for a set of algorithms.
/// Returns roofline points showing whether each algorithm is memory- or compute-bound.
pub fn roofline_analysis(
    algorithms: &[(String, f64, f64)], // (name, ops_count, bytes_transferred)
    peak_bandwidth: f64,               // bytes/sec
    peak_compute: f64,                 // ops/sec
) -> Vec<RooflinePoint> {
    algorithms
        .iter()
        .map(|(name, ops, bytes)| {
            let oi = if *bytes > 0.0 { ops / bytes } else { f64::MAX };
            let bandwidth_ceiling = peak_bandwidth * oi;
            let compute_ceiling = peak_compute;
            let achievable = bandwidth_ceiling.min(compute_ceiling);
            let bottleneck = if bandwidth_ceiling < compute_ceiling {
                RooflineBottleneck::MemoryBound
            } else {
                RooflineBottleneck::ComputeBound
            };
            RooflinePoint {
                algorithm: name.clone(),
                operational_intensity: oi,
                achieved_performance: achievable,
                peak_bandwidth_bound: bandwidth_ceiling,
                peak_compute_bound: compute_ceiling,
                bottleneck,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_basic() {
        assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_empty() {
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn test_mean_single() {
        assert!((mean(&[42.0]) - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_odd() {
        assert!((median(&[3.0, 1.0, 2.0]) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_even() {
        assert!((median(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_median_empty() {
        assert_eq!(median(&[]), 0.0);
    }

    #[test]
    fn test_variance_known() {
        // Variance of [2, 4, 4, 4, 5, 5, 7, 9] = 4.571...
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = variance(&data);
        assert!((v - 4.571428571).abs() < 0.001);
    }

    #[test]
    fn test_stddev_known() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = stddev(&data);
        assert!((s - 2.13809).abs() < 0.001);
    }

    #[test]
    fn test_variance_single() {
        assert_eq!(variance(&[5.0]), 0.0);
    }

    #[test]
    fn test_geometric_mean() {
        // geometric_mean([2, 8]) = sqrt(16) = 4
        let gm = geometric_mean(&[2.0, 8.0]);
        assert!((gm - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_mean_equal() {
        let gm = geometric_mean(&[5.0, 5.0, 5.0]);
        assert!((gm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p50 = percentile(&data, 50.0);
        assert!((p50 - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_extremes() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        assert!((percentile(&data, 0.0) - 10.0).abs() < 1e-10);
        assert!((percentile(&data, 100.0) - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_25_75() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let q1 = percentile(&data, 25.0);
        let q3 = percentile(&data, 75.0);
        assert!(q1 < q3);
        assert!(q1 > 1.0);
        assert!(q3 < 12.0);
    }

    #[test]
    fn test_confidence_interval_95() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let (lower, upper) = confidence_interval(&data, 0.95);
        let m = mean(&data);
        assert!(lower < m);
        assert!(upper > m);
        assert!(lower > 0.0);
        assert!(upper < 101.0);
    }

    #[test]
    fn test_confidence_interval_single() {
        let (lower, upper) = confidence_interval(&[5.0], 0.95);
        assert!((lower - 5.0).abs() < 1e-10);
        assert!((upper - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_detect_outliers_none() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let outliers = detect_outliers(&data);
        assert!(outliers.is_empty());
    }

    #[test]
    fn test_detect_outliers_present() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 100.0];
        let outliers = detect_outliers(&data);
        assert!(!outliers.is_empty());
        assert!(outliers.iter().any(|(_, v)| (*v - 100.0).abs() < 1e-10));
    }

    #[test]
    fn test_remove_outliers() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 100.0];
        let cleaned = remove_outliers(&data);
        assert!(cleaned.len() < data.len());
        assert!(!cleaned.contains(&100.0));
    }

    #[test]
    fn test_min_max() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        assert!((min(&data) - 1.0).abs() < 1e-10);
        assert!((max(&data) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_summary_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = SummaryStats::from_data(&data);
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.median - 3.0).abs() < 1e-10);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_summary_stats_display() {
        let data = vec![1.0, 2.0, 3.0];
        let stats = SummaryStats::from_data(&data);
        let s = format!("{}", stats);
        assert!(s.contains("n=3"));
        assert!(s.contains("mean="));
    }

    #[test]
    fn test_welch_t_test_significant() {
        let a = vec![10.0, 12.0, 11.0, 13.0, 14.0, 10.0, 11.0, 12.0];
        let b = vec![20.0, 22.0, 21.0, 23.0, 24.0, 20.0, 21.0, 22.0];
        let result = welch_t_test(&a, &b);
        assert!(result.t_statistic < 0.0); // a has lower mean
        assert!(result.significant);
        assert!(result.degrees_of_freedom > 0.0);
    }

    #[test]
    fn test_welch_t_test_not_significant() {
        let a = vec![5.0, 5.1, 4.9, 5.0, 5.05];
        let b = vec![5.0, 4.95, 5.1, 5.0, 4.98];
        let result = welch_t_test(&a, &b);
        assert!(!result.significant);
    }

    #[test]
    fn test_mann_whitney_u_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let u = mann_whitney_u(&a, &b);
        // a values all rank below b values, so U = 0
        assert!((u - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_mann_whitney_u_reversed() {
        let a = vec![4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0];
        let u = mann_whitney_u(&a, &b);
        // a values all rank above b values, so U = n_a * n_b = 9
        assert!((u - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_bootstrap_ci_constant() {
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let (lower, upper) = bootstrap_ci(&data, 0.95, 100);
        assert!((lower - 5.0).abs() < 1e-10);
        assert!((upper - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_bootstrap_ci_range() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (lower, upper) = bootstrap_ci(&data, 0.95, 1000);
        let m = mean(&data);
        assert!(lower <= m);
        assert!(upper >= m);
    }

    #[test]
    fn test_effect_size_zero() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = effect_size(&a, &b);
        assert!((d - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_effect_size_large() {
        let a = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = effect_size(&a, &b);
        assert!(d > 0.0); // a has higher mean
        // With equal variances and 9-unit difference, d should be large
        assert!(d > 2.0);
    }

    #[test]
    fn test_roofline_analysis_memory_bound() {
        let algos = vec![
            ("bfs".to_string(), 1000.0, 10000.0), // low OI → memory-bound
        ];
        let points = roofline_analysis(&algos, 100.0, 1000.0);
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].bottleneck, RooflineBottleneck::MemoryBound);
    }

    #[test]
    fn test_roofline_analysis_compute_bound() {
        let algos = vec![
            ("fft".to_string(), 100000.0, 100.0), // high OI → compute-bound
        ];
        let points = roofline_analysis(&algos, 100.0, 1000.0);
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].bottleneck, RooflineBottleneck::ComputeBound);
    }

    #[test]
    fn test_roofline_analysis_multiple() {
        let algos = vec![
            ("sort".to_string(), 500.0, 2000.0),
            ("scan".to_string(), 100.0, 50.0),
        ];
        let points = roofline_analysis(&algos, 50.0, 500.0);
        assert_eq!(points.len(), 2);
        assert!(points[0].operational_intensity > 0.0);
        assert!(points[1].operational_intensity > 0.0);
    }
}
