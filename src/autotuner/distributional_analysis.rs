//! PGO distributional analysis.
//!
//! Addresses the critique that "PGO methodology lacks distributional analysis"
//! by adding statistical analysis of how PGO decisions depend on input distributions.
//!
//! Analyzes:
//! - Crossover threshold sensitivity to input distribution
//! - Hash family selection stability under distributional shift
//! - Confidence intervals for PGO recommendations
//! - Distribution characterization (uniformity, skewness, stride patterns)

use std::collections::HashMap;
use crate::autotuner::profile_guided::{ProfileData, ProfileGuidedOptimizer};
use crate::autotuner::cache_probe::CacheHierarchy;

/// Distribution characterization for PGO inputs.
#[derive(Debug, Clone)]
pub struct DistributionProfile {
    pub name: String,
    pub mean: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub entropy: f64,
    pub stride_coefficient: f64,
}

/// Result of distributional sensitivity analysis.
#[derive(Debug, Clone)]
pub struct DistributionalAnalysis {
    /// Number of distributions tested.
    pub distributions_tested: usize,
    /// Crossover thresholds per distribution.
    pub crossover_thresholds: Vec<(String, usize)>,
    /// Hash family recommendation stability.
    pub hash_family_stability: f64,
    /// Block size recommendation stability.
    pub block_size_stability: f64,
    /// Overall sensitivity score (0 = robust, 1 = highly sensitive).
    pub sensitivity_score: f64,
    /// Per-distribution profiles.
    pub profiles: Vec<DistributionProfile>,
    /// Confidence interval for crossover threshold.
    pub crossover_ci_lower: usize,
    pub crossover_ci_upper: usize,
}

/// Generate different input distributions for testing PGO robustness.
pub fn generate_test_distributions(n: usize, seed: u64) -> Vec<(String, Vec<u64>)> {
    let mut distributions = Vec::new();
    let mut rng = seed;

    // 1. Uniform random
    let uniform: Vec<u64> = (0..n).map(|_| {
        rng = xorshift(rng);
        rng % (n as u64 * 8)
    }).collect();
    distributions.push(("uniform_random".into(), uniform));

    // 2. Sequential (sorted)
    let sequential: Vec<u64> = (0..n).map(|i| i as u64 * 8).collect();
    distributions.push(("sequential".into(), sequential));

    // 3. Reverse sorted
    let reverse: Vec<u64> = (0..n).map(|i| (n - 1 - i) as u64 * 8).collect();
    distributions.push(("reverse_sorted".into(), reverse));

    // 4. Strided
    let stride = 7u64;
    let strided: Vec<u64> = (0..n).map(|i| (i as u64 * stride) % (n as u64 * 8)).collect();
    distributions.push(("strided_7".into(), strided));

    // 5. Clustered (hot spots)
    rng = seed ^ 0xDEAD;
    let clustered: Vec<u64> = (0..n).map(|i| {
        let cluster = (i % 4) as u64;
        rng = xorshift(rng);
        cluster * (n as u64 * 2) + (rng % 64)
    }).collect();
    distributions.push(("clustered".into(), clustered));

    // 6. Power-law (Zipfian)
    rng = seed ^ 0xBEEF;
    let zipfian: Vec<u64> = (0..n).map(|_| {
        rng = xorshift(rng);
        let u = (rng % 10000) as f64 / 10000.0;
        let alpha = 1.5;
        let x = ((1.0 - u).powf(-1.0 / (alpha - 1.0))) as u64;
        x.min(n as u64 * 8)
    }).collect();
    distributions.push(("zipfian".into(), zipfian));

    distributions
}

fn xorshift(mut x: u64) -> u64 {
    if x == 0 { x = 1; }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Characterize a distribution.
pub fn characterize_distribution(name: &str, data: &[u64]) -> DistributionProfile {
    let n = data.len() as f64;
    if n == 0.0 {
        return DistributionProfile {
            name: name.into(), mean: 0.0, variance: 0.0,
            skewness: 0.0, kurtosis: 0.0, entropy: 0.0, stride_coefficient: 0.0,
        };
    }

    let mean = data.iter().map(|&x| x as f64).sum::<f64>() / n;
    let variance = data.iter().map(|&x| {
        let d = x as f64 - mean;
        d * d
    }).sum::<f64>() / n;
    let std_dev = variance.sqrt().max(1e-10);

    let skewness = data.iter().map(|&x| {
        let d = (x as f64 - mean) / std_dev;
        d * d * d
    }).sum::<f64>() / n;

    let kurtosis = data.iter().map(|&x| {
        let d = (x as f64 - mean) / std_dev;
        d * d * d * d
    }).sum::<f64>() / n - 3.0;

    // Stride coefficient: correlation of consecutive differences
    let stride_coeff = if data.len() > 2 {
        let diffs: Vec<f64> = data.windows(2).map(|w| w[1] as f64 - w[0] as f64).collect();
        let diff_mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let diff_var = diffs.iter().map(|&d| (d - diff_mean).powi(2)).sum::<f64>() / diffs.len() as f64;
        1.0 - (diff_var / (variance + 1e-10)).min(1.0)
    } else {
        0.0
    };

    // Shannon entropy of binned distribution
    let num_bins = 64usize;
    let max_val = data.iter().max().copied().unwrap_or(1) as f64;
    let bin_width = (max_val / num_bins as f64).max(1.0);
    let mut bins = vec![0u64; num_bins];
    for &x in data {
        let bin = ((x as f64 / bin_width) as usize).min(num_bins - 1);
        bins[bin] += 1;
    }
    let entropy = bins.iter().map(|&c| {
        if c == 0 { 0.0 } else {
            let p = c as f64 / n;
            -p * p.ln()
        }
    }).sum::<f64>();

    DistributionProfile {
        name: name.into(), mean, variance, skewness, kurtosis, entropy, stride_coefficient: stride_coeff,
    }
}

/// Run distributional sensitivity analysis for PGO.
pub fn analyze_pgo_sensitivity(
    hierarchy: &CacheHierarchy,
    base_size: usize,
) -> DistributionalAnalysis {
    let distributions = generate_test_distributions(base_size, 42);
    let optimizer = ProfileGuidedOptimizer::new(hierarchy.clone()).with_sample_fraction(1.0);

    let mut crossover_thresholds = Vec::new();
    let mut profiles = Vec::new();
    let mut hash_recommendations: HashMap<String, usize> = HashMap::new();
    let mut block_sizes: Vec<usize> = Vec::new();

    for (name, data) in &distributions {
        let profile = optimizer.profile(data);
        let knobs = optimizer.optimize_from_profile(&profile);

        // Estimate crossover threshold where HP beats CO
        let crossover = estimate_crossover(data, &profile);
        crossover_thresholds.push((name.clone(), crossover));

        *hash_recommendations.entry(format!("{:?}", knobs.hash_family)).or_insert(0) += 1;
        block_sizes.push(knobs.block_size);

        profiles.push(characterize_distribution(name, data));
    }

    // Hash family stability: fraction that agree with the mode
    let mode_count = hash_recommendations.values().max().copied().unwrap_or(0);
    let total = hash_recommendations.values().sum::<usize>().max(1);
    let hash_family_stability = mode_count as f64 / total as f64;

    // Block size stability: coefficient of variation
    let bs_mean = block_sizes.iter().sum::<usize>() as f64 / block_sizes.len().max(1) as f64;
    let bs_var = block_sizes.iter().map(|&b| {
        let d = b as f64 - bs_mean;
        d * d
    }).sum::<f64>() / block_sizes.len().max(1) as f64;
    let block_size_stability = 1.0 - (bs_var.sqrt() / bs_mean.max(1.0)).min(1.0);

    // Crossover CI (simple: min/max as conservative interval)
    let thresholds: Vec<usize> = crossover_thresholds.iter().map(|(_, t)| *t).collect();
    let ci_lower = thresholds.iter().min().copied().unwrap_or(0);
    let ci_upper = thresholds.iter().max().copied().unwrap_or(0);

    // Overall sensitivity: how much do recommendations vary?
    let sensitivity = 1.0 - (hash_family_stability * block_size_stability);

    DistributionalAnalysis {
        distributions_tested: distributions.len(),
        crossover_thresholds,
        hash_family_stability,
        block_size_stability,
        sensitivity_score: sensitivity,
        profiles,
        crossover_ci_lower: ci_lower,
        crossover_ci_upper: ci_upper,
    }
}

/// Estimate the crossover point where hash-partition overhead exceeds benefit.
fn estimate_crossover(data: &[u64], profile: &ProfileData) -> usize {
    let wss = profile.working_set_size.max(1);
    let locality = profile.spatial_locality_score.max(0.01);
    let crossover = (wss as f64 * 4.0 / locality) as usize;
    crossover.max(256).min(data.len() * 16)
}

/// Detailed crossover sensitivity analysis across distributions.
#[derive(Debug, Clone)]
pub struct CrossoverSensitivity {
    pub per_distribution: Vec<(String, usize)>,
    /// Sensitivity ratio: max(n*) / min(n*).
    pub sensitivity_ratio: f64,
    pub high_sensitivity: bool,
    pub mean_threshold: f64,
    pub std_threshold: f64,
    pub ci_95: (f64, f64),
}

/// Compute detailed crossover sensitivity analysis.
pub fn analyze_crossover_sensitivity(
    hierarchy: &CacheHierarchy,
    base_size: usize,
) -> CrossoverSensitivity {
    let distributions = generate_test_distributions(base_size, 42);
    let optimizer = ProfileGuidedOptimizer::new(hierarchy.clone()).with_sample_fraction(1.0);

    let mut thresholds = Vec::new();
    let mut per_dist = Vec::new();

    for (name, data) in &distributions {
        let profile = optimizer.profile(data);
        let crossover = estimate_crossover(data, &profile);
        thresholds.push(crossover as f64);
        per_dist.push((name.clone(), crossover));
    }

    let min_t = thresholds.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_t = thresholds.iter().cloned().fold(0.0f64, f64::max);
    let sensitivity_ratio = if min_t > 0.0 { max_t / min_t } else { f64::INFINITY };

    let (mean, ci_lo, ci_hi) = confidence_interval(&thresholds, 0.95);
    let n = thresholds.len() as f64;
    let variance = thresholds.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);

    CrossoverSensitivity {
        per_distribution: per_dist,
        sensitivity_ratio,
        high_sensitivity: sensitivity_ratio > 2.0,
        mean_threshold: mean,
        std_threshold: variance.sqrt(),
        ci_95: (ci_lo, ci_hi),
    }
}

/// Per-algorithm crossover analysis result.
#[derive(Debug, Clone)]
pub struct AlgorithmCrossover {
    pub algorithm_name: String,
    pub crossover_n: Option<usize>,
    pub hp_ratios: Vec<(usize, f64)>,
    pub co_ratios: Vec<(usize, f64)>,
}

/// Find the input size where cache-oblivious beats hash-partition.
pub fn find_algorithm_crossover(
    name: &str,
    hp_misses: &[(usize, f64)],
    co_misses: &[(usize, f64)],
) -> AlgorithmCrossover {
    let mut crossover_n = None;
    for (hp, co) in hp_misses.iter().zip(co_misses.iter()) {
        if co.1 < hp.1 {
            crossover_n = Some(hp.0);
            break;
        }
    }
    AlgorithmCrossover {
        algorithm_name: name.to_string(),
        crossover_n,
        hp_ratios: hp_misses.to_vec(),
        co_ratios: co_misses.to_vec(),
    }
}

/// Welch's t-test for comparing two sample means.
pub fn welch_t_test(sample1: &[f64], sample2: &[f64]) -> (f64, f64) {
    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;
    if n1 < 2.0 || n2 < 2.0 {
        return (0.0, 1.0);
    }

    let mean1 = sample1.iter().sum::<f64>() / n1;
    let mean2 = sample2.iter().sum::<f64>() / n2;
    let var1 = sample1.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let var2 = sample2.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

    let se = (var1 / n1 + var2 / n2).sqrt();
    if se < 1e-15 {
        return (0.0, 1.0);
    }

    let t = (mean1 - mean2) / se;

    // Welch-Satterthwaite degrees of freedom
    let numerator = (var1 / n1 + var2 / n2).powi(2);
    let denom = (var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0);
    let df = if denom > 0.0 { numerator / denom } else { n1 + n2 - 2.0 };

    // Approximate p-value using normal distribution for large df
    let p_value = if df > 30.0 {
        2.0 * (1.0 - normal_cdf(t.abs()))
    } else {
        // Conservative: use 2-sided t-distribution approximation
        2.0 * (1.0 - normal_cdf(t.abs() * (1.0 - 1.0 / (4.0 * df))))
    };

    (t, p_value.max(0.0).min(1.0))
}

/// Approximate standard normal CDF.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Approximation of the error function (Abramowitz & Stegun).
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    sign * y
}

/// Confidence interval for a sample mean.
pub fn confidence_interval(samples: &[f64], confidence: f64) -> (f64, f64, f64) {
    let n = samples.len() as f64;
    if n < 2.0 {
        let mean = samples.first().copied().unwrap_or(0.0);
        return (mean, mean, mean);
    }

    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let se = (variance / n).sqrt();

    // z-score for confidence level (approximate)
    let z = match confidence {
        c if c >= 0.99 => 2.576,
        c if c >= 0.95 => 1.960,
        c if c >= 0.90 => 1.645,
        _ => 1.960,
    };

    (mean, mean - z * se, mean + z * se)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_characterize_uniform() {
        let data: Vec<u64> = (0..1000).collect();
        let profile = characterize_distribution("uniform", &data);
        assert!(profile.mean > 0.0);
        assert!(profile.variance > 0.0);
        assert!(profile.entropy > 0.0);
    }

    #[test]
    fn test_characterize_constant() {
        let data: Vec<u64> = vec![42; 100];
        let profile = characterize_distribution("constant", &data);
        assert!((profile.variance).abs() < 1e-10);
    }

    #[test]
    fn test_generate_distributions() {
        let dists = generate_test_distributions(100, 42);
        assert_eq!(dists.len(), 6);
        for (name, data) in &dists {
            assert_eq!(data.len(), 100, "Distribution {} has wrong length", name);
        }
    }

    #[test]
    fn test_welch_t_test_identical() {
        let s1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s2: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (t, p) = welch_t_test(&s1, &s2);
        assert!(t.abs() < 1e-10, "t-statistic should be ~0 for identical samples");
        assert!(p > 0.99, "p-value should be ~1 for identical samples");
    }

    #[test]
    fn test_welch_t_test_different() {
        let s1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s2: Vec<f64> = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let (t, p) = welch_t_test(&s1, &s2);
        assert!(t.abs() > 2.0, "t-statistic should be large for different samples");
        assert!(p < 0.05, "p-value should be small for different samples");
    }

    #[test]
    fn test_confidence_interval() {
        let samples: Vec<f64> = vec![10.0, 12.0, 11.0, 13.0, 11.0, 12.0, 10.0, 13.0];
        let (mean, lower, upper) = confidence_interval(&samples, 0.95);
        assert!(lower < mean);
        assert!(upper > mean);
        assert!((mean - 11.5).abs() < 1.0);
    }

    #[test]
    fn test_pgo_sensitivity_analysis() {
        let hierarchy = CacheHierarchy::default_hierarchy();
        let analysis = analyze_pgo_sensitivity(&hierarchy, 256);
        assert_eq!(analysis.distributions_tested, 6);
        assert!(analysis.sensitivity_score >= 0.0);
        assert!(analysis.sensitivity_score <= 1.0);
        assert!(analysis.crossover_ci_lower <= analysis.crossover_ci_upper);
    }

    #[test]
    fn test_normal_cdf_bounds() {
        assert!(normal_cdf(0.0) > 0.49 && normal_cdf(0.0) < 0.51);
        assert!(normal_cdf(3.0) > 0.99);
        assert!(normal_cdf(-3.0) < 0.01);
    }

    #[test]
    fn test_erf_symmetry() {
        for x in [0.5, 1.0, 1.5, 2.0] {
            let pos = erf(x);
            let neg = erf(-x);
            assert!((pos + neg).abs() < 1e-10, "erf should be anti-symmetric");
        }
    }

    #[test]
    fn test_crossover_sensitivity() {
        let hierarchy = CacheHierarchy::default_hierarchy();
        let sensitivity = analyze_crossover_sensitivity(&hierarchy, 256);
        assert_eq!(sensitivity.per_distribution.len(), 6);
        assert!(sensitivity.sensitivity_ratio >= 1.0);
        assert!(sensitivity.mean_threshold > 0.0);
        assert!(sensitivity.ci_95.0 <= sensitivity.ci_95.1);
    }

    #[test]
    fn test_crossover_sensitivity_ratio() {
        let hierarchy = CacheHierarchy::default_hierarchy();
        let sensitivity = analyze_crossover_sensitivity(&hierarchy, 512);
        assert!(sensitivity.sensitivity_ratio.is_finite());
        assert!(sensitivity.sensitivity_ratio >= 1.0);
    }

    #[test]
    fn test_algorithm_crossover_hp_always_wins() {
        let hp = vec![(256, 0.1), (1024, 0.15), (4096, 0.2)];
        let co = vec![(256, 0.5), (1024, 0.6), (4096, 0.7)];
        let result = find_algorithm_crossover("test_algo", &hp, &co);
        assert!(result.crossover_n.is_none());
    }

    #[test]
    fn test_algorithm_crossover_found() {
        let hp = vec![(256, 0.1), (1024, 0.3), (4096, 0.6)];
        let co = vec![(256, 0.5), (1024, 0.2), (4096, 0.1)];
        let result = find_algorithm_crossover("test_algo", &hp, &co);
        assert_eq!(result.crossover_n, Some(1024));
    }
}
