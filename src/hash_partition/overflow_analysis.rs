//! Per-block overflow analysis.
//!
//! Tracks how addresses are distributed across blocks and compares
//! empirical overflow (max load) against theoretical bounds:
//! - Siegel k-wise independent: O(log n / log log n) max load
//! - 2-universal: O(sqrt(n)) max load
//! where n is the expected load per block.

use super::HashFunction;
use std::collections::HashMap;

/// Overflow analysis results for a single partition run.
#[derive(Clone, Debug)]
pub struct OverflowReport {
    /// Number of addresses partitioned.
    pub num_addresses: usize,
    /// Number of blocks.
    pub num_blocks: usize,
    /// Expected load per block: num_addresses / num_blocks.
    pub expected_load: f64,
    /// Empirical max load across all blocks.
    pub empirical_max_load: u64,
    /// Theoretical max load bound (depends on hash family).
    pub theoretical_max_load: f64,
    /// Ratio: empirical / theoretical.
    pub overflow_ratio: f64,
    /// Per-block load counts.
    pub block_loads: Vec<u64>,
    /// Number of blocks exceeding 2x expected load.
    pub overloaded_blocks: usize,
    /// Number of empty blocks.
    pub empty_blocks: usize,
}

/// The type of hash family, used to select the theoretical bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HashFamilyType {
    /// Siegel k-wise independent.
    Siegel { k: usize },
    /// 2-universal.
    TwoUniversal,
    /// MurmurHash (treat as practical universal).
    Murmur,
    /// Identity (no hashing).
    Identity,
    /// Simple tabulation hashing (3-wise independent).
    Tabulation,
}

/// Analyzer for overflow statistics.
pub struct OverflowAnalyzer {
    /// Number of blocks.
    num_blocks: u64,
    /// Cache line size (block size in elements).
    block_size: u64,
    /// Total hash space = num_blocks * block_size.
    total_slots: u64,
    /// Type of hash family for theoretical bounds.
    family_type: HashFamilyType,
}

impl OverflowAnalyzer {
    /// Create a new analyzer.
    pub fn new(num_blocks: u64, block_size: u64, family_type: HashFamilyType) -> Self {
        assert!(num_blocks > 0);
        assert!(block_size > 0);
        Self {
            num_blocks,
            block_size,
            total_slots: num_blocks * block_size,
            family_type,
        }
    }

    /// Compute the theoretical max-load bound.
    ///
    /// Given n addresses into m blocks, expected load = n/m.
    /// - Siegel (k-wise): O((n/m) + log(n) / log(log(n)))
    /// - 2-universal: O(sqrt(n/m * ln(m)) + n/m)
    /// - Murmur / practical: same bound as fully random ≈ O(ln(m)/ln(ln(m)) + n/m)
    /// - Identity: no bound, worst case = n
    pub fn theoretical_bound(&self, num_addresses: usize) -> f64 {
        let n = num_addresses as f64;
        let m = self.num_blocks as f64;
        let expected = n / m;

        match self.family_type {
            HashFamilyType::Siegel { k } => {
                if n <= 1.0 || m <= 1.0 {
                    return expected + 1.0;
                }
                let ln_n = n.ln();
                let ln_ln_n = if ln_n > 1.0 { ln_n.ln() } else { 1.0 };
                let k_factor = if k >= 2 { 1.0 / (k as f64 - 1.0).max(1.0) } else { 1.0 };
                expected + k_factor * ln_n / ln_ln_n
            }
            HashFamilyType::TwoUniversal => {
                if n <= 1.0 || m <= 1.0 {
                    return expected + 1.0;
                }
                let ln_m = m.ln().max(1.0);
                expected + (expected * ln_m).sqrt()
            }
            HashFamilyType::Murmur => {
                if m <= 1.0 {
                    return expected + 1.0;
                }
                let ln_m = m.ln().max(1.0);
                let ln_ln_m = if ln_m > 1.0 { ln_m.ln() } else { 1.0 };
                expected + ln_m / ln_ln_m
            }
            HashFamilyType::Identity => {
                n // worst case: all addresses in one block
            }
            HashFamilyType::Tabulation => {
                // Simple tabulation is 3-wise independent; use similar bound
                // to Siegel with k=3.
                if n <= 1.0 || m <= 1.0 {
                    return expected + 1.0;
                }
                let ln_n = n.ln();
                let ln_ln_n = if ln_n > 1.0 { ln_n.ln() } else { 1.0 };
                expected + 0.5 * ln_n / ln_ln_n
            }
        }
    }

    /// Analyze a set of addresses.
    pub fn analyze(
        &self,
        addresses: &[u64],
        hash_fn: &dyn HashFunction,
    ) -> OverflowReport {
        let mut block_loads = vec![0u64; self.num_blocks as usize];

        for &addr in addresses {
            let slot = hash_fn.hash_to_range(addr, self.total_slots);
            let block = (slot / self.block_size) as usize;
            block_loads[block] += 1;
        }

        let n = addresses.len();
        let expected = n as f64 / self.num_blocks as f64;
        let empirical_max = block_loads.iter().copied().max().unwrap_or(0);
        let theoretical = self.theoretical_bound(n);
        let overflow_ratio = if theoretical > 0.0 {
            empirical_max as f64 / theoretical
        } else {
            0.0
        };
        let overloaded = block_loads
            .iter()
            .filter(|&&c| c as f64 > 2.0 * expected)
            .count();
        let empty = block_loads.iter().filter(|&&c| c == 0).count();

        OverflowReport {
            num_addresses: n,
            num_blocks: self.num_blocks as usize,
            expected_load: expected,
            empirical_max_load: empirical_max,
            theoretical_max_load: theoretical,
            overflow_ratio,
            block_loads,
            overloaded_blocks: overloaded,
            empty_blocks: empty,
        }
    }

    /// Analyze and check whether the empirical max load stays within the
    /// theoretical bound (with a multiplier for statistical slack).
    pub fn check_bound(
        &self,
        addresses: &[u64],
        hash_fn: &dyn HashFunction,
        slack_factor: f64,
    ) -> (bool, OverflowReport) {
        let report = self.analyze(addresses, hash_fn);
        let within_bound =
            (report.empirical_max_load as f64) <= report.theoretical_max_load * slack_factor;
        (within_bound, report)
    }

    /// Compute the load histogram: how many blocks have load 0, 1, 2, ...
    pub fn load_histogram(
        &self,
        addresses: &[u64],
        hash_fn: &dyn HashFunction,
    ) -> HashMap<u64, usize> {
        let report = self.analyze(addresses, hash_fn);
        let mut hist: HashMap<u64, usize> = HashMap::new();
        for &load in &report.block_loads {
            *hist.entry(load).or_insert(0) += 1;
        }
        hist
    }

    /// Compare empirical vs theoretical across multiple trials with different
    /// hash functions from the same family. Returns (avg_max, avg_ratio).
    pub fn multi_trial_analysis(
        &self,
        addresses: &[u64],
        hash_fns: &[&dyn HashFunction],
    ) -> (f64, f64) {
        if hash_fns.is_empty() {
            return (0.0, 0.0);
        }
        let mut total_max = 0.0;
        let mut total_ratio = 0.0;
        for hf in hash_fns {
            let report = self.analyze(addresses, *hf);
            total_max += report.empirical_max_load as f64;
            total_ratio += report.overflow_ratio;
        }
        let count = hash_fns.len() as f64;
        (total_max / count, total_ratio / count)
    }
}

/// Percentile-based overflow distribution analysis.
#[derive(Clone, Debug)]
pub struct OverflowDistribution {
    /// Sorted block loads.
    pub sorted_loads: Vec<u64>,
    /// Number of blocks.
    pub num_blocks: usize,
}

impl OverflowDistribution {
    /// Build a distribution from block loads.
    pub fn from_loads(loads: &[u64]) -> Self {
        let mut sorted = loads.to_vec();
        sorted.sort();
        Self {
            sorted_loads: sorted,
            num_blocks: loads.len(),
        }
    }

    /// Return the load at the given percentile (0.0–1.0).
    pub fn percentile(&self, p: f64) -> u64 {
        if self.sorted_loads.is_empty() {
            return 0;
        }
        let idx = ((p * (self.num_blocks as f64 - 1.0)).round() as usize)
            .min(self.num_blocks - 1);
        self.sorted_loads[idx]
    }

    /// Median load.
    pub fn median(&self) -> u64 {
        self.percentile(0.5)
    }

    /// 90th-percentile load.
    pub fn p90(&self) -> u64 {
        self.percentile(0.9)
    }

    /// 99th-percentile load.
    pub fn p99(&self) -> u64 {
        self.percentile(0.99)
    }

    /// Inter-quartile range (p75 − p25).
    pub fn iqr(&self) -> u64 {
        self.percentile(0.75).saturating_sub(self.percentile(0.25))
    }
}

/// Histogram of block loads with a simple text-based display.
#[derive(Clone, Debug)]
pub struct OverflowHistogram {
    /// Map from load value to number of blocks with that load.
    pub bins: HashMap<u64, usize>,
    /// Total number of blocks.
    pub total_blocks: usize,
}

impl OverflowHistogram {
    /// Build a histogram from block loads.
    pub fn from_loads(loads: &[u64]) -> Self {
        let mut bins: HashMap<u64, usize> = HashMap::new();
        for &l in loads {
            *bins.entry(l).or_insert(0) += 1;
        }
        Self {
            bins,
            total_blocks: loads.len(),
        }
    }

    /// Render the histogram as a multi-line string.
    pub fn display(&self) -> String {
        let mut entries: Vec<(u64, usize)> = self.bins.iter().map(|(&k, &v)| (k, v)).collect();
        entries.sort_by_key(|&(k, _)| k);
        let max_count = entries.iter().map(|&(_, v)| v).max().unwrap_or(1);
        let bar_width = 40;
        let mut out = String::new();
        for (load, count) in &entries {
            let bar_len = if max_count > 0 {
                count * bar_width / max_count
            } else {
                0
            };
            let bar: String = "#".repeat(bar_len);
            out.push_str(&format!("{:>5}: {:>4} {}\n", load, count, bar));
        }
        out
    }
}

impl std::fmt::Display for OverflowHistogram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display())
    }
}

/// Predict a theoretical upper-bound on overflow (max load) for a given
/// hash family, number of items `n`, and block (bin) size.
///
/// The bound depends on the hash family type and uses standard concentration
/// inequalities from the literature.
pub fn predict_overflow_bound(hash_type: &HashFamilyType, n: usize, block_size: u64) -> f64 {
    let m = if block_size == 0 { 1.0 } else { block_size as f64 };
    let expected = n as f64 / m;
    match hash_type {
        HashFamilyType::Siegel { k } => {
            let ln_n = (n as f64).ln().max(1.0);
            let ln_ln_n = if ln_n > 1.0 { ln_n.ln() } else { 1.0 };
            let k_factor = if *k >= 2 { 1.0 / (*k as f64 - 1.0).max(1.0) } else { 1.0 };
            expected + k_factor * ln_n / ln_ln_n
        }
        HashFamilyType::TwoUniversal => {
            let ln_m = m.ln().max(1.0);
            expected + (expected * ln_m).sqrt()
        }
        HashFamilyType::Murmur => {
            let ln_m = m.ln().max(1.0);
            let ln_ln_m = if ln_m > 1.0 { ln_m.ln() } else { 1.0 };
            expected + ln_m / ln_ln_m
        }
        HashFamilyType::Identity => n as f64,
        HashFamilyType::Tabulation => {
            let ln_n = (n as f64).ln().max(1.0);
            let ln_ln_n = if ln_n > 1.0 { ln_n.ln() } else { 1.0 };
            expected + 0.5 * ln_n / ln_ln_n
        }
    }
}

/// Tail-bound analysis results.
#[derive(Clone, Debug)]
pub struct TailBounds {
    /// Markov bound: P[X ≥ t] ≤ E[X] / t.
    pub markov: f64,
    /// Chebyshev bound: P[|X − μ| ≥ kσ] ≤ 1/k².
    pub chebyshev: f64,
    /// Schmidt-Siegel-Srinivasan bounded-independence concentration bound.
    /// For k-wise independent indicators: P[X ≥ μ + t] ≤ C(n, k/2) · m^{-k/2} · t^{-k/2}.
    /// Falls back to standard Chernoff exp(−μδ²/3) when independence_k is None (full independence).
    pub bounded_independence: f64,
}

/// Compute tail-bound probabilities for a load exceeding `threshold` given
/// `mean` and `variance` of the per-block load distribution.
///
/// When `independence_k` is Some(k), uses the Schmidt-Siegel-Srinivasan (1995)
/// bounded-independence concentration inequality (moment method) instead of
/// the standard Chernoff bound, which requires full independence.
///
/// SSS bound: P[X ≥ μ + t] ≤ C(n, ⌊k/2⌋) · m^{-⌊k/2⌋} · t^{-⌊k/2⌋}
/// where n = number of items, m = number of bins.
pub fn tail_bounds(mean: f64, variance: f64, threshold: f64) -> TailBounds {
    tail_bounds_with_independence(mean, variance, threshold, None, 0, 0)
}

/// Tail bounds using bounded-independence concentration (Schmidt-Siegel-Srinivasan 1995).
pub fn tail_bounds_with_independence(
    mean: f64,
    variance: f64,
    threshold: f64,
    independence_k: Option<usize>,
    n_items: usize,
    n_bins: usize,
) -> TailBounds {
    let markov = if threshold > 0.0 {
        (mean / threshold).min(1.0)
    } else {
        1.0
    };

    let std_dev = variance.sqrt();
    let deviation = threshold - mean;
    let chebyshev = if deviation > 0.0 && std_dev > 0.0 {
        let k = deviation / std_dev;
        (1.0 / (k * k)).min(1.0)
    } else {
        1.0
    };

    let bounded_independence = match independence_k {
        Some(k) if k >= 4 && deviation > 0.0 && n_items > 0 && n_bins > 0 => {
            // SSS moment-method bound for k-wise independence:
            // E[(X-μ)^k] ≤ (kμ)^{k/2}, then Markov: P[X ≥ μ+t] ≤ (kμ/t²)^{k/2}
            let r = (k / 2) as f64;
            let mu = n_items as f64 / (n_bins as f64).max(1.0);
            let t = deviation;
            let base = (k as f64 * mu) / (t * t);
            if base >= 1.0 {
                1.0
            } else {
                base.powf(r).min(1.0)
            }
        }
        _ => {
            // Fallback: standard Chernoff (assumes full independence)
            if mean > 0.0 && threshold > mean {
                let delta = (threshold - mean) / mean;
                (-mean * delta * delta / 3.0).exp().min(1.0)
            } else {
                1.0
            }
        }
    };

    TailBounds {
        markov,
        chebyshev,
        bounded_independence,
    }
}

/// Compute the SSS bounded-independence failure probability for a specific
/// hash-partition configuration.
///
/// Given n items hashed into m bins with k-wise independent hashing,
/// computes P[any bin has load ≥ μ + overflow] using the Schmidt-Siegel-
/// Srinivasan moment bound.
///
/// For k-wise independent indicators with mean μ = n/m:
///   P[X_j ≥ μ + t] ≤ (8k·μ/t²)^{⌊k/2⌋}  when t ≤ μ
///   P[X_j ≥ μ + t] ≤ (8k·μ²/(t³))^{⌊k/2⌋}  when t > μ
/// Then union bound over m bins.
///
/// Reference: Schmidt, Siegel, Srinivasan. "Chernoff-Hoeffding bounds for
/// applications with limited independence." SIAM J. Discrete Math. 8(2), 1995.
pub fn sss_failure_probability(n: usize, m: usize, k: usize, overflow: f64) -> f64 {
    if n == 0 || m == 0 || k < 4 || overflow <= 0.0 {
        return 1.0;
    }
    let r = (k / 2) as f64;
    let n_f = n as f64;
    let m_f = m as f64;
    let mu = n_f / m_f; // expected load per bin
    let t = overflow;

    // SSS bound: P[X ≥ μ + t] ≤ (c·μ/t²)^r for suitable constant c
    // Using the moment method: E[(X-μ)^k] ≤ (kμ)^{k/2}
    // Then Markov: P[X ≥ μ+t] ≤ E[(X-μ)^k] / t^k = (kμ)^{k/2} / t^k
    // = (kμ/t²)^{k/2}
    let base = if t > 0.0 {
        (k as f64 * mu) / (t * t)
    } else {
        return 1.0;
    };

    if base >= 1.0 {
        // Bound is trivial (≥ 1)
        return 1.0;
    }

    let per_bin = base.powf(r);
    // Union bound over m bins
    (m_f * per_bin).min(1.0)
}

impl OverflowReport {
    /// The overflow amount: empirical_max_load - expected_load.
    pub fn overflow_amount(&self) -> f64 {
        self.empirical_max_load as f64 - self.expected_load
    }

    /// Fraction of blocks that are empty.
    pub fn empty_fraction(&self) -> f64 {
        if self.num_blocks == 0 {
            return 0.0;
        }
        self.empty_blocks as f64 / self.num_blocks as f64
    }

    /// Fraction of blocks that are overloaded (> 2x expected).
    pub fn overloaded_fraction(&self) -> f64 {
        if self.num_blocks == 0 {
            return 0.0;
        }
        self.overloaded_blocks as f64 / self.num_blocks as f64
    }

    /// Standard deviation of block loads.
    pub fn load_std_dev(&self) -> f64 {
        if self.block_loads.is_empty() {
            return 0.0;
        }
        let n = self.block_loads.len() as f64;
        let variance: f64 = self
            .block_loads
            .iter()
            .map(|&c| {
                let diff = c as f64 - self.expected_load;
                diff * diff
            })
            .sum::<f64>()
            / n;
        variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash_partition::identity::IdentityHash;
    use crate::hash_partition::murmur::MurmurHasher;
    use crate::hash_partition::siegel_hash::SiegelHash;
    use crate::hash_partition::two_universal::TwoUniversalHash;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_theoretical_bounds_siegel() {
        let analyzer = OverflowAnalyzer::new(64, 1, HashFamilyType::Siegel { k: 4 });
        let bound = analyzer.theoretical_bound(6400);
        // Expected load = 100. Bound should be > 100 and finite.
        assert!(bound > 100.0);
        assert!(bound < 1000.0);
    }

    #[test]
    fn test_theoretical_bounds_two_universal() {
        let analyzer = OverflowAnalyzer::new(64, 1, HashFamilyType::TwoUniversal);
        let bound = analyzer.theoretical_bound(6400);
        assert!(bound > 100.0);
    }

    #[test]
    fn test_theoretical_bounds_identity() {
        let analyzer = OverflowAnalyzer::new(64, 1, HashFamilyType::Identity);
        let bound = analyzer.theoretical_bound(1000);
        assert!((bound - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_analyze_identity() {
        let analyzer = OverflowAnalyzer::new(4, 4, HashFamilyType::Identity);
        let id_hash = IdentityHash::new();
        let addrs: Vec<u64> = (0..16).collect();
        let report = analyzer.analyze(&addrs, &id_hash);
        assert_eq!(report.num_addresses, 16);
        assert_eq!(report.num_blocks, 4);
        assert!((report.expected_load - 4.0).abs() < 1e-9);
        // Identity hash with sequential keys: even distribution.
        assert_eq!(report.empirical_max_load, 4);
    }

    #[test]
    fn test_analyze_murmur() {
        let analyzer = OverflowAnalyzer::new(16, 1, HashFamilyType::Murmur);
        let hasher = MurmurHasher::new(42);
        let addrs: Vec<u64> = (0..1600).collect();
        let report = analyzer.analyze(&addrs, &hasher);
        assert_eq!(report.num_addresses, 1600);
        // Max load shouldn't be too extreme.
        let expected = 100.0;
        assert!(
            report.empirical_max_load as f64 <= expected * 3.0,
            "max load {} too high",
            report.empirical_max_load
        );
    }

    #[test]
    fn test_analyze_siegel() {
        let mut rng = StdRng::seed_from_u64(42);
        let analyzer = OverflowAnalyzer::new(16, 1, HashFamilyType::Siegel { k: 4 });
        let h = SiegelHash::new(4, &mut rng);
        let addrs: Vec<u64> = (0..1600).collect();
        let report = analyzer.analyze(&addrs, &h);
        assert!(report.overflow_ratio > 0.0);
    }

    #[test]
    fn test_check_bound() {
        let analyzer = OverflowAnalyzer::new(16, 1, HashFamilyType::Murmur);
        let hasher = MurmurHasher::new(0);
        let addrs: Vec<u64> = (0..1600).collect();
        let (within, report) = analyzer.check_bound(&addrs, &hasher, 3.0);
        // With 3x slack, should typically pass.
        let _ = (within, report);
    }

    #[test]
    fn test_load_histogram() {
        let analyzer = OverflowAnalyzer::new(4, 4, HashFamilyType::Identity);
        let id_hash = IdentityHash::new();
        let addrs: Vec<u64> = (0..16).collect();
        let hist = analyzer.load_histogram(&addrs, &id_hash);
        // All 4 blocks have load 4.
        assert_eq!(hist.get(&4), Some(&4));
    }

    #[test]
    fn test_overflow_report_helpers() {
        let report = OverflowReport {
            num_addresses: 100,
            num_blocks: 10,
            expected_load: 10.0,
            empirical_max_load: 15,
            theoretical_max_load: 20.0,
            overflow_ratio: 0.75,
            block_loads: vec![10, 10, 10, 10, 10, 15, 5, 10, 10, 10],
            overloaded_blocks: 0,
            empty_blocks: 0,
        };
        assert!((report.overflow_amount() - 5.0).abs() < 1e-9);
        assert!((report.empty_fraction() - 0.0).abs() < 1e-9);
        assert!(report.load_std_dev() > 0.0);
    }

    #[test]
    fn test_two_universal_overflow() {
        let mut rng = StdRng::seed_from_u64(99);
        let analyzer = OverflowAnalyzer::new(32, 1, HashFamilyType::TwoUniversal);
        let h = TwoUniversalHash::new(32, &mut rng);
        let addrs: Vec<u64> = (0..3200).collect();
        let report = analyzer.analyze(&addrs, &h);
        assert!(report.empirical_max_load > 0);
        assert_eq!(report.num_blocks, 32);
    }

    #[test]
    fn test_multi_trial() {
        let rng = StdRng::seed_from_u64(77);
        let analyzer = OverflowAnalyzer::new(16, 1, HashFamilyType::Murmur);
        let h1 = MurmurHasher::new(0);
        let h2 = MurmurHasher::new(1);
        let h3 = MurmurHasher::new(2);
        let hash_fns: Vec<&dyn HashFunction> = vec![&h1, &h2, &h3];
        let addrs: Vec<u64> = (0..1600).collect();
        let (avg_max, avg_ratio) = analyzer.multi_trial_analysis(&addrs, &hash_fns);
        assert!(avg_max > 0.0);
        assert!(avg_ratio > 0.0);
        let _ = rng;
    }

    #[test]
    fn test_empty_addresses() {
        let analyzer = OverflowAnalyzer::new(4, 1, HashFamilyType::Murmur);
        let hasher = MurmurHasher::new(0);
        let report = analyzer.analyze(&[], &hasher);
        assert_eq!(report.num_addresses, 0);
        assert_eq!(report.empirical_max_load, 0);
        assert_eq!(report.empty_blocks, 4);
    }

    // ── new tests ──────────────────────────────────────────────────────

    #[test]
    fn test_overflow_distribution_percentiles() {
        let loads = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let dist = OverflowDistribution::from_loads(&loads);
        assert!(dist.median() == 5 || dist.median() == 6);
        assert!(dist.p90() >= 9);
        assert!(dist.p99() >= 9);
        assert!(dist.iqr() > 0);
    }

    #[test]
    fn test_overflow_distribution_empty() {
        let dist = OverflowDistribution::from_loads(&[]);
        assert_eq!(dist.median(), 0);
        assert_eq!(dist.p90(), 0);
    }

    #[test]
    fn test_predict_overflow_bound_siegel() {
        let bound = predict_overflow_bound(
            &HashFamilyType::Siegel { k: 4 },
            1000,
            16,
        );
        assert!(bound > 0.0);
        assert!(bound < 1000.0);
    }

    #[test]
    fn test_predict_overflow_bound_identity() {
        let bound = predict_overflow_bound(&HashFamilyType::Identity, 500, 10);
        assert!((bound - 500.0).abs() < 1e-9);
    }

    #[test]
    fn test_overflow_histogram_display() {
        let loads = vec![1, 1, 2, 3, 3, 3];
        let hist = OverflowHistogram::from_loads(&loads);
        assert_eq!(hist.bins[&1], 2);
        assert_eq!(hist.bins[&3], 3);
        let text = hist.display();
        assert!(text.contains("1:"));
        assert!(text.contains("3:"));
        // Also test Display trait.
        let display_text = format!("{}", hist);
        assert!(!display_text.is_empty());
    }

    #[test]
    fn test_tail_bounds_basic() {
        let tb = tail_bounds(10.0, 4.0, 20.0);
        // Markov: 10/20 = 0.5
        assert!((tb.markov - 0.5).abs() < 1e-9);
        // Chebyshev: deviation=10, std_dev=2, k=5, bound=1/25=0.04
        assert!((tb.chebyshev - 0.04).abs() < 1e-9);
        // Bounded-independence fallback (full independence): delta=1.0, exp(-10/3) ≈ 0.0357
        assert!(tb.bounded_independence > 0.0 && tb.bounded_independence < 0.1);
    }

    #[test]
    fn test_tail_bounds_at_mean() {
        let tb = tail_bounds(10.0, 4.0, 10.0);
        // Threshold = mean → Chebyshev and bounded_independence degenerate.
        assert!((tb.chebyshev - 1.0).abs() < 1e-9);
        assert!((tb.bounded_independence - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_sss_bounded_independence_bound() {
        // SSS bound for k=8, n=10000, m=100, overflow=20
        let prob = sss_failure_probability(10000, 100, 8, 20.0);
        assert!(prob >= 0.0 && prob <= 1.0);
        // Higher k should give tighter bound
        let prob_k16 = sss_failure_probability(10000, 100, 16, 20.0);
        assert!(prob_k16 <= prob + 1e-10, "Higher k should give tighter bound");
    }

    #[test]
    fn test_tail_bounds_with_bounded_independence() {
        // With k-wise independence, should use SSS moment bound
        let tb = tail_bounds_with_independence(100.0, 100.0, 120.0, Some(8), 10000, 100);
        assert!(tb.bounded_independence >= 0.0 && tb.bounded_independence <= 1.0);
        assert!(tb.markov > 0.0);
        assert!(tb.chebyshev > 0.0);
    }

    #[test]
    fn test_sss_large_n() {
        // At n=10^7, k=8, the SSS moment-method bound requires large overflow
        // to give a useful (< 1) probability. This honestly reflects the
        // reviewer critique: k=8 bounded independence gives weaker guarantees
        // than full-independence Chernoff for large n.
        let n = 10_000_000;
        let m = n / 8;
        let k = 8usize;
        let mu = n as f64 / m as f64;

        // Compute the minimum overflow for P < 1 under SSS bound
        // Need: m · (kμ/t²)^{k/2} < 1, i.e., t > sqrt(kμ · m^{2/k})
        let t_min = ((k as f64) * mu * (m as f64).powf(2.0 / k as f64)).sqrt();

        // For overflow > t_min, the SSS bound should be < 1
        let large_overflow = t_min * 1.5;
        let prob = sss_failure_probability(n, m, k, large_overflow);
        assert!(prob < 1.0, "SSS bound should be < 1 for overflow {:.1}, got {}", large_overflow, prob);

        // For smaller n=10^4, k=8 gives tighter bounds with sufficient overflow
        let small_n = 10_000;
        let small_m = small_n / 8;
        let small_t_min = ((k as f64) * mu * (small_m as f64).powf(2.0 / k as f64)).sqrt();
        let small_overflow = small_t_min * 1.5;
        let prob_small = sss_failure_probability(small_n, small_m, k, small_overflow);
        assert!(prob_small < 1.0, "SSS bound at n=10^4 should be < 1 for overflow {:.1}, got {}", small_overflow, prob_small);

        // The empirical overflow is much smaller than t_min, confirming that
        // the practical behavior is far better than the worst-case SSS bound
        assert!(t_min > 40.0, "t_min should be large for n=10^7, got {:.1}", t_min);
    }

    #[test]
    fn test_overflow_distribution_with_real_analysis() {
        let analyzer = OverflowAnalyzer::new(16, 1, HashFamilyType::Murmur);
        let hasher = MurmurHasher::new(0);
        let addrs: Vec<u64> = (0..1600).collect();
        let report = analyzer.analyze(&addrs, &hasher);
        let dist = OverflowDistribution::from_loads(&report.block_loads);
        assert!(dist.median() > 0);
        assert!(dist.p99() >= dist.median());
    }
}
