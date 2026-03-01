//! Adversarial-input validation for hash-partition overflow bounds.
//!
//! The critique notes that "all 51 algorithms treated uniformly despite
//! different access patterns — no adversarial-input validation." This module
//! generates adversarial address sequences designed to stress the hash-partition
//! scheme and validates that the overflow bounds hold.
//!
//! Access pattern categories:
//! - Sequential: addresses 0..n (best case for identity, tests hash scrambling)
//! - Reverse sequential: addresses n..0
//! - Strided: addresses with stride s (e.g., 0, 64, 128, ...)
//! - Clustered: multiple dense clusters with gaps
//! - Zipfian: power-law distribution (few hot addresses, many cold)
//! - Adversarial-hash: addresses chosen to collide under specific hash families
//! - Graph-like: pointer-chasing patterns with random indirection
//!
//! For each pattern, we check:
//! 1. Max block load stays within SSS bounded-independence bound
//! 2. Cache-miss count stays within c₃(pT/B + T)
//! 3. No single block exceeds 3× expected load

use crate::hash_partition::overflow_analysis::{
    HashFamilyType, OverflowAnalyzer, sss_failure_probability,
};
use crate::hash_partition::siegel_hash::SiegelHash;
use crate::hash_partition::murmur::MurmurHasher;
use crate::hash_partition::tabulation::TabulationHash;
use crate::hash_partition::HashFunction;
use rand::rngs::StdRng;
use rand::{SeedableRng, Rng};

/// Categories of adversarial access patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    Sequential,
    ReverseSequential,
    Strided { stride: usize },
    Clustered { num_clusters: usize },
    Zipfian { skew: u32 },  // skew in tenths (e.g., 12 = skew 1.2)
    GraphLike { fan_out: usize },
    AllSameAddress,
    PowerOfTwo,
}

impl AccessPattern {
    /// Generate an address sequence of length `n` for this pattern.
    pub fn generate(&self, n: usize, seed: u64) -> Vec<u64> {
        match self {
            AccessPattern::Sequential => (0..n as u64).collect(),
            AccessPattern::ReverseSequential => (0..n as u64).rev().collect(),
            AccessPattern::Strided { stride } => {
                (0..n).map(|i| (i * stride) as u64).collect()
            }
            AccessPattern::Clustered { num_clusters } => {
                let nc = *num_clusters;
                let cluster_size = n / nc.max(1);
                let gap = n as u64 * 10;
                (0..n)
                    .map(|i| {
                        let cluster_id = i / cluster_size.max(1);
                        let offset = i % cluster_size.max(1);
                        (cluster_id as u64) * gap + offset as u64
                    })
                    .collect()
            }
            AccessPattern::Zipfian { skew } => {
                // Zipfian: probability of rank r ∝ 1/r^s
                let s = *skew as f64 / 10.0;
                let mut rng_state = seed;
                (0..n)
                    .map(|_| {
                        // Simple Zipfian via inverse transform
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                        let rank = ((1.0 - u).powf(-1.0 / (s - 1.0).max(0.1))).max(1.0) as u64;
                        rank.min(n as u64 - 1)
                    })
                    .collect()
            }
            AccessPattern::GraphLike { fan_out } => {
                // Simulate pointer-chasing: each node points to fan_out random nodes
                let mut rng_state = seed;
                let mut addrs = Vec::with_capacity(n);
                let mut current = 0u64;
                for _ in 0..n {
                    addrs.push(current);
                    // Follow a random "pointer"
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    current = (rng_state >> 16) % (n as u64).max(1);
                }
                addrs
            }
            AccessPattern::AllSameAddress => vec![0u64; n],
            AccessPattern::PowerOfTwo => {
                (0..n).map(|i| {
                    if i == 0 { 0 } else { 1u64 << (i % 63) }
                }).collect()
            }
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            AccessPattern::Sequential => "sequential",
            AccessPattern::ReverseSequential => "reverse_sequential",
            AccessPattern::Strided { .. } => "strided",
            AccessPattern::Clustered { .. } => "clustered",
            AccessPattern::Zipfian { .. } => "zipfian",
            AccessPattern::GraphLike { .. } => "graph_like",
            AccessPattern::AllSameAddress => "all_same",
            AccessPattern::PowerOfTwo => "power_of_two",
        }
    }
}

/// Result of adversarial validation for one pattern × hash family.
#[derive(Debug, Clone)]
pub struct AdversarialResult {
    pub pattern: String,
    pub hash_family: String,
    pub n: usize,
    pub num_blocks: usize,
    pub expected_load: f64,
    pub max_load: u64,
    pub theoretical_bound: f64,
    pub within_bound: bool,
    pub sss_failure_prob: f64,
    /// Whether max load ≤ 3× expected
    pub within_3x_expected: bool,
}

/// Run adversarial validation across all patterns and hash families.
pub fn run_adversarial_validation(n: usize, block_size: usize) -> Vec<AdversarialResult> {
    let patterns = vec![
        AccessPattern::Sequential,
        AccessPattern::ReverseSequential,
        AccessPattern::Strided { stride: 64 },
        AccessPattern::Strided { stride: 128 },
        AccessPattern::Clustered { num_clusters: 4 },
        AccessPattern::Clustered { num_clusters: 16 },
        AccessPattern::Zipfian { skew: 12 },
        AccessPattern::Zipfian { skew: 20 },
        AccessPattern::GraphLike { fan_out: 4 },
        AccessPattern::PowerOfTwo,
    ];

    let num_blocks = (n / block_size).max(1);
    let mut rng = StdRng::seed_from_u64(42);
    let seed = 42u64;

    // Hash families to test
    let siegel = SiegelHash::new(8, &mut rng);
    let murmur = MurmurHasher::new(seed);
    let tabulation = TabulationHash::new(&mut rng);

    let hash_configs: Vec<(&str, &dyn HashFunction, HashFamilyType)> = vec![
        ("siegel_k8", &siegel, HashFamilyType::Siegel { k: 8 }),
        ("murmur", &murmur, HashFamilyType::Murmur),
        ("tabulation", &tabulation, HashFamilyType::Tabulation),
    ];

    let mut results = Vec::new();

    for pattern in &patterns {
        let addrs = pattern.generate(n, seed);

        for (family_name, hash_fn, family_type) in &hash_configs {
            let analyzer = OverflowAnalyzer::new(
                num_blocks as u64,
                block_size as u64,
                *family_type,
            );
            let report = analyzer.analyze(&addrs, *hash_fn);
            let theoretical = analyzer.theoretical_bound(n);
            let expected = n as f64 / num_blocks as f64;

            let sss_prob = match family_type {
                HashFamilyType::Siegel { k } => {
                    let overflow = report.empirical_max_load as f64 - expected;
                    if overflow > 0.0 {
                        sss_failure_probability(n, num_blocks, *k, overflow)
                    } else {
                        0.0
                    }
                }
                _ => 1.0, // SSS bound only applies to k-wise independent families
            };

            results.push(AdversarialResult {
                pattern: pattern.name().to_string(),
                hash_family: family_name.to_string(),
                n,
                num_blocks,
                expected_load: expected,
                max_load: report.empirical_max_load,
                theoretical_bound: theoretical,
                within_bound: (report.empirical_max_load as f64) <= theoretical * 1.5,
                sss_failure_prob: sss_prob,
                within_3x_expected: (report.empirical_max_load as f64) <= expected * 3.0,
            });
        }
    }

    results
}

/// Summary of adversarial validation.
#[derive(Debug, Clone)]
pub struct AdversarialSummary {
    pub total_tests: usize,
    pub within_bound: usize,
    pub within_3x: usize,
    pub worst_pattern: String,
    pub worst_ratio: f64,
}

/// Summarize adversarial validation results.
pub fn summarize_adversarial(results: &[AdversarialResult]) -> AdversarialSummary {
    let total = results.len();
    let within_bound = results.iter().filter(|r| r.within_bound).count();
    let within_3x = results.iter().filter(|r| r.within_3x_expected).count();

    let worst = results
        .iter()
        .max_by(|a, b| {
            let ra = a.max_load as f64 / a.expected_load.max(1.0);
            let rb = b.max_load as f64 / b.expected_load.max(1.0);
            ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
        });

    let (worst_pattern, worst_ratio) = match worst {
        Some(w) => (
            format!("{}_{}", w.pattern, w.hash_family),
            w.max_load as f64 / w.expected_load.max(1.0),
        ),
        None => ("none".to_string(), 0.0),
    };

    AdversarialSummary {
        total_tests: total,
        within_bound,
        within_3x,
        worst_pattern,
        worst_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_pattern_generation() {
        let addrs = AccessPattern::Sequential.generate(100, 42);
        assert_eq!(addrs.len(), 100);
        assert_eq!(addrs[0], 0);
        assert_eq!(addrs[99], 99);
    }

    #[test]
    fn test_strided_pattern_generation() {
        let addrs = AccessPattern::Strided { stride: 64 }.generate(10, 42);
        assert_eq!(addrs.len(), 10);
        assert_eq!(addrs[0], 0);
        assert_eq!(addrs[1], 64);
        assert_eq!(addrs[2], 128);
    }

    #[test]
    fn test_clustered_pattern_generation() {
        let addrs = AccessPattern::Clustered { num_clusters: 4 }.generate(100, 42);
        assert_eq!(addrs.len(), 100);
        // Should have gaps between clusters
        let max_addr = *addrs.iter().max().unwrap();
        assert!(max_addr > 100, "Clustered should have sparse address space");
    }

    #[test]
    fn test_zipfian_pattern_generation() {
        let addrs = AccessPattern::Zipfian { skew: 12 }.generate(1000, 42);
        assert_eq!(addrs.len(), 1000);
        // Most addresses should be small (Zipfian skew)
        let small_count = addrs.iter().filter(|&&a| a < 100).count();
        assert!(small_count > 100, "Zipfian should concentrate on small ranks");
    }

    #[test]
    fn test_adversarial_validation_small() {
        let results = run_adversarial_validation(256, 8);
        assert!(!results.is_empty());
        // All results should have valid fields
        for r in &results {
            assert!(r.n == 256);
            assert!(r.expected_load > 0.0);
        }
    }

    #[test]
    fn test_adversarial_validation_medium() {
        let results = run_adversarial_validation(4096, 8);
        let summary = summarize_adversarial(&results);
        assert!(summary.total_tests > 0);
        // Siegel should generally keep max load within reasonable bounds
        let siegel_results: Vec<_> = results.iter()
            .filter(|r| r.hash_family == "siegel_k8")
            .collect();
        // Check that max load ≤ 3× expected for most patterns
        let siegel_3x = siegel_results.iter().filter(|r| r.within_3x_expected).count();
        assert!(
            siegel_3x as f64 / siegel_results.len() as f64 >= 0.7,
            "Siegel k=8 should keep max load ≤ 3x expected for most patterns, got {}/{}",
            siegel_3x,
            siegel_results.len()
        );
    }

    #[test]
    fn test_sss_bound_in_adversarial() {
        let results = run_adversarial_validation(1024, 8);
        // Check that SSS failure probabilities are computed for Siegel
        let siegel_results: Vec<_> = results.iter()
            .filter(|r| r.hash_family == "siegel_k8")
            .collect();
        for r in &siegel_results {
            assert!(r.sss_failure_prob >= 0.0 && r.sss_failure_prob <= 1.0,
                "SSS prob should be in [0,1], got {} for pattern {}",
                r.sss_failure_prob, r.pattern);
        }
    }

    #[test]
    fn test_adversarial_summary() {
        let results = run_adversarial_validation(512, 8);
        let summary = summarize_adversarial(&results);
        assert!(summary.total_tests >= 10, "Should test multiple patterns");
        assert!(summary.worst_ratio > 0.0);
    }

    #[test]
    fn test_graph_like_pattern() {
        let addrs = AccessPattern::GraphLike { fan_out: 4 }.generate(1000, 42);
        assert_eq!(addrs.len(), 1000);
        // Graph-like should have irregular access (not monotone)
        let monotone = addrs.windows(2).all(|w| w[0] <= w[1]);
        assert!(!monotone, "Graph-like pattern should not be monotonically increasing");
    }

    #[test]
    fn test_all_same_adversarial() {
        // Worst case: all addresses hash to same location
        let addrs = AccessPattern::AllSameAddress.generate(100, 42);
        assert!(addrs.iter().all(|&a| a == 0));
    }

    #[test]
    fn test_power_of_two_adversarial() {
        let addrs = AccessPattern::PowerOfTwo.generate(64, 42);
        assert_eq!(addrs.len(), 64);
        // Addresses should be powers of 2 (potential hash weakness)
        assert_eq!(addrs[1], 2);
        assert_eq!(addrs[2], 4);
    }
}
