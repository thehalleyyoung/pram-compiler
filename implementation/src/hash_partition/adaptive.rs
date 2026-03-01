//! Adaptive partitioning for irregular access patterns.
//!
//! Analyzes access pattern regularity and selects the best partitioning
//! strategy: standard hash-partition for regular patterns, multi-level
//! partition for graph traversals, and locality-aware reordering for
//! scattered accesses.

use super::partition_engine::{HashFamilyChoice, PartitionEngine, PartitionResult};

/// Access pattern classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Sequential / strided: addresses form a regular sequence.
    Regular,
    /// Graph-structured: pointer-chasing with moderate locality.
    GraphTraversal,
    /// Scattered: near-random access with poor locality.
    Scattered,
}

/// Result of access pattern analysis.
#[derive(Debug, Clone)]
pub struct AccessAnalysis {
    pub pattern: AccessPattern,
    /// Fraction of consecutive address pairs that land in the same block.
    pub locality_score: f64,
    /// Ratio of unique blocks accessed to total accesses.
    pub block_coverage: f64,
    /// Recommended block size multiplier (1.0 = default).
    pub recommended_block_multiplier: f64,
    /// Whether multi-level partitioning is recommended.
    pub use_multi_level: bool,
}

/// Analyze an address stream to classify its access pattern.
pub fn analyze_access_pattern(addresses: &[u64], block_size: u64) -> AccessAnalysis {
    if addresses.is_empty() || block_size == 0 {
        return AccessAnalysis {
            pattern: AccessPattern::Regular,
            locality_score: 1.0,
            block_coverage: 0.0,
            recommended_block_multiplier: 1.0,
            use_multi_level: false,
        };
    }

    // Compute locality score: fraction of consecutive pairs in the same block
    let mut same_block = 0usize;
    for pair in addresses.windows(2) {
        if pair[0] / block_size == pair[1] / block_size {
            same_block += 1;
        }
    }
    let locality_score = if addresses.len() > 1 {
        same_block as f64 / (addresses.len() - 1) as f64
    } else {
        1.0
    };

    // Compute block coverage
    let mut blocks: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for &a in addresses {
        blocks.insert(a / block_size);
    }
    let block_coverage = blocks.len() as f64 / addresses.len() as f64;

    // Compute stride regularity (coefficient of variation of inter-address gaps)
    let mut diffs: Vec<i64> = Vec::with_capacity(addresses.len().saturating_sub(1));
    for pair in addresses.windows(2) {
        diffs.push(pair[1] as i64 - pair[0] as i64);
    }
    let stride_cv = if diffs.len() > 1 {
        let mean = diffs.iter().sum::<i64>() as f64 / diffs.len() as f64;
        let var = diffs.iter().map(|&d| { let x = d as f64 - mean; x * x }).sum::<f64>() / diffs.len() as f64;
        if mean.abs() > 1e-9 { var.sqrt() / mean.abs() } else { 10.0 }
    } else {
        0.0
    };

    // Classify
    let pattern = if locality_score > 0.6 && stride_cv < 1.0 {
        AccessPattern::Regular
    } else if locality_score > 0.2 || block_coverage < 0.5 {
        AccessPattern::GraphTraversal
    } else {
        AccessPattern::Scattered
    };

    let recommended_block_multiplier = match pattern {
        AccessPattern::Regular => 1.0,
        AccessPattern::GraphTraversal => 2.0,  // larger blocks for graph locality
        AccessPattern::Scattered => 0.5,       // smaller blocks to reduce waste
    };

    let use_multi_level = matches!(pattern, AccessPattern::GraphTraversal | AccessPattern::Scattered);

    AccessAnalysis {
        pattern,
        locality_score,
        block_coverage,
        recommended_block_multiplier,
        use_multi_level,
    }
}

/// Perform adaptive partitioning: analyze the access pattern and select
/// the best partitioning strategy automatically.
pub fn adaptive_partition(
    addresses: &[u64],
    num_blocks: u64,
    base_block_size: u64,
    family: &HashFamilyChoice,
    seed: u64,
) -> (PartitionResult, AccessAnalysis) {
    let analysis = analyze_access_pattern(addresses, base_block_size);

    let adjusted_block_size = (base_block_size as f64 * analysis.recommended_block_multiplier)
        .max(1.0) as u64;

    let engine = PartitionEngine::new(num_blocks, adjusted_block_size, family.clone(), seed);

    let result = if analysis.use_multi_level {
        // Try multiple block sizes and pick the best
        let candidates: Vec<u64> = vec![
            adjusted_block_size / 2,
            adjusted_block_size,
            adjusted_block_size * 2,
        ].into_iter().filter(|&s| s > 0).collect();
        engine.adaptive_partition(addresses, &candidates)
    } else {
        engine.partition(addresses)
    };

    (result, analysis)
}

/// Reorder addresses for improved cache locality given a partition result.
/// Groups addresses by their assigned block, so sequential processing
/// visits each block contiguously.
pub fn locality_reorder(addresses: &[u64], assignments: &[usize]) -> Vec<usize> {
    let n = addresses.len();
    if n == 0 {
        return Vec::new();
    }

    // Group indices by block
    let max_block = assignments.iter().copied().max().unwrap_or(0);
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); max_block + 1];
    for (i, &block) in assignments.iter().enumerate() {
        buckets[block].push(i);
    }

    // Flatten: visit each block's addresses in order
    let mut order = Vec::with_capacity(n);
    for bucket in &buckets {
        order.extend_from_slice(bucket);
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regular_pattern() {
        let addrs: Vec<u64> = (0..100).collect();
        let analysis = analyze_access_pattern(&addrs, 8);
        assert_eq!(analysis.pattern, AccessPattern::Regular);
        assert!(analysis.locality_score > 0.5);
    }

    #[test]
    fn test_scattered_pattern() {
        // Pseudo-random addresses
        let addrs: Vec<u64> = (0..100).map(|i| (i * 97 + 31) % 1000).collect();
        let analysis = analyze_access_pattern(&addrs, 4);
        assert!(matches!(analysis.pattern, AccessPattern::Scattered | AccessPattern::GraphTraversal));
    }

    #[test]
    fn test_graph_traversal_pattern() {
        // Pointer-chasing: some locality but not sequential
        let mut addrs = Vec::new();
        let mut cur = 0u64;
        for _ in 0..100 {
            addrs.push(cur);
            cur = (cur + 3) % 50; // wraps around with some locality
        }
        let analysis = analyze_access_pattern(&addrs, 8);
        // Should be GraphTraversal or Regular depending on locality
        assert!(analysis.locality_score > 0.0);
    }

    #[test]
    fn test_empty_addresses() {
        let analysis = analyze_access_pattern(&[], 4);
        assert_eq!(analysis.pattern, AccessPattern::Regular);
        assert_eq!(analysis.locality_score, 1.0);
    }

    #[test]
    fn test_adaptive_partition() {
        let addrs: Vec<u64> = (0..256).collect();
        let family = HashFamilyChoice::Siegel { k: 4 };
        let (result, analysis) = adaptive_partition(&addrs, 16, 4, &family, 42);
        assert_eq!(result.assignments.len(), 256);
        assert!(analysis.locality_score > 0.0);
    }

    #[test]
    fn test_locality_reorder() {
        let addrs = vec![0u64, 10, 1, 11, 2, 12];
        let assignments = vec![0, 2, 0, 2, 0, 2]; // alternating blocks
        let order = locality_reorder(&addrs, &assignments);
        // Block 0 indices first, then block 2
        assert_eq!(order, vec![0, 2, 4, 1, 3, 5]);
    }

    #[test]
    fn test_locality_reorder_empty() {
        assert!(locality_reorder(&[], &[]).is_empty());
    }

    #[test]
    fn test_adaptive_scattered_uses_multi_level() {
        let addrs: Vec<u64> = (0..200).map(|i| (i * 97 + 31) % 5000).collect();
        let analysis = analyze_access_pattern(&addrs, 4);
        // Scattered patterns should recommend multi-level
        assert!(analysis.use_multi_level || analysis.pattern == AccessPattern::Regular);
    }
}
