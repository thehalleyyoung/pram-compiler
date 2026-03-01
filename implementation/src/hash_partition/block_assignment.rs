//! Assign PRAM addresses to cache-line-aligned blocks.
//!
//! Each address is hashed, then the hash output is divided into
//! cache-line-sized blocks of B elements.

use std::collections::HashMap;
use super::{BlockId, HashFunction};

/// Assigns PRAM addresses to cache-line-aligned blocks.
///
/// The assignment works as follows:
/// 1. Hash the address to a value in [0, total_slots).
/// 2. Divide the hashed value by block size B to get the block ID.
pub struct BlockAssigner {
    /// Cache line size in elements.
    block_size: u64,
    /// Total number of slots in the hash space (num_blocks * block_size).
    total_slots: u64,
    /// Total number of blocks.
    num_blocks: u64,
}

impl BlockAssigner {
    /// Create a new block assigner.
    ///
    /// - `num_blocks`: total number of cache-line blocks.
    /// - `block_size`: number of elements per cache line (B).
    pub fn new(num_blocks: u64, block_size: u64) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        assert!(num_blocks > 0, "num_blocks must be > 0");
        let total_slots = num_blocks * block_size;
        Self {
            block_size,
            total_slots,
            num_blocks,
        }
    }

    /// Return the block size B.
    pub fn block_size(&self) -> u64 {
        self.block_size
    }

    /// Return the number of blocks.
    pub fn num_blocks(&self) -> u64 {
        self.num_blocks
    }

    /// Return the total number of hash slots.
    pub fn total_slots(&self) -> u64 {
        self.total_slots
    }

    /// Assign a single address to a block.
    pub fn assign(&self, address: u64, hash_fn: &dyn HashFunction) -> BlockId {
        let slot = hash_fn.hash_to_range(address, self.total_slots);
        (slot / self.block_size) as BlockId
    }

    /// Assign a contiguous range of addresses [start, start+count) to blocks.
    pub fn assign_range(
        &self,
        start: u64,
        count: u64,
        hash_fn: &dyn HashFunction,
    ) -> Vec<BlockId> {
        (0..count)
            .map(|i| self.assign(start + i, hash_fn))
            .collect()
    }

    /// Assign a slice of addresses to blocks.
    pub fn assign_batch(
        &self,
        addresses: &[u64],
        hash_fn: &dyn HashFunction,
    ) -> Vec<BlockId> {
        addresses.iter().map(|&a| self.assign(a, hash_fn)).collect()
    }

    /// Count how many addresses land in each block.
    pub fn block_counts(
        &self,
        addresses: &[u64],
        hash_fn: &dyn HashFunction,
    ) -> Vec<u64> {
        let mut counts = vec![0u64; self.num_blocks as usize];
        for &addr in addresses {
            let block = self.assign(addr, hash_fn);
            counts[block] += 1;
        }
        counts
    }

    /// Compute per-block statistics: (min, max, mean, variance) of block loads.
    pub fn block_statistics(
        &self,
        addresses: &[u64],
        hash_fn: &dyn HashFunction,
    ) -> BlockStatistics {
        let counts = self.block_counts(addresses, hash_fn);
        BlockStatistics::from_counts(&counts)
    }

    /// Return a map from BlockId to the list of addresses assigned to it.
    pub fn block_membership(
        &self,
        addresses: &[u64],
        hash_fn: &dyn HashFunction,
    ) -> HashMap<BlockId, Vec<u64>> {
        let mut map: HashMap<BlockId, Vec<u64>> = HashMap::new();
        for &addr in addresses {
            let block = self.assign(addr, hash_fn);
            map.entry(block).or_default().push(addr);
        }
        map
    }

    /// Check if a block assignment respects cache-line alignment.
    /// Returns true if the slot index is a multiple of block_size.
    pub fn is_aligned(&self, address: u64, hash_fn: &dyn HashFunction) -> bool {
        let slot = hash_fn.hash_to_range(address, self.total_slots);
        slot % self.block_size == 0
    }

    /// Return the slot within its block for an address.
    pub fn slot_offset(&self, address: u64, hash_fn: &dyn HashFunction) -> u64 {
        let slot = hash_fn.hash_to_range(address, self.total_slots);
        slot % self.block_size
    }
}

/// Assigns addresses to blocks using contiguous ranges rather than hashing.
///
/// Address `a` is mapped to block `(a / range_size) % num_blocks`.
pub struct RangeBlockAssigner {
    /// Number of contiguous addresses per block.
    range_size: u64,
    /// Total number of blocks.
    num_blocks: u64,
}

impl RangeBlockAssigner {
    /// Create a new range-based assigner.
    pub fn new(num_blocks: u64, range_size: u64) -> Self {
        assert!(num_blocks > 0);
        assert!(range_size > 0);
        Self {
            range_size,
            num_blocks,
        }
    }

    /// Assign a single address.
    pub fn assign(&self, address: u64) -> BlockId {
        ((address / self.range_size) % self.num_blocks) as BlockId
    }

    /// Assign a batch of addresses.
    pub fn assign_batch(&self, addresses: &[u64]) -> Vec<BlockId> {
        addresses.iter().map(|&a| self.assign(a)).collect()
    }

    /// Return the range size.
    pub fn range_size(&self) -> u64 {
        self.range_size
    }

    /// Return the number of blocks.
    pub fn num_blocks(&self) -> u64 {
        self.num_blocks
    }
}

/// Cache hierarchy level descriptor.
#[derive(Clone, Debug)]
pub struct CacheLevel {
    /// Human-readable name (e.g. "L1", "L2", "L3").
    pub name: String,
    /// Size of this cache in elements.
    pub size_elements: u64,
    /// Line size in elements (same unit as size_elements).
    pub line_size: u64,
}

/// A block assigner that is aware of a multi-level cache hierarchy.
///
/// It assigns addresses to blocks sized to the smallest cache line and then
/// provides helpers to compute which higher-level cache set each block maps
/// to.
pub struct CacheAwareBlockAssigner {
    /// Levels ordered from smallest (L1) to largest (L3).
    levels: Vec<CacheLevel>,
    /// Inner block assigner based on the L1 line size.
    inner: BlockAssigner,
}

impl CacheAwareBlockAssigner {
    /// Create a cache-aware assigner.
    ///
    /// `num_blocks` is the total number of L1-sized blocks.
    /// `levels` describes the cache hierarchy from L1 (smallest) outward.
    pub fn new(num_blocks: u64, levels: Vec<CacheLevel>) -> Self {
        assert!(!levels.is_empty());
        let block_size = levels[0].line_size;
        let inner = BlockAssigner::new(num_blocks, block_size);
        Self { levels, inner }
    }

    /// Assign an address to an L1 block.
    pub fn assign(&self, address: u64, hash_fn: &dyn HashFunction) -> BlockId {
        self.inner.assign(address, hash_fn)
    }

    /// Return the cache level descriptors.
    pub fn levels(&self) -> &[CacheLevel] {
        &self.levels
    }

    /// Determine which set at `level_index` a given L1 block maps to.
    ///
    /// Computed as `block_id * L1_line_size / level.line_size % (level.size / level.line_size)`.
    pub fn cache_set(&self, block_id: BlockId, level_index: usize) -> u64 {
        let level = &self.levels[level_index];
        let num_sets = level.size_elements / level.line_size;
        if num_sets == 0 {
            return 0;
        }
        let l1_line = self.levels[0].line_size;
        let addr_proxy = block_id as u64 * l1_line;
        (addr_proxy / level.line_size) % num_sets
    }

    /// Return a reference to the inner `BlockAssigner`.
    pub fn inner(&self) -> &BlockAssigner {
        &self.inner
    }
}

/// Extended statistics including a load histogram and entropy.
#[derive(Clone, Debug)]
pub struct BlockStats {
    /// Per-load-value count: how many blocks have each load.
    pub histogram: HashMap<u64, usize>,
    /// Shannon entropy of the block load distribution (bits).
    pub entropy: f64,
    /// Max load factor: max_load / mean_load.
    pub max_load_factor: f64,
    /// Mean load.
    pub mean_load: f64,
    /// Number of blocks.
    pub num_blocks: usize,
}

/// Compute extended block statistics from a slice of block assignments.
pub fn block_statistics(assignments: &[BlockId]) -> BlockStats {
    let mut counts_map: HashMap<BlockId, u64> = HashMap::new();
    for &b in assignments {
        *counts_map.entry(b).or_insert(0) += 1;
    }

    let num_blocks = if counts_map.is_empty() {
        0
    } else {
        *counts_map.keys().max().unwrap() + 1
    };

    let mut counts = vec![0u64; num_blocks];
    for (&b, &c) in &counts_map {
        if b < num_blocks {
            counts[b] = c;
        }
    }

    let n = assignments.len() as f64;
    let mean = if num_blocks > 0 {
        n / num_blocks as f64
    } else {
        0.0
    };
    let max_load = counts.iter().copied().max().unwrap_or(0);
    let max_load_factor = if mean > 0.0 {
        max_load as f64 / mean
    } else {
        0.0
    };

    // Histogram
    let mut histogram: HashMap<u64, usize> = HashMap::new();
    for &c in &counts {
        *histogram.entry(c).or_insert(0) += 1;
    }

    // Entropy
    let entropy = if n > 0.0 {
        counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n;
                -p * p.ln()
            })
            .sum::<f64>()
            / 2.0f64.ln()
    } else {
        0.0
    };

    BlockStats {
        histogram,
        entropy,
        max_load_factor,
        mean_load: mean,
        num_blocks,
    }
}

/// Rearrange a slice of block assignments for improved spatial locality.
///
/// The optimisation sorts addresses that share the same block so they are
/// contiguous in the output, reducing potential cache-miss penalties during
/// sequential scans.
pub fn optimize_assignment(assignments: &mut [BlockId]) {
    assignments.sort();
}

/// Summary statistics for block load distribution.
#[derive(Clone, Debug)]
pub struct BlockStatistics {
    pub min_load: u64,
    pub max_load: u64,
    pub mean_load: f64,
    pub variance: f64,
    pub total_items: u64,
    pub num_blocks: usize,
    pub empty_blocks: usize,
}

impl BlockStatistics {
    /// Compute statistics from a vector of per-block counts.
    pub fn from_counts(counts: &[u64]) -> Self {
        if counts.is_empty() {
            return Self {
                min_load: 0,
                max_load: 0,
                mean_load: 0.0,
                variance: 0.0,
                total_items: 0,
                num_blocks: 0,
                empty_blocks: 0,
            };
        }

        let total: u64 = counts.iter().sum();
        let n = counts.len() as f64;
        let mean = total as f64 / n;
        let min = counts.iter().copied().min().unwrap_or(0);
        let max = counts.iter().copied().max().unwrap_or(0);
        let variance = counts
            .iter()
            .map(|&c| {
                let diff = c as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / n;
        let empty = counts.iter().filter(|&&c| c == 0).count();

        Self {
            min_load: min,
            max_load: max,
            mean_load: mean,
            variance,
            total_items: total,
            num_blocks: counts.len(),
            empty_blocks: empty,
        }
    }

    /// Standard deviation of block loads.
    pub fn std_dev(&self) -> f64 {
        self.variance.sqrt()
    }

    /// Coefficient of variation (std_dev / mean).
    pub fn coeff_of_variation(&self) -> f64 {
        if self.mean_load == 0.0 {
            return 0.0;
        }
        self.std_dev() / self.mean_load
    }

    /// Load factor: max_load / mean_load.
    pub fn load_factor(&self) -> f64 {
        if self.mean_load == 0.0 {
            return 0.0;
        }
        self.max_load as f64 / self.mean_load
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash_partition::identity::IdentityHash;
    use crate::hash_partition::murmur::MurmurHasher;
    use crate::hash_partition::siegel_hash::SiegelHash;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_basic_assignment() {
        let assigner = BlockAssigner::new(4, 4); // 4 blocks of size 4 = 16 slots
        let id_hash = IdentityHash::new();
        // Identity: address 0 -> slot 0 -> block 0
        assert_eq!(assigner.assign(0, &id_hash), 0);
        // address 5 -> slot 5 -> block 1
        assert_eq!(assigner.assign(5, &id_hash), 1);
        // address 15 -> slot 15 -> block 3
        assert_eq!(assigner.assign(15, &id_hash), 3);
    }

    #[test]
    fn test_assign_range() {
        let assigner = BlockAssigner::new(4, 4);
        let id_hash = IdentityHash::new();
        let blocks = assigner.assign_range(0, 16, &id_hash);
        assert_eq!(blocks.len(), 16);
        // Each group of 4 consecutive addresses should go to same block.
        for i in 0..4 {
            assert_eq!(blocks[i], 0);
        }
        for i in 4..8 {
            assert_eq!(blocks[i], 1);
        }
    }

    #[test]
    fn test_assign_batch() {
        let assigner = BlockAssigner::new(8, 2);
        let id_hash = IdentityHash::new();
        let addrs = vec![0, 1, 2, 3, 14, 15];
        let blocks = assigner.assign_batch(&addrs, &id_hash);
        assert_eq!(blocks[0], 0);
        assert_eq!(blocks[1], 0);
        assert_eq!(blocks[2], 1);
        assert_eq!(blocks[3], 1);
        assert_eq!(blocks[4], 7);
        assert_eq!(blocks[5], 7);
    }

    #[test]
    fn test_block_counts() {
        let assigner = BlockAssigner::new(4, 4);
        let id_hash = IdentityHash::new();
        let addrs: Vec<u64> = (0..16).collect();
        let counts = assigner.block_counts(&addrs, &id_hash);
        assert_eq!(counts, vec![4, 4, 4, 4]);
    }

    #[test]
    fn test_block_statistics() {
        let assigner = BlockAssigner::new(4, 4);
        let id_hash = IdentityHash::new();
        let addrs: Vec<u64> = (0..16).collect();
        let stats = assigner.block_statistics(&addrs, &id_hash);
        assert_eq!(stats.min_load, 4);
        assert_eq!(stats.max_load, 4);
        assert!((stats.mean_load - 4.0).abs() < 1e-9);
        assert!(stats.variance < 1e-9);
        assert_eq!(stats.empty_blocks, 0);
    }

    #[test]
    fn test_block_membership() {
        let assigner = BlockAssigner::new(4, 4);
        let id_hash = IdentityHash::new();
        let addrs: Vec<u64> = (0..8).collect();
        let membership = assigner.block_membership(&addrs, &id_hash);
        assert_eq!(membership[&0], vec![0, 1, 2, 3]);
        assert_eq!(membership[&1], vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_slot_offset() {
        let assigner = BlockAssigner::new(4, 4);
        let id_hash = IdentityHash::new();
        assert_eq!(assigner.slot_offset(0, &id_hash), 0);
        assert_eq!(assigner.slot_offset(1, &id_hash), 1);
        assert_eq!(assigner.slot_offset(4, &id_hash), 0);
        assert_eq!(assigner.slot_offset(7, &id_hash), 3);
    }

    #[test]
    fn test_is_aligned() {
        let assigner = BlockAssigner::new(4, 4);
        let id_hash = IdentityHash::new();
        assert!(assigner.is_aligned(0, &id_hash));
        assert!(!assigner.is_aligned(1, &id_hash));
        assert!(assigner.is_aligned(4, &id_hash));
    }

    #[test]
    fn test_murmur_assignment_in_range() {
        let assigner = BlockAssigner::new(32, 8);
        let hasher = MurmurHasher::new(0);
        for addr in 0..1000u64 {
            let block = assigner.assign(addr, &hasher);
            assert!(block < 32);
        }
    }

    #[test]
    fn test_siegel_assignment_balance() {
        let mut rng = StdRng::seed_from_u64(42);
        let assigner = BlockAssigner::new(16, 4);
        let h = SiegelHash::new(4, &mut rng);
        let addrs: Vec<u64> = (0..1000).collect();
        let stats = assigner.block_statistics(&addrs, &h);
        // With a good hash, load factor should be reasonable.
        assert!(
            stats.load_factor() < 5.0,
            "load factor {} too high",
            stats.load_factor()
        );
    }

    #[test]
    fn test_statistics_helpers() {
        let stats = BlockStatistics::from_counts(&[10, 20, 30, 40]);
        assert_eq!(stats.min_load, 10);
        assert_eq!(stats.max_load, 40);
        assert!((stats.mean_load - 25.0).abs() < 1e-9);
        assert!(stats.std_dev() > 0.0);
        assert!(stats.coeff_of_variation() > 0.0);
        assert!((stats.load_factor() - 40.0 / 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_counts() {
        let stats = BlockStatistics::from_counts(&[]);
        assert_eq!(stats.num_blocks, 0);
        assert_eq!(stats.total_items, 0);
    }

    #[test]
    fn test_wrap_around() {
        let assigner = BlockAssigner::new(4, 4); // 16 total slots
        let id_hash = IdentityHash::new();
        // address 20 % 16 = 4 -> block 1
        assert_eq!(assigner.assign(20, &id_hash), 1);
    }

    // ── new tests ──────────────────────────────────────────────────────

    #[test]
    fn test_range_block_assigner_basic() {
        let rba = RangeBlockAssigner::new(4, 8);
        assert_eq!(rba.assign(0), 0);
        assert_eq!(rba.assign(7), 0);
        assert_eq!(rba.assign(8), 1);
        assert_eq!(rba.assign(16), 2);
        assert_eq!(rba.assign(24), 3);
        assert_eq!(rba.assign(32), 0); // wraps around
    }

    #[test]
    fn test_range_block_assigner_batch() {
        let rba = RangeBlockAssigner::new(4, 4);
        let addrs: Vec<u64> = (0..16).collect();
        let blocks = rba.assign_batch(&addrs);
        assert_eq!(blocks, vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
    }

    #[test]
    fn test_cache_aware_assigner() {
        let levels = vec![
            CacheLevel {
                name: "L1".to_string(),
                size_elements: 256,
                line_size: 4,
            },
            CacheLevel {
                name: "L2".to_string(),
                size_elements: 1024,
                line_size: 8,
            },
        ];
        let ca = CacheAwareBlockAssigner::new(64, levels);
        let id_hash = IdentityHash::new();
        let block = ca.assign(10, &id_hash);
        assert!(block < 64);
        // Check cache set computation doesn't panic.
        let _set = ca.cache_set(block, 0);
        let _set2 = ca.cache_set(block, 1);
    }

    #[test]
    fn test_block_statistics_fn() {
        let assignments = vec![0, 0, 1, 1, 1, 2, 3, 3];
        let stats = block_statistics(&assignments);
        assert_eq!(stats.num_blocks, 4);
        assert!((stats.mean_load - 2.0).abs() < 1e-9);
        assert!((stats.max_load_factor - 1.5).abs() < 1e-9); // max=3, mean=2
        assert!(stats.entropy > 0.0);
        assert_eq!(stats.histogram[&2], 2); // blocks 0 and 3 have load 2
        assert_eq!(stats.histogram[&3], 1); // block 1 has load 3
        assert_eq!(stats.histogram[&1], 1); // block 2 has load 1
    }

    #[test]
    fn test_optimize_assignment() {
        let mut assignments = vec![3, 1, 0, 2, 1, 3, 0, 2];
        optimize_assignment(&mut assignments);
        assert_eq!(assignments, vec![0, 0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn test_block_statistics_empty() {
        let stats = block_statistics(&[]);
        assert_eq!(stats.num_blocks, 0);
        assert!((stats.mean_load - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_cache_aware_levels() {
        let levels = vec![
            CacheLevel {
                name: "L1".to_string(),
                size_elements: 128,
                line_size: 4,
            },
        ];
        let ca = CacheAwareBlockAssigner::new(32, levels);
        assert_eq!(ca.levels().len(), 1);
        assert_eq!(ca.levels()[0].name, "L1");
    }
}
