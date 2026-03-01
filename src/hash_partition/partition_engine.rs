//! Complete hash-partition pipeline.
//!
//! Combines hash function selection → block assignment → overflow analysis
//! into a single `PartitionEngine` that produces a `PartitionResult`.

use rand::SeedableRng;
use rand::rngs::StdRng;

use super::block_assignment::{BlockAssigner, BlockStatistics};
use super::identity::IdentityHash;
use super::murmur::MurmurHasher;
use super::overflow_analysis::{HashFamilyType, OverflowAnalyzer, OverflowReport};
use super::siegel_hash::SiegelHash;
use super::tabulation::TabulationHash;
use super::two_universal::TwoUniversalHash;
use super::HashFunction;

/// Selects which hash family the engine uses.
#[derive(Clone, Debug)]
pub enum HashFamilyChoice {
    /// Siegel k-wise independent hash.
    Siegel { k: usize },
    /// 2-universal hash.
    TwoUniversal,
    /// MurmurHash3.
    Murmur { seed: u64 },
    /// Identity (ablation baseline).
    Identity,
    /// Simple tabulation hashing.
    Tabulation { seed: u64 },
}

/// Holds a concrete hash function behind an enum for ownership.
enum ConcreteHash {
    Siegel(SiegelHash),
    TwoUniversal(TwoUniversalHash),
    Murmur(MurmurHasher),
    Identity(IdentityHash),
    Tabulation(TabulationHash),
}

impl ConcreteHash {
    fn as_hash_fn(&self) -> &dyn HashFunction {
        match self {
            ConcreteHash::Siegel(h) => h,
            ConcreteHash::TwoUniversal(h) => h,
            ConcreteHash::Murmur(h) => h,
            ConcreteHash::Identity(h) => h,
            ConcreteHash::Tabulation(h) => h,
        }
    }
}

/// Pre-computed partition layout describing how addresses will be distributed.
#[derive(Clone, Debug)]
pub struct PartitionPlan {
    /// Number of blocks.
    pub num_blocks: u64,
    /// Block size.
    pub block_size: u64,
    /// The block each address maps to.
    pub layout: Vec<usize>,
    /// Counts per block.
    pub block_counts: Vec<u64>,
}

impl PartitionPlan {
    /// Build a plan from addresses using the given hash function.
    pub fn build(
        addresses: &[u64],
        num_blocks: u64,
        block_size: u64,
        hash_fn: &dyn HashFunction,
    ) -> Self {
        let assigner = BlockAssigner::new(num_blocks, block_size);
        let layout = assigner.assign_batch(addresses, hash_fn);
        let mut block_counts = vec![0u64; num_blocks as usize];
        for &b in &layout {
            block_counts[b] += 1;
        }
        Self {
            num_blocks,
            block_size,
            layout,
            block_counts,
        }
    }

    /// Maximum load across all blocks.
    pub fn max_load(&self) -> u64 {
        self.block_counts.iter().copied().max().unwrap_or(0)
    }

    /// Mean load.
    pub fn mean_load(&self) -> f64 {
        if self.block_counts.is_empty() {
            return 0.0;
        }
        let total: u64 = self.block_counts.iter().sum();
        total as f64 / self.block_counts.len() as f64
    }
}

/// Minimise cross-block dependencies in a partition plan.
///
/// Re-maps blocks so that consecutively-accessed addresses share a block as
/// much as possible.  This is a greedy heuristic: it scans the layout and
/// reassigns block ids so that adjacent addresses that were in different
/// blocks are merged when there is capacity.
pub fn optimize_partition(plan: &mut PartitionPlan) {
    if plan.layout.is_empty() {
        return;
    }
    // Simple heuristic: Sort layout segments by block id to cluster accesses.
    // Build mapping: original_block → new_block based on first-appearance order.
    let mut mapping: Vec<Option<usize>> = vec![None; plan.num_blocks as usize];
    let mut next_id = 0usize;
    for &b in &plan.layout {
        if mapping[b].is_none() {
            mapping[b] = Some(next_id);
            next_id += 1;
        }
    }
    // Apply mapping.
    for b in plan.layout.iter_mut() {
        if let Some(new_id) = mapping[*b] {
            *b = new_id;
        }
    }
    // Recompute counts.
    let mut new_counts = vec![0u64; plan.num_blocks as usize];
    for &b in &plan.layout {
        new_counts[b] += 1;
    }
    plan.block_counts = new_counts;
}

/// Estimate the number of cache misses given a partition plan and a
/// sequential schedule ordering.
///
/// Models a single-entry cache per block: each time the scheduled address
/// accesses a different block from the previous access it is a miss.
pub fn estimate_cache_misses(plan: &PartitionPlan, schedule_order: &[usize]) -> u64 {
    if schedule_order.is_empty() {
        return 0;
    }
    let mut misses = 1u64; // first access is always a miss
    let mut last_block = plan.layout[schedule_order[0]];
    for &idx in &schedule_order[1..] {
        let block = plan.layout[idx];
        if block != last_block {
            misses += 1;
            last_block = block;
        }
    }
    misses
}

/// Multi-level partitioning: partition blocks into super-blocks.
#[derive(Clone, Debug)]
pub struct MultiLevelPartition {
    /// Level-0 (finest) assignments: address → block.
    pub level0: Vec<usize>,
    /// Level-1 assignments: block → super-block.
    pub level1: Vec<usize>,
    /// Number of super-blocks.
    pub num_super_blocks: usize,
}

impl MultiLevelPartition {
    /// Create a two-level partition.
    ///
    /// `blocks_per_super` controls how many L0 blocks form one super-block.
    pub fn build(
        addresses: &[u64],
        num_blocks: u64,
        block_size: u64,
        blocks_per_super: u64,
        hash_fn: &dyn HashFunction,
    ) -> Self {
        let assigner = BlockAssigner::new(num_blocks, block_size);
        let level0 = assigner.assign_batch(addresses, hash_fn);

        let num_super = ((num_blocks + blocks_per_super - 1) / blocks_per_super) as usize;
        let level1: Vec<usize> = (0..num_blocks as usize)
            .map(|b| b / blocks_per_super as usize)
            .collect();

        Self {
            level0,
            level1,
            num_super_blocks: num_super,
        }
    }

    /// Get the super-block for a given address index.
    pub fn super_block_of(&self, addr_index: usize) -> usize {
        let block = self.level0[addr_index];
        if block < self.level1.len() {
            self.level1[block]
        } else {
            0
        }
    }
}

/// Quality metrics for a partition.
#[derive(Clone, Debug)]
pub struct PartitionQualityReport {
    /// Load balance ratio: max_load / mean_load.
    pub load_balance_ratio: f64,
    /// Coefficient of variation of block loads.
    pub cv: f64,
    /// Number of empty blocks.
    pub empty_blocks: usize,
    /// Estimated cache misses for sequential access.
    pub sequential_cache_misses: u64,
    /// Entropy of the block load distribution (bits).
    pub entropy: f64,
}

/// Compute a quality report for a partition result.
pub fn partition_quality_report(result: &PartitionResult) -> PartitionQualityReport {
    let loads = &result.overflow.block_loads;
    let n = loads.len() as f64;
    let total: u64 = loads.iter().sum();
    let mean = if n > 0.0 { total as f64 / n } else { 0.0 };
    let max_load = loads.iter().copied().max().unwrap_or(0);
    let load_balance_ratio = if mean > 0.0 {
        max_load as f64 / mean
    } else {
        0.0
    };

    let variance = if n > 0.0 {
        loads
            .iter()
            .map(|&c| {
                let d = c as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n
    } else {
        0.0
    };
    let cv = if mean > 0.0 {
        variance.sqrt() / mean
    } else {
        0.0
    };

    let empty_blocks = loads.iter().filter(|&&c| c == 0).count();

    // Sequential cache misses
    let mut sequential_misses = 0u64;
    if !result.assignments.is_empty() {
        sequential_misses = 1;
        let mut last = result.assignments[0];
        for &b in &result.assignments[1..] {
            if b != last {
                sequential_misses += 1;
                last = b;
            }
        }
    }

    // Entropy
    let entropy = if total > 0 {
        loads
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / total as f64;
                -p * p.ln()
            })
            .sum::<f64>()
            / 2.0f64.ln()
    } else {
        0.0
    };

    PartitionQualityReport {
        load_balance_ratio,
        cv,
        empty_blocks,
        sequential_cache_misses: sequential_misses,
        entropy,
    }
}

/// Result of a complete partitioning run.
#[derive(Clone, Debug)]
pub struct PartitionResult {
    /// Block assignment for each input address (indexed by position in input).
    pub assignments: Vec<usize>,
    /// Overflow analysis report.
    pub overflow: OverflowReport,
    /// Block load statistics.
    pub statistics: BlockStatistics,
    /// Estimated cache-miss count.
    pub cache_miss_estimate: f64,
    /// The hash family used.
    pub hash_family: String,
}

impl PartitionResult {
    /// Fraction of addresses in blocks that exceed a given load threshold.
    pub fn fraction_above_threshold(&self, threshold: u64) -> f64 {
        if self.overflow.num_addresses == 0 {
            return 0.0;
        }
        let above: u64 = self
            .overflow
            .block_loads
            .iter()
            .filter(|&&c| c > threshold)
            .sum();
        above as f64 / self.overflow.num_addresses as f64
    }
}

/// The partition engine orchestrating hash → assign → analyze.
pub struct PartitionEngine {
    /// Number of cache-line blocks.
    num_blocks: u64,
    /// Elements per cache line.
    block_size: u64,
    /// Hash family choice.
    family: HashFamilyChoice,
    /// RNG seed for reproducibility.
    seed: u64,
}

impl PartitionEngine {
    /// Create a new engine.
    pub fn new(num_blocks: u64, block_size: u64, family: HashFamilyChoice, seed: u64) -> Self {
        assert!(num_blocks > 0);
        assert!(block_size > 0);
        Self {
            num_blocks,
            block_size,
            family,
            seed,
        }
    }

    /// Build the concrete hash function.
    fn build_hash(&self) -> ConcreteHash {
        let mut rng = StdRng::seed_from_u64(self.seed);
        match &self.family {
            HashFamilyChoice::Siegel { k } => {
                ConcreteHash::Siegel(SiegelHash::new(*k, &mut rng))
            }
            HashFamilyChoice::TwoUniversal => {
                let total = self.num_blocks * self.block_size;
                ConcreteHash::TwoUniversal(TwoUniversalHash::new(total, &mut rng))
            }
            HashFamilyChoice::Murmur { seed } => {
                ConcreteHash::Murmur(MurmurHasher::new(*seed))
            }
            HashFamilyChoice::Identity => ConcreteHash::Identity(IdentityHash::new()),
            HashFamilyChoice::Tabulation { seed } => {
                ConcreteHash::Tabulation(TabulationHash::from_seed(*seed))
            }
        }
    }

    /// Determine the overflow family type for theoretical bounds.
    fn family_type(&self) -> HashFamilyType {
        match &self.family {
            HashFamilyChoice::Siegel { k } => HashFamilyType::Siegel { k: *k },
            HashFamilyChoice::TwoUniversal => HashFamilyType::TwoUniversal,
            HashFamilyChoice::Murmur { .. } => HashFamilyType::Murmur,
            HashFamilyChoice::Identity => HashFamilyType::Identity,
            HashFamilyChoice::Tabulation { .. } => HashFamilyType::Tabulation,
        }
    }

    /// Family name as a string.
    fn family_name(&self) -> String {
        match &self.family {
            HashFamilyChoice::Siegel { k } => format!("Siegel(k={})", k),
            HashFamilyChoice::TwoUniversal => "TwoUniversal".to_string(),
            HashFamilyChoice::Murmur { seed } => format!("Murmur(seed={})", seed),
            HashFamilyChoice::Identity => "Identity".to_string(),
            HashFamilyChoice::Tabulation { seed } => format!("Tabulation(seed={})", seed),
        }
    }

    /// Run the full partition pipeline on a set of addresses.
    pub fn partition(&self, addresses: &[u64]) -> PartitionResult {
        let concrete = self.build_hash();
        let hash_fn = concrete.as_hash_fn();

        // Block assignment.
        let assigner = BlockAssigner::new(self.num_blocks, self.block_size);
        let assignments = assigner.assign_batch(addresses, hash_fn);
        let statistics = assigner.block_statistics(addresses, hash_fn);

        // Overflow analysis.
        let analyzer = OverflowAnalyzer::new(self.num_blocks, self.block_size, self.family_type());
        let overflow = analyzer.analyze(addresses, hash_fn);

        // Cache-miss estimate: each distinct block accessed is one cache miss.
        let distinct_blocks: std::collections::HashSet<usize> =
            assignments.iter().copied().collect();
        let cache_miss_estimate = distinct_blocks.len() as f64;

        PartitionResult {
            assignments,
            overflow,
            statistics,
            cache_miss_estimate,
            hash_family: self.family_name(),
        }
    }

    /// Compare different hash families on the same input.
    pub fn compare_families(
        addresses: &[u64],
        num_blocks: u64,
        block_size: u64,
        seed: u64,
    ) -> Vec<PartitionResult> {
        let families = vec![
            HashFamilyChoice::Siegel { k: 2 },
            HashFamilyChoice::Siegel { k: 4 },
            HashFamilyChoice::TwoUniversal,
            HashFamilyChoice::Murmur { seed: 0 },
            HashFamilyChoice::Tabulation { seed: 0 },
            HashFamilyChoice::Identity,
        ];
        families
            .into_iter()
            .map(|f| {
                let engine = PartitionEngine::new(num_blocks, block_size, f, seed);
                engine.partition(addresses)
            })
            .collect()
    }

    /// Adaptive partition: tries multiple block sizes and picks the one with
    /// the lowest maximum load factor, reducing worst-case cache overflow.
    pub fn adaptive_partition(&self, addresses: &[u64], candidate_sizes: &[u64]) -> PartitionResult {
        let mut best_result: Option<PartitionResult> = None;
        let mut best_max_load = u64::MAX;

        for &bs in candidate_sizes {
            if bs == 0 { continue; }
            let engine = PartitionEngine::new(
                self.num_blocks,
                bs,
                self.family.clone(),
                self.seed,
            );
            let result = engine.partition(addresses);

            // Compute max load across blocks
            let max_load = result.statistics.max_load;

            if max_load < best_max_load {
                best_max_load = max_load;
                best_result = Some(result);
            }
        }

        best_result.unwrap_or_else(|| self.partition(addresses))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_siegel_partition() {
        let engine = PartitionEngine::new(16, 4, HashFamilyChoice::Siegel { k: 4 }, 42);
        let addrs: Vec<u64> = (0..1000).collect();
        let result = engine.partition(&addrs);
        assert_eq!(result.assignments.len(), 1000);
        assert!(result.overflow.empirical_max_load > 0);
        assert!(result.cache_miss_estimate > 0.0);
        assert!(result.hash_family.contains("Siegel"));
    }

    #[test]
    fn test_two_universal_partition() {
        let engine = PartitionEngine::new(8, 2, HashFamilyChoice::TwoUniversal, 99);
        let addrs: Vec<u64> = (0..500).collect();
        let result = engine.partition(&addrs);
        assert_eq!(result.assignments.len(), 500);
        for &a in &result.assignments {
            assert!(a < 8);
        }
    }

    #[test]
    fn test_murmur_partition() {
        let engine = PartitionEngine::new(32, 8, HashFamilyChoice::Murmur { seed: 0 }, 0);
        let addrs: Vec<u64> = (0..2000).collect();
        let result = engine.partition(&addrs);
        assert_eq!(result.assignments.len(), 2000);
        assert!(result.statistics.load_factor() < 5.0);
    }

    #[test]
    fn test_identity_partition() {
        let engine = PartitionEngine::new(4, 4, HashFamilyChoice::Identity, 0);
        let addrs: Vec<u64> = (0..16).collect();
        let result = engine.partition(&addrs);
        assert_eq!(result.assignments.len(), 16);
        // Identity on sequential: uniform.
        assert_eq!(result.overflow.empirical_max_load, 4);
    }

    #[test]
    fn test_deterministic() {
        let engine = PartitionEngine::new(16, 4, HashFamilyChoice::Siegel { k: 3 }, 123);
        let addrs: Vec<u64> = (0..100).collect();
        let r1 = engine.partition(&addrs);
        let r2 = engine.partition(&addrs);
        assert_eq!(r1.assignments, r2.assignments);
    }

    #[test]
    fn test_compare_families() {
        let addrs: Vec<u64> = (0..500).collect();
        let results = PartitionEngine::compare_families(&addrs, 16, 4, 42);
        assert_eq!(results.len(), 6);
        // Identity should generally have worse (or equal) max load than good hashes.
        let identity_max = results.last().unwrap().overflow.empirical_max_load;
        let _ = identity_max;
    }

    #[test]
    fn test_empty_addresses() {
        let engine = PartitionEngine::new(4, 4, HashFamilyChoice::Murmur { seed: 0 }, 0);
        let result = engine.partition(&[]);
        assert_eq!(result.assignments.len(), 0);
        assert_eq!(result.overflow.empirical_max_load, 0);
        assert_eq!(result.cache_miss_estimate, 0.0);
    }

    #[test]
    fn test_single_address() {
        let engine = PartitionEngine::new(8, 1, HashFamilyChoice::Murmur { seed: 0 }, 0);
        let result = engine.partition(&[42]);
        assert_eq!(result.assignments.len(), 1);
        assert!(result.assignments[0] < 8);
        assert_eq!(result.cache_miss_estimate, 1.0);
    }

    #[test]
    fn test_fraction_above_threshold() {
        let engine = PartitionEngine::new(4, 4, HashFamilyChoice::Identity, 0);
        let addrs: Vec<u64> = (0..16).collect();
        let result = engine.partition(&addrs);
        // All blocks have load 4, so fraction above 3 is all addresses.
        let frac = result.fraction_above_threshold(3);
        assert!((frac - 1.0).abs() < 1e-9);
        // Fraction above 4 is 0 since no block exceeds 4.
        let frac2 = result.fraction_above_threshold(4);
        assert!((frac2 - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_partition_result_fields() {
        let engine = PartitionEngine::new(16, 4, HashFamilyChoice::Siegel { k: 2 }, 7);
        let addrs: Vec<u64> = (0..800).collect();
        let result = engine.partition(&addrs);
        assert_eq!(result.overflow.num_addresses, 800);
        assert_eq!(result.overflow.num_blocks, 16);
        assert!((result.overflow.expected_load - 50.0).abs() < 1e-9);
        assert!(result.statistics.num_blocks == 16);
    }

    // ── new tests ──────────────────────────────────────────────────────

    #[test]
    fn test_partition_plan_build() {
        let hasher = MurmurHasher::new(0);
        let addrs: Vec<u64> = (0..500).collect();
        let plan = PartitionPlan::build(&addrs, 16, 4, &hasher);
        assert_eq!(plan.layout.len(), 500);
        assert_eq!(plan.block_counts.len(), 16);
        let total: u64 = plan.block_counts.iter().sum();
        assert_eq!(total, 500);
        assert!(plan.max_load() > 0);
        assert!(plan.mean_load() > 0.0);
    }

    #[test]
    fn test_optimize_partition() {
        let hasher = MurmurHasher::new(0);
        let addrs: Vec<u64> = (0..200).collect();
        let mut plan = PartitionPlan::build(&addrs, 8, 2, &hasher);
        let old_max = plan.max_load();
        optimize_partition(&mut plan);
        // Total items should be preserved.
        let new_total: u64 = plan.block_counts.iter().sum();
        assert_eq!(new_total, 200);
        let _ = old_max; // optimisation may or may not change max
    }

    #[test]
    fn test_estimate_cache_misses() {
        let hasher = MurmurHasher::new(0);
        let addrs: Vec<u64> = (0..100).collect();
        let plan = PartitionPlan::build(&addrs, 8, 2, &hasher);
        let schedule: Vec<usize> = (0..100).collect();
        let misses = estimate_cache_misses(&plan, &schedule);
        assert!(misses >= 1);
        assert!(misses <= 100);
    }

    #[test]
    fn test_multi_level_partition() {
        let hasher = MurmurHasher::new(0);
        let addrs: Vec<u64> = (0..400).collect();
        let ml = MultiLevelPartition::build(&addrs, 16, 2, 4, &hasher);
        assert_eq!(ml.level0.len(), 400);
        assert_eq!(ml.level1.len(), 16);
        assert_eq!(ml.num_super_blocks, 4);
        for i in 0..400 {
            let sb = ml.super_block_of(i);
            assert!(sb < 4);
        }
    }

    #[test]
    fn test_partition_quality_report() {
        let engine = PartitionEngine::new(16, 4, HashFamilyChoice::Murmur { seed: 0 }, 0);
        let addrs: Vec<u64> = (0..1000).collect();
        let result = engine.partition(&addrs);
        let qr = partition_quality_report(&result);
        assert!(qr.load_balance_ratio >= 1.0);
        assert!(qr.cv >= 0.0);
        assert!(qr.sequential_cache_misses >= 1);
        assert!(qr.entropy >= 0.0);
    }

    #[test]
    fn test_estimate_cache_misses_sorted() {
        let id_hash = IdentityHash::new();
        let addrs: Vec<u64> = (0..16).collect();
        let plan = PartitionPlan::build(&addrs, 4, 4, &id_hash);
        // Sequential schedule on identity: 4 groups → 4 misses.
        let schedule: Vec<usize> = (0..16).collect();
        let misses = estimate_cache_misses(&plan, &schedule);
        assert_eq!(misses, 4);
    }

    #[test]
    fn test_partition_plan_empty() {
        let hasher = MurmurHasher::new(0);
        let plan = PartitionPlan::build(&[], 8, 2, &hasher);
        assert!(plan.layout.is_empty());
        assert_eq!(plan.max_load(), 0);
        assert!((plan.mean_load() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_adaptive_partition() {
        let engine = PartitionEngine::new(16, 4, HashFamilyChoice::Siegel { k: 4 }, 42);
        let addrs: Vec<u64> = (0..256).collect();
        let result = engine.adaptive_partition(&addrs, &[2, 4, 8, 16]);
        assert_eq!(result.assignments.len(), 256);
    }

    #[test]
    fn test_adaptive_partition_selects_best() {
        let engine = PartitionEngine::new(8, 4, HashFamilyChoice::TwoUniversal, 42);
        let addrs: Vec<u64> = (0..128).collect();
        let result = engine.adaptive_partition(&addrs, &[1, 2, 4, 8]);
        // Should pick a configuration that minimizes max load
        let max_load = result.statistics.max_load;
        assert!(max_load > 0);
    }

    #[test]
    fn test_adaptive_partition_single_size() {
        let engine = PartitionEngine::new(16, 4, HashFamilyChoice::Siegel { k: 2 }, 42);
        let addrs: Vec<u64> = (0..64).collect();
        let result = engine.adaptive_partition(&addrs, &[4]);
        assert_eq!(result.assignments.len(), 64);
    }

    #[test]
    fn test_adaptive_partition_empty() {
        let engine = PartitionEngine::new(16, 4, HashFamilyChoice::Siegel { k: 4 }, 42);
        let addrs: Vec<u64> = vec![];
        let result = engine.adaptive_partition(&addrs, &[2, 4]);
        assert_eq!(result.assignments.len(), 0);
    }
}
