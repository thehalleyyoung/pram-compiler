//! Software cache-line simulator for measuring memory access patterns.

use std::collections::VecDeque;
use std::fmt;

/// Result of a single cache access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheResult {
    Hit,
    Miss,
}

/// Accumulated cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub total_accesses: u64,
    pub hits: u64,
    pub misses: u64,
}

impl CacheStats {
    pub fn miss_rate(&self) -> f64 {
        if self.total_accesses == 0 {
            0.0
        } else {
            self.misses as f64 / self.total_accesses as f64
        }
    }

    pub fn hit_rate(&self) -> f64 {
        if self.total_accesses == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_accesses as f64
        }
    }
}

impl fmt::Display for CacheStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "accesses={}, hits={}, misses={}, miss_rate={:.4}",
            self.total_accesses,
            self.hits,
            self.misses,
            self.miss_rate()
        )
    }
}

// ---------------------------------------------------------------------------
// Fully-associative LRU cache
// ---------------------------------------------------------------------------

/// A single cache line in the fully-associative cache.
#[derive(Debug, Clone)]
struct CacheLine {
    tag: u64,
    valid: bool,
}

/// Fully-associative LRU cache simulator.
#[derive(Debug)]
pub struct CacheSimulator {
    cache_line_size: u64,
    num_cache_lines: usize,
    /// LRU order: front = most recently used, back = least recently used
    lines: VecDeque<CacheLine>,
    stats: CacheStats,
}

impl CacheSimulator {
    /// Create a new fully-associative LRU cache simulator.
    ///
    /// * `cache_line_size` – size of each cache line in bytes (must be power of 2)
    /// * `num_cache_lines` – total number of cache lines
    pub fn new(cache_line_size: u64, num_cache_lines: usize) -> Self {
        assert!(cache_line_size.is_power_of_two(), "cache_line_size must be a power of 2");
        assert!(num_cache_lines > 0, "num_cache_lines must be > 0");

        let mut lines = VecDeque::with_capacity(num_cache_lines);
        for _ in 0..num_cache_lines {
            lines.push_back(CacheLine { tag: 0, valid: false });
        }

        Self {
            cache_line_size,
            num_cache_lines,
            lines,
            stats: CacheStats::default(),
        }
    }

    /// Access an address and return whether it was a hit or miss.
    pub fn access(&mut self, address: u64) -> CacheResult {
        let tag = address / self.cache_line_size;
        self.stats.total_accesses += 1;

        // Search for the tag in the cache (linear scan of deque)
        if let Some(pos) = self.lines.iter().position(|l| l.valid && l.tag == tag) {
            // Hit – move to front (MRU position)
            let line = self.lines.remove(pos).unwrap();
            self.lines.push_front(line);
            self.stats.hits += 1;
            CacheResult::Hit
        } else {
            // Miss – evict LRU (back) and insert at front
            self.lines.pop_back();
            self.lines.push_front(CacheLine { tag, valid: true });
            self.stats.misses += 1;
            CacheResult::Miss
        }
    }

    /// Simulate a sequence of memory accesses.
    pub fn access_sequence(&mut self, addresses: &[u64]) -> Vec<CacheResult> {
        addresses.iter().map(|&addr| self.access(addr)).collect()
    }

    /// Return a snapshot of accumulated statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Reset the cache (invalidate all lines, zero stats).
    pub fn reset(&mut self) {
        for line in &mut self.lines {
            line.valid = false;
            line.tag = 0;
        }
        self.stats = CacheStats::default();
    }

    pub fn cache_line_size(&self) -> u64 {
        self.cache_line_size
    }

    pub fn num_cache_lines(&self) -> usize {
        self.num_cache_lines
    }
}

// ---------------------------------------------------------------------------
// Direct-mapped cache
// ---------------------------------------------------------------------------

/// Direct-mapped cache simulator.
#[derive(Debug)]
pub struct DirectMappedCache {
    cache_line_size: u64,
    num_lines: usize,
    /// Each slot: (valid, tag)
    lines: Vec<(bool, u64)>,
    stats: CacheStats,
}

impl DirectMappedCache {
    pub fn new(cache_line_size: u64, num_lines: usize) -> Self {
        assert!(cache_line_size.is_power_of_two());
        assert!(num_lines.is_power_of_two(), "num_lines must be a power of 2 for direct mapping");
        Self {
            cache_line_size,
            num_lines,
            lines: vec![(false, 0); num_lines],
            stats: CacheStats::default(),
        }
    }

    pub fn access(&mut self, address: u64) -> CacheResult {
        let block_addr = address / self.cache_line_size;
        let index = (block_addr as usize) % self.num_lines;
        let tag = block_addr / (self.num_lines as u64);
        self.stats.total_accesses += 1;

        let (valid, stored_tag) = &self.lines[index];
        if *valid && *stored_tag == tag {
            self.stats.hits += 1;
            CacheResult::Hit
        } else {
            self.lines[index] = (true, tag);
            self.stats.misses += 1;
            CacheResult::Miss
        }
    }

    pub fn access_sequence(&mut self, addresses: &[u64]) -> Vec<CacheResult> {
        addresses.iter().map(|&addr| self.access(addr)).collect()
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    pub fn reset(&mut self) {
        for slot in &mut self.lines {
            *slot = (false, 0);
        }
        self.stats = CacheStats::default();
    }
}

// ---------------------------------------------------------------------------
// Set-associative cache
// ---------------------------------------------------------------------------

/// An individual set in the set-associative cache, using LRU eviction.
#[derive(Debug, Clone)]
struct CacheSet {
    /// LRU order: front = MRU, back = LRU
    ways: VecDeque<CacheLine>,
    associativity: usize,
}

impl CacheSet {
    fn new(associativity: usize) -> Self {
        let mut ways = VecDeque::with_capacity(associativity);
        for _ in 0..associativity {
            ways.push_back(CacheLine { tag: 0, valid: false });
        }
        Self { ways, associativity }
    }

    fn access(&mut self, tag: u64) -> CacheResult {
        if let Some(pos) = self.ways.iter().position(|l| l.valid && l.tag == tag) {
            let line = self.ways.remove(pos).unwrap();
            self.ways.push_front(line);
            CacheResult::Hit
        } else {
            self.ways.pop_back();
            self.ways.push_front(CacheLine { tag, valid: true });
            CacheResult::Miss
        }
    }

    fn reset(&mut self) {
        for way in &mut self.ways {
            way.valid = false;
            way.tag = 0;
        }
    }
}

/// Set-associative LRU cache simulator.
#[derive(Debug)]
pub struct SetAssociativeCache {
    cache_line_size: u64,
    num_sets: usize,
    associativity: usize,
    sets: Vec<CacheSet>,
    stats: CacheStats,
}

impl SetAssociativeCache {
    /// Create a new set-associative cache.
    ///
    /// * `cache_line_size` – bytes per cache line (power of 2)
    /// * `num_sets` – number of sets (power of 2)
    /// * `associativity` – ways per set
    pub fn new(cache_line_size: u64, num_sets: usize, associativity: usize) -> Self {
        assert!(cache_line_size.is_power_of_two());
        assert!(num_sets.is_power_of_two(), "num_sets must be power of 2");
        assert!(associativity > 0);

        let sets = (0..num_sets).map(|_| CacheSet::new(associativity)).collect();
        Self {
            cache_line_size,
            num_sets,
            associativity,
            sets,
            stats: CacheStats::default(),
        }
    }

    pub fn access(&mut self, address: u64) -> CacheResult {
        let block_addr = address / self.cache_line_size;
        let set_index = (block_addr as usize) % self.num_sets;
        let tag = block_addr / (self.num_sets as u64);
        self.stats.total_accesses += 1;

        let result = self.sets[set_index].access(tag);
        match result {
            CacheResult::Hit => self.stats.hits += 1,
            CacheResult::Miss => self.stats.misses += 1,
        }
        result
    }

    pub fn access_sequence(&mut self, addresses: &[u64]) -> Vec<CacheResult> {
        addresses.iter().map(|&addr| self.access(addr)).collect()
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    pub fn reset(&mut self) {
        for set in &mut self.sets {
            set.reset();
        }
        self.stats = CacheStats::default();
    }

    pub fn total_lines(&self) -> usize {
        self.num_sets * self.associativity
    }
}

// ---------------------------------------------------------------------------
// Convenience: simulate a full address trace and return miss count
// ---------------------------------------------------------------------------

/// Simulate a sequence of byte-addressed memory accesses on a fully-associative LRU cache
/// and return the total number of cache misses.
pub fn count_cache_misses(
    addresses: &[u64],
    cache_line_size: u64,
    num_cache_lines: usize,
) -> u64 {
    let mut sim = CacheSimulator::new(cache_line_size, num_cache_lines);
    sim.access_sequence(addresses);
    sim.stats().misses
}

/// Generate a sequential stride access pattern (useful for testing).
pub fn sequential_stride_pattern(start: u64, count: usize, stride: u64) -> Vec<u64> {
    (0..count).map(|i| start + (i as u64) * stride).collect()
}

/// Generate a repeated-block access pattern: accesses cycle through `block_size`
/// consecutive cache-line-aligned addresses, repeated `repeats` times.
pub fn repeated_block_pattern(
    base: u64,
    block_size: usize,
    repeats: usize,
    elem_size: u64,
) -> Vec<u64> {
    let mut addrs = Vec::with_capacity(block_size * repeats);
    for _ in 0..repeats {
        for i in 0..block_size {
            addrs.push(base + (i as u64) * elem_size);
        }
    }
    addrs
}

// ---------------------------------------------------------------------------
// Multi-level cache hierarchy (L1 / L2 / L3)
// ---------------------------------------------------------------------------

/// Simulates a three-level inclusive cache hierarchy (L1 → L2 → L3).
#[derive(Debug)]
pub struct MultiLevelCache {
    l1: CacheSimulator,
    l2: CacheSimulator,
    l3: CacheSimulator,
    l1_stats: CacheStats,
    l2_stats: CacheStats,
    l3_stats: CacheStats,
}

impl MultiLevelCache {
    /// Create a new multi-level cache.
    ///
    /// * `l1_lines` – number of cache lines in L1
    /// * `l2_lines` – number of cache lines in L2
    /// * `l3_lines` – number of cache lines in L3
    /// * `line_size` – cache line size in bytes (shared across all levels, power of 2)
    pub fn new(l1_lines: usize, l2_lines: usize, l3_lines: usize, line_size: u64) -> Self {
        Self {
            l1: CacheSimulator::new(line_size, l1_lines),
            l2: CacheSimulator::new(line_size, l2_lines),
            l3: CacheSimulator::new(line_size, l3_lines),
            l1_stats: CacheStats::default(),
            l2_stats: CacheStats::default(),
            l3_stats: CacheStats::default(),
        }
    }

    /// Access an address through the cache hierarchy.
    ///
    /// Returns `(CacheResult, level)` where `level` indicates which cache
    /// satisfied the request: 1 = L1 hit, 2 = L2 hit, 3 = L3 hit,
    /// 0 = miss at all levels.
    pub fn access(&mut self, addr: u64) -> (CacheResult, u8) {
        self.l1_stats.total_accesses += 1;

        // Try L1
        if self.l1.access(addr) == CacheResult::Hit {
            self.l1_stats.hits += 1;
            return (CacheResult::Hit, 1);
        }
        self.l1_stats.misses += 1;

        // L1 miss → try L2
        self.l2_stats.total_accesses += 1;
        if self.l2.access(addr) == CacheResult::Hit {
            self.l2_stats.hits += 1;
            return (CacheResult::Hit, 2);
        }
        self.l2_stats.misses += 1;

        // L2 miss → try L3
        self.l3_stats.total_accesses += 1;
        if self.l3.access(addr) == CacheResult::Hit {
            self.l3_stats.hits += 1;
            return (CacheResult::Hit, 3);
        }
        self.l3_stats.misses += 1;

        (CacheResult::Miss, 0)
    }

    /// Return a human-readable summary of per-level statistics.
    pub fn stats_summary(&self) -> String {
        format!(
            "L1: {} | L2: {} | L3: {}",
            self.l1_stats, self.l2_stats, self.l3_stats
        )
    }
}

// ---------------------------------------------------------------------------
// Cache trace recorder
// ---------------------------------------------------------------------------

/// Wraps a [`CacheSimulator`] and records every access result.
#[derive(Debug)]
pub struct CacheTraceRecorder {
    inner: CacheSimulator,
    trace: Vec<(u64, CacheResult)>,
}

impl CacheTraceRecorder {
    pub fn new(cache_line_size: u64, num_lines: usize) -> Self {
        Self {
            inner: CacheSimulator::new(cache_line_size, num_lines),
            trace: Vec::new(),
        }
    }

    pub fn access(&mut self, addr: u64) -> CacheResult {
        let result = self.inner.access(addr);
        self.trace.push((addr, result));
        result
    }

    pub fn trace(&self) -> &[(u64, CacheResult)] {
        &self.trace
    }

    /// Indices in the trace that were cache hits.
    pub fn hit_indices(&self) -> Vec<usize> {
        self.trace
            .iter()
            .enumerate()
            .filter_map(|(i, (_, r))| if *r == CacheResult::Hit { Some(i) } else { None })
            .collect()
    }

    /// Indices in the trace that were cache misses.
    pub fn miss_indices(&self) -> Vec<usize> {
        self.trace
            .iter()
            .enumerate()
            .filter_map(|(i, (_, r))| if *r == CacheResult::Miss { Some(i) } else { None })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Trace replay
// ---------------------------------------------------------------------------

/// Reset `cache`, replay every address in `trace`, and return the final stats.
pub fn replay_trace(trace: &[u64], cache: &mut CacheSimulator) -> CacheStats {
    cache.reset();
    cache.access_sequence(trace);
    cache.stats().clone()
}

// ---------------------------------------------------------------------------
// Bélády's optimal (MIN) algorithm
// ---------------------------------------------------------------------------

use std::collections::{HashMap, HashSet, BinaryHeap};

/// Compute the minimum number of cache misses achievable for `trace` on a
/// cache that can hold `cache_size` distinct pages (Bélády's offline optimal
/// algorithm).
pub fn optimal_cache_misses(trace: &[u64], cache_size: usize) -> u64 {
    if cache_size == 0 {
        return trace.len() as u64;
    }

    // Pre-compute, for each position, the next occurrence of the same page.
    let n = trace.len();
    let mut next_use: Vec<usize> = vec![n; n];
    let mut last_seen: HashMap<u64, usize> = HashMap::new();
    for i in (0..n).rev() {
        if let Some(&prev) = last_seen.get(&trace[i]) {
            next_use[i] = prev;
        }
        last_seen.insert(trace[i], i);
    }

    let mut cache_set: HashSet<u64> = HashSet::with_capacity(cache_size);
    // Max-heap of (next_use, page) so we can quickly find the page whose next
    // use is farthest in the future.
    let mut heap: BinaryHeap<(usize, u64)> = BinaryHeap::new();
    let mut misses: u64 = 0;

    for i in 0..n {
        let page = trace[i];
        if cache_set.contains(&page) {
            // Hit – push updated next-use (stale entries are handled lazily).
            heap.push((next_use[i], page));
            continue;
        }

        // Miss
        misses += 1;

        if cache_set.len() == cache_size {
            // Evict: pop until we find a page still in the cache.
            while let Some((_, evict_page)) = heap.pop() {
                if cache_set.remove(&evict_page) {
                    break;
                }
            }
        }

        cache_set.insert(page);
        heap.push((next_use[i], page));
    }

    misses
}

// ---------------------------------------------------------------------------
// Realistic cache configuration (models real x86 hardware)
// ---------------------------------------------------------------------------

/// Cache model selection for benchmarks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheModel {
    /// Fully-associative LRU (theoretical, optimistic).
    FullyAssociative,
    /// Set-associative LRU (realistic hardware model).
    SetAssociative,
}

/// Realistic x86-like cache hierarchy parameters.
#[derive(Debug, Clone)]
pub struct RealisticCacheConfig {
    /// L1 line size (bytes).
    pub line_size: u64,
    /// L1 number of sets.
    pub l1_sets: usize,
    /// L1 associativity (ways per set).
    pub l1_ways: usize,
}

impl Default for RealisticCacheConfig {
    fn default() -> Self {
        // Typical Intel L1d: 32 KB, 8-way, 64-byte lines → 64 sets
        Self {
            line_size: 64,
            l1_sets: 64,
            l1_ways: 8,
        }
    }
}

impl RealisticCacheConfig {
    /// Total L1 cache lines.
    pub fn total_lines(&self) -> usize {
        self.l1_sets * self.l1_ways
    }
}

/// Count cache misses using a realistic set-associative model.
pub fn count_cache_misses_realistic(
    addresses: &[u64],
    config: &RealisticCacheConfig,
) -> u64 {
    let mut cache = SetAssociativeCache::new(config.line_size, config.l1_sets, config.l1_ways);
    cache.access_sequence(addresses);
    cache.stats().misses
}

/// Compare fully-associative vs set-associative miss counts for a trace.
#[derive(Debug, Clone)]
pub struct CacheModelComparison {
    pub fa_misses: u64,
    pub sa_misses: u64,
    pub total_accesses: usize,
    pub fa_miss_rate: f64,
    pub sa_miss_rate: f64,
    /// Ratio sa_misses / fa_misses (>1 means set-associative has more misses).
    pub conflict_overhead: f64,
}

/// Compare fully-associative and set-associative cache models on the same trace.
pub fn compare_cache_models(
    addresses: &[u64],
    config: &RealisticCacheConfig,
) -> CacheModelComparison {
    let fa_misses = count_cache_misses(addresses, config.line_size, config.total_lines());
    let sa_misses = count_cache_misses_realistic(addresses, config);
    let n = addresses.len();
    let fa_rate = if n > 0 { fa_misses as f64 / n as f64 } else { 0.0 };
    let sa_rate = if n > 0 { sa_misses as f64 / n as f64 } else { 0.0 };
    let overhead = if fa_misses > 0 { sa_misses as f64 / fa_misses as f64 } else { 1.0 };
    CacheModelComparison {
        fa_misses,
        sa_misses,
        total_accesses: n,
        fa_miss_rate: fa_rate,
        sa_miss_rate: sa_rate,
        conflict_overhead: overhead,
    }
}

// ---------------------------------------------------------------------------
// Stack (reuse) distance analysis
// ---------------------------------------------------------------------------

/// For each access in `trace`, compute the *reuse distance*: the number of
/// distinct addresses accessed since the previous access to the same address.
/// The first access to any address gets `usize::MAX`.
pub fn stack_distance_analysis(trace: &[u64]) -> Vec<usize> {
    let mut distances = Vec::with_capacity(trace.len());
    // Stack of distinct recent addresses (front = most recent).
    let mut stack: Vec<u64> = Vec::new();

    for &addr in trace {
        if let Some(pos) = stack.iter().position(|&a| a == addr) {
            distances.push(pos);
            stack.remove(pos);
        } else {
            distances.push(usize::MAX);
        }
        stack.insert(0, addr);
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Fully-associative tests ---

    #[test]
    fn test_cold_misses() {
        let mut cache = CacheSimulator::new(64, 4);
        // Access 4 different cache lines → 4 compulsory misses
        for i in 0..4 {
            assert_eq!(cache.access(i * 64), CacheResult::Miss);
        }
        assert_eq!(cache.stats().misses, 4);
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn test_hits_after_load() {
        let mut cache = CacheSimulator::new(64, 4);
        cache.access(0);
        assert_eq!(cache.access(0), CacheResult::Hit);
        assert_eq!(cache.access(32), CacheResult::Hit); // same cache line as 0
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = CacheSimulator::new(64, 2);
        cache.access(0);        // line 0 → miss
        cache.access(64);       // line 1 → miss
        cache.access(128);      // line 2 → miss, evicts line 0 (LRU)
        assert_eq!(cache.access(0), CacheResult::Miss); // line 0 was evicted
        assert_eq!(cache.access(128), CacheResult::Hit); // line 2 still present
    }

    #[test]
    fn test_lru_ordering() {
        let mut cache = CacheSimulator::new(64, 2);
        cache.access(0);   // [0]
        cache.access(64);  // [64, 0]
        cache.access(0);   // hit → [0, 64], 64 is now LRU
        cache.access(128); // miss → evicts 64
        assert_eq!(cache.access(0), CacheResult::Hit);
        assert_eq!(cache.access(64), CacheResult::Miss);
    }

    #[test]
    fn test_reset() {
        let mut cache = CacheSimulator::new(64, 4);
        cache.access(0);
        cache.access(64);
        cache.reset();
        assert_eq!(cache.stats().total_accesses, 0);
        assert_eq!(cache.access(0), CacheResult::Miss);
    }

    #[test]
    fn test_sequential_pattern_all_misses() {
        // Accessing N distinct cache lines with a cache of size < N → all misses after warmup
        let addrs = sequential_stride_pattern(0, 100, 64);
        let misses = count_cache_misses(&addrs, 64, 8);
        assert_eq!(misses, 100); // all cold misses (100 distinct lines, only 8 fit)
    }

    #[test]
    fn test_repeated_block_pattern() {
        // Accessing a small block repeatedly should warm up and then hit
        let addrs = repeated_block_pattern(0, 4, 10, 64);
        let mut cache = CacheSimulator::new(64, 4);
        let results = cache.access_sequence(&addrs);
        // First 4 are misses, rest should be hits
        assert_eq!(results[0], CacheResult::Miss);
        assert_eq!(results[1], CacheResult::Miss);
        assert_eq!(results[2], CacheResult::Miss);
        assert_eq!(results[3], CacheResult::Miss);
        for r in &results[4..] {
            assert_eq!(*r, CacheResult::Hit);
        }
    }

    #[test]
    fn test_miss_rate() {
        let mut cache = CacheSimulator::new(64, 4);
        cache.access(0);
        cache.access(0);
        cache.access(0);
        cache.access(0);
        let stats = cache.stats();
        assert_eq!(stats.total_accesses, 4);
        assert_eq!(stats.misses, 1);
        assert!((stats.miss_rate() - 0.25).abs() < 1e-10);
        assert!((stats.hit_rate() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_cache_line_size_getter() {
        let cache = CacheSimulator::new(128, 16);
        assert_eq!(cache.cache_line_size(), 128);
        assert_eq!(cache.num_cache_lines(), 16);
    }

    // --- Direct-mapped cache tests ---

    #[test]
    fn test_direct_mapped_cold_miss() {
        let mut cache = DirectMappedCache::new(64, 4);
        assert_eq!(cache.access(0), CacheResult::Miss);
        assert_eq!(cache.access(64), CacheResult::Miss);
        assert_eq!(cache.access(0), CacheResult::Hit);
    }

    #[test]
    fn test_direct_mapped_conflict() {
        let mut cache = DirectMappedCache::new(64, 4);
        // Lines that map to the same set: 0 and 256 both index=0
        cache.access(0);
        cache.access(256); // evicts 0
        assert_eq!(cache.access(0), CacheResult::Miss);
    }

    #[test]
    fn test_direct_mapped_reset() {
        let mut cache = DirectMappedCache::new(64, 4);
        cache.access(0);
        cache.reset();
        assert_eq!(cache.stats().total_accesses, 0);
        assert_eq!(cache.access(0), CacheResult::Miss);
    }

    #[test]
    fn test_direct_mapped_sequence() {
        let mut cache = DirectMappedCache::new(64, 4);
        let addrs = vec![0, 64, 128, 192, 0, 64];
        let results = cache.access_sequence(&addrs);
        assert_eq!(results[0], CacheResult::Miss);
        assert_eq!(results[4], CacheResult::Hit); // 0 still in set 0
    }

    // --- Set-associative cache tests ---

    #[test]
    fn test_set_assoc_basic() {
        let mut cache = SetAssociativeCache::new(64, 4, 2);
        assert_eq!(cache.total_lines(), 8);
        assert_eq!(cache.access(0), CacheResult::Miss);
        assert_eq!(cache.access(0), CacheResult::Hit);
    }

    #[test]
    fn test_set_assoc_ways() {
        // 2-way with 4 sets: set index = block_addr % 4
        let mut cache = SetAssociativeCache::new(64, 4, 2);
        // block 0, block 4, block 8 all map to set 0
        cache.access(0 * 64);   // set 0, tag=0 → miss. ways=[0, _]
        cache.access(4 * 64);   // set 0, tag=1 → miss. ways=[1, 0]
        // both should be present
        assert_eq!(cache.access(0 * 64), CacheResult::Hit);  // ways=[0, 1]
        // tag 1 is now LRU
        cache.access(8 * 64);   // tag=2 → miss, evicts tag=1. ways=[2, 0]
        assert_eq!(cache.access(4 * 64), CacheResult::Miss); // tag=1 evicted, ways=[1, 2]
        assert_eq!(cache.access(8 * 64), CacheResult::Hit);  // tag=2 still present
    }

    #[test]
    fn test_set_assoc_reset() {
        let mut cache = SetAssociativeCache::new(64, 4, 2);
        cache.access(0);
        cache.access(64);
        cache.reset();
        assert_eq!(cache.stats().total_accesses, 0);
        assert_eq!(cache.access(0), CacheResult::Miss);
    }

    #[test]
    fn test_set_assoc_sequence() {
        let mut cache = SetAssociativeCache::new(64, 4, 2);
        let addrs: Vec<u64> = (0..8).map(|i| i * 64).collect();
        let results = cache.access_sequence(&addrs);
        // All cold misses
        assert!(results.iter().all(|r| *r == CacheResult::Miss));
        assert_eq!(cache.stats().misses, 8);
    }

    #[test]
    fn test_cache_stats_display() {
        let stats = CacheStats {
            total_accesses: 100,
            hits: 80,
            misses: 20,
        };
        let s = format!("{}", stats);
        assert!(s.contains("accesses=100"));
        assert!(s.contains("misses=20"));
    }

    #[test]
    fn test_same_line_different_offsets() {
        let mut cache = CacheSimulator::new(64, 4);
        cache.access(0);
        // Bytes 0..63 are in the same line
        for offset in 1..64 {
            assert_eq!(cache.access(offset), CacheResult::Hit);
        }
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_count_cache_misses_helper() {
        let addrs: Vec<u64> = (0..10).map(|i| i * 64).collect();
        let misses = count_cache_misses(&addrs, 64, 16);
        assert_eq!(misses, 10); // 10 compulsory misses, all fit
    }

    #[test]
    fn test_sequential_stride_pattern_gen() {
        let pat = sequential_stride_pattern(100, 5, 8);
        assert_eq!(pat, vec![100, 108, 116, 124, 132]);
    }

    #[test]
    fn test_repeated_block_pattern_gen() {
        let pat = repeated_block_pattern(0, 3, 2, 8);
        assert_eq!(pat, vec![0, 8, 16, 0, 8, 16]);
    }

    // --- Multi-level cache tests ---

    #[test]
    fn test_multi_level_l1_hit() {
        let mut ml = MultiLevelCache::new(4, 8, 16, 64);
        // First access → miss everywhere
        let (res, level) = ml.access(0);
        assert_eq!(res, CacheResult::Miss);
        assert_eq!(level, 0);
        // Second access to same line → L1 hit
        let (res, level) = ml.access(0);
        assert_eq!(res, CacheResult::Hit);
        assert_eq!(level, 1);
    }

    #[test]
    fn test_multi_level_l2_hit() {
        // L1 holds 2 lines, L2 holds 4, L3 holds 8, line_size=64
        let mut ml = MultiLevelCache::new(2, 4, 8, 64);
        // Load lines 0,1,2 → line 0 evicted from L1 but stays in L2
        ml.access(0 * 64);
        ml.access(1 * 64);
        ml.access(2 * 64); // evicts line 0 from L1
        let (res, level) = ml.access(0 * 64);
        assert_eq!(res, CacheResult::Hit);
        assert_eq!(level, 2); // found in L2
    }

    #[test]
    fn test_multi_level_stats_summary() {
        let mut ml = MultiLevelCache::new(2, 4, 8, 64);
        ml.access(0);
        ml.access(0);
        let summary = ml.stats_summary();
        assert!(summary.contains("L1:"));
        assert!(summary.contains("L2:"));
        assert!(summary.contains("L3:"));
    }

    // --- Trace recorder tests ---

    #[test]
    fn test_trace_recorder() {
        let mut rec = CacheTraceRecorder::new(64, 4);
        rec.access(0);
        rec.access(64);
        rec.access(0); // hit
        rec.access(128);

        assert_eq!(rec.trace().len(), 4);
        assert_eq!(rec.hit_indices(), vec![2]);
        assert_eq!(rec.miss_indices(), vec![0, 1, 3]);
    }

    // --- Replay trace test ---

    #[test]
    fn test_replay_trace() {
        let trace = vec![0u64, 64, 0, 64, 128];
        let mut cache = CacheSimulator::new(64, 4);
        // Dirty the cache first
        cache.access(999);
        let stats = replay_trace(&trace, &mut cache);
        // After replay, stats reflect only the trace
        assert_eq!(stats.total_accesses, 5);
        assert_eq!(stats.misses, 3); // 0 miss, 64 miss, 0 hit, 64 hit, 128 miss
        assert_eq!(stats.hits, 2);
    }

    // --- Optimal (Bélády) tests ---

    #[test]
    fn test_optimal_cache_misses_basic() {
        // trace: 0,1,2,3,0,1,4,0,1,2,3,4  with cache_size=3
        // Classic Bélády example
        let trace: Vec<u64> = vec![0, 1, 2, 3, 0, 1, 4, 0, 1, 2, 3, 4];
        let misses = optimal_cache_misses(&trace, 3);
        // Optimal should do no worse than LRU
        let lru_misses = count_cache_misses(&trace, 1, 3);
        assert!(misses <= lru_misses, "optimal {} should be <= LRU {}", misses, lru_misses);
    }

    #[test]
    fn test_optimal_all_hits_when_cache_large() {
        let trace: Vec<u64> = vec![0, 1, 2, 3, 0, 1, 2, 3];
        // Cache big enough to hold all distinct pages → only compulsory misses
        let misses = optimal_cache_misses(&trace, 4);
        assert_eq!(misses, 4); // 4 compulsory misses, then all hits
    }

    // --- Stack distance tests ---

    #[test]
    fn test_stack_distance_analysis() {
        // trace: A B A C B
        let trace = vec![10u64, 20, 10, 30, 20];
        let dists = stack_distance_analysis(&trace);
        // 10: first access → MAX
        assert_eq!(dists[0], usize::MAX);
        // 20: first access → MAX
        assert_eq!(dists[1], usize::MAX);
        // 10: last seen 1 distinct addr ago (20) → distance 1
        assert_eq!(dists[2], 1);
        // 30: first access → MAX
        assert_eq!(dists[3], usize::MAX);
        // 20: last seen 2 distinct addrs ago (10, 30) → distance 2
        assert_eq!(dists[4], 2);
    }

    // --- Realistic cache model tests ---

    #[test]
    fn test_realistic_cache_config_default() {
        let config = RealisticCacheConfig::default();
        assert_eq!(config.line_size, 64);
        assert_eq!(config.l1_sets, 64);
        assert_eq!(config.l1_ways, 8);
        assert_eq!(config.total_lines(), 512);
    }

    #[test]
    fn test_count_cache_misses_realistic() {
        let config = RealisticCacheConfig::default();
        let addrs: Vec<u64> = (0..10).map(|i| i * 64).collect();
        let misses = count_cache_misses_realistic(&addrs, &config);
        assert_eq!(misses, 10); // all compulsory misses
    }

    #[test]
    fn test_realistic_sa_more_misses_than_fa() {
        // Conflict-heavy pattern: addresses that map to the same set
        let config = RealisticCacheConfig { line_size: 64, l1_sets: 64, l1_ways: 2 };
        // All map to set 0: addresses 0, 64*64, 2*64*64, ...
        let stride = 64 * 64; // same set index
        let addrs: Vec<u64> = (0..100).map(|i| (i as u64) * stride).collect();
        let fa_misses = count_cache_misses(&addrs, 64, config.total_lines());
        let sa_misses = count_cache_misses_realistic(&addrs, &config);
        // SA should have more misses due to set conflicts
        assert!(sa_misses >= fa_misses,
            "SA ({}) should have >= FA ({}) misses on conflict pattern", sa_misses, fa_misses);
    }

    #[test]
    fn test_compare_cache_models() {
        let config = RealisticCacheConfig::default();
        let addrs: Vec<u64> = (0..100).map(|i| i * 64).collect();
        let cmp = compare_cache_models(&addrs, &config);
        assert_eq!(cmp.total_accesses, 100);
        assert!(cmp.fa_miss_rate > 0.0);
        assert!(cmp.sa_miss_rate >= cmp.fa_miss_rate);
        assert!(cmp.conflict_overhead >= 1.0);
    }

    #[test]
    fn test_compare_cache_models_sequential() {
        // Sequential pattern should show minimal conflict overhead
        let config = RealisticCacheConfig::default();
        let addrs: Vec<u64> = (0..1000).map(|i| i * 8).collect();
        let cmp = compare_cache_models(&addrs, &config);
        // Sequential pattern has good spatial locality, low conflict
        assert!(cmp.conflict_overhead <= 2.0,
            "Sequential pattern should have low conflict overhead, got {}", cmp.conflict_overhead);
    }
}
