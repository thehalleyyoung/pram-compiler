//! Profile-guided optimization for hash-partition compilation.

use crate::autotuner::cache_probe::CacheHierarchy;
use crate::autotuner::param_optimizer::{TuningKnobs, TuningHashFamily};
use crate::benchmark::cache_sim::{CacheSimulator, SetAssociativeCache};

/// Profile data collected from a sample run.
#[derive(Debug, Clone)]
pub struct ProfileData {
    pub access_pattern: Vec<u64>,
    pub cache_misses_l1: usize,
    pub cache_misses_l2: usize,
    pub working_set_size: usize,
    pub spatial_locality_score: f64,
    pub temporal_locality_score: f64,
    pub stride_regularity: f64,
}

pub struct ProfileGuidedOptimizer {
    hierarchy: CacheHierarchy,
    sample_fraction: f64,
}

impl ProfileGuidedOptimizer {
    pub fn new(hierarchy: CacheHierarchy) -> Self {
        Self { hierarchy, sample_fraction: 0.1 }
    }

    pub fn with_sample_fraction(mut self, frac: f64) -> Self {
        self.sample_fraction = frac.max(0.01).min(1.0);
        self
    }

    pub fn profile(&self, addresses: &[u64]) -> ProfileData {
        let sample_size = ((addresses.len() as f64 * self.sample_fraction) as usize).max(100);
        let sample: Vec<u64> = addresses.iter().take(sample_size).copied().collect();

        let l1 = self.hierarchy.l1d();
        let l1_lines = l1.size_bytes / l1.line_size;
        let mut l1_cache = CacheSimulator::new(l1.line_size as u64, l1_lines);
        l1_cache.access_sequence(&sample);
        let l1_stats = l1_cache.stats();

        let l2_info = self.hierarchy.level(2).cloned()
            .unwrap_or(self.hierarchy.l1d().clone());
        let l2_lines = l2_info.size_bytes / l2_info.line_size;
        let mut l2_cache = SetAssociativeCache::new(l2_info.line_size as u64, l2_lines, l2_info.associativity);
        l2_cache.access_sequence(&sample);
        let l2_stats = l2_cache.stats();

        let spatial = compute_spatial_locality(&sample, l1.line_size);
        let temporal = compute_temporal_locality(&sample);
        let stride = compute_stride_regularity(&sample);
        let wss = compute_working_set_size(&sample, l1.line_size);

        ProfileData {
            access_pattern: sample,
            cache_misses_l1: l1_stats.misses as usize,
            cache_misses_l2: l2_stats.misses as usize,
            working_set_size: wss,
            spatial_locality_score: spatial,
            temporal_locality_score: temporal,
            stride_regularity: stride,
        }
    }

    pub fn optimize_from_profile(&self, profile: &ProfileData) -> TuningKnobs {
        let l1 = self.hierarchy.l1d();

        let hash_family = if profile.stride_regularity > 0.8 {
            TuningHashFamily::Murmur
        } else if profile.spatial_locality_score > 0.7 {
            TuningHashFamily::TwoUniversal
        } else {
            TuningHashFamily::Siegel
        };

        let wss_ratio = profile.working_set_size as f64 / l1.size_bytes as f64;
        let block_size = if wss_ratio < 0.5 {
            l1.line_size * 4
        } else if wss_ratio < 2.0 {
            l1.line_size * 2
        } else {
            l1.line_size
        };

        let tile_frac = if profile.temporal_locality_score > 0.7 { 0.75 } else { 0.5 };
        let tile_size = self.hierarchy.optimal_tile_size(1, 8, tile_frac);

        let prefetch_distance = if profile.stride_regularity > 0.5 { 8 } else { 2 };

        TuningKnobs {
            hash_family,
            block_size,
            tile_size,
            prefetch_distance,
            loop_unroll_factor: 4,
            multi_level_partition: profile.working_set_size > l1.size_bytes,
        }
    }

    pub fn optimize(&self, addresses: &[u64]) -> (ProfileData, TuningKnobs) {
        let profile = self.profile(addresses);
        let knobs = self.optimize_from_profile(&profile);
        (profile, knobs)
    }
}

fn compute_spatial_locality(addresses: &[u64], line_size: usize) -> f64 {
    if addresses.len() < 2 { return 1.0; }
    let mut same_line = 0usize;
    for i in 1..addresses.len() {
        if addresses[i] / line_size as u64 == addresses[i - 1] / line_size as u64 {
            same_line += 1;
        }
    }
    same_line as f64 / (addresses.len() - 1) as f64
}

fn compute_temporal_locality(addresses: &[u64]) -> f64 {
    if addresses.len() < 2 { return 1.0; }
    let mut reuse_count = 0usize;
    let window = 64.min(addresses.len());
    for i in 1..addresses.len() {
        let start = if i > window { i - window } else { 0 };
        if addresses[start..i].contains(&addresses[i]) {
            reuse_count += 1;
        }
    }
    reuse_count as f64 / (addresses.len() - 1) as f64
}

fn compute_stride_regularity(addresses: &[u64]) -> f64 {
    if addresses.len() < 3 { return 0.0; }
    let mut stride_counts = std::collections::HashMap::new();
    for i in 1..addresses.len() {
        let stride = addresses[i] as i64 - addresses[i - 1] as i64;
        *stride_counts.entry(stride).or_insert(0usize) += 1;
    }
    let max_count = stride_counts.values().max().copied().unwrap_or(0);
    max_count as f64 / (addresses.len() - 1) as f64
}

fn compute_working_set_size(addresses: &[u64], line_size: usize) -> usize {
    let lines: std::collections::HashSet<u64> = addresses.iter()
        .map(|a| a / line_size as u64)
        .collect();
    lines.len() * line_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_sequential() {
        let h = CacheHierarchy::default_hierarchy();
        let opt = ProfileGuidedOptimizer::new(h);
        let addrs: Vec<u64> = (0..1000).map(|i| i * 8).collect();
        let profile = opt.profile(&addrs);
        assert!(profile.spatial_locality_score >= 0.0);
        assert!(profile.working_set_size > 0);
    }

    #[test]
    fn test_optimize_from_profile() {
        let h = CacheHierarchy::default_hierarchy();
        let opt = ProfileGuidedOptimizer::new(h);
        let addrs: Vec<u64> = (0..1000).map(|i| i * 8).collect();
        let (_, knobs) = opt.optimize(&addrs);
        assert!(knobs.block_size >= 64);
        assert!(knobs.tile_size > 0);
    }

    #[test]
    fn test_spatial_locality() {
        let sequential: Vec<u64> = (0..100).collect();
        assert!(compute_spatial_locality(&sequential, 64) > 0.9);
    }

    #[test]
    fn test_stride_regularity() {
        let strided: Vec<u64> = (0..100).map(|i| i * 8).collect();
        assert!(compute_stride_regularity(&strided) > 0.9);
    }
}
