//! Adaptive hash family selection.

use crate::hash_partition::partition_engine::{HashFamilyChoice, PartitionEngine};
use crate::autotuner::param_optimizer::TuningHashFamily;

const OVERFLOW_THRESHOLD_RATIO: f64 = 2.0;

#[derive(Debug, Clone)]
pub struct AdaptiveHashResult {
    pub selected_family: TuningHashFamily,
    pub overflow_max: f64,
    pub overflow_mean: f64,
    pub candidates_tested: usize,
    pub improvement_over_default: f64,
}

pub struct AdaptiveHashSelector {
    block_size: usize,
    max_candidates: usize,
}

impl AdaptiveHashSelector {
    pub fn new(block_size: usize) -> Self {
        Self { block_size, max_candidates: 3 }
    }

    pub fn select(&self, addresses: &[u64]) -> AdaptiveHashResult {
        let candidates = [
            TuningHashFamily::Siegel,
            TuningHashFamily::TwoUniversal,
            TuningHashFamily::Murmur,
        ];

        let mut best_family = TuningHashFamily::Siegel;
        let mut best_overflow = f64::MAX;
        let mut best_mean = 0.0;
        let mut tested = 0;

        let num_blocks = (addresses.len() / self.block_size).max(1) as u64;

        for family in candidates.iter().take(self.max_candidates) {
            let engine = PartitionEngine::new(
                num_blocks, self.block_size as u64, family.to_partition_choice(), 42,
            );
            let result = engine.partition(addresses);
            tested += 1;

            let max_load = result.overflow.empirical_max_load as f64;
            let expected = result.overflow.expected_load;
            let overflow = if expected > 0.0 { max_load / expected } else { max_load };

            if overflow < best_overflow {
                best_overflow = overflow;
                best_mean = result.statistics.mean_load;
                best_family = family.clone();
            }
        }

        let default_engine = PartitionEngine::new(
            num_blocks, self.block_size as u64,
            HashFamilyChoice::Siegel { k: 8 }, 42,
        );
        let default_result = default_engine.partition(addresses);
        let default_overflow = if default_result.overflow.expected_load > 0.0 {
            default_result.overflow.empirical_max_load as f64 / default_result.overflow.expected_load
        } else {
            default_result.overflow.empirical_max_load as f64
        };

        let improvement = if best_overflow > 0.0 {
            (default_overflow - best_overflow) / default_overflow
        } else {
            0.0
        };

        AdaptiveHashResult {
            selected_family: best_family,
            overflow_max: best_overflow,
            overflow_mean: best_mean,
            candidates_tested: tested,
            improvement_over_default: improvement,
        }
    }

    pub fn needs_adaptation(&self, max_overflow_ratio: f64) -> bool {
        max_overflow_ratio > OVERFLOW_THRESHOLD_RATIO
    }

    pub fn adapt_partition(
        &self,
        addresses: &[u64],
        current_family: &TuningHashFamily,
    ) -> Option<AdaptiveHashResult> {
        let num_blocks = (addresses.len() / self.block_size).max(1) as u64;
        let engine = PartitionEngine::new(
            num_blocks, self.block_size as u64, current_family.to_partition_choice(), 42,
        );
        let result = engine.partition(addresses);

        let overflow_ratio = if result.overflow.expected_load > 0.0 {
            result.overflow.empirical_max_load as f64 / result.overflow.expected_load
        } else {
            result.overflow.empirical_max_load as f64
        };

        if self.needs_adaptation(overflow_ratio) {
            Some(self.select(addresses))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct MultiLevelAdaptiveHash {
    pub l1_family: TuningHashFamily,
    pub l2_family: TuningHashFamily,
    pub l3_family: TuningHashFamily,
    pub l1_block_size: usize,
    pub l2_block_size: usize,
}

impl MultiLevelAdaptiveHash {
    pub fn select_for_hierarchy(addresses: &[u64], l1_line: usize, l2_line: usize) -> Self {
        let l1_selector = AdaptiveHashSelector::new(l1_line);
        let l2_selector = AdaptiveHashSelector::new(l2_line);

        let l1_result = l1_selector.select(addresses);
        let l2_result = l2_selector.select(addresses);

        MultiLevelAdaptiveHash {
            l1_family: l1_result.selected_family,
            l2_family: l2_result.selected_family,
            l3_family: TuningHashFamily::Siegel,
            l1_block_size: l1_line,
            l2_block_size: l2_line,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_selection() {
        let selector = AdaptiveHashSelector::new(64);
        let addrs: Vec<u64> = (0..1000).collect();
        let result = selector.select(&addrs);
        assert!(result.candidates_tested > 0);
        assert!(result.overflow_max >= 1.0);
    }

    #[test]
    fn test_needs_adaptation() {
        let selector = AdaptiveHashSelector::new(64);
        assert!(!selector.needs_adaptation(1.5));
        assert!(selector.needs_adaptation(3.0));
    }

    #[test]
    fn test_multi_level() {
        let addrs: Vec<u64> = (0..1000).collect();
        let ml = MultiLevelAdaptiveHash::select_for_hierarchy(&addrs, 64, 64);
        assert_eq!(ml.l1_block_size, 64);
    }
}
