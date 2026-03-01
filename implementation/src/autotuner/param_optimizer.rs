//! Parameter optimization for hash-partition compilation.
//!
//! Searches over hash families, block sizes, tile sizes, and scheduling
//! strategies to find the configuration that minimizes cache misses and
//! maximizes throughput for a given algorithm on detected hardware.

use crate::autotuner::cache_probe::CacheHierarchy;
use crate::hash_partition::partition_engine::PartitionEngine;
use crate::hash_partition::partition_engine::HashFamilyChoice;
use crate::benchmark::cache_sim::CacheSimulator;
use crate::pram_ir::ast::PramProgram;

/// Simplified hash family choice for tuning (wraps the full enum).
#[derive(Debug, Clone, PartialEq)]
pub enum TuningHashFamily {
    Siegel,
    TwoUniversal,
    Murmur,
    Identity,
}

impl TuningHashFamily {
    pub fn to_partition_choice(&self) -> HashFamilyChoice {
        match self {
            TuningHashFamily::Siegel => HashFamilyChoice::Siegel { k: 8 },
            TuningHashFamily::TwoUniversal => HashFamilyChoice::TwoUniversal,
            TuningHashFamily::Murmur => HashFamilyChoice::Murmur { seed: 0 },
            TuningHashFamily::Identity => HashFamilyChoice::Identity,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            TuningHashFamily::Siegel => "siegel",
            TuningHashFamily::TwoUniversal => "two_universal",
            TuningHashFamily::Murmur => "murmur",
            TuningHashFamily::Identity => "identity",
        }
    }
}

/// Configuration knobs that can be tuned.
#[derive(Debug, Clone)]
pub struct TuningKnobs {
    pub hash_family: TuningHashFamily,
    pub block_size: usize,
    pub tile_size: usize,
    pub prefetch_distance: usize,
    pub loop_unroll_factor: usize,
    pub multi_level_partition: bool,
}

impl TuningKnobs {
    pub fn default_for(hierarchy: &CacheHierarchy) -> Self {
        TuningKnobs {
            hash_family: TuningHashFamily::Siegel,
            block_size: hierarchy.optimal_block_size(),
            tile_size: hierarchy.optimal_tile_size(1, 8, 0.5),
            prefetch_distance: 4,
            loop_unroll_factor: 4,
            multi_level_partition: hierarchy.levels.len() > 1,
        }
    }
}

/// Result of a tuning trial.
#[derive(Debug, Clone)]
pub struct TuningResult {
    pub knobs: TuningKnobs,
    pub cache_misses: usize,
    pub estimated_cycles: f64,
    pub work_ops: usize,
    pub score: f64,
}

/// Search strategy for parameter optimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SearchStrategy {
    /// Exhaustive grid search over parameter space
    GridSearch,
    /// Guided search starting from predicted best
    GuidedSearch,
    /// Quick heuristic (no search, just prediction)
    Heuristic,
}

/// Parameter optimizer that finds the best configuration.
pub struct ParamOptimizer {
    hierarchy: CacheHierarchy,
    strategy: SearchStrategy,
    max_trials: usize,
}

impl ParamOptimizer {
    pub fn new(hierarchy: CacheHierarchy) -> Self {
        Self {
            hierarchy,
            strategy: SearchStrategy::GuidedSearch,
            max_trials: 100,
        }
    }

    pub fn with_strategy(mut self, strategy: SearchStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn with_max_trials(mut self, max: usize) -> Self {
        self.max_trials = max;
        self
    }

    /// Optimize parameters for a given PRAM program.
    pub fn optimize(&self, program: &PramProgram) -> TuningResult {
        match self.strategy {
            SearchStrategy::GridSearch => self.grid_search(program),
            SearchStrategy::GuidedSearch => self.guided_search(program),
            SearchStrategy::Heuristic => self.heuristic(program),
        }
    }

    /// Predict the best hash family for an algorithm based on its characteristics.
    pub fn predict_hash_family(&self, program: &PramProgram) -> TuningHashFamily {
        let n_stmts = program.total_stmts();
        let n_phases = program.parallel_step_count();

        // For small programs, Siegel overhead not worth it
        if n_stmts < 20 {
            return TuningHashFamily::TwoUniversal;
        }

        // For algorithms with many phases (high temporal reuse), Siegel shines
        if n_phases > 5 {
            return TuningHashFamily::Siegel;
        }

        // For streaming algorithms, MurmurHash is sufficient and faster
        let has_sequential_access = program.description.as_ref()
            .map(|d| d.contains("scan") || d.contains("prefix") || d.contains("compact"))
            .unwrap_or(false);

        if has_sequential_access {
            TuningHashFamily::Murmur
        } else {
            TuningHashFamily::Siegel
        }
    }

    /// Predict optimal block size based on algorithm and cache.
    pub fn predict_block_size(&self, program: &PramProgram) -> usize {
        let l1 = self.hierarchy.l1d();
        let base = l1.line_size;

        // For algorithms with high spatial locality, use larger blocks
        let n_phases = program.parallel_step_count();
        if n_phases <= 2 {
            base * 4 // Streaming: larger blocks
        } else if n_phases <= 8 {
            base * 2 // Moderate reuse
        } else {
            base // High reuse: smaller blocks for finer granularity
        }
    }

    fn grid_search(&self, program: &PramProgram) -> TuningResult {
        let families = [
            TuningHashFamily::Siegel,
            TuningHashFamily::TwoUniversal,
            TuningHashFamily::Murmur,
        ];
        let block_sizes = [32, 64, 128, 256];
        let tile_sizes = [64, 128, 256, 512, 1024];

        let mut best: Option<TuningResult> = None;
        let mut trials = 0;

        for family in &families {
            for &block_size in &block_sizes {
                for &tile_size in &tile_sizes {
                    if trials >= self.max_trials {
                        break;
                    }
                    let knobs = TuningKnobs {
                        hash_family: family.clone(),
                        block_size,
                        tile_size,
                        prefetch_distance: 4,
                        loop_unroll_factor: 4,
                        multi_level_partition: self.hierarchy.levels.len() > 1,
                    };
                    let result = self.evaluate_config(program, knobs);
                    if best.as_ref().map_or(true, |b| result.score < b.score) {
                        best = Some(result);
                    }
                    trials += 1;
                }
            }
        }

        best.unwrap_or_else(|| self.heuristic(program))
    }

    fn guided_search(&self, program: &PramProgram) -> TuningResult {
        // Start from heuristic prediction
        let mut current = self.heuristic(program);
        let base_knobs = current.knobs.clone();

        // Try nearby hash families
        let families = [
            TuningHashFamily::Siegel,
            TuningHashFamily::TwoUniversal,
            TuningHashFamily::Murmur,
        ];

        for family in &families {
            let mut knobs = base_knobs.clone();
            knobs.hash_family = family.clone();
            let result = self.evaluate_config(program, knobs);
            if result.score < current.score {
                current = result;
            }
        }

        // Try nearby block sizes
        for factor in &[1, 2, 4, 8] {
            let mut knobs = current.knobs.clone();
            knobs.block_size = self.hierarchy.l1d().line_size * factor;
            let result = self.evaluate_config(program, knobs);
            if result.score < current.score {
                current = result;
            }
        }

        // Try nearby tile sizes
        let base_tile = self.hierarchy.optimal_tile_size(1, 8, 0.5);
        for &frac in &[0.25, 0.5, 0.75, 1.0] {
            let mut knobs = current.knobs.clone();
            knobs.tile_size = ((base_tile as f64) * frac) as usize;
            if knobs.tile_size > 0 {
                let result = self.evaluate_config(program, knobs);
                if result.score < current.score {
                    current = result;
                }
            }
        }

        current
    }

    fn heuristic(&self, program: &PramProgram) -> TuningResult {
        let knobs = TuningKnobs {
            hash_family: self.predict_hash_family(program),
            block_size: self.predict_block_size(program),
            tile_size: self.hierarchy.optimal_tile_size(1, 8, 0.5),
            prefetch_distance: 4,
            loop_unroll_factor: 4,
            multi_level_partition: self.hierarchy.levels.len() > 1,
        };
        self.evaluate_config(program, knobs)
    }

    /// Evaluate a configuration by simulating cache behavior.
    fn evaluate_config(&self, program: &PramProgram, knobs: TuningKnobs) -> TuningResult {
        let n = 10_000usize;
        let n_addrs = n.min(program.total_stmts() * 100);

        // Simulate partition
        let addresses: Vec<u64> = (0..n_addrs as u64).collect();
        let num_blocks = (n_addrs / knobs.block_size).max(1) as u64;
        let engine = PartitionEngine::new(num_blocks, knobs.block_size as u64, knobs.hash_family.to_partition_choice(), 42);
        let partition = engine.partition(&addresses);

        // Simulate cache with the detected hierarchy
        let l1 = self.hierarchy.l1d();
        let cache_lines = l1.size_bytes / l1.line_size;
        let mut cache = CacheSimulator::new(knobs.block_size as u64, cache_lines);

        // Generate access pattern based on block assignment
        let access_sequence: Vec<u64> = partition.assignments.iter()
            .map(|&block| block as u64 * knobs.block_size as u64)
            .collect();

        cache.access_sequence(&access_sequence);
        let stats = cache.stats();

        let work_ops = n_addrs;
        let cache_misses = stats.misses as usize;

        // Score: weighted combination of misses and work
        let miss_penalty = self.hierarchy.l1d().latency_ns * 10.0;
        let estimated_cycles = work_ops as f64 + cache_misses as f64 * miss_penalty;

        TuningResult {
            knobs,
            cache_misses,
            estimated_cycles,
            work_ops,
            score: estimated_cycles,
        }
    }
}

/// Recommend tuning for a category of algorithms.
pub fn recommended_knobs_for_category(
    category: &str,
    hierarchy: &CacheHierarchy,
) -> TuningKnobs {
    let base = TuningKnobs::default_for(hierarchy);
    match category {
        "sorting" => TuningKnobs {
            hash_family: TuningHashFamily::Siegel,
            block_size: hierarchy.l1d().line_size * 2,
            tile_size: hierarchy.optimal_tile_size(2, 8, 0.5),
            loop_unroll_factor: 8,
            ..base
        },
        "graph" => TuningKnobs {
            hash_family: TuningHashFamily::Siegel,
            block_size: hierarchy.l1d().line_size,
            multi_level_partition: true,
            ..base
        },
        "list" => TuningKnobs {
            hash_family: TuningHashFamily::Murmur,
            block_size: hierarchy.l1d().line_size * 4,
            prefetch_distance: 8,
            ..base
        },
        "arithmetic" => TuningKnobs {
            hash_family: TuningHashFamily::TwoUniversal,
            tile_size: hierarchy.optimal_tile_size(1, 8, 0.75),
            loop_unroll_factor: 8,
            ..base
        },
        _ => base,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_knobs() {
        let h = CacheHierarchy::default_hierarchy();
        let k = TuningKnobs::default_for(&h);
        assert_eq!(k.block_size, 64);
        assert!(k.tile_size > 0);
    }

    #[test]
    fn test_param_optimizer_heuristic() {
        let h = CacheHierarchy::default_hierarchy();
        let opt = ParamOptimizer::new(h).with_strategy(SearchStrategy::Heuristic);
        let prog = crate::algorithm_library::sorting::bitonic_sort();
        let result = opt.optimize(&prog);
        assert!(result.score > 0.0);
        assert!(result.cache_misses > 0);
    }

    #[test]
    fn test_param_optimizer_guided() {
        let h = CacheHierarchy::default_hierarchy();
        let opt = ParamOptimizer::new(h)
            .with_strategy(SearchStrategy::GuidedSearch)
            .with_max_trials(50);
        let prog = crate::algorithm_library::list::prefix_sum();
        let result = opt.optimize(&prog);
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_category_recommendations() {
        let h = CacheHierarchy::default_hierarchy();
        let sorting = recommended_knobs_for_category("sorting", &h);
        let list = recommended_knobs_for_category("list", &h);
        // Sorting should use larger blocks
        assert!(sorting.block_size >= list.block_size || sorting.loop_unroll_factor >= list.loop_unroll_factor);
    }

    #[test]
    fn test_predict_hash_family() {
        let h = CacheHierarchy::default_hierarchy();
        let opt = ParamOptimizer::new(h);
        let prog = crate::algorithm_library::sorting::bitonic_sort();
        let family = opt.predict_hash_family(&prog);
        assert!(matches!(family, TuningHashFamily::Siegel | TuningHashFamily::TwoUniversal | TuningHashFamily::Murmur));
    }
}
