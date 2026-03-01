//! Input-size-dependent autotuning.
//!
//! Instead of selecting one configuration per algorithm across all input sizes,
//! this module selects configurations based on input size buckets with smooth
//! interpolation between bucket configs, online adaptation, config-switch cost
//! modelling, multi-objective Pareto optimization, and regression-based
//! prediction for unseen input sizes.

use crate::autotuner::cache_probe::CacheHierarchy;
use crate::autotuner::param_optimizer::{TuningKnobs, TuningHashFamily, TuningResult};
use crate::hash_partition::partition_engine::{PartitionEngine, HashFamilyChoice};
use crate::hash_partition::independence::required_independence;
use crate::benchmark::cache_sim::CacheSimulator;
use crate::pram_ir::ast::PramProgram;
use std::collections::HashMap;

/// Input-size buckets for tuning decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SizeBucket {
    /// n ≤ 256
    Tiny,
    /// 257 ≤ n ≤ 4096
    Small,
    /// 4097 ≤ n ≤ 65536
    Medium,
    /// 65537 ≤ n ≤ 1048576
    Large,
    /// n > 1048576
    Huge,
}

impl SizeBucket {
    pub fn from_size(n: usize) -> Self {
        match n {
            0..=256 => SizeBucket::Tiny,
            257..=4096 => SizeBucket::Small,
            4097..=65536 => SizeBucket::Medium,
            65537..=1048576 => SizeBucket::Large,
            _ => SizeBucket::Huge,
        }
    }

    pub fn representative_size(&self) -> usize {
        match self {
            SizeBucket::Tiny => 128,
            SizeBucket::Small => 1024,
            SizeBucket::Medium => 16384,
            SizeBucket::Large => 262144,
            SizeBucket::Huge => 4194304,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            SizeBucket::Tiny => "tiny",
            SizeBucket::Small => "small",
            SizeBucket::Medium => "medium",
            SizeBucket::Large => "large",
            SizeBucket::Huge => "huge",
        }
    }
}

/// Per-bucket tuning configuration.
#[derive(Debug, Clone)]
pub struct BucketConfig {
    pub bucket: SizeBucket,
    pub knobs: TuningKnobs,
    pub independence_k: usize,
    pub estimated_score: f64,
}

/// Input-dependent autotuner that selects configurations per size bucket.
pub struct InputDependentTuner {
    hierarchy: CacheHierarchy,
    epsilon: f64,
}

impl InputDependentTuner {
    pub fn new(hierarchy: CacheHierarchy) -> Self {
        Self { hierarchy, epsilon: 0.5 }
    }

    pub fn with_epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps.max(0.01).min(2.0);
        self
    }

    /// Tune for a specific input size.
    pub fn tune_for_size(&self, program: &PramProgram, n: usize) -> BucketConfig {
        let bucket = SizeBucket::from_size(n);
        let l1 = self.hierarchy.l1d();
        let base_block_size = l1.line_size;

        // Determine block count and independence parameter
        let num_blocks = (n / base_block_size).max(1);
        let k = required_independence(n, num_blocks, self.epsilon);

        // Select hash family based on size bucket
        let (hash_family, block_size, tile_size) = match bucket {
            SizeBucket::Tiny => {
                // Small inputs: overhead dominates, use simple hashing
                (TuningHashFamily::TwoUniversal, base_block_size * 4, base_block_size * 8)
            }
            SizeBucket::Small => {
                // Moderate: MurmurHash is fast and sufficient
                (TuningHashFamily::Murmur, base_block_size * 2, base_block_size * 4)
            }
            SizeBucket::Medium => {
                // Medium: Siegel provides better bounds
                let phases = program.parallel_step_count();
                if phases > 5 {
                    (TuningHashFamily::Siegel, base_block_size, base_block_size * 2)
                } else {
                    (TuningHashFamily::Murmur, base_block_size * 2, base_block_size * 4)
                }
            }
            SizeBucket::Large | SizeBucket::Huge => {
                // Large: Siegel k-wise for optimal bounds
                (TuningHashFamily::Siegel, base_block_size, base_block_size)
            }
        };

        // Determine multi-level partition usage
        let use_multi_level = bucket == SizeBucket::Large || bucket == SizeBucket::Huge;

        let knobs = TuningKnobs {
            hash_family: hash_family.clone(),
            block_size,
            tile_size,
            prefetch_distance: if matches!(bucket, SizeBucket::Large | SizeBucket::Huge) { 8 } else { 4 },
            loop_unroll_factor: if matches!(bucket, SizeBucket::Tiny) { 2 } else { 4 },
            multi_level_partition: use_multi_level,
        };

        // Estimate score via cache simulation
        let score = self.evaluate_config(&knobs, n);

        BucketConfig {
            bucket,
            knobs,
            independence_k: k,
            estimated_score: score,
        }
    }

    /// Generate a complete tuning schedule for all size buckets.
    pub fn full_schedule(&self, program: &PramProgram) -> Vec<BucketConfig> {
        let buckets = [
            SizeBucket::Tiny,
            SizeBucket::Small,
            SizeBucket::Medium,
            SizeBucket::Large,
            SizeBucket::Huge,
        ];

        buckets.iter()
            .map(|b| self.tune_for_size(program, b.representative_size()))
            .collect()
    }

    /// Evaluate a configuration by simulating cache behavior at a given size.
    fn evaluate_config(&self, knobs: &TuningKnobs, n: usize) -> f64 {
        let addresses: Vec<u64> = (0..n as u64).collect();
        let num_blocks = (n / knobs.block_size).max(1) as u64;
        let engine = PartitionEngine::new(
            num_blocks,
            knobs.block_size as u64,
            knobs.hash_family.to_partition_choice(),
            42,
        );
        let partition = engine.partition(&addresses);

        let l1 = self.hierarchy.l1d();
        let cache_lines = l1.size_bytes / l1.line_size;
        let mut cache = CacheSimulator::new(knobs.block_size as u64, cache_lines);

        let access_seq: Vec<u64> = partition.assignments.iter()
            .map(|&b| b as u64 * knobs.block_size as u64)
            .collect();
        cache.access_sequence(&access_seq);
        let stats = cache.stats();

        let miss_penalty = l1.latency_ns * 10.0;
        n as f64 + stats.misses as f64 * miss_penalty
    }
}

/// Comparison of per-bucket vs single-config tuning.
#[derive(Debug, Clone)]
pub struct TuningComparison {
    pub single_config_score: f64,
    pub per_bucket_score: f64,
    pub improvement_pct: f64,
    pub bucket_configs: Vec<BucketConfig>,
}

/// Compare input-dependent vs fixed tuning for a program.
pub fn compare_tuning_strategies(
    program: &PramProgram,
    hierarchy: &CacheHierarchy,
) -> TuningComparison {
    let tuner = InputDependentTuner::new(hierarchy.clone());
    let schedule = tuner.full_schedule(program);

    // Per-bucket total score
    let per_bucket_total: f64 = schedule.iter().map(|c| c.estimated_score).sum();

    // Single-config score (use medium as representative)
    let medium_config = schedule.iter()
        .find(|c| c.bucket == SizeBucket::Medium)
        .cloned()
        .unwrap_or_else(|| schedule[0].clone());

    let single_total: f64 = [SizeBucket::Tiny, SizeBucket::Small, SizeBucket::Medium, SizeBucket::Large, SizeBucket::Huge]
        .iter()
        .map(|b| tuner.evaluate_config(&medium_config.knobs, b.representative_size()))
        .sum();

    let improvement = if single_total > 0.0 {
        (single_total - per_bucket_total) / single_total * 100.0
    } else {
        0.0
    };

    TuningComparison {
        single_config_score: single_total,
        per_bucket_score: per_bucket_total,
        improvement_pct: improvement,
        bucket_configs: schedule,
    }
}

// ---------------------------------------------------------------------------
// 1. Interpolated configuration selection
// ---------------------------------------------------------------------------

/// A configuration produced by interpolating between two bucket configs.
#[derive(Debug, Clone)]
pub struct InterpolatedConfig {
    pub lower_bucket: SizeBucket,
    pub upper_bucket: SizeBucket,
    pub fraction: f64,
    pub knobs: TuningKnobs,
    pub independence_k: usize,
    pub estimated_score: f64,
}

/// Linearly interpolate numeric knobs between two `TuningKnobs`.
/// `fraction` ∈ [0, 1]: 0 → `low`, 1 → `high`.
/// Hash family is taken from the closer config (< 0.5 → low, else high).
pub fn interpolate_config(low: &TuningKnobs, high: &TuningKnobs, fraction: f64) -> TuningKnobs {
    let t = fraction.max(0.0).min(1.0);
    let lerp = |a: usize, b: usize| -> usize {
        ((a as f64) * (1.0 - t) + (b as f64) * t).round().max(1.0) as usize
    };
    TuningKnobs {
        hash_family: if t < 0.5 { low.hash_family.clone() } else { high.hash_family.clone() },
        block_size: lerp(low.block_size, high.block_size),
        tile_size: lerp(low.tile_size, high.tile_size),
        prefetch_distance: lerp(low.prefetch_distance, high.prefetch_distance),
        loop_unroll_factor: lerp(low.loop_unroll_factor, high.loop_unroll_factor),
        multi_level_partition: if t < 0.5 { low.multi_level_partition } else { high.multi_level_partition },
    }
}

impl InputDependentTuner {
    /// Tune for a specific input size with smooth interpolation between the
    /// two nearest bucket configs instead of hard bucket boundaries.
    pub fn tune_for_size_interpolated(&self, program: &PramProgram, n: usize) -> InterpolatedConfig {
        let boundaries: &[(usize, SizeBucket)] = &[
            (0, SizeBucket::Tiny),
            (256, SizeBucket::Tiny),
            (257, SizeBucket::Small),
            (4096, SizeBucket::Small),
            (4097, SizeBucket::Medium),
            (65536, SizeBucket::Medium),
            (65537, SizeBucket::Large),
            (1048576, SizeBucket::Large),
            (1048577, SizeBucket::Huge),
        ];

        // Find the two surrounding representative sizes.
        let ordered_reps: &[(SizeBucket, usize)] = &[
            (SizeBucket::Tiny, 128),
            (SizeBucket::Small, 1024),
            (SizeBucket::Medium, 16384),
            (SizeBucket::Large, 262144),
            (SizeBucket::Huge, 4194304),
        ];

        let (low_idx, high_idx) = {
            let mut lo = 0usize;
            let mut hi = ordered_reps.len() - 1;
            for (i, &(_, rep)) in ordered_reps.iter().enumerate() {
                if n >= rep {
                    lo = i;
                }
            }
            hi = (lo + 1).min(ordered_reps.len() - 1);
            (lo, hi)
        };

        let (low_bucket, low_rep) = ordered_reps[low_idx];
        let (high_bucket, high_rep) = ordered_reps[high_idx];

        let low_cfg = self.tune_for_size(program, low_rep);
        let high_cfg = self.tune_for_size(program, high_rep);

        let fraction = if low_rep == high_rep {
            0.0
        } else {
            ((n as f64) - (low_rep as f64)) / ((high_rep as f64) - (low_rep as f64))
        };
        let fraction = fraction.max(0.0).min(1.0);

        let knobs = interpolate_config(&low_cfg.knobs, &high_cfg.knobs, fraction);
        let k = {
            let lo_k = low_cfg.independence_k as f64;
            let hi_k = high_cfg.independence_k as f64;
            (lo_k * (1.0 - fraction) + hi_k * fraction).round() as usize
        };
        let score = self.evaluate_config(&knobs, n);

        InterpolatedConfig {
            lower_bucket: low_bucket,
            upper_bucket: high_bucket,
            fraction,
            knobs,
            independence_k: k,
            estimated_score: score,
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Online adaptation
// ---------------------------------------------------------------------------

/// Exponential-moving-average tracker used by `OnlineAdapter`.
#[derive(Debug, Clone)]
struct EmaTracker {
    alpha: f64,
    value: Option<f64>,
}

impl EmaTracker {
    fn new(alpha: f64) -> Self {
        Self { alpha: alpha.max(0.01).min(1.0), value: None }
    }

    fn update(&mut self, sample: f64) {
        self.value = Some(match self.value {
            Some(v) => self.alpha * sample + (1.0 - self.alpha) * v,
            None => sample,
        });
    }

    fn get(&self) -> Option<f64> {
        self.value
    }
}

/// Per-algorithm performance record.
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    pub input_size: usize,
    pub score: f64,
    pub config_name: String,
}

/// Online adapter that tracks performance as input sizes change and
/// switches configurations when predicted improvement exceeds a threshold.
pub struct OnlineAdapter {
    hierarchy: CacheHierarchy,
    threshold: f64,
    ema: EmaTracker,
    history: Vec<PerformanceRecord>,
    current_config: Option<BucketConfig>,
}

impl OnlineAdapter {
    pub fn new(hierarchy: CacheHierarchy, threshold: f64) -> Self {
        Self {
            hierarchy,
            threshold: threshold.max(0.0),
            ema: EmaTracker::new(0.3),
            history: Vec::new(),
            current_config: None,
        }
    }

    /// Record a performance observation and possibly switch config.
    /// Returns `true` if the configuration was switched.
    pub fn observe(&mut self, program: &PramProgram, n: usize, measured_score: f64) -> bool {
        self.ema.update(measured_score);
        let config_name = self.current_config.as_ref()
            .map(|c| c.bucket.name().to_string())
            .unwrap_or_else(|| "none".to_string());
        self.history.push(PerformanceRecord {
            input_size: n,
            score: measured_score,
            config_name,
        });

        let tuner = InputDependentTuner::new(self.hierarchy.clone());
        let candidate = tuner.tune_for_size(program, n);
        let predicted_score = candidate.estimated_score;

        let ema_val = self.ema.get().unwrap_or(measured_score);
        let improvement = (ema_val - predicted_score) / ema_val;

        if improvement > self.threshold || self.current_config.is_none() {
            self.current_config = Some(candidate);
            true
        } else {
            false
        }
    }

    pub fn current_config(&self) -> Option<&BucketConfig> {
        self.current_config.as_ref()
    }

    pub fn history(&self) -> &[PerformanceRecord] {
        &self.history
    }

    pub fn ema_score(&self) -> Option<f64> {
        self.ema.get()
    }
}

// ---------------------------------------------------------------------------
// 3. Cost model for config switching
// ---------------------------------------------------------------------------

/// Estimated cost of switching between two configurations.
#[derive(Debug, Clone)]
pub struct ConfigSwitchCost {
    pub hash_reinit_cost: f64,
    pub rehash_cost: f64,
    pub tile_switch_cost: f64,
    pub total: f64,
}

/// Estimate the cost of switching from `old` to `new` for input of size `n`.
///
/// * Hash family change  → O(k) re-initialisation
/// * Block size change   → O(n/B) rehashing
/// * Tile size change    → O(1)
pub fn estimate_switch_cost(old: &TuningKnobs, new: &TuningKnobs, n: usize, k: usize) -> ConfigSwitchCost {
    let hash_reinit_cost = if old.hash_family != new.hash_family {
        k as f64
    } else {
        0.0
    };

    let rehash_cost = if old.block_size != new.block_size {
        let b = new.block_size.max(1);
        (n as f64) / (b as f64)
    } else {
        0.0
    };

    let tile_switch_cost = if old.tile_size != new.tile_size { 1.0 } else { 0.0 };

    ConfigSwitchCost {
        hash_reinit_cost,
        rehash_cost,
        tile_switch_cost,
        total: hash_reinit_cost + rehash_cost + tile_switch_cost,
    }
}

/// Decide whether to switch: `expected_improvement > switch_cost / amortization_window`.
pub fn should_switch(
    expected_improvement: f64,
    switch_cost: &ConfigSwitchCost,
    amortization_window: usize,
) -> bool {
    let amortised = switch_cost.total / (amortization_window.max(1) as f64);
    expected_improvement > amortised
}

// ---------------------------------------------------------------------------
// 4. Multi-objective (Pareto) optimisation
// ---------------------------------------------------------------------------

/// A point in the (cache_misses, work_overhead, memory_usage) objective space.
#[derive(Debug, Clone)]
pub struct ParetoConfig {
    pub knobs: TuningKnobs,
    pub cache_misses: f64,
    pub work_overhead: f64,
    pub memory_usage: f64,
}

impl ParetoConfig {
    /// Weighted scalarisation of the three objectives.
    pub fn weighted_score(&self, w_cache: f64, w_work: f64, w_mem: f64) -> f64 {
        w_cache * self.cache_misses + w_work * self.work_overhead + w_mem * self.memory_usage
    }
}

/// Return `true` if `a` dominates `b` (all objectives ≤, at least one <).
fn dominates(a: &ParetoConfig, b: &ParetoConfig) -> bool {
    let le = a.cache_misses <= b.cache_misses
        && a.work_overhead <= b.work_overhead
        && a.memory_usage <= b.memory_usage;
    let lt = a.cache_misses < b.cache_misses
        || a.work_overhead < b.work_overhead
        || a.memory_usage < b.memory_usage;
    le && lt
}

/// Find the Pareto-optimal (non-dominated) subset from a set of configs.
pub fn find_pareto_optimal(configs: &[ParetoConfig]) -> Vec<ParetoConfig> {
    let mut frontier: Vec<ParetoConfig> = Vec::new();
    for c in configs {
        let dominated_by_any = configs.iter().any(|other| {
            !std::ptr::eq(c, other) && dominates(other, c)
        });
        if !dominated_by_any {
            frontier.push(c.clone());
        }
    }
    frontier
}

/// Select the best config from the Pareto frontier according to user weights.
pub fn select_from_pareto(
    frontier: &[ParetoConfig],
    w_cache: f64,
    w_work: f64,
    w_mem: f64,
) -> Option<ParetoConfig> {
    frontier.iter()
        .min_by(|a, b| {
            a.weighted_score(w_cache, w_work, w_mem)
                .partial_cmp(&b.weighted_score(w_cache, w_work, w_mem))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned()
}

// ---------------------------------------------------------------------------
// 5. Regression-based prediction
// ---------------------------------------------------------------------------

/// Simple linear regression predictor: cache_misses ≈ β₀ + β₁·n
/// for each hash family.
#[derive(Debug, Clone)]
pub struct RegressionPredictor {
    /// (hash_family_name, intercept, slope)
    models: HashMap<String, (f64, f64)>,
    /// Raw data per family for cross-validation.
    data: HashMap<String, Vec<(f64, f64)>>,
}

impl RegressionPredictor {
    pub fn new() -> Self {
        Self { models: HashMap::new(), data: HashMap::new() }
    }

    /// Add a data point (input_size, cache_misses) for a hash family.
    pub fn add_sample(&mut self, family: &str, n: f64, cache_misses: f64) {
        self.data.entry(family.to_string()).or_default().push((n, cache_misses));
    }

    /// Fit ordinary-least-squares models for every family with ≥ 2 points.
    pub fn fit(&mut self) {
        for (family, pts) in &self.data {
            if pts.len() < 2 {
                continue;
            }
            let n_f = pts.len() as f64;
            let sum_x: f64 = pts.iter().map(|(x, _)| x).sum();
            let sum_y: f64 = pts.iter().map(|(_, y)| y).sum();
            let sum_xy: f64 = pts.iter().map(|(x, y)| x * y).sum();
            let sum_xx: f64 = pts.iter().map(|(x, _)| x * x).sum();
            let denom = n_f * sum_xx - sum_x * sum_x;
            if denom.abs() < 1e-12 {
                self.models.insert(family.clone(), (sum_y / n_f, 0.0));
                continue;
            }
            let slope = (n_f * sum_xy - sum_x * sum_y) / denom;
            let intercept = (sum_y - slope * sum_x) / n_f;
            self.models.insert(family.clone(), (intercept, slope));
        }
    }

    /// Predict cache misses for a given family and input size.
    pub fn predict(&self, family: &str, n: f64) -> Option<f64> {
        self.models.get(family).map(|&(b0, b1)| b0 + b1 * n)
    }

    /// Predict the best hash family for a given input size.
    pub fn predict_best_family(&self, n: f64) -> Option<String> {
        self.models.iter()
            .map(|(fam, &(b0, b1))| (fam.clone(), b0 + b1 * n))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(fam, _)| fam)
    }

    /// Leave-one-out cross-validation mean absolute error per family.
    pub fn cross_validate(&self) -> HashMap<String, f64> {
        let mut errors: HashMap<String, f64> = HashMap::new();
        for (family, pts) in &self.data {
            if pts.len() < 3 {
                continue;
            }
            let mut total_err = 0.0;
            for i in 0..pts.len() {
                // Fit without point i
                let rest: Vec<_> = pts.iter().enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, p)| *p)
                    .collect();
                let n_f = rest.len() as f64;
                let sum_x: f64 = rest.iter().map(|(x, _)| x).sum();
                let sum_y: f64 = rest.iter().map(|(_, y)| y).sum();
                let sum_xy: f64 = rest.iter().map(|(x, y)| x * y).sum();
                let sum_xx: f64 = rest.iter().map(|(x, _)| x * x).sum();
                let denom = n_f * sum_xx - sum_x * sum_x;
                let (b0, b1) = if denom.abs() < 1e-12 {
                    (sum_y / n_f, 0.0)
                } else {
                    let s = (n_f * sum_xy - sum_x * sum_y) / denom;
                    let i_ = (sum_y - s * sum_x) / n_f;
                    (i_, s)
                };
                let pred = b0 + b1 * pts[i].0;
                total_err += (pred - pts[i].1).abs();
            }
            errors.insert(family.clone(), total_err / pts.len() as f64);
        }
        errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autotuner::cache_probe::CacheHierarchy;

    fn test_hierarchy() -> CacheHierarchy {
        CacheHierarchy::default_hierarchy()
    }

    fn test_prog() -> PramProgram {
        crate::algorithm_library::sorting::bitonic_sort()
    }

    // --- Original tests ------------------------------------------------

    #[test]
    fn test_size_bucket_classification() {
        assert_eq!(SizeBucket::from_size(100), SizeBucket::Tiny);
        assert_eq!(SizeBucket::from_size(1000), SizeBucket::Small);
        assert_eq!(SizeBucket::from_size(10000), SizeBucket::Medium);
        assert_eq!(SizeBucket::from_size(100000), SizeBucket::Large);
        assert_eq!(SizeBucket::from_size(10000000), SizeBucket::Huge);
    }

    #[test]
    fn test_tune_for_tiny() {
        let h = test_hierarchy();
        let tuner = InputDependentTuner::new(h);
        let prog = test_prog();
        let config = tuner.tune_for_size(&prog, 128);
        assert_eq!(config.bucket, SizeBucket::Tiny);
        assert_eq!(config.knobs.hash_family, TuningHashFamily::TwoUniversal);
    }

    #[test]
    fn test_tune_for_large() {
        let h = test_hierarchy();
        let tuner = InputDependentTuner::new(h);
        let prog = test_prog();
        let config = tuner.tune_for_size(&prog, 500000);
        assert_eq!(config.bucket, SizeBucket::Large);
        assert_eq!(config.knobs.hash_family, TuningHashFamily::Siegel);
        assert!(config.knobs.multi_level_partition);
    }

    #[test]
    fn test_full_schedule() {
        let h = test_hierarchy();
        let tuner = InputDependentTuner::new(h);
        let prog = test_prog();
        let schedule = tuner.full_schedule(&prog);
        assert_eq!(schedule.len(), 5);
        let families: Vec<_> = schedule.iter().map(|c| c.knobs.hash_family.clone()).collect();
        assert_ne!(families[0], families[4],
            "Tiny and Huge should use different hash families");
    }

    #[test]
    fn test_independence_k_varies() {
        let h = test_hierarchy();
        let tuner = InputDependentTuner::new(h);
        let prog = test_prog();
        let tiny = tuner.tune_for_size(&prog, 128);
        let large = tuner.tune_for_size(&prog, 500000);
        assert!(large.independence_k >= tiny.independence_k,
            "Large inputs should need higher k");
    }

    #[test]
    fn test_compare_strategies() {
        let h = test_hierarchy();
        let prog = test_prog();
        let cmp = compare_tuning_strategies(&prog, &h);
        assert!(cmp.single_config_score > 0.0);
        assert!(cmp.per_bucket_score > 0.0);
        assert_eq!(cmp.bucket_configs.len(), 5);
    }

    #[test]
    fn test_representative_sizes() {
        assert!(SizeBucket::Tiny.representative_size() < SizeBucket::Small.representative_size());
        assert!(SizeBucket::Small.representative_size() < SizeBucket::Medium.representative_size());
        assert!(SizeBucket::Medium.representative_size() < SizeBucket::Large.representative_size());
    }

    // --- 1. Interpolated config tests ----------------------------------

    #[test]
    fn test_interpolate_config_endpoints() {
        let low = TuningKnobs {
            hash_family: TuningHashFamily::TwoUniversal,
            block_size: 64, tile_size: 128,
            prefetch_distance: 4, loop_unroll_factor: 2,
            multi_level_partition: false,
        };
        let high = TuningKnobs {
            hash_family: TuningHashFamily::Siegel,
            block_size: 256, tile_size: 512,
            prefetch_distance: 8, loop_unroll_factor: 8,
            multi_level_partition: true,
        };
        let at_zero = interpolate_config(&low, &high, 0.0);
        assert_eq!(at_zero.block_size, 64);
        assert_eq!(at_zero.hash_family, TuningHashFamily::TwoUniversal);

        let at_one = interpolate_config(&low, &high, 1.0);
        assert_eq!(at_one.block_size, 256);
        assert_eq!(at_one.hash_family, TuningHashFamily::Siegel);
    }

    #[test]
    fn test_interpolate_config_midpoint() {
        let low = TuningKnobs {
            hash_family: TuningHashFamily::TwoUniversal,
            block_size: 100, tile_size: 200,
            prefetch_distance: 4, loop_unroll_factor: 2,
            multi_level_partition: false,
        };
        let high = TuningKnobs {
            hash_family: TuningHashFamily::Siegel,
            block_size: 200, tile_size: 400,
            prefetch_distance: 8, loop_unroll_factor: 4,
            multi_level_partition: true,
        };
        let mid = interpolate_config(&low, &high, 0.5);
        assert_eq!(mid.block_size, 150);
        assert_eq!(mid.tile_size, 300);
    }

    #[test]
    fn test_interpolate_config_clamps_fraction() {
        let low = TuningKnobs {
            hash_family: TuningHashFamily::TwoUniversal,
            block_size: 64, tile_size: 128,
            prefetch_distance: 4, loop_unroll_factor: 2,
            multi_level_partition: false,
        };
        let high = low.clone();
        let out = interpolate_config(&low, &high, 5.0);
        assert_eq!(out.block_size, 64);
    }

    #[test]
    fn test_tune_for_size_interpolated_returns_valid() {
        let h = test_hierarchy();
        let tuner = InputDependentTuner::new(h);
        let prog = test_prog();
        let ic = tuner.tune_for_size_interpolated(&prog, 2048);
        assert!(ic.knobs.block_size > 0);
        assert!(ic.estimated_score > 0.0);
        assert!(ic.fraction >= 0.0 && ic.fraction <= 1.0);
    }

    #[test]
    fn test_tune_for_size_interpolated_tiny_edge() {
        let h = test_hierarchy();
        let tuner = InputDependentTuner::new(h);
        let prog = test_prog();
        let ic = tuner.tune_for_size_interpolated(&prog, 10);
        // Should resolve to Tiny bucket area
        assert_eq!(ic.lower_bucket, SizeBucket::Tiny);
    }

    // --- 2. Online adaptation tests ------------------------------------

    #[test]
    fn test_online_adapter_initial_switch() {
        let h = test_hierarchy();
        let mut adapter = OnlineAdapter::new(h, 0.05);
        let prog = test_prog();
        let switched = adapter.observe(&prog, 1000, 50000.0);
        assert!(switched, "First observation should always set a config");
        assert!(adapter.current_config().is_some());
    }

    #[test]
    fn test_online_adapter_history_recorded() {
        let h = test_hierarchy();
        let mut adapter = OnlineAdapter::new(h, 0.05);
        let prog = test_prog();
        adapter.observe(&prog, 500, 30000.0);
        adapter.observe(&prog, 1000, 40000.0);
        assert_eq!(adapter.history().len(), 2);
    }

    #[test]
    fn test_online_adapter_ema_updates() {
        let h = test_hierarchy();
        let mut adapter = OnlineAdapter::new(h, 0.05);
        let prog = test_prog();
        adapter.observe(&prog, 1000, 100.0);
        let ema1 = adapter.ema_score().unwrap();
        adapter.observe(&prog, 1000, 200.0);
        let ema2 = adapter.ema_score().unwrap();
        // EMA should move toward 200
        assert!(ema2 > ema1);
    }

    // --- 3. Config switch cost tests -----------------------------------

    #[test]
    fn test_switch_cost_same_config() {
        let cfg = TuningKnobs {
            hash_family: TuningHashFamily::Siegel,
            block_size: 64, tile_size: 128,
            prefetch_distance: 4, loop_unroll_factor: 4,
            multi_level_partition: false,
        };
        let cost = estimate_switch_cost(&cfg, &cfg, 10000, 8);
        assert_eq!(cost.total, 0.0);
    }

    #[test]
    fn test_switch_cost_hash_change() {
        let old = TuningKnobs {
            hash_family: TuningHashFamily::Murmur,
            block_size: 64, tile_size: 128,
            prefetch_distance: 4, loop_unroll_factor: 4,
            multi_level_partition: false,
        };
        let new = TuningKnobs {
            hash_family: TuningHashFamily::Siegel,
            ..old.clone()
        };
        let cost = estimate_switch_cost(&old, &new, 10000, 8);
        assert_eq!(cost.hash_reinit_cost, 8.0);
        assert_eq!(cost.rehash_cost, 0.0);
    }

    #[test]
    fn test_switch_cost_block_size_change() {
        let old = TuningKnobs {
            hash_family: TuningHashFamily::Siegel,
            block_size: 64, tile_size: 128,
            prefetch_distance: 4, loop_unroll_factor: 4,
            multi_level_partition: false,
        };
        let new = TuningKnobs { block_size: 128, ..old.clone() };
        let cost = estimate_switch_cost(&old, &new, 12800, 8);
        assert!((cost.rehash_cost - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_should_switch_decision() {
        let cost = ConfigSwitchCost {
            hash_reinit_cost: 8.0,
            rehash_cost: 100.0,
            tile_switch_cost: 1.0,
            total: 109.0,
        };
        // Large amortization window ⇒ small amortised cost
        assert!(should_switch(50.0, &cost, 1000));
        // Tiny window ⇒ cost dominates
        assert!(!should_switch(50.0, &cost, 1));
    }

    // --- 4. Multi-objective / Pareto tests -----------------------------

    #[test]
    fn test_pareto_single_point() {
        let configs = vec![ParetoConfig {
            knobs: TuningKnobs {
                hash_family: TuningHashFamily::Siegel,
                block_size: 64, tile_size: 128,
                prefetch_distance: 4, loop_unroll_factor: 4,
                multi_level_partition: false,
            },
            cache_misses: 10.0, work_overhead: 5.0, memory_usage: 100.0,
        }];
        let front = find_pareto_optimal(&configs);
        assert_eq!(front.len(), 1);
    }

    #[test]
    fn test_pareto_dominance_removes_inferior() {
        let knobs = TuningKnobs {
            hash_family: TuningHashFamily::Siegel,
            block_size: 64, tile_size: 128,
            prefetch_distance: 4, loop_unroll_factor: 4,
            multi_level_partition: false,
        };
        let configs = vec![
            ParetoConfig { knobs: knobs.clone(), cache_misses: 5.0, work_overhead: 5.0, memory_usage: 5.0 },
            ParetoConfig { knobs: knobs.clone(), cache_misses: 10.0, work_overhead: 10.0, memory_usage: 10.0 },
        ];
        let front = find_pareto_optimal(&configs);
        assert_eq!(front.len(), 1);
        assert_eq!(front[0].cache_misses, 5.0);
    }

    #[test]
    fn test_pareto_non_dominated_both_kept() {
        let knobs = TuningKnobs {
            hash_family: TuningHashFamily::Siegel,
            block_size: 64, tile_size: 128,
            prefetch_distance: 4, loop_unroll_factor: 4,
            multi_level_partition: false,
        };
        let configs = vec![
            ParetoConfig { knobs: knobs.clone(), cache_misses: 5.0, work_overhead: 20.0, memory_usage: 10.0 },
            ParetoConfig { knobs: knobs.clone(), cache_misses: 20.0, work_overhead: 5.0, memory_usage: 10.0 },
        ];
        let front = find_pareto_optimal(&configs);
        assert_eq!(front.len(), 2);
    }

    #[test]
    fn test_select_from_pareto_weights() {
        let knobs = TuningKnobs {
            hash_family: TuningHashFamily::Siegel,
            block_size: 64, tile_size: 128,
            prefetch_distance: 4, loop_unroll_factor: 4,
            multi_level_partition: false,
        };
        let frontier = vec![
            ParetoConfig { knobs: knobs.clone(), cache_misses: 1.0, work_overhead: 100.0, memory_usage: 10.0 },
            ParetoConfig { knobs: knobs.clone(), cache_misses: 100.0, work_overhead: 1.0, memory_usage: 10.0 },
        ];
        // Prefer low cache misses
        let best = select_from_pareto(&frontier, 1.0, 0.0, 0.0).unwrap();
        assert_eq!(best.cache_misses, 1.0);
        // Prefer low work overhead
        let best2 = select_from_pareto(&frontier, 0.0, 1.0, 0.0).unwrap();
        assert_eq!(best2.work_overhead, 1.0);
    }

    // --- 5. Regression predictor tests ---------------------------------

    #[test]
    fn test_regression_fit_and_predict() {
        let mut pred = RegressionPredictor::new();
        // y = 2x + 10
        pred.add_sample("siegel", 100.0, 210.0);
        pred.add_sample("siegel", 200.0, 410.0);
        pred.add_sample("siegel", 300.0, 610.0);
        pred.fit();
        let p = pred.predict("siegel", 400.0).unwrap();
        assert!((p - 810.0).abs() < 1.0);
    }

    #[test]
    fn test_regression_best_family() {
        let mut pred = RegressionPredictor::new();
        // Family A: high misses
        pred.add_sample("A", 100.0, 500.0);
        pred.add_sample("A", 200.0, 1000.0);
        // Family B: low misses
        pred.add_sample("B", 100.0, 50.0);
        pred.add_sample("B", 200.0, 100.0);
        pred.fit();
        assert_eq!(pred.predict_best_family(300.0).unwrap(), "B");
    }

    #[test]
    fn test_regression_cross_validation() {
        let mut pred = RegressionPredictor::new();
        for i in 0..10 {
            let x = (i as f64) * 100.0;
            pred.add_sample("lin", x, 3.0 * x + 5.0);
        }
        pred.fit();
        let cv = pred.cross_validate();
        let mae = cv.get("lin").unwrap();
        assert!(*mae < 1.0, "LOO-CV MAE should be very small for perfect linear data");
    }

    #[test]
    fn test_regression_no_data_returns_none() {
        let pred = RegressionPredictor::new();
        assert!(pred.predict("nonexistent", 100.0).is_none());
    }
}
