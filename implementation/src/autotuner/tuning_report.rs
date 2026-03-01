//! Tuning report generation.
//!
//! Produces JSON and human-readable reports of tuning decisions,
//! including detected hardware, selected parameters, and expected performance.

use serde::{Serialize, Deserialize};
use crate::autotuner::cache_probe::{CacheHierarchy, CacheLevelInfo, DetectionMethod};
use crate::autotuner::param_optimizer::{TuningKnobs, TuningHashFamily};

/// Complete tuning report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningReport {
    pub hardware: HardwareReport,
    pub selected_params: SelectedParams,
    pub per_algorithm: Vec<AlgorithmTuning>,
    pub summary: TuningSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareReport {
    pub detection_method: String,
    pub cache_levels: Vec<CacheLevelReport>,
    pub page_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheLevelReport {
    pub level: usize,
    pub size_kb: f64,
    pub line_size: usize,
    pub associativity: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectedParams {
    pub hash_family: String,
    pub block_size: usize,
    pub tile_size: usize,
    pub prefetch_distance: usize,
    pub loop_unroll_factor: usize,
    pub multi_level: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmTuning {
    pub algorithm: String,
    pub category: String,
    pub hash_family: String,
    pub block_size: usize,
    pub estimated_cache_misses: usize,
    pub estimated_speedup: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningSummary {
    pub total_algorithms: usize,
    pub avg_estimated_speedup: f64,
    pub algorithms_above_2x: usize,
    pub coverage_pct: f64,
}

impl TuningReport {
    /// Create a report from a cache hierarchy and tuning results.
    pub fn new(hierarchy: &CacheHierarchy, knobs: &TuningKnobs) -> Self {
        let cache_levels: Vec<CacheLevelReport> = hierarchy.levels.iter()
            .map(|l| CacheLevelReport {
                level: l.level,
                size_kb: l.size_bytes as f64 / 1024.0,
                line_size: l.line_size,
                associativity: l.associativity,
            })
            .collect();

        let detection_str = match hierarchy.detection_method {
            DetectionMethod::SystemInfo => "system_info",
            DetectionMethod::LatencyProbe => "latency_probe",
            DetectionMethod::Default => "default",
        };

        let hash_str = knobs.hash_family.name();

        TuningReport {
            hardware: HardwareReport {
                detection_method: detection_str.to_string(),
                cache_levels,
                page_size: hierarchy.page_size,
            },
            selected_params: SelectedParams {
                hash_family: hash_str.to_string(),
                block_size: knobs.block_size,
                tile_size: knobs.tile_size,
                prefetch_distance: knobs.prefetch_distance,
                loop_unroll_factor: knobs.loop_unroll_factor,
                multi_level: knobs.multi_level_partition,
            },
            per_algorithm: Vec::new(),
            summary: TuningSummary {
                total_algorithms: 0,
                avg_estimated_speedup: 0.0,
                algorithms_above_2x: 0,
                coverage_pct: 0.0,
            },
        }
    }

    /// Add an algorithm tuning result.
    pub fn add_algorithm(&mut self, alg: AlgorithmTuning) {
        self.per_algorithm.push(alg);
        self.update_summary();
    }

    fn update_summary(&mut self) {
        let n = self.per_algorithm.len();
        self.summary.total_algorithms = n;
        if n > 0 {
            let total_speedup: f64 = self.per_algorithm.iter()
                .map(|a| a.estimated_speedup)
                .sum();
            self.summary.avg_estimated_speedup = total_speedup / n as f64;
            self.summary.algorithms_above_2x = self.per_algorithm.iter()
                .filter(|a| a.estimated_speedup >= 2.0)
                .count();
            self.summary.coverage_pct = self.summary.algorithms_above_2x as f64 / n as f64 * 100.0;
        }
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Format as human-readable text.
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        s.push_str("=== PRAM Compiler Auto-Tuning Report ===\n\n");

        s.push_str("Hardware Detection:\n");
        s.push_str(&format!("  Method: {}\n", self.hardware.detection_method));
        for cl in &self.hardware.cache_levels {
            s.push_str(&format!(
                "  L{}: {:.0} KB, {} B lines, {}-way\n",
                cl.level, cl.size_kb, cl.line_size, cl.associativity
            ));
        }

        s.push_str("\nSelected Parameters:\n");
        s.push_str(&format!("  Hash family: {}\n", self.selected_params.hash_family));
        s.push_str(&format!("  Block size: {} B\n", self.selected_params.block_size));
        s.push_str(&format!("  Tile size: {}\n", self.selected_params.tile_size));
        s.push_str(&format!("  Prefetch distance: {}\n", self.selected_params.prefetch_distance));
        s.push_str(&format!("  Multi-level: {}\n", self.selected_params.multi_level));

        if !self.per_algorithm.is_empty() {
            s.push_str("\nPer-Algorithm Results:\n");
            s.push_str(&format!(
                "  {:30} {:15} {:>10} {:>10}\n",
                "Algorithm", "Hash Family", "Misses", "Speedup"
            ));
            s.push_str(&format!("  {:-<70}\n", ""));
            for alg in &self.per_algorithm {
                s.push_str(&format!(
                    "  {:30} {:15} {:>10} {:>9.2}x\n",
                    alg.algorithm, alg.hash_family, alg.estimated_cache_misses, alg.estimated_speedup
                ));
            }
        }

        s.push_str(&format!("\nSummary:\n"));
        s.push_str(&format!("  Algorithms tuned: {}\n", self.summary.total_algorithms));
        s.push_str(&format!(
            "  Avg estimated speedup: {:.2}x\n",
            self.summary.avg_estimated_speedup
        ));
        s.push_str(&format!(
            "  Above 2x target: {}/{} ({:.0}%)\n",
            self.summary.algorithms_above_2x,
            self.summary.total_algorithms,
            self.summary.coverage_pct
        ));

        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_creation() {
        let h = CacheHierarchy::default_hierarchy();
        let k = TuningKnobs::default_for(&h);
        let report = TuningReport::new(&h, &k);
        assert_eq!(report.hardware.cache_levels.len(), 3);
        assert!(report.to_json().len() > 10);
    }

    #[test]
    fn test_report_with_algorithms() {
        let h = CacheHierarchy::default_hierarchy();
        let k = TuningKnobs::default_for(&h);
        let mut report = TuningReport::new(&h, &k);

        report.add_algorithm(AlgorithmTuning {
            algorithm: "bitonic_sort".to_string(),
            category: "sorting".to_string(),
            hash_family: "siegel".to_string(),
            block_size: 64,
            estimated_cache_misses: 1500,
            estimated_speedup: 2.5,
        });

        report.add_algorithm(AlgorithmTuning {
            algorithm: "prefix_sum".to_string(),
            category: "list".to_string(),
            hash_family: "murmur".to_string(),
            block_size: 256,
            estimated_cache_misses: 200,
            estimated_speedup: 3.1,
        });

        assert_eq!(report.summary.total_algorithms, 2);
        assert_eq!(report.summary.algorithms_above_2x, 2);
        assert_eq!(report.summary.coverage_pct, 100.0);

        let text = report.to_text();
        assert!(text.contains("bitonic_sort"));
        assert!(text.contains("prefix_sum"));
    }

    #[test]
    fn test_report_json_roundtrip() {
        let h = CacheHierarchy::default_hierarchy();
        let k = TuningKnobs::default_for(&h);
        let report = TuningReport::new(&h, &k);
        let json = report.to_json();
        let parsed: TuningReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.hardware.cache_levels.len(), 3);
    }
}
