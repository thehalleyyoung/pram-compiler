//! Regression tracking for algorithm performance across tuning iterations.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// A single performance measurement point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePoint {
    pub algorithm: String,
    pub input_size: usize,
    pub cache_misses: usize,
    pub work_ops: usize,
    pub ratio_vs_baseline: f64,
    pub iteration: usize,
    pub timestamp: String,
}

/// Track performance regressions across tuning iterations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTracker {
    history: Vec<PerformancePoint>,
    baselines: HashMap<String, f64>,
}

impl RegressionTracker {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            baselines: HashMap::new(),
        }
    }

    /// Set baseline performance for an algorithm.
    pub fn set_baseline(&mut self, algorithm: &str, ratio: f64) {
        self.baselines.insert(algorithm.to_string(), ratio);
    }

    /// Record a performance measurement.
    pub fn record(&mut self, point: PerformancePoint) {
        self.history.push(point);
    }

    /// Check for regressions: performance worse than previous iteration.
    pub fn check_regressions(&self, algorithm: &str) -> Vec<String> {
        let points: Vec<&PerformancePoint> = self.history.iter()
            .filter(|p| p.algorithm == algorithm)
            .collect();

        let mut regressions = Vec::new();
        for i in 1..points.len() {
            if points[i].ratio_vs_baseline > points[i - 1].ratio_vs_baseline * 1.1 {
                regressions.push(format!(
                    "Regression in {} at iteration {}: {:.2}x -> {:.2}x",
                    algorithm, points[i].iteration,
                    points[i - 1].ratio_vs_baseline,
                    points[i].ratio_vs_baseline
                ));
            }
        }
        regressions
    }

    /// Get improvement trend for an algorithm.
    pub fn improvement_trend(&self, algorithm: &str) -> Option<f64> {
        let points: Vec<&PerformancePoint> = self.history.iter()
            .filter(|p| p.algorithm == algorithm)
            .collect();

        if points.len() < 2 {
            return None;
        }

        let first = points.first()?.ratio_vs_baseline;
        let last = points.last()?.ratio_vs_baseline;

        Some((first - last) / first * 100.0)
    }

    /// Get the latest performance for all algorithms.
    pub fn latest_performance(&self) -> HashMap<String, f64> {
        let mut latest: HashMap<String, (usize, f64)> = HashMap::new();
        for point in &self.history {
            let entry = latest.entry(point.algorithm.clone())
                .or_insert((0, point.ratio_vs_baseline));
            if point.iteration >= entry.0 {
                *entry = (point.iteration, point.ratio_vs_baseline);
            }
        }
        latest.into_iter().map(|(k, (_, v))| (k, v)).collect()
    }

    /// Count algorithms meeting the 2x target in latest iteration.
    pub fn count_meeting_target(&self, target: f64) -> (usize, usize) {
        let latest = self.latest_performance();
        let meeting = latest.values().filter(|&&r| r <= target).count();
        (meeting, latest.len())
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regression_tracker() {
        let mut tracker = RegressionTracker::new();
        tracker.set_baseline("bitonic_sort", 3.0);

        tracker.record(PerformancePoint {
            algorithm: "bitonic_sort".to_string(),
            input_size: 10000,
            cache_misses: 5000,
            work_ops: 100000,
            ratio_vs_baseline: 3.0,
            iteration: 0,
            timestamp: "2024-01-01".to_string(),
        });

        tracker.record(PerformancePoint {
            algorithm: "bitonic_sort".to_string(),
            input_size: 10000,
            cache_misses: 3000,
            work_ops: 100000,
            ratio_vs_baseline: 1.8,
            iteration: 1,
            timestamp: "2024-01-02".to_string(),
        });

        let regressions = tracker.check_regressions("bitonic_sort");
        assert!(regressions.is_empty()); // Improved, not regressed

        let trend = tracker.improvement_trend("bitonic_sort");
        assert!(trend.unwrap() > 0.0); // Positive improvement
    }

    #[test]
    fn test_count_meeting_target() {
        let mut tracker = RegressionTracker::new();
        tracker.record(PerformancePoint {
            algorithm: "a".to_string(),
            input_size: 1000,
            cache_misses: 100,
            work_ops: 1000,
            ratio_vs_baseline: 1.5,
            iteration: 0,
            timestamp: "t".to_string(),
        });
        tracker.record(PerformancePoint {
            algorithm: "b".to_string(),
            input_size: 1000,
            cache_misses: 100,
            work_ops: 1000,
            ratio_vs_baseline: 2.5,
            iteration: 0,
            timestamp: "t".to_string(),
        });

        let (meeting, total) = tracker.count_meeting_target(2.0);
        assert_eq!(meeting, 1);
        assert_eq!(total, 2);
    }
}
