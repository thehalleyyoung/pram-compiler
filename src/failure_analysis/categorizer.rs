//! Categorize algorithm failures by root cause pattern.

use crate::failure_analysis::analyzer::{AlgorithmAnalysis, FailureCategory};
use serde::{Serialize, Deserialize};

/// Summary statistics by failure category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryStats {
    pub category: String,
    pub count: usize,
    pub avg_severity: f64,
    pub affected_algorithms: Vec<String>,
}

/// Complete categorization report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategorizationReport {
    pub total_algorithms: usize,
    pub algorithms_passing: usize,
    pub algorithms_failing: usize,
    pub pass_rate: f64,
    pub category_breakdown: Vec<CategoryStats>,
    pub most_common_category: String,
    pub highest_severity_category: String,
}

/// Categorize a batch of analysis results.
pub fn categorize_results(analyses: &[AlgorithmAnalysis]) -> CategorizationReport {
    let total = analyses.len();
    let passing = analyses.iter().filter(|a| a.meets_2x_target).count();
    let failing = total - passing;

    let mut category_map: std::collections::HashMap<String, Vec<(String, u8)>> =
        std::collections::HashMap::new();

    for analysis in analyses {
        for failure in &analysis.failures {
            category_map
                .entry(failure.name().to_string())
                .or_default()
                .push((analysis.algorithm_name.clone(), failure.severity()));
        }
    }

    let mut category_breakdown: Vec<CategoryStats> = category_map
        .into_iter()
        .map(|(cat, entries)| {
            let count = entries.len();
            let avg_severity = entries.iter().map(|(_, s)| *s as f64).sum::<f64>() / count as f64;
            let affected: Vec<String> = entries.into_iter().map(|(name, _)| name).collect();
            CategoryStats {
                category: cat,
                count,
                avg_severity,
                affected_algorithms: affected,
            }
        })
        .collect();

    category_breakdown.sort_by(|a, b| b.count.cmp(&a.count));

    let most_common = category_breakdown
        .first()
        .map(|c| c.category.clone())
        .unwrap_or_default();
    let highest_severity = category_breakdown
        .iter()
        .max_by(|a, b| a.avg_severity.partial_cmp(&b.avg_severity).unwrap())
        .map(|c| c.category.clone())
        .unwrap_or_default();

    CategorizationReport {
        total_algorithms: total,
        algorithms_passing: passing,
        algorithms_failing: failing,
        pass_rate: if total > 0 { passing as f64 / total as f64 * 100.0 } else { 0.0 },
        category_breakdown,
        most_common_category: most_common,
        highest_severity_category: highest_severity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::failure_analysis::analyzer::FailureAnalyzer;

    #[test]
    fn test_categorize_results() {
        let analyzer = FailureAnalyzer::new();
        let programs = vec![
            crate::algorithm_library::sorting::bitonic_sort(),
            crate::algorithm_library::list::prefix_sum(),
            crate::algorithm_library::graph::shiloach_vishkin(),
        ];
        let analyses: Vec<_> = programs.iter().map(|p| analyzer.analyze(p)).collect();
        let report = categorize_results(&analyses);
        assert_eq!(report.total_algorithms, 3);
        assert!(report.pass_rate >= 0.0);
    }
}
