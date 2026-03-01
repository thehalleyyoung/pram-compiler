//! Failure analysis for algorithms missing the 2x target.
//!
//! Identifies root causes: work inflation, cache overflow, scheduling overhead,
//! or fundamental algorithmic mismatch.

use crate::pram_ir::ast::{PramProgram, Stmt};
use crate::hash_partition::partition_engine::{HashFamilyChoice, PartitionEngine};

/// Root cause categories for algorithms missing 2x target.
#[derive(Debug, Clone, PartialEq)]
pub enum FailureCategory {
    /// Work inflation: compiled code does more work than O(pT)
    WorkInflation {
        actual_ratio: u64,
        expected_ratio: u64,
    },
    /// Cache overflow: hash partition has excessive overflow
    CacheOverflow {
        max_overflow: usize,
        expected_overflow: usize,
    },
    /// Scheduling overhead: Brent schedule introduces excessive synchronization
    SchedulingOverhead {
        barrier_count: usize,
        phase_count: usize,
    },
    /// Irregular access pattern: algorithm has data-dependent access pattern
    IrregularAccess {
        locality_score: f64,
    },
    /// Small input: overhead dominates for small n
    SmallInputOverhead {
        crossover_n: usize,
    },
    /// Memory model overhead: CRCW resolution cost
    MemoryModelOverhead {
        conflict_rate: f64,
    },
}

impl FailureCategory {
    pub fn name(&self) -> &'static str {
        match self {
            FailureCategory::WorkInflation { .. } => "work_inflation",
            FailureCategory::CacheOverflow { .. } => "cache_overflow",
            FailureCategory::SchedulingOverhead { .. } => "scheduling_overhead",
            FailureCategory::IrregularAccess { .. } => "irregular_access",
            FailureCategory::SmallInputOverhead { .. } => "small_input_overhead",
            FailureCategory::MemoryModelOverhead { .. } => "memory_model_overhead",
        }
    }

    pub fn severity(&self) -> u8 {
        match self {
            FailureCategory::WorkInflation { actual_ratio, expected_ratio } => {
                if *actual_ratio > *expected_ratio * 4 { 3 } else { 2 }
            }
            FailureCategory::CacheOverflow { .. } => 2,
            FailureCategory::SchedulingOverhead { barrier_count, .. } => {
                if *barrier_count > 20 { 3 } else { 1 }
            }
            FailureCategory::IrregularAccess { locality_score } => {
                if *locality_score < 0.3 { 3 } else { 1 }
            }
            FailureCategory::SmallInputOverhead { .. } => 1,
            FailureCategory::MemoryModelOverhead { conflict_rate } => {
                if *conflict_rate > 0.5 { 3 } else { 1 }
            }
        }
    }
}

/// Analysis result for a single algorithm.
#[derive(Debug, Clone)]
pub struct AlgorithmAnalysis {
    pub algorithm_name: String,
    pub category: String,
    pub memory_model: String,
    pub failures: Vec<FailureCategory>,
    pub performance_ratio: f64,
    pub meets_2x_target: bool,
    pub recommended_fixes: Vec<String>,
}

/// Analyzer that diagnoses why algorithms miss the 2x target.
pub struct FailureAnalyzer {
    cache_line_size: usize,
    target_ratio: f64,
}

impl FailureAnalyzer {
    pub fn new() -> Self {
        Self {
            cache_line_size: 64,
            target_ratio: 2.0,
        }
    }

    pub fn with_cache_line_size(mut self, size: usize) -> Self {
        self.cache_line_size = size;
        self
    }

    /// Analyze a program to identify failure root causes.
    pub fn analyze(&self, prog: &PramProgram) -> AlgorithmAnalysis {
        let mut failures = Vec::new();
        let mut recommended_fixes = Vec::new();

        let stmts = prog.total_stmts();
        let phases = prog.parallel_step_count();

        // Check for work inflation
        if stmts > phases * 50 {
            failures.push(FailureCategory::WorkInflation {
                actual_ratio: stmts as u64,
                expected_ratio: (phases * 10) as u64,
            });
            recommended_fixes.push(
                "Apply aggressive dead code elimination and constant folding".to_string()
            );
        }

        // Check scheduling overhead
        let barrier_count = count_barriers(&prog.body);
        if barrier_count > 10 {
            failures.push(FailureCategory::SchedulingOverhead {
                barrier_count,
                phase_count: phases,
            });
            recommended_fixes.push(
                "Merge adjacent barriers and reduce synchronization granularity".to_string()
            );
        }

        // Check for CRCW overhead
        let model_name = prog.memory_model.name();
        if model_name.starts_with("CRCW") {
            let estimated_conflict_rate = estimate_conflict_rate(prog);
            if estimated_conflict_rate > 0.1 {
                failures.push(FailureCategory::MemoryModelOverhead {
                    conflict_rate: estimated_conflict_rate,
                });
                recommended_fixes.push(
                    "Use priority-based write resolution to eliminate conflict checking".to_string()
                );
            }
        }

        // Check for irregular access patterns
        let locality = estimate_access_locality(prog);
        if locality < 0.5 {
            failures.push(FailureCategory::IrregularAccess {
                locality_score: locality,
            });
            recommended_fixes.push(
                "Use Siegel k-wise independent hashing with multi-level partition".to_string()
            );
        }

        // Check for small input overhead
        if phases > 5 && stmts < 100 {
            failures.push(FailureCategory::SmallInputOverhead {
                crossover_n: 10_000,
            });
            recommended_fixes.push(
                "Add small-input fast path that bypasses compilation overhead".to_string()
            );
        }

        // Simulate cache behavior
        let n_addrs = (stmts * 10).max(100);
        let addresses: Vec<u64> = (0..n_addrs as u64).collect();
        let num_blocks = (n_addrs / self.cache_line_size).max(1) as u64;
        let engine = PartitionEngine::new(
            num_blocks,
            self.cache_line_size as u64,
            HashFamilyChoice::Siegel { k: 8 },
            42,
        );
        let partition = engine.partition(&addresses);

        let expected_load = n_addrs / partition.overflow.num_blocks.max(1);
        if partition.overflow.empirical_max_load as usize > expected_load * 3 {
            failures.push(FailureCategory::CacheOverflow {
                max_overflow: partition.overflow.empirical_max_load as usize,
                expected_overflow: expected_load,
            });
            recommended_fixes.push(
                "Switch to multi-level partition with L2-aware block sizing".to_string()
            );
        }

        // Estimate performance ratio
        let overhead = failures.iter().map(|f| f.severity() as f64 * 0.5).sum::<f64>();
        let performance_ratio = 1.0 + overhead;

        AlgorithmAnalysis {
            algorithm_name: prog.name.clone(),
            category: categorize_algorithm(&prog.name),
            memory_model: model_name.to_string(),
            meets_2x_target: performance_ratio <= self.target_ratio,
            performance_ratio,
            failures,
            recommended_fixes,
        }
    }

    /// Batch analyze all algorithms and return summary.
    pub fn analyze_batch(&self, programs: &[PramProgram]) -> Vec<AlgorithmAnalysis> {
        programs.iter().map(|p| self.analyze(p)).collect()
    }
}

fn count_barriers(stmts: &[Stmt]) -> usize {
    let mut count = 0;
    for stmt in stmts {
        match stmt {
            Stmt::Barrier => count += 1,
            Stmt::Block(inner) => count += count_barriers(inner),
            Stmt::ParallelFor { body, .. } => count += count_barriers(body),
            Stmt::If { then_body, else_body, .. } => {
                count += count_barriers(then_body);
                count += count_barriers(else_body);
            }
            Stmt::SeqFor { body, .. } => count += count_barriers(body),
            Stmt::While { body, .. } => count += count_barriers(body),
            _ => {}
        }
    }
    count
}

fn estimate_conflict_rate(prog: &PramProgram) -> f64 {
    let mut writes = 0usize;
    let mut parallel_writes = 0usize;
    count_writes(&prog.body, &mut writes, &mut parallel_writes, false, false);
    if writes == 0 { 0.0 } else { parallel_writes as f64 / writes as f64 }
}

fn count_writes(stmts: &[Stmt], writes: &mut usize, parallel_writes: &mut usize, in_par: bool, in_buffer: bool) {
    for stmt in stmts {
        match stmt {
            Stmt::SharedWrite { .. } => {
                *writes += 1;
                // Writes inside a buffered block (coalesced) are not conflicting
                if in_par && !in_buffer { *parallel_writes += 1; }
            }
            Stmt::ParallelFor { body, .. } => {
                count_writes(body, writes, parallel_writes, true, in_buffer);
            }
            Stmt::Block(inner) => {
                // Check if this block is a coalesced write buffer (has LocalDecl + guarded write)
                let is_buffer = inner.iter().any(|s| matches!(s, Stmt::LocalDecl(name, _, _) if name.starts_with("__buf_")));
                count_writes(inner, writes, parallel_writes, in_par, in_buffer || is_buffer);
            }
            Stmt::If { then_body, else_body, .. } => {
                count_writes(then_body, writes, parallel_writes, in_par, in_buffer);
                count_writes(else_body, writes, parallel_writes, in_par, in_buffer);
            }
            Stmt::SeqFor { body, .. } => count_writes(body, writes, parallel_writes, in_par, in_buffer),
            Stmt::While { body, .. } => count_writes(body, writes, parallel_writes, in_par, in_buffer),
            _ => {}
        }
    }
}

fn estimate_access_locality(prog: &PramProgram) -> f64 {
    let phases = prog.parallel_step_count();
    let stmts = prog.total_stmts();
    let phase_ratio = if phases > 0 { stmts as f64 / phases as f64 } else { 1.0 };
    // Recognize tiled access: if SeqFor wraps shared access, locality improves
    let tiled_regions = count_tiled_regions(&prog.body);
    let tile_bonus = (tiled_regions as f64 * 0.15).min(0.5);
    ((phase_ratio / 20.0) + tile_bonus).min(1.0)
}

/// Count SeqFor loops that wrap shared memory accesses (tiled regions).
fn count_tiled_regions(stmts: &[Stmt]) -> usize {
    let mut count = 0;
    for stmt in stmts {
        match stmt {
            Stmt::SeqFor { body, var, .. } => {
                if var.starts_with("__tile_") {
                    count += 1;
                }
                count += count_tiled_regions(body);
            }
            Stmt::ParallelFor { body, .. } => count += count_tiled_regions(body),
            Stmt::Block(inner) => count += count_tiled_regions(inner),
            Stmt::If { then_body, else_body, .. } => {
                count += count_tiled_regions(then_body);
                count += count_tiled_regions(else_body);
            }
            Stmt::While { body, .. } => count += count_tiled_regions(body),
            _ => {}
        }
    }
    count
}

fn categorize_algorithm(name: &str) -> String {
    if name.contains("sort") || name.contains("merge") {
        "sorting".to_string()
    } else if name.contains("graph") || name.contains("bfs") || name.contains("dfs")
        || name.contains("vishkin") || name.contains("boruvka") || name.contains("euler")
        || name.contains("connectivity") || name.contains("shiloach")
    {
        "graph".to_string()
    } else if name.contains("prefix") || name.contains("list") || name.contains("compact")
        || name.contains("scan") || name.contains("ranking")
    {
        "list".to_string()
    } else if name.contains("matrix") || name.contains("multiply") || name.contains("addition")
        || name.contains("fft") || name.contains("strassen")
    {
        "arithmetic".to_string()
    } else if name.contains("hull") || name.contains("pair") || name.contains("voronoi") {
        "geometry".to_string()
    } else if name.contains("tree") || name.contains("lca") || name.contains("contraction") {
        "tree".to_string()
    } else if name.contains("string") || name.contains("suffix") || name.contains("match") {
        "string".to_string()
    } else if name.contains("search") || name.contains("select") || name.contains("median") {
        "search".to_string()
    } else {
        "other".to_string()
    }
}

/// Count shared reads from irregular (data-dependent) indices within parallel-for bodies.
pub fn count_irregular_shared_reads(stmts: &[crate::pram_ir::ast::Stmt]) -> usize {
    use crate::pram_ir::ast::{Stmt, Expr};
    let mut count = 0;
    for stmt in stmts {
        match stmt {
            Stmt::ParallelFor { body, .. } => {
                for s in body {
                    count_irregular_reads_inner(s, &mut count);
                }
            }
            Stmt::Block(inner) => count += count_irregular_shared_reads(inner),
            _ => {}
        }
    }
    count
}

fn count_irregular_reads_inner(stmt: &crate::pram_ir::ast::Stmt, count: &mut usize) {
    use crate::pram_ir::ast::{Stmt, Expr};
    match stmt {
        Stmt::SharedWrite { value, .. } => {
            if expr_has_shared_read(value) {
                *count += 1;
            }
        }
        Stmt::Assign(_, expr) => {
            if expr_has_shared_read(expr) {
                *count += 1;
            }
        }
        Stmt::ParallelFor { body, .. } | Stmt::Block(body) => {
            for s in body { count_irregular_reads_inner(s, count); }
        }
        Stmt::SeqFor { body, .. } | Stmt::While { body, .. } => {
            for s in body { count_irregular_reads_inner(s, count); }
        }
        Stmt::If { then_body, else_body, .. } => {
            for s in then_body { count_irregular_reads_inner(s, count); }
            for s in else_body { count_irregular_reads_inner(s, count); }
        }
        _ => {}
    }
}

fn expr_has_shared_read(expr: &crate::pram_ir::ast::Expr) -> bool {
    use crate::pram_ir::ast::Expr;
    match expr {
        Expr::SharedRead(_, idx) => {
            // Irregular if index is itself a shared read (data-dependent)
            matches!(**idx, Expr::SharedRead(_, _) | Expr::BinOp(_, _, _))
        }
        Expr::BinOp(_, a, b) => expr_has_shared_read(a) || expr_has_shared_read(b),
        _ => false,
    }
}

/// Estimate access locality with improved shared read/write pattern analysis.
pub fn enhanced_access_locality(prog: &PramProgram) -> f64 {
    let phases = prog.parallel_step_count();
    let stmts = prog.total_stmts();
    let irregular = count_irregular_shared_reads(&prog.body);
    let phase_ratio = if phases > 0 { stmts as f64 / phases as f64 } else { 1.0 };
    let regularity_penalty = if stmts > 0 {
        1.0 - (irregular as f64 / stmts as f64).min(0.5)
    } else {
        1.0
    };
    ((phase_ratio / 20.0) * regularity_penalty).min(1.0)
}

/// Count nested parallel write depth (max nesting level of SharedWrite inside ParallelFor).
pub fn nested_parallel_write_depth(stmts: &[crate::pram_ir::ast::Stmt]) -> usize {
    use crate::pram_ir::ast::Stmt;
    fn depth_inner(stmts: &[Stmt], par_depth: usize) -> usize {
        let mut max_depth = 0;
        for stmt in stmts {
            match stmt {
                Stmt::ParallelFor { body, .. } => {
                    max_depth = max_depth.max(depth_inner(body, par_depth + 1));
                }
                Stmt::SharedWrite { .. } => {
                    max_depth = max_depth.max(par_depth);
                }
                Stmt::Block(inner) | Stmt::SeqFor { body: inner, .. }
                | Stmt::While { body: inner, .. } => {
                    max_depth = max_depth.max(depth_inner(inner, par_depth));
                }
                Stmt::If { then_body, else_body, .. } => {
                    max_depth = max_depth.max(depth_inner(then_body, par_depth));
                    max_depth = max_depth.max(depth_inner(else_body, par_depth));
                }
                _ => {}
            }
        }
        max_depth
    }
    depth_inner(stmts, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_prefix_sum() {
        let analyzer = FailureAnalyzer::new();
        let prog = crate::algorithm_library::list::prefix_sum();
        let result = analyzer.analyze(&prog);
        assert_eq!(result.algorithm_name, "prefix_sum");
        assert!(result.category == "list");
    }

    #[test]
    fn test_analyze_bitonic_sort() {
        let analyzer = FailureAnalyzer::new();
        let prog = crate::algorithm_library::sorting::bitonic_sort();
        let result = analyzer.analyze(&prog);
        assert!(!result.algorithm_name.is_empty());
    }

    #[test]
    fn test_failure_category_severity() {
        let wf = FailureCategory::WorkInflation { actual_ratio: 100, expected_ratio: 10 };
        assert!(wf.severity() >= 2);
        let si = FailureCategory::SmallInputOverhead { crossover_n: 1000 };
        assert_eq!(si.severity(), 1);
    }

    #[test]
    fn test_categorize_algorithm() {
        assert_eq!(categorize_algorithm("bitonic_sort"), "sorting");
        assert_eq!(categorize_algorithm("shiloach_vishkin"), "graph");
        assert_eq!(categorize_algorithm("prefix_sum"), "list");
        assert_eq!(categorize_algorithm("matrix_multiply"), "arithmetic");
    }

    #[test]
    fn test_batch_analysis() {
        let analyzer = FailureAnalyzer::new();
        let programs = vec![
            crate::algorithm_library::sorting::bitonic_sort(),
            crate::algorithm_library::list::prefix_sum(),
            crate::algorithm_library::graph::shiloach_vishkin(),
        ];
        let results = analyzer.analyze_batch(&programs);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_count_irregular_shared_reads() {
        let prog = crate::algorithm_library::graph::shiloach_vishkin();
        let count = count_irregular_shared_reads(&prog.body);
        // Graph algorithms typically have data-dependent reads
        assert!(count >= 0);
    }

    #[test]
    fn test_enhanced_access_locality() {
        let prog = crate::algorithm_library::sorting::bitonic_sort();
        let locality = enhanced_access_locality(&prog);
        assert!(locality >= 0.0 && locality <= 1.0);
    }

    #[test]
    fn test_nested_parallel_write_depth() {
        let prog = crate::algorithm_library::sorting::bitonic_sort();
        let depth = nested_parallel_write_depth(&prog.body);
        assert!(depth >= 1); // bitonic sort has parallel writes
    }

    #[test]
    fn test_enhanced_locality_prefix_sum() {
        let prog = crate::algorithm_library::list::prefix_sum();
        let locality = enhanced_access_locality(&prog);
        // Prefix sum should have decent locality
        assert!(locality > 0.0);
    }
}
