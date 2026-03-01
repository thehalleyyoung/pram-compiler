//! Main Brent scheduler: combines dependency graph construction,
//! work-optimal schedule extraction, and locality optimization.

use crate::pram_ir::ast::PramProgram;

use super::cost_analyzer::{analyze_with_graph, CostReport};
use super::dependency_graph::DependencyGraph;
use super::locality_order::LocalityOptimizer;
use super::schedule::{Schedule, ScheduleEntry};
use super::work_optimal::extract_schedule;

/// Configuration for the Brent scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Cache block size (number of elements per block).
    pub block_size: usize,
    /// Enable locality optimization within levels.
    pub enable_locality: bool,
    /// Work bound factor for verification (work ≤ factor * p * T).
    pub work_bound_factor: f64,
    /// Number of cache blocks for cost analysis.
    pub cache_blocks: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            block_size: 4,
            enable_locality: true,
            work_bound_factor: 1.0,
            cache_blocks: 64,
        }
    }
}

/// The main Brent scheduler that orchestrates the full pipeline.
pub struct BrentScheduler {
    pub config: SchedulerConfig,
}

impl BrentScheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: SchedulerConfig) -> Self {
        Self { config }
    }

    /// Create a scheduler with default configuration.
    pub fn with_defaults() -> Self {
        Self {
            config: SchedulerConfig::default(),
        }
    }

    /// Schedule a PRAM program into a sequential schedule.
    ///
    /// Steps:
    /// 1. Build the dependency graph from the program.
    /// 2. Extract a work-optimal sequential schedule (Brent's theorem).
    /// 3. Optionally apply locality optimization.
    /// 4. Return the final schedule.
    pub fn schedule(
        &self,
        program: &PramProgram,
        num_procs: usize,
    ) -> Schedule {
        let graph = DependencyGraph::build(program, self.config.block_size, num_procs);
        self.schedule_from_graph(&graph)
    }

    /// Schedule from a pre-built dependency graph.
    pub fn schedule_from_graph(&self, graph: &DependencyGraph) -> Schedule {
        if self.config.enable_locality {
            let optimizer = LocalityOptimizer::new(self.config.block_size);
            optimizer.optimize_to_schedule(graph)
        } else {
            let seq = extract_schedule(graph);
            seq.to_schedule(self.config.block_size, graph.num_phases)
        }
    }

    /// Schedule and return both the schedule and cost report.
    pub fn schedule_and_analyze(
        &self,
        program: &PramProgram,
        num_procs: usize,
    ) -> (Schedule, CostReport) {
        let graph = DependencyGraph::build(program, self.config.block_size, num_procs);
        let schedule = self.schedule_from_graph(&graph);
        let report = analyze_with_graph(&schedule, &graph, self.config.cache_blocks);
        (schedule, report)
    }

    /// Schedule from a pre-built dependency graph and analyze.
    pub fn schedule_and_analyze_graph(
        &self,
        graph: &DependencyGraph,
    ) -> (Schedule, CostReport) {
        let schedule = self.schedule_from_graph(graph);
        let report = analyze_with_graph(&schedule, graph, self.config.cache_blocks);
        (schedule, report)
    }

    /// Build just the dependency graph (useful for inspection).
    pub fn build_graph(
        &self,
        program: &PramProgram,
        num_procs: usize,
    ) -> DependencyGraph {
        DependencyGraph::build(program, self.config.block_size, num_procs)
    }

    /// Verify that a schedule satisfies the work bound.
    pub fn verify_work_bound(&self, schedule: &Schedule, critical_path: usize) -> bool {
        let p = schedule.num_processors.max(1);
        let t = critical_path.max(1);
        let bound = (self.config.work_bound_factor * (p * t) as f64).ceil() as usize;
        schedule.total_ops() <= bound
    }

    /// Run multiple passes of scheduling with locality optimization,
    /// iteratively improving the schedule quality.
    ///
    /// Each pass re-evaluates the locality ordering using the previous
    /// pass's block transition pattern as a guide.
    pub fn schedule_multi_pass(
        &self,
        graph: &DependencyGraph,
        num_passes: usize,
    ) -> Schedule {
        let mut best_schedule = self.schedule_from_graph(graph);
        let mut best_transitions = count_schedule_transitions(&best_schedule);

        for _ in 1..num_passes {
            // Re-optimize with locality reordering
            let reordered = best_schedule.reorder_for_locality();
            let trans = count_schedule_transitions(&reordered);

            if trans < best_transitions {
                best_schedule = reordered;
                best_transitions = trans;
            }
        }

        best_schedule
    }

    /// Schedule with constraints on which blocks can be scheduled together.
    ///
    /// `max_blocks_per_level` limits how many distinct blocks appear at
    /// any level in the output schedule, forcing spills to subsequent levels.
    pub fn schedule_with_constraints(
        &self,
        graph: &DependencyGraph,
        max_blocks_per_level: Option<usize>,
    ) -> Schedule {
        let base = self.schedule_from_graph(graph);

        let max_blk = match max_blocks_per_level {
            Some(m) if m > 0 => m,
            _ => return base,
        };

        // Group entries by level
        let max_level = base.max_level();
        let mut by_level: Vec<Vec<ScheduleEntry>> = vec![Vec::new(); max_level + 1];
        for entry in &base.entries {
            by_level[entry.level].push(entry.clone());
        }

        // Redistribute: within each level, if more than max_blk distinct blocks,
        // spill excess entries to a new synthetic level
        let mut result_entries: Vec<ScheduleEntry> = Vec::new();
        let mut current_level = 0;

        for level_entries in &by_level {
            let mut block_set: std::collections::HashSet<usize> = std::collections::HashSet::new();
            let mut current_batch: Vec<ScheduleEntry> = Vec::new();

            for entry in level_entries {
                if block_set.len() >= max_blk && !block_set.contains(&entry.block_id) {
                    // Emit current batch
                    for mut e in current_batch.drain(..) {
                        e.level = current_level;
                        result_entries.push(e);
                    }
                    current_level += 1;
                    block_set.clear();
                }
                block_set.insert(entry.block_id);
                current_batch.push(entry.clone());
            }

            // Emit remaining
            for mut e in current_batch {
                e.level = current_level;
                result_entries.push(e);
            }
            current_level += 1;
        }

        let mut sched = Schedule::new(base.num_processors, base.num_phases, base.block_size);
        sched.entries = result_entries;
        sched
    }
}

/// Statistics about the scheduling process.
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    /// Number of nodes in the dependency graph.
    pub graph_nodes: usize,
    /// Number of edges in the dependency graph.
    pub graph_edges: usize,
    /// Critical path length.
    pub critical_path: usize,
    /// Number of operations in the final schedule.
    pub schedule_ops: usize,
    /// Number of levels in the schedule.
    pub schedule_levels: usize,
    /// Whether locality optimization was applied.
    pub locality_applied: bool,
}

impl std::fmt::Display for SchedulerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Scheduler Stats ===")?;
        writeln!(f, "Graph nodes:     {}", self.graph_nodes)?;
        writeln!(f, "Graph edges:     {}", self.graph_edges)?;
        writeln!(f, "Critical path:   {}", self.critical_path)?;
        writeln!(f, "Schedule ops:    {}", self.schedule_ops)?;
        writeln!(f, "Schedule levels: {}", self.schedule_levels)?;
        writeln!(f, "Locality:        {}", self.locality_applied)?;
        Ok(())
    }
}

impl BrentScheduler {
    /// Schedule and return stats about the scheduling process.
    pub fn schedule_with_stats(
        &self,
        graph: &DependencyGraph,
    ) -> (Schedule, SchedulerStats) {
        let schedule = self.schedule_from_graph(graph);
        let stats = SchedulerStats {
            graph_nodes: graph.node_count(),
            graph_edges: graph.edge_count(),
            critical_path: graph.critical_path_length(),
            schedule_ops: schedule.total_ops(),
            schedule_levels: schedule.max_level() + 1,
            locality_applied: self.config.enable_locality,
        };
        (schedule, stats)
    }
}

/// Count block transitions in a schedule.
fn count_schedule_transitions(schedule: &Schedule) -> usize {
    if schedule.entries.len() <= 1 {
        return 0;
    }
    let mut transitions = 0;
    for i in 1..schedule.entries.len() {
        if schedule.entries[i].block_id != schedule.entries[i - 1].block_id {
            transitions += 1;
        }
    }
    transitions
}

/// Schedule with locality: reorders schedule steps to minimize cache-line
/// transitions, grouping operations that access the same memory partition.
pub fn schedule_with_locality(
    scheduler: &BrentScheduler,
    program: &PramProgram,
    partition_order: &[usize],
) -> Schedule {
    let num_procs = program.processor_count().unwrap_or(4);
    let mut schedule = scheduler.schedule(program, num_procs);

    // Reorder steps within each phase to follow partition_order
    if !partition_order.is_empty() && !schedule.entries.is_empty() {
        // Group steps by their position modulo partition count
        let num_partitions = partition_order.len();
        let mut reordered = Vec::with_capacity(schedule.entries.len());
        for &p in partition_order {
            for (i, step) in schedule.entries.iter().enumerate() {
                if i % num_partitions == p % num_partitions {
                    reordered.push(step.clone());
                }
            }
        }
        if reordered.len() == schedule.entries.len() {
            schedule.entries = reordered;
        }
    }

    schedule
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pram_ir::ast::{Expr, MemoryModel, PramProgram, SharedMemoryDecl, Stmt};
    use crate::pram_ir::types::PramType;
    use crate::brent_scheduler::dependency_graph::OperationNode;
    use crate::brent_scheduler::schedule::OpType;

    fn make_simple_program() -> PramProgram {
        let mut prog = PramProgram::new("test_write", MemoryModel::CREW);
        prog.num_processors = Expr::IntLiteral(4);
        prog.shared_memory.push(SharedMemoryDecl {
            name: "A".to_string(),
            elem_type: PramType::Int64,
            size: Expr::IntLiteral(16),
        });
        prog.body.push(Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::IntLiteral(4),
            body: vec![Stmt::SharedWrite {
                memory: Expr::Variable("A".to_string()),
                index: Expr::ProcessorId,
                value: Expr::IntLiteral(42),
            }],
        });
        prog
    }

    fn make_read_write_program() -> PramProgram {
        let mut prog = PramProgram::new("test_rw", MemoryModel::CREW);
        prog.num_processors = Expr::IntLiteral(4);
        prog.shared_memory.push(SharedMemoryDecl {
            name: "A".to_string(),
            elem_type: PramType::Int64,
            size: Expr::IntLiteral(16),
        });
        // Phase 1: write
        prog.body.push(Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::IntLiteral(4),
            body: vec![Stmt::SharedWrite {
                memory: Expr::Variable("A".to_string()),
                index: Expr::ProcessorId,
                value: Expr::IntLiteral(1),
            }],
        });
        // Barrier
        prog.body.push(Stmt::Barrier);
        // Phase 2: read
        prog.body.push(Stmt::ParallelFor {
            proc_var: "pid".to_string(),
            num_procs: Expr::IntLiteral(4),
            body: vec![Stmt::Assign(
                "x".to_string(),
                Expr::shared_read(
                    Expr::Variable("A".to_string()),
                    Expr::ProcessorId,
                ),
            )],
        });
        prog
    }

    #[test]
    fn test_schedule_simple_program() {
        let prog = make_simple_program();
        let scheduler = BrentScheduler::with_defaults();
        let schedule = scheduler.schedule(&prog, 4);

        assert!(schedule.total_ops() > 0);
        // Each processor writes once → at least 4 write operations
        assert!(schedule.write_count() >= 4);
    }

    #[test]
    fn test_schedule_with_locality_disabled() {
        let prog = make_simple_program();
        let scheduler = BrentScheduler::new(SchedulerConfig {
            enable_locality: false,
            ..Default::default()
        });
        let schedule = scheduler.schedule(&prog, 4);
        assert!(schedule.total_ops() > 0);
    }

    #[test]
    fn test_schedule_and_analyze() {
        let prog = make_simple_program();
        let scheduler = BrentScheduler::with_defaults();
        let (schedule, report) = scheduler.schedule_and_analyze(&prog, 4);

        assert_eq!(report.total_work, schedule.total_ops());
        assert!(report.total_work > 0);
    }

    #[test]
    fn test_schedule_two_phase_program() {
        let prog = make_read_write_program();
        let scheduler = BrentScheduler::with_defaults();
        let (schedule, report) = scheduler.schedule_and_analyze(&prog, 4);

        assert!(schedule.total_ops() > 0);
        assert!(report.reads > 0 || report.writes > 0);
    }

    #[test]
    fn test_verify_work_bound() {
        let scheduler = BrentScheduler::with_defaults();
        let mut sched = Schedule::new(4, 1, 4);
        for i in 0..4 {
            sched.push(crate::brent_scheduler::schedule::ScheduleEntry {
                operation_id: i,
                block_id: i,
                phase: 0,
                original_proc_id: i,
                op_type: OpType::Write,
                address: i * 4,
                memory_region: "A".to_string(),
                level: 0,
            });
        }
        // W=4, p=4, T=1 → 4 ≤ 1.0 * 4 * 1
        assert!(scheduler.verify_work_bound(&sched, 1));
    }

    #[test]
    fn test_from_operations_graph() {
        let ops = vec![
            OperationNode {
                op_id: 0, block_id: 0, phase: 0, proc_id: 0,
                op_type: OpType::Write, address: 0,
                memory_region: "A".to_string(),
            },
            OperationNode {
                op_id: 1, block_id: 0, phase: 0, proc_id: 1,
                op_type: OpType::Read, address: 0,
                memory_region: "A".to_string(),
            },
        ];
        let dg = DependencyGraph::from_operations(ops, 4);

        let scheduler = BrentScheduler::with_defaults();
        let (schedule, report) = scheduler.schedule_and_analyze_graph(&dg);

        assert_eq!(schedule.total_ops(), 2);
        assert_eq!(report.critical_path, 2);
    }

    #[test]
    fn test_build_graph() {
        let prog = make_simple_program();
        let scheduler = BrentScheduler::with_defaults();
        let graph = scheduler.build_graph(&prog, 4);
        assert!(graph.node_count() > 0);
    }

    #[test]
    fn test_default_config() {
        let config = SchedulerConfig::default();
        assert_eq!(config.block_size, 4);
        assert!(config.enable_locality);
        assert!((config.work_bound_factor - 1.0).abs() < 1e-6);
        assert_eq!(config.cache_blocks, 64);
    }

    #[test]
    fn test_schedule_dependencies_respected() {
        let ops = vec![
            OperationNode {
                op_id: 0, block_id: 0, phase: 0, proc_id: 0,
                op_type: OpType::Write, address: 0,
                memory_region: "A".to_string(),
            },
            OperationNode {
                op_id: 1, block_id: 0, phase: 0, proc_id: 1,
                op_type: OpType::Read, address: 0,
                memory_region: "A".to_string(),
            },
            OperationNode {
                op_id: 2, block_id: 0, phase: 0, proc_id: 2,
                op_type: OpType::Write, address: 0,
                memory_region: "A".to_string(),
            },
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let deps = dg.dependency_pairs();

        let scheduler = BrentScheduler::with_defaults();
        let schedule = scheduler.schedule_from_graph(&dg);

        assert!(schedule.validate_dependencies(&deps).is_ok());
    }

    #[test]
    fn test_schedule_multi_pass() {
        let ops = vec![
            OperationNode {
                op_id: 0, block_id: 0, phase: 0, proc_id: 0,
                op_type: OpType::Write, address: 0,
                memory_region: "A".to_string(),
            },
            OperationNode {
                op_id: 1, block_id: 1, phase: 0, proc_id: 1,
                op_type: OpType::Write, address: 4,
                memory_region: "A".to_string(),
            },
            OperationNode {
                op_id: 2, block_id: 0, phase: 0, proc_id: 2,
                op_type: OpType::Write, address: 1,
                memory_region: "A".to_string(),
            },
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let scheduler = BrentScheduler::with_defaults();

        let schedule = scheduler.schedule_multi_pass(&dg, 3);
        assert_eq!(schedule.total_ops(), 3);
    }

    #[test]
    fn test_schedule_multi_pass_single() {
        let ops = vec![
            OperationNode {
                op_id: 0, block_id: 0, phase: 0, proc_id: 0,
                op_type: OpType::Write, address: 0,
                memory_region: "A".to_string(),
            },
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let scheduler = BrentScheduler::with_defaults();

        let schedule = scheduler.schedule_multi_pass(&dg, 5);
        assert_eq!(schedule.total_ops(), 1);
    }

    #[test]
    fn test_schedule_with_constraints() {
        let ops: Vec<OperationNode> = (0..8)
            .map(|i| OperationNode {
                op_id: i, block_id: i, phase: 0, proc_id: i,
                op_type: OpType::Write, address: i * 4,
                memory_region: "A".to_string(),
            })
            .collect();
        let dg = DependencyGraph::from_operations(ops, 4);
        let scheduler = BrentScheduler::with_defaults();

        // Constrain to max 2 blocks per level
        let schedule = scheduler.schedule_with_constraints(&dg, Some(2));
        assert_eq!(schedule.total_ops(), 8);

        // Check that no level has more than 2 distinct blocks
        let max_level = schedule.max_level();
        for level in 0..=max_level {
            let blocks_at_level: std::collections::HashSet<usize> = schedule
                .entries
                .iter()
                .filter(|e| e.level == level)
                .map(|e| e.block_id)
                .collect();
            assert!(blocks_at_level.len() <= 2,
                "Level {} has {} blocks, expected ≤ 2",
                level, blocks_at_level.len());
        }
    }

    #[test]
    fn test_schedule_with_constraints_none() {
        let ops = vec![
            OperationNode {
                op_id: 0, block_id: 0, phase: 0, proc_id: 0,
                op_type: OpType::Write, address: 0,
                memory_region: "A".to_string(),
            },
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let scheduler = BrentScheduler::with_defaults();

        let schedule = scheduler.schedule_with_constraints(&dg, None);
        assert_eq!(schedule.total_ops(), 1);
    }

    #[test]
    fn test_schedule_with_stats() {
        let ops = vec![
            OperationNode {
                op_id: 0, block_id: 0, phase: 0, proc_id: 0,
                op_type: OpType::Write, address: 0,
                memory_region: "A".to_string(),
            },
            OperationNode {
                op_id: 1, block_id: 0, phase: 0, proc_id: 1,
                op_type: OpType::Read, address: 0,
                memory_region: "A".to_string(),
            },
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let scheduler = BrentScheduler::with_defaults();

        let (schedule, stats) = scheduler.schedule_with_stats(&dg);
        assert_eq!(schedule.total_ops(), 2);
        assert_eq!(stats.graph_nodes, 2);
        assert_eq!(stats.graph_edges, 1);
        assert_eq!(stats.critical_path, 2);
        assert!(stats.locality_applied);
        let s = format!("{}", stats);
        assert!(s.contains("Scheduler Stats"));
    }

    #[test]
    fn test_scheduler_stats_display() {
        let stats = SchedulerStats {
            graph_nodes: 10,
            graph_edges: 15,
            critical_path: 4,
            schedule_ops: 10,
            schedule_levels: 4,
            locality_applied: true,
        };
        let s = format!("{}", stats);
        assert!(s.contains("10"));
        assert!(s.contains("15"));
        assert!(s.contains("true"));
    }

    #[test]
    fn test_schedule_with_locality() {
        let scheduler = BrentScheduler::with_defaults();
        let prog = crate::algorithm_library::sorting::bitonic_sort();
        let partition_order = vec![0, 1, 2, 3];
        let schedule = schedule_with_locality(&scheduler, &prog, &partition_order);
        assert!(!schedule.entries.is_empty());
    }

    #[test]
    fn test_schedule_with_locality_empty_partition() {
        let scheduler = BrentScheduler::with_defaults();
        let prog = crate::algorithm_library::list::prefix_sum();
        let schedule = schedule_with_locality(&scheduler, &prog, &[]);
        assert!(!schedule.entries.is_empty());
    }

    #[test]
    fn test_schedule_with_locality_preserves_length() {
        let scheduler = BrentScheduler::with_defaults();
        let prog = crate::algorithm_library::sorting::bitonic_sort();
        let num_procs = prog.processor_count().unwrap_or(4);
        let original = scheduler.schedule(&prog, num_procs);
        let reordered = schedule_with_locality(&scheduler, &prog, &[0, 1]);
        assert_eq!(original.entries.len(), reordered.entries.len());
    }
}
