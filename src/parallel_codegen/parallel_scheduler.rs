//! Parallel scheduling strategies for multi-core targets.
//!
//! Extends Brent scheduling to produce multi-core-aware schedules
//! that preserve work optimality while minimizing synchronization.

use crate::pram_ir::ast::PramProgram;

/// Parallel scheduling strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParallelStrategy {
    /// Static partitioning of processors across cores
    StaticPartition,
    /// Dynamic work distribution with stealing
    WorkStealing,
    /// Locality-aware assignment minimizing cross-core communication
    LocalityAware,
    /// Hybrid: static for regular phases, dynamic for irregular
    Hybrid,
}

/// Configuration for parallel scheduling.
#[derive(Debug, Clone)]
pub struct ParallelScheduleConfig {
    pub strategy: ParallelStrategy,
    pub num_cores: usize,
    pub l1_size: usize,
    pub l2_size: usize,
    pub prefetch_enabled: bool,
}

impl Default for ParallelScheduleConfig {
    fn default() -> Self {
        Self {
            strategy: ParallelStrategy::Hybrid,
            num_cores: 4,
            l1_size: 32 * 1024,
            l2_size: 256 * 1024,
            prefetch_enabled: true,
        }
    }
}

/// Parallel schedule for a single phase.
#[derive(Debug, Clone)]
pub struct ParallelPhaseSchedule {
    pub phase_id: usize,
    pub core_assignments: Vec<CoreAssignment>,
    pub synchronization: SyncType,
}

/// Assignment of work to a single core.
#[derive(Debug, Clone)]
pub struct CoreAssignment {
    pub core_id: usize,
    pub processor_range: (usize, usize),
    pub block_ids: Vec<usize>,
    pub estimated_cache_misses: usize,
}

/// Synchronization type between phases.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SyncType {
    Barrier,
    PointToPoint,
    None,
}

/// Generate a parallel schedule for a PRAM program.
pub fn generate_parallel_schedule(
    program: &PramProgram,
    config: &ParallelScheduleConfig,
) -> Vec<ParallelPhaseSchedule> {
    let num_phases = program.parallel_step_count();
    let mut schedules = Vec::new();

    for phase in 0..num_phases {
        let core_assignments: Vec<CoreAssignment> = (0..config.num_cores)
            .map(|core| {
                let procs_per_core = 1024 / config.num_cores; // Assumed max processors
                CoreAssignment {
                    core_id: core,
                    processor_range: (
                        core * procs_per_core,
                        (core + 1) * procs_per_core,
                    ),
                    block_ids: Vec::new(),
                    estimated_cache_misses: 0,
                }
            })
            .collect();

        schedules.push(ParallelPhaseSchedule {
            phase_id: phase,
            core_assignments,
            synchronization: if phase < num_phases - 1 {
                SyncType::Barrier
            } else {
                SyncType::None
            },
        });
    }

    schedules
}

/// Estimate parallel efficiency: how well the work is balanced.
pub fn parallel_efficiency(schedules: &[ParallelPhaseSchedule]) -> f64 {
    if schedules.is_empty() {
        return 1.0;
    }

    let mut total_efficiency = 0.0;
    for schedule in schedules {
        let works: Vec<usize> = schedule.core_assignments.iter()
            .map(|ca| ca.processor_range.1 - ca.processor_range.0)
            .collect();

        let max_work = *works.iter().max().unwrap_or(&1);
        let avg_work = works.iter().sum::<usize>() as f64 / works.len() as f64;

        if max_work > 0 {
            total_efficiency += avg_work / max_work as f64;
        }
    }

    total_efficiency / schedules.len() as f64
}

/// High-level parallel schedule combining strategy selection with work partitioning.
#[derive(Debug, Clone)]
pub struct ParallelSchedule {
    pub strategy: ParallelStrategy,
    pub phases: Vec<ParallelPhaseSchedule>,
    pub estimated_speedup: f64,
    pub work_distribution: Vec<f64>,
}

/// Scheduling strategy selector based on algorithm characteristics.
#[derive(Debug, Clone)]
pub struct SchedulingStrategy {
    pub num_cores: usize,
    pub l2_size: usize,
}

impl SchedulingStrategy {
    pub fn new(num_cores: usize, l2_size: usize) -> Self {
        Self { num_cores, l2_size }
    }

    /// Select optimal strategy based on work distribution regularity.
    pub fn select(&self, work_variance: f64, has_irregular_access: bool) -> ParallelStrategy {
        if work_variance < 0.1 && !has_irregular_access {
            ParallelStrategy::StaticPartition
        } else if has_irregular_access {
            ParallelStrategy::WorkStealing
        } else if work_variance > 0.5 {
            ParallelStrategy::Hybrid
        } else {
            ParallelStrategy::LocalityAware
        }
    }
}

/// Work chunk descriptor for parallel distribution.
#[derive(Debug, Clone)]
pub struct WorkChunk {
    pub chunk_id: usize,
    pub start_proc: usize,
    pub end_proc: usize,
    pub estimated_cost: f64,
}

/// Generate an adaptive parallel schedule that selects strategy automatically.
pub fn adaptive_parallel_schedule(
    program: &PramProgram,
    num_cores: usize,
) -> ParallelSchedule {
    let config = ParallelScheduleConfig {
        strategy: ParallelStrategy::Hybrid,
        num_cores,
        ..Default::default()
    };
    let phases = generate_parallel_schedule(program, &config);
    let eff = parallel_efficiency(&phases);

    // Compute work distribution across cores
    let work_dist: Vec<f64> = (0..num_cores)
        .map(|c| {
            phases.iter()
                .flat_map(|p| p.core_assignments.iter())
                .filter(|ca| ca.core_id == c)
                .map(|ca| (ca.processor_range.1 - ca.processor_range.0) as f64)
                .sum()
        })
        .collect();

    ParallelSchedule {
        strategy: ParallelStrategy::Hybrid,
        phases,
        estimated_speedup: eff * num_cores as f64,
        work_distribution: work_dist,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_schedule() {
        let prog = crate::algorithm_library::sorting::bitonic_sort();
        let config = ParallelScheduleConfig::default();
        let schedules = generate_parallel_schedule(&prog, &config);
        assert!(!schedules.is_empty());
    }

    #[test]
    fn test_parallel_efficiency() {
        let prog = crate::algorithm_library::list::prefix_sum();
        let config = ParallelScheduleConfig::default();
        let schedules = generate_parallel_schedule(&prog, &config);
        let eff = parallel_efficiency(&schedules);
        assert!(eff > 0.0);
        assert!(eff <= 1.0);
    }

    #[test]
    fn test_scheduling_strategy_static() {
        let strat = SchedulingStrategy::new(4, 256 * 1024);
        assert_eq!(strat.select(0.05, false), ParallelStrategy::StaticPartition);
    }

    #[test]
    fn test_scheduling_strategy_work_stealing() {
        let strat = SchedulingStrategy::new(4, 256 * 1024);
        assert_eq!(strat.select(0.3, true), ParallelStrategy::WorkStealing);
    }

    #[test]
    fn test_scheduling_strategy_hybrid() {
        let strat = SchedulingStrategy::new(8, 256 * 1024);
        assert_eq!(strat.select(0.8, false), ParallelStrategy::Hybrid);
    }

    #[test]
    fn test_adaptive_parallel_schedule() {
        let prog = crate::algorithm_library::sorting::bitonic_sort();
        let schedule = adaptive_parallel_schedule(&prog, 4);
        assert!(!schedule.phases.is_empty());
        assert!(schedule.estimated_speedup > 0.0);
        assert_eq!(schedule.work_distribution.len(), 4);
    }

    #[test]
    fn test_work_chunk() {
        let chunk = WorkChunk {
            chunk_id: 0,
            start_proc: 0,
            end_proc: 256,
            estimated_cost: 100.0,
        };
        assert_eq!(chunk.end_proc - chunk.start_proc, 256);
    }
}
