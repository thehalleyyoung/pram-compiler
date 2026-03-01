//! Work-stealing scheduler for multi-core PRAM execution.
//!
//! Implements a Chase-Lev work-stealing deque for dynamic load balancing
//! of PRAM parallel steps across hardware threads.

use std::collections::VecDeque;

/// A task in the work-stealing scheduler.
#[derive(Debug, Clone)]
pub struct Task {
    pub id: usize,
    pub phase: usize,
    pub processor_range: (usize, usize),
    pub block_id: usize,
    pub estimated_work: usize,
}

/// Work-stealing deque for a single worker thread.
#[derive(Debug)]
pub struct WorkStealingDeque {
    tasks: VecDeque<Task>,
    worker_id: usize,
}

impl WorkStealingDeque {
    pub fn new(worker_id: usize) -> Self {
        Self {
            tasks: VecDeque::new(),
            worker_id,
        }
    }

    /// Push a task to the bottom of the deque (local push).
    pub fn push(&mut self, task: Task) {
        self.tasks.push_back(task);
    }

    /// Pop a task from the bottom (local pop - LIFO for locality).
    pub fn pop(&mut self) -> Option<Task> {
        self.tasks.pop_back()
    }

    /// Steal a task from the top (remote steal - FIFO for load balance).
    pub fn steal(&mut self) -> Option<Task> {
        self.tasks.pop_front()
    }

    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tasks.len()
    }
}

/// Multi-core schedule generated from PRAM parallel steps.
#[derive(Debug, Clone)]
pub struct MultiCoreSchedule {
    pub worker_count: usize,
    pub phases: Vec<PhaseSchedule>,
    pub total_tasks: usize,
    pub total_work: usize,
}

/// Schedule for a single phase across workers.
#[derive(Debug, Clone)]
pub struct PhaseSchedule {
    pub phase_id: usize,
    pub worker_tasks: Vec<Vec<Task>>,
    pub barrier_after: bool,
}

/// Generate a multi-core schedule from PRAM program parameters.
pub fn schedule_multicore(
    num_pram_processors: usize,
    num_phases: usize,
    num_workers: usize,
    block_assignments: &[(usize, usize)], // (address, block_id)
) -> MultiCoreSchedule {
    let mut phases = Vec::new();
    let mut total_tasks = 0;
    let mut total_work = 0;

    let procs_per_worker = (num_pram_processors + num_workers - 1) / num_workers;

    for phase in 0..num_phases {
        let mut worker_tasks: Vec<Vec<Task>> = vec![Vec::new(); num_workers];

        for worker in 0..num_workers {
            let start = worker * procs_per_worker;
            let end = ((worker + 1) * procs_per_worker).min(num_pram_processors);

            if start < end {
                // Group by block for locality
                let mut block_groups: std::collections::HashMap<usize, Vec<usize>> =
                    std::collections::HashMap::new();

                for proc_id in start..end {
                    if proc_id < block_assignments.len() {
                        let block = block_assignments[proc_id].1;
                        block_groups.entry(block).or_default().push(proc_id);
                    }
                }

                for (block_id, procs) in block_groups {
                    let task = Task {
                        id: total_tasks,
                        phase,
                        processor_range: (*procs.first().unwrap(), *procs.last().unwrap() + 1),
                        block_id,
                        estimated_work: procs.len(),
                    };
                    total_work += procs.len();
                    total_tasks += 1;
                    worker_tasks[worker].push(task);
                }
            }
        }

        phases.push(PhaseSchedule {
            phase_id: phase,
            worker_tasks,
            barrier_after: true,
        });
    }

    MultiCoreSchedule {
        worker_count: num_workers,
        phases,
        total_tasks,
        total_work,
    }
}

/// Emit C code for a work-stealing scheduler.
pub fn emit_work_stealing_c(schedule: &MultiCoreSchedule) -> String {
    let mut code = String::new();

    code.push_str("/* Work-stealing scheduler for multi-core execution */\n");
    code.push_str("#include <omp.h>\n\n");

    code.push_str(&format!(
        "void execute_parallel(int num_workers) {{\n"
    ));
    code.push_str(&format!("    omp_set_num_threads({});\n\n", schedule.worker_count));

    for phase in &schedule.phases {
        code.push_str(&format!("    /* Phase {} */\n", phase.phase_id));
        code.push_str("    #pragma omp parallel\n");
        code.push_str("    {\n");
        code.push_str("        int tid = omp_get_thread_num();\n");

        // Each worker processes its assigned tasks
        code.push_str("        #pragma omp for schedule(dynamic, 1) nowait\n");
        let total_tasks: usize = phase.worker_tasks.iter().map(|w| w.len()).sum();
        code.push_str(&format!("        for (int t = 0; t < {}; t++) {{\n", total_tasks));
        code.push_str("            /* Process task t */\n");
        code.push_str("        }\n");

        if phase.barrier_after {
            code.push_str("        #pragma omp barrier\n");
        }
        code.push_str("    }\n\n");
    }

    code.push_str("}\n");
    code
}

/// Calculate theoretical speedup for the multi-core schedule.
pub fn theoretical_speedup(schedule: &MultiCoreSchedule) -> f64 {
    if schedule.phases.is_empty() {
        return 1.0;
    }

    let sequential_work = schedule.total_work as f64;
    let mut parallel_time = 0.0;

    for phase in &schedule.phases {
        let max_worker_work: f64 = phase.worker_tasks.iter()
            .map(|tasks| tasks.iter().map(|t| t.estimated_work).sum::<usize>() as f64)
            .fold(0.0f64, f64::max);
        parallel_time += max_worker_work;
    }

    if parallel_time > 0.0 {
        sequential_work / parallel_time
    } else {
        1.0
    }
}

/// Simulator for work-stealing behavior analysis.
#[derive(Debug)]
pub struct WorkStealingSimulator {
    pub num_workers: usize,
    pub steal_latency: f64,
    pub local_latency: f64,
}

impl WorkStealingSimulator {
    pub fn new(num_workers: usize) -> Self {
        Self {
            num_workers,
            steal_latency: 10.0,  // cycles
            local_latency: 1.0,   // cycle
        }
    }

    /// Simulate work-stealing execution and return (total_time, steal_count).
    pub fn simulate(&self, tasks: &[usize]) -> (f64, usize) {
        if tasks.is_empty() || self.num_workers == 0 {
            return (0.0, 0);
        }

        let mut worker_loads = vec![0.0f64; self.num_workers];
        let mut steal_count = 0;

        // Initial distribution
        for (i, &work) in tasks.iter().enumerate() {
            worker_loads[i % self.num_workers] += work as f64 * self.local_latency;
        }

        // Simulate stealing: overloaded workers share with underloaded
        let avg_load: f64 = worker_loads.iter().sum::<f64>() / self.num_workers as f64;
        for i in 0..self.num_workers {
            if worker_loads[i] > avg_load * 1.5 {
                // Find least loaded worker
                let min_idx = worker_loads.iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                if min_idx != i {
                    let transfer = (worker_loads[i] - avg_load) / 2.0;
                    worker_loads[i] -= transfer;
                    worker_loads[min_idx] += transfer + self.steal_latency;
                    steal_count += 1;
                }
            }
        }

        let total_time = worker_loads.iter().cloned().fold(0.0f64, f64::max);
        (total_time, steal_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_stealing_deque() {
        let mut deque = WorkStealingDeque::new(0);
        deque.push(Task {
            id: 0, phase: 0, processor_range: (0, 10),
            block_id: 0, estimated_work: 10,
        });
        deque.push(Task {
            id: 1, phase: 0, processor_range: (10, 20),
            block_id: 1, estimated_work: 10,
        });

        assert_eq!(deque.len(), 2);

        // Local pop (LIFO)
        let t = deque.pop().unwrap();
        assert_eq!(t.id, 1);

        // Remote steal (FIFO)
        let t = deque.steal().unwrap();
        assert_eq!(t.id, 0);

        assert!(deque.is_empty());
    }

    #[test]
    fn test_schedule_multicore() {
        let blocks: Vec<(usize, usize)> = (0..100).map(|i| (i, i % 10)).collect();
        let schedule = schedule_multicore(100, 5, 4, &blocks);
        assert_eq!(schedule.worker_count, 4);
        assert_eq!(schedule.phases.len(), 5);
        assert!(schedule.total_work > 0);
    }

    #[test]
    fn test_theoretical_speedup() {
        let blocks: Vec<(usize, usize)> = (0..100).map(|i| (i, i % 10)).collect();
        let schedule = schedule_multicore(100, 5, 4, &blocks);
        let speedup = theoretical_speedup(&schedule);
        assert!(speedup >= 1.0);
        assert!(speedup <= 4.0);
    }

    #[test]
    fn test_emit_work_stealing_c() {
        let blocks: Vec<(usize, usize)> = (0..20).map(|i| (i, i % 5)).collect();
        let schedule = schedule_multicore(20, 2, 2, &blocks);
        let code = emit_work_stealing_c(&schedule);
        assert!(code.contains("omp"));
        assert!(code.contains("Phase 0"));
    }

    #[test]
    fn test_work_stealing_simulator_balanced() {
        let sim = WorkStealingSimulator::new(4);
        let tasks = vec![100, 100, 100, 100];
        let (time, steals) = sim.simulate(&tasks);
        assert!(time > 0.0);
        assert_eq!(steals, 0); // Balanced, no stealing needed
    }

    #[test]
    fn test_work_stealing_simulator_imbalanced() {
        let sim = WorkStealingSimulator::new(4);
        let tasks = vec![1000, 10, 10, 10];
        let (time, steals) = sim.simulate(&tasks);
        assert!(time > 0.0);
        assert!(steals > 0); // Imbalanced, should steal
    }

    #[test]
    fn test_work_stealing_simulator_empty() {
        let sim = WorkStealingSimulator::new(4);
        let (time, steals) = sim.simulate(&[]);
        assert_eq!(time, 0.0);
        assert_eq!(steals, 0);
    }

    #[test]
    fn test_work_stealing_simulator_single_worker() {
        let sim = WorkStealingSimulator::new(1);
        let tasks = vec![100, 200, 300];
        let (time, steals) = sim.simulate(&tasks);
        assert!(time > 0.0);
        assert_eq!(steals, 0); // Can't steal from yourself
    }
}
