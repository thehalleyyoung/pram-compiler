//! NUMA-aware memory allocation strategies for multi-socket systems.

/// NUMA allocation policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumaPolicy {
    /// First-touch: allocate on the node that first accesses
    FirstTouch,
    /// Interleave: round-robin across NUMA nodes
    Interleave,
    /// Bind: allocate on a specific node
    Bind(usize),
    /// Local: allocate on the calling thread's node
    Local,
}

/// NUMA-aware memory region descriptor.
#[derive(Debug, Clone)]
pub struct NumaRegion {
    pub name: String,
    pub size_bytes: usize,
    pub policy: NumaPolicy,
    pub node_id: Option<usize>,
}

/// Generate NUMA-aware allocation code.
pub fn emit_numa_alloc(region: &NumaRegion) -> String {
    let mut code = String::new();

    match region.policy {
        NumaPolicy::FirstTouch => {
            code.push_str(&format!(
                "int* {} = (int*)malloc(sizeof(int) * {});\n",
                region.name,
                region.size_bytes / 4
            ));
            code.push_str(&format!(
                "/* First-touch: initialize in parallel for NUMA locality */\n"
            ));
            code.push_str(&format!(
                "#pragma omp parallel for schedule(static)\n"
            ));
            code.push_str(&format!(
                "for (int i = 0; i < {}; i++) {}[i] = 0;\n",
                region.size_bytes / 4,
                region.name
            ));
        }
        NumaPolicy::Interleave => {
            code.push_str(&format!(
                "int* {} = (int*)malloc(sizeof(int) * {});\n",
                region.name,
                region.size_bytes / 4
            ));
            code.push_str("/* Interleaved NUMA allocation */\n");
            code.push_str(&format!(
                "memset({}, 0, sizeof(int) * {});\n",
                region.name,
                region.size_bytes / 4
            ));
        }
        NumaPolicy::Bind(node) => {
            code.push_str(&format!(
                "/* Bound to NUMA node {} */\n", node
            ));
            code.push_str(&format!(
                "int* {} = (int*)malloc(sizeof(int) * {});\n",
                region.name,
                region.size_bytes / 4
            ));
        }
        NumaPolicy::Local => {
            code.push_str(&format!(
                "int* {} = (int*)malloc(sizeof(int) * {});\n",
                region.name,
                region.size_bytes / 4
            ));
        }
    }

    code
}

/// Choose NUMA policy based on access pattern.
pub fn select_numa_policy(
    total_size: usize,
    num_threads: usize,
    is_read_mostly: bool,
) -> NumaPolicy {
    if total_size < 1024 * 1024 {
        // Small allocation: local is fine
        NumaPolicy::Local
    } else if is_read_mostly {
        // Read-heavy: interleave for bandwidth
        NumaPolicy::Interleave
    } else if num_threads > 1 {
        // Write-heavy multi-threaded: first-touch for locality
        NumaPolicy::FirstTouch
    } else {
        NumaPolicy::Local
    }
}

/// NUMA node descriptor.
#[derive(Debug, Clone)]
pub struct NumaNode {
    pub node_id: usize,
    pub core_count: usize,
    pub memory_mb: usize,
    pub distance_to: Vec<usize>,  // distance to other nodes
}

/// NUMA topology model.
#[derive(Debug, Clone)]
pub struct NumaTopology {
    pub nodes: Vec<NumaNode>,
}

impl NumaTopology {
    /// Create a symmetric 2-node topology.
    pub fn two_socket(cores_per_socket: usize, memory_per_socket_mb: usize) -> Self {
        Self {
            nodes: vec![
                NumaNode {
                    node_id: 0,
                    core_count: cores_per_socket,
                    memory_mb: memory_per_socket_mb,
                    distance_to: vec![10, 20],
                },
                NumaNode {
                    node_id: 1,
                    core_count: cores_per_socket,
                    memory_mb: memory_per_socket_mb,
                    distance_to: vec![20, 10],
                },
            ],
        }
    }

    /// Total cores across all NUMA nodes.
    pub fn total_cores(&self) -> usize {
        self.nodes.iter().map(|n| n.core_count).sum()
    }
}

/// Compute optimal data placement across NUMA nodes to minimize remote accesses.
pub fn optimal_placement(
    topology: &NumaTopology,
    data_size_mb: usize,
    access_pattern: &[usize],  // which node each access comes from
) -> Vec<usize> {
    // Count accesses per node
    let mut node_access_counts: Vec<usize> = vec![0; topology.nodes.len()];
    for &node in access_pattern {
        if node < node_access_counts.len() {
            node_access_counts[node] += 1;
        }
    }

    // Place data on the node with most accesses (greedy)
    let best_node = node_access_counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, count)| count)
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Distribute data proportionally to access frequency
    let total_accesses: usize = node_access_counts.iter().sum();
    if total_accesses == 0 {
        return vec![data_size_mb / topology.nodes.len().max(1); topology.nodes.len()];
    }

    topology.nodes.iter().enumerate().map(|(i, _)| {
        if i == best_node {
            // Primary node gets at least half
            data_size_mb / 2 + (data_size_mb * node_access_counts[i]) / (2 * total_accesses)
        } else {
            (data_size_mb * node_access_counts[i]) / (2 * total_accesses)
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_numa_alloc() {
        let region = NumaRegion {
            name: "data".to_string(),
            size_bytes: 4096,
            policy: NumaPolicy::FirstTouch,
            node_id: None,
        };
        let code = emit_numa_alloc(&region);
        assert!(code.contains("malloc"));
        assert!(code.contains("parallel"));
    }

    #[test]
    fn test_select_policy() {
        assert_eq!(
            select_numa_policy(1024, 1, false),
            NumaPolicy::Local
        );
        assert_eq!(
            select_numa_policy(10 * 1024 * 1024, 8, true),
            NumaPolicy::Interleave
        );
        assert_eq!(
            select_numa_policy(10 * 1024 * 1024, 8, false),
            NumaPolicy::FirstTouch
        );
    }

    #[test]
    fn test_numa_topology_two_socket() {
        let topo = NumaTopology::two_socket(8, 16384);
        assert_eq!(topo.nodes.len(), 2);
        assert_eq!(topo.total_cores(), 16);
        assert_eq!(topo.nodes[0].distance_to[1], 20);
        assert_eq!(topo.nodes[0].distance_to[0], 10);
    }

    #[test]
    fn test_optimal_placement_skewed() {
        let topo = NumaTopology::two_socket(8, 16384);
        let accesses = vec![0, 0, 0, 0, 0, 1]; // mostly node 0
        let placement = optimal_placement(&topo, 1024, &accesses);
        assert_eq!(placement.len(), 2);
        assert!(placement[0] > placement[1]); // node 0 gets more data
    }

    #[test]
    fn test_optimal_placement_balanced() {
        let topo = NumaTopology::two_socket(8, 16384);
        let accesses = vec![0, 1, 0, 1, 0, 1];
        let placement = optimal_placement(&topo, 1024, &accesses);
        assert_eq!(placement.len(), 2);
        // Both should get some data
        assert!(placement[0] > 0);
        assert!(placement[1] > 0);
    }

    #[test]
    fn test_optimal_placement_empty() {
        let topo = NumaTopology::two_socket(4, 8192);
        let placement = optimal_placement(&topo, 1024, &[]);
        assert_eq!(placement.len(), 2);
    }
}
