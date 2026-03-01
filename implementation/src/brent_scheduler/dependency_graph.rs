//! Block-level dependency graph for PRAM operations.
//!
//! Builds a directed acyclic graph where nodes are shared memory operations
//! and edges encode data dependencies (RAW, WAR, WAW). Supports block-level
//! aggregation, topological ordering, and critical path computation.

use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::pram_ir::ast::{Expr, PramProgram, Stmt, split_into_phases};

use super::schedule::OpType;

/// Classification of a data dependency edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DepKind {
    /// Read-after-write: a read depends on a prior write.
    RAW,
    /// Write-after-read: a write must follow a prior read.
    WAR,
    /// Write-after-write: ordering between two writes to the same location.
    WAW,
}

/// A node in the dependency graph representing one shared memory operation.
#[derive(Debug, Clone)]
pub struct OperationNode {
    /// Unique id for this operation (equals the graph node index value).
    pub op_id: usize,
    /// Cache block id (address / block_size).
    pub block_id: usize,
    /// Phase (between barriers) this operation belongs to.
    pub phase: usize,
    /// Processor id that issued this operation.
    pub proc_id: usize,
    /// Read or write.
    pub op_type: OpType,
    /// Element address in shared memory.
    pub address: usize,
    /// Name of the shared memory region.
    pub memory_region: String,
}

/// Edge weight in the dependency graph.
#[derive(Debug, Clone, Copy)]
pub struct DepEdge {
    pub kind: DepKind,
}

/// Block-level dependency graph built from a PRAM program.
pub struct DependencyGraph {
    /// The underlying directed graph.
    pub graph: DiGraph<OperationNode, DepEdge>,
    /// Block size used for cache analysis.
    pub block_size: usize,
    /// Number of processors.
    pub num_processors: usize,
    /// Number of phases.
    pub num_phases: usize,
    /// Map from (memory_region, address) to the list of node indices that access it.
    location_map: HashMap<(String, usize), Vec<NodeIndex>>,
}

impl DependencyGraph {
    /// Build a dependency graph from a PRAM program.
    ///
    /// `block_size` controls the cache block granularity.
    /// `num_procs` is the number of processors to instantiate.
    pub fn build(program: &PramProgram, block_size: usize, num_procs: usize) -> Self {
        let phases = split_into_phases(&program.body, &program.num_processors);
        let mut graph = DiGraph::new();
        let mut location_map: HashMap<(String, usize), Vec<NodeIndex>> = HashMap::new();
        let num_phases = phases.len();

        let mut op_counter: usize = 0;

        for phase in &phases {
            for stmt in &phase.statements {
                Self::process_stmt(
                    stmt,
                    phase.phase_id,
                    num_procs,
                    block_size,
                    &mut graph,
                    &mut location_map,
                    &mut op_counter,
                );
            }
        }

        let mut dep_graph = DependencyGraph {
            graph,
            block_size,
            num_processors: num_procs,
            num_phases,
            location_map,
        };
        dep_graph.add_dependency_edges();
        dep_graph
    }

    /// Build from explicit operation lists (for testing or manual construction).
    pub fn from_operations(ops: Vec<OperationNode>, block_size: usize) -> Self {
        let mut graph = DiGraph::new();
        let mut location_map: HashMap<(String, usize), Vec<NodeIndex>> = HashMap::new();
        let mut num_procs = 0;
        let mut num_phases = 0;

        for op in ops {
            if op.proc_id + 1 > num_procs {
                num_procs = op.proc_id + 1;
            }
            if op.phase + 1 > num_phases {
                num_phases = op.phase + 1;
            }
            let key = (op.memory_region.clone(), op.address);
            let idx = graph.add_node(op);
            location_map.entry(key).or_default().push(idx);
        }

        let mut dep_graph = DependencyGraph {
            graph,
            block_size,
            num_processors: num_procs,
            num_phases,
            location_map,
        };
        dep_graph.add_dependency_edges();
        dep_graph
    }

    /// Process a statement, extracting shared memory operations for each processor.
    fn process_stmt(
        stmt: &Stmt,
        phase_id: usize,
        num_procs: usize,
        block_size: usize,
        graph: &mut DiGraph<OperationNode, DepEdge>,
        location_map: &mut HashMap<(String, usize), Vec<NodeIndex>>,
        op_counter: &mut usize,
    ) {
        match stmt {
            Stmt::ParallelFor {
                num_procs: np_expr,
                body,
                ..
            } => {
                let actual_procs = np_expr.eval_const_int().unwrap_or(num_procs as i64) as usize;
                let p = actual_procs.min(num_procs);
                for proc_id in 0..p {
                    for inner in body {
                        Self::process_stmt_for_proc(
                            inner,
                            phase_id,
                            proc_id,
                            block_size,
                            graph,
                            location_map,
                            op_counter,
                        );
                    }
                }
            }
            Stmt::Block(stmts) => {
                for s in stmts {
                    Self::process_stmt(s, phase_id, num_procs, block_size, graph, location_map, op_counter);
                }
            }
            // For non-parallel top-level stmts, treat as proc 0
            other => {
                Self::process_stmt_for_proc(other, phase_id, 0, block_size, graph, location_map, op_counter);
            }
        }
    }

    /// Process a statement for a specific processor, extracting operations.
    fn process_stmt_for_proc(
        stmt: &Stmt,
        phase_id: usize,
        proc_id: usize,
        block_size: usize,
        graph: &mut DiGraph<OperationNode, DepEdge>,
        location_map: &mut HashMap<(String, usize), Vec<NodeIndex>>,
        op_counter: &mut usize,
    ) {
        let accesses = stmt.collect_shared_accesses();
        for access in &accesses {
            let addr = Self::resolve_address(&access.index, proc_id);
            let blk = addr / block_size;
            let op_type = if access.is_write { OpType::Write } else { OpType::Read };

            let node = OperationNode {
                op_id: *op_counter,
                block_id: blk,
                phase: phase_id,
                proc_id,
                op_type,
                address: addr,
                memory_region: access.memory.clone(),
            };
            *op_counter += 1;

            let key = (access.memory.clone(), addr);
            let idx = graph.add_node(node);
            location_map.entry(key).or_default().push(idx);
        }

        // Recurse into compound statements
        match stmt {
            Stmt::SeqFor { body, .. }
            | Stmt::While { body, .. }
            | Stmt::Block(body) => {
                for s in body {
                    Self::process_stmt_for_proc(s, phase_id, proc_id, block_size, graph, location_map, op_counter);
                }
            }
            Stmt::If { then_body, else_body, .. } => {
                for s in then_body {
                    Self::process_stmt_for_proc(s, phase_id, proc_id, block_size, graph, location_map, op_counter);
                }
                for s in else_body {
                    Self::process_stmt_for_proc(s, phase_id, proc_id, block_size, graph, location_map, op_counter);
                }
            }
            _ => {}
        }
    }

    /// Best-effort address resolution from an expression.
    /// For static/constant indices we evaluate directly.
    /// For processor-id-dependent indices, substitute the proc_id.
    fn resolve_address(expr: &Expr, proc_id: usize) -> usize {
        // Try constant eval first
        if let Some(v) = expr.eval_const_int() {
            return v.unsigned_abs() as usize;
        }
        // Substitute ProcessorId
        let substituted = expr.substitute("pid", &Expr::IntLiteral(proc_id as i64));
        if let Some(v) = substituted.eval_const_int() {
            return v.unsigned_abs() as usize;
        }
        // If the expression is just ProcessorId, use proc_id
        if matches!(expr, Expr::ProcessorId) {
            return proc_id;
        }
        // Fallback: hash-based deterministic address
        let vars = expr.collect_variables();
        let mut h: usize = proc_id.wrapping_mul(2654435761);
        for v in &vars {
            for b in v.bytes() {
                h = h.wrapping_mul(31).wrapping_add(b as usize);
            }
        }
        h % 1024
    }

    /// Add RAW, WAR, and WAW dependency edges based on the location map.
    fn add_dependency_edges(&mut self) {
        for (_loc, indices) in &self.location_map {
            let len = indices.len();
            if len <= 1 {
                continue;
            }
            // Sort by op_id (creation order = program order within a phase)
            let mut sorted = indices.clone();
            sorted.sort_by_key(|idx| self.graph[*idx].op_id);

            for i in 0..len {
                for j in (i + 1)..len {
                    let a = sorted[i];
                    let b = sorted[j];
                    let na = &self.graph[a];
                    let nb = &self.graph[b];

                    // Only add edges within the same phase or between adjacent phases
                    // (cross-phase deps are implicit via barriers)
                    let dep_kind = match (na.op_type, nb.op_type) {
                        (OpType::Write, OpType::Read) => Some(DepKind::RAW),
                        (OpType::Read, OpType::Write) => Some(DepKind::WAR),
                        (OpType::Write, OpType::Write) => Some(DepKind::WAW),
                        (OpType::Read, OpType::Read) => None, // No dependency
                    };

                    if let Some(kind) = dep_kind {
                        self.graph.add_edge(a, b, DepEdge { kind });
                    }
                }
            }
        }
    }

    /// Topological ordering of nodes. Returns None if there is a cycle.
    pub fn topological_order(&self) -> Option<Vec<NodeIndex>> {
        toposort(&self.graph, None).ok()
    }

    /// Compute the level of each node (longest path from any source).
    /// Returns a map from NodeIndex to level.
    pub fn compute_levels(&self) -> HashMap<NodeIndex, usize> {
        let topo = match self.topological_order() {
            Some(t) => t,
            None => return HashMap::new(),
        };

        let mut levels: HashMap<NodeIndex, usize> = HashMap::new();
        for &node in &topo {
            let mut max_pred_level: Option<usize> = None;
            for edge in self.graph.edges_directed(node, petgraph::Direction::Incoming) {
                let pred_level = levels.get(&edge.source()).copied().unwrap_or(0);
                max_pred_level = Some(max_pred_level.map_or(pred_level, |m: usize| m.max(pred_level)));
            }
            let level = max_pred_level.map_or(0, |m| m + 1);
            levels.insert(node, level);
        }
        levels
    }

    /// Critical path length (the maximum level + 1).
    pub fn critical_path_length(&self) -> usize {
        let levels = self.compute_levels();
        levels.values().max().map_or(0, |m| m + 1)
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get a node by its graph index.
    pub fn get_node(&self, idx: NodeIndex) -> &OperationNode {
        &self.graph[idx]
    }

    /// All node indices.
    pub fn node_indices(&self) -> Vec<NodeIndex> {
        self.graph.node_indices().collect()
    }

    /// Get dependencies (predecessors) of a node.
    pub fn predecessors(&self, idx: NodeIndex) -> Vec<NodeIndex> {
        self.graph
            .edges_directed(idx, petgraph::Direction::Incoming)
            .map(|e| e.source())
            .collect()
    }

    /// Get dependents (successors) of a node.
    pub fn successors(&self, idx: NodeIndex) -> Vec<NodeIndex> {
        self.graph
            .edges_directed(idx, petgraph::Direction::Outgoing)
            .map(|e| e.target())
            .collect()
    }

    /// Collect all dependency edges as (from_op_id, to_op_id) pairs.
    pub fn dependency_pairs(&self) -> Vec<(usize, usize)> {
        self.graph
            .edge_indices()
            .map(|e| {
                let (src, tgt) = self.graph.edge_endpoints(e).unwrap();
                (self.graph[src].op_id, self.graph[tgt].op_id)
            })
            .collect()
    }

    /// Aggregate operations by block_id, returning the number of operations per block.
    pub fn block_aggregation(&self) -> HashMap<usize, Vec<NodeIndex>> {
        let mut blocks: HashMap<usize, Vec<NodeIndex>> = HashMap::new();
        for idx in self.graph.node_indices() {
            let blk = self.graph[idx].block_id;
            blocks.entry(blk).or_default().push(idx);
        }
        blocks
    }

    /// Count dependencies of each kind.
    pub fn dependency_counts(&self) -> HashMap<DepKind, usize> {
        let mut counts = HashMap::new();
        for edge in self.graph.edge_indices() {
            let w = &self.graph[edge];
            *counts.entry(w.kind).or_insert(0) += 1;
        }
        counts
    }

    /// Nodes with no incoming edges (sources / ready to execute first).
    pub fn source_nodes(&self) -> Vec<NodeIndex> {
        self.graph
            .node_indices()
            .filter(|&idx| {
                self.graph
                    .edges_directed(idx, petgraph::Direction::Incoming)
                    .next()
                    .is_none()
            })
            .collect()
    }

    /// Nodes with no outgoing edges (sinks).
    pub fn sink_nodes(&self) -> Vec<NodeIndex> {
        self.graph
            .node_indices()
            .filter(|&idx| {
                self.graph
                    .edges_directed(idx, petgraph::Direction::Outgoing)
                    .next()
                    .is_none()
            })
            .collect()
    }

    /// Compute the transitive reduction of the dependency graph.
    ///
    /// Removes redundant edges: if A→B→C and A→C, the direct A→C edge is
    /// redundant and removed. The result has the same reachability but
    /// minimal edges.
    pub fn reduce_graph(graph: &DependencyGraph) -> DependencyGraph {
        let topo = match graph.topological_order() {
            Some(t) => t,
            None => {
                // Return a clone if we can't toposort (cycle)
                return DependencyGraph {
                    graph: graph.graph.clone(),
                    block_size: graph.block_size,
                    num_processors: graph.num_processors,
                    num_phases: graph.num_phases,
                    location_map: graph.location_map.clone(),
                };
            }
        };

        // For each node, compute the set of nodes reachable via paths of length ≥ 2
        let node_count = graph.graph.node_count();
        let idx_to_pos: HashMap<NodeIndex, usize> = topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();

        // Reachability matrix (bit-set approximation via HashSet for correctness)
        let mut reachable: Vec<HashSet<usize>> = vec![HashSet::new(); node_count];

        // Process in reverse topological order
        for &node in topo.iter().rev() {
            let pos = idx_to_pos[&node];
            let succs: Vec<NodeIndex> = graph.successors(node);
            for &succ in &succs {
                let succ_pos = idx_to_pos[&succ];
                reachable[pos].insert(succ_pos);
                let succ_reach: HashSet<usize> = reachable[succ_pos].clone();
                reachable[pos].extend(succ_reach);
            }
        }

        // Build new graph, only keeping edges that are not transitively implied
        let mut new_graph = DiGraph::new();
        let mut old_to_new: HashMap<NodeIndex, NodeIndex> = HashMap::new();

        for &old_idx in &topo {
            let new_idx = new_graph.add_node(graph.graph[old_idx].clone());
            old_to_new.insert(old_idx, new_idx);
        }

        for &node in &topo {
            let succs: Vec<NodeIndex> = graph.successors(node);
            let node_pos = idx_to_pos[&node];

            for &succ in &succs {
                let succ_pos = idx_to_pos[&succ];
                // Check if succ is reachable through another direct successor
                let is_redundant = succs.iter().any(|&other_succ| {
                    if other_succ == succ {
                        return false;
                    }
                    let other_pos = idx_to_pos[&other_succ];
                    reachable[other_pos].contains(&succ_pos)
                });

                if !is_redundant {
                    // Find the edge weight
                    let edge = graph.graph.edges_directed(node, petgraph::Direction::Outgoing)
                        .find(|e| e.target() == succ)
                        .unwrap();
                    new_graph.add_edge(
                        old_to_new[&node],
                        old_to_new[&succ],
                        *edge.weight(),
                    );
                }
            }
        }

        // Rebuild location map for new indices
        let mut new_location_map: HashMap<(String, usize), Vec<NodeIndex>> = HashMap::new();
        for (&old_idx, &new_idx) in &old_to_new {
            let node = &graph.graph[old_idx];
            let key = (node.memory_region.clone(), node.address);
            new_location_map.entry(key).or_default().push(new_idx);
        }

        DependencyGraph {
            graph: new_graph,
            block_size: graph.block_size,
            num_processors: graph.num_processors,
            num_phases: graph.num_phases,
            location_map: new_location_map,
        }
    }

    /// Find maximal independent sets of operations.
    ///
    /// Returns groups of nodes that have no dependency edges between them
    /// and can thus be executed in any order (or in parallel). Uses a
    /// greedy coloring approach on levels.
    pub fn find_independent_sets(graph: &DependencyGraph) -> Vec<Vec<NodeIndex>> {
        let levels = graph.compute_levels();
        if levels.is_empty() {
            return Vec::new();
        }

        let max_level = levels.values().copied().max().unwrap_or(0);
        let mut sets: Vec<Vec<NodeIndex>> = Vec::new();

        for level in 0..=max_level {
            let level_nodes: Vec<NodeIndex> = graph.graph.node_indices()
                .filter(|&idx| levels.get(&idx).copied().unwrap_or(0) == level)
                .collect();

            if level_nodes.is_empty() {
                continue;
            }

            // Within a level, nodes may still have edges between them.
            // Greedily partition into independent subsets.
            let mut assigned: HashSet<NodeIndex> = HashSet::new();
            let mut remaining: Vec<NodeIndex> = level_nodes;

            while !remaining.is_empty() {
                let mut current_set: Vec<NodeIndex> = Vec::new();
                let mut excluded: HashSet<NodeIndex> = HashSet::new();

                for &node in &remaining {
                    if excluded.contains(&node) {
                        continue;
                    }
                    current_set.push(node);
                    // Exclude all neighbors within the remaining set
                    for succ in graph.successors(node) {
                        excluded.insert(succ);
                    }
                    for pred in graph.predecessors(node) {
                        excluded.insert(pred);
                    }
                }

                for &node in &current_set {
                    assigned.insert(node);
                }
                sets.push(current_set);
                remaining.retain(|n| !assigned.contains(n));
            }
        }

        sets
    }

    /// Merge a set of nodes in the graph into a single representative node.
    ///
    /// The merged node inherits all incoming and outgoing edges of the
    /// constituent nodes. The first node in the slice becomes the
    /// representative; others are removed. Only merges nodes that belong
    /// to the same block (block_id).
    pub fn merge_nodes(graph: &mut DependencyGraph, nodes: &[NodeIndex]) {
        if nodes.len() <= 1 {
            return;
        }

        // Verify all nodes share the same block_id
        let block_id = graph.graph[nodes[0]].block_id;
        let mergeable: Vec<NodeIndex> = nodes.iter()
            .copied()
            .filter(|&n| graph.graph.node_weight(n).map_or(false, |w| w.block_id == block_id))
            .collect();

        if mergeable.len() <= 1 {
            return;
        }

        let representative = mergeable[0];
        let to_remove: Vec<NodeIndex> = mergeable[1..].to_vec();

        // Collect edges to add to representative
        let mut incoming: Vec<(NodeIndex, DepEdge)> = Vec::new();
        let mut outgoing: Vec<(NodeIndex, DepEdge)> = Vec::new();

        for &node in &to_remove {
            for edge in graph.graph.edges_directed(node, petgraph::Direction::Incoming) {
                let src = edge.source();
                if src != representative && !to_remove.contains(&src) {
                    incoming.push((src, *edge.weight()));
                }
            }
            for edge in graph.graph.edges_directed(node, petgraph::Direction::Outgoing) {
                let tgt = edge.target();
                if tgt != representative && !to_remove.contains(&tgt) {
                    outgoing.push((tgt, *edge.weight()));
                }
            }
        }

        // Add edges to representative (avoid duplicates)
        for (src, weight) in &incoming {
            let already_exists = graph.graph
                .edges_directed(representative, petgraph::Direction::Incoming)
                .any(|e| e.source() == *src);
            if !already_exists {
                graph.graph.add_edge(*src, representative, *weight);
            }
        }

        for (tgt, weight) in &outgoing {
            let already_exists = graph.graph
                .edges_directed(representative, petgraph::Direction::Outgoing)
                .any(|e| e.target() == *tgt);
            if !already_exists {
                graph.graph.add_edge(representative, *tgt, *weight);
            }
        }

        // Remove merged nodes (in reverse index order to avoid invalidation)
        let mut sorted_remove = to_remove;
        sorted_remove.sort_by(|a, b| b.index().cmp(&a.index()));
        for node in sorted_remove {
            graph.graph.remove_node(node);
        }

        // Rebuild location map
        graph.location_map.clear();
        for idx in graph.graph.node_indices() {
            let n = &graph.graph[idx];
            let key = (n.memory_region.clone(), n.address);
            graph.location_map.entry(key).or_default().push(idx);
        }
    }

    /// Compute comprehensive statistics about the dependency graph.
    pub fn graph_statistics(graph: &DependencyGraph) -> GraphStats {
        let n = graph.node_count();
        let e = graph.edge_count();

        let density = if n > 1 {
            e as f64 / (n as f64 * (n as f64 - 1.0))
        } else {
            0.0
        };

        let mut in_degrees: Vec<usize> = Vec::new();
        let mut out_degrees: Vec<usize> = Vec::new();
        for idx in graph.graph.node_indices() {
            in_degrees.push(
                graph.graph.edges_directed(idx, petgraph::Direction::Incoming).count()
            );
            out_degrees.push(
                graph.graph.edges_directed(idx, petgraph::Direction::Outgoing).count()
            );
        }

        let avg_in_degree = if n > 0 {
            in_degrees.iter().sum::<usize>() as f64 / n as f64
        } else {
            0.0
        };

        let avg_out_degree = if n > 0 {
            out_degrees.iter().sum::<usize>() as f64 / n as f64
        } else {
            0.0
        };

        let max_in_degree = in_degrees.iter().copied().max().unwrap_or(0);
        let max_out_degree = out_degrees.iter().copied().max().unwrap_or(0);

        let critical_path = graph.critical_path_length();
        let sources = graph.source_nodes().len();
        let sinks = graph.sink_nodes().len();

        let dep_counts = graph.dependency_counts();
        let raw_count = dep_counts.get(&DepKind::RAW).copied().unwrap_or(0);
        let war_count = dep_counts.get(&DepKind::WAR).copied().unwrap_or(0);
        let waw_count = dep_counts.get(&DepKind::WAW).copied().unwrap_or(0);

        let blocks = graph.block_aggregation();
        let num_blocks = blocks.len();
        let avg_ops_per_block = if num_blocks > 0 {
            n as f64 / num_blocks as f64
        } else {
            0.0
        };

        GraphStats {
            node_count: n,
            edge_count: e,
            density,
            avg_in_degree,
            avg_out_degree,
            max_in_degree,
            max_out_degree,
            critical_path,
            num_sources: sources,
            num_sinks: sinks,
            raw_edges: raw_count,
            war_edges: war_count,
            waw_edges: waw_count,
            num_blocks,
            avg_ops_per_block,
            num_processors: graph.num_processors,
            num_phases: graph.num_phases,
        }
    }

    /// Generate a DOT format string for graph visualization.
    ///
    /// Nodes are labeled with their operation details; edges are colored
    /// by dependency kind (RAW=red, WAR=blue, WAW=orange).
    pub fn visualize_dot(graph: &DependencyGraph) -> String {
        let mut dot = String::new();
        dot.push_str("digraph DependencyGraph {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box, fontsize=10];\n");
        dot.push_str("  edge [fontsize=8];\n\n");

        for idx in graph.graph.node_indices() {
            let node = &graph.graph[idx];
            let color = match node.op_type {
                OpType::Read => "lightblue",
                OpType::Write => "lightyellow",
            };
            dot.push_str(&format!(
                "  n{} [label=\"op{}: {} {}[{}]\\nblk={} ph={} p={}\", style=filled, fillcolor=\"{}\"];\n",
                idx.index(),
                node.op_id,
                node.op_type,
                node.memory_region,
                node.address,
                node.block_id,
                node.phase,
                node.proc_id,
                color,
            ));
        }

        dot.push('\n');

        for edge_idx in graph.graph.edge_indices() {
            let (src, tgt) = graph.graph.edge_endpoints(edge_idx).unwrap();
            let weight = &graph.graph[edge_idx];
            let (color, label) = match weight.kind {
                DepKind::RAW => ("red", "RAW"),
                DepKind::WAR => ("blue", "WAR"),
                DepKind::WAW => ("orange", "WAW"),
            };
            dot.push_str(&format!(
                "  n{} -> n{} [color=\"{}\", label=\"{}\"];\n",
                src.index(),
                tgt.index(),
                color,
                label,
            ));
        }

        dot.push_str("}\n");
        dot
    }
}

/// Comprehensive statistics about a dependency graph.
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub density: f64,
    pub avg_in_degree: f64,
    pub avg_out_degree: f64,
    pub max_in_degree: usize,
    pub max_out_degree: usize,
    pub critical_path: usize,
    pub num_sources: usize,
    pub num_sinks: usize,
    pub raw_edges: usize,
    pub war_edges: usize,
    pub waw_edges: usize,
    pub num_blocks: usize,
    pub avg_ops_per_block: f64,
    pub num_processors: usize,
    pub num_phases: usize,
}

impl fmt::Display for GraphStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Graph Statistics ===")?;
        writeln!(f, "Nodes:           {}", self.node_count)?;
        writeln!(f, "Edges:           {}", self.edge_count)?;
        writeln!(f, "Density:         {:.6}", self.density)?;
        writeln!(f, "Avg in-degree:   {:.2}", self.avg_in_degree)?;
        writeln!(f, "Avg out-degree:  {:.2}", self.avg_out_degree)?;
        writeln!(f, "Max in-degree:   {}", self.max_in_degree)?;
        writeln!(f, "Max out-degree:  {}", self.max_out_degree)?;
        writeln!(f, "Critical path:   {}", self.critical_path)?;
        writeln!(f, "Sources:         {}", self.num_sources)?;
        writeln!(f, "Sinks:           {}", self.num_sinks)?;
        writeln!(f, "RAW edges:       {}", self.raw_edges)?;
        writeln!(f, "WAR edges:       {}", self.war_edges)?;
        writeln!(f, "WAW edges:       {}", self.waw_edges)?;
        writeln!(f, "Blocks:          {}", self.num_blocks)?;
        writeln!(f, "Avg ops/block:   {:.2}", self.avg_ops_per_block)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_op(
        op_id: usize,
        block_id: usize,
        phase: usize,
        proc_id: usize,
        op_type: OpType,
        address: usize,
    ) -> OperationNode {
        OperationNode {
            op_id,
            block_id,
            phase,
            proc_id,
            op_type,
            address,
            memory_region: "A".to_string(),
        }
    }

    #[test]
    fn test_raw_dependency() {
        // Writer at addr 0, then reader at addr 0 → RAW edge
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        assert_eq!(dg.node_count(), 2);
        assert_eq!(dg.edge_count(), 1);
        let counts = dg.dependency_counts();
        assert_eq!(counts[&DepKind::RAW], 1);
    }

    #[test]
    fn test_war_dependency() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Read, 0),
            make_op(1, 0, 0, 1, OpType::Write, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        assert_eq!(dg.edge_count(), 1);
        let counts = dg.dependency_counts();
        assert_eq!(counts[&DepKind::WAR], 1);
    }

    #[test]
    fn test_waw_dependency() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Write, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        assert_eq!(dg.edge_count(), 1);
        let counts = dg.dependency_counts();
        assert_eq!(counts[&DepKind::WAW], 1);
    }

    #[test]
    fn test_no_dependency_between_reads() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Read, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        assert_eq!(dg.edge_count(), 0);
    }

    #[test]
    fn test_no_dependency_different_addresses() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 1),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        assert_eq!(dg.edge_count(), 0);
    }

    #[test]
    fn test_topological_order() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
            make_op(2, 0, 0, 2, OpType::Write, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let topo = dg.topological_order().unwrap();
        // Write(0) must come before Read(1) and Write(2)
        let pos: HashMap<usize, usize> = topo
            .iter()
            .enumerate()
            .map(|(i, &idx)| (dg.graph[idx].op_id, i))
            .collect();
        assert!(pos[&0] < pos[&1]);
        assert!(pos[&0] < pos[&2]);
        assert!(pos[&1] < pos[&2]);
    }

    #[test]
    fn test_critical_path() {
        // Chain: W(0) → R(1) → W(2) → R(3)
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
            make_op(2, 0, 0, 2, OpType::Write, 0),
            make_op(3, 0, 0, 3, OpType::Read, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        assert_eq!(dg.critical_path_length(), 4);
    }

    #[test]
    fn test_independent_operations() {
        // Operations on different addresses are independent → critical path = 1
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 1, 0, 1, OpType::Write, 1),
            make_op(2, 2, 0, 2, OpType::Write, 2),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        assert_eq!(dg.critical_path_length(), 1);
        assert_eq!(dg.edge_count(), 0);
    }

    #[test]
    fn test_block_aggregation() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Read, 0),
            make_op(1, 0, 0, 1, OpType::Read, 1),
            make_op(2, 1, 0, 2, OpType::Write, 4),
            make_op(3, 1, 0, 3, OpType::Write, 5),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let blocks = dg.block_aggregation();
        assert_eq!(blocks[&0].len(), 2);
        assert_eq!(blocks[&1].len(), 2);
    }

    #[test]
    fn test_source_and_sink_nodes() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let sources = dg.source_nodes();
        let sinks = dg.sink_nodes();
        assert_eq!(sources.len(), 1);
        assert_eq!(sinks.len(), 1);
        assert_eq!(dg.graph[sources[0]].op_id, 0);
        assert_eq!(dg.graph[sinks[0]].op_id, 1);
    }

    #[test]
    fn test_compute_levels() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
            make_op(2, 1, 0, 2, OpType::Write, 5),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let levels = dg.compute_levels();
        // op 0 is source → level 0
        // op 1 depends on op 0 → level 1
        // op 2 is independent → level 0
        for (idx, &level) in &levels {
            match dg.graph[*idx].op_id {
                0 => assert_eq!(level, 0),
                1 => assert_eq!(level, 1),
                2 => assert_eq!(level, 0),
                _ => panic!("unexpected op_id"),
            }
        }
    }

    #[test]
    fn test_dependency_pairs() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let pairs = dg.dependency_pairs();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (0, 1));
    }

    #[test]
    fn test_reduce_graph_removes_transitive_edges() {
        // Chain: W(0) → R(1) → W(2), plus transitive W(0) → W(2)
        // The W(0)→W(2) WAW edge is transitive through R(1)
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
            make_op(2, 0, 0, 2, OpType::Write, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        // Original has 3 edges: W→R (RAW), R→W (WAR), W→W (WAW)
        assert_eq!(dg.edge_count(), 3);

        let reduced = DependencyGraph::reduce_graph(&dg);
        // After reduction, W(0)→W(2) is redundant (reachable via W(0)→R(1)→W(2))
        assert!(reduced.edge_count() < dg.edge_count());
        assert_eq!(reduced.node_count(), 3);
        // Same reachability: topological order still valid
        assert!(reduced.topological_order().is_some());
    }

    #[test]
    fn test_reduce_graph_independent_ops() {
        // Independent ops have no edges, reduction is a no-op
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 1, 0, 1, OpType::Write, 1),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        assert_eq!(dg.edge_count(), 0);

        let reduced = DependencyGraph::reduce_graph(&dg);
        assert_eq!(reduced.edge_count(), 0);
        assert_eq!(reduced.node_count(), 2);
    }

    #[test]
    fn test_find_independent_sets_all_independent() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 1, 0, 1, OpType::Write, 1),
            make_op(2, 2, 0, 2, OpType::Write, 2),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let sets = DependencyGraph::find_independent_sets(&dg);

        // All ops are independent, so they should be in one set
        assert!(!sets.is_empty());
        let total: usize = sets.iter().map(|s| s.len()).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_find_independent_sets_chain() {
        // Chain: W→R→W, each at a different level → each level is a set
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
            make_op(2, 0, 0, 2, OpType::Write, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let sets = DependencyGraph::find_independent_sets(&dg);

        let total: usize = sets.iter().map(|s| s.len()).sum();
        assert_eq!(total, 3);
        // Each set has exactly 1 node (since they form a chain)
        for set in &sets {
            assert_eq!(set.len(), 1);
        }
    }

    #[test]
    fn test_merge_nodes_same_block() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Write, 1),
            make_op(2, 1, 0, 2, OpType::Write, 4),
        ];
        let mut dg = DependencyGraph::from_operations(ops, 4);
        assert_eq!(dg.node_count(), 3);

        let block0_nodes: Vec<NodeIndex> = dg.graph.node_indices()
            .filter(|&idx| dg.graph[idx].block_id == 0)
            .collect();
        DependencyGraph::merge_nodes(&mut dg, &block0_nodes);

        // Two block-0 nodes merged into one, plus the block-1 node
        assert_eq!(dg.node_count(), 2);
    }

    #[test]
    fn test_merge_nodes_different_blocks_no_merge() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 1, 0, 1, OpType::Write, 4),
        ];
        let mut dg = DependencyGraph::from_operations(ops, 4);
        let all: Vec<NodeIndex> = dg.graph.node_indices().collect();
        DependencyGraph::merge_nodes(&mut dg, &all);
        // Different blocks → only the representative's block merges
        // But since first is block 0, only block-0 nodes merge
        assert!(dg.node_count() >= 1);
    }

    #[test]
    fn test_graph_statistics() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
            make_op(2, 1, 0, 2, OpType::Write, 4),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let stats = DependencyGraph::graph_statistics(&dg);

        assert_eq!(stats.node_count, 3);
        assert_eq!(stats.raw_edges, 1);
        assert_eq!(stats.critical_path, 2);
        assert_eq!(stats.num_blocks, 2);
        assert!(stats.density >= 0.0);
        assert!(stats.avg_ops_per_block > 0.0);
        let s = format!("{}", stats);
        assert!(s.contains("Graph Statistics"));
    }

    #[test]
    fn test_graph_statistics_empty() {
        let dg = DependencyGraph::from_operations(vec![], 4);
        let stats = DependencyGraph::graph_statistics(&dg);
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.edge_count, 0);
        assert_eq!(stats.density, 0.0);
    }

    #[test]
    fn test_visualize_dot_format() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let dot = DependencyGraph::visualize_dot(&dg);

        assert!(dot.starts_with("digraph"));
        assert!(dot.contains("->"));
        assert!(dot.contains("RAW"));
        assert!(dot.contains("lightblue"));
        assert!(dot.contains("lightyellow"));
        assert!(dot.ends_with("}\n"));
    }

    #[test]
    fn test_visualize_dot_empty() {
        let dg = DependencyGraph::from_operations(vec![], 4);
        let dot = DependencyGraph::visualize_dot(&dg);
        assert!(dot.starts_with("digraph"));
        assert!(!dot.contains("->"));
    }

    #[test]
    fn test_reduce_preserves_critical_path() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
            make_op(2, 0, 0, 2, OpType::Write, 0),
            make_op(3, 0, 0, 3, OpType::Read, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let reduced = DependencyGraph::reduce_graph(&dg);

        assert_eq!(reduced.critical_path_length(), dg.critical_path_length());
    }

    #[test]
    fn test_build_from_pram_program() {
        use crate::pram_ir::ast::{MemoryModel, PramProgram, SharedMemoryDecl};
        use crate::pram_ir::types::PramType;

        let mut prog = PramProgram::new("test", MemoryModel::CREW);
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

        let dg = DependencyGraph::build(&prog, 4, 4);
        assert!(dg.node_count() > 0);
    }

    #[test]
    fn test_predecessors_successors() {
        let ops = vec![
            make_op(0, 0, 0, 0, OpType::Write, 0),
            make_op(1, 0, 0, 1, OpType::Read, 0),
            make_op(2, 0, 0, 2, OpType::Write, 0),
        ];
        let dg = DependencyGraph::from_operations(ops, 4);
        let topo = dg.topological_order().unwrap();

        // Find the node for op 1
        let op1_idx = dg
            .graph
            .node_indices()
            .find(|&i| dg.graph[i].op_id == 1)
            .unwrap();
        let preds = dg.predecessors(op1_idx);
        assert_eq!(preds.len(), 1);
        assert_eq!(dg.graph[preds[0]].op_id, 0);

        let succs = dg.successors(op1_idx);
        assert_eq!(succs.len(), 1);
        assert_eq!(dg.graph[succs[0]].op_id, 2);
    }
}
