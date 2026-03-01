//! Algorithm library: a catalog of classic PRAM algorithms encoded as `PramProgram` ASTs.

pub mod arithmetic;
pub mod connectivity;
pub mod geometry;
pub mod graph;
pub mod list;
pub mod search;
pub mod selection;
pub mod sorting;
pub mod string_algo;
pub mod tree;

use crate::pram_ir::ast::PramProgram;

/// Descriptor for an algorithm in the catalog.
#[derive(Debug, Clone)]
pub struct AlgorithmEntry {
    pub name: &'static str,
    pub category: &'static str,
    pub memory_model: &'static str,
    pub time_bound: &'static str,
    pub description: &'static str,
    pub builder: fn() -> PramProgram,
}

/// Return every algorithm in the library.
pub fn catalog() -> Vec<AlgorithmEntry> {
    vec![
        // ── Sorting ──────────────────────────────────────────────
        AlgorithmEntry {
            name: "cole_merge_sort",
            category: "sorting",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Cole's pipelined merge sort",
            builder: sorting::cole_merge_sort,
        },
        AlgorithmEntry {
            name: "bitonic_sort",
            category: "sorting",
            memory_model: "EREW",
            time_bound: "O(log^2 n)",
            description: "Batcher's bitonic sort",
            builder: sorting::bitonic_sort,
        },
        AlgorithmEntry {
            name: "sample_sort",
            category: "sorting",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Parallel sample sort",
            builder: sorting::sample_sort,
        },
        AlgorithmEntry {
            name: "odd_even_merge_sort",
            category: "sorting",
            memory_model: "EREW",
            time_bound: "O(log^2 n)",
            description: "Odd-even merge sort",
            builder: sorting::odd_even_merge_sort,
        },
        // ── Graph ────────────────────────────────────────────────
        AlgorithmEntry {
            name: "shiloach_vishkin",
            category: "graph",
            memory_model: "CRCW-Arbitrary",
            time_bound: "O(log n)",
            description: "Shiloach-Vishkin connected components",
            builder: graph::shiloach_vishkin,
        },
        AlgorithmEntry {
            name: "boruvka_mst",
            category: "graph",
            memory_model: "CRCW-Priority",
            time_bound: "O(log n)",
            description: "Borůvka's minimum spanning tree",
            builder: graph::boruvka_mst,
        },
        AlgorithmEntry {
            name: "parallel_bfs",
            category: "graph",
            memory_model: "CREW",
            time_bound: "O(D)",
            description: "Level-synchronous parallel BFS",
            builder: graph::parallel_bfs,
        },
        AlgorithmEntry {
            name: "euler_tour",
            category: "graph",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Euler tour construction on trees",
            builder: graph::euler_tour,
        },
        // ── Connectivity ─────────────────────────────────────────
        AlgorithmEntry {
            name: "vishkin_connectivity",
            category: "connectivity",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Vishkin's deterministic connectivity",
            builder: connectivity::vishkin_connectivity,
        },
        AlgorithmEntry {
            name: "ear_decomposition",
            category: "connectivity",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Reif's ear decomposition",
            builder: connectivity::ear_decomposition,
        },
        // ── List ─────────────────────────────────────────────────
        AlgorithmEntry {
            name: "list_ranking",
            category: "list",
            memory_model: "EREW",
            time_bound: "O(log n)",
            description: "Pointer-jumping list ranking",
            builder: list::list_ranking,
        },
        AlgorithmEntry {
            name: "prefix_sum",
            category: "list",
            memory_model: "EREW",
            time_bound: "O(log n)",
            description: "Parallel prefix sum (scan)",
            builder: list::prefix_sum,
        },
        AlgorithmEntry {
            name: "compact",
            category: "list",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Parallel compaction",
            builder: list::compact,
        },
        // ── Arithmetic ───────────────────────────────────────────
        AlgorithmEntry {
            name: "parallel_addition",
            category: "arithmetic",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Carry-lookahead parallel addition",
            builder: arithmetic::parallel_addition,
        },
        AlgorithmEntry {
            name: "parallel_multiplication",
            category: "arithmetic",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Parallel integer multiplication",
            builder: arithmetic::parallel_multiplication,
        },
        AlgorithmEntry {
            name: "matrix_multiply",
            category: "arithmetic",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Parallel matrix multiplication",
            builder: arithmetic::matrix_multiply,
        },
        AlgorithmEntry {
            name: "matrix_vector_multiply",
            category: "arithmetic",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Parallel matrix-vector multiply",
            builder: arithmetic::matrix_vector_multiply,
        },
        // ── Geometry ─────────────────────────────────────────────
        AlgorithmEntry {
            name: "convex_hull",
            category: "geometry",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Parallel convex hull",
            builder: geometry::convex_hull,
        },
        AlgorithmEntry {
            name: "closest_pair",
            category: "geometry",
            memory_model: "CREW",
            time_bound: "O(log^2 n)",
            description: "Parallel closest pair",
            builder: geometry::closest_pair,
        },
        // ── Tree ─────────────────────────────────────────────────
        AlgorithmEntry {
            name: "tree_contraction",
            category: "tree",
            memory_model: "EREW",
            time_bound: "O(log n)",
            description: "Rake/compress tree contraction",
            builder: tree::tree_contraction,
        },
        AlgorithmEntry {
            name: "lca",
            category: "tree",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Lowest common ancestor",
            builder: tree::lca,
        },
        // ── String ───────────────────────────────────────────────
        AlgorithmEntry {
            name: "string_matching",
            category: "string",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Parallel string matching",
            builder: string_algo::string_matching,
        },
        AlgorithmEntry {
            name: "suffix_array",
            category: "string",
            memory_model: "CREW",
            time_bound: "O(log^2 n)",
            description: "Parallel suffix array construction",
            builder: string_algo::suffix_array,
        },
        // ── Selection ────────────────────────────────────────────
        AlgorithmEntry {
            name: "parallel_selection",
            category: "selection",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Parallel kth-element selection",
            builder: selection::parallel_selection,
        },
        // ── Search ───────────────────────────────────────────────
        AlgorithmEntry {
            name: "parallel_binary_search",
            category: "search",
            memory_model: "CREW",
            time_bound: "O(log n / log log n)",
            description: "Parallel binary search",
            builder: search::parallel_binary_search,
        },
        AlgorithmEntry {
            name: "parallel_interpolation_search",
            category: "search",
            memory_model: "CREW",
            time_bound: "O(log log n)",
            description: "Parallel interpolation search",
            builder: search::parallel_interpolation_search,
        },
        // ── New Sorting ──────────────────────────────────────────
        AlgorithmEntry {
            name: "radix_sort",
            category: "sorting",
            memory_model: "EREW",
            time_bound: "O(b log n)",
            description: "Parallel radix sort via prefix sums",
            builder: sorting::radix_sort,
        },
        AlgorithmEntry {
            name: "aks_sorting_network",
            category: "sorting",
            memory_model: "EREW",
            time_bound: "O(log n)",
            description: "AKS optimal sorting network",
            builder: sorting::aks_sorting_network,
        },
        AlgorithmEntry {
            name: "flashsort",
            category: "sorting",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Distribution-based parallel flashsort",
            builder: sorting::flashsort,
        },
        // ── New Graph ────────────────────────────────────────────
        AlgorithmEntry {
            name: "parallel_dfs",
            category: "graph",
            memory_model: "CREW",
            time_bound: "O(D log n)",
            description: "Parallel depth-first search",
            builder: graph::parallel_dfs,
        },
        AlgorithmEntry {
            name: "graph_coloring",
            category: "graph",
            memory_model: "CRCW-Common",
            time_bound: "O(Delta log n)",
            description: "Parallel graph coloring",
            builder: graph::graph_coloring,
        },
        AlgorithmEntry {
            name: "maximal_independent_set",
            category: "graph",
            memory_model: "CRCW-Arbitrary",
            time_bound: "O(log n)",
            description: "Luby's parallel MIS algorithm",
            builder: graph::maximal_independent_set,
        },
        AlgorithmEntry {
            name: "shortest_path",
            category: "graph",
            memory_model: "CREW",
            time_bound: "O(n log n)",
            description: "Parallel Bellman-Ford shortest paths",
            builder: graph::shortest_path,
        },
        // ── New Connectivity ─────────────────────────────────────
        AlgorithmEntry {
            name: "biconnected_components",
            category: "connectivity",
            memory_model: "CREW",
            time_bound: "O(log^2 n)",
            description: "Parallel biconnected components",
            builder: connectivity::biconnected_components,
        },
        AlgorithmEntry {
            name: "strongly_connected",
            category: "connectivity",
            memory_model: "CREW",
            time_bound: "O(log^2 n)",
            description: "Parallel strongly connected components",
            builder: connectivity::strongly_connected,
        },
        // ── New List ─────────────────────────────────────────────
        AlgorithmEntry {
            name: "segmented_scan",
            category: "list",
            memory_model: "EREW",
            time_bound: "O(log n)",
            description: "Parallel segmented prefix sum",
            builder: list::segmented_scan,
        },
        AlgorithmEntry {
            name: "list_split",
            category: "list",
            memory_model: "EREW",
            time_bound: "O(log n)",
            description: "Parallel list splitting",
            builder: list::list_split,
        },
        AlgorithmEntry {
            name: "symmetry_breaking",
            category: "list",
            memory_model: "EREW",
            time_bound: "O(log* n)",
            description: "Deterministic symmetry breaking (Cole-Vishkin)",
            builder: list::symmetry_breaking,
        },
        // ── New Arithmetic ───────────────────────────────────────
        AlgorithmEntry {
            name: "parallel_prefix_multiplication",
            category: "arithmetic",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Parallel prefix multiplication",
            builder: arithmetic::parallel_prefix_multiplication,
        },
        AlgorithmEntry {
            name: "strassen_matrix_multiply",
            category: "arithmetic",
            memory_model: "CREW",
            time_bound: "O(log^2 n)",
            description: "Strassen's parallel matrix multiply",
            builder: arithmetic::strassen_matrix_multiply,
        },
        AlgorithmEntry {
            name: "fft",
            category: "arithmetic",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Parallel FFT butterfly computation",
            builder: arithmetic::fft,
        },
        // ── New Geometry ─────────────────────────────────────────
        AlgorithmEntry {
            name: "line_segment_intersection",
            category: "geometry",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Parallel line segment intersection",
            builder: geometry::line_segment_intersection,
        },
        AlgorithmEntry {
            name: "voronoi_diagram",
            category: "geometry",
            memory_model: "CREW",
            time_bound: "O(log^2 n)",
            description: "Parallel Voronoi diagram",
            builder: geometry::voronoi_diagram,
        },
        AlgorithmEntry {
            name: "point_location",
            category: "geometry",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Parallel point location",
            builder: geometry::point_location,
        },
        // ── New Tree ─────────────────────────────────────────────
        AlgorithmEntry {
            name: "tree_isomorphism",
            category: "tree",
            memory_model: "CREW",
            time_bound: "O(log^2 n)",
            description: "Parallel tree isomorphism testing",
            builder: tree::tree_isomorphism,
        },
        AlgorithmEntry {
            name: "centroid_decomposition",
            category: "tree",
            memory_model: "EREW",
            time_bound: "O(log^2 n)",
            description: "Parallel centroid decomposition",
            builder: tree::centroid_decomposition,
        },
        // ── New String ───────────────────────────────────────────
        AlgorithmEntry {
            name: "lcp_array",
            category: "string",
            memory_model: "CREW",
            time_bound: "O(log^2 n)",
            description: "Parallel LCP array construction",
            builder: string_algo::lcp_array,
        },
        AlgorithmEntry {
            name: "string_sorting",
            category: "string",
            memory_model: "CREW",
            time_bound: "O(L log n)",
            description: "Parallel string sorting (MSD radix)",
            builder: string_algo::string_sorting,
        },
        // ── New Selection ────────────────────────────────────────
        AlgorithmEntry {
            name: "parallel_median",
            category: "selection",
            memory_model: "CREW",
            time_bound: "O(log^2 n)",
            description: "Parallel weighted median finding",
            builder: selection::parallel_median,
        },
        AlgorithmEntry {
            name: "parallel_partition",
            category: "selection",
            memory_model: "EREW",
            time_bound: "O(log n)",
            description: "Parallel array partitioning",
            builder: selection::parallel_partition,
        },
        // ── New Search ───────────────────────────────────────────
        AlgorithmEntry {
            name: "parallel_batch_search",
            category: "search",
            memory_model: "CREW",
            time_bound: "O(log n)",
            description: "Parallel batch search for multiple keys",
            builder: search::parallel_batch_search,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_non_empty() {
        let cat = catalog();
        assert!(cat.len() >= 45);
    }

    #[test]
    fn test_all_algorithms_build() {
        for entry in catalog() {
            let prog = (entry.builder)();
            assert_eq!(prog.name, entry.name, "name mismatch for {}", entry.name);
            assert!(!prog.parameters.is_empty(), "{} has no parameters", entry.name);
        }
    }
}
