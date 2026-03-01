//! Hand-optimized sequential baselines for comparison with compiled PRAM code.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Sorting
// ---------------------------------------------------------------------------

/// In-place introsort (quicksort + heapsort fallback + insertion sort for small arrays).
pub fn baseline_sort(arr: &mut [i64]) {
    if arr.len() <= 1 {
        return;
    }
    let max_depth = (2.0 * (arr.len() as f64).log2()) as usize;
    introsort_impl(arr, max_depth);
}

fn introsort_impl(arr: &mut [i64], depth_limit: usize) {
    let n = arr.len();
    if n <= 16 {
        insertion_sort(arr);
        return;
    }
    if depth_limit == 0 {
        heapsort(arr);
        return;
    }
    let pivot_idx = partition(arr);
    introsort_impl(&mut arr[..pivot_idx], depth_limit - 1);
    introsort_impl(&mut arr[pivot_idx + 1..], depth_limit - 1);
}

fn insertion_sort(arr: &mut [i64]) {
    for i in 1..arr.len() {
        let key = arr[i];
        let mut j = i;
        while j > 0 && arr[j - 1] > key {
            arr[j] = arr[j - 1];
            j -= 1;
        }
        arr[j] = key;
    }
}

fn partition(arr: &mut [i64]) -> usize {
    let n = arr.len();
    // Median-of-three pivot selection
    let mid = n / 2;
    let last = n - 1;
    if arr[0] > arr[mid] {
        arr.swap(0, mid);
    }
    if arr[0] > arr[last] {
        arr.swap(0, last);
    }
    if arr[mid] > arr[last] {
        arr.swap(mid, last);
    }
    arr.swap(mid, last);
    let pivot = arr[last];

    let mut i = 0usize;
    for j in 0..last {
        if arr[j] <= pivot {
            arr.swap(i, j);
            i += 1;
        }
    }
    arr.swap(i, last);
    i
}

fn heapsort(arr: &mut [i64]) {
    let n = arr.len();
    // Build max-heap
    for i in (0..n / 2).rev() {
        sift_down(arr, i, n);
    }
    // Extract elements
    for end in (1..n).rev() {
        arr.swap(0, end);
        sift_down(arr, 0, end);
    }
}

fn sift_down(arr: &mut [i64], mut root: usize, end: usize) {
    loop {
        let left = 2 * root + 1;
        if left >= end {
            break;
        }
        let right = left + 1;
        let mut largest = root;
        if arr[left] > arr[largest] {
            largest = left;
        }
        if right < end && arr[right] > arr[largest] {
            largest = right;
        }
        if largest == root {
            break;
        }
        arr.swap(root, largest);
        root = largest;
    }
}

// ---------------------------------------------------------------------------
// Prefix sum
// ---------------------------------------------------------------------------

/// Sequential inclusive prefix sum (in-place).
pub fn baseline_prefix_sum(arr: &mut [i64]) {
    for i in 1..arr.len() {
        arr[i] += arr[i - 1];
    }
}

/// Sequential exclusive prefix sum (in-place, first element becomes 0).
pub fn baseline_exclusive_prefix_sum(arr: &mut [i64]) {
    if arr.is_empty() {
        return;
    }
    let mut running = 0i64;
    for val in arr.iter_mut() {
        let tmp = *val;
        *val = running;
        running += tmp;
    }
}

// ---------------------------------------------------------------------------
// Connected components (union-find)
// ---------------------------------------------------------------------------

/// Union-Find data structure with path compression and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

/// Compute connected components of an undirected graph.
///
/// * `edges` – list of (u, v) pairs
/// * `n` – number of vertices (0..n)
///
/// Returns a vector of length `n` where `result[v]` is the component representative for vertex `v`.
pub fn baseline_connected_components(edges: &[(usize, usize)], n: usize) -> Vec<usize> {
    let mut uf = UnionFind::new(n);
    for &(u, v) in edges {
        uf.union(u, v);
    }
    // Flatten: make every node point directly to its root
    let mut components = vec![0usize; n];
    for i in 0..n {
        components[i] = uf.find(i);
    }
    components
}

// ---------------------------------------------------------------------------
// List ranking
// ---------------------------------------------------------------------------

/// Sequential list ranking: given a linked list encoded as `next[i]` (with a sentinel
/// where `next[i] == i` or `next[i] >= n` for the tail), and initial weights,
/// compute the suffix-sum of weights along each list.
///
/// Returns a vector where `result[i]` = sum of weights from node `i` to the end of the list.
pub fn baseline_list_ranking(next: &[usize], weights: &[i64]) -> Vec<i64> {
    let n = next.len();
    if n == 0 {
        return Vec::new();
    }

    let mut result = weights.to_vec();

    // Find the head of each list: nodes that are not pointed to by anyone else.
    let mut is_pointed_to = vec![false; n];
    for i in 0..n {
        if next[i] < n && next[i] != i {
            is_pointed_to[next[i]] = true;
        }
    }

    // For each head, walk the list and accumulate
    for start in 0..n {
        if is_pointed_to[start] {
            continue; // not a head
        }
        // Collect the list nodes in order
        let mut nodes = Vec::new();
        let mut cur = start;
        loop {
            nodes.push(cur);
            if next[cur] >= n || next[cur] == cur {
                break;
            }
            cur = next[cur];
        }
        // Compute suffix sums
        for i in (0..nodes.len().saturating_sub(1)).rev() {
            result[nodes[i]] += result[nodes[i + 1]];
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Matrix multiply
// ---------------------------------------------------------------------------

/// Standard O(n³) matrix multiplication with cache-friendly loop order (ikj).
///
/// Matrices are stored as flat row-major vectors of size n×n.
/// Returns `C = A × B`.
pub fn baseline_matrix_multiply(a: &[i64], b: &[i64], n: usize) -> Vec<i64> {
    assert_eq!(a.len(), n * n);
    assert_eq!(b.len(), n * n);

    let mut c = vec![0i64; n * n];

    // ikj order for better cache locality on row-major layout
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            if a_ik == 0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }

    c
}

// ---------------------------------------------------------------------------
// Binary search
// ---------------------------------------------------------------------------

/// Standard binary search on a sorted slice. Returns the index if found.
pub fn baseline_binary_search(arr: &[i64], target: i64) -> Option<usize> {
    let mut lo = 0usize;
    let mut hi = arr.len();

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match arr[mid].cmp(&target) {
            std::cmp::Ordering::Equal => return Some(mid),
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
        }
    }

    None
}

// ---------------------------------------------------------------------------
// BFS
// ---------------------------------------------------------------------------

/// Sequential BFS from a given start vertex.
///
/// * `adj` – adjacency lists: `adj[u]` is the list of neighbors of `u`.
/// * `start` – starting vertex.
/// * `n` – number of vertices.
///
/// Returns a vector of distances; unreachable vertices get distance -1.
pub fn baseline_bfs(adj: &[Vec<usize>], start: usize, n: usize) -> Vec<i64> {
    let mut dist = vec![-1i64; n];
    dist[start] = 0;

    let mut queue = VecDeque::new();
    queue.push_back(start);

    while let Some(u) = queue.pop_front() {
        for &v in &adj[u] {
            if dist[v] == -1 {
                dist[v] = dist[u] + 1;
                queue.push_back(v);
            }
        }
    }

    dist
}

// ---------------------------------------------------------------------------
// Merge Sort (bottom-up, in-place with temporary buffer)
// ---------------------------------------------------------------------------

pub fn baseline_merge_sort(arr: &mut [i64]) {
    let n = arr.len();
    if n <= 1 {
        return;
    }
    let mut buf = arr.to_vec();
    let mut width = 1;
    while width < n {
        let mut i = 0;
        while i < n {
            let left = i;
            let mid = (i + width).min(n);
            let right = (i + 2 * width).min(n);
            let (mut l, mut r) = (left, mid);
            let mut k = left;
            while l < mid && r < right {
                if arr[l] <= arr[r] {
                    buf[k] = arr[l];
                    l += 1;
                } else {
                    buf[k] = arr[r];
                    r += 1;
                }
                k += 1;
            }
            while l < mid {
                buf[k] = arr[l];
                l += 1;
                k += 1;
            }
            while r < right {
                buf[k] = arr[r];
                r += 1;
                k += 1;
            }
            i += 2 * width;
        }
        arr.copy_from_slice(&buf);
        width *= 2;
    }
}

// ---------------------------------------------------------------------------
// Convex Hull (Graham scan)
// ---------------------------------------------------------------------------

pub fn baseline_convex_hull(points: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let n = points.len();
    if n < 2 {
        return points.to_vec();
    }

    // Find the lowest point (and leftmost among ties).
    let mut pts: Vec<(i64, i64)> = points.to_vec();
    let mut pivot = 0;
    for i in 1..n {
        if pts[i].1 < pts[pivot].1 || (pts[i].1 == pts[pivot].1 && pts[i].0 < pts[pivot].0) {
            pivot = i;
        }
    }
    pts.swap(0, pivot);
    let origin = pts[0];

    // Cross product helper: (o->a) x (o->b)
    let cross =
        |o: (i64, i64), a: (i64, i64), b: (i64, i64)| -> i64 {
            (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
        };

    // Sort by polar angle around origin; break ties by distance.
    pts[1..].sort_by(|a, b| {
        let c = cross(origin, *a, *b);
        if c != 0 {
            return if c > 0 {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            };
        }
        let da = (a.0 - origin.0).pow(2) + (a.1 - origin.1).pow(2);
        let db = (b.0 - origin.0).pow(2) + (b.1 - origin.1).pow(2);
        da.cmp(&db)
    });

    // Remove collinear points that are closer to origin (keep farthest).
    let mut cleaned: Vec<(i64, i64)> = vec![origin];
    for i in 1..n {
        // Skip points collinear with the previous (same angle), keep farthest.
        if i + 1 < n && cross(origin, pts[i], pts[i + 1]) == 0 {
            continue;
        }
        cleaned.push(pts[i]);
    }

    if cleaned.len() < 3 {
        return cleaned;
    }

    let mut stack: Vec<(i64, i64)> = Vec::new();
    for &p in &cleaned {
        while stack.len() > 1 && cross(stack[stack.len() - 2], stack[stack.len() - 1], p) <= 0 {
            stack.pop();
        }
        stack.push(p);
    }

    stack
}

// ---------------------------------------------------------------------------
// KMP String Matching
// ---------------------------------------------------------------------------

pub fn baseline_string_match(text: &[u8], pattern: &[u8]) -> Vec<usize> {
    let m = pattern.len();
    if m == 0 {
        return vec![];
    }
    let n = text.len();
    if n < m {
        return vec![];
    }

    // Build failure (partial-match) table.
    let mut fail = vec![0usize; m];
    let mut k = 0usize;
    for i in 1..m {
        while k > 0 && pattern[k] != pattern[i] {
            k = fail[k - 1];
        }
        if pattern[k] == pattern[i] {
            k += 1;
        }
        fail[i] = k;
    }

    // Scan text.
    let mut results = Vec::new();
    let mut q = 0usize;
    for i in 0..n {
        while q > 0 && pattern[q] != text[i] {
            q = fail[q - 1];
        }
        if pattern[q] == text[i] {
            q += 1;
        }
        if q == m {
            results.push(i + 1 - m);
            q = fail[q - 1];
        }
    }
    results
}

// ---------------------------------------------------------------------------
// FFT (iterative Cooley-Tukey, radix-2)
// ---------------------------------------------------------------------------

pub fn baseline_fft(data: &mut [(f64, f64)]) {
    let n = data.len();
    assert!(n.is_power_of_two(), "FFT length must be a power of 2");
    if n <= 1 {
        return;
    }
    let log_n = n.trailing_zeros() as usize;

    // Bit-reversal permutation.
    for i in 0..n {
        let mut rev = 0usize;
        for bit in 0..log_n {
            if i & (1 << bit) != 0 {
                rev |= 1 << (log_n - 1 - bit);
            }
        }
        if i < rev {
            data.swap(i, rev);
        }
    }

    // Butterfly stages.
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * std::f64::consts::PI / (len as f64);
        let wn = (angle.cos(), angle.sin());
        let mut i = 0;
        while i < n {
            let mut w = (1.0_f64, 0.0_f64);
            for j in 0..half {
                let u = data[i + j];
                let t = (
                    w.0 * data[i + j + half].0 - w.1 * data[i + j + half].1,
                    w.0 * data[i + j + half].1 + w.1 * data[i + j + half].0,
                );
                data[i + j] = (u.0 + t.0, u.1 + t.1);
                data[i + j + half] = (u.0 - t.0, u.1 - t.1);
                w = (w.0 * wn.0 - w.1 * wn.1, w.0 * wn.1 + w.1 * wn.0);
            }
            i += len;
        }
        len *= 2;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Sorting ---

    #[test]
    fn test_sort_empty() {
        let mut arr: Vec<i64> = vec![];
        baseline_sort(&mut arr);
        assert!(arr.is_empty());
    }

    #[test]
    fn test_sort_single() {
        let mut arr = vec![42];
        baseline_sort(&mut arr);
        assert_eq!(arr, vec![42]);
    }

    #[test]
    fn test_sort_sorted() {
        let mut arr = vec![1, 2, 3, 4, 5];
        baseline_sort(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_sort_reverse() {
        let mut arr = vec![5, 4, 3, 2, 1];
        baseline_sort(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_sort_duplicates() {
        let mut arr = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        baseline_sort(&mut arr);
        let expected = vec![1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9];
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_sort_large() {
        let mut arr: Vec<i64> = (0..1000).rev().collect();
        baseline_sort(&mut arr);
        let expected: Vec<i64> = (0..1000).collect();
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_sort_all_same() {
        let mut arr = vec![7, 7, 7, 7, 7];
        baseline_sort(&mut arr);
        assert_eq!(arr, vec![7, 7, 7, 7, 7]);
    }

    #[test]
    fn test_sort_negative() {
        let mut arr = vec![-3, -1, -4, -1, -5, -9];
        baseline_sort(&mut arr);
        assert_eq!(arr, vec![-9, -5, -4, -3, -1, -1]);
    }

    // --- Prefix sum ---

    #[test]
    fn test_prefix_sum_basic() {
        let mut arr = vec![1, 2, 3, 4, 5];
        baseline_prefix_sum(&mut arr);
        assert_eq!(arr, vec![1, 3, 6, 10, 15]);
    }

    #[test]
    fn test_prefix_sum_empty() {
        let mut arr: Vec<i64> = vec![];
        baseline_prefix_sum(&mut arr);
        assert!(arr.is_empty());
    }

    #[test]
    fn test_prefix_sum_single() {
        let mut arr = vec![42];
        baseline_prefix_sum(&mut arr);
        assert_eq!(arr, vec![42]);
    }

    #[test]
    fn test_exclusive_prefix_sum() {
        let mut arr = vec![1, 2, 3, 4, 5];
        baseline_exclusive_prefix_sum(&mut arr);
        assert_eq!(arr, vec![0, 1, 3, 6, 10]);
    }

    // --- Connected components ---

    #[test]
    fn test_cc_no_edges() {
        let comps = baseline_connected_components(&[], 5);
        // Each vertex is its own component
        for i in 0..5 {
            assert_eq!(comps[i], i);
        }
    }

    #[test]
    fn test_cc_single_component() {
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
        let comps = baseline_connected_components(&edges, 5);
        let root = comps[0];
        assert!(comps.iter().all(|&c| c == root));
    }

    #[test]
    fn test_cc_two_components() {
        let edges = vec![(0, 1), (2, 3)];
        let comps = baseline_connected_components(&edges, 4);
        assert_eq!(comps[0], comps[1]);
        assert_eq!(comps[2], comps[3]);
        assert_ne!(comps[0], comps[2]);
    }

    #[test]
    fn test_cc_cycle() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let comps = baseline_connected_components(&edges, 3);
        assert_eq!(comps[0], comps[1]);
        assert_eq!(comps[1], comps[2]);
    }

    // --- List ranking ---

    #[test]
    fn test_list_ranking_simple() {
        // List: 0 -> 1 -> 2 -> 3 (3 is tail)
        let next = vec![1, 2, 3, 3];
        let weights = vec![1, 2, 3, 4];
        let result = baseline_list_ranking(&next, &weights);
        // suffix sums: [1+2+3+4, 2+3+4, 3+4, 4] = [10, 9, 7, 4]
        assert_eq!(result, vec![10, 9, 7, 4]);
    }

    #[test]
    fn test_list_ranking_single() {
        let next = vec![0]; // self-loop
        let weights = vec![5];
        let result = baseline_list_ranking(&next, &weights);
        assert_eq!(result, vec![5]);
    }

    #[test]
    fn test_list_ranking_empty() {
        let result = baseline_list_ranking(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_list_ranking_two_lists() {
        // List 1: 0 -> 1 (1 is tail)
        // List 2: 2 -> 3 (3 is tail)
        let next = vec![1, 1, 3, 3];
        let weights = vec![10, 20, 30, 40];
        let result = baseline_list_ranking(&next, &weights);
        assert_eq!(result[0], 30); // 10 + 20
        assert_eq!(result[1], 20);
        assert_eq!(result[2], 70); // 30 + 40
        assert_eq!(result[3], 40);
    }

    // --- Matrix multiply ---

    #[test]
    fn test_matmul_identity() {
        // I × A = A
        let n = 3;
        let identity = vec![1, 0, 0, 0, 1, 0, 0, 0, 1];
        let a = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let result = baseline_matrix_multiply(&identity, &a, n);
        assert_eq!(result, a);
    }

    #[test]
    fn test_matmul_2x2() {
        // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let result = baseline_matrix_multiply(&a, &b, 2);
        assert_eq!(result, vec![19, 22, 43, 50]);
    }

    #[test]
    fn test_matmul_zeros() {
        let n = 3;
        let zero = vec![0i64; n * n];
        let a = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let result = baseline_matrix_multiply(&a, &zero, n);
        assert_eq!(result, zero);
    }

    #[test]
    fn test_matmul_1x1() {
        let result = baseline_matrix_multiply(&[3], &[7], 1);
        assert_eq!(result, vec![21]);
    }

    // --- Binary search ---

    #[test]
    fn test_binary_search_found() {
        let arr = vec![1, 3, 5, 7, 9, 11, 13, 15];
        assert_eq!(baseline_binary_search(&arr, 7), Some(3));
        assert_eq!(baseline_binary_search(&arr, 1), Some(0));
        assert_eq!(baseline_binary_search(&arr, 15), Some(7));
    }

    #[test]
    fn test_binary_search_not_found() {
        let arr = vec![1, 3, 5, 7, 9];
        assert_eq!(baseline_binary_search(&arr, 4), None);
        assert_eq!(baseline_binary_search(&arr, 0), None);
        assert_eq!(baseline_binary_search(&arr, 10), None);
    }

    #[test]
    fn test_binary_search_empty() {
        assert_eq!(baseline_binary_search(&[], 5), None);
    }

    #[test]
    fn test_binary_search_single() {
        assert_eq!(baseline_binary_search(&[42], 42), Some(0));
        assert_eq!(baseline_binary_search(&[42], 43), None);
    }

    #[test]
    fn test_binary_search_large() {
        let arr: Vec<i64> = (0..10000).map(|i| i * 2).collect();
        assert_eq!(baseline_binary_search(&arr, 5000), Some(2500));
        assert_eq!(baseline_binary_search(&arr, 5001), None);
    }

    // --- BFS ---

    #[test]
    fn test_bfs_simple() {
        // 0 -- 1 -- 2 -- 3
        let adj = vec![
            vec![1],
            vec![0, 2],
            vec![1, 3],
            vec![2],
        ];
        let dist = baseline_bfs(&adj, 0, 4);
        assert_eq!(dist, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_bfs_disconnected() {
        // 0 -- 1, 2 -- 3 (two components)
        let adj = vec![vec![1], vec![0], vec![3], vec![2]];
        let dist = baseline_bfs(&adj, 0, 4);
        assert_eq!(dist[0], 0);
        assert_eq!(dist[1], 1);
        assert_eq!(dist[2], -1);
        assert_eq!(dist[3], -1);
    }

    #[test]
    fn test_bfs_cycle() {
        // 0 -- 1 -- 2 -- 0
        let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let dist = baseline_bfs(&adj, 0, 3);
        assert_eq!(dist, vec![0, 1, 1]);
    }

    #[test]
    fn test_bfs_single_node() {
        let adj = vec![vec![]];
        let dist = baseline_bfs(&adj, 0, 1);
        assert_eq!(dist, vec![0]);
    }

    #[test]
    fn test_bfs_star_graph() {
        // 0 connected to 1, 2, 3, 4
        let adj = vec![
            vec![1, 2, 3, 4],
            vec![0],
            vec![0],
            vec![0],
            vec![0],
        ];
        let dist = baseline_bfs(&adj, 0, 5);
        assert_eq!(dist, vec![0, 1, 1, 1, 1]);
    }

    // --- Merge Sort ---

    #[test]
    fn test_merge_sort_basic() {
        let mut arr = vec![5, 3, 8, 1, 2, 7, 4, 6];
        baseline_merge_sort(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_merge_sort_empty_and_single() {
        let mut empty: Vec<i64> = vec![];
        baseline_merge_sort(&mut empty);
        assert!(empty.is_empty());

        let mut single = vec![42];
        baseline_merge_sort(&mut single);
        assert_eq!(single, vec![42]);
    }

    // --- Convex Hull ---

    #[test]
    fn test_convex_hull_square() {
        let points = vec![(0, 0), (1, 0), (1, 1), (0, 1)];
        let hull = baseline_convex_hull(&points);
        assert_eq!(hull.len(), 4);
        // All four points are on the hull
        for p in &points {
            assert!(hull.contains(p));
        }
    }

    #[test]
    fn test_convex_hull_with_interior() {
        let points = vec![(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)];
        let hull = baseline_convex_hull(&points);
        assert_eq!(hull.len(), 4);
        assert!(!hull.contains(&(2, 2)));
    }

    #[test]
    fn test_convex_hull_fewer_than_3() {
        assert_eq!(baseline_convex_hull(&[]), vec![]);
        assert_eq!(baseline_convex_hull(&[(1, 2)]), vec![(1, 2)]);
        let hull2 = baseline_convex_hull(&[(0, 0), (1, 1)]);
        assert_eq!(hull2.len(), 2);
    }

    // --- KMP String Matching ---

    #[test]
    fn test_string_match_basic() {
        let text = b"abcabcabc";
        let pat = b"abc";
        assert_eq!(baseline_string_match(text, pat), vec![0, 3, 6]);
    }

    #[test]
    fn test_string_match_no_match() {
        assert_eq!(baseline_string_match(b"hello", b"xyz"), Vec::<usize>::new());
        assert_eq!(baseline_string_match(b"ab", b"abc"), Vec::<usize>::new());
        assert_eq!(baseline_string_match(b"test", b""), Vec::<usize>::new());
    }

    #[test]
    fn test_string_match_overlapping() {
        assert_eq!(baseline_string_match(b"aaaa", b"aa"), vec![0, 1, 2]);
    }

    // --- FFT ---

    #[test]
    fn test_fft_impulse() {
        // FFT of [1, 0, 0, 0] should be [1, 1, 1, 1]
        let mut data = vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
        baseline_fft(&mut data);
        for (re, im) in &data {
            assert!((re - 1.0).abs() < 1e-9, "real part should be 1.0, got {re}");
            assert!(im.abs() < 1e-9, "imag part should be 0.0, got {im}");
        }
    }

    #[test]
    fn test_fft_constant() {
        // FFT of [3, 3, 3, 3] should be [12, 0, 0, 0]
        let mut data = vec![(3.0, 0.0); 4];
        baseline_fft(&mut data);
        assert!((data[0].0 - 12.0).abs() < 1e-9);
        for i in 1..4 {
            assert!(data[i].0.abs() < 1e-9);
            assert!(data[i].1.abs() < 1e-9);
        }
    }
}
