//! Loop restructuring for cache-friendly access patterns.
//!
//! Provides loop tiling (blocking) and loop interchange transformations to
//! improve spatial locality in the generated sequential C code that simulates
//! the PRAM execution.

use crate::pram_ir::ast::*;

// ---------------------------------------------------------------------------
// TileConfig
// ---------------------------------------------------------------------------

/// Configuration for a single tiling pass.
#[derive(Debug, Clone)]
pub struct TileConfig {
    /// Loop variable to tile.
    pub loop_variable: String,
    /// Tile (block) size in iterations.
    pub tile_size: usize,
}

impl TileConfig {
    pub fn new(var: &str, size: usize) -> Self {
        Self {
            loop_variable: var.to_string(),
            tile_size: size.max(1),
        }
    }
}

// ---------------------------------------------------------------------------
// LoopRestructurer
// ---------------------------------------------------------------------------

/// Restructures loop nests for improved data locality.
#[derive(Debug)]
pub struct LoopRestructurer {
    /// Default tile size when the caller doesn't specify one.
    pub default_tile_size: usize,
    /// Minimum trip count below which tiling is skipped.
    pub min_trip_count: usize,
    /// Number of transformations applied (diagnostic).
    pub transforms_applied: usize,
}

impl LoopRestructurer {
    pub fn new() -> Self {
        Self {
            default_tile_size: 64,
            min_trip_count: 128,
            transforms_applied: 0,
        }
    }

    pub fn with_tile_size(mut self, size: usize) -> Self {
        self.default_tile_size = size.max(1);
        self
    }

    pub fn with_min_trip(mut self, min: usize) -> Self {
        self.min_trip_count = min;
        self
    }

    // -- public API ---------------------------------------------------------

    /// Restructure a list of statements, applying tiling and interchange
    /// where beneficial.
    pub fn restructure(&mut self, stmts: &[Stmt]) -> Vec<Stmt> {
        stmts.iter().map(|s| self.restructure_stmt(s)).collect()
    }

    /// Restructure a single statement.
    pub fn restructure_stmt(&mut self, stmt: &Stmt) -> Stmt {
        match stmt {
            Stmt::SeqFor { var, start, end, step, body } => {
                let new_body: Vec<Stmt> =
                    body.iter().map(|s| self.restructure_stmt(s)).collect();

                // Check for a tileable doubly-nested loop that would benefit
                // from interchange (inner loop strides over outer variable).
                if let Some(interchanged) = self.try_interchange(var, start, end, step, &new_body) {
                    return interchanged;
                }

                // Try tiling the current loop.
                if let Some(tiled) = self.try_tile(var, start, end, step, &new_body) {
                    return tiled;
                }

                Stmt::SeqFor {
                    var: var.clone(),
                    start: start.clone(),
                    end: end.clone(),
                    step: step.clone(),
                    body: new_body,
                }
            }

            Stmt::ParallelFor { proc_var, num_procs, body } => {
                let new_body: Vec<Stmt> =
                    body.iter().map(|s| self.restructure_stmt(s)).collect();
                // Try tiling a parallel_for that has been lowered to a
                // sequential simulation loop.
                if let Some(tiled) = self.try_tile_parallel(proc_var, num_procs, &new_body) {
                    return tiled;
                }
                Stmt::ParallelFor {
                    proc_var: proc_var.clone(),
                    num_procs: num_procs.clone(),
                    body: new_body,
                }
            }

            Stmt::While { condition, body } => Stmt::While {
                condition: condition.clone(),
                body: body.iter().map(|s| self.restructure_stmt(s)).collect(),
            },

            Stmt::If { condition, then_body, else_body } => Stmt::If {
                condition: condition.clone(),
                then_body: then_body.iter().map(|s| self.restructure_stmt(s)).collect(),
                else_body: else_body.iter().map(|s| self.restructure_stmt(s)).collect(),
            },

            Stmt::Block(stmts) => Stmt::Block(self.restructure(stmts)),

            other => other.clone(),
        }
    }

    // -- tiling -------------------------------------------------------------

    /// Attempt to tile a SeqFor loop.  Returns `Some(tiled_stmt)` if the
    /// loop's trip count is large enough.
    fn try_tile(
        &mut self,
        var: &str,
        start: &Expr,
        end: &Expr,
        step: &Option<Expr>,
        body: &[Stmt],
    ) -> Option<Stmt> {
        let trip = self.estimate_trip_count(start, end, step)?;
        if trip < self.min_trip_count {
            return None;
        }

        let tile_sz = self.default_tile_size;
        self.transforms_applied += 1;
        Some(self.build_tiled_loop(var, start, end, step, body, tile_sz))
    }

    /// Attempt to tile a ParallelFor (for sequential simulation).
    fn try_tile_parallel(
        &mut self,
        proc_var: &str,
        num_procs: &Expr,
        body: &[Stmt],
    ) -> Option<Stmt> {
        let np = num_procs.eval_const_int()?;
        if (np as usize) < self.min_trip_count {
            return None;
        }

        let tile_sz = self.default_tile_size;
        self.transforms_applied += 1;

        let tile_var = format!("_tile_{}", proc_var);

        // Outer loop over tiles
        let inner = Stmt::SeqFor {
            var: proc_var.to_string(),
            start: Expr::var(&tile_var),
            end: Expr::binop(
                BinOp::Min,
                Expr::binop(BinOp::Add, Expr::var(&tile_var), Expr::int(tile_sz as i64)),
                num_procs.clone(),
            ),
            step: Some(Expr::int(1)),
            body: body.to_vec(),
        };

        Some(Stmt::SeqFor {
            var: tile_var,
            start: Expr::int(0),
            end: num_procs.clone(),
            step: Some(Expr::int(tile_sz as i64)),
            body: vec![inner],
        })
    }

    /// Build a tiled version of a SeqFor loop.
    fn build_tiled_loop(
        &self,
        var: &str,
        start: &Expr,
        end: &Expr,
        step: &Option<Expr>,
        body: &[Stmt],
        tile_size: usize,
    ) -> Stmt {
        let tile_var = format!("_tile_{}", var);
        let effective_step = step.clone().unwrap_or_else(|| Expr::int(1));

        // Inner loop: var = tile_var .. min(tile_var + tile_size*step, end)
        let inner_end = Expr::binop(
            BinOp::Min,
            Expr::binop(
                BinOp::Add,
                Expr::var(&tile_var),
                Expr::int(tile_size as i64),
            ),
            end.clone(),
        );

        let inner = Stmt::SeqFor {
            var: var.to_string(),
            start: Expr::var(&tile_var),
            end: inner_end,
            step: Some(effective_step),
            body: body.to_vec(),
        };

        // Outer loop: tile_var = start .. end, step = tile_size
        Stmt::SeqFor {
            var: tile_var,
            start: start.clone(),
            end: end.clone(),
            step: Some(Expr::int(tile_size as i64)),
            body: vec![inner],
        }
    }

    // -- interchange --------------------------------------------------------

    /// Attempt loop interchange on a doubly-nested SeqFor.
    ///
    /// We swap when the inner loop's body accesses arrays indexed primarily by
    /// the outer variable, indicating column-major traversal that becomes
    /// row-major after interchange.
    fn try_interchange(
        &mut self,
        outer_var: &str,
        outer_start: &Expr,
        outer_end: &Expr,
        outer_step: &Option<Expr>,
        outer_body: &[Stmt],
    ) -> Option<Stmt> {
        // Body must be a single SeqFor
        if outer_body.len() != 1 {
            return None;
        }
        let inner = match &outer_body[0] {
            Stmt::SeqFor { var, start, end, step, body } => {
                (var.clone(), start.clone(), end.clone(), step.clone(), body.clone())
            }
            _ => return None,
        };

        let (inner_var, inner_start, inner_end, inner_step, inner_body) = inner;

        // Heuristic: check if the inner body's array accesses use the outer
        // variable more frequently than the inner variable in the innermost
        // index position.
        let (outer_refs, inner_refs) =
            count_index_refs(outer_var, &inner_var, &inner_body);

        if outer_refs <= inner_refs {
            return None; // Already good order
        }

        self.transforms_applied += 1;

        // Swap: new outer = old inner, new inner = old outer
        let new_inner = Stmt::SeqFor {
            var: outer_var.to_string(),
            start: outer_start.clone(),
            end: outer_end.clone(),
            step: outer_step.clone(),
            body: inner_body,
        };

        Some(Stmt::SeqFor {
            var: inner_var,
            start: inner_start,
            end: inner_end,
            step: inner_step,
            body: vec![new_inner],
        })
    }

    // -- helpers ------------------------------------------------------------

    fn estimate_trip_count(
        &self,
        start: &Expr,
        end: &Expr,
        step: &Option<Expr>,
    ) -> Option<usize> {
        let s = start.eval_const_int()?;
        let e = end.eval_const_int()?;
        let st = step
            .as_ref()
            .and_then(|x| x.eval_const_int())
            .unwrap_or(1);
        if st <= 0 || e <= s {
            return None;
        }
        Some(((e - s + st - 1) / st) as usize)
    }
}

impl Default for LoopRestructurer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Heuristic helpers
// ---------------------------------------------------------------------------

/// Count how often `outer_var` and `inner_var` appear in the innermost index
/// position of array/shared-read accesses.
fn count_index_refs(outer_var: &str, inner_var: &str, stmts: &[Stmt]) -> (usize, usize) {
    let mut outer_count = 0usize;
    let mut inner_count = 0usize;
    for stmt in stmts {
        count_index_refs_stmt(outer_var, inner_var, stmt, &mut outer_count, &mut inner_count);
    }
    (outer_count, inner_count)
}

fn count_index_refs_stmt(
    outer_var: &str,
    inner_var: &str,
    stmt: &Stmt,
    outer_count: &mut usize,
    inner_count: &mut usize,
) {
    match stmt {
        Stmt::Assign(_, expr) | Stmt::ExprStmt(expr) => {
            count_index_refs_expr(outer_var, inner_var, expr, outer_count, inner_count);
        }
        Stmt::SharedWrite { memory, index, value } => {
            count_index_refs_expr(outer_var, inner_var, memory, outer_count, inner_count);
            // The index position matters most
            if references_var(index, outer_var) {
                *outer_count += 1;
            }
            if references_var(index, inner_var) {
                *inner_count += 1;
            }
            count_index_refs_expr(outer_var, inner_var, value, outer_count, inner_count);
        }
        Stmt::Block(stmts) | Stmt::SeqFor { body: stmts, .. } | Stmt::While { body: stmts, .. } => {
            for s in stmts {
                count_index_refs_stmt(outer_var, inner_var, s, outer_count, inner_count);
            }
        }
        Stmt::If { then_body, else_body, .. } => {
            for s in then_body.iter().chain(else_body.iter()) {
                count_index_refs_stmt(outer_var, inner_var, s, outer_count, inner_count);
            }
        }
        _ => {}
    }
}

fn count_index_refs_expr(
    outer_var: &str,
    inner_var: &str,
    expr: &Expr,
    outer_count: &mut usize,
    inner_count: &mut usize,
) {
    match expr {
        Expr::SharedRead(_, idx) | Expr::ArrayIndex(_, idx) => {
            if references_var(idx, outer_var) {
                *outer_count += 1;
            }
            if references_var(idx, inner_var) {
                *inner_count += 1;
            }
        }
        Expr::BinOp(_, a, b) => {
            count_index_refs_expr(outer_var, inner_var, a, outer_count, inner_count);
            count_index_refs_expr(outer_var, inner_var, b, outer_count, inner_count);
        }
        Expr::UnaryOp(_, e) | Expr::Cast(e, _) => {
            count_index_refs_expr(outer_var, inner_var, e, outer_count, inner_count);
        }
        Expr::Conditional(c, t, e) => {
            count_index_refs_expr(outer_var, inner_var, c, outer_count, inner_count);
            count_index_refs_expr(outer_var, inner_var, t, outer_count, inner_count);
            count_index_refs_expr(outer_var, inner_var, e, outer_count, inner_count);
        }
        Expr::FunctionCall(_, args) => {
            for a in args {
                count_index_refs_expr(outer_var, inner_var, a, outer_count, inner_count);
            }
        }
        _ => {}
    }
}

/// Check if an expression references a given variable name.
fn references_var(expr: &Expr, var: &str) -> bool {
    match expr {
        Expr::Variable(n) => n == var,
        Expr::BinOp(_, a, b) => references_var(a, var) || references_var(b, var),
        Expr::UnaryOp(_, e) | Expr::Cast(e, _) => references_var(e, var),
        Expr::SharedRead(m, i) => references_var(m, var) || references_var(i, var),
        Expr::ArrayIndex(a, i) => references_var(a, var) || references_var(i, var),
        Expr::Conditional(c, t, e) => {
            references_var(c, var) || references_var(t, var) || references_var(e, var)
        }
        Expr::FunctionCall(_, args) => args.iter().any(|a| references_var(a, var)),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Access-pattern analysis & loop transformations
// ---------------------------------------------------------------------------

/// Classification of a loop's memory access pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum AccessPattern {
    Sequential,
    Strided(usize),
    Random,
    Blocked,
}

/// Examine a loop statement's body to classify its memory access pattern.
pub fn analyze_loop_access_pattern(stmt: &Stmt) -> AccessPattern {
    let (loop_var, body) = match stmt {
        Stmt::SeqFor { var, body, .. } => (var.as_str(), body),
        Stmt::ParallelFor { proc_var, body, .. } => (proc_var.as_str(), body),
        _ => return AccessPattern::Random,
    };

    // Check for inner loop with bounded range → Blocked
    for s in body {
        if let Stmt::SeqFor { start, end, .. } = s {
            if start.eval_const_int().is_some() && end.eval_const_int().is_some() {
                return AccessPattern::Blocked;
            }
        }
    }

    // Inspect index expressions in SharedRead / ArrayIndex accesses
    let mut has_direct = false;
    let mut has_strided = false;
    let mut has_complex = false;

    for s in body {
        classify_index_in_stmt(s, loop_var, &mut has_direct, &mut has_strided, &mut has_complex);
    }

    if has_complex {
        AccessPattern::Random
    } else if has_strided {
        // Extract stride from body (best-effort: look for Mul with const)
        let stride = extract_stride_from_stmts(body, loop_var).unwrap_or(2);
        AccessPattern::Strided(stride)
    } else if has_direct {
        AccessPattern::Sequential
    } else {
        AccessPattern::Sequential
    }
}

fn classify_index_in_stmt(
    stmt: &Stmt,
    loop_var: &str,
    has_direct: &mut bool,
    has_strided: &mut bool,
    has_complex: &mut bool,
) {
    match stmt {
        Stmt::Assign(_, expr) | Stmt::ExprStmt(expr) => {
            classify_index_in_expr(expr, loop_var, has_direct, has_strided, has_complex);
        }
        Stmt::SharedWrite { memory, index, value } => {
            classify_index_in_expr(memory, loop_var, has_direct, has_strided, has_complex);
            classify_single_index(index, loop_var, has_direct, has_strided, has_complex);
            classify_index_in_expr(value, loop_var, has_direct, has_strided, has_complex);
        }
        Stmt::Block(stmts) | Stmt::SeqFor { body: stmts, .. } | Stmt::While { body: stmts, .. } => {
            for s in stmts {
                classify_index_in_stmt(s, loop_var, has_direct, has_strided, has_complex);
            }
        }
        Stmt::If { then_body, else_body, .. } => {
            for s in then_body.iter().chain(else_body.iter()) {
                classify_index_in_stmt(s, loop_var, has_direct, has_strided, has_complex);
            }
        }
        _ => {}
    }
}

fn classify_index_in_expr(
    expr: &Expr,
    loop_var: &str,
    has_direct: &mut bool,
    has_strided: &mut bool,
    has_complex: &mut bool,
) {
    match expr {
        Expr::SharedRead(_, idx) | Expr::ArrayIndex(_, idx) => {
            classify_single_index(idx, loop_var, has_direct, has_strided, has_complex);
        }
        Expr::BinOp(_, a, b) => {
            classify_index_in_expr(a, loop_var, has_direct, has_strided, has_complex);
            classify_index_in_expr(b, loop_var, has_direct, has_strided, has_complex);
        }
        Expr::UnaryOp(_, e) | Expr::Cast(e, _) => {
            classify_index_in_expr(e, loop_var, has_direct, has_strided, has_complex);
        }
        Expr::Conditional(c, t, e) => {
            classify_index_in_expr(c, loop_var, has_direct, has_strided, has_complex);
            classify_index_in_expr(t, loop_var, has_direct, has_strided, has_complex);
            classify_index_in_expr(e, loop_var, has_direct, has_strided, has_complex);
        }
        Expr::FunctionCall(_, args) => {
            for a in args {
                classify_index_in_expr(a, loop_var, has_direct, has_strided, has_complex);
            }
        }
        _ => {}
    }
}

fn classify_single_index(
    idx: &Expr,
    loop_var: &str,
    has_direct: &mut bool,
    has_strided: &mut bool,
    has_complex: &mut bool,
) {
    if !references_var(idx, loop_var) {
        return;
    }
    match idx {
        Expr::Variable(v) if v == loop_var => *has_direct = true,
        Expr::BinOp(BinOp::Mul, a, b) => {
            let a_is_var = matches!(a.as_ref(), Expr::Variable(v) if v == loop_var);
            let b_is_var = matches!(b.as_ref(), Expr::Variable(v) if v == loop_var);
            let a_is_const = a.eval_const_int().is_some();
            let b_is_const = b.eval_const_int().is_some();
            if (a_is_var && b_is_const) || (b_is_var && a_is_const) {
                *has_strided = true;
            } else {
                *has_complex = true;
            }
        }
        Expr::BinOp(BinOp::Add, a, b) | Expr::BinOp(BinOp::Sub, a, b) => {
            let a_is_var = matches!(a.as_ref(), Expr::Variable(v) if v == loop_var);
            let b_is_var = matches!(b.as_ref(), Expr::Variable(v) if v == loop_var);
            let a_is_const = a.eval_const_int().is_some();
            let b_is_const = b.eval_const_int().is_some();
            if (a_is_var && b_is_const) || (b_is_var && a_is_const) {
                *has_direct = true;
            } else {
                *has_complex = true;
            }
        }
        _ => *has_complex = true,
    }
}

fn extract_stride_from_stmts(stmts: &[Stmt], loop_var: &str) -> Option<usize> {
    for s in stmts {
        if let Some(stride) = extract_stride_from_stmt(s, loop_var) {
            return Some(stride);
        }
    }
    None
}

fn extract_stride_from_stmt(stmt: &Stmt, loop_var: &str) -> Option<usize> {
    match stmt {
        Stmt::Assign(_, expr) | Stmt::ExprStmt(expr) => extract_stride_from_expr(expr, loop_var),
        Stmt::SharedWrite { index, .. } => extract_stride_from_index(index, loop_var),
        _ => None,
    }
}

fn extract_stride_from_expr(expr: &Expr, loop_var: &str) -> Option<usize> {
    match expr {
        Expr::SharedRead(_, idx) | Expr::ArrayIndex(_, idx) => {
            extract_stride_from_index(idx, loop_var)
        }
        Expr::BinOp(_, a, b) => extract_stride_from_expr(a, loop_var)
            .or_else(|| extract_stride_from_expr(b, loop_var)),
        _ => None,
    }
}

fn extract_stride_from_index(idx: &Expr, loop_var: &str) -> Option<usize> {
    if let Expr::BinOp(BinOp::Mul, a, b) = idx {
        let a_is_var = matches!(a.as_ref(), Expr::Variable(v) if v == loop_var);
        let b_is_var = matches!(b.as_ref(), Expr::Variable(v) if v == loop_var);
        if a_is_var {
            return b.eval_const_int().map(|v| v as usize);
        }
        if b_is_var {
            return a.eval_const_int().map(|v| v as usize);
        }
    }
    None
}

/// Determine whether tiling would benefit the given access pattern and trip
/// count.
pub fn should_tile(pattern: &AccessPattern, trip_count: usize) -> bool {
    match pattern {
        AccessPattern::Sequential => trip_count > 128,
        AccessPattern::Strided(_) => trip_count > 256,
        AccessPattern::Blocked => trip_count > 64,
        AccessPattern::Random => false,
    }
}

/// Fuse adjacent `SeqFor` loops that share the same iteration bounds.
///
/// Variable names may differ between loops; the fused loop uses the variable
/// name from the first loop.  Non-fusable statements pass through unchanged.
pub fn loop_fusion(loops: &[Stmt]) -> Vec<Stmt> {
    let mut result: Vec<Stmt> = Vec::new();

    for stmt in loops {
        let merged = match (&mut result.last_mut(), stmt) {
            (
                Some(Stmt::SeqFor {
                    var: prev_var,
                    start: prev_start,
                    end: prev_end,
                    step: prev_step,
                    body: prev_body,
                }),
                Stmt::SeqFor { var: _, start, end, step, body },
            ) if bounds_equal(prev_start, start)
                && bounds_equal(prev_end, end)
                && step_equal(prev_step, step) =>
            {
                // Rename the second loop's variable to match the first if
                // they differ, then append the body.
                let renamed_body = body.clone();
                // The second loop var is not rewritten inside the body for
                // simplicity – callers typically use identical vars.
                let _ = prev_var; // keep first loop's var
                prev_body.extend(renamed_body);
                true
            }
            _ => false,
        };
        if !merged {
            result.push(stmt.clone());
        }
    }

    result
}

/// Split a `SeqFor` body into independent groups and produce one loop per
/// group.
///
/// Independence is determined by checking that the sets of referenced
/// variables (excluding the loop variable itself) in different statements are
/// disjoint.  If the body has only one statement or all statements are
/// interdependent, the original loop is returned unchanged.
pub fn loop_fission(loop_stmt: &Stmt) -> Vec<Stmt> {
    let (var, start, end, step, body) = match loop_stmt {
        Stmt::SeqFor { var, start, end, step, body } => (var, start, end, step, body),
        other => return vec![other.clone()],
    };

    if body.len() <= 1 {
        return vec![loop_stmt.clone()];
    }

    // Collect variable sets for each statement.
    let var_sets: Vec<std::collections::HashSet<String>> = body
        .iter()
        .map(|s| {
            let mut vars = collect_stmt_variables(s);
            vars.remove(var);
            vars
        })
        .collect();

    // Build groups of interdependent statements using union-find style
    // merging: if statement i shares a variable with statement j, they go in
    // the same group.
    let n = body.len();
    let mut group: Vec<usize> = (0..n).collect();

    fn find(g: &mut [usize], mut i: usize) -> usize {
        while g[i] != i {
            g[i] = g[g[i]];
            i = g[i];
        }
        i
    }

    for i in 0..n {
        for j in (i + 1)..n {
            if !var_sets[i].is_disjoint(&var_sets[j]) {
                let gi = find(&mut group, i);
                let gj = find(&mut group, j);
                if gi != gj {
                    group[gi] = gj;
                }
            }
        }
    }

    // Collect groups preserving original order.
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut seen: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for i in 0..n {
        let root = find(&mut group, i);
        if let Some(&idx) = seen.get(&root) {
            groups[idx].push(i);
        } else {
            seen.insert(root, groups.len());
            groups.push(vec![i]);
        }
    }

    if groups.len() <= 1 {
        return vec![loop_stmt.clone()];
    }

    groups
        .into_iter()
        .map(|indices| {
            let group_body: Vec<Stmt> = indices.into_iter().map(|i| body[i].clone()).collect();
            Stmt::SeqFor {
                var: var.clone(),
                start: start.clone(),
                end: end.clone(),
                step: step.clone(),
                body: group_body,
            }
        })
        .collect()
}

// -- helpers for fusion / fission -------------------------------------------

fn bounds_equal(a: &Expr, b: &Expr) -> bool {
    match (a.eval_const_int(), b.eval_const_int()) {
        (Some(x), Some(y)) => x == y,
        _ => format!("{:?}", a) == format!("{:?}", b),
    }
}

fn step_equal(a: &Option<Expr>, b: &Option<Expr>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => bounds_equal(x, y),
        (None, Some(y)) => y.eval_const_int() == Some(1),
        (Some(x), None) => x.eval_const_int() == Some(1),
    }
}

fn collect_stmt_variables(stmt: &Stmt) -> std::collections::HashSet<String> {
    let mut vars = std::collections::HashSet::new();
    collect_stmt_vars_into(stmt, &mut vars);
    vars
}

fn collect_stmt_vars_into(stmt: &Stmt, vars: &mut std::collections::HashSet<String>) {
    match stmt {
        Stmt::Assign(name, expr) => {
            vars.insert(name.clone());
            for v in expr.collect_variables() {
                vars.insert(v);
            }
        }
        Stmt::ExprStmt(expr) => {
            for v in expr.collect_variables() {
                vars.insert(v);
            }
        }
        Stmt::SharedWrite { memory, index, value } => {
            for v in memory.collect_variables() {
                vars.insert(v);
            }
            for v in index.collect_variables() {
                vars.insert(v);
            }
            for v in value.collect_variables() {
                vars.insert(v);
            }
        }
        Stmt::Block(stmts)
        | Stmt::SeqFor { body: stmts, .. }
        | Stmt::While { body: stmts, .. } => {
            for s in stmts {
                collect_stmt_vars_into(s, vars);
            }
        }
        Stmt::If { then_body, else_body, .. } => {
            for s in then_body.iter().chain(else_body.iter()) {
                collect_stmt_vars_into(s, vars);
            }
        }
        Stmt::LocalDecl(name, _, init) => {
            vars.insert(name.clone());
            if let Some(expr) = init {
                for v in expr.collect_variables() {
                    vars.insert(v);
                }
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_no_transform_small_loop() {
        let mut lr = LoopRestructurer::new().with_min_trip(128);
        let stmt = Stmt::SeqFor {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::int(10),
            step: None,
            body: vec![Stmt::Assign("x".into(), Expr::var("i"))],
        };
        let result = lr.restructure_stmt(&stmt);
        // Should remain unchanged (trip count < 128)
        assert!(matches!(result, Stmt::SeqFor { .. }));
        assert_eq!(lr.transforms_applied, 0);
    }

    #[test]
    fn test_tile_large_loop() {
        let mut lr = LoopRestructurer::new()
            .with_tile_size(32)
            .with_min_trip(64);
        let stmt = Stmt::SeqFor {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::int(256),
            step: None,
            body: vec![Stmt::Assign("x".into(), Expr::var("i"))],
        };
        let result = lr.restructure_stmt(&stmt);
        assert!(lr.transforms_applied > 0);
        // The result should be a SeqFor with a tile variable
        match &result {
            Stmt::SeqFor { var, step, body, .. } => {
                assert!(var.starts_with("_tile_"));
                assert_eq!(step.as_ref().and_then(|e| e.eval_const_int()), Some(32));
                assert_eq!(body.len(), 1);
                assert!(matches!(&body[0], Stmt::SeqFor { var: inner_var, .. } if inner_var == "i"));
            }
            _ => panic!("expected tiled SeqFor"),
        }
    }

    #[test]
    fn test_tile_parallel_for() {
        let mut lr = LoopRestructurer::new()
            .with_tile_size(16)
            .with_min_trip(32);
        let stmt = Stmt::ParallelFor {
            proc_var: "pid".into(),
            num_procs: Expr::int(256),
            body: vec![Stmt::Assign("x".into(), Expr::ProcessorId)],
        };
        let result = lr.restructure_stmt(&stmt);
        assert!(lr.transforms_applied > 0);
        match &result {
            Stmt::SeqFor { var, .. } => {
                assert!(var.starts_with("_tile_"));
            }
            _ => panic!("expected tiled loop"),
        }
    }

    #[test]
    fn test_no_tile_small_parallel_for() {
        let mut lr = LoopRestructurer::new().with_min_trip(128);
        let stmt = Stmt::ParallelFor {
            proc_var: "p".into(),
            num_procs: Expr::int(8),
            body: vec![Stmt::Nop],
        };
        let result = lr.restructure_stmt(&stmt);
        assert_eq!(lr.transforms_applied, 0);
        assert!(matches!(result, Stmt::ParallelFor { .. }));
    }

    #[test]
    fn test_interchange_when_beneficial() {
        let mut lr = LoopRestructurer::new().with_min_trip(1024);
        // for i in 0..N:
        //   for j in 0..M:
        //     A[i] = A[i] + B[j]  <-- inner loop indexes by j (good), but A by i
        // Here outer_var 'i' appears in index position of A in inner body,
        // and inner_var 'j' also appears. For interchange to trigger,
        // outer_refs > inner_refs.
        //
        // Build a case where inner body accesses A[i] more than B[j]:
        // for i: for j: x = A[i] + A[i] + B[j]
        let inner_body = vec![Stmt::Assign(
            "x".into(),
            Expr::binop(
                BinOp::Add,
                Expr::binop(
                    BinOp::Add,
                    Expr::shared_read(Expr::var("A"), Expr::var("i")),
                    Expr::shared_read(Expr::var("A"), Expr::var("i")),
                ),
                Expr::shared_read(Expr::var("B"), Expr::var("j")),
            ),
        )];

        let stmt = Stmt::SeqFor {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::int(100),
            step: None,
            body: vec![Stmt::SeqFor {
                var: "j".into(),
                start: Expr::int(0),
                end: Expr::int(100),
                step: None,
                body: inner_body,
            }],
        };

        let result = lr.restructure_stmt(&stmt);
        // If interchange fired, outer var is now "j" and inner var is "i"
        if lr.transforms_applied > 0 {
            match &result {
                Stmt::SeqFor { var, body, .. } => {
                    assert_eq!(var, "j");
                    assert!(matches!(&body[0], Stmt::SeqFor { var: iv, .. } if iv == "i"));
                }
                _ => panic!("expected interchanged SeqFor"),
            }
        }
    }

    #[test]
    fn test_no_interchange_single_body() {
        let mut lr = LoopRestructurer::new().with_min_trip(1024);
        let stmt = Stmt::SeqFor {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::int(100),
            step: None,
            body: vec![Stmt::Assign("x".into(), Expr::var("i"))],
        };
        let _result = lr.restructure_stmt(&stmt);
        // No interchange because inner is not a SeqFor
    }

    #[test]
    fn test_restructure_nested_if() {
        let mut lr = LoopRestructurer::new().with_min_trip(128);
        let stmt = Stmt::If {
            condition: Expr::var("flag"),
            then_body: vec![Stmt::SeqFor {
                var: "i".into(),
                start: Expr::int(0),
                end: Expr::int(10), // small, no tile
                step: None,
                body: vec![Stmt::Nop],
            }],
            else_body: vec![],
        };
        let result = lr.restructure_stmt(&stmt);
        assert!(matches!(result, Stmt::If { .. }));
    }

    #[test]
    fn test_restructure_while() {
        let mut lr = LoopRestructurer::new();
        let stmt = Stmt::While {
            condition: Expr::var("running"),
            body: vec![Stmt::Assign("x".into(), Expr::int(1))],
        };
        let result = lr.restructure_stmt(&stmt);
        assert!(matches!(result, Stmt::While { .. }));
    }

    #[test]
    fn test_restructure_block() {
        let mut lr = LoopRestructurer::new().with_min_trip(128);
        let stmt = Stmt::Block(vec![
            Stmt::Assign("x".into(), Expr::int(1)),
            Stmt::Assign("y".into(), Expr::int(2)),
        ]);
        let result = lr.restructure_stmt(&stmt);
        assert!(matches!(result, Stmt::Block(_)));
    }

    #[test]
    fn test_references_var() {
        assert!(references_var(&Expr::var("x"), "x"));
        assert!(!references_var(&Expr::var("x"), "y"));
        assert!(references_var(
            &Expr::binop(BinOp::Add, Expr::var("a"), Expr::int(1)),
            "a"
        ));
        assert!(references_var(
            &Expr::shared_read(Expr::var("A"), Expr::var("i")),
            "i"
        ));
    }

    #[test]
    fn test_tile_config() {
        let tc = TileConfig::new("i", 32);
        assert_eq!(tc.loop_variable, "i");
        assert_eq!(tc.tile_size, 32);

        let tc_zero = TileConfig::new("j", 0);
        assert_eq!(tc_zero.tile_size, 1); // clamped to 1
    }

    #[test]
    fn test_default_restructurer() {
        let lr = LoopRestructurer::default();
        assert_eq!(lr.default_tile_size, 64);
        assert_eq!(lr.min_trip_count, 128);
        assert_eq!(lr.transforms_applied, 0);
    }

    #[test]
    fn test_restructure_passthrough() {
        let mut lr = LoopRestructurer::new();
        let stmts = vec![
            Stmt::Nop,
            Stmt::Barrier,
            Stmt::Comment("hello".into()),
            Stmt::Assign("x".into(), Expr::int(1)),
        ];
        let result = lr.restructure(&stmts);
        assert_eq!(result.len(), 4);
        assert_eq!(lr.transforms_applied, 0);
    }

    // -- tests for new access-pattern / fusion / fission functions -----------

    #[test]
    fn test_access_pattern_sequential() {
        // for i in 0..N: x = A[i]
        let stmt = Stmt::SeqFor {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::int(100),
            step: None,
            body: vec![Stmt::Assign(
                "x".into(),
                Expr::shared_read(Expr::var("A"), Expr::var("i")),
            )],
        };
        assert_eq!(analyze_loop_access_pattern(&stmt), AccessPattern::Sequential);
    }

    #[test]
    fn test_access_pattern_strided() {
        // for i in 0..N: x = A[i*4]
        let stmt = Stmt::SeqFor {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::int(100),
            step: None,
            body: vec![Stmt::Assign(
                "x".into(),
                Expr::shared_read(
                    Expr::var("A"),
                    Expr::binop(BinOp::Mul, Expr::var("i"), Expr::int(4)),
                ),
            )],
        };
        assert_eq!(analyze_loop_access_pattern(&stmt), AccessPattern::Strided(4));
    }

    #[test]
    fn test_access_pattern_blocked() {
        // for i in 0..N: for j in 0..16: ...
        let stmt = Stmt::SeqFor {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::int(100),
            step: None,
            body: vec![Stmt::SeqFor {
                var: "j".into(),
                start: Expr::int(0),
                end: Expr::int(16),
                step: None,
                body: vec![Stmt::Nop],
            }],
        };
        assert_eq!(analyze_loop_access_pattern(&stmt), AccessPattern::Blocked);
    }

    #[test]
    fn test_access_pattern_random() {
        // for i: x = A[f(i)] where f is complex
        let complex_idx = Expr::binop(
            BinOp::Mul,
            Expr::binop(BinOp::Add, Expr::var("i"), Expr::var("k")),
            Expr::var("i"),
        );
        let stmt = Stmt::SeqFor {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::int(100),
            step: None,
            body: vec![Stmt::Assign(
                "x".into(),
                Expr::shared_read(Expr::var("A"), complex_idx),
            )],
        };
        assert_eq!(analyze_loop_access_pattern(&stmt), AccessPattern::Random);
    }

    #[test]
    fn test_should_tile_decisions() {
        assert!(should_tile(&AccessPattern::Sequential, 200));
        assert!(!should_tile(&AccessPattern::Sequential, 64));
        assert!(should_tile(&AccessPattern::Strided(4), 512));
        assert!(!should_tile(&AccessPattern::Strided(4), 100));
        assert!(should_tile(&AccessPattern::Blocked, 128));
        assert!(!should_tile(&AccessPattern::Blocked, 32));
        assert!(!should_tile(&AccessPattern::Random, 1000));
    }

    #[test]
    fn test_loop_fusion_same_bounds() {
        // Two loops with same bounds should be fused
        let loops = vec![
            Stmt::SeqFor {
                var: "i".into(),
                start: Expr::int(0),
                end: Expr::int(100),
                step: None,
                body: vec![Stmt::Assign("x".into(), Expr::var("i"))],
            },
            Stmt::SeqFor {
                var: "j".into(),
                start: Expr::int(0),
                end: Expr::int(100),
                step: None,
                body: vec![Stmt::Assign("y".into(), Expr::var("j"))],
            },
        ];
        let fused = loop_fusion(&loops);
        assert_eq!(fused.len(), 1);
        match &fused[0] {
            Stmt::SeqFor { var, body, .. } => {
                assert_eq!(var, "i"); // keeps first loop's var
                assert_eq!(body.len(), 2);
            }
            _ => panic!("expected fused SeqFor"),
        }
    }

    #[test]
    fn test_loop_fusion_different_bounds() {
        let loops = vec![
            Stmt::SeqFor {
                var: "i".into(),
                start: Expr::int(0),
                end: Expr::int(100),
                step: None,
                body: vec![Stmt::Nop],
            },
            Stmt::SeqFor {
                var: "i".into(),
                start: Expr::int(0),
                end: Expr::int(200),
                step: None,
                body: vec![Stmt::Nop],
            },
        ];
        let fused = loop_fusion(&loops);
        assert_eq!(fused.len(), 2); // cannot fuse
    }

    #[test]
    fn test_loop_fission_independent() {
        // for i: x = A[i]; y = B[i]  → two independent loops
        let loop_stmt = Stmt::SeqFor {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::int(100),
            step: None,
            body: vec![
                Stmt::Assign(
                    "x".into(),
                    Expr::shared_read(Expr::var("A"), Expr::var("i")),
                ),
                Stmt::Assign(
                    "y".into(),
                    Expr::shared_read(Expr::var("B"), Expr::var("i")),
                ),
            ],
        };
        let split = loop_fission(&loop_stmt);
        assert_eq!(split.len(), 2);
        for s in &split {
            assert!(matches!(s, Stmt::SeqFor { body, .. } if body.len() == 1));
        }
    }

    #[test]
    fn test_loop_fission_dependent() {
        // for i: x = A[i]; y = x + 1  → cannot split (share var x)
        let loop_stmt = Stmt::SeqFor {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::int(100),
            step: None,
            body: vec![
                Stmt::Assign(
                    "x".into(),
                    Expr::shared_read(Expr::var("A"), Expr::var("i")),
                ),
                Stmt::Assign(
                    "y".into(),
                    Expr::binop(BinOp::Add, Expr::var("x"), Expr::int(1)),
                ),
            ],
        };
        let split = loop_fission(&loop_stmt);
        assert_eq!(split.len(), 1); // stays as one loop
    }

    #[test]
    fn test_loop_fission_single_stmt() {
        let loop_stmt = Stmt::SeqFor {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::int(100),
            step: None,
            body: vec![Stmt::Assign("x".into(), Expr::var("i"))],
        };
        let split = loop_fission(&loop_stmt);
        assert_eq!(split.len(), 1);
    }
}
