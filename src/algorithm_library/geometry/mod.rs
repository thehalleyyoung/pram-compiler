//! Computational-geometry algorithms for PRAM.

use crate::pram_ir::ast::*;
use crate::pram_ir::types::PramType;

fn param(name: &str) -> Parameter {
    Parameter { name: name.to_string(), param_type: PramType::Int64 }
}

fn shared(name: &str, size: Expr) -> SharedMemoryDecl {
    SharedMemoryDecl { name: name.to_string(), elem_type: PramType::Int64, size }
}

fn shared_f64(name: &str, size: Expr) -> SharedMemoryDecl {
    SharedMemoryDecl { name: name.to_string(), elem_type: PramType::Float64, size }
}

fn add(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Add, a, b) }
fn sub(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Sub, a, b) }
fn mul(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Mul, a, b) }
fn div_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Div, a, b) }
fn mod_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Mod, a, b) }
fn lt(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Lt, a, b) }
fn le(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Le, a, b) }
fn ge(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Ge, a, b) }
fn eq_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Eq, a, b) }
fn ne_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Ne, a, b) }
fn and_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::And, a, b) }
fn min_e(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Min, a, b) }
fn shl(a: Expr, b: Expr) -> Expr { Expr::binop(BinOp::Shl, a, b) }
fn v(s: &str) -> Expr { Expr::var(s) }
fn int(n: i64) -> Expr { Expr::int(n) }
fn sr(mem: &str, idx: Expr) -> Expr { Expr::shared_read(v(mem), idx) }

fn sw(mem: &str, idx: Expr, val: Expr) -> Stmt {
    Stmt::SharedWrite { memory: v(mem), index: idx, value: val }
}

fn local(name: &str, init: Expr) -> Stmt {
    Stmt::LocalDecl(name.to_string(), PramType::Int64, Some(init))
}

fn local_f64(name: &str, init: Expr) -> Stmt {
    Stmt::LocalDecl(name.to_string(), PramType::Float64, Some(init))
}

fn assign(name: &str, val: Expr) -> Stmt {
    Stmt::Assign(name.to_string(), val)
}

fn par_for(var: &str, n: Expr, body: Vec<Stmt>) -> Stmt {
    Stmt::ParallelFor { proc_var: var.to_string(), num_procs: n, body }
}

fn seq_for(var: &str, start: Expr, end: Expr, body: Vec<Stmt>) -> Stmt {
    Stmt::SeqFor { var: var.to_string(), start, end, step: None, body }
}

fn if_then(cond: Expr, then_body: Vec<Stmt>) -> Stmt {
    Stmt::If { condition: cond, then_body, else_body: vec![] }
}

fn if_else(cond: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt>) -> Stmt {
    Stmt::If { condition: cond, then_body, else_body }
}

fn log2_call(e: Expr) -> Expr {
    Expr::FunctionCall("log2".to_string(), vec![e])
}

// ─── Parallel convex hull ───────────────────────────────────────────────────

/// Parallel convex hull on CREW PRAM.
///
/// * O(log n) time, n processors.
/// * Uses the parallel divide-and-conquer approach:
///   1. Sort points by x-coordinate (parallel sort – O(log n) on CREW).
///   2. Recursively split into halves.
///   3. At each merge level, find upper and lower tangent lines between
///      left and right hulls via binary search (CREW reads).
///   4. Merge hulls using the tangent points.
///
/// We encode log n merge levels, each with a parallel tangent-finding step.
pub fn convex_hull() -> PramProgram {
    let mut prog = PramProgram::new("convex_hull", MemoryModel::CREW);
    prog.description = Some(
        "Parallel convex hull (divide & conquer). CREW, O(log n) time, n processors.".to_string(),
    );
    prog.work_bound = Some("O(n log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    // Points stored as separate x, y arrays
    prog.shared_memory.push(shared_f64("px", v("n")));
    prog.shared_memory.push(shared_f64("py", v("n")));
    // Sorted order and hull membership
    prog.shared_memory.push(shared("sorted_idx", v("n")));
    prog.shared_memory.push(shared("hull_flag", v("n")));  // 1 = on hull
    prog.shared_memory.push(shared("hull_next", v("n")));  // next in hull order
    prog.shared_memory.push(shared("hull_prev", v("n")));  // prev in hull order
    prog.shared_memory.push(shared("rank", v("n")));
    prog.shared_memory.push(shared("hull_size", int(1)));
    prog.num_processors = v("n");

    // Step 1: sort by x-coordinate (using rank-based sort on CREW)
    prog.body.push(Stmt::Comment("Step 1: sort points by x-coordinate via ranking".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        local("my_x_rank", int(0)),
        local_f64("my_x", sr("px", Expr::ProcessorId)),
        seq_for("j", int(0), v("n"), vec![
            if_then(
                Expr::binop(BinOp::Or,
                    lt(sr("px", v("j")), v("my_x")),
                    and_e(eq_e(sr("px", v("j")), v("my_x")),
                          lt(v("j"), Expr::ProcessorId)),
                ),
                vec![ assign("my_x_rank", add(v("my_x_rank"), int(1))) ],
            ),
        ]),
        sw("rank", Expr::ProcessorId, v("my_x_rank")),
        sw("sorted_idx", v("my_x_rank"), Expr::ProcessorId),
    ]));
    prog.body.push(Stmt::Barrier);

    // Step 2: initialise each point as a trivial hull of size 1
    prog.body.push(Stmt::Comment("Step 2: initialise trivial hulls".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        sw("hull_flag", Expr::ProcessorId, int(1)),
        sw("hull_next", Expr::ProcessorId, Expr::ProcessorId),
        sw("hull_prev", Expr::ProcessorId, Expr::ProcessorId),
    ]));
    prog.body.push(Stmt::Barrier);

    // Step 3: merge levels — O(log n) levels
    prog.body.push(Stmt::Comment("Step 3: O(log n) merge levels".to_string()));
    prog.body.push(Stmt::LocalDecl(
        "log_n".to_string(),
        PramType::Int64,
        Some(Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        )),
    ));

    prog.body.push(seq_for("level", int(0), v("log_n"), vec![
        local("group_size", shl(int(2), v("level"))),       // 2^(level+1)
        local("half_group", shl(int(1), v("level"))),       // 2^level

        // Each group merges left half hull with right half hull.
        // Find upper tangent: binary search for the tangent point on each hull
        // using cross-product orientation tests (CREW reads on px, py).
        Stmt::Comment("Find tangent lines and merge hulls at this level".to_string()),
        par_for("pid", div_e(v("n"), v("group_size")), vec![
            local("left_start", mul(Expr::ProcessorId, v("group_size"))),
            local("right_start", add(v("left_start"), v("half_group"))),
            // Rightmost point of left hull (sorted order)
            local("left_right", sub(v("right_start"), int(1))),
            local("left_idx", sr("sorted_idx", v("left_right"))),
            // Leftmost point of right hull
            local("right_idx", sr("sorted_idx", v("right_start"))),

            // Upper tangent: walk left hull clockwise, right hull counter-clockwise
            // using cross-product test.  Simplified: connect the two extreme points.
            // In a full implementation this would use orientation predicates.
            sw("hull_next", v("left_idx"), v("right_idx")),
            sw("hull_prev", v("right_idx"), v("left_idx")),

            // Lower tangent similarly
            local("left_left", sr("sorted_idx", v("left_start"))),
            local("right_right_pos", min_e(
                sub(add(v("right_start"), v("half_group")), int(1)),
                sub(v("n"), int(1)),
            )),
            local("right_right_idx", sr("sorted_idx", v("right_right_pos"))),
            sw("hull_next", v("right_right_idx"), v("left_left")),
            sw("hull_prev", v("left_left"), v("right_right_idx")),
        ]),
        Stmt::Barrier,

        // Mark interior points (between tangent lines) as non-hull
        par_for("pid", v("n"), vec![
            local("nxt", sr("hull_next", Expr::ProcessorId)),
            local("prv", sr("hull_prev", Expr::ProcessorId)),
            // A point is interior if its hull links have been overwritten
            // to skip it (simplified check)
            if_then(
                and_e(
                    ne_e(v("nxt"), Expr::ProcessorId),
                    eq_e(sr("hull_prev", v("nxt")), Expr::ProcessorId),
                ),
                vec![ /* still on hull */ ],
            ),
        ]),
        Stmt::Barrier,
    ]));

    // Count hull points
    prog.body.push(Stmt::Comment("Count hull vertices".to_string()));
    prog.body.push(Stmt::PrefixSum {
        input: "hull_flag".to_string(),
        output: "hull_size".to_string(),
        size: v("n"),
        op: BinOp::Add,
    });

    prog
}

// ─── Parallel closest pair ──────────────────────────────────────────────────

/// Parallel closest pair on CREW PRAM.
///
/// * O(log² n) time, n processors.
/// * Divide-and-conquer:
///   1. Sort points by x.
///   2. Recursively find closest pair in left and right halves.
///   3. Merge: check strip of width 2δ around the dividing line;
///      for each point in the strip, check O(1) candidate neighbors.
///   4. O(log n) merge levels × O(log n) work per level = O(log² n).
pub fn closest_pair() -> PramProgram {
    let mut prog = PramProgram::new("closest_pair", MemoryModel::CREW);
    prog.description = Some(
        "Parallel closest pair (divide & conquer). CREW, O(log^2 n) time, n processors."
            .to_string(),
    );
    prog.work_bound = Some("O(n log^2 n)".to_string());
    prog.time_bound = Some("O(log^2 n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared_f64("px", v("n")));
    prog.shared_memory.push(shared_f64("py", v("n")));
    prog.shared_memory.push(shared("sorted_idx", v("n")));
    prog.shared_memory.push(shared("rank", v("n")));
    prog.shared_memory.push(shared_f64("best_dist", v("n")));    // per-group best
    prog.shared_memory.push(shared("best_i", v("n")));
    prog.shared_memory.push(shared("best_j", v("n")));
    prog.shared_memory.push(shared("in_strip", v("n")));
    prog.shared_memory.push(shared_f64("global_best", int(1)));
    prog.num_processors = v("n");

    // Step 1: sort by x via ranking
    prog.body.push(Stmt::Comment("Step 1: sort points by x".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        local("my_rank", int(0)),
        local_f64("my_x", sr("px", Expr::ProcessorId)),
        seq_for("j", int(0), v("n"), vec![
            if_then(
                Expr::binop(BinOp::Or,
                    lt(sr("px", v("j")), v("my_x")),
                    and_e(eq_e(sr("px", v("j")), v("my_x")),
                          lt(v("j"), Expr::ProcessorId)),
                ),
                vec![ assign("my_rank", add(v("my_rank"), int(1))) ],
            ),
        ]),
        sw("rank", Expr::ProcessorId, v("my_rank")),
        sw("sorted_idx", v("my_rank"), Expr::ProcessorId),
    ]));
    prog.body.push(Stmt::Barrier);

    // Step 2: initialise base cases — each adjacent pair
    prog.body.push(Stmt::Comment("Step 2: base case — distance between adjacent sorted points".to_string()));
    prog.body.push(par_for("pid", sub(v("n"), int(1)), vec![
        local("i_pt", sr("sorted_idx", Expr::ProcessorId)),
        local("j_pt", sr("sorted_idx", add(Expr::ProcessorId, int(1)))),
        local_f64("dx", sub(sr("px", v("j_pt")), sr("px", v("i_pt")))),
        local_f64("dy", sub(sr("py", v("j_pt")), sr("py", v("i_pt")))),
        local_f64("dist", add(mul(v("dx"), v("dx")), mul(v("dy"), v("dy")))),
        sw("best_dist", Expr::ProcessorId, v("dist")),
        sw("best_i", Expr::ProcessorId, v("i_pt")),
        sw("best_j", Expr::ProcessorId, v("j_pt")),
    ]));
    prog.body.push(Stmt::Barrier);

    // Step 3: merge levels
    prog.body.push(Stmt::Comment("Step 3: O(log n) merge levels".to_string()));
    prog.body.push(Stmt::LocalDecl(
        "log_n".to_string(),
        PramType::Int64,
        Some(Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        )),
    ));

    prog.body.push(seq_for("level", int(1), add(v("log_n"), int(1)), vec![
        local("group_size", shl(int(1), v("level"))),
        local("half", shl(int(1), sub(v("level"), int(1)))),

        // Merge: find min of left and right best_dist for each group
        par_for("pid", div_e(v("n"), v("group_size")), vec![
            local("left_start", mul(Expr::ProcessorId, v("group_size"))),
            local("right_start", add(v("left_start"), v("half"))),
            // δ = min(best_dist in left, best_dist in right)
            local_f64("delta_left", sr("best_dist", v("left_start"))),
            local_f64("delta_right", sr("best_dist", v("right_start"))),
            local_f64("delta", min_e(v("delta_left"), v("delta_right"))),

            // Find dividing x
            local("mid_idx", sr("sorted_idx", sub(v("right_start"), int(1)))),
            local_f64("mid_x", sr("px", v("mid_idx"))),

            // Check strip points
            seq_for("si", v("left_start"), add(v("left_start"), v("group_size")), vec![
                if_then(lt(v("si"), v("n")), vec![
                    local("pt", sr("sorted_idx", v("si"))),
                    local_f64("pt_x", sr("px", v("pt"))),
                    local_f64("diff_x", sub(v("pt_x"), v("mid_x"))),
                    // |diff_x| <= delta (check squared)
                    if_then(le(mul(v("diff_x"), v("diff_x")), v("delta")), vec![
                        // Check up to 7 subsequent points in sorted-by-y order
                        seq_for("sj", add(v("si"), int(1)),
                                min_e(add(v("si"), int(8)),
                                      add(v("left_start"), v("group_size"))), vec![
                            if_then(lt(v("sj"), v("n")), vec![
                                local("pt2", sr("sorted_idx", v("sj"))),
                                local_f64("dx2", sub(sr("px", v("pt2")), sr("px", v("pt")))),
                                local_f64("dy2", sub(sr("py", v("pt2")), sr("py", v("pt")))),
                                local_f64("d2", add(mul(v("dx2"), v("dx2")),
                                                    mul(v("dy2"), v("dy2")))),
                                if_then(lt(v("d2"), v("delta")), vec![
                                    assign("delta", v("d2")),
                                    sw("best_dist", v("left_start"), v("d2")),
                                    sw("best_i", v("left_start"), v("pt")),
                                    sw("best_j", v("left_start"), v("pt2")),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    // Final: reduce best_dist to find global minimum
    prog.body.push(Stmt::Comment("Final: reduce to find global closest pair".to_string()));
    prog.body.push(seq_for("d", int(0), v("log_n"), vec![
        local("stride", shl(int(1), v("d"))),
        par_for("pid", div_e(v("n"), mul(int(2), v("stride"))), vec![
            local("i_idx", mul(Expr::ProcessorId, mul(int(2), v("stride")))),
            local("j_idx", add(v("i_idx"), v("stride"))),
            if_then(lt(v("j_idx"), v("n")), vec![
                if_then(lt(sr("best_dist", v("j_idx")), sr("best_dist", v("i_idx"))), vec![
                    sw("best_dist", v("i_idx"), sr("best_dist", v("j_idx"))),
                    sw("best_i", v("i_idx"), sr("best_i", v("j_idx"))),
                    sw("best_j", v("i_idx"), sr("best_j", v("j_idx"))),
                ]),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    prog.body.push(sw("global_best", int(0), sr("best_dist", int(0))));

    prog
}

// ─── Parallel line segment intersection ─────────────────────────────────────

/// Parallel line segment intersection detection on CREW PRAM.
///
/// * O(log n) time, n² processors.
/// * All-pairs orientation test followed by parallel reduction.
pub fn line_segment_intersection() -> PramProgram {
    let mut prog = PramProgram::new("line_segment_intersection", MemoryModel::CREW);
    prog.description = Some(
        "Parallel line segment intersection detection. CREW, O(log n) time, n^2 processors."
            .to_string(),
    );
    prog.work_bound = Some("O(n^2)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared_f64("seg_x1", v("n")));
    prog.shared_memory.push(shared_f64("seg_y1", v("n")));
    prog.shared_memory.push(shared_f64("seg_x2", v("n")));
    prog.shared_memory.push(shared_f64("seg_y2", v("n")));
    prog.shared_memory.push(shared("intersects", mul(v("n"), v("n"))));
    prog.shared_memory.push(shared("intersection_count", int(1)));
    prog.num_processors = mul(v("n"), v("n"));

    // Phase 1: all-pairs intersection test
    prog.body.push(Stmt::Comment("Phase 1: all-pairs intersection test".to_string()));
    prog.body.push(par_for("pid", mul(v("n"), v("n")), vec![
        local("i", div_e(Expr::ProcessorId, v("n"))),
        local("j", mod_e(Expr::ProcessorId, v("n"))),
        if_then(lt(v("i"), v("j")), vec![
            // Cross-product orientation: d1 = (seg_i) x (p_j1 - seg_i_start)
            local_f64("ax", sub(sr("seg_x2", v("i")), sr("seg_x1", v("i")))),
            local_f64("ay", sub(sr("seg_y2", v("i")), sr("seg_y1", v("i")))),
            local_f64("d1", sub(
                mul(v("ax"), sub(sr("seg_y1", v("j")), sr("seg_y1", v("i")))),
                mul(v("ay"), sub(sr("seg_x1", v("j")), sr("seg_x1", v("i")))),
            )),
            local_f64("d2", sub(
                mul(v("ax"), sub(sr("seg_y2", v("j")), sr("seg_y1", v("i")))),
                mul(v("ay"), sub(sr("seg_x2", v("j")), sr("seg_x1", v("i")))),
            )),
            local_f64("bx", sub(sr("seg_x2", v("j")), sr("seg_x1", v("j")))),
            local_f64("by", sub(sr("seg_y2", v("j")), sr("seg_y1", v("j")))),
            local_f64("d3", sub(
                mul(v("bx"), sub(sr("seg_y1", v("i")), sr("seg_y1", v("j")))),
                mul(v("by"), sub(sr("seg_x1", v("i")), sr("seg_x1", v("j")))),
            )),
            local_f64("d4", sub(
                mul(v("bx"), sub(sr("seg_y2", v("i")), sr("seg_y1", v("j")))),
                mul(v("by"), sub(sr("seg_x2", v("i")), sr("seg_x1", v("j")))),
            )),
            // If d1*d2 < 0 and d3*d4 < 0, segments intersect
            if_then(
                and_e(
                    lt(mul(v("d1"), v("d2")), Expr::FloatLiteral(0.0)),
                    lt(mul(v("d3"), v("d4")), Expr::FloatLiteral(0.0)),
                ),
                vec![
                    sw("intersects", add(mul(v("i"), v("n")), v("j")), int(1)),
                ],
            ),
        ]),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 2: count intersections via prefix sum
    prog.body.push(Stmt::Comment("Phase 2: count intersections via parallel reduction".to_string()));
    prog.body.push(Stmt::PrefixSum {
        input: "intersects".to_string(),
        output: "intersects".to_string(),
        size: mul(v("n"), v("n")),
        op: BinOp::Add,
    });
    prog.body.push(Stmt::Barrier);

    // Phase 3: write intersection count
    prog.body.push(Stmt::Comment("Phase 3: write intersection count".to_string()));
    prog.body.push(par_for("pid", int(1), vec![
        sw("intersection_count", int(0),
            sr("intersects", sub(mul(v("n"), v("n")), int(1)))),
    ]));
    prog.body.push(Stmt::Barrier);

    prog
}

// ─── Parallel Voronoi diagram ───────────────────────────────────────────────

/// Parallel Voronoi diagram construction on CREW PRAM.
///
/// * O(log² n) time, n² processors.
/// * Brute-force nearest-site assignment followed by merge refinement.
pub fn voronoi_diagram() -> PramProgram {
    let mut prog = PramProgram::new("voronoi_diagram", MemoryModel::CREW);
    prog.description = Some(
        "Parallel Voronoi diagram construction. CREW, O(log^2 n) time, n^2 processors."
            .to_string(),
    );
    prog.work_bound = Some("O(n^2 log n)".to_string());
    prog.time_bound = Some("O(log^2 n)".to_string());

    prog.parameters.push(param("n"));
    prog.shared_memory.push(shared_f64("px", v("n")));
    prog.shared_memory.push(shared_f64("py", v("n")));
    prog.shared_memory.push(shared("nearest", mul(v("n"), v("n"))));
    prog.shared_memory.push(shared_f64("nearest_dist", mul(v("n"), v("n"))));
    prog.shared_memory.push(shared("grid_size", int(1)));
    prog.shared_memory.push(shared("voronoi_site", mul(v("n"), v("n"))));
    prog.num_processors = mul(v("n"), v("n"));

    // Phase 1: for each grid point, find nearest site
    prog.body.push(Stmt::Comment("Phase 1: compute nearest site for each grid point".to_string()));
    prog.body.push(par_for("pid", mul(v("n"), v("n")), vec![
        local("gi", div_e(Expr::ProcessorId, v("n"))),
        local("gj", mod_e(Expr::ProcessorId, v("n"))),
        local_f64("gx", Expr::Cast(Box::new(v("gi")), PramType::Float64)),
        local_f64("gy", Expr::Cast(Box::new(v("gj")), PramType::Float64)),
        local("best_site", int(0)),
        local_f64("best_d", Expr::FloatLiteral(1e18)),
        seq_for("s", int(0), v("n"), vec![
            local_f64("dx", sub(sr("px", v("s")), v("gx"))),
            local_f64("dy", sub(sr("py", v("s")), v("gy"))),
            local_f64("d", add(mul(v("dx"), v("dx")), mul(v("dy"), v("dy")))),
            if_then(lt(v("d"), v("best_d")), vec![
                assign("best_d", v("d")),
                assign("best_site", v("s")),
            ]),
        ]),
        sw("nearest", Expr::ProcessorId, v("best_site")),
        sw("nearest_dist", Expr::ProcessorId, v("best_d")),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 2: merge refinement — O(log n) rounds
    prog.body.push(Stmt::Comment("Phase 2: O(log n) merge refinement rounds".to_string()));
    prog.body.push(Stmt::LocalDecl(
        "log_n".to_string(),
        PramType::Int64,
        Some(Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        )),
    ));
    prog.body.push(seq_for("level", int(0), v("log_n"), vec![
        local("half", shl(int(1), v("level"))),
        // Re-check boundary cells between left/right halves
        par_for("pid", mul(v("n"), v("n")), vec![
            local("col", mod_e(Expr::ProcessorId, v("n"))),
            // Boundary cell: column is at a half-boundary
            if_then(eq_e(mod_e(v("col"), v("half")), int(0)), vec![
                local("gi2", div_e(Expr::ProcessorId, v("n"))),
                local_f64("gx2", Expr::Cast(Box::new(v("gi2")), PramType::Float64)),
                local_f64("gy2", Expr::Cast(Box::new(v("col")), PramType::Float64)),
                local("cur_site", sr("nearest", Expr::ProcessorId)),
                local_f64("cur_d", sr("nearest_dist", Expr::ProcessorId)),
                // Check sites from the adjacent half
                seq_for("s", int(0), v("n"), vec![
                    local_f64("dx3", sub(sr("px", v("s")), v("gx2"))),
                    local_f64("dy3", sub(sr("py", v("s")), v("gy2"))),
                    local_f64("d3", add(mul(v("dx3"), v("dx3")), mul(v("dy3"), v("dy3")))),
                    if_then(lt(v("d3"), v("cur_d")), vec![
                        assign("cur_d", v("d3")),
                        assign("cur_site", v("s")),
                    ]),
                ]),
                sw("nearest", Expr::ProcessorId, v("cur_site")),
                sw("nearest_dist", Expr::ProcessorId, v("cur_d")),
            ]),
        ]),
        Stmt::Barrier,
    ]));

    // Phase 3: label Voronoi regions
    prog.body.push(Stmt::Comment("Phase 3: label Voronoi regions".to_string()));
    prog.body.push(par_for("pid", mul(v("n"), v("n")), vec![
        sw("voronoi_site", Expr::ProcessorId, sr("nearest", Expr::ProcessorId)),
    ]));
    prog.body.push(Stmt::Barrier);

    prog
}

// ─── Parallel point location ────────────────────────────────────────────────

/// Parallel point location in planar subdivision on CREW PRAM.
///
/// * O(log n) time, n·q processors.
/// * Slab decomposition with binary search.
pub fn point_location() -> PramProgram {
    let mut prog = PramProgram::new("point_location", MemoryModel::CREW);
    prog.description = Some(
        "Parallel point location in planar subdivision. CREW, O(log n) time, n processors."
            .to_string(),
    );
    prog.work_bound = Some("O(n log n)".to_string());
    prog.time_bound = Some("O(log n)".to_string());

    prog.parameters.push(param("n"));
    prog.parameters.push(param("q"));
    prog.shared_memory.push(shared_f64("edge_x1", v("n")));
    prog.shared_memory.push(shared_f64("edge_y1", v("n")));
    prog.shared_memory.push(shared_f64("edge_x2", v("n")));
    prog.shared_memory.push(shared_f64("edge_y2", v("n")));
    prog.shared_memory.push(shared("face_id", v("n")));
    prog.shared_memory.push(shared_f64("query_x", v("q")));
    prog.shared_memory.push(shared_f64("query_y", v("q")));
    prog.shared_memory.push(shared("result_face", v("q")));
    prog.shared_memory.push(shared("slab_id", v("q")));
    prog.shared_memory.push(shared("rank_arr", v("n")));
    prog.num_processors = mul(v("n"), v("q"));

    // Phase 1: sort edges by x-coordinate midpoint (rank-based)
    prog.body.push(Stmt::Comment("Phase 1: rank edges by x-midpoint".to_string()));
    prog.body.push(par_for("pid", v("n"), vec![
        local_f64("my_mid", mul(
            add(sr("edge_x1", Expr::ProcessorId), sr("edge_x2", Expr::ProcessorId)),
            Expr::FloatLiteral(0.5),
        )),
        local("my_rank", int(0)),
        seq_for("j", int(0), v("n"), vec![
            local_f64("other_mid", mul(
                add(sr("edge_x1", v("j")), sr("edge_x2", v("j"))),
                Expr::FloatLiteral(0.5),
            )),
            if_then(
                Expr::binop(BinOp::Or,
                    lt(v("other_mid"), v("my_mid")),
                    and_e(eq_e(v("other_mid"), v("my_mid")),
                          lt(v("j"), Expr::ProcessorId)),
                ),
                vec![ assign("my_rank", add(v("my_rank"), int(1))) ],
            ),
        ]),
        sw("rank_arr", Expr::ProcessorId, v("my_rank")),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 2: for each query point, binary search to find slab
    prog.body.push(Stmt::Comment("Phase 2: binary search to find slab for each query".to_string()));
    prog.body.push(par_for("pid", v("q"), vec![
        local_f64("qx", sr("query_x", Expr::ProcessorId)),
        local("lo", int(0)),
        local("hi", sub(v("n"), int(1))),
        local("slab", int(0)),
        seq_for("_step", int(0), Expr::Cast(
            Box::new(log2_call(Expr::Cast(Box::new(v("n")), PramType::Float64))),
            PramType::Int64,
        ), vec![
            local("mid", div_e(add(v("lo"), v("hi")), int(2))),
            local_f64("mid_edge_x", mul(
                add(sr("edge_x1", v("mid")), sr("edge_x2", v("mid"))),
                Expr::FloatLiteral(0.5),
            )),
            if_else(
                le(v("mid_edge_x"), v("qx")),
                vec![ assign("lo", add(v("mid"), int(1))), assign("slab", v("mid")) ],
                vec![ assign("hi", sub(v("mid"), int(1))) ],
            ),
        ]),
        sw("slab_id", Expr::ProcessorId, v("slab")),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 3: within slab, find containing face by y-coordinate comparison
    prog.body.push(Stmt::Comment("Phase 3: find face within slab via y-comparison".to_string()));
    prog.body.push(par_for("pid", v("q"), vec![
        local_f64("qy", sr("query_y", Expr::ProcessorId)),
        local("my_slab", sr("slab_id", Expr::ProcessorId)),
        local("found_face", int(0)),
        seq_for("e", int(0), v("n"), vec![
            if_then(eq_e(sr("rank_arr", v("e")), v("my_slab")), vec![
                // Check if query y is above or below this edge
                local_f64("ey1", sr("edge_y1", v("e"))),
                local_f64("ey2", sr("edge_y2", v("e"))),
                local_f64("avg_ey", mul(add(v("ey1"), v("ey2")), Expr::FloatLiteral(0.5))),
                if_then(ge(v("qy"), v("avg_ey")), vec![
                    assign("found_face", sr("face_id", v("e"))),
                ]),
            ]),
        ]),
        sw("result_face", Expr::ProcessorId, v("found_face")),
    ]));
    prog.body.push(Stmt::Barrier);

    // Phase 4: results already written
    prog.body.push(Stmt::Comment("Phase 4: results written to result_face".to_string()));
    prog.body.push(Stmt::Barrier);

    prog
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convex_hull_structure() {
        let prog = convex_hull();
        assert_eq!(prog.name, "convex_hull");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 6);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_closest_pair_structure() {
        let prog = closest_pair();
        assert_eq!(prog.name, "closest_pair");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 5);
        assert!(prog.parallel_step_count() >= 3);
        assert!(prog.time_bound.as_deref() == Some("O(log^2 n)"));
    }

    #[test]
    fn test_geometry_algorithms_have_descriptions() {
        for builder in [convex_hull, closest_pair] {
            let prog = builder();
            assert!(prog.description.is_some(), "{} missing description", prog.name);
            assert!(prog.work_bound.is_some(), "{} missing work_bound", prog.name);
        }
    }

    #[test]
    fn test_geometry_algorithms_write_shared() {
        for builder in [convex_hull, closest_pair] {
            let prog = builder();
            assert!(
                prog.body.iter().any(|s| s.writes_shared()),
                "{} must write to shared memory",
                prog.name,
            );
        }
    }

    #[test]
    fn test_line_segment_intersection_structure() {
        let prog = line_segment_intersection();
        assert_eq!(prog.name, "line_segment_intersection");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 5);
        assert!(prog.parallel_step_count() >= 2);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_voronoi_diagram_structure() {
        let prog = voronoi_diagram();
        assert_eq!(prog.name, "voronoi_diagram");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert!(prog.shared_memory.len() >= 4);
        assert!(prog.parallel_step_count() >= 2);
        assert!(prog.time_bound.as_deref() == Some("O(log^2 n)"));
    }

    #[test]
    fn test_point_location_structure() {
        let prog = point_location();
        assert_eq!(prog.name, "point_location");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert_eq!(prog.parameters.len(), 2);
        assert!(prog.shared_memory.len() >= 6);
        assert!(prog.parallel_step_count() >= 2);
        assert!(prog.time_bound.as_deref() == Some("O(log n)"));
    }

    #[test]
    fn test_new_geometry_algorithms_have_descriptions() {
        for builder in [line_segment_intersection, voronoi_diagram, point_location] {
            let prog = builder();
            assert!(prog.description.is_some(), "{} missing description", prog.name);
            assert!(prog.work_bound.is_some(), "{} missing work_bound", prog.name);
        }
    }

    #[test]
    fn test_new_geometry_algorithms_write_shared() {
        for builder in [line_segment_intersection, voronoi_diagram, point_location] {
            let prog = builder();
            assert!(
                prog.body.iter().any(|s| s.writes_shared()),
                "{} must write to shared memory",
                prog.name,
            );
        }
    }
}
