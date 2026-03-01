//! Memory layout computation for serialized PRAM shared-memory state.
//!
//! Maps each `SharedMemoryDecl` to a flat C array with proper alignment and
//! computes base addresses so every region can be laid out in a single
//! contiguous allocation (or, more commonly, as individual `pram_calloc`
//! calls with per-region declarations).

use std::fmt::Write;

use crate::pram_ir::ast::*;
use crate::pram_ir::types::*;

// ---------------------------------------------------------------------------
// SharedRegionLayout
// ---------------------------------------------------------------------------

/// Describes the C-level layout of one shared-memory region.
#[derive(Debug, Clone)]
pub struct SharedRegionLayout {
    /// Name of the region (matches `SharedMemoryDecl::name`).
    pub name: String,
    /// Base byte-offset inside a hypothetical flat buffer.
    pub base_offset: usize,
    /// Size of a single element in bytes.
    pub element_size: usize,
    /// Number of elements.
    pub count: usize,
    /// Total bytes occupied (including any tail padding for alignment of the
    /// *next* region).
    pub total_bytes: usize,
    /// The element's PramType.
    pub elem_type: PramType,
    /// Required alignment in bytes.
    pub alignment: usize,
}

impl SharedRegionLayout {
    /// The C type string for the element type.
    pub fn c_type(&self) -> String {
        pram_type_to_c(&self.elem_type)
    }

    /// The C declaration for this region as a heap-allocated array.
    pub fn c_decl(&self) -> String {
        let cty = self.c_type();
        format!(
            "{}* {} = ({}*)pram_calloc({}, sizeof({}));",
            cty, self.name, cty, self.count, cty
        )
    }

    /// The C statement to free this region.
    pub fn c_free(&self) -> String {
        format!("pram_free({});", self.name)
    }
}

// ---------------------------------------------------------------------------
// MemoryLayout
// ---------------------------------------------------------------------------

/// Complete memory layout for all shared-memory regions of a PRAM program.
#[derive(Debug, Clone)]
pub struct MemoryLayout {
    pub regions: Vec<SharedRegionLayout>,
    /// Total bytes needed if laid out contiguously.
    pub total_bytes: usize,
}

impl MemoryLayout {
    /// Compute a `MemoryLayout` from the shared-memory declarations of a
    /// `PramProgram`.
    ///
    /// Region sizes that are given as constant expressions are evaluated;
    /// non-constant sizes fall back to a caller-supplied `default_size`.
    pub fn compute(decls: &[SharedMemoryDecl], default_size: usize) -> Self {
        let mut regions = Vec::with_capacity(decls.len());
        let mut offset: usize = 0;

        for decl in decls {
            let elem_size = decl.elem_type.size_bytes();
            let align = decl.elem_type.alignment();
            let count = decl
                .size
                .eval_const_int()
                .map(|v| v.max(1) as usize)
                .unwrap_or(default_size);

            // Align the current offset.
            let aligned_offset = align_up(offset, align);

            let raw_bytes = elem_size * count;
            // Pad total_bytes so the *next* region starts aligned to at least 8.
            let total_bytes = align_up(raw_bytes, 8.max(align));

            regions.push(SharedRegionLayout {
                name: decl.name.clone(),
                base_offset: aligned_offset,
                element_size: elem_size,
                count,
                total_bytes,
                elem_type: decl.elem_type.clone(),
                alignment: align,
            });

            offset = aligned_offset + total_bytes;
        }

        MemoryLayout {
            total_bytes: offset,
            regions,
        }
    }

    /// Emit C variable declarations for every region.
    pub fn generate_declarations(&self) -> String {
        let mut out = String::new();
        for r in &self.regions {
            writeln!(out, "{}", r.c_decl()).unwrap();
        }
        out
    }

    /// Emit C initialisation code (zero-fill is implicit with `calloc`, so
    /// this only emits a comment plus any required sentinel values for
    /// CRCW-Priority writer-PID tracking arrays).
    pub fn generate_init_code(&self) -> String {
        let mut out = String::new();
        writeln!(out, "/* Initialize shared memory regions */").unwrap();
        for r in &self.regions {
            writeln!(
                out,
                "/* {}: {} elements of {} ({} bytes each) */",
                r.name,
                r.count,
                pram_type_to_c(&r.elem_type),
                r.element_size
            )
            .unwrap();
        }
        out
    }

    /// Emit C cleanup code that frees every region.
    pub fn generate_free_code(&self) -> String {
        let mut out = String::new();
        writeln!(out, "/* Free shared memory regions */").unwrap();
        for r in &self.regions {
            writeln!(out, "{}", r.c_free()).unwrap();
        }
        out
    }

    /// Generate CRCW-Priority auxiliary arrays for a given region.
    /// Returns (writer_pid_decl, staging_decl, init_code, free_code).
    pub fn generate_crcw_priority_arrays(&self, region_name: &str) -> Option<(String, String, String, String)> {
        let r = self.regions.iter().find(|r| r.name == region_name)?;
        let cty = r.c_type();
        let wpid_name = format!("_wpid_{}", region_name);
        let stg_name = format!("_stg_{}", region_name);
        let count_s = r.count.to_string();

        let wpid_decl = format!(
            "int64_t* {} = (int64_t*)pram_calloc({}, sizeof(int64_t));",
            wpid_name, count_s
        );
        let stg_decl = format!(
            "{}* {} = ({}*)pram_calloc({}, sizeof({}));",
            cty, stg_name, cty, count_s, cty
        );

        let mut init = String::new();
        writeln!(
            init,
            "for (int64_t _i = 0; _i < {}; _i++) {{ {}[_i] = INT64_MAX; }}",
            count_s, wpid_name
        )
        .unwrap();

        let mut free_code = String::new();
        writeln!(free_code, "pram_free({});", wpid_name).unwrap();
        writeln!(free_code, "pram_free({});", stg_name).unwrap();

        Some((wpid_decl, stg_decl, init, free_code))
    }

    /// Look up a region by name.
    pub fn get_region(&self, name: &str) -> Option<&SharedRegionLayout> {
        self.regions.iter().find(|r| r.name == name)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Round `value` up to the next multiple of `align`.
fn align_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    (value + align - 1) / align * align
}

/// Map a `PramType` to its C99 type string.
pub fn pram_type_to_c(ty: &PramType) -> String {
    match ty {
        PramType::Int64 => "int64_t".to_string(),
        PramType::Int32 => "int32_t".to_string(),
        PramType::Float64 => "double".to_string(),
        PramType::Float32 => "float".to_string(),
        PramType::Bool => "bool".to_string(),
        PramType::ProcessorId => "int64_t".to_string(),
        PramType::Unit => "void".to_string(),
        PramType::SharedMemory(inner) => format!("{}*", pram_type_to_c(inner)),
        PramType::SharedRef(inner) => format!("{}*", pram_type_to_c(inner)),
        PramType::Array(inner, _) => format!("{}*", pram_type_to_c(inner)),
        PramType::Tuple(_) => "void*".to_string(),
        PramType::Struct(name, _) => format!("struct {}", name),
    }
}

/// Return the C99 default-zero expression for a type (e.g. `0`, `0.0`, `false`).
pub fn c_default_value(ty: &PramType) -> &'static str {
    match ty {
        PramType::Int64 | PramType::Int32 | PramType::ProcessorId => "0",
        PramType::Float64 | PramType::Float32 => "0.0",
        PramType::Bool => "false",
        _ => "0",
    }
}

// ---------------------------------------------------------------------------
// LayoutOptimizer
// ---------------------------------------------------------------------------

/// Reorders memory regions by element_size descending (largest alignment first)
/// for cache friendliness, then recomputes offsets.
pub struct LayoutOptimizer;

impl LayoutOptimizer {
    pub fn optimize(layout: &MemoryLayout) -> MemoryLayout {
        let mut sorted_regions = layout.regions.clone();
        sorted_regions.sort_by(|a, b| b.element_size.cmp(&a.element_size));

        let mut offset: usize = 0;
        for r in &mut sorted_regions {
            let aligned_offset = align_up(offset, r.alignment);
            r.base_offset = aligned_offset;
            r.total_bytes = align_up(r.element_size * r.count, 8.max(r.alignment));
            offset = aligned_offset + r.total_bytes;
        }

        MemoryLayout {
            total_bytes: offset,
            regions: sorted_regions,
        }
    }
}

// ---------------------------------------------------------------------------
// compute_padding
// ---------------------------------------------------------------------------

/// Compute padding bytes needed to reach the given alignment from the current offset.
pub fn compute_padding(alignment: usize, current_offset: usize) -> usize {
    if alignment == 0 {
        return 0;
    }
    let remainder = current_offset % alignment;
    if remainder == 0 {
        0
    } else {
        alignment - remainder
    }
}

// ---------------------------------------------------------------------------
// LayoutReport
// ---------------------------------------------------------------------------

/// Summary report of a memory layout.
#[derive(Debug, Clone)]
pub struct LayoutReport {
    pub total_memory: usize,
    pub padding_overhead: usize,
    pub region_count: usize,
    /// (name, offset, size) for each region.
    pub region_map: Vec<(String, usize, usize)>,
}

impl LayoutReport {
    /// Generate a report from a `MemoryLayout`.
    pub fn generate(layout: &MemoryLayout) -> LayoutReport {
        let mut padding_overhead: usize = 0;
        let mut region_map = Vec::with_capacity(layout.regions.len());

        for r in &layout.regions {
            let raw_bytes = r.element_size * r.count;
            padding_overhead += r.total_bytes.saturating_sub(raw_bytes);
            region_map.push((r.name.clone(), r.base_offset, r.total_bytes));
        }

        // Also account for inter-region alignment gaps.
        for i in 1..layout.regions.len() {
            let prev_end = layout.regions[i - 1].base_offset + layout.regions[i - 1].total_bytes;
            let gap = layout.regions[i].base_offset.saturating_sub(prev_end);
            padding_overhead += gap;
        }

        LayoutReport {
            total_memory: layout.total_bytes,
            padding_overhead,
            region_count: layout.regions.len(),
            region_map,
        }
    }
}

// ---------------------------------------------------------------------------
// generate_memcpy_init
// ---------------------------------------------------------------------------

/// Generate C code that uses memcpy/loops to initialize regions with given values.
pub fn generate_memcpy_init(layout: &MemoryLayout, initial_values: &[(String, Vec<i64>)]) -> String {
    let mut out = String::new();
    writeln!(out, "/* memcpy-based initialization */").unwrap();

    for (name, values) in initial_values {
        if let Some(r) = layout.get_region(name) {
            let cty = pram_type_to_c(&r.elem_type);
            let count = values.len().min(r.count);

            if count > 0 {
                writeln!(out, "{{").unwrap();
                writeln!(out, "    {} _init_{}_vals[] = {{{}}};", cty, name,
                    values.iter().take(count).map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
                ).unwrap();
                writeln!(
                    out,
                    "    memcpy({}, _init_{}_vals, {} * sizeof({}));",
                    name, name, count, cty
                ).unwrap();
                writeln!(out, "}}").unwrap();
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// estimate_memory_footprint
// ---------------------------------------------------------------------------

/// Return total bytes used by all regions including padding.
pub fn estimate_memory_footprint(layout: &MemoryLayout) -> usize {
    layout.total_bytes
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_decls() -> Vec<SharedMemoryDecl> {
        vec![
            SharedMemoryDecl {
                name: "A".to_string(),
                elem_type: PramType::Int64,
                size: Expr::IntLiteral(1024),
            },
            SharedMemoryDecl {
                name: "B".to_string(),
                elem_type: PramType::Float64,
                size: Expr::IntLiteral(512),
            },
            SharedMemoryDecl {
                name: "flags".to_string(),
                elem_type: PramType::Bool,
                size: Expr::IntLiteral(256),
            },
        ]
    }

    #[test]
    fn test_compute_layout_regions() {
        let layout = MemoryLayout::compute(&sample_decls(), 1024);
        assert_eq!(layout.regions.len(), 3);
        assert_eq!(layout.regions[0].name, "A");
        assert_eq!(layout.regions[1].name, "B");
        assert_eq!(layout.regions[2].name, "flags");
    }

    #[test]
    fn test_region_sizes() {
        let layout = MemoryLayout::compute(&sample_decls(), 1024);
        assert_eq!(layout.regions[0].count, 1024);
        assert_eq!(layout.regions[0].element_size, 8);
        assert_eq!(layout.regions[1].count, 512);
        assert_eq!(layout.regions[1].element_size, 8);
        assert_eq!(layout.regions[2].count, 256);
        assert_eq!(layout.regions[2].element_size, 1);
    }

    #[test]
    fn test_alignment() {
        let layout = MemoryLayout::compute(&sample_decls(), 1024);
        for r in &layout.regions {
            assert_eq!(r.base_offset % r.alignment, 0, "region {} misaligned", r.name);
        }
    }

    #[test]
    fn test_total_bytes_positive() {
        let layout = MemoryLayout::compute(&sample_decls(), 1024);
        assert!(layout.total_bytes > 0);
    }

    #[test]
    fn test_non_overlapping() {
        let layout = MemoryLayout::compute(&sample_decls(), 1024);
        for i in 0..layout.regions.len() {
            for j in (i + 1)..layout.regions.len() {
                let a = &layout.regions[i];
                let b = &layout.regions[j];
                let a_end = a.base_offset + a.total_bytes;
                assert!(a_end <= b.base_offset, "regions {} and {} overlap", a.name, b.name);
            }
        }
    }

    #[test]
    fn test_generate_declarations() {
        let layout = MemoryLayout::compute(&sample_decls(), 1024);
        let decls = layout.generate_declarations();
        assert!(decls.contains("int64_t* A"));
        assert!(decls.contains("pram_calloc(1024"));
        assert!(decls.contains("double* B"));
        assert!(decls.contains("bool* flags"));
    }

    #[test]
    fn test_generate_free_code() {
        let layout = MemoryLayout::compute(&sample_decls(), 1024);
        let free = layout.generate_free_code();
        assert!(free.contains("pram_free(A)"));
        assert!(free.contains("pram_free(B)"));
        assert!(free.contains("pram_free(flags)"));
    }

    #[test]
    fn test_generate_init_code() {
        let layout = MemoryLayout::compute(&sample_decls(), 1024);
        let init = layout.generate_init_code();
        assert!(init.contains("Initialize shared memory"));
        assert!(init.contains("A:"));
    }

    #[test]
    fn test_default_size_fallback() {
        let decls = vec![SharedMemoryDecl {
            name: "X".to_string(),
            elem_type: PramType::Int32,
            size: Expr::Variable("n".to_string()), // not constant
        }];
        let layout = MemoryLayout::compute(&decls, 2048);
        assert_eq!(layout.regions[0].count, 2048);
    }

    #[test]
    fn test_pram_type_to_c() {
        assert_eq!(pram_type_to_c(&PramType::Int64), "int64_t");
        assert_eq!(pram_type_to_c(&PramType::Int32), "int32_t");
        assert_eq!(pram_type_to_c(&PramType::Float64), "double");
        assert_eq!(pram_type_to_c(&PramType::Float32), "float");
        assert_eq!(pram_type_to_c(&PramType::Bool), "bool");
        assert_eq!(pram_type_to_c(&PramType::Unit), "void");
        assert_eq!(
            pram_type_to_c(&PramType::SharedMemory(Box::new(PramType::Int64))),
            "int64_t*"
        );
    }

    #[test]
    fn test_c_default_value() {
        assert_eq!(c_default_value(&PramType::Int64), "0");
        assert_eq!(c_default_value(&PramType::Float64), "0.0");
        assert_eq!(c_default_value(&PramType::Bool), "false");
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(7, 4), 8);
        assert_eq!(align_up(0, 1), 0);
    }

    #[test]
    fn test_region_c_decl() {
        let layout = MemoryLayout::compute(&sample_decls(), 1024);
        let decl = layout.regions[0].c_decl();
        assert!(decl.contains("int64_t* A"));
        assert!(decl.contains("pram_calloc(1024, sizeof(int64_t))"));
    }

    #[test]
    fn test_region_c_free() {
        let layout = MemoryLayout::compute(&sample_decls(), 1024);
        assert_eq!(layout.regions[0].c_free(), "pram_free(A);");
    }

    #[test]
    fn test_get_region() {
        let layout = MemoryLayout::compute(&sample_decls(), 1024);
        assert!(layout.get_region("A").is_some());
        assert!(layout.get_region("B").is_some());
        assert!(layout.get_region("Z").is_none());
    }

    #[test]
    fn test_crcw_priority_arrays() {
        let layout = MemoryLayout::compute(&sample_decls(), 1024);
        let (wpid, stg, init, free) = layout.generate_crcw_priority_arrays("A").unwrap();
        assert!(wpid.contains("_wpid_A"));
        assert!(stg.contains("_stg_A"));
        assert!(init.contains("INT64_MAX"));
        assert!(free.contains("pram_free(_wpid_A)"));
        assert!(free.contains("pram_free(_stg_A)"));
    }

    #[test]
    fn test_empty_layout() {
        let layout = MemoryLayout::compute(&[], 1024);
        assert!(layout.regions.is_empty());
        assert_eq!(layout.total_bytes, 0);
        assert!(layout.generate_declarations().is_empty());
    }

    #[test]
    fn test_single_bool_region_alignment() {
        let decls = vec![SharedMemoryDecl {
            name: "f".to_string(),
            elem_type: PramType::Bool,
            size: Expr::IntLiteral(10),
        }];
        let layout = MemoryLayout::compute(&decls, 1024);
        assert_eq!(layout.regions[0].alignment, 1);
        assert_eq!(layout.regions[0].count, 10);
        // total_bytes padded to at least 8
        assert!(layout.regions[0].total_bytes >= 10);
        assert_eq!(layout.regions[0].total_bytes % 8, 0);
    }

    // -----------------------------------------------------------------------
    // Tests for new additions
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_padding() {
        assert_eq!(compute_padding(8, 0), 0);
        assert_eq!(compute_padding(8, 1), 7);
        assert_eq!(compute_padding(8, 8), 0);
        assert_eq!(compute_padding(8, 9), 7);
        assert_eq!(compute_padding(4, 5), 3);
        assert_eq!(compute_padding(1, 17), 0);
        assert_eq!(compute_padding(0, 42), 0);
    }

    #[test]
    fn test_layout_optimizer_reorders_by_element_size() {
        let decls = vec![
            SharedMemoryDecl {
                name: "small".to_string(),
                elem_type: PramType::Bool,
                size: Expr::IntLiteral(100),
            },
            SharedMemoryDecl {
                name: "large".to_string(),
                elem_type: PramType::Int64,
                size: Expr::IntLiteral(100),
            },
            SharedMemoryDecl {
                name: "medium".to_string(),
                elem_type: PramType::Int32,
                size: Expr::IntLiteral(100),
            },
        ];
        let layout = MemoryLayout::compute(&decls, 1024);
        let optimized = LayoutOptimizer::optimize(&layout);

        assert_eq!(optimized.regions[0].name, "large");
        assert_eq!(optimized.regions[1].name, "medium");
        assert_eq!(optimized.regions[2].name, "small");
        // All regions should remain properly aligned.
        for r in &optimized.regions {
            assert_eq!(r.base_offset % r.alignment, 0, "region {} misaligned after optimize", r.name);
        }
    }

    #[test]
    fn test_layout_report_basic() {
        let layout = MemoryLayout::compute(&sample_decls(), 1024);
        let report = LayoutReport::generate(&layout);

        assert_eq!(report.total_memory, layout.total_bytes);
        assert_eq!(report.region_count, 3);
        assert_eq!(report.region_map.len(), 3);
        assert_eq!(report.region_map[0].0, "A");
        assert_eq!(report.region_map[1].0, "B");
        assert_eq!(report.region_map[2].0, "flags");
    }

    #[test]
    fn test_layout_report_empty() {
        let layout = MemoryLayout::compute(&[], 1024);
        let report = LayoutReport::generate(&layout);

        assert_eq!(report.total_memory, 0);
        assert_eq!(report.padding_overhead, 0);
        assert_eq!(report.region_count, 0);
        assert!(report.region_map.is_empty());
    }

    #[test]
    fn test_generate_memcpy_init() {
        let layout = MemoryLayout::compute(&sample_decls(), 1024);
        let init_vals = vec![
            ("A".to_string(), vec![1, 2, 3]),
            ("B".to_string(), vec![10, 20]),
        ];
        let code = generate_memcpy_init(&layout, &init_vals);
        assert!(code.contains("memcpy(A, _init_A_vals, 3 * sizeof(int64_t))"));
        assert!(code.contains("memcpy(B, _init_B_vals, 2 * sizeof(double))"));
        assert!(code.contains("1, 2, 3"));
        assert!(code.contains("10, 20"));
    }

    #[test]
    fn test_generate_memcpy_init_unknown_region() {
        let layout = MemoryLayout::compute(&sample_decls(), 1024);
        let init_vals = vec![("nonexistent".to_string(), vec![1, 2, 3])];
        let code = generate_memcpy_init(&layout, &init_vals);
        // Should not contain memcpy for unknown region.
        assert!(!code.contains("memcpy(nonexistent"));
    }

    #[test]
    fn test_estimate_memory_footprint() {
        let layout = MemoryLayout::compute(&sample_decls(), 1024);
        let footprint = estimate_memory_footprint(&layout);
        assert_eq!(footprint, layout.total_bytes);
        assert!(footprint > 0);
    }

    #[test]
    fn test_optimizer_preserves_region_count() {
        let layout = MemoryLayout::compute(&sample_decls(), 1024);
        let optimized = LayoutOptimizer::optimize(&layout);
        assert_eq!(optimized.regions.len(), layout.regions.len());
    }
}
