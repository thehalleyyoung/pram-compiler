//! C code templates for the PRAM-to-C compiler.
//!
//! Provides reusable code fragments: standard headers, helper macros,
//! memory-allocation wrappers, loop scaffolding, and a `CTemplate` builder
//! that stitches them into a complete C99 translation unit.

use std::fmt::Write;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Standard C headers needed by every generated program.
pub const STANDARD_HEADERS: &[&str] = &[
    "#include <stdio.h>",
    "#include <stdlib.h>",
    "#include <string.h>",
    "#include <stdint.h>",
    "#include <stdbool.h>",
    "#include <assert.h>",
];

/// Helper macros injected at the top of the generated file.
pub const HELPER_MACROS: &str = "\
#ifndef PRAM_MIN
#define PRAM_MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef PRAM_MAX
#define PRAM_MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef PRAM_SWAP
#define PRAM_SWAP(type, a, b) do { type _tmp = (a); (a) = (b); (b) = _tmp; } while(0)
#endif
";

/// Safe-malloc wrapper emitted once per translation unit.
pub const SAFE_MALLOC_WRAPPER: &str = "\
static void* pram_malloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr && size > 0) {
        fprintf(stderr, \"pram_malloc: out of memory (requested %zu bytes)\\n\", size);
        exit(1);
    }
    return ptr;
}
";

/// Safe-calloc wrapper.
pub const SAFE_CALLOC_WRAPPER: &str = "\
static void* pram_calloc(size_t count, size_t size) {
    void* ptr = calloc(count, size);
    if (!ptr && count > 0 && size > 0) {
        fprintf(stderr, \"pram_calloc: out of memory (requested %zu * %zu bytes)\\n\", count, size);
        exit(1);
    }
    return ptr;
}
";

/// Safe-free wrapper (NULL-safe).
pub const SAFE_FREE_WRAPPER: &str = "\
static void pram_free(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}
";

/// Timing helpers (optional, guarded by PRAM_TIMING).
pub const TIMING_HELPERS: &str = "\
#ifdef PRAM_TIMING
#include <time.h>
static struct timespec _pram_ts_start, _pram_ts_end;
static inline void pram_timer_start(void) {
    clock_gettime(CLOCK_MONOTONIC, &_pram_ts_start);
}
static inline double pram_timer_stop(void) {
    clock_gettime(CLOCK_MONOTONIC, &_pram_ts_end);
    return (_pram_ts_end.tv_sec - _pram_ts_start.tv_sec)
         + (_pram_ts_end.tv_nsec - _pram_ts_start.tv_nsec) * 1e-9;
}
#endif
";

// ---------------------------------------------------------------------------
// CTemplate – three-section code builder
// ---------------------------------------------------------------------------

/// A three-section template for a C translation unit.
///
/// * **header** – includes, macros, forward declarations
/// * **body**   – function definitions, the main algorithm code
/// * **footer** – cleanup, main-function return, closing braces
#[derive(Debug, Clone)]
pub struct CTemplate {
    pub header: String,
    pub body: String,
    pub footer: String,
}

impl CTemplate {
    /// Create an empty template.
    pub fn new() -> Self {
        Self {
            header: String::new(),
            body: String::new(),
            footer: String::new(),
        }
    }

    /// Create a template pre-populated with standard headers and helper macros.
    pub fn with_standard_preamble() -> Self {
        let mut t = Self::new();
        t.add_standard_headers();
        t.add_helper_macros();
        t.add_memory_wrappers();
        t
    }

    // -- header helpers -----------------------------------------------------

    pub fn add_standard_headers(&mut self) {
        for h in STANDARD_HEADERS {
            writeln!(self.header, "{}", h).unwrap();
        }
        writeln!(self.header).unwrap();
    }

    pub fn add_helper_macros(&mut self) {
        self.header.push_str(HELPER_MACROS);
        self.header.push('\n');
    }

    pub fn add_memory_wrappers(&mut self) {
        self.header.push_str(SAFE_MALLOC_WRAPPER);
        self.header.push('\n');
        self.header.push_str(SAFE_CALLOC_WRAPPER);
        self.header.push('\n');
        self.header.push_str(SAFE_FREE_WRAPPER);
        self.header.push('\n');
    }

    pub fn add_timing_helpers(&mut self) {
        self.header.push_str(TIMING_HELPERS);
        self.header.push('\n');
    }

    pub fn add_header_line(&mut self, line: &str) {
        writeln!(self.header, "{}", line).unwrap();
    }

    // -- body helpers -------------------------------------------------------

    pub fn add_body_line(&mut self, line: &str) {
        writeln!(self.body, "{}", line).unwrap();
    }

    pub fn add_body(&mut self, code: &str) {
        self.body.push_str(code);
    }

    // -- footer helpers -----------------------------------------------------

    pub fn add_footer_line(&mut self, line: &str) {
        writeln!(self.footer, "{}", line).unwrap();
    }

    pub fn add_footer(&mut self, code: &str) {
        self.footer.push_str(code);
    }

    // -- rendering ----------------------------------------------------------

    /// Concatenate all sections into the final source string.
    pub fn render(&self) -> String {
        let mut out =
            String::with_capacity(self.header.len() + self.body.len() + self.footer.len() + 4);
        out.push_str(&self.header);
        out.push_str(&self.body);
        out.push_str(&self.footer);
        out
    }
}

impl Default for CTemplate {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Snippet generators – reusable code fragments
// ---------------------------------------------------------------------------

/// Emit an `#include "..."` directive.
pub fn include_local(path: &str) -> String {
    format!("#include \"{}\"", path)
}

/// Emit an `#include <...>` directive.
pub fn include_system(header: &str) -> String {
    format!("#include <{}>", header)
}

/// Generate a C array allocation statement using `pram_calloc`.
///
/// ```c
/// int64_t* arr = (int64_t*)pram_calloc(count, sizeof(int64_t));
/// ```
pub fn alloc_array(c_type: &str, var_name: &str, count_expr: &str) -> String {
    format!(
        "{}* {} = ({}*)pram_calloc({}, sizeof({}));",
        c_type, var_name, c_type, count_expr, c_type
    )
}

/// Generate a deallocation statement.
pub fn free_array(var_name: &str) -> String {
    format!("pram_free({});", var_name)
}

/// Generate a sequential for-loop header.
///
/// ```c
/// for (int64_t i = 0; i < n; i += 1) {
/// ```
pub fn for_loop_header(var: &str, start: &str, end: &str, step: &str) -> String {
    format!(
        "for (int64_t {} = {}; {} < {}; {} += {}) {{",
        var, start, var, end, var, step
    )
}

/// Generate a while-loop header.
pub fn while_loop_header(condition: &str) -> String {
    format!("while ({}) {{", condition)
}

/// Generate a function declaration.
///
/// ```c
/// static int64_t my_func(int64_t a, int64_t b) {
/// ```
pub fn function_decl(
    return_type: &str,
    name: &str,
    params: &[(&str, &str)], // (type, name) pairs
    is_static: bool,
) -> String {
    let prefix = if is_static { "static " } else { "" };
    let param_str: String = params
        .iter()
        .map(|(ty, nm)| format!("{} {}", ty, nm))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{}{} {}({}) {{", prefix, return_type, name, param_str)
}

/// Generate a `main` function scaffold.
pub fn main_function_scaffold(body: &str) -> String {
    let mut out = String::new();
    writeln!(out, "int main(int argc, char* argv[]) {{").unwrap();
    for line in body.lines() {
        writeln!(out, "    {}", line).unwrap();
    }
    writeln!(out, "    return 0;").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

/// Generate a comment block.
pub fn comment_block(lines: &[&str]) -> String {
    let mut out = String::from("/*\n");
    for line in lines {
        writeln!(out, " * {}", line).unwrap();
    }
    out.push_str(" */\n");
    out
}

/// Generate a single-line comment.
pub fn line_comment(text: &str) -> String {
    format!("// {}", text)
}

/// Generate a CRCW-Priority write block.
///
/// For CRCW-Priority, only the processor with the lowest ID may write.
/// We implement this with a staging buffer + min-PID tracker.
pub fn crcw_priority_write_block(
    target_array: &str,
    index_expr: &str,
    value_expr: &str,
    pid_expr: &str,
    writer_pid_array: &str,
    staging_array: &str,
) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "if ({pid} < {wpid}[{idx}]) {{",
        pid = pid_expr,
        wpid = writer_pid_array,
        idx = index_expr
    )
    .unwrap();
    writeln!(
        out,
        "    {wpid}[{idx}] = {pid};",
        wpid = writer_pid_array,
        idx = index_expr,
        pid = pid_expr
    )
    .unwrap();
    writeln!(
        out,
        "    {stg}[{idx}] = {val};",
        stg = staging_array,
        idx = index_expr,
        val = value_expr
    )
    .unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(
        out,
        "/* After all processors: {target}[{idx}] = {stg}[{idx}]; */",
        target = target_array,
        idx = index_expr,
        stg = staging_array
    )
    .unwrap();
    out
}

/// Generate code to commit staged CRCW-Priority writes.
pub fn crcw_priority_commit(
    target_array: &str,
    staging_array: &str,
    writer_pid_array: &str,
    size_expr: &str,
) -> String {
    let mut out = String::new();
    writeln!(out, "for (int64_t _ci = 0; _ci < {}; _ci++) {{", size_expr).unwrap();
    writeln!(out, "    if ({}[_ci] < INT64_MAX) {{", writer_pid_array).unwrap();
    writeln!(
        out,
        "        {}[_ci] = {}[_ci];",
        target_array, staging_array
    )
    .unwrap();
    writeln!(out, "        {}[_ci] = INT64_MAX;", writer_pid_array).unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

/// Generate barrier comment (in sequential simulation, barriers are no-ops).
pub fn barrier_comment(phase: usize) -> String {
    format!("/* ---- barrier (end of phase {}) ---- */", phase)
}

// ---------------------------------------------------------------------------
// Build / profiling helpers
// ---------------------------------------------------------------------------

/// Generate a Makefile that compiles the C output with gcc.
pub fn generate_makefile(output_name: &str) -> String {
    let mut out = String::new();
    writeln!(out, "CC = gcc").unwrap();
    writeln!(out, "CFLAGS = -O2 -Wall -std=c99").unwrap();
    writeln!(out, "TARGET = {}", output_name).unwrap();
    writeln!(out, "SRC = {}.c", output_name).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "all: $(TARGET)").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "$(TARGET): $(SRC)").unwrap();
    writeln!(out, "\t$(CC) $(CFLAGS) -o $(TARGET) $(SRC)").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "clean:").unwrap();
    writeln!(out, "\trm -f $(TARGET)").unwrap();
    writeln!(out).unwrap();
    writeln!(out, ".PHONY: all clean").unwrap();
    out
}

/// Generate a shell script template for running with `perf stat`.
pub fn generate_perf_template() -> String {
    let mut out = String::new();
    writeln!(out, "#!/bin/bash").unwrap();
    writeln!(out, "# Performance measurement script").unwrap();
    writeln!(out, "set -euo pipefail").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "BINARY=\"${{1:?Usage: $0 <binary>}}\"").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "perf stat \\").unwrap();
    writeln!(out, "  -e cache-references,cache-misses \\").unwrap();
    writeln!(out, "  -e branch-instructions,branch-misses \\").unwrap();
    writeln!(out, "  -e instructions,cycles \\").unwrap();
    writeln!(out, "  \"$BINARY\"").unwrap();
    out
}

/// Generate a shell script template for running with valgrind cachegrind.
pub fn generate_valgrind_template() -> String {
    let mut out = String::new();
    writeln!(out, "#!/bin/bash").unwrap();
    writeln!(out, "# Valgrind cachegrind profiling script").unwrap();
    writeln!(out, "set -euo pipefail").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "BINARY=\"${{1:?Usage: $0 <binary>}}\"").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "valgrind --tool=cachegrind \\").unwrap();
    writeln!(out, "  --cachegrind-out-file=cachegrind.out.%p \\").unwrap();
    writeln!(out, "  \"$BINARY\"").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "echo \"Results written to cachegrind.out.*\"").unwrap();
    writeln!(
        out,
        "echo \"Run 'cg_annotate cachegrind.out.<pid>' to view.\""
    )
    .unwrap();
    out
}

// ---------------------------------------------------------------------------
// CStdlibFunction – metadata for C standard library functions
// ---------------------------------------------------------------------------

/// Common C standard library functions and their required headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CStdlibFunction {
    Printf,
    Fprintf,
    Malloc,
    Calloc,
    Free,
    Memcpy,
    Memset,
    Qsort,
    Bsearch,
    Abs,
    Exit,
    ClockGettime,
}

impl CStdlibFunction {
    /// Return the C header required for this function.
    pub fn header(&self) -> &'static str {
        match self {
            CStdlibFunction::Printf | CStdlibFunction::Fprintf => "<stdio.h>",
            CStdlibFunction::Malloc
            | CStdlibFunction::Calloc
            | CStdlibFunction::Free
            | CStdlibFunction::Qsort
            | CStdlibFunction::Bsearch
            | CStdlibFunction::Abs
            | CStdlibFunction::Exit => "<stdlib.h>",
            CStdlibFunction::Memcpy | CStdlibFunction::Memset => "<string.h>",
            CStdlibFunction::ClockGettime => "<time.h>",
        }
    }

    /// Return the C function signature.
    pub fn signature(&self) -> &'static str {
        match self {
            CStdlibFunction::Printf => "int printf(const char *format, ...)",
            CStdlibFunction::Fprintf => "int fprintf(FILE *stream, const char *format, ...)",
            CStdlibFunction::Malloc => "void *malloc(size_t size)",
            CStdlibFunction::Calloc => "void *calloc(size_t count, size_t size)",
            CStdlibFunction::Free => "void free(void *ptr)",
            CStdlibFunction::Memcpy => "void *memcpy(void *dest, const void *src, size_t n)",
            CStdlibFunction::Memset => "void *memset(void *s, int c, size_t n)",
            CStdlibFunction::Qsort => "void qsort(void *base, size_t nel, size_t width, int (*compar)(const void *, const void *))",
            CStdlibFunction::Bsearch => "void *bsearch(const void *key, const void *base, size_t nel, size_t width, int (*compar)(const void *, const void *))",
            CStdlibFunction::Abs => "int abs(int j)",
            CStdlibFunction::Exit => "void exit(int status)",
            CStdlibFunction::ClockGettime => "int clock_gettime(clockid_t clk_id, struct timespec *tp)",
        }
    }
}

/// Collect and deduplicate the required `#include` directives for a set of
/// standard-library functions.
pub fn generate_required_includes(functions: &[CStdlibFunction]) -> String {
    let mut seen = std::collections::BTreeSet::new();
    for f in functions {
        seen.insert(f.header());
    }
    let mut out = String::new();
    for h in &seen {
        writeln!(out, "#include {}", h).unwrap();
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_new_is_empty() {
        let t = CTemplate::new();
        assert!(t.header.is_empty());
        assert!(t.body.is_empty());
        assert!(t.footer.is_empty());
        assert!(t.render().is_empty());
    }

    #[test]
    fn test_standard_preamble_includes_headers() {
        let t = CTemplate::with_standard_preamble();
        let rendered = t.render();
        assert!(rendered.contains("#include <stdio.h>"));
        assert!(rendered.contains("#include <stdlib.h>"));
        assert!(rendered.contains("#include <string.h>"));
        assert!(rendered.contains("#include <stdint.h>"));
        assert!(rendered.contains("#include <stdbool.h>"));
    }

    #[test]
    fn test_standard_preamble_includes_macros() {
        let t = CTemplate::with_standard_preamble();
        let rendered = t.render();
        assert!(rendered.contains("PRAM_MIN"));
        assert!(rendered.contains("PRAM_MAX"));
        assert!(rendered.contains("PRAM_SWAP"));
    }

    #[test]
    fn test_standard_preamble_includes_wrappers() {
        let t = CTemplate::with_standard_preamble();
        let rendered = t.render();
        assert!(rendered.contains("pram_malloc"));
        assert!(rendered.contains("pram_calloc"));
        assert!(rendered.contains("pram_free"));
    }

    #[test]
    fn test_timing_helpers() {
        let mut t = CTemplate::new();
        t.add_timing_helpers();
        let rendered = t.render();
        assert!(rendered.contains("pram_timer_start"));
        assert!(rendered.contains("pram_timer_stop"));
        assert!(rendered.contains("PRAM_TIMING"));
    }

    #[test]
    fn test_alloc_array() {
        let code = alloc_array("int64_t", "arr", "n");
        assert_eq!(
            code,
            "int64_t* arr = (int64_t*)pram_calloc(n, sizeof(int64_t));"
        );
    }

    #[test]
    fn test_free_array() {
        assert_eq!(free_array("arr"), "pram_free(arr);");
    }

    #[test]
    fn test_for_loop_header() {
        let h = for_loop_header("i", "0", "n", "1");
        assert_eq!(h, "for (int64_t i = 0; i < n; i += 1) {");
    }

    #[test]
    fn test_while_loop_header() {
        assert_eq!(while_loop_header("x > 0"), "while (x > 0) {");
    }

    #[test]
    fn test_function_decl_static() {
        let d = function_decl(
            "int64_t",
            "add",
            &[("int64_t", "a"), ("int64_t", "b")],
            true,
        );
        assert_eq!(d, "static int64_t add(int64_t a, int64_t b) {");
    }

    #[test]
    fn test_function_decl_non_static() {
        let d = function_decl("void", "init", &[], false);
        assert_eq!(d, "void init() {");
    }

    #[test]
    fn test_main_function_scaffold() {
        let code = main_function_scaffold("printf(\"hello\\n\");");
        assert!(code.contains("int main("));
        assert!(code.contains("printf(\"hello\\n\");"));
        assert!(code.contains("return 0;"));
    }

    #[test]
    fn test_comment_block() {
        let c = comment_block(&["Line 1", "Line 2"]);
        assert!(c.contains("/*"));
        assert!(c.contains(" * Line 1"));
        assert!(c.contains(" * Line 2"));
        assert!(c.contains(" */"));
    }

    #[test]
    fn test_line_comment() {
        assert_eq!(line_comment("hello"), "// hello");
    }

    #[test]
    fn test_include_directives() {
        assert_eq!(include_local("myheader.h"), "#include \"myheader.h\"");
        assert_eq!(include_system("math.h"), "#include <math.h>");
    }

    #[test]
    fn test_render_order() {
        let mut t = CTemplate::new();
        t.add_header_line("/* HEADER */");
        t.add_body_line("/* BODY */");
        t.add_footer_line("/* FOOTER */");
        let r = t.render();
        let h = r.find("HEADER").unwrap();
        let b = r.find("BODY").unwrap();
        let f = r.find("FOOTER").unwrap();
        assert!(h < b);
        assert!(b < f);
    }

    #[test]
    fn test_crcw_priority_write_block() {
        let code = crcw_priority_write_block("A", "idx", "val", "pid", "_wpid_A", "_stg_A");
        assert!(code.contains("if (pid < _wpid_A[idx])"));
        assert!(code.contains("_stg_A[idx] = val;"));
    }

    #[test]
    fn test_crcw_priority_commit() {
        let code = crcw_priority_commit("A", "_stg_A", "_wpid_A", "n");
        assert!(code.contains("for (int64_t _ci = 0; _ci < n; _ci++)"));
        assert!(code.contains("A[_ci] = _stg_A[_ci];"));
    }

    #[test]
    fn test_barrier_comment() {
        let c = barrier_comment(3);
        assert!(c.contains("phase 3"));
    }

    #[test]
    fn test_template_body_append() {
        let mut t = CTemplate::new();
        t.add_body("int x = 0;\n");
        t.add_body("int y = 1;\n");
        assert_eq!(t.body, "int x = 0;\nint y = 1;\n");
    }

    #[test]
    fn test_template_footer_append() {
        let mut t = CTemplate::new();
        t.add_footer("return 0;\n");
        t.add_footer("}\n");
        assert_eq!(t.footer, "return 0;\n}\n");
    }

    // -- new tests for generate_makefile, perf/valgrind templates, CStdlibFunction --

    #[test]
    fn test_generate_makefile_structure() {
        let mf = generate_makefile("pram_sim");
        assert!(mf.contains("CC = gcc"));
        assert!(mf.contains("CFLAGS = -O2 -Wall -std=c99"));
        assert!(mf.contains("TARGET = pram_sim"));
        assert!(mf.contains("SRC = pram_sim.c"));
        assert!(mf.contains("$(CC) $(CFLAGS) -o $(TARGET) $(SRC)"));
        assert!(mf.contains("clean:"));
        assert!(mf.contains("rm -f $(TARGET)"));
        assert!(mf.contains(".PHONY: all clean"));
    }

    #[test]
    fn test_generate_perf_template() {
        let script = generate_perf_template();
        assert!(script.starts_with("#!/bin/bash"));
        assert!(script.contains("perf stat"));
        assert!(script.contains("cache-misses"));
        assert!(script.contains("branch-misses"));
        assert!(script.contains("instructions,cycles"));
    }

    #[test]
    fn test_generate_valgrind_template() {
        let script = generate_valgrind_template();
        assert!(script.starts_with("#!/bin/bash"));
        assert!(script.contains("valgrind --tool=cachegrind"));
        assert!(script.contains("cachegrind-out-file"));
        assert!(script.contains("cg_annotate"));
    }

    #[test]
    fn test_cstdlib_function_headers() {
        assert_eq!(CStdlibFunction::Printf.header(), "<stdio.h>");
        assert_eq!(CStdlibFunction::Fprintf.header(), "<stdio.h>");
        assert_eq!(CStdlibFunction::Malloc.header(), "<stdlib.h>");
        assert_eq!(CStdlibFunction::Free.header(), "<stdlib.h>");
        assert_eq!(CStdlibFunction::Memcpy.header(), "<string.h>");
        assert_eq!(CStdlibFunction::ClockGettime.header(), "<time.h>");
    }

    #[test]
    fn test_cstdlib_function_signatures() {
        assert!(CStdlibFunction::Printf.signature().contains("printf"));
        assert!(CStdlibFunction::Malloc.signature().contains("malloc"));
        assert!(CStdlibFunction::Qsort.signature().contains("qsort"));
        assert!(CStdlibFunction::Memset.signature().contains("memset"));
        assert!(CStdlibFunction::Exit.signature().contains("exit"));
        assert!(CStdlibFunction::ClockGettime
            .signature()
            .contains("clock_gettime"));
    }

    #[test]
    fn test_generate_required_includes_deduplicates() {
        let funcs = [
            CStdlibFunction::Printf,
            CStdlibFunction::Fprintf, // also stdio.h
            CStdlibFunction::Malloc,
            CStdlibFunction::Free, // also stdlib.h
            CStdlibFunction::Memcpy,
            CStdlibFunction::ClockGettime,
        ];
        let includes = generate_required_includes(&funcs);
        // Each header should appear exactly once
        assert_eq!(includes.matches("<stdio.h>").count(), 1);
        assert_eq!(includes.matches("<stdlib.h>").count(), 1);
        assert_eq!(includes.matches("<string.h>").count(), 1);
        assert_eq!(includes.matches("<time.h>").count(), 1);
    }

    #[test]
    fn test_generate_required_includes_empty() {
        let includes = generate_required_includes(&[]);
        assert!(includes.is_empty());
    }

    #[test]
    fn test_generate_required_includes_single() {
        let includes = generate_required_includes(&[CStdlibFunction::ClockGettime]);
        assert_eq!(includes.trim(), "#include <time.h>");
    }
}
