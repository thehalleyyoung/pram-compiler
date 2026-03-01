# PRAM Compiler — Public API Reference

> Crate: `pram_compiler` v0.1.0

This document covers every public struct, enum, trait, and function exposed by the
`pram_compiler` crate, organized by module.

---

## Table of Contents

1. [CLI Interface](#cli-interface)
2. [Module: `pram_ir`](#module-pram_ir)
3. [Module: `hash_partition`](#module-hash_partition)
4. [Module: `brent_scheduler`](#module-brent_scheduler)
5. [Module: `staged_specializer`](#module-staged_specializer)
6. [Module: `codegen`](#module-codegen)
7. [Module: `algorithm_library`](#module-algorithm_library)
8. [Module: `benchmark`](#module-benchmark)

---

## CLI Interface

The binary is built with [clap](https://docs.rs/clap) derive and exposes four
subcommands.

### `compile` — Compile a PRAM algorithm to C

```bash
cargo run -- compile \
  --algorithm <NAME> \
  --output <PATH>           # default: output.c
  --hash-family <FAMILY>    # siegel | two-universal | murmur | identity (default: siegel)
  --cache-line-size <BYTES> # default: 64
  --opt-level <0-3>         # default: 2
  --instrument              # include timing instrumentation
```

### `benchmark` — Run benchmarks

```bash
cargo run -- benchmark \
  --algorithm <NAME|all>    # default: all
  --sizes <CSV>             # default: 1000,10000,100000
  --trials <N>              # default: 5
  --format <FMT>            # table | csv | json (default: table)
```

### `verify` — Verify work and cache-miss bounds

```bash
cargo run -- verify \
  --algorithm <NAME|all>    # default: all
  --sizes <CSV>             # default: 1000,10000
```

### `list-algorithms` — List available algorithms

```bash
cargo run -- list-algorithms
cargo run -- list-algorithms --verbose   # show model, stmts, phases, bounds
```

### CLI Types

```rust
/// Top-level CLI argument struct.
pub struct Cli {
    pub command: Commands,
}

/// Subcommands.
pub enum Commands {
    Compile { algorithm, output, hash_family, cache_line_size, opt_level, instrument },
    Benchmark { algorithm, sizes, trials, format },
    Verify { algorithm, sizes },
    ListAlgorithms { verbose },
}
```

### Helper Functions

```rust
/// Look up an algorithm by name from the built-in library.
pub fn get_algorithm(name: &str) -> Option<PramProgram>;

/// Get all available algorithm names.
pub fn list_algorithm_names() -> Vec<&'static str>;

/// Execute the compile pipeline.
pub fn execute_compile(
    algorithm: &str, output: &str, hash_family: &str,
    cache_line_size: usize, opt_level: usize, instrument: bool,
) -> Result<(), String>;

/// Execute the benchmark pipeline.
pub fn execute_benchmark(
    algorithm: &str, sizes_str: &str, trials: usize, format: &str,
) -> Result<(), String>;

/// Execute the verify pipeline.
pub fn execute_verify(algorithm: &str, sizes_str: &str) -> Result<(), String>;

/// Execute the list-algorithms command.
pub fn execute_list_algorithms(verbose: bool);

/// Run the CLI from parsed arguments.
pub fn run(cli: Cli) -> Result<(), String>;
```

---

## Module: `pram_ir`

The PRAM intermediate representation: AST, type system, parser, validator,
printer, and builder.

### Re-exports

```rust
pub use ast::{PramProgram, Stmt, Expr, MemoryModel, BinOp, UnaryOp, ...};
pub use types::{PramType, TypeEnv, TypeError, SourceLocation};
```

### `pram_ir::ast` — Core AST

#### `MemoryModel`

```rust
pub enum MemoryModel {
    EREW,
    CREW,
    CRCWPriority,
    CRCWArbitrary,
    CRCWCommon,
}
```

Methods: `name()`, `allows_concurrent_read()`, `allows_concurrent_write()`.

#### `WriteResolution`

```rust
pub enum WriteResolution {
    Priority,   // lowest processor ID wins
    Arbitrary,  // nondeterministic
    Common,     // all writers must agree
}
```

#### `BinOp` / `UnaryOp`

```rust
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    And, Or, Xor, Shl, Shr,
    Eq, Ne, Lt, Le, Gt, Ge,
    LogicalAnd, LogicalOr,
    Min, Max, Pow,
}

pub enum UnaryOp {
    Neg, Not, BitwiseNot, Abs, Sqrt, Log2, Floor, Ceil,
}
```

#### `Expr`

```rust
pub enum Expr {
    IntLiteral(i64),
    FloatLiteral(f64),
    BoolLiteral(bool),
    Var(String),
    BinOp { op: BinOp, left: Box<Expr>, right: Box<Expr> },
    UnaryOp { op: UnaryOp, operand: Box<Expr> },
    SharedRead { array: String, index: Box<Expr> },
    ProcessorId,
    NumProcessors,
    FunctionCall { name: String, args: Vec<Expr> },
    ArrayIndex { array: Box<Expr>, index: Box<Expr> },
    Conditional { condition: Box<Expr>, then_expr: Box<Expr>, else_expr: Box<Expr> },
    Cast { expr: Box<Expr>, target_type: PramType },
    // ... additional variants
}
```

#### `Stmt`

```rust
pub enum Stmt {
    Assign(String, Expr),
    SharedWrite { array: String, index: Expr, value: Expr },
    ParallelFor { proc_var: String, num_procs: Expr, body: Vec<Stmt> },
    SeqFor { var: String, start: Expr, end: Expr, step: Option<Expr>, body: Vec<Stmt> },
    While { condition: Expr, body: Vec<Stmt> },
    If { condition: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt> },
    Barrier,
    LocalDecl(String, PramType, Option<Expr>),
    AllocShared { name: String, elem_type: PramType, size: Expr },
    FreeShared(String),
    Return(Option<Expr>),
    Block(Vec<Stmt>),
    Comment(String),
    // ... additional variants
}
```

Methods on `Stmt`: `stmt_count()`, `writes_shared()`, `reads_shared()`.

#### `PramProgram`

```rust
pub struct PramProgram {
    pub name: String,
    pub memory_model: MemoryModel,
    pub parameters: Vec<Parameter>,
    pub shared_memory: Vec<SharedMemoryDecl>,
    pub body: Vec<Stmt>,
    pub num_processors: Expr,
    pub work_bound: Option<String>,
    pub time_bound: Option<String>,
    pub description: Option<String>,
}
```

Methods:
- `new(name, model) -> Self`
- `total_stmts() -> usize`
- `uses_concurrent_writes() -> bool`
- `uses_concurrent_reads() -> bool`
- `parallel_step_count() -> usize`
- `shared_region_names() -> Vec<String>`
- `add_parameter(name, ty)`

#### `SharedMemoryDecl` / `Parameter` / `SharedAccess` / `ParallelPhase`

```rust
pub struct SharedMemoryDecl { pub name: String, pub elem_type: PramType, pub size: Expr }
pub struct Parameter { pub name: String, pub param_type: PramType }
pub struct SharedAccess { /* read/write tracking per array */ }
pub struct ParallelPhase { pub num_procs: Expr, pub body: Vec<Stmt> }

pub fn split_into_phases(body: &[Stmt], num_procs: &Expr) -> Vec<ParallelPhase>;
```

### `pram_ir::types` — Type System

```rust
pub enum PramType {
    Int, Float, Bool, Void,
    Array(Box<PramType>),
    Pointer(Box<PramType>),
    // ...
}

pub struct TypeEnv { /* variable → type bindings */ }
pub struct TypeError { pub message: String, pub location: Option<SourceLocation> }
pub struct SourceLocation { pub line: usize, pub column: usize }

pub enum TypeCompatibility { Exact, Coercible, Incompatible }
pub fn check_compatibility(target: &PramType, source: &PramType) -> TypeCompatibility;
pub fn typecheck_expr(expr: &Expr, env: &TypeEnv) -> Result<PramType, TypeError>;
```

### `pram_ir::memory_model` — Memory Model Semantics

```rust
pub enum ConflictType { ReadWrite, WriteWrite, WriteRead }
pub struct AccessRecord { /* processor, address, read/write */ }
pub struct PendingWrite { /* buffered write for CRCW resolution */ }
pub struct ConflictReport { /* detected conflicts in a parallel step */ }
pub struct MemoryModelChecker { /* validates accesses against EREW/CREW/CRCW rules */ }

pub struct CRCWResolver;  // resolves concurrent writes by policy

pub struct MemoryState { /* tracks shared memory contents */ }
pub struct StepSimulator { /* simulates one parallel step */ }
pub struct SimAccess { pub processor: usize, pub address: u64, pub is_write: bool }
pub struct SimResult { /* post-step state + violations */ }
pub struct ModelViolation { pub severity: ViolationSeverity, pub message: String }
pub enum ViolationSeverity { Error, Warning, Info }

pub fn resolve_crcw_writes(
    writes: &[(usize, i64)], policy: WriteResolution,
) -> Vec<(u64, i64)>;

pub enum AccessPattern { Sequential, Strided, Random, Mixed }
```

### `pram_ir::parser`

```rust
pub struct ParseError { pub message: String, pub line: usize, pub column: usize }

/// Parse a single PRAM program from source text.
pub fn parse_program(source: &str) -> Result<PramProgram, ParseError>;

/// Parse multiple PRAM programs from a single source file.
pub fn parse_multiple(source: &str) -> Result<Vec<PramProgram>, ParseError>;
```

### `pram_ir::validator`

```rust
pub enum ValidationErrorKind { TypeError, ModelViolation, StructuralError, ... }
pub struct ValidationError { pub kind: ValidationErrorKind, pub message: String }
pub enum IssueSeverity { Error, Warning, Info }
pub struct ValidationIssue { pub severity: IssueSeverity, pub message: String }
pub struct ValidationReport { /* aggregates all issues */ }

/// Run all validation checks on a program.
pub fn validate_program(program: &PramProgram) -> Vec<ValidationError>;

/// Check shared memory access patterns against the declared memory model.
pub fn validate_memory_accesses(program: &PramProgram) -> Vec<ValidationIssue>;

/// Check that processor index expressions stay in bounds.
pub fn validate_processor_bounds(program: &PramProgram) -> Vec<ValidationIssue>;

/// Check that barriers are correctly placed between parallel phases.
pub fn validate_barrier_structure(program: &PramProgram) -> Vec<ValidationIssue>;

/// Check for potential non-termination.
pub fn validate_termination(program: &PramProgram) -> Vec<ValidationIssue>;
```

### `pram_ir::printer`

```rust
pub struct PrettyPrinterConfig { pub indent: usize, pub max_width: usize, ... }
pub struct PramPrinter { /* configurable pretty-printer */ }

pub fn print_compact(program: &PramProgram) -> String;
pub fn print_annotated(program: &PramProgram) -> String;
pub fn print_expr_precedence(expr: &Expr) -> String;
pub fn print_to_file(program: &PramProgram, path: &str) -> std::io::Result<()>;
```

### `pram_ir::builder` — Fluent Builder API

```rust
pub struct PramBuilder { /* internal state */ }

impl PramBuilder {
    pub fn new(name: &str, model: MemoryModel) -> Self;

    // Metadata
    pub fn add_param(&mut self, name: &str, ty: PramType) -> &mut Self;
    pub fn add_shared_memory(&mut self, name: &str, elem_ty: PramType, size: Expr) -> &mut Self;
    pub fn set_num_processors(&mut self, expr: Expr) -> &mut Self;
    pub fn set_work_bound(&mut self, bound: &str) -> &mut Self;
    pub fn set_time_bound(&mut self, bound: &str) -> &mut Self;
    pub fn set_description(&mut self, desc: &str) -> &mut Self;

    // Statement construction
    pub fn add_stmt(&mut self, stmt: Stmt) -> &mut Self;
    pub fn parallel_for(&mut self, proc_var: &str, num_procs: Expr, body: Vec<Stmt>) -> &mut Self;
    pub fn seq_for(&mut self, var: &str, start: Expr, end: Expr, step: Option<Expr>, body: Vec<Stmt>) -> &mut Self;
    pub fn while_loop(&mut self, condition: Expr, body: Vec<Stmt>) -> &mut Self;
    pub fn if_then_else(&mut self, condition: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt>) -> &mut Self;
    pub fn assign(&mut self, var: &str, expr: Expr) -> &mut Self;
    pub fn local_decl(&mut self, name: &str, ty: PramType, init: Option<Expr>) -> &mut Self;
    pub fn barrier(&mut self) -> &mut Self;
    pub fn alloc_shared(&mut self, name: &str, elem_ty: PramType, size: Expr) -> &mut Self;
    pub fn free_shared(&mut self, name: &str) -> &mut Self;

    // Finalize
    pub fn build(self) -> PramProgram;
}
```

**Example — building a prefix-sum program:**

```rust
use pram_compiler::pram_ir::builder::PramBuilder;
use pram_compiler::pram_ir::ast::*;
use pram_compiler::pram_ir::types::PramType;

let program = {
    let mut b = PramBuilder::new("prefix_sum", MemoryModel::EREW);
    b.add_param("n", PramType::Int)
     .add_shared_memory("A", PramType::Int, Expr::Var("n".into()))
     .set_num_processors(Expr::Var("n".into()))
     .set_work_bound("O(n log n)")
     .set_time_bound("O(log n)")
     .set_description("Parallel prefix sum via pointer jumping");
    // ... add parallel_for, barriers, etc.
    b.build()
};
```

---

## Module: `hash_partition`

Stage 1 of the pipeline. Assigns PRAM addresses to cache-aligned blocks.

### Trait: `HashFunction`

```rust
/// Trait implemented by all hash families.
pub trait HashFunction {
    fn hash(&self, key: u64) -> u64;
    fn name(&self) -> &str;
}

pub type BlockId = usize;
```

### `hash_partition::siegel_hash` — Siegel k-wise Independent Hashing

```rust
pub struct SiegelHash {
    /* k-wise independent polynomial hash over GF(2^64) */
}

pub struct SiegelHashFamily { /* generates SiegelHash instances */ }
pub struct SiegelHashBatch { /* batch evaluation for multiple keys */ }
pub struct HashQualityReport {
    pub max_load: u64,
    pub mean_load: f64,
    pub chi_squared: f64,
    // ...
}

pub fn evaluate_polynomial_mod(coeffs: &[u64], x: u64) -> u64;
pub fn collision_probability(k: usize, n: usize, m: u64) -> f64;
pub fn generate_coefficients(k: usize, seed: u64) -> Vec<u64>;
pub fn hash_quality_report(hash: &SiegelHash, m: u64, keys: &[u64]) -> HashQualityReport;
pub fn coefficient_of_variation(hash: &SiegelHash, m: u64, keys: &[u64]) -> f64;
pub fn empirical_collision_rate<R: Rng>(hash: &SiegelHash, m: u64, n: usize, rng: &mut R) -> f64;
pub fn distribution_chi_squared(hash: &SiegelHash, m: u64, keys: &[u64]) -> f64;
```

### `hash_partition::two_universal` — 2-Universal Hashing

```rust
pub struct TwoUniversalHash { /* h(x) = (ax + b) mod p mod m */ }
pub struct TwoUniversalFamily { /* generates TwoUniversalHash instances */ }
pub struct TabulationHash { /* simple tabulation hashing */ }
pub struct Tabulation4Hash { /* 4-way tabulation hashing */ }
pub struct VarianceAnalysis { pub mean: f64, pub variance: f64, pub max_load: u64, ... }

pub fn empirical_collision_probability<R: Rng>(...) -> f64;
pub fn theoretical_collision_bound(m: u64) -> f64;
pub fn bucket_load_distribution(hash: &TwoUniversalHash, keys: &[u64], num_buckets: u64) -> Vec<u64>;
pub fn max_bucket_load(counts: &[u64]) -> u64;
pub fn bucket_load_variance(counts: &[u64]) -> f64;
pub fn expected_max_load(n: u64, m: u64) -> f64;
pub fn variance_analysis(hash: &TwoUniversalHash, keys: &[u64], m: u64) -> VarianceAnalysis;
```

### `hash_partition::murmur` — MurmurHash3

```rust
pub struct MurmurHasher { /* wraps MurmurHash3 as a HashFunction */ }

pub fn fmix64(key: u64) -> u64;
pub fn fmix64_inv(key: u64) -> u64;
pub fn murmur3_64(key: u64, seed: u64) -> u64;
pub fn murmur3_64_bytes(data: &[u8], seed: u64) -> u64;
pub fn murmur3_128(key: u64, seed: u64) -> (u64, u64);
pub fn murmur3_bulk(keys: &[u64], seed: u64) -> Vec<u64>;
pub fn murmur3_string(data: &[u8], seed: u64) -> u64;

pub struct ChiSquaredResult { pub statistic: f64, pub p_value: f64, ... }
pub fn chi_squared_uniformity_test(hash: &MurmurHasher, ...) -> ChiSquaredResult;
pub fn avalanche_matrix(seed: u64, num_samples: usize) -> Vec<Vec<f64>>;
pub fn avalanche_quality(matrix: &[Vec<f64>]) -> f64;
pub fn bit_bias(seed: u64, num_keys: u64) -> [f64; 64];
```

### `hash_partition::identity` — Identity Hash (Baseline)

```rust
pub struct IdentityHash;  // implements HashFunction; hash(x) = x
```

### `hash_partition::block_assignment` — Block Assignment

```rust
pub struct BlockAssigner {
    /* assigns addresses to cache-line-aligned blocks */
}

pub struct RangeBlockAssigner { /* contiguous range-based assignment */ }

pub struct CacheLevel { pub size: usize, pub line_size: usize, pub associativity: usize }
pub struct CacheAwareBlockAssigner { /* multi-level cache-aware assignment */ }

pub struct BlockStats { pub num_blocks: usize, pub max_load: usize, pub mean_load: f64, ... }
pub struct BlockStatistics { /* extended statistics */ }

pub fn block_statistics(assignments: &[BlockId]) -> BlockStats;
pub fn optimize_assignment(assignments: &mut [BlockId]);
```

### `hash_partition::overflow_analysis` — Overflow Analysis

```rust
pub enum HashFamilyType { Siegel, TwoUniversal, Murmur, Identity }

pub struct OverflowReport {
    pub max_overflow: usize,
    pub mean_overflow: f64,
    pub theoretical_bound: f64,
    // ...
}

pub struct OverflowAnalyzer { /* analyzes per-block overflow for a hash family */ }
pub struct OverflowDistribution { /* CDF of overflow values */ }
pub struct OverflowHistogram { /* histogram of overflow counts */ }
pub struct TailBounds { pub markov: f64, pub chebyshev: f64, pub chernoff: f64 }

pub fn predict_overflow_bound(hash_type: &HashFamilyType, n: usize, block_size: u64) -> f64;
pub fn tail_bounds(mean: f64, variance: f64, threshold: f64) -> TailBounds;
```

### `hash_partition::partition_engine` — Partition Engine

```rust
pub enum HashFamilyChoice { Siegel { k: usize }, TwoUniversal, Murmur { seed: u64 }, Identity }

pub struct PartitionPlan { /* planned partition: hash choice + block size + layout */ }
pub struct PartitionResult { /* executed partition: assignments + overflow stats */ }
pub struct PartitionQualityReport { pub num_blocks: usize, pub max_overflow: usize, ... }
pub struct MultiLevelPartition { /* hierarchical partition for multi-level caches */ }

pub struct PartitionEngine { /* orchestrates hash selection, assignment, and analysis */ }

pub fn optimize_partition(plan: &mut PartitionPlan);
pub fn estimate_cache_misses(plan: &PartitionPlan, schedule_order: &[usize]) -> u64;
pub fn partition_quality_report(result: &PartitionResult) -> PartitionQualityReport;
```

**Example — partitioning an address space:**

```rust
use pram_compiler::hash_partition::partition_engine::*;

let engine = PartitionEngine::new(HashFamilyChoice::Siegel { k: 20 }, 64);
let result = engine.partition(addresses, num_addresses);
let report = partition_quality_report(&result);
println!("Blocks: {}, Max overflow: {}", report.num_blocks, report.max_overflow);
```

---

## Module: `brent_scheduler`

Stage 2 of the pipeline. Extracts work-optimal sequential schedules from
block-level dependency graphs.

### `brent_scheduler::dependency_graph`

```rust
pub enum DepKind { ReadAfterWrite, WriteAfterRead, WriteAfterWrite, Control }

pub struct OperationNode {
    pub id: usize,
    pub block_id: usize,
    pub op_type: OpType,
    pub processor: usize,
    pub address: Option<u64>,
    // ...
}

pub struct DepEdge { pub from: usize, pub to: usize, pub kind: DepKind }

pub struct DependencyGraph {
    /* petgraph-based DAG of operations with dependency edges */
}

pub struct GraphStats { pub num_nodes: usize, pub num_edges: usize, pub critical_path: usize, ... }
```

### `brent_scheduler::scheduler`

```rust
pub struct SchedulerConfig {
    pub enable_locality_opt: bool,
    pub cache_line_size: usize,
    pub cache_size: usize,
    pub block_reorder: bool,
    pub max_lookahead: usize,
    // ...
}

pub struct BrentScheduler { /* configurable scheduler */ }
pub struct SchedulerStats { pub total_ops: usize, pub cache_misses: usize, ... }
```

### `brent_scheduler::schedule`

```rust
pub enum OpType {
    Read { array: String, index: u64 },
    Write { array: String, index: u64, value: i64 },
    Compute,
    Barrier,
    // ...
}

pub struct ScheduleEntry {
    pub op_id: usize,
    pub block_id: usize,
    pub op_type: OpType,
    pub processor: usize,
    // ...
}

pub struct Schedule {
    pub entries: Vec<ScheduleEntry>,
    /* + analysis methods */
}

pub struct ScheduleAnalysis { pub total_work: usize, pub cache_transitions: usize, ... }
```

### `brent_scheduler::work_optimal`

```rust
pub struct SequentialSchedule { pub ops: Vec<ScheduledOp>, pub critical_path: usize }
pub struct ScheduledOp { pub node_id: usize, pub level: usize, pub block_id: usize }

/// Extract a topological-order work-optimal schedule.
pub fn extract_schedule(graph: &DependencyGraph) -> SequentialSchedule;

/// Extract with custom comparator for tie-breaking.
pub fn extract_schedule_with_order<F>(graph: &DependencyGraph, cmp: F) -> SequentialSchedule
    where F: FnMut(&OperationNode, &OperationNode) -> std::cmp::Ordering;

/// Extract with explicit priority function.
pub fn extract_schedule_with_priorities(
    graph: &DependencyGraph, priorities: &[usize],
) -> SequentialSchedule;

pub fn compute_level_widths(schedule: &SequentialSchedule) -> Vec<usize>;
pub fn balance_levels(schedule: &SequentialSchedule) -> SequentialSchedule;
pub fn verify_schedule_correctness(
    graph: &DependencyGraph, schedule: &SequentialSchedule,
) -> bool;
```

### `brent_scheduler::locality_order`

```rust
pub struct LocalityOptimizer { /* reorders operations for cache locality */ }

pub fn estimate_cache_misses(entries: &[ScheduleEntry], cache_blocks: usize) -> usize;
pub fn estimate_misses_seq(schedule: &SequentialSchedule, cache_blocks: usize) -> usize;
pub fn count_block_transitions(entries: &[ScheduleEntry]) -> usize;
pub fn count_transitions_seq(schedule: &SequentialSchedule) -> usize;
pub fn compare_orderings(orig: &[ScheduleEntry], opt: &[ScheduleEntry], cache_blocks: usize) -> (usize, usize);
pub fn hilbert_curve_order(ops: &[ScheduleEntry]) -> Vec<usize>;
pub fn reuse_distance_analysis(entries: &[ScheduleEntry]) -> Vec<usize>;
pub fn optimal_block_ordering_greedy(blocks: &[Vec<ScheduleEntry>]) -> Vec<usize>;

pub struct ComparisonReport { /* compares multiple ordering strategies */ }
pub fn compare_multiple_orderings(entries: &[ScheduleEntry], cache_blocks: usize) -> ComparisonReport;
```

### `brent_scheduler::cost_analyzer`

```rust
pub struct CostReport { pub total_work: u64, pub cache_misses: u64, pub critical_path: usize, ... }

pub fn analyze_schedule(schedule: &Schedule, cache_config: &CacheConfig) -> CostReport;
pub fn analyze_with_graph(graph: &DependencyGraph, schedule: &Schedule) -> CostReport;
pub fn compare_schedules(a: &CostReport, b: &CostReport) -> ScheduleComparison;

pub struct ScheduleComparison { pub work_ratio: f64, pub cache_miss_ratio: f64, ... }
pub struct CacheConfig { pub line_size: usize, pub cache_size: usize, pub associativity: usize }
pub struct CacheAnalysis { pub cold_misses: u64, pub capacity_misses: u64, pub conflict_misses: u64, ... }
pub struct BottleneckReport { /* identifies scheduling bottlenecks */ }

pub fn detailed_cache_analysis(schedule: &Schedule, config: &CacheConfig) -> CacheAnalysis;
pub fn predict_execution_time(report: &CostReport, ...) -> f64;
pub fn bottleneck_analysis(schedule: &Schedule, graph: &DependencyGraph) -> BottleneckReport;
pub fn sensitivity_analysis(schedule: &Schedule, ...) -> Vec<CostReport>;
```

---

## Module: `staged_specializer`

Stage 3a. Partial evaluation pipeline that eliminates simulation overhead.

### `staged_specializer::specializer`

```rust
pub struct SpecializerConfig {
    pub enable_dispatch: bool,       // parallel_for → sequential
    pub enable_arbitration: bool,    // memory model → direct ops
    pub enable_residualize: bool,    // hash lookups → direct access
    pub enable_partial_eval: bool,   // constant prop + DCE + strength reduction
    pub enable_work_check: bool,     // verify work preservation
    pub unroll_threshold: usize,     // default: 32
    pub work_c1: usize,             // work bound constant (default: 4)
    pub work_c2_per_region: usize,  // per-region overhead (default: 10)
}

impl SpecializerConfig {
    pub fn default() -> Self;       // all passes enabled
    pub fn all_disabled() -> Self;  // all passes disabled
}

pub struct SpecializationResult {
    pub body: Vec<Stmt>,
    pub pre_work: usize,
    pub post_work: usize,
    pub work_preserved: bool,
    pub passes_applied: Vec<String>,
    pub warnings: Vec<String>,
}

pub struct Specializer { /* orchestrates all transformation passes */ }

impl Specializer {
    pub fn new(config: SpecializerConfig) -> Self;
    pub fn with_default_config() -> Self;
    pub fn specialize(&self, program: &PramProgram) -> SpecializationResult;
}
```

### `staged_specializer::specializer_ir` — Staged IR

```rust
pub enum SType { Int, Float, Bool, Array(Box<SType>), Void }
pub type BlockId = usize;

pub enum SExpr {
    Const(i64), FConst(f64), BConst(bool),
    Var(String), BinOp { ... }, UnaryOp { ... },
    Load { base: String, offset: Box<SExpr> },
    // ...
}

pub struct MemoryAccessDescriptor { pub array: String, pub index: SExpr, pub is_write: bool }

pub enum SInstr {
    Assign(String, SExpr),
    Store { base: String, offset: SExpr, value: SExpr },
    Branch { cond: SExpr, true_block: BlockId, false_block: BlockId },
    Loop { header: BlockId, body: BlockId, exit: BlockId },
    Call { func: String, args: Vec<SExpr> },
    // ...
}

pub struct SBlock { pub id: BlockId, pub instrs: Vec<SInstr>, pub terminator: SInstr }
pub struct SProgram { pub blocks: Vec<SBlock>, pub entry: BlockId }
```

### `staged_specializer::partial_eval`

```rust
pub enum KnownValue { Int(i64), Float(f64), Bool(bool), Unknown }
pub struct PartialEnv { /* maps variables to known values */ }
pub struct PartialEvaluator { /* constant propagation + dead code elimination */ }

pub fn loop_unrolling(stmt: &Stmt, max_iters: usize) -> Vec<Stmt>;
pub fn inline_function_calls(stmts: &[Stmt], defs: &HashMap<String, Vec<Stmt>>) -> Vec<Stmt>;
pub fn algebraic_simplify(expr: &Expr) -> Expr;
pub fn common_subexpression_elimination(stmts: &[Stmt]) -> Vec<Stmt>;
```

### `staged_specializer::hash_residualize`

```rust
pub struct BlockAssignment { pub address_to_block: Vec<usize>, pub block_size: usize }

pub enum AddressMap {
    Direct { offset: usize },
    Hashed { block_id: usize, intra_offset: usize },
    // ...
}

pub struct ResidualizePass { /* replaces hash lookups with direct array indexing */ }
pub struct BatchResidualizer { /* batch residualization across multiple arrays */ }
pub struct ResidualSavings { pub ops_removed: usize, pub ops_remaining: usize }

pub fn direct_index_expr(block_id: usize, offset: usize, block_size: usize) -> Expr;
pub fn contiguous_direct_index(addr_expr: &Expr) -> Expr;
pub fn compute_block_id_and_offset(addr_expr: &Expr, block_size: usize) -> (Expr, Expr);
pub fn estimate_savings(program: &PramProgram, assignments: &BlockAssignment) -> ResidualSavings;
pub fn verify_residualization(before: &[Stmt], after: &[Stmt]) -> bool;
```

### `staged_specializer::processor_dispatch`

```rust
pub struct DispatchConfig { pub unroll_threshold: usize, pub strip_mine_factor: usize, ... }
pub struct ProcessorDispatch { /* transforms parallel_for into sequential code */ }
pub struct DispatchStats { pub parallel_fors_converted: usize, pub ops_unrolled: usize, ... }

pub fn analyze_dispatch(stmts: &[Stmt]) -> DispatchStats;
pub fn partial_unroll(stmt: &Stmt, factor: usize) -> Vec<Stmt>;
pub fn strip_mine(stmt: &Stmt, factor: usize) -> Vec<Stmt>;
```

### `staged_specializer::model_arbitration`

```rust
pub struct ModelArbitrationPass { /* eliminates CRCW conflict resolution overhead */ }
pub struct ArbitrationStats { pub writes_resolved: usize, pub reads_simplified: usize, ... }
pub struct WriteConflict { pub address: u64, pub writers: Vec<usize> }

pub fn write_resolution_for(model: MemoryModel) -> Option<WriteResolution>;
pub fn requires_write_arbitration(model: MemoryModel) -> bool;
pub fn requires_read_arbitration(model: MemoryModel) -> bool;
pub fn priority_winner(proc_ids: &[usize]) -> Option<usize>;
pub fn common_value_check(values: &[i64]) -> bool;
pub fn analyze_arbitration(stmts: &[Stmt], model: MemoryModel) -> ArbitrationStats;
pub fn detect_write_conflicts(stmts: &[Stmt]) -> Vec<WriteConflict>;
pub fn optimize_write_ordering(stmts: &[Stmt], model: MemoryModel) -> Vec<Stmt>;
```

### `staged_specializer::work_preservation`

```rust
pub struct WorkCount { pub arithmetic: u64, pub memory: u64, pub control: u64, pub total: u64 }
pub struct WorkCounter;  // counts operations in statement trees

pub struct WorkBoundChecker { /* asserts W_post ≤ c₁ · W_pre + c₂ */ }
pub struct WorkBoundViolation { pub expected: u64, pub actual: u64, pub message: String }

pub struct PhaseWork { pub phase_id: usize, pub work: WorkCount }
pub struct DetailedWorkReport { pub phases: Vec<PhaseWork>, pub total: WorkCount }
pub struct WorkInflation { pub ratio: f64, pub absolute_increase: u64 }

pub fn analyze_work_per_phase(body: &[Stmt]) -> Vec<PhaseWork>;
pub fn estimate_cache_work(body: &[Stmt], cache_line_size: usize) -> u64;
pub fn work_inflation_analysis(pre: &WorkCount, post: &WorkCount) -> WorkInflation;
```

---

## Module: `codegen`

Stage 3b. Emits standalone C99 source files.

### `codegen::generator`

```rust
pub struct GeneratorConfig {
    pub opt_level: u8,            // 0 = none, 1 = fold, 2 = fold + loop opt
    pub include_timing: bool,     // timing instrumentation
    pub include_assertions: bool, // runtime assertions
    pub tile_size: usize,         // loop tile size (0 = disable)
    pub min_trip_for_tiling: usize,
    pub default_region_size: usize,
}

impl GeneratorConfig {
    pub fn new() -> Self;       // opt_level=2, assertions on
    pub fn debug() -> Self;     // opt_level=0, assertions on
    pub fn release() -> Self;   // opt_level=2, assertions off
}

pub struct CodeGenerator { pub config: GeneratorConfig }

impl CodeGenerator {
    pub fn new(config: GeneratorConfig) -> Self;
    pub fn generate(&self, program: &PramProgram) -> String;
}

pub struct GenerationReport { pub lines: usize, pub bytes: usize, pub optimizations: Vec<String> }
pub fn generate_with_variants(program: &PramProgram, configs: &[GeneratorConfig]) -> Vec<String>;
pub fn estimate_output_size(program: &PramProgram) -> usize;
pub fn validate_generated_code(c_code: &str) -> Vec<String>;
```

**Example — end-to-end compilation:**

```rust
use pram_compiler::algorithm_library;
use pram_compiler::codegen::generator::{CodeGenerator, GeneratorConfig};

let program = algorithm_library::sorting::bitonic_sort();

let config = GeneratorConfig {
    opt_level: 2,
    include_timing: true,
    include_assertions: false,
    ..GeneratorConfig::default()
};

let generator = CodeGenerator::new(config);
let c_code = generator.generate(&program);
std::fs::write("bitonic_sort.c", &c_code).unwrap();
```

### `codegen::c_emitter`

```rust
pub struct CEmitter { /* converts AST nodes to C99 source text */ }
```

### `codegen::memory_layout`

```rust
pub struct SharedRegionLayout { pub name: String, pub offset: usize, pub size: usize, ... }
pub struct MemoryLayout { pub regions: Vec<SharedRegionLayout>, pub total_size: usize }
pub struct LayoutOptimizer;  // reorders regions to minimize padding
pub struct LayoutReport { pub total_bytes: usize, pub padding_bytes: usize, ... }

pub fn pram_type_to_c(ty: &PramType) -> String;
pub fn c_default_value(ty: &PramType) -> &'static str;
pub fn compute_padding(alignment: usize, current_offset: usize) -> usize;
pub fn generate_memcpy_init(layout: &MemoryLayout, initial_values: &[(String, Vec<i64>)]) -> String;
pub fn estimate_memory_footprint(layout: &MemoryLayout) -> usize;
```

### `codegen::loop_restructure`

```rust
pub struct TileConfig { pub tile_size: usize, pub min_trip: usize, ... }
pub struct LoopRestructurer { /* loop tiling, fusion, fission */ }

pub enum AccessPattern { Sequential, Strided, Random, Mixed }
pub fn analyze_loop_access_pattern(stmt: &Stmt) -> AccessPattern;
pub fn should_tile(pattern: &AccessPattern, trip_count: usize) -> bool;
pub fn loop_fusion(loops: &[Stmt]) -> Vec<Stmt>;
pub fn loop_fission(loop_stmt: &Stmt) -> Vec<Stmt>;
```

### `codegen::constant_fold`

```rust
pub struct ConstantFolder { /* folds constant expressions at compile time */ }
pub struct FoldingStats { pub expressions_folded: usize, ... }

pub fn fold_program(program: &PramProgram) -> PramProgram;
pub fn count_folded(before: &Expr, after: &Expr) -> usize;
pub fn deep_fold(expr: &Expr) -> Expr;
```

### `codegen::template` — C Code Templates

```rust
pub const STANDARD_HEADERS: &[&str];
pub const HELPER_MACROS: &str;
pub const SAFE_MALLOC_WRAPPER: &str;
pub const SAFE_CALLOC_WRAPPER: &str;
pub const SAFE_FREE_WRAPPER: &str;
pub const TIMING_HELPERS: &str;

pub struct CTemplate { /* assembles header + body + footer */ }

// Template helpers
pub fn include_local(path: &str) -> String;
pub fn include_system(header: &str) -> String;
pub fn alloc_array(c_type: &str, var_name: &str, count_expr: &str) -> String;
pub fn free_array(var_name: &str) -> String;
pub fn for_loop_header(var: &str, start: &str, end: &str, step: &str) -> String;
pub fn while_loop_header(condition: &str) -> String;
pub fn function_decl(return_type: &str, name: &str, params: &[(String, String)]) -> String;
pub fn main_function_scaffold(body: &str) -> String;
pub fn comment_block(lines: &[&str]) -> String;
pub fn line_comment(text: &str) -> String;
pub fn crcw_priority_write_block(...) -> String;
pub fn crcw_priority_commit(...) -> String;
pub fn barrier_comment(phase: usize) -> String;
pub fn generate_makefile(output_name: &str) -> String;
pub fn generate_perf_template() -> String;
pub fn generate_valgrind_template() -> String;

pub enum CStdlibFunction { Malloc, Calloc, Free, Printf, Memcpy, Memset, ... }
pub fn generate_required_includes(functions: &[CStdlibFunction]) -> String;
```

---

## Module: `algorithm_library`

26+ classic PRAM algorithms, each returning a `PramProgram`.

### Catalog

```rust
pub struct AlgorithmEntry {
    pub name: &'static str,
    pub category: &'static str,
    pub memory_model: MemoryModel,
    // ...
}

pub fn catalog() -> Vec<AlgorithmEntry>;
```

### Sorting (`algorithm_library::sorting`)

```rust
pub fn bitonic_sort() -> PramProgram;          // EREW, O(n log²n) work, O(log²n) time
pub fn cole_merge_sort() -> PramProgram;       // CREW, O(n log n) work, O(log n) time
pub fn sample_sort() -> PramProgram;           // CREW
pub fn odd_even_merge_sort() -> PramProgram;   // EREW
```

### Graph (`algorithm_library::graph`)

```rust
pub fn shiloach_vishkin() -> PramProgram;      // CRCW, connected components
pub fn boruvka_mst() -> PramProgram;           // CRCW-Priority, minimum spanning tree
pub fn parallel_bfs() -> PramProgram;          // CREW, breadth-first search
pub fn euler_tour() -> PramProgram;            // CREW, Euler tour construction
```

### Connectivity (`algorithm_library::connectivity`)

```rust
pub fn vishkin_connectivity() -> PramProgram;  // CREW, deterministic connectivity
pub fn ear_decomposition() -> PramProgram;     // CREW, Reif's ear decomposition
```

### List (`algorithm_library::list`)

```rust
pub fn prefix_sum() -> PramProgram;            // EREW, parallel prefix sum / scan
pub fn list_ranking() -> PramProgram;          // EREW, pointer jumping
pub fn compact() -> PramProgram;               // CREW, parallel compaction
```

### Arithmetic (`algorithm_library::arithmetic`)

```rust
pub fn parallel_addition() -> PramProgram;     // CREW, carry-lookahead
pub fn parallel_multiplication() -> PramProgram; // CREW
pub fn matrix_multiply() -> PramProgram;       // CREW
pub fn matrix_vector_multiply() -> PramProgram; // CREW
```

### Geometry (`algorithm_library::geometry`)

```rust
pub fn convex_hull() -> PramProgram;           // CREW
pub fn closest_pair() -> PramProgram;          // CREW
```

### Tree (`algorithm_library::tree`)

```rust
pub fn tree_contraction() -> PramProgram;      // EREW
pub fn lca() -> PramProgram;                   // CREW, lowest common ancestor
```

### String (`algorithm_library::string_algo`)

```rust
pub fn string_matching() -> PramProgram;       // CREW
pub fn suffix_array() -> PramProgram;          // CREW
```

### Search (`algorithm_library::search`)

```rust
pub fn parallel_binary_search() -> PramProgram;        // CREW
pub fn parallel_interpolation_search() -> PramProgram;  // CREW
```

### Selection (`algorithm_library::selection`)

```rust
pub fn parallel_selection() -> PramProgram;    // CREW, kth element
```

---

## Module: `benchmark`

Measurement, verification, and reporting infrastructure.

### `benchmark::cache_sim` — Cache Simulation

```rust
pub enum CacheResult { Hit, ColdMiss, CapacityMiss, ConflictMiss }
pub struct CacheStats { pub hits: u64, pub misses: u64, pub cold_misses: u64, ... }
pub struct CacheSimulator { /* fully-associative LRU cache simulator */ }
pub struct DirectMappedCache { /* direct-mapped cache simulator */ }
pub struct SetAssociativeCache { /* N-way set-associative cache simulator */ }
pub struct MultiLevelCache { /* L1/L2/L3 hierarchy simulator */ }
pub struct CacheTraceRecorder { /* records address traces for replay */ }

pub fn count_cache_misses(addresses: &[u64], cache_size: usize, line_size: usize) -> u64;
pub fn sequential_stride_pattern(start: u64, count: usize, stride: u64) -> Vec<u64>;
pub fn repeated_block_pattern(blocks: &[u64], repeats: usize) -> Vec<u64>;
pub fn replay_trace(trace: &[u64], cache: &mut CacheSimulator) -> CacheStats;
pub fn optimal_cache_misses(trace: &[u64], cache_size: usize) -> u64;
pub fn stack_distance_analysis(trace: &[u64]) -> Vec<usize>;
```

### `benchmark::operation_counter`

```rust
pub struct OpCounts { pub arithmetic: u64, pub memory: u64, pub control: u64, pub total: u64 }
pub struct CategoryCounts { pub reads: u64, pub writes: u64, pub adds: u64, pub muls: u64, ... }

pub fn count_expr(expr: &Expr) -> OpCounts;
pub fn count_stmt(stmt: &Stmt) -> OpCounts;
pub fn count_program(program: &PramProgram) -> OpCounts;
pub fn count_by_category(program: &PramProgram) -> CategoryCounts;
pub fn operation_density(program: &PramProgram) -> f64;
pub fn work_distribution(program: &PramProgram) -> Vec<(String, u64)>;
```

### `benchmark::bound_verifier`

```rust
pub struct BoundResult { pub passed: bool, pub actual: u64, pub bound: u64, pub ratio: f64 }
pub struct VerificationEntry { pub algorithm: String, pub size: usize, pub result: BoundResult }
pub struct VerificationReport { pub entries: Vec<VerificationEntry> }
pub struct BoundCheckConfig { pub work_constant: f64, pub cache_constant: f64, ... }
pub struct BoundVerifier { /* checks work and cache bounds on every run */ }

pub fn verify_work_bound(actual_ops: u64, p: u64, t: u64, constant: f64) -> BoundResult;
pub fn verify_cache_bound(actual_misses: u64, p: u64, t: u64, b: u64, constant: f64) -> BoundResult;
pub fn work_satisfies_bound(actual_ops: u64, p: u64, t: u64) -> bool;
pub fn cache_satisfies_bound(actual_misses: u64, p: u64, t: u64, b: u64) -> bool;
pub fn compute_work_constant(actual_ops: u64, p: u64, t: u64) -> f64;
pub fn compute_cache_constant(actual_misses: u64, p: u64, t: u64, b: u64) -> f64;
```

### `benchmark::harness`

```rust
pub struct BenchmarkConfig { pub warmup: usize, pub trials: usize, pub sizes: Vec<u64>, ... }
pub struct BenchmarkResult { pub algorithm: String, pub size: u64, pub time_ns: u64, ... }
pub struct Timer { /* high-resolution timer */ }
pub struct BenchmarkHarness { /* orchestrates benchmark runs */ }
pub struct BenchmarkSuite { pub name: String, pub configs: Vec<BenchmarkConfig> }
pub struct TimingStats { pub mean: f64, pub median: f64, pub stddev: f64, ... }

pub type WorkloadFn = Box<dyn FnMut(u64) -> (u64, Vec<u64>)>;

pub fn time_fn<F: FnMut()>(f: F) -> u64;
pub fn time_fn_median<F: FnMut()>(f: F, warmup: usize, trials: usize) -> u64;
pub fn simple_sequential_workload(n: u64) -> (u64, Vec<u64>);
pub fn pseudo_random_workload(n: u64, range: u64) -> (u64, Vec<u64>);
pub fn run_suite<F>(suite: &BenchmarkSuite, workload: F) -> Vec<BenchmarkResult>;
pub fn warmup<F: FnMut()>(f: &mut F, iterations: usize);
pub fn statistical_benchmark<F: FnMut() -> u64>(f: F, trials: usize) -> TimingStats;
```

### `benchmark::baseline` — Hand-Optimized Sequential Baselines

```rust
pub fn baseline_sort(arr: &mut [i64]);
pub fn baseline_prefix_sum(arr: &mut [i64]);
pub fn baseline_exclusive_prefix_sum(arr: &mut [i64]);
pub fn baseline_connected_components(edges: &[(usize, usize)], n: usize) -> Vec<usize>;
pub fn baseline_list_ranking(next: &[usize], weights: &[i64]) -> Vec<i64>;
pub fn baseline_matrix_multiply(a: &[i64], b: &[i64], n: usize) -> Vec<i64>;
pub fn baseline_binary_search(arr: &[i64], target: i64) -> Option<usize>;
pub fn baseline_bfs(adj: &[Vec<usize>], start: usize, n: usize) -> Vec<i64>;
pub fn baseline_merge_sort(arr: &mut [i64]);
pub fn baseline_convex_hull(points: &[(i64, i64)]) -> Vec<(i64, i64)>;
pub fn baseline_string_match(text: &[u8], pattern: &[u8]) -> Vec<usize>;
pub fn baseline_fft(data: &mut [(f64, f64)]);
```

### `benchmark::reporter`

```rust
pub struct ReporterConfig { pub format: String, pub include_header: bool, ... }
pub struct Reporter { /* formats benchmark results for output */ }
pub struct AlgorithmSummary { pub name: String, pub mean_time: f64, pub speedup: f64, ... }

pub fn report_csv(results: &[BenchmarkResult]) -> String;
pub fn report_json(results: &[BenchmarkResult]) -> String;
pub fn report_table(results: &[BenchmarkResult]) -> String;
pub fn summarize_by_algorithm(results: &[BenchmarkResult]) -> Vec<AlgorithmSummary>;
```

### `benchmark::statistics`

```rust
pub fn mean(data: &[f64]) -> f64;
pub fn median(data: &[f64]) -> f64;
pub fn variance(data: &[f64]) -> f64;
pub fn stddev(data: &[f64]) -> f64;
pub fn geometric_mean(data: &[f64]) -> f64;
pub fn percentile(data: &[f64], p: f64) -> f64;
pub fn confidence_interval(data: &[f64], confidence_level: f64) -> (f64, f64);
pub fn detect_outliers(data: &[f64]) -> Vec<(usize, f64)>;
pub fn remove_outliers(data: &[f64]) -> Vec<f64>;
pub fn min(data: &[f64]) -> f64;
pub fn max(data: &[f64]) -> f64;

pub struct SummaryStats { pub mean: f64, pub median: f64, pub stddev: f64, pub min: f64, pub max: f64, ... }
pub struct TTestResult { pub t_statistic: f64, pub p_value: f64, pub significant: bool }

pub fn welch_t_test(a: &[f64], b: &[f64]) -> TTestResult;
pub fn mann_whitney_u(a: &[f64], b: &[f64]) -> f64;
pub fn bootstrap_ci(data: &[f64], confidence: f64, iterations: usize) -> (f64, f64);
pub fn effect_size(a: &[f64], b: &[f64]) -> f64;
```
