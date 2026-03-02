# API Reference

## CLI Interface

All commands run via `cargo run --release --` from the project root.

### `compile`
Compile a PRAM algorithm to C code.

```bash
cargo run --release -- compile --algorithm <NAME> [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--algorithm` | required | Algorithm name (use `list-algorithms` to see all) |
| `--output` | `output.c` | Output file path |
| `--target` | `sequential` | `sequential`, `parallel`, or `adaptive` |
| `--hash-family` | `siegel` | `siegel`, `two-universal`, `murmur`, `tabulation`, `identity` |
| `--cache-line-size` | `64` | Cache line size in bytes |
| `--opt-level` | `2` | Optimization level (0-3) |
| `--instrument` | false | Include timing instrumentation |

### `list-algorithms`
```bash
cargo run --release -- list-algorithms [--verbose]
```

### `verify`
Verify work and cache-miss bounds via simulation.
```bash
cargo run --release -- verify --algorithm <NAME|all> --sizes <SIZES>
```

### `rayon-baseline`
Compare hash-partition approach against real rayon parallel baselines.
```bash
cargo run --release -- rayon-baseline --output-dir <DIR> --sizes <SIZES> --trials <N>
```

### `large-scale-eval`
Run large-scale evaluation with L1/L2 cache simulation and rayon comparison.
```bash
cargo run --release -- large-scale-eval --output-dir <DIR> --sizes <SIZES> --trials <N>
```

### `scalability-benchmark`
Run scalability evaluation at realistic input sizes.
```bash
cargo run --release -- scalability-benchmark --output-dir <DIR> --sizes <SIZES> --trials <N>
```

### `gap-analysis`
Analyze theory-practice gap in hash load distributions.
```bash
cargo run --release -- gap-analysis --output-dir <DIR> --sizes <SIZES> --k <K>
```

### `run-experiments`
Full evaluation across all 51 algorithms.
```bash
cargo run --release -- run-experiments --output-dir <DIR> --sizes <SIZES>
```

### `compare`
Compare hash-partition against cache-oblivious baselines (cache simulation).
```bash
cargo run --release -- compare --sizes <SIZES> --output <FILE>
```

### `statistical-compare`
Welch's t-test comparing hash-partition vs baselines.
```bash
cargo run --release -- statistical-compare --size <N> --trials <T> --output <FILE>
```

### `autotune`
Select optimal hash family per algorithm.
```bash
cargo run --release -- autotune --output <FILE>
```

### `analyze-failures`
Diagnose and auto-repair failing algorithms via IR transformations.
```bash
cargo run --release -- analyze-failures --output <FILE>
```

### `hardware-benchmark`
Generate simulated cache counter measurements with CSV data.
```bash
cargo run --release -- hardware-benchmark --output-dir <DIR> --sizes <SIZES>
```

### `benchmark`
Run performance benchmarks (compilation time).
```bash
cargo run --release -- benchmark --algorithm <NAME> --sizes <SIZES> --trials <N> --format <FMT>
```

## Rust Library API

### Core Types

```rust
use pram_compiler::pram_ir::ast::{PramProgram, MemoryModel, Stmt, Expr};
use pram_compiler::codegen::adaptive::{AdaptiveCompiler, CompilationTarget};

enum MemoryModel { EREW, CREW, CRCWPriority, CRCWArbitrary, CRCWCommon }
// Note: CRCWArbitrary is implemented as deterministic lowest-index-writer
// resolution (equivalent to Priority). This is sound for cache-miss analysis.

enum CompilationTarget {
    Sequential,
    Parallel { num_threads: usize },
    Adaptive { crossover_n: usize },
}
```

### Compilation

```rust
use pram_compiler::codegen::generator::{CodeGenerator, GeneratorConfig};
use pram_compiler::codegen::adaptive::{AdaptiveCompiler, CompilationTarget};

let gen = CodeGenerator::new(GeneratorConfig::default());
let c_code = gen.generate(&program);

let compiler = AdaptiveCompiler::new();
let code = compiler.compile(&program, &CompilationTarget::Sequential);
```

### Hash Partition Engine

```rust
use pram_compiler::hash_partition::partition_engine::{PartitionEngine, HashFamilyChoice};

let engine = PartitionEngine::new(num_blocks, block_size, HashFamilyChoice::Siegel { k: 8 }, seed);
let result = engine.partition(&addresses);
```

### Theorem Regime Analysis

```rust
use pram_compiler::hash_partition::theorem_regime::{
    analyze_theorem_regime, analyze_theorem_applicability, TheoremRegime,
};

let result = analyze_theorem_regime(65536, 512, 64, 8);
// result.regime: CacheResident, Transitional, or CapacityDominated
// result.bound_holds: whether c₃ ≤ 4 bound is satisfied
```

### Rayon Baseline Comparison

```rust
use pram_compiler::benchmark::rayon_baselines::{
    run_rayon_baseline_evaluation, rayon_baselines_to_csv,
};

let summary = run_rayon_baseline_evaluation(&[1024, 65536, 262144], 5);
println!("Geometric mean HP/Rayon ratio: {:.2}x", summary.geometric_mean_ratio);
```

### Large-Scale Evaluation

```rust
use pram_compiler::benchmark::large_scale::{
    run_large_scale_evaluation, large_scale_to_csv,
};

let summary = run_large_scale_evaluation(&[1024, 65536, 1048576], 3);
println!("Avg cache bound ratio: {:.4}", summary.avg_cache_bound_ratio);
```

### Algorithm Library

```rust
use pram_compiler::algorithm_library;

let entries = algorithm_library::catalog(); // 51 algorithms
let prog = algorithm_library::sorting::bitonic_sort();
let prog = algorithm_library::graph::shiloach_vishkin();
```

### Work Preservation Verification

```rust
use pram_compiler::staged_specializer::work_preservation::{WorkCounter, WorkBoundChecker};

let pre_work = WorkCounter::count(&original_body);
let post_work = WorkCounter::count(&transformed_body);
let checker = WorkBoundChecker::new();
let result = checker.check(&pre_work, &post_work);
```

### Property-Based Testing

```rust
use pram_compiler::staged_specializer::property_tests::run_all_property_tests;

let results = run_all_property_tests(); // 2,700+ trials across 6 properties
for r in &results { assert!(r.all_passed()); }
```

### Cache Simulation

```rust
use pram_compiler::benchmark::cache_sim::{CacheSimulator, SetAssociativeCache};

let mut sim = CacheSimulator::new(64, 512); // 64B lines, 512 lines
sim.access_sequence(&trace);
let stats = sim.stats();
println!("Miss rate: {:.4}", stats.miss_rate());
```
