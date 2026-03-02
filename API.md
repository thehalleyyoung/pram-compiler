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

### `run-experiments`
Full evaluation across all 51 algorithms.
```bash
cargo run --release -- run-experiments --output-dir <DIR> --sizes <SIZES>
```

### `compare`
Compare hash-partition against baselines (cache simulation).
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
Generate hardware counter measurements with CSV data.
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
let code = compiler.compile(&program, &CompilationTarget::Adaptive { crossover_n: 10000 });
```

### Hash Partition Engine

```rust
use pram_compiler::hash_partition::partition_engine::{PartitionEngine, HashFamilyChoice};

let engine = PartitionEngine::new(num_blocks, block_size, HashFamilyChoice::Siegel { k: 8 }, seed);
let result = engine.partition(&addresses);
let result = engine.adaptive_partition(&addresses, &[64, 128, 256]);
```

### Failure Analysis and Repair

```rust
use pram_compiler::failure_analysis::analyzer::FailureAnalyzer;
use pram_compiler::failure_analysis::fixer::apply_fixes;

let analyzer = FailureAnalyzer::new();
let analysis = analyzer.analyze(&program);
let fix_result = apply_fixes(&mut program, &analysis);
```

### Profile-Guided Compilation

```rust
use pram_compiler::codegen::adaptive::ProfileGuidedCompiler;

let pgc = ProfileGuidedCompiler::new();
let result = pgc.compile_with_profiling(&program, &CompilationTarget::Adaptive { crossover_n: 10000 });
```

### Translation Validation

```rust
use pram_compiler::staged_specializer::translation_validation::TranslationValidator;
use pram_compiler::staged_specializer::work_preservation::WorkCounter;

let validator = TranslationValidator::new();
let pre_work = WorkCounter::count(&original_body);
let post_work = WorkCounter::count(&transformed_body);

// Structural validation
let result = validator.validate_structural(&original_body, &transformed_body, &pre_work, &post_work);

// Full validation: structural + semantic equivalence on random inputs
let result = validator.validate_full(
    &original_body, &transformed_body, &pre_work, &post_work,
    MemoryModel::EREW, &[("A".to_string(), 64)], 20,
);

// Confluence check (pass order independence)
let confluence = validator.validate_confluence(&body, MemoryModel::EREW);
```

### Semantic Preservation Verification

```rust
use pram_compiler::failure_analysis::semantic_preservation::{
    verify_semantic_preservation, ValidatedTransform, PreservationWitness,
};

let result = verify_semantic_preservation(
    &original, &transformed, MemoryModel::EREW, &[("A".into(), 64)], 100,
);

let mut vt = ValidatedTransform::new(PreservationWitness::write_coalescing());
let passed = vt.apply_and_verify(
    &original, &transformed, MemoryModel::CRCWPriority, &[("A".into(), 64)],
);
```

### Compositional Pass Verification

```rust
use pram_compiler::staged_specializer::compositional_verification::verify_pass_composition;

let result = verify_pass_composition(
    &original_body, MemoryModel::EREW, &[("A".to_string(), 8)], 10,
);
assert!(result.all_preserved);
```

### Realistic Cache Simulation

```rust
use pram_compiler::benchmark::cache_sim::{
    RealisticCacheConfig, count_cache_misses_realistic, compare_cache_models,
};

let config = RealisticCacheConfig::default(); // 8-way, 64 sets, 64B lines
let misses = count_cache_misses_realistic(&addresses, &config);
let cmp = compare_cache_models(&addresses, &config);
```

### Adversarial-Input Validation

```rust
use pram_compiler::benchmark::adversarial::{run_adversarial_validation, summarize_adversarial};

let results = run_adversarial_validation(4096, 8);
let summary = summarize_adversarial(&results);
```

### Operational Semantics

```rust
use pram_compiler::pram_ir::operational_semantics::{Store, exec_stmt, eval_to_value};

let mut store = Store::new();
store.alloc_shared("A", 1024);
exec_stmt(&stmt, &mut store, Some(0), Some(4), MemoryModel::CRCWPriority).unwrap();
```

### Algorithm Library

```rust
use pram_compiler::algorithm_library;

let entries = algorithm_library::catalog(); // 51 algorithms
let prog = algorithm_library::sorting::bitonic_sort();
let prog = algorithm_library::graph::shiloach_vishkin();
```

### Parallel Batch Compilation

```rust
use pram_compiler::codegen::parallel_batch::{compile_batch, compile_multi_target};

let results = compile_batch(&programs, &CompilationTarget::Sequential);
let results = compile_multi_target(&program, &targets);
```

### Hardware Counter Benchmarks

```rust
use pram_compiler::benchmark::hardware_counters::{measure_hardware_counters, counters_to_csv};

let counters = measure_hardware_counters(&programs, &sizes, l1_size, l1_line, l1_assoc, l2_size, l2_line, l2_assoc);
let csv = counters_to_csv(&counters);
```

### PGO Distributional Analysis

```rust
use pram_compiler::autotuner::distributional_analysis::{
    analyze_pgo_sensitivity, analyze_crossover_sensitivity, find_algorithm_crossover,
};

let analysis = analyze_pgo_sensitivity(&hierarchy, 1024);
let sensitivity = analyze_crossover_sensitivity(&hierarchy, 1024);
let crossover = find_algorithm_crossover("bitonic_sort", &hp_misses, &co_misses);
```

### SSS Bounded-Independence Bounds

```rust
use pram_compiler::hash_partition::overflow_analysis::{sss_failure_probability, tail_bounds_with_independence};

let prob = sss_failure_probability(10000, 1250, 8, 30.0);
let tb = tail_bounds_with_independence(8.0, 8.0, 20.0, Some(8), 10000, 1250);
```

### Property-Based Testing

```rust
use pram_compiler::staged_specializer::property_tests::run_all_property_tests;

let results = run_all_property_tests(); // 2,700+ trials across 6 properties
for r in &results { assert!(r.all_passed()); }
```

### Scalability Benchmarks

```rust
use pram_compiler::benchmark::scalability::{
    run_scalability_evaluation, benchmark_sort_comparison, scalability_to_csv,
};

let summary = run_scalability_evaluation(&[1024, 16384, 262144], 5);
let csv = scalability_to_csv(&summary);

// Compare hash-partition vs best-available baselines
let sort_cmp = benchmark_sort_comparison(65536, 5);
println!("Speedup vs introsort: {:.2}x", sort_cmp.speedup);
```

### Theory-Practice Gap Analysis

```rust
use pram_compiler::benchmark::load_distribution::{
    analyze_theory_practice_gap, analyze_load_distribution, AccessPatternType,
};

let report = analyze_theory_practice_gap(&[1000, 10000, 100000], 8);
println!("Gap explanation: {}", report.gap_explanation);
println!("Structural regularity factor: {:.2}", report.structural_regularity_factor);

// Per-pattern analysis
let result = analyze_load_distribution(AccessPatternType::Sequential, 10000, 1250, 8);
println!("SSS bound: {:.1}, Empirical max: {}", result.sss_theoretical_bound, result.empirical_max_load);
```
