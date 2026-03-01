# PRAM Compiler Implementation

Rust implementation of the PRAM-to-cache-efficient-code compiler.

## Build

```bash
cargo build --release
```

## Test

```bash
cargo test --release  # 1,476 tests
```

## Quick Demo

```bash
# Compile Cole's merge sort to sequential C:
cargo run --release -- compile --algorithm cole_merge_sort --output sort.c

# Compile to parallel OpenMP C:
cargo run --release -- compile --algorithm shiloach_vishkin --target parallel --output cc.c

# Verify all 51 algorithms:
cargo run --release -- verify -a all --sizes 1024,4096

# Statistical comparison with bootstrap CIs:
cargo run --release -- statistical-compare --size 16384 --trials 10

# Run full evaluation:
cargo run --release -- run-experiments --output-dir results --sizes 256,1024,4096,16384
```

## Module Structure

- `src/pram_ir/` - PRAM intermediate representation (AST, operational semantics, type system)
- `src/hash_partition/` - Hash families (Siegel, 2-universal, tabulation, Murmur, identity)
- `src/brent_scheduler/` - Work-optimal scheduling with locality ordering and CRCW resolution
- `src/staged_specializer/` - Partial evaluation, work preservation, translation validation
- `src/codegen/` - C code generation (sequential, OpenMP, adaptive)
- `src/algorithm_library/` - 51 PRAM algorithms across 10 families
- `src/benchmark/` - Cache simulation, baseline comparison, statistics, adversarial validation
- `src/autotuner/` - Cache hierarchy detection, parameter optimization, PGO
- `src/failure_analysis/` - Failure diagnosis, automated repair, semantic preservation witnesses
- `src/parallel_codegen/` - OpenMP emission, work-stealing, NUMA
