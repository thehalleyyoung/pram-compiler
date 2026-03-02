# PRAM Hash-Partition Compiler

Compile PRAM algorithms to cache-efficient sequential C. Supports EREW, CREW, and CRCW memory models.

## 30-Second Quickstart

```bash
cargo build --release

# Compile bitonic sort to cache-efficient C:
cargo run --release -- compile --algorithm bitonic_sort --output sort.c

# Run scalability benchmark (n up to 262,144):
cargo run --release -- scalability-benchmark --sizes 1024,16384,262144

# Verify cache-miss bounds on all 51 algorithms:
cargo run --release -- verify -a all --sizes 1024,4096,16384

# Analyze theory-practice gap in hash load distributions:
cargo run --release -- gap-analysis --sizes 1000,10000,100000
```

## What It Does

Takes a PRAM algorithm specification and produces cache-efficient sequential C code through three stages:

```
PRAM IR  →  Hash-Partition Analysis  →  Brent Scheduling  →  C99/OpenMP Code
             (Siegel k-wise hashing,    (locality ordering,   (sequential or
              adaptive partitioning)      CRCW resolution)     parallel output)
```

**Hash-Partition Locality Theorem**: Siegel k-wise independent hashing maps PRAM addresses to cache-line-aligned blocks, yielding Q ≤ c₃(pT/B + T) cache misses with c₃ ≤ 4.

## Key Results

| Metric | Value |
|--------|-------|
| Algorithms compiled | **51/51 (100%)** — 7 fixed by automated IR repair |
| Avg cache bound ratio | **0.49** (well within 2× theoretical bound) |
| Avg L1 miss rate | **0.77%** |
| Max input size tested | **262,144** |
| Tests passing | **1,497** |
| Property-test trials | **2,700+** across 6 properties |

**Honest comparison**: The compiler produces code that is cache-efficient relative to theoretical bounds, but hand-optimized baselines (introsort, Union-Find, Cooley-Tukey FFT) are significantly faster on wall-clock time. The value is *automated compilation from high-level PRAM specs* with provable cache-miss guarantees, not beating hand-tuned code.

## Commands

| Command | Description |
|---------|-------------|
| `compile --algorithm NAME` | Compile PRAM algorithm to C |
| `scalability-benchmark` | Benchmark at realistic sizes (up to 10⁶) |
| `gap-analysis` | Analyze theory-practice gap in hash loads |
| `verify -a all` | Verify work and cache bounds |
| `statistical-compare` | Welch's t-test comparison vs baselines |
| `compare --sizes SIZES` | Compare vs cache-oblivious baselines |
| `autotune` | Select optimal hash family per algorithm |
| `hardware-benchmark` | Generate hardware counter CSV data |
| `list-algorithms` | Show all 51 algorithms |

## Project Structure

```
src/
├── pram_ir/            # IR: AST, operational semantics, type system
├── hash_partition/     # 5 hash families (Siegel, 2-universal, tabulation, Murmur, identity)
├── brent_scheduler/    # Work-optimal scheduling, CRCW conflict resolution
├── staged_specializer/ # Partial evaluation, work preservation verification
├── codegen/            # C99 and OpenMP code generation
├── algorithm_library/  # 51 PRAM algorithms across 10 families
├── benchmark/          # Cache simulation, scalability, load distribution analysis
├── autotuner/          # Cache probing, parameter optimization
├── failure_analysis/   # Automated diagnosis and repair
└── parallel_codegen/   # OpenMP emission, work-stealing
```

## Limitations

- **Compilation overhead**: Hash partition compilation adds overhead vs hand-optimized baselines. The compiler targets *correctness* and *cache-efficiency guarantees*, not minimal wall-clock time.
- **SSS bound gap**: The k=8 SSS concentration bound overshoots empirical overflow by ~2× on average for structured PRAM access patterns. An adaptive k module computes tighter bounds when needed.
- **Work-Preservation**: Empirically validated via 2,700+ property-based tests across 6 properties; not mechanized in a proof assistant.
- **Cache simulation**: Benchmarks use software cache simulation (LRU, 8-way set-associative), not hardware performance counters.

## Building

Requires Rust 2021 edition (stable). No external dependencies beyond Cargo.

```bash
cargo build --release
cargo test   # 1,497 tests
```
