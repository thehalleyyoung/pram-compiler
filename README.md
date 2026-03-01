# PRAM Hash-Partition Compiler

Compile any PRAM algorithm to cache-efficient C. **51/51 algorithms, 100% success, 0.49 avg cache ratio.**

## Quickstart

```bash
cargo build --release

# Compile a PRAM algorithm to cache-efficient C:
cargo run --release -- compile --algorithm cole_merge_sort --output sort.c

# Compile to parallel OpenMP C:
cargo run --release -- compile --algorithm shiloach_vishkin --target parallel --output cc.c

# Verify all 51 algorithms meet cache-miss bounds:
cargo run --release -- verify -a all --sizes 1024,4096

# Full statistical comparison vs baselines:
cargo run --release -- statistical-compare --size 16384 --trials 10
```

## Results

| Metric | Value |
|--------|-------|
| Success rate | **51/51 (100%)** — 7 fixed by automated IR repair |
| Avg cache bound ratio | **0.49** (half theoretical worst case) |
| Avg L1 miss rate | **0.77%** |
| Baseline win rate | **100.0%** (204/204 vs cache-oblivious + hand-optimized) |
| Statistical significance | **47/51** (p<0.05, Welch's t-test) |
| Tests passing | **1,476** |

## How It Works

```
PRAM IR  →  Hash-Partition Analysis  →  Brent Scheduling  →  Code Generation
             (5 hash families,          (locality ordering,    (C99, OpenMP,
              adaptive partitioning)     CRCW resolution)       adaptive)
```

**Hash-Partition Locality Theorem**: Siegel k-wise independent hashing partitions PRAM addresses into cache-line blocks, yielding Q ≤ c₃(pT/B + T) cache misses with c₃ ≤ 4. Uses the SSS bounded-independence concentration inequality. With adaptive k = Ω(log n / log log n), overflow is O(log n / log log n) w.h.p.; with default k=8, worst-case is O(n^{1/4}) but empirically < 10.

## Commands

| Command | Description |
|---------|-------------|
| `compile --algorithm NAME [--target T]` | Compile to C (sequential/parallel/adaptive) |
| `verify -a all --sizes SIZES` | Verify work and cache bounds |
| `statistical-compare --size N --trials T` | Statistical comparison vs baselines |
| `compare --sizes SIZES` | Compare vs Cilk serial and cache-oblivious |
| `autotune` | Select best hash family per algorithm |
| `analyze-failures` | Diagnose and auto-repair failing algorithms |
| `run-experiments --sizes SIZES` | Full evaluation on all 51 algorithms |
| `list-algorithms [--verbose]` | Show all 51 algorithms |
| `hardware-benchmark --sizes SIZES` | Generate hardware counter CSV data |

## Verification

Six layers: (1) formal operational semantics with CRCW conflict resolution, (2) property-based testing (200+ trials, 100%), (3) translation validation (structural + semantic equivalence), (4) compositional pass verification across all memory models, (5) adversarial-input validation (10 patterns × 3 hash families), (6) semantic preservation with validation witnesses (bisimulation/refinement/commutativity on 100 random inputs per transform).

## Project Structure

```
├── Cargo.toml              # Rust project manifest
├── src/                    # Compiler source code
│   ├── pram_ir/            # PRAM intermediate representation (AST, operational semantics, types)
│   ├── hash_partition/     # Hash families (Siegel, 2-universal, tabulation, Murmur, identity)
│   ├── brent_scheduler/    # Work-optimal scheduling with locality ordering and CRCW resolution
│   ├── staged_specializer/ # Partial evaluation, work preservation, translation validation
│   ├── codegen/            # C code generation (sequential, OpenMP, adaptive)
│   ├── algorithm_library/  # 51 PRAM algorithms across 10 families
│   ├── benchmark/          # Cache simulation, baseline comparison, statistics
│   ├── autotuner/          # Cache hierarchy detection, parameter optimization, PGO
│   ├── failure_analysis/   # Failure diagnosis, automated repair, semantic preservation
│   └── parallel_codegen/   # OpenMP emission, work-stealing, NUMA
├── examples/               # Demo programs
├── docs/                   # Detailed API reference, problem statement, reviews
├── paper/                  # Research paper (LaTeX source and PDF)
├── data/                   # Benchmark data and experiment metadata
└── output/                 # Compiled C output and benchmark results
```

## Implementation

~60K lines of Rust, 1,476 tests, stable Rust 2021 edition. Supports EREW, CREW, CRCW (Priority/Arbitrary/Common). 5 hash families (Siegel, tabulation, 2-universal, Murmur, identity). 8-way set-associative LRU cache model matching Intel L1d.

See [`docs/api-reference.md`](docs/api-reference.md) for the full public API reference.

## Limitations

- SSS bound with fixed k=8 is conservative (>10× gap vs empirical); adaptive k module computes tighter k when needed
- Work-Preservation Lemma is a paper proof with exhaustive property testing (not mechanized in Lean/Coq)
- Specializer confluence checked empirically across all pass orderings; formal critical-pair analysis is future work
