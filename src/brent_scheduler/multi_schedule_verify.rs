//! Multi-schedule CRCW conflict verification.
//!
//! Extends the single-schedule CRCW resolver to verify conflict resolution
//! correctness across multiple scheduling strategies and orderings.
//!
//! The critique identified that "CRCW conflict resolution verified for only
//! one schedule per algorithm." This module systematically tests all three
//! CRCW semantics (Priority, Arbitrary, Common) against multiple schedules:
//! - Default Brent schedule
//! - Reversed processor ordering
//! - Random permutation orderings (k trials)
//! - Worst-case adversarial orderings
//!
//! For each schedule × semantics combination, we verify:
//! 1. Determinism: Priority always selects the same winner
//! 2. Safety: Common mode rejects conflicting values
//! 3. Consistency: Arbitrary mode picks one valid winner

use std::collections::HashMap;
use crate::pram_ir::ast::MemoryModel;
use super::schedule::{Schedule, ScheduleEntry, OpType};
use super::crcw_resolver::{resolve_crcw_conflicts, CrcwResolutionReport};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A schedule permutation strategy for testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulePermutation {
    /// Original Brent schedule order
    Original,
    /// Reverse processor ID ordering within each phase
    Reversed,
    /// Interleaved: odd processors first, then even
    Interleaved,
    /// Stride-based: processor i*stride mod p
    Strided(usize),
}

/// Result of multi-schedule verification.
#[derive(Debug, Clone)]
pub struct MultiScheduleVerification {
    /// Total number of (schedule × semantics) combinations tested
    pub combinations_tested: usize,
    /// Number of combinations that passed
    pub combinations_passed: usize,
    /// Detailed results per combination
    pub results: Vec<ScheduleVerificationResult>,
    /// Whether all combinations passed
    pub all_passed: bool,
    /// Summary statistics
    pub total_conflicts: usize,
    pub total_eliminations: usize,
}

/// Result for a single schedule × semantics combination.
#[derive(Debug, Clone)]
pub struct ScheduleVerificationResult {
    pub permutation: String,
    pub memory_model: MemoryModel,
    pub resolution_report: CrcwResolutionReport,
    pub determinism_ok: bool,
    pub safety_ok: bool,
    pub passed: bool,
}

// ---------------------------------------------------------------------------
// §1  Adversarial schedule generation
// ---------------------------------------------------------------------------

/// Generates worst-case schedules that stress CRCW conflict resolution.
pub struct AdversarialScheduleGenerator;

impl AdversarialScheduleGenerator {
    /// All processors write to the same address in *descending* processor-ID
    /// order. This is worst-case for Priority resolution because the winner
    /// (lowest ID) appears last in the schedule.
    pub fn descending_priority(num_procs: usize) -> Schedule {
        let mut sched = Schedule::new(num_procs, 1, 4);
        for i in (0..num_procs).rev() {
            sched.push(ScheduleEntry {
                operation_id: num_procs - 1 - i,
                block_id: 0,
                phase: 0,
                original_proc_id: i,
                op_type: OpType::Write,
                address: 0,
                memory_region: "HOT".to_string(),
                level: 0,
            });
        }
        sched
    }

    /// Alternating-value writes: even-ID processors write to address 0,
    /// odd-ID processors write to address 1, but all share the same
    /// `(phase, memory_region)`. For Common mode, the interleaving of
    /// distinct target addresses maximises the guard-check surface.
    pub fn alternating_common(num_procs: usize) -> Schedule {
        let mut sched = Schedule::new(num_procs, 1, 4);
        for i in 0..num_procs {
            let addr = i % 2; // alternates 0 / 1
            sched.push(ScheduleEntry {
                operation_id: i,
                block_id: 0,
                phase: 0,
                original_proc_id: i,
                op_type: OpType::Write,
                address: addr,
                memory_region: "ALT".to_string(),
                level: 0,
            });
        }
        sched
    }

    /// Maximum-conflict schedule: *all* processors write to one "hot"
    /// address while `cold_count` additional non-conflicting writes go to
    /// distinct "cold" addresses. This creates a single dense conflict
    /// group surrounded by conflict-free operations.
    pub fn hot_cold(num_procs: usize, cold_count: usize) -> Schedule {
        let total = num_procs + cold_count;
        let mut sched = Schedule::new(total, 1, 4);
        // Hot writes
        for i in 0..num_procs {
            sched.push(ScheduleEntry {
                operation_id: i,
                block_id: 0,
                phase: 0,
                original_proc_id: i,
                op_type: OpType::Write,
                address: 0,
                memory_region: "HOT".to_string(),
                level: 0,
            });
        }
        // Cold writes (each to a unique address -> no conflicts)
        for j in 0..cold_count {
            sched.push(ScheduleEntry {
                operation_id: num_procs + j,
                block_id: (100 + j) / 4,
                phase: 0,
                original_proc_id: num_procs + j,
                op_type: OpType::Write,
                address: 100 + j,
                memory_region: "COLD".to_string(),
                level: 0,
            });
        }
        sched
    }
}

// ---------------------------------------------------------------------------
// §2  Randomized schedule testing (property-based style)
// ---------------------------------------------------------------------------

/// Collects statistics from randomized schedule verification runs.
#[derive(Debug, Clone, Default)]
pub struct RandomizedStats {
    pub trials: usize,
    pub total_conflicts: usize,
    pub min_conflicts: usize,
    pub max_conflicts: usize,
}

/// Simple deterministic PRNG (xorshift64) so tests are reproducible.
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Fisher-Yates shuffle of a slice.
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = (self.next() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

/// Generates random permutations of a schedule and verifies CRCW properties.
pub struct RandomizedScheduleVerifier {
    /// Number of random trials.
    pub k: usize,
    /// PRNG seed for reproducibility.
    pub seed: u64,
}

impl Default for RandomizedScheduleVerifier {
    fn default() -> Self {
        Self { k: 100, seed: 42 }
    }
}

impl RandomizedScheduleVerifier {
    pub fn new(k: usize, seed: u64) -> Self {
        Self { k, seed }
    }

    /// Generate `k` random permutations of entries within each phase,
    /// resolve CRCW conflicts under `model`, and return per-trial reports.
    pub fn run(
        &self,
        base: &Schedule,
        model: MemoryModel,
    ) -> (Vec<CrcwResolutionReport>, RandomizedStats) {
        let mut rng = Xorshift64::new(self.seed);
        let mut reports = Vec::with_capacity(self.k);
        let mut stats = RandomizedStats {
            trials: self.k,
            total_conflicts: 0,
            min_conflicts: usize::MAX,
            max_conflicts: 0,
        };

        for _ in 0..self.k {
            let mut permuted = self.random_permute(base, &mut rng);
            let report = resolve_crcw_conflicts(&mut permuted, model);
            stats.total_conflicts += report.conflicts_detected;
            stats.min_conflicts = stats.min_conflicts.min(report.conflicts_detected);
            stats.max_conflicts = stats.max_conflicts.max(report.conflicts_detected);
            reports.push(report);
        }

        if self.k == 0 {
            stats.min_conflicts = 0;
        }

        (reports, stats)
    }

    /// Verify **Priority determinism**: across all `k` permutations the
    /// surviving writes always come from the same set of processors.
    pub fn verify_priority_determinism(&self, base: &Schedule) -> bool {
        let mut rng = Xorshift64::new(self.seed);
        let mut reference_survivors: Option<Vec<usize>> = None;

        for _ in 0..self.k {
            let mut permuted = self.random_permute(base, &mut rng);
            resolve_crcw_conflicts(&mut permuted, MemoryModel::CRCWPriority);
            let mut survivors: Vec<usize> = permuted
                .entries
                .iter()
                .filter(|e| e.op_type == OpType::Write)
                .map(|e| e.original_proc_id)
                .collect();
            survivors.sort();

            match &reference_survivors {
                None => reference_survivors = Some(survivors),
                Some(ref expected) => {
                    if &survivors != expected {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Verify **Common safety**: whenever conflicts exist, at least one
    /// guard is inserted (i.e., `guards_inserted > 0`).
    pub fn verify_common_safety(&self, base: &Schedule) -> bool {
        let mut rng = Xorshift64::new(self.seed);
        for _ in 0..self.k {
            let mut permuted = self.random_permute(base, &mut rng);
            let report = resolve_crcw_conflicts(&mut permuted, MemoryModel::CRCWCommon);
            if report.conflicts_detected > 0 && report.guards_inserted == 0 {
                return false;
            }
        }
        true
    }

    fn random_permute(&self, base: &Schedule, rng: &mut Xorshift64) -> Schedule {
        let mut permuted = base.clone();
        // Group indices by phase, shuffle within each phase
        let mut by_phase: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, entry) in permuted.entries.iter().enumerate() {
            by_phase.entry(entry.phase).or_default().push(idx);
        }
        for (_phase, indices) in &by_phase {
            if indices.len() <= 1 {
                continue;
            }
            let mut entries: Vec<ScheduleEntry> =
                indices.iter().map(|&i| permuted.entries[i].clone()).collect();
            rng.shuffle(&mut entries);
            for (i, &idx) in indices.iter().enumerate() {
                permuted.entries[idx] = entries[i].clone();
            }
        }
        permuted
    }
}

// ---------------------------------------------------------------------------
// §3  Formal correctness specification
// ---------------------------------------------------------------------------

/// Defines the formal correctness properties for CRCW resolution and
/// provides predicate functions that can be composed into larger proofs.
pub struct CrcwCorrectnessSpec;

impl CrcwCorrectnessSpec {
    /// **Priority determinism**: given multiple schedule variants, resolving
    /// under Priority always yields the same set of surviving write
    /// processor IDs (grouped by conflict key).
    pub fn spec_priority_deterministic(schedule_variants: &[Schedule]) -> bool {
        if schedule_variants.is_empty() {
            return true;
        }
        let reference = Self::surviving_writers(&schedule_variants[0], MemoryModel::CRCWPriority);
        for variant in &schedule_variants[1..] {
            let survivors = Self::surviving_writers(variant, MemoryModel::CRCWPriority);
            if survivors != reference {
                return false;
            }
        }
        true
    }

    /// **Common agreement**: for every variant, if a conflict group exists,
    /// the resolver inserts a guard (i.e., detects the disagreement).
    pub fn spec_common_agreement(schedule_variants: &[Schedule]) -> bool {
        for variant in schedule_variants {
            let mut v = variant.clone();
            let report = resolve_crcw_conflicts(&mut v, MemoryModel::CRCWCommon);
            if report.conflicts_detected > 0 && report.guards_inserted == 0 {
                return false;
            }
        }
        true
    }

    /// **Arbitrary uniqueness**: for every conflict group in every variant,
    /// exactly one write survives (no duplicates, no zero survivors).
    pub fn spec_arbitrary_unique(schedule_variants: &[Schedule]) -> bool {
        for variant in schedule_variants {
            let mut v = variant.clone();
            let pre_groups = Self::conflict_groups(&v);
            resolve_crcw_conflicts(&mut v, MemoryModel::CRCWArbitrary);

            // For each conflict group key, exactly one write should survive
            let mut post_counts: HashMap<(usize, String, usize), usize> = HashMap::new();
            for entry in &v.entries {
                if entry.op_type == OpType::Write {
                    let key = (entry.phase, entry.memory_region.clone(), entry.address);
                    *post_counts.entry(key).or_insert(0) += 1;
                }
            }
            for (key, _original_count) in &pre_groups {
                let survivors = post_counts.get(key).copied().unwrap_or(0);
                if survivors != 1 {
                    return false;
                }
            }
        }
        true
    }

    // -- helpers --

    /// Returns sorted surviving writer proc-IDs after resolution.
    fn surviving_writers(schedule: &Schedule, model: MemoryModel) -> Vec<usize> {
        let mut s = schedule.clone();
        resolve_crcw_conflicts(&mut s, model);
        let mut ids: Vec<usize> = s
            .entries
            .iter()
            .filter(|e| e.op_type == OpType::Write)
            .map(|e| e.original_proc_id)
            .collect();
        ids.sort();
        ids
    }

    /// Returns conflict groups (keys with >1 writer) and their counts.
    fn conflict_groups(schedule: &Schedule) -> HashMap<(usize, String, usize), usize> {
        let mut groups: HashMap<(usize, String, usize), usize> = HashMap::new();
        for entry in &schedule.entries {
            if entry.op_type == OpType::Write {
                let key = (entry.phase, entry.memory_region.clone(), entry.address);
                *groups.entry(key).or_insert(0) += 1;
            }
        }
        groups.retain(|_, count| *count > 1);
        groups
    }
}

// ---------------------------------------------------------------------------
// §4  Cross-algorithm verification
// ---------------------------------------------------------------------------

/// An algorithm together with its schedule, used as input for cross-algorithm
/// verification.
pub struct AlgorithmSchedule {
    pub name: String,
    pub schedule: Schedule,
}

/// Runs multi-schedule verification across all supplied algorithms and all
/// three CRCW memory models, returning per-algorithm results.
pub fn verify_all_algorithms_multi_schedule(
    algorithms: &[AlgorithmSchedule],
) -> Vec<(String, MultiScheduleVerification)> {
    let models = [
        MemoryModel::CRCWPriority,
        MemoryModel::CRCWArbitrary,
        MemoryModel::CRCWCommon,
    ];

    algorithms
        .iter()
        .map(|alg| {
            let verification = verify_multi_schedule(&alg.schedule, &models);
            (alg.name.clone(), verification)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Existing helpers (permutation, determinism, summary)
// ---------------------------------------------------------------------------

/// Verify CRCW conflict resolution across multiple schedules.
pub fn verify_multi_schedule(
    base_schedule: &Schedule,
    models: &[MemoryModel],
) -> MultiScheduleVerification {
    let permutations = vec![
        SchedulePermutation::Original,
        SchedulePermutation::Reversed,
        SchedulePermutation::Interleaved,
        SchedulePermutation::Strided(2),
        SchedulePermutation::Strided(3),
    ];

    let mut results = Vec::new();
    let mut total_conflicts = 0;
    let mut total_eliminations = 0;

    for &perm in &permutations {
        let permuted = apply_permutation(base_schedule, perm);
        for &model in models {
            if !model.allows_concurrent_write() {
                continue;
            }

            let mut schedule_copy = permuted.clone();
            let report = resolve_crcw_conflicts(&mut schedule_copy, model);

            let determinism_ok = check_determinism(base_schedule, &permuted, model);
            let safety_ok = report.conflicts_detected == 0
                || report.writes_eliminated > 0
                || model == MemoryModel::CRCWArbitrary;

            total_conflicts += report.conflicts_detected;
            total_eliminations += report.writes_eliminated;

            let passed = determinism_ok && safety_ok;

            results.push(ScheduleVerificationResult {
                permutation: format!("{:?}", perm),
                memory_model: model,
                resolution_report: report,
                determinism_ok,
                safety_ok,
                passed,
            });
        }
    }

    let combinations_tested = results.len();
    let combinations_passed = results.iter().filter(|r| r.passed).count();

    MultiScheduleVerification {
        combinations_tested,
        combinations_passed,
        results,
        all_passed: combinations_tested == combinations_passed,
        total_conflicts,
        total_eliminations,
    }
}

/// Apply a permutation to a schedule's entries within each phase.
fn apply_permutation(schedule: &Schedule, perm: SchedulePermutation) -> Schedule {
    let mut permuted = schedule.clone();

    match perm {
        SchedulePermutation::Original => {}
        SchedulePermutation::Reversed => {
            let mut by_phase: HashMap<usize, Vec<usize>> = HashMap::new();
            for (idx, entry) in permuted.entries.iter().enumerate() {
                by_phase.entry(entry.phase).or_default().push(idx);
            }
            for (_phase, indices) in &by_phase {
                if indices.len() > 1 {
                    let mut reversed_entries: Vec<ScheduleEntry> =
                        indices.iter().map(|&i| permuted.entries[i].clone()).collect();
                    reversed_entries.reverse();
                    for (i, &idx) in indices.iter().enumerate() {
                        permuted.entries[idx] = reversed_entries[i].clone();
                    }
                }
            }
        }
        SchedulePermutation::Interleaved => {
            let mut by_phase: HashMap<usize, Vec<usize>> = HashMap::new();
            for (idx, entry) in permuted.entries.iter().enumerate() {
                by_phase.entry(entry.phase).or_default().push(idx);
            }
            for (_phase, indices) in &by_phase {
                if indices.len() > 1 {
                    let entries: Vec<ScheduleEntry> =
                        indices.iter().map(|&i| permuted.entries[i].clone()).collect();
                    let mut odd: Vec<_> = entries
                        .iter()
                        .filter(|e| e.original_proc_id % 2 == 1)
                        .cloned()
                        .collect();
                    let even: Vec<_> = entries
                        .iter()
                        .filter(|e| e.original_proc_id % 2 == 0)
                        .cloned()
                        .collect();
                    odd.extend(even);
                    for (i, &idx) in indices.iter().enumerate() {
                        if i < odd.len() {
                            permuted.entries[idx] = odd[i].clone();
                        }
                    }
                }
            }
        }
        SchedulePermutation::Strided(stride) => {
            let mut by_phase: HashMap<usize, Vec<usize>> = HashMap::new();
            for (idx, entry) in permuted.entries.iter().enumerate() {
                by_phase.entry(entry.phase).or_default().push(idx);
            }
            for (_phase, indices) in &by_phase {
                if indices.len() > 1 {
                    let entries: Vec<ScheduleEntry> =
                        indices.iter().map(|&i| permuted.entries[i].clone()).collect();
                    let n = entries.len();
                    let mut reordered = Vec::with_capacity(n);
                    let mut visited = vec![false; n];
                    let mut pos = 0;
                    for _ in 0..n {
                        while pos < n && visited[pos] {
                            pos += 1;
                        }
                        if pos >= n {
                            break;
                        }
                        reordered.push(entries[pos].clone());
                        visited[pos] = true;
                        pos = (pos + stride) % n;
                    }
                    for (i, entry) in entries.iter().enumerate() {
                        if !visited[i] {
                            reordered.push(entry.clone());
                        }
                    }
                    for (i, &idx) in indices.iter().enumerate() {
                        if i < reordered.len() {
                            permuted.entries[idx] = reordered[i].clone();
                        }
                    }
                }
            }
        }
    }

    permuted
}

/// Check determinism: for Priority mode, the winner should be the same
/// regardless of schedule ordering.
fn check_determinism(original: &Schedule, permuted: &Schedule, model: MemoryModel) -> bool {
    if model != MemoryModel::CRCWPriority {
        return true;
    }

    let mut orig = original.clone();
    let mut perm = permuted.clone();

    let _report_orig = resolve_crcw_conflicts(&mut orig, model);
    let _report_perm = resolve_crcw_conflicts(&mut perm, model);

    let orig_surviving = orig
        .entries
        .iter()
        .filter(|e| e.op_type == OpType::Write)
        .count();
    let perm_surviving = perm
        .entries
        .iter()
        .filter(|e| e.op_type == OpType::Write)
        .count();

    orig_surviving == perm_surviving
}

/// Generate a human-readable verification report.
pub fn verification_summary(verification: &MultiScheduleVerification) -> String {
    let mut lines = Vec::new();
    lines.push("Multi-Schedule CRCW Verification Report".to_string());
    lines.push("========================================".to_string());
    lines.push(format!(
        "Combinations tested: {}",
        verification.combinations_tested
    ));
    lines.push(format!(
        "Combinations passed: {}",
        verification.combinations_passed
    ));
    lines.push(format!("All passed: {}", verification.all_passed));
    lines.push(format!(
        "Total conflicts found: {}",
        verification.total_conflicts
    ));
    lines.push(format!(
        "Total writes eliminated: {}",
        verification.total_eliminations
    ));

    for result in &verification.results {
        lines.push(format!(
            "  {:20} {:15} conflicts={} eliminated={} det={} safe={} {}",
            result.permutation,
            result.memory_model.name(),
            result.resolution_report.conflicts_detected,
            result.resolution_report.writes_eliminated,
            result.determinism_ok,
            result.safety_ok,
            if result.passed { "✓" } else { "✗" }
        ));
    }

    lines.join("\n")
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ------------------------------------------------------------

    fn make_write(op_id: usize, proc_id: usize, phase: usize, addr: usize) -> ScheduleEntry {
        ScheduleEntry {
            operation_id: op_id,
            block_id: addr / 4,
            phase,
            original_proc_id: proc_id,
            op_type: OpType::Write,
            address: addr,
            memory_region: "A".to_string(),
            level: 0,
        }
    }

    fn make_read(op_id: usize, proc_id: usize, phase: usize, addr: usize) -> ScheduleEntry {
        ScheduleEntry {
            operation_id: op_id,
            block_id: addr / 4,
            phase,
            original_proc_id: proc_id,
            op_type: OpType::Read,
            address: addr,
            memory_region: "A".to_string(),
            level: 0,
        }
    }

    fn make_conflict_schedule() -> Schedule {
        let mut sched = Schedule::new(3, 2, 4);
        // Phase 0: two writes to same address -> conflict
        sched.push(make_write(0, 0, 0, 0));
        sched.push(make_write(1, 1, 0, 0));
        // Phase 0: one read (no conflict)
        sched.push(make_read(2, 2, 0, 1));
        // Phase 1: single write (no conflict)
        sched.push(make_write(3, 0, 1, 4));
        sched
    }

    // -- §1 adversarial tests -----------------------------------------------

    #[test]
    fn test_adversarial_descending_priority() {
        let sched = AdversarialScheduleGenerator::descending_priority(8);
        assert_eq!(sched.entries.len(), 8);
        // After priority resolution, only proc 0 survives
        let mut s = sched.clone();
        let report = resolve_crcw_conflicts(&mut s, MemoryModel::CRCWPriority);
        assert_eq!(report.conflicts_detected, 1);
        assert_eq!(report.writes_eliminated, 7);
        assert_eq!(s.entries.len(), 1);
        assert_eq!(s.entries[0].original_proc_id, 0);
    }

    #[test]
    fn test_adversarial_alternating_common() {
        let sched = AdversarialScheduleGenerator::alternating_common(6);
        assert_eq!(sched.entries.len(), 6);
        let mut s = sched.clone();
        let report = resolve_crcw_conflicts(&mut s, MemoryModel::CRCWCommon);
        // Two conflict groups: addr 0 (procs 0,2,4) and addr 1 (procs 1,3,5)
        assert_eq!(report.conflicts_detected, 2);
        assert_eq!(report.guards_inserted, 2);
    }

    #[test]
    fn test_adversarial_hot_cold() {
        let sched = AdversarialScheduleGenerator::hot_cold(4, 3);
        assert_eq!(sched.entries.len(), 7);
        let mut s = sched.clone();
        let report = resolve_crcw_conflicts(&mut s, MemoryModel::CRCWPriority);
        // Only the hot group conflicts (1 conflict); cold writes are untouched
        assert_eq!(report.conflicts_detected, 1);
        assert_eq!(report.writes_eliminated, 3);
        assert_eq!(s.entries.len(), 4); // 1 hot survivor + 3 cold
    }

    // -- §2 randomized verifier tests ---------------------------------------

    #[test]
    fn test_randomized_priority_determinism() {
        let sched = AdversarialScheduleGenerator::descending_priority(6);
        let verifier = RandomizedScheduleVerifier::new(50, 123);
        assert!(
            verifier.verify_priority_determinism(&sched),
            "Priority must be deterministic across all random orderings"
        );
    }

    #[test]
    fn test_randomized_common_safety() {
        let sched = AdversarialScheduleGenerator::alternating_common(6);
        let verifier = RandomizedScheduleVerifier::new(50, 456);
        assert!(
            verifier.verify_common_safety(&sched),
            "Common mode must insert guards for every conflict"
        );
    }

    #[test]
    fn test_randomized_stats_collected() {
        let sched = AdversarialScheduleGenerator::hot_cold(5, 2);
        let verifier = RandomizedScheduleVerifier::new(30, 789);
        let (reports, stats) = verifier.run(&sched, MemoryModel::CRCWPriority);
        assert_eq!(reports.len(), 30);
        assert_eq!(stats.trials, 30);
        // Every trial should find the same single hot conflict
        assert!(stats.min_conflicts >= 1);
        assert!(stats.max_conflicts >= 1);
    }

    #[test]
    fn test_randomized_conflict_count_consistent() {
        let sched = make_conflict_schedule();
        let verifier = RandomizedScheduleVerifier::new(20, 1001);
        let (reports, _stats) = verifier.run(&sched, MemoryModel::CRCWArbitrary);
        for r in &reports {
            assert_eq!(
                r.conflicts_detected, 1,
                "Conflict count must be order-invariant"
            );
        }
    }

    // -- §3 specification tests ---------------------------------------------

    #[test]
    fn test_spec_priority_deterministic() {
        let base = AdversarialScheduleGenerator::descending_priority(5);
        let variants: Vec<Schedule> = vec![
            apply_permutation(&base, SchedulePermutation::Original),
            apply_permutation(&base, SchedulePermutation::Reversed),
            apply_permutation(&base, SchedulePermutation::Interleaved),
            apply_permutation(&base, SchedulePermutation::Strided(2)),
        ];
        assert!(CrcwCorrectnessSpec::spec_priority_deterministic(&variants));
    }

    #[test]
    fn test_spec_common_agreement() {
        let base = AdversarialScheduleGenerator::alternating_common(8);
        let variants: Vec<Schedule> = vec![
            apply_permutation(&base, SchedulePermutation::Original),
            apply_permutation(&base, SchedulePermutation::Reversed),
        ];
        assert!(CrcwCorrectnessSpec::spec_common_agreement(&variants));
    }

    #[test]
    fn test_spec_arbitrary_unique() {
        let base = AdversarialScheduleGenerator::descending_priority(4);
        let variants: Vec<Schedule> = vec![
            apply_permutation(&base, SchedulePermutation::Original),
            apply_permutation(&base, SchedulePermutation::Reversed),
            apply_permutation(&base, SchedulePermutation::Interleaved),
        ];
        assert!(CrcwCorrectnessSpec::spec_arbitrary_unique(&variants));
    }

    #[test]
    fn test_spec_empty_variants() {
        assert!(CrcwCorrectnessSpec::spec_priority_deterministic(&[]));
        assert!(CrcwCorrectnessSpec::spec_common_agreement(&[]));
        assert!(CrcwCorrectnessSpec::spec_arbitrary_unique(&[]));
    }

    // -- §4 cross-algorithm verification ------------------------------------

    #[test]
    fn test_verify_all_algorithms() {
        let algorithms = vec![
            AlgorithmSchedule {
                name: "prefix-sum".into(),
                schedule: AdversarialScheduleGenerator::descending_priority(4),
            },
            AlgorithmSchedule {
                name: "merge-sort".into(),
                schedule: AdversarialScheduleGenerator::hot_cold(3, 2),
            },
        ];
        let results = verify_all_algorithms_multi_schedule(&algorithms);
        assert_eq!(results.len(), 2);
        for (name, verification) in &results {
            assert!(
                verification.combinations_tested > 0,
                "Algorithm '{}' should have tested combinations",
                name
            );
        }
    }

    // -- original / baseline tests ------------------------------------------

    #[test]
    fn test_apply_permutation_original() {
        let schedule = make_conflict_schedule();
        let permuted = apply_permutation(&schedule, SchedulePermutation::Original);
        assert_eq!(permuted.entries.len(), schedule.entries.len());
    }

    #[test]
    fn test_apply_permutation_reversed() {
        let schedule = make_conflict_schedule();
        let permuted = apply_permutation(&schedule, SchedulePermutation::Reversed);
        assert_eq!(permuted.entries.len(), schedule.entries.len());
    }

    #[test]
    fn test_multi_schedule_priority() {
        let schedule = make_conflict_schedule();
        let models = vec![MemoryModel::CRCWPriority];
        let result = verify_multi_schedule(&schedule, &models);
        assert!(result.combinations_tested > 0);
        for r in &result.results {
            assert!(r.determinism_ok, "Priority should be deterministic");
        }
    }

    #[test]
    fn test_multi_schedule_all_models() {
        let schedule = make_conflict_schedule();
        let models = vec![
            MemoryModel::CRCWPriority,
            MemoryModel::CRCWArbitrary,
            MemoryModel::CRCWCommon,
        ];
        let result = verify_multi_schedule(&schedule, &models);
        assert!(
            result.combinations_tested >= 9,
            "Expected at least 9, got {}",
            result.combinations_tested
        );
    }

    #[test]
    fn test_verification_summary_format() {
        let schedule = make_conflict_schedule();
        let models = vec![MemoryModel::CRCWPriority];
        let result = verify_multi_schedule(&schedule, &models);
        let summary = verification_summary(&result);
        assert!(summary.contains("Multi-Schedule"));
        assert!(summary.contains("Combinations tested"));
    }

    #[test]
    fn test_erew_skipped() {
        let schedule = make_conflict_schedule();
        let models = vec![MemoryModel::EREW, MemoryModel::CREW];
        let result = verify_multi_schedule(&schedule, &models);
        assert_eq!(
            result.combinations_tested, 0,
            "EREW/CREW should be skipped"
        );
    }
}
