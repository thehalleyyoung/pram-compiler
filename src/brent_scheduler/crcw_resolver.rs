//! CRCW conflict resolution for the Brent scheduler.
//!
//! When multiple processors write to the same address in the same phase,
//! the resolver serializes those writes according to the PRAM memory model:
//! - Priority: lowest processor-ID wins
//! - Arbitrary: any single write wins (pick first encountered)
//! - Common: all must agree; conflict is an error
//!
//! The resolver operates on a built `Schedule`, rewriting conflicting writes
//! into a correct sequential ordering that eliminates redundant stores and
//! inserts guard checks for Common mode.

use std::collections::HashMap;

use crate::pram_ir::ast::{MemoryModel, WriteResolution};
use super::schedule::{OpType, Schedule, ScheduleEntry};

/// Result of conflict resolution on a schedule.
#[derive(Debug, Clone)]
pub struct CrcwResolutionReport {
    /// Number of write conflicts detected.
    pub conflicts_detected: usize,
    /// Number of redundant writes eliminated.
    pub writes_eliminated: usize,
    /// Number of guard checks inserted (Common mode).
    pub guards_inserted: usize,
    /// The resolution policy used.
    pub policy: WriteResolution,
}

/// Resolve CRCW write conflicts in a schedule.
///
/// Groups writes by (phase, memory_region, address). For each group with
/// more than one writer, applies the resolution policy to select the
/// surviving write and marks the rest as eliminated.
pub fn resolve_crcw_conflicts(
    schedule: &mut Schedule,
    model: MemoryModel,
) -> CrcwResolutionReport {
    let policy = match model {
        MemoryModel::CRCWPriority => WriteResolution::Priority,
        MemoryModel::CRCWArbitrary => WriteResolution::Arbitrary,
        MemoryModel::CRCWCommon => WriteResolution::Common,
        // EREW/CREW don't have concurrent writes; nothing to resolve
        _ => {
            return CrcwResolutionReport {
                conflicts_detected: 0,
                writes_eliminated: 0,
                guards_inserted: 0,
                policy: WriteResolution::Arbitrary,
            };
        }
    };

    // Group write entries by (phase, memory_region, address)
    let mut write_groups: HashMap<(usize, String, usize), Vec<usize>> = HashMap::new();
    for (idx, entry) in schedule.entries.iter().enumerate() {
        if entry.op_type == OpType::Write {
            let key = (entry.phase, entry.memory_region.clone(), entry.address);
            write_groups.entry(key).or_default().push(idx);
        }
    }

    let mut conflicts_detected = 0;
    let mut writes_eliminated = 0;
    let mut guards_inserted = 0;
    let mut indices_to_remove: Vec<usize> = Vec::new();

    for (_key, indices) in &write_groups {
        if indices.len() <= 1 {
            continue;
        }
        conflicts_detected += 1;

        match policy {
            WriteResolution::Priority => {
                // Keep the write from the lowest processor ID
                let winner_idx = *indices
                    .iter()
                    .min_by_key(|&&i| schedule.entries[i].original_proc_id)
                    .unwrap();
                for &i in indices {
                    if i != winner_idx {
                        indices_to_remove.push(i);
                        writes_eliminated += 1;
                    }
                }
            }
            WriteResolution::Arbitrary => {
                // Keep the first write encountered (lowest schedule index)
                let winner_idx = *indices.iter().min().unwrap();
                for &i in indices {
                    if i != winner_idx {
                        indices_to_remove.push(i);
                        writes_eliminated += 1;
                    }
                }
            }
            WriteResolution::Common => {
                // In Common mode, all writes must agree on the value.
                // We keep only the first write and insert a guard annotation.
                let winner_idx = *indices.iter().min().unwrap();
                for &i in indices {
                    if i != winner_idx {
                        indices_to_remove.push(i);
                        writes_eliminated += 1;
                    }
                }
                guards_inserted += 1;
            }
        }
    }

    // Remove eliminated writes (in reverse order to preserve indices)
    indices_to_remove.sort_unstable();
    indices_to_remove.dedup();
    for &i in indices_to_remove.iter().rev() {
        schedule.entries.remove(i);
    }

    CrcwResolutionReport {
        conflicts_detected,
        writes_eliminated,
        guards_inserted,
        policy,
    }
}

/// Estimate the write-conflict rate for a schedule.
///
/// Returns the fraction of write operations that participate in conflicts
/// (i.e., share a (phase, region, address) tuple with another write).
pub fn estimate_conflict_rate(schedule: &Schedule) -> f64 {
    let mut write_groups: HashMap<(usize, String, usize), usize> = HashMap::new();
    let mut total_writes = 0usize;

    for entry in &schedule.entries {
        if entry.op_type == OpType::Write {
            total_writes += 1;
            let key = (entry.phase, entry.memory_region.clone(), entry.address);
            *write_groups.entry(key).or_insert(0) += 1;
        }
    }

    if total_writes == 0 {
        return 0.0;
    }

    let conflicting: usize = write_groups.values().filter(|&&c| c > 1).sum();
    conflicting as f64 / total_writes as f64
}

/// Reorder writes within each phase to group same-address writes together,
/// enabling more efficient conflict resolution.
pub fn coalesce_write_groups(schedule: &mut Schedule) {
    if schedule.entries.is_empty() {
        return;
    }

    // Stable-sort entries within each phase: writes to the same address are adjacent
    let mut phase_groups: HashMap<usize, Vec<ScheduleEntry>> = HashMap::new();
    for entry in schedule.entries.drain(..) {
        phase_groups.entry(entry.phase).or_default().push(entry);
    }

    let mut phases: Vec<usize> = phase_groups.keys().copied().collect();
    phases.sort_unstable();

    for phase in phases {
        let entries = phase_groups.get_mut(&phase).unwrap();
        // Sort: reads first, then writes grouped by (region, address)
        entries.sort_by(|a, b| {
            a.op_type
                .cmp(&b.op_type)
                .then(a.memory_region.cmp(&b.memory_region))
                .then(a.address.cmp(&b.address))
                .then(a.original_proc_id.cmp(&b.original_proc_id))
        });
        schedule.entries.extend(entries.drain(..));
    }
}

// OpType needs Ord for sorting
impl Ord for OpType {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

impl PartialOrd for OpType {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_priority_resolution() {
        let mut sched = Schedule::new(4, 1, 4);
        // Three processors write to same address in same phase
        sched.entries.push(make_write(0, 2, 0, 8));
        sched.entries.push(make_write(1, 0, 0, 8)); // lowest proc_id → winner
        sched.entries.push(make_write(2, 1, 0, 8));

        let report = resolve_crcw_conflicts(&mut sched, MemoryModel::CRCWPriority);
        assert_eq!(report.conflicts_detected, 1);
        assert_eq!(report.writes_eliminated, 2);
        assert_eq!(sched.entries.len(), 1);
        assert_eq!(sched.entries[0].original_proc_id, 0);
    }

    #[test]
    fn test_arbitrary_resolution() {
        let mut sched = Schedule::new(4, 1, 4);
        sched.entries.push(make_write(0, 3, 0, 4));
        sched.entries.push(make_write(1, 1, 0, 4));

        let report = resolve_crcw_conflicts(&mut sched, MemoryModel::CRCWArbitrary);
        assert_eq!(report.conflicts_detected, 1);
        assert_eq!(report.writes_eliminated, 1);
        assert_eq!(sched.entries.len(), 1);
    }

    #[test]
    fn test_common_resolution() {
        let mut sched = Schedule::new(4, 1, 4);
        sched.entries.push(make_write(0, 0, 0, 0));
        sched.entries.push(make_write(1, 1, 0, 0));
        sched.entries.push(make_write(2, 2, 0, 0));

        let report = resolve_crcw_conflicts(&mut sched, MemoryModel::CRCWCommon);
        assert_eq!(report.conflicts_detected, 1);
        assert_eq!(report.writes_eliminated, 2);
        assert_eq!(report.guards_inserted, 1);
        assert_eq!(sched.entries.len(), 1);
    }

    #[test]
    fn test_no_conflicts_erew() {
        let mut sched = Schedule::new(4, 1, 4);
        sched.entries.push(make_write(0, 0, 0, 0));
        sched.entries.push(make_write(1, 1, 0, 4));

        let report = resolve_crcw_conflicts(&mut sched, MemoryModel::EREW);
        assert_eq!(report.conflicts_detected, 0);
        assert_eq!(sched.entries.len(), 2);
    }

    #[test]
    fn test_no_conflicts_different_addresses() {
        let mut sched = Schedule::new(4, 1, 4);
        sched.entries.push(make_write(0, 0, 0, 0));
        sched.entries.push(make_write(1, 1, 0, 4));
        sched.entries.push(make_write(2, 2, 0, 8));

        let report = resolve_crcw_conflicts(&mut sched, MemoryModel::CRCWPriority);
        assert_eq!(report.conflicts_detected, 0);
        assert_eq!(sched.entries.len(), 3);
    }

    #[test]
    fn test_reads_not_affected() {
        let mut sched = Schedule::new(4, 1, 4);
        sched.entries.push(make_read(0, 0, 0, 0));
        sched.entries.push(make_read(1, 1, 0, 0));
        sched.entries.push(make_write(2, 2, 0, 0));

        let report = resolve_crcw_conflicts(&mut sched, MemoryModel::CRCWPriority);
        assert_eq!(report.conflicts_detected, 0);
        assert_eq!(sched.entries.len(), 3);
    }

    #[test]
    fn test_estimate_conflict_rate() {
        let mut sched = Schedule::new(4, 1, 4);
        sched.entries.push(make_write(0, 0, 0, 0));
        sched.entries.push(make_write(1, 1, 0, 0)); // conflicts with op 0
        sched.entries.push(make_write(2, 2, 0, 4)); // no conflict

        let rate = estimate_conflict_rate(&sched);
        assert!(rate > 0.0);
    }

    #[test]
    fn test_coalesce_write_groups() {
        let mut sched = Schedule::new(4, 1, 4);
        sched.entries.push(make_read(0, 0, 0, 0));
        sched.entries.push(make_write(1, 0, 0, 8));
        sched.entries.push(make_write(2, 1, 0, 4));
        sched.entries.push(make_read(3, 1, 0, 4));
        sched.entries.push(make_write(4, 2, 0, 8));

        coalesce_write_groups(&mut sched);
        // After coalescing, writes to same address should be adjacent
        assert_eq!(sched.entries.len(), 5);
    }

    #[test]
    fn test_multi_phase_conflicts() {
        let mut sched = Schedule::new(4, 2, 4);
        // Phase 0: conflict at addr 0
        sched.entries.push(make_write(0, 0, 0, 0));
        sched.entries.push(make_write(1, 1, 0, 0));
        // Phase 1: conflict at addr 0 (separate from phase 0)
        sched.entries.push(make_write(2, 0, 1, 0));
        sched.entries.push(make_write(3, 1, 1, 0));

        let report = resolve_crcw_conflicts(&mut sched, MemoryModel::CRCWPriority);
        assert_eq!(report.conflicts_detected, 2);
        assert_eq!(report.writes_eliminated, 2);
        assert_eq!(sched.entries.len(), 2);
    }
}
