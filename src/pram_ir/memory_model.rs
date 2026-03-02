//! Memory model semantics for CRCW/CREW/EREW PRAM models.
//!
//! Provides access tracking, conflict detection, and CRCW write resolution.

use std::collections::HashMap;
use super::ast::{MemoryModel, WriteResolution};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Type of memory access conflict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictType {
    /// Multiple processors reading the same address (violates EREW).
    ConcurrentRead,
    /// Multiple processors writing the same address (violates EREW/CREW).
    ConcurrentWrite,
}

impl std::fmt::Display for ConflictType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConflictType::ConcurrentRead => write!(f, "concurrent read"),
            ConflictType::ConcurrentWrite => write!(f, "concurrent write"),
        }
    }
}

/// Records a single memory access by a processor in a parallel step.
#[derive(Debug, Clone)]
pub struct AccessRecord {
    pub processor_id: usize,
    pub memory_name: String,
    pub address: usize,
    pub is_write: bool,
    pub value: Option<i64>,
}

impl AccessRecord {
    /// Create a read access record.
    pub fn read(pid: usize, mem: &str, addr: usize) -> Self {
        Self {
            processor_id: pid,
            memory_name: mem.to_string(),
            address: addr,
            is_write: false,
            value: None,
        }
    }

    /// Create a write access record.
    pub fn write(pid: usize, mem: &str, addr: usize, val: i64) -> Self {
        Self {
            processor_id: pid,
            memory_name: mem.to_string(),
            address: addr,
            is_write: true,
            value: Some(val),
        }
    }
}

/// A pending write for CRCW resolution.
#[derive(Debug, Clone)]
pub struct PendingWrite {
    pub processor_id: usize,
    pub memory_name: String,
    pub address: usize,
    pub value: i64,
}

/// Report of a detected memory access conflict.
#[derive(Debug, Clone)]
pub struct ConflictReport {
    pub memory_name: String,
    pub address: usize,
    pub conflict_type: ConflictType,
    pub processor_ids: Vec<usize>,
}

impl std::fmt::Display for ConflictReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} conflict on {}[{}] by processors {:?}",
            self.conflict_type, self.memory_name, self.address, self.processor_ids
        )
    }
}

// ---------------------------------------------------------------------------
// MemoryModelChecker
// ---------------------------------------------------------------------------

/// Tracks memory accesses within a single parallel step and detects conflicts
/// based on the memory model.
pub struct MemoryModelChecker {
    model: MemoryModel,
    reads: HashMap<(String, usize), Vec<usize>>,
    writes: HashMap<(String, usize), Vec<(usize, i64)>>,
}

impl MemoryModelChecker {
    /// Create a new checker for the given memory model.
    pub fn new(model: MemoryModel) -> Self {
        Self {
            model,
            reads: HashMap::new(),
            writes: HashMap::new(),
        }
    }

    /// The memory model being checked.
    pub fn model(&self) -> MemoryModel {
        self.model
    }

    /// Record a read access by a processor.
    pub fn record_read(&mut self, processor_id: usize, memory_name: &str, address: usize) {
        self.reads
            .entry((memory_name.to_string(), address))
            .or_default()
            .push(processor_id);
    }

    /// Record a write access by a processor.
    pub fn record_write(
        &mut self,
        processor_id: usize,
        memory_name: &str,
        address: usize,
        value: i64,
    ) {
        self.writes
            .entry((memory_name.to_string(), address))
            .or_default()
            .push((processor_id, value));
    }

    /// Check for all conflicts based on the memory model.
    pub fn check_conflicts(&self) -> Vec<ConflictReport> {
        let mut conflicts = Vec::new();
        self.check_read_conflicts(&mut conflicts);
        self.check_write_conflicts(&mut conflicts);
        conflicts
    }

    fn check_read_conflicts(&self, conflicts: &mut Vec<ConflictReport>) {
        if self.model.allows_concurrent_read() {
            return;
        }
        for ((mem, addr), pids) in &self.reads {
            if pids.len() > 1 {
                conflicts.push(ConflictReport {
                    memory_name: mem.clone(),
                    address: *addr,
                    conflict_type: ConflictType::ConcurrentRead,
                    processor_ids: pids.clone(),
                });
            }
        }
    }

    fn check_write_conflicts(&self, conflicts: &mut Vec<ConflictReport>) {
        if self.model.allows_concurrent_write() {
            return;
        }
        for ((mem, addr), writers) in &self.writes {
            if writers.len() > 1 {
                conflicts.push(ConflictReport {
                    memory_name: mem.clone(),
                    address: *addr,
                    conflict_type: ConflictType::ConcurrentWrite,
                    processor_ids: writers.iter().map(|(pid, _)| *pid).collect(),
                });
            }
        }
    }

    /// Whether any conflicts exist.
    pub fn has_conflicts(&self) -> bool {
        !self.check_conflicts().is_empty()
    }

    /// Clear all recorded accesses for a new step.
    pub fn clear(&mut self) {
        self.reads.clear();
        self.writes.clear();
    }

    /// Static helper: check a slice of access records for conflicts.
    pub fn check_step(model: MemoryModel, accesses: &[AccessRecord]) -> Vec<ConflictReport> {
        let mut checker = Self::new(model);
        for access in accesses {
            if access.is_write {
                checker.record_write(
                    access.processor_id,
                    &access.memory_name,
                    access.address,
                    access.value.unwrap_or(0),
                );
            } else {
                checker.record_read(access.processor_id, &access.memory_name, access.address);
            }
        }
        checker.check_conflicts()
    }

    /// Get all pending writes grouped by (memory, address).
    pub fn pending_writes(&self) -> HashMap<(String, usize), Vec<PendingWrite>> {
        let mut result = HashMap::new();
        for ((mem, addr), writers) in &self.writes {
            let pending: Vec<PendingWrite> = writers
                .iter()
                .map(|(pid, val)| PendingWrite {
                    processor_id: *pid,
                    memory_name: mem.clone(),
                    address: *addr,
                    value: *val,
                })
                .collect();
            result.insert((mem.clone(), *addr), pending);
        }
        result
    }

    /// Number of distinct read locations.
    pub fn read_location_count(&self) -> usize {
        self.reads.len()
    }

    /// Number of distinct write locations.
    pub fn write_location_count(&self) -> usize {
        self.writes.len()
    }

    /// Read counts per (memory, address).
    pub fn read_counts(&self) -> HashMap<(String, usize), usize> {
        self.reads
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect()
    }

    /// Write counts per (memory, address).
    pub fn write_counts(&self) -> HashMap<(String, usize), usize> {
        self.writes
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect()
    }

    /// Get all processors that read a specific location.
    pub fn readers_of(&self, memory: &str, address: usize) -> Vec<usize> {
        self.reads
            .get(&(memory.to_string(), address))
            .cloned()
            .unwrap_or_default()
    }

    /// Get all processors that write to a specific location.
    pub fn writers_of(&self, memory: &str, address: usize) -> Vec<(usize, i64)> {
        self.writes
            .get(&(memory.to_string(), address))
            .cloned()
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// CRCWResolver
// ---------------------------------------------------------------------------

/// Resolves concurrent writes under various CRCW policies.
pub struct CRCWResolver;

impl CRCWResolver {
    /// Priority resolution: lowest processor ID wins.
    pub fn resolve_priority(writes: &[PendingWrite]) -> Option<PendingWrite> {
        writes.iter().min_by_key(|w| w.processor_id).cloned()
    }

    /// Arbitrary resolution: non-deterministic winner.
    pub fn resolve_arbitrary(writes: &[PendingWrite]) -> Option<PendingWrite> {
        if writes.is_empty() {
            return None;
        }
        use rand::Rng;
        let idx = rand::thread_rng().gen_range(0..writes.len());
        Some(writes[idx].clone())
    }

    /// Deterministic arbitrary resolution (for testing): picks the writer at the given index.
    pub fn resolve_arbitrary_deterministic(
        writes: &[PendingWrite],
        index: usize,
    ) -> Option<PendingWrite> {
        if writes.is_empty() {
            return None;
        }
        Some(writes[index % writes.len()].clone())
    }

    /// Common resolution: all writers must write the same value.
    /// Returns `Err` if values disagree.
    pub fn resolve_common(writes: &[PendingWrite]) -> Result<Option<PendingWrite>, String> {
        if writes.is_empty() {
            return Ok(None);
        }
        let first_val = writes[0].value;
        for w in &writes[1..] {
            if w.value != first_val {
                return Err(format!(
                    "CRCW-Common conflict: processor {} wrote {} but processor {} wrote {} to {}[{}]",
                    writes[0].processor_id,
                    first_val,
                    w.processor_id,
                    w.value,
                    w.memory_name,
                    w.address,
                ));
            }
        }
        Ok(Some(writes[0].clone()))
    }

    /// Dispatch to the appropriate resolution policy.
    pub fn resolve(
        resolution: WriteResolution,
        writes: &[PendingWrite],
    ) -> Result<Option<PendingWrite>, String> {
        match resolution {
            WriteResolution::Priority => Ok(Self::resolve_priority(writes)),
            WriteResolution::Arbitrary => Ok(Self::resolve_arbitrary(writes)),
            WriteResolution::Common => Self::resolve_common(writes),
        }
    }

    /// Resolve every write-group in `pending` and return the winning writes.
    pub fn resolve_all(
        resolution: WriteResolution,
        pending: &HashMap<(String, usize), Vec<PendingWrite>>,
    ) -> Result<Vec<PendingWrite>, String> {
        let mut winners = Vec::new();
        for writes in pending.values() {
            if writes.len() <= 1 {
                if let Some(w) = writes.first() {
                    winners.push(w.clone());
                }
            } else {
                if let Some(w) = Self::resolve(resolution, writes)? {
                    winners.push(w);
                }
            }
        }
        Ok(winners)
    }
}

// ---------------------------------------------------------------------------
// MemoryState – simulated shared memory
// ---------------------------------------------------------------------------

/// Simulated shared memory state for step-by-step execution.
#[derive(Debug, Clone)]
pub struct MemoryState {
    regions: HashMap<String, Vec<i64>>,
}

impl Default for MemoryState {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryState {
    pub fn new() -> Self {
        Self {
            regions: HashMap::new(),
        }
    }

    /// Allocate a region filled with `init_value`.
    pub fn allocate(&mut self, name: &str, size: usize, init_value: i64) {
        self.regions
            .insert(name.to_string(), vec![init_value; size]);
    }

    /// Allocate a region with the given contents.
    pub fn allocate_with(&mut self, name: &str, contents: Vec<i64>) {
        self.regions.insert(name.to_string(), contents);
    }

    /// Read a single cell.
    pub fn read(&self, name: &str, addr: usize) -> Result<i64, String> {
        let region = self
            .regions
            .get(name)
            .ok_or_else(|| format!("Unknown memory region: {}", name))?;
        if addr >= region.len() {
            return Err(format!(
                "Out of bounds: {}[{}] (size {})",
                name,
                addr,
                region.len()
            ));
        }
        Ok(region[addr])
    }

    /// Write a single cell.
    pub fn write(&mut self, name: &str, addr: usize, value: i64) -> Result<(), String> {
        let region = self
            .regions
            .get_mut(name)
            .ok_or_else(|| format!("Unknown memory region: {}", name))?;
        if addr >= region.len() {
            return Err(format!(
                "Out of bounds: {}[{}] (size {})",
                name,
                addr,
                region.len()
            ));
        }
        region[addr] = value;
        Ok(())
    }

    /// Size of a region, or `None` if it does not exist.
    pub fn region_size(&self, name: &str) -> Option<usize> {
        self.regions.get(name).map(|r| r.len())
    }

    /// Borrow the contents of a region.
    pub fn region_contents(&self, name: &str) -> Option<&[i64]> {
        self.regions.get(name).map(|r| r.as_slice())
    }

    /// Whether a region exists.
    pub fn has_region(&self, name: &str) -> bool {
        self.regions.contains_key(name)
    }

    /// All region names.
    pub fn region_names(&self) -> Vec<String> {
        self.regions.keys().cloned().collect()
    }
}

// ---------------------------------------------------------------------------
// StepSimulator
// ---------------------------------------------------------------------------

/// Simulates parallel steps with access checking and CRCW resolution.
pub struct StepSimulator {
    model: MemoryModel,
    memory: MemoryState,
}

impl StepSimulator {
    pub fn new(model: MemoryModel, memory: MemoryState) -> Self {
        Self { model, memory }
    }

    pub fn model(&self) -> MemoryModel {
        self.model
    }

    pub fn memory(&self) -> &MemoryState {
        &self.memory
    }

    pub fn memory_mut(&mut self) -> &mut MemoryState {
        &mut self.memory
    }

    pub fn into_memory(self) -> MemoryState {
        self.memory
    }

    /// Execute a parallel step.
    ///
    /// Returns any conflicts detected.  For non-CRCW models, writes are **not**
    /// applied when conflicts exist.  For CRCW models, the appropriate resolution
    /// policy is used.
    pub fn execute_step(
        &mut self,
        accesses: &[AccessRecord],
    ) -> Result<Vec<ConflictReport>, String> {
        let conflicts = MemoryModelChecker::check_step(self.model, accesses);

        if !conflicts.is_empty() && !self.model.allows_concurrent_write() {
            return Ok(conflicts);
        }

        // Group writes by (memory, address).
        let mut write_groups: HashMap<(String, usize), Vec<PendingWrite>> = HashMap::new();
        for access in accesses.iter().filter(|a| a.is_write) {
            write_groups
                .entry((access.memory_name.clone(), access.address))
                .or_default()
                .push(PendingWrite {
                    processor_id: access.processor_id,
                    memory_name: access.memory_name.clone(),
                    address: access.address,
                    value: access.value.unwrap_or(0),
                });
        }

        let resolution = match self.model {
            MemoryModel::CRCWPriority => WriteResolution::Priority,
            MemoryModel::CRCWArbitrary => WriteResolution::Arbitrary,
            MemoryModel::CRCWCommon => WriteResolution::Common,
            _ => WriteResolution::Priority,
        };

        for writes in write_groups.values() {
            if writes.len() == 1 {
                self.memory
                    .write(&writes[0].memory_name, writes[0].address, writes[0].value)?;
            } else {
                let winner = CRCWResolver::resolve(resolution, writes)?;
                if let Some(w) = winner {
                    self.memory.write(&w.memory_name, w.address, w.value)?;
                }
            }
        }

        Ok(conflicts)
    }
}

// ---------------------------------------------------------------------------
// SimAccess / SimResult – step simulation types
// ---------------------------------------------------------------------------

/// A simulated memory access for step-level simulation.
#[derive(Debug, Clone)]
pub struct SimAccess {
    pub proc_id: usize,
    pub memory: String,
    pub address: u64,
    pub is_write: bool,
    pub value: i64,
}

/// Result of simulating a parallel step.
#[derive(Debug, Clone)]
pub struct SimResult {
    /// Final memory state after the step.
    pub final_memory_state: HashMap<(String, u64), i64>,
    /// Detected conflicts during the step.
    pub conflicts: Vec<ConflictReport>,
    /// Writes that were actually resolved and applied.
    pub resolved_writes: Vec<(String, u64, i64)>,
}

/// A violation detected during program-level memory model checking.
#[derive(Debug, Clone)]
pub struct ModelViolation {
    pub location: String,
    pub description: String,
    pub severity: ViolationSeverity,
}

/// Severity of a model violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViolationSeverity {
    Error,
    Warning,
}

impl std::fmt::Display for ModelViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] {}: {}", self.severity, self.location, self.description)
    }
}

impl MemoryModelChecker {
    /// Simulate a complete parallel step, returning the final state and conflicts.
    pub fn simulate_step(
        model: MemoryModel,
        accesses: &[SimAccess],
        initial_state: &HashMap<(String, u64), i64>,
    ) -> SimResult {
        let mut checker = MemoryModelChecker::new(model);
        let mut state = initial_state.clone();

        for a in accesses {
            if a.is_write {
                checker.record_write(a.proc_id, &a.memory, a.address as usize, a.value);
            } else {
                checker.record_read(a.proc_id, &a.memory, a.address as usize);
            }
        }

        let conflicts = checker.check_conflicts();

        let mut resolved_writes = Vec::new();
        if conflicts.is_empty() || model.allows_concurrent_write() {
            let mut write_groups: HashMap<(String, u64), Vec<(usize, i64)>> = HashMap::new();
            for a in accesses.iter().filter(|a| a.is_write) {
                write_groups
                    .entry((a.memory.clone(), a.address))
                    .or_default()
                    .push((a.proc_id, a.value));
            }

            for ((mem, addr), writers) in &write_groups {
                if writers.len() == 1 {
                    let val = writers[0].1;
                    state.insert((mem.clone(), *addr), val);
                    resolved_writes.push((mem.clone(), *addr, val));
                } else {
                    let resolved = resolve_crcw_writes(
                        writers,
                        match model {
                            MemoryModel::CRCWPriority => WriteResolution::Priority,
                            MemoryModel::CRCWArbitrary => WriteResolution::Arbitrary,
                            MemoryModel::CRCWCommon => WriteResolution::Common,
                            _ => WriteResolution::Priority,
                        },
                    );
                    for (_, v) in &resolved {
                        state.insert((mem.clone(), *addr), *v);
                        resolved_writes.push((mem.clone(), *addr, *v));
                    }
                }
            }
        }

        SimResult {
            final_memory_state: state,
            conflicts,
            resolved_writes,
        }
    }

    /// Check an entire PRAM program for potential memory model violations.
    pub fn check_program(program: &super::ast::PramProgram) -> Vec<ModelViolation> {
        let mut violations = Vec::new();
        let model = program.memory_model;

        for stmt in &program.body {
            check_stmt_for_violations(stmt, model, &mut violations, "body");
        }

        violations
    }
}

fn check_stmt_for_violations(
    stmt: &super::ast::Stmt,
    model: MemoryModel,
    violations: &mut Vec<ModelViolation>,
    context: &str,
) {
    use super::ast::Stmt;
    match stmt {
        Stmt::ParallelFor { proc_var, body, .. } => {
            for s in body {
                if let Stmt::SharedWrite { index, .. } = s {
                    let vars = index.collect_variables();
                    if !vars.contains(&proc_var.to_string())
                        && !model.allows_concurrent_write()
                    {
                        violations.push(ModelViolation {
                            location: context.to_string(),
                            description: format!(
                                "Write index does not depend on '{}'; potential concurrent write under {}",
                                proc_var, model
                            ),
                            severity: ViolationSeverity::Warning,
                        });
                    }
                }
            }

            for s in body {
                check_stmt_for_violations(s, model, violations, &format!("{}::{}", context, proc_var));
            }
        }
        Stmt::If { then_body, else_body, .. } => {
            for s in then_body {
                check_stmt_for_violations(s, model, violations, context);
            }
            for s in else_body {
                check_stmt_for_violations(s, model, violations, context);
            }
        }
        Stmt::SeqFor { body, .. } | Stmt::While { body, .. } | Stmt::Block(body) => {
            for s in body {
                check_stmt_for_violations(s, model, violations, context);
            }
        }
        _ => {}
    }
}

/// Resolve concurrent writes under a given CRCW policy.
///
/// Input: slice of `(processor_id, value)` tuples, all targeting the same address.
/// Returns: resolved `(relative_address_0, value)` pairs (at most 1 entry).
pub fn resolve_crcw_writes(writes: &[(usize, i64)], policy: WriteResolution) -> Vec<(u64, i64)> {
    if writes.is_empty() {
        return vec![];
    }
    if writes.len() == 1 {
        return vec![(0, writes[0].1)];
    }
    match policy {
        WriteResolution::Priority => {
            let winner = writes.iter().min_by_key(|(pid, _)| *pid).unwrap();
            vec![(0, winner.1)]
        }
        WriteResolution::Common => {
            let first_val = writes[0].1;
            if writes.iter().all(|(_, v)| *v == first_val) {
                vec![(0, first_val)]
            } else {
                vec![]
            }
        }
        WriteResolution::Arbitrary => {
            let winner = writes.iter().min_by_key(|(pid, _)| *pid).unwrap();
            vec![(0, winner.1)]
        }
    }
}

// ---------------------------------------------------------------------------
// AccessPattern
// ---------------------------------------------------------------------------

/// Describes the access pattern of a parallel step for static analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessPattern {
    /// Each processor accesses a unique address.
    Exclusive,
    /// All processors access the same address.
    Broadcast,
    /// Cannot be determined statically.
    DataDependent,
    /// Addresses form a permutation of processor IDs.
    Permutation,
}

impl AccessPattern {
    /// Whether this pattern is valid under `model` for reads (`!is_write`) or writes.
    pub fn is_valid_for(&self, model: MemoryModel, is_write: bool) -> bool {
        match self {
            AccessPattern::Exclusive | AccessPattern::Permutation => true,
            AccessPattern::Broadcast => {
                if is_write {
                    model.allows_concurrent_write()
                } else {
                    model.allows_concurrent_read()
                }
            }
            AccessPattern::DataDependent => {
                if is_write {
                    model.allows_concurrent_write()
                } else {
                    model.allows_concurrent_read()
                }
            }
        }
    }

    /// Classify from a set of `(processor_id, address)` pairs.
    pub fn classify(accesses: &[(usize, usize)]) -> Self {
        if accesses.is_empty() {
            return AccessPattern::Exclusive;
        }

        // All same address?
        let first_addr = accesses[0].1;
        if accesses.len() > 1 && accesses.iter().all(|(_, a)| *a == first_addr) {
            return AccessPattern::Broadcast;
        }

        let mut addrs: Vec<usize> = accesses.iter().map(|(_, a)| *a).collect();
        addrs.sort();
        let unique_count = {
            addrs.dedup();
            addrs.len()
        };

        if unique_count == accesses.len() {
            // All distinct addresses – might be a permutation.
            let mut pids: Vec<usize> = accesses.iter().map(|(p, _)| *p).collect();
            pids.sort();
            let mut sorted_addrs = addrs.clone();
            sorted_addrs.sort();
            if pids == sorted_addrs {
                return AccessPattern::Permutation;
            }
            return AccessPattern::Exclusive;
        }

        AccessPattern::DataDependent
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- MemoryModelChecker ------------------------------------------------

    #[test]
    fn test_erew_no_conflicts() {
        let accesses = vec![
            AccessRecord::read(0, "A", 0),
            AccessRecord::read(1, "A", 1),
            AccessRecord::write(2, "A", 2, 42),
        ];
        let conflicts = MemoryModelChecker::check_step(MemoryModel::EREW, &accesses);
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_erew_read_conflict() {
        let accesses = vec![
            AccessRecord::read(0, "A", 0),
            AccessRecord::read(1, "A", 0),
        ];
        let conflicts = MemoryModelChecker::check_step(MemoryModel::EREW, &accesses);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].conflict_type, ConflictType::ConcurrentRead);
        assert_eq!(conflicts[0].processor_ids.len(), 2);
    }

    #[test]
    fn test_erew_write_conflict() {
        let accesses = vec![
            AccessRecord::write(0, "A", 0, 1),
            AccessRecord::write(1, "A", 0, 2),
        ];
        let conflicts = MemoryModelChecker::check_step(MemoryModel::EREW, &accesses);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].conflict_type, ConflictType::ConcurrentWrite);
    }

    #[test]
    fn test_crew_allows_concurrent_read() {
        let accesses = vec![
            AccessRecord::read(0, "A", 0),
            AccessRecord::read(1, "A", 0),
            AccessRecord::read(2, "A", 0),
        ];
        let conflicts = MemoryModelChecker::check_step(MemoryModel::CREW, &accesses);
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_crew_write_conflict() {
        let accesses = vec![
            AccessRecord::write(0, "A", 0, 10),
            AccessRecord::write(1, "A", 0, 20),
        ];
        let conflicts = MemoryModelChecker::check_step(MemoryModel::CREW, &accesses);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].conflict_type, ConflictType::ConcurrentWrite);
    }

    #[test]
    fn test_crcw_no_conflict_report() {
        let accesses = vec![
            AccessRecord::read(0, "A", 0),
            AccessRecord::read(1, "A", 0),
            AccessRecord::write(2, "A", 0, 1),
            AccessRecord::write(3, "A", 0, 2),
        ];
        let conflicts =
            MemoryModelChecker::check_step(MemoryModel::CRCWPriority, &accesses);
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_checker_clear() {
        let mut checker = MemoryModelChecker::new(MemoryModel::EREW);
        checker.record_read(0, "A", 0);
        checker.record_read(1, "A", 0);
        assert!(checker.has_conflicts());
        checker.clear();
        assert!(!checker.has_conflicts());
    }

    #[test]
    fn test_checker_readers_writers() {
        let mut checker = MemoryModelChecker::new(MemoryModel::CREW);
        checker.record_read(0, "A", 5);
        checker.record_read(3, "A", 5);
        checker.record_write(1, "A", 7, 99);
        assert_eq!(checker.readers_of("A", 5), vec![0, 3]);
        assert_eq!(checker.writers_of("A", 7), vec![(1, 99)]);
        assert_eq!(checker.read_location_count(), 1);
        assert_eq!(checker.write_location_count(), 1);
    }

    // -- CRCWResolver ------------------------------------------------------

    #[test]
    fn test_priority_resolution() {
        let writes = vec![
            PendingWrite { processor_id: 3, memory_name: "A".into(), address: 0, value: 30 },
            PendingWrite { processor_id: 1, memory_name: "A".into(), address: 0, value: 10 },
            PendingWrite { processor_id: 5, memory_name: "A".into(), address: 0, value: 50 },
        ];
        let winner = CRCWResolver::resolve_priority(&writes).unwrap();
        assert_eq!(winner.processor_id, 1);
        assert_eq!(winner.value, 10);
    }

    #[test]
    fn test_common_resolution_agree() {
        let writes = vec![
            PendingWrite { processor_id: 0, memory_name: "A".into(), address: 0, value: 42 },
            PendingWrite { processor_id: 1, memory_name: "A".into(), address: 0, value: 42 },
            PendingWrite { processor_id: 2, memory_name: "A".into(), address: 0, value: 42 },
        ];
        let winner = CRCWResolver::resolve_common(&writes).unwrap().unwrap();
        assert_eq!(winner.value, 42);
    }

    #[test]
    fn test_common_resolution_disagree() {
        let writes = vec![
            PendingWrite { processor_id: 0, memory_name: "A".into(), address: 0, value: 1 },
            PendingWrite { processor_id: 1, memory_name: "A".into(), address: 0, value: 2 },
        ];
        assert!(CRCWResolver::resolve_common(&writes).is_err());
    }

    #[test]
    fn test_arbitrary_deterministic() {
        let writes = vec![
            PendingWrite { processor_id: 0, memory_name: "A".into(), address: 0, value: 10 },
            PendingWrite { processor_id: 1, memory_name: "A".into(), address: 0, value: 20 },
            PendingWrite { processor_id: 2, memory_name: "A".into(), address: 0, value: 30 },
        ];
        let w = CRCWResolver::resolve_arbitrary_deterministic(&writes, 2).unwrap();
        assert_eq!(w.value, 30);
    }

    #[test]
    fn test_resolve_empty() {
        let empty: Vec<PendingWrite> = vec![];
        assert!(CRCWResolver::resolve_priority(&empty).is_none());
        assert!(CRCWResolver::resolve_arbitrary(&empty).is_none());
        assert!(CRCWResolver::resolve_common(&empty).unwrap().is_none());
    }

    // -- MemoryState -------------------------------------------------------

    #[test]
    fn test_memory_state_basic() {
        let mut mem = MemoryState::new();
        mem.allocate("A", 10, 0);
        assert_eq!(mem.region_size("A"), Some(10));
        assert!(mem.has_region("A"));
        assert!(!mem.has_region("B"));
        mem.write("A", 3, 42).unwrap();
        assert_eq!(mem.read("A", 3).unwrap(), 42);
        assert_eq!(mem.read("A", 0).unwrap(), 0);
    }

    #[test]
    fn test_memory_state_out_of_bounds() {
        let mut mem = MemoryState::new();
        mem.allocate("A", 5, 0);
        assert!(mem.read("A", 5).is_err());
        assert!(mem.write("A", 10, 1).is_err());
    }

    #[test]
    fn test_memory_state_unknown_region() {
        let mem = MemoryState::new();
        assert!(mem.read("X", 0).is_err());
    }

    #[test]
    fn test_memory_allocate_with() {
        let mut mem = MemoryState::new();
        mem.allocate_with("B", vec![10, 20, 30]);
        assert_eq!(mem.region_contents("B"), Some(&[10, 20, 30][..]));
    }

    // -- StepSimulator -----------------------------------------------------

    #[test]
    fn test_simulator_exclusive_writes() {
        let mut mem = MemoryState::new();
        mem.allocate("A", 4, 0);
        let mut sim = StepSimulator::new(MemoryModel::EREW, mem);
        let accesses = vec![
            AccessRecord::write(0, "A", 0, 10),
            AccessRecord::write(1, "A", 1, 20),
            AccessRecord::write(2, "A", 2, 30),
            AccessRecord::write(3, "A", 3, 40),
        ];
        let conflicts = sim.execute_step(&accesses).unwrap();
        assert!(conflicts.is_empty());
        assert_eq!(sim.memory().read("A", 0).unwrap(), 10);
        assert_eq!(sim.memory().read("A", 3).unwrap(), 40);
    }

    #[test]
    fn test_simulator_crcw_priority() {
        let mut mem = MemoryState::new();
        mem.allocate("A", 1, 0);
        let mut sim = StepSimulator::new(MemoryModel::CRCWPriority, mem);
        let accesses = vec![
            AccessRecord::write(2, "A", 0, 200),
            AccessRecord::write(0, "A", 0, 100),
            AccessRecord::write(1, "A", 0, 150),
        ];
        let conflicts = sim.execute_step(&accesses).unwrap();
        assert!(conflicts.is_empty());
        // Lowest PID (0) wins.
        assert_eq!(sim.memory().read("A", 0).unwrap(), 100);
    }

    #[test]
    fn test_simulator_crew_conflict_blocks_write() {
        let mut mem = MemoryState::new();
        mem.allocate("A", 1, 0);
        let mut sim = StepSimulator::new(MemoryModel::CREW, mem);
        let accesses = vec![
            AccessRecord::write(0, "A", 0, 10),
            AccessRecord::write(1, "A", 0, 20),
        ];
        let conflicts = sim.execute_step(&accesses).unwrap();
        assert_eq!(conflicts.len(), 1);
        // Original value should be unchanged because writes were not applied.
        assert_eq!(sim.memory().read("A", 0).unwrap(), 0);
    }

    // -- AccessPattern -----------------------------------------------------

    #[test]
    fn test_classify_broadcast() {
        let accesses = vec![(0, 5), (1, 5), (2, 5)];
        assert_eq!(AccessPattern::classify(&accesses), AccessPattern::Broadcast);
    }

    #[test]
    fn test_classify_exclusive() {
        let accesses = vec![(0, 0), (1, 1), (2, 2)];
        assert_eq!(
            AccessPattern::classify(&accesses),
            AccessPattern::Permutation
        );
    }

    #[test]
    fn test_classify_data_dependent() {
        // Two processors access the same address but not all.
        let accesses = vec![(0, 0), (1, 0), (2, 1)];
        assert_eq!(
            AccessPattern::classify(&accesses),
            AccessPattern::DataDependent
        );
    }

    #[test]
    fn test_pattern_validity() {
        assert!(AccessPattern::Exclusive.is_valid_for(MemoryModel::EREW, true));
        assert!(!AccessPattern::Broadcast.is_valid_for(MemoryModel::EREW, false));
        assert!(AccessPattern::Broadcast.is_valid_for(MemoryModel::CREW, false));
        assert!(!AccessPattern::Broadcast.is_valid_for(MemoryModel::CREW, true));
        assert!(AccessPattern::Broadcast.is_valid_for(MemoryModel::CRCWPriority, true));
    }

    #[test]
    fn test_conflict_report_display() {
        let r = ConflictReport {
            memory_name: "A".into(),
            address: 3,
            conflict_type: ConflictType::ConcurrentWrite,
            processor_ids: vec![0, 1, 2],
        };
        let s = format!("{}", r);
        assert!(s.contains("concurrent write"));
        assert!(s.contains("A[3]"));
    }

    #[test]
    fn test_resolve_all_priority() {
        let mut pending = HashMap::new();
        pending.insert(
            ("A".into(), 0),
            vec![
                PendingWrite { processor_id: 2, memory_name: "A".into(), address: 0, value: 20 },
                PendingWrite { processor_id: 0, memory_name: "A".into(), address: 0, value: 10 },
            ],
        );
        pending.insert(
            ("A".into(), 1),
            vec![
                PendingWrite { processor_id: 1, memory_name: "A".into(), address: 1, value: 99 },
            ],
        );
        let winners =
            CRCWResolver::resolve_all(WriteResolution::Priority, &pending).unwrap();
        assert_eq!(winners.len(), 2);
        for w in &winners {
            match w.address {
                0 => assert_eq!(w.value, 10),
                1 => assert_eq!(w.value, 99),
                _ => panic!("unexpected address"),
            }
        }
    }

    // -- SimAccess / SimResult / resolve_crcw_writes -----------------------

    #[test]
    fn test_simulate_step_exclusive() {
        let accesses = vec![
            SimAccess { proc_id: 0, memory: "A".into(), address: 0, is_write: true, value: 10 },
            SimAccess { proc_id: 1, memory: "A".into(), address: 1, is_write: true, value: 20 },
        ];
        let initial = HashMap::new();
        let result = MemoryModelChecker::simulate_step(MemoryModel::EREW, &accesses, &initial);
        assert!(result.conflicts.is_empty());
        assert_eq!(result.resolved_writes.len(), 2);
        assert_eq!(*result.final_memory_state.get(&("A".into(), 0)).unwrap(), 10);
        assert_eq!(*result.final_memory_state.get(&("A".into(), 1)).unwrap(), 20);
    }

    #[test]
    fn test_simulate_step_crcw_priority() {
        let accesses = vec![
            SimAccess { proc_id: 2, memory: "A".into(), address: 0, is_write: true, value: 200 },
            SimAccess { proc_id: 0, memory: "A".into(), address: 0, is_write: true, value: 100 },
        ];
        let initial = HashMap::new();
        let result = MemoryModelChecker::simulate_step(MemoryModel::CRCWPriority, &accesses, &initial);
        assert!(result.conflicts.is_empty());
        assert_eq!(*result.final_memory_state.get(&("A".into(), 0)).unwrap(), 100);
    }

    #[test]
    fn test_simulate_step_erew_conflict() {
        let accesses = vec![
            SimAccess { proc_id: 0, memory: "A".into(), address: 0, is_write: true, value: 10 },
            SimAccess { proc_id: 1, memory: "A".into(), address: 0, is_write: true, value: 20 },
        ];
        let initial = HashMap::new();
        let result = MemoryModelChecker::simulate_step(MemoryModel::EREW, &accesses, &initial);
        assert!(!result.conflicts.is_empty());
        // Writes should not be applied.
        assert!(result.resolved_writes.is_empty());
    }

    #[test]
    fn test_resolve_crcw_writes_priority() {
        let writes = vec![(3, 30i64), (1, 10), (5, 50)];
        let result = resolve_crcw_writes(&writes, WriteResolution::Priority);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 10);
    }

    #[test]
    fn test_resolve_crcw_writes_common_agree() {
        let writes = vec![(0, 42i64), (1, 42), (2, 42)];
        let result = resolve_crcw_writes(&writes, WriteResolution::Common);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 42);
    }

    #[test]
    fn test_resolve_crcw_writes_common_disagree() {
        let writes = vec![(0, 1i64), (1, 2)];
        let result = resolve_crcw_writes(&writes, WriteResolution::Common);
        assert!(result.is_empty());
    }

    #[test]
    fn test_resolve_crcw_writes_empty() {
        let writes: Vec<(usize, i64)> = vec![];
        let result = resolve_crcw_writes(&writes, WriteResolution::Priority);
        assert!(result.is_empty());
    }

    #[test]
    fn test_resolve_crcw_writes_single() {
        let writes = vec![(0, 99i64)];
        let result = resolve_crcw_writes(&writes, WriteResolution::Priority);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 99);
    }

    #[test]
    fn test_model_violation_display() {
        let v = ModelViolation {
            location: "body::p".into(),
            description: "potential conflict".into(),
            severity: ViolationSeverity::Warning,
        };
        let s = format!("{}", v);
        assert!(s.contains("Warning"));
        assert!(s.contains("potential conflict"));
    }

    #[test]
    fn test_check_program_erew_violations() {
        use crate::pram_ir::ast::*;
        let mut prog = PramProgram::new("test", MemoryModel::EREW);
        prog.body.push(Stmt::ParallelFor {
            proc_var: "p".into(),
            num_procs: Expr::int(4),
            body: vec![Stmt::SharedWrite {
                memory: Expr::var("A"),
                index: Expr::int(0), // constant index => concurrent write
                value: Expr::int(1),
            }],
        });
        let violations = MemoryModelChecker::check_program(&prog);
        assert!(!violations.is_empty());
    }
}
