//! Hash-based partition module for PRAM address → cache-line block mapping.
//!
//! Provides multiple hash families (Siegel k-wise independent, 2-universal,
//! MurmurHash3, identity) and a partition engine that assigns PRAM addresses
//! to cache-line-aligned blocks with overflow analysis.

pub mod siegel_hash;
pub mod two_universal;
pub mod murmur;
pub mod identity;
pub mod tabulation;
pub mod block_assignment;
pub mod overflow_analysis;
pub mod partition_engine;
pub mod independence;
pub mod adaptive;

/// Trait that all hash function implementations must satisfy.
pub trait HashFunction {
    /// Hash a 64-bit key to a 64-bit output.
    fn hash(&self, key: u64) -> u64;

    /// Hash a 64-bit key into the range [0, range).
    fn hash_to_range(&self, key: u64, range: u64) -> u64 {
        if range == 0 {
            return 0;
        }
        self.hash(key) % range
    }
}

/// Block identifier type (index of a cache-line-aligned block).
pub type BlockId = usize;
