//! Identity hash for ablation studies.
//!
//! Returns the input key unchanged (or key mod range).
//! Used as a baseline to measure the impact of hash quality
//! on partition balance and overflow.

use super::HashFunction;

/// Identity hash: h(x) = x.
#[derive(Clone, Debug, Default)]
pub struct IdentityHash;

impl IdentityHash {
    /// Create a new identity hash.
    pub fn new() -> Self {
        Self
    }
}

impl HashFunction for IdentityHash {
    fn hash(&self, key: u64) -> u64 {
        key
    }

    fn hash_to_range(&self, key: u64, range: u64) -> u64 {
        if range == 0 {
            return 0;
        }
        key % range
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let h = IdentityHash::new();
        assert_eq!(h.hash(0), 0);
        assert_eq!(h.hash(42), 42);
        assert_eq!(h.hash(u64::MAX), u64::MAX);
    }

    #[test]
    fn test_identity_range() {
        let h = IdentityHash::new();
        assert_eq!(h.hash_to_range(10, 8), 2);
        assert_eq!(h.hash_to_range(16, 16), 0);
        assert_eq!(h.hash_to_range(0, 100), 0);
    }

    #[test]
    fn test_identity_zero_range() {
        let h = IdentityHash::new();
        assert_eq!(h.hash_to_range(42, 0), 0);
    }

    #[test]
    fn test_identity_default() {
        let h = IdentityHash::default();
        assert_eq!(h.hash(99), 99);
    }
}
