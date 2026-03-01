//! Simple tabulation hashing.
//!
//! Splits a 64-bit key into eight 8-bit chunks and XORs together independent
//! random lookup-table entries — the standard practical alternative to Siegel
//! hashing with high independence guarantees (3-wise independent, strong
//! concentration bounds in practice).

use rand::Rng;
use super::HashFunction;

/// Number of byte positions in a u64 key.
const NUM_CHUNKS: usize = 8;
/// Number of possible values per byte.
const TABLE_SIZE: usize = 256;

/// Simple tabulation hash function.
///
/// Maintains `NUM_CHUNKS` tables of 256 random u64 values each.
/// For a key, the hash is computed as:
///   h(key) = T_0[key & 0xFF] ^ T_1[(key >> 8) & 0xFF] ^ ... ^ T_7[(key >> 56) & 0xFF]
#[derive(Clone, Debug)]
pub struct TabulationHash {
    tables: [[u64; TABLE_SIZE]; NUM_CHUNKS],
}

impl TabulationHash {
    /// Create a new tabulation hash with random tables drawn from the given RNG.
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        let mut tables = [[0u64; TABLE_SIZE]; NUM_CHUNKS];
        for table in tables.iter_mut() {
            for entry in table.iter_mut() {
                *entry = rng.gen();
            }
        }
        Self { tables }
    }

    /// Create a tabulation hash from a deterministic seed.
    pub fn from_seed(seed: u64) -> Self {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(seed);
        Self::new(&mut rng)
    }
}

impl HashFunction for TabulationHash {
    #[inline]
    fn hash(&self, key: u64) -> u64 {
        let mut h = 0u64;
        for i in 0..NUM_CHUNKS {
            let byte = ((key >> (i * 8)) & 0xFF) as usize;
            h ^= self.tables[i][byte];
        }
        h
    }

    fn hash_to_range(&self, key: u64, range: u64) -> u64 {
        if range == 0 {
            return 0;
        }
        self.hash(key) % range
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::collections::HashSet;

    fn make_hasher() -> TabulationHash {
        TabulationHash::from_seed(42)
    }

    #[test]
    fn test_deterministic() {
        let h = make_hasher();
        assert_eq!(h.hash(12345), h.hash(12345));
        assert_eq!(h.hash(0), h.hash(0));
        assert_eq!(h.hash(u64::MAX), h.hash(u64::MAX));
    }

    #[test]
    fn test_from_seed_reproducible() {
        let h1 = TabulationHash::from_seed(99);
        let h2 = TabulationHash::from_seed(99);
        for key in [0, 1, 42, 1000, u64::MAX] {
            assert_eq!(h1.hash(key), h2.hash(key));
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let h1 = TabulationHash::from_seed(0);
        let h2 = TabulationHash::from_seed(1);
        // At least one of several keys should produce different hashes.
        let any_differ = (0..100u64).any(|k| h1.hash(k) != h2.hash(k));
        assert!(any_differ, "different seeds should produce different functions");
    }

    #[test]
    fn test_different_keys_differ() {
        let h = make_hasher();
        let h0 = h.hash(0);
        let h1 = h.hash(1);
        assert_ne!(h0, h1, "distinct keys should (almost certainly) hash differently");
    }

    #[test]
    fn test_hash_to_range_bounds() {
        let h = make_hasher();
        for key in 0..1000u64 {
            assert!(h.hash_to_range(key, 16) < 16);
            assert!(h.hash_to_range(key, 1) == 0);
            assert!(h.hash_to_range(key, 256) < 256);
        }
    }

    #[test]
    fn test_hash_to_range_zero() {
        let h = make_hasher();
        assert_eq!(h.hash_to_range(42, 0), 0);
        assert_eq!(h.hash_to_range(0, 0), 0);
    }

    #[test]
    fn test_distribution_quality() {
        let h = make_hasher();
        let m = 16u64;
        let n = 10_000u64;
        let mut counts = vec![0u64; m as usize];
        for key in 0..n {
            counts[h.hash_to_range(key, m) as usize] += 1;
        }
        let expected = n as f64 / m as f64;
        for (i, &c) in counts.iter().enumerate() {
            let ratio = c as f64 / expected;
            assert!(
                ratio > 0.5 && ratio < 2.0,
                "bucket {} has {} items (expected ~{:.0})",
                i, c, expected,
            );
        }
    }

    #[test]
    fn test_large_key() {
        let h = make_hasher();
        // Should not panic on extreme values.
        let _ = h.hash(u64::MAX);
        let _ = h.hash(u64::MAX - 1);
        let _ = h.hash_to_range(u64::MAX, 1000);
    }

    #[test]
    fn test_zero_key() {
        let h = make_hasher();
        // h(0) = T_0[0] ^ T_1[0] ^ ... ^ T_7[0]; should be a valid u64.
        let val = h.hash(0);
        let _ = val;
    }

    #[test]
    fn test_collision_resistance() {
        let h = make_hasher();
        let n = 5000u64;
        let hashes: HashSet<u64> = (0..n).map(|k| h.hash(k)).collect();
        // With 64-bit outputs and 5000 keys, we expect essentially no collisions.
        assert!(
            hashes.len() as u64 >= n - 5,
            "too many collisions: {} unique out of {}",
            hashes.len(),
            n,
        );
    }

    #[test]
    fn test_new_with_rng() {
        let mut rng = StdRng::seed_from_u64(77);
        let h = TabulationHash::new(&mut rng);
        // Basic smoke test: different keys map differently.
        let vals: HashSet<u64> = (0..100u64).map(|k| h.hash(k)).collect();
        assert!(vals.len() > 90, "hashes should be diverse");
    }

    #[test]
    fn test_hash_function_trait_object() {
        let h = make_hasher();
        let dyn_h: &dyn HashFunction = &h;
        let v1 = dyn_h.hash(42);
        let v2 = dyn_h.hash(42);
        assert_eq!(v1, v2);
        assert!(dyn_h.hash_to_range(42, 10) < 10);
    }

    #[test]
    fn test_byte_chunk_independence() {
        // Flipping a single byte of the key should change the hash.
        let h = make_hasher();
        let base = 0u64;
        let base_hash = h.hash(base);
        for i in 0..8 {
            let flipped = base ^ (0xFF_u64 << (i * 8));
            assert_ne!(
                h.hash(flipped), base_hash,
                "flipping byte {} should change hash", i,
            );
        }
    }
}
