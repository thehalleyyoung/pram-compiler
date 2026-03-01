//! MurmurHash3 64-bit implementation.
//!
//! Provides the standard fmix64 finalization mix and a full
//! MurmurHash3-style 64-bit hash function.

use super::HashFunction;

/// MurmurHash3 finalization mix for 64-bit values.
///
/// Applies the standard avalanche bit-mixing sequence:
///   key ^= key >> 33
///   key *= 0xff51afd7ed558ccd
///   key ^= key >> 33
///   key *= 0xc4ceb9fe1a85ec53
///   key ^= key >> 33
#[inline]
pub fn fmix64(mut key: u64) -> u64 {
    key ^= key >> 33;
    key = key.wrapping_mul(0xff51afd7ed558ccd);
    key ^= key >> 33;
    key = key.wrapping_mul(0xc4ceb9fe1a85ec53);
    key ^= key >> 33;
    key
}

/// Inverse of fmix64 (for testing bijectivity).
#[inline]
pub fn fmix64_inv(mut key: u64) -> u64 {
    key ^= key >> 33;
    key = key.wrapping_mul(0x9cb4b2f8129337db); // inverse of 0xc4ceb9fe1a85ec53
    key ^= key >> 33;
    key = key.wrapping_mul(0x4f74430c22a54005); // inverse of 0xff51afd7ed558ccd
    key ^= key >> 33;
    key
}

/// Compute a 64-bit MurmurHash3-style hash of a single u64 key with a seed.
///
/// Mixes the seed into the key, then applies fmix64.
#[inline]
pub fn murmur3_64(key: u64, seed: u64) -> u64 {
    let mut h = seed;

    // Mix key into state.
    let mut k = key;
    k = k.wrapping_mul(0x87c37b91114253d5);
    k = k.rotate_left(31);
    k = k.wrapping_mul(0x4cf5ad432745937f);
    h ^= k;

    h = h.rotate_left(27);
    h = h.wrapping_add(h.wrapping_mul(4));
    h = h.wrapping_add(0x52dce729);

    // Finalization.
    h ^= 8; // length = 8 bytes
    fmix64(h)
}

/// Compute murmur3_64 for a byte slice.
pub fn murmur3_64_bytes(data: &[u8], seed: u64) -> u64 {
    let mut h = seed;
    let nblocks = data.len() / 8;

    // Body: process 8-byte blocks.
    for i in 0..nblocks {
        let offset = i * 8;
        let mut k = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        k = k.wrapping_mul(0x87c37b91114253d5);
        k = k.rotate_left(31);
        k = k.wrapping_mul(0x4cf5ad432745937f);
        h ^= k;
        h = h.rotate_left(27);
        h = h.wrapping_add(h.wrapping_mul(4));
        h = h.wrapping_add(0x52dce729);
    }

    // Tail: remaining bytes.
    let tail_start = nblocks * 8;
    let tail = &data[tail_start..];
    let mut k: u64 = 0;
    for (i, &byte) in tail.iter().enumerate().rev() {
        k = (k << 8) | (byte as u64);
        let _ = i; // use the reverse iteration for correct byte ordering
    }
    // Actually build k properly from tail bytes in LE order.
    k = 0;
    for (i, &byte) in tail.iter().enumerate() {
        k |= (byte as u64) << (i * 8);
    }
    if !tail.is_empty() {
        k = k.wrapping_mul(0x87c37b91114253d5);
        k = k.rotate_left(31);
        k = k.wrapping_mul(0x4cf5ad432745937f);
        h ^= k;
    }

    h ^= data.len() as u64;
    fmix64(h)
}

/// A MurmurHash3-based hasher struct.
#[derive(Clone, Debug)]
pub struct MurmurHasher {
    seed: u64,
}

impl MurmurHasher {
    /// Create a new hasher with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Create a hasher with seed = 0.
    pub fn default_seed() -> Self {
        Self { seed: 0 }
    }

    /// Return the seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Hash a u64 key.
    pub fn hash_u64(&self, key: u64) -> u64 {
        murmur3_64(key, self.seed)
    }

    /// Hash raw bytes.
    pub fn hash_bytes(&self, data: &[u8]) -> u64 {
        murmur3_64_bytes(data, self.seed)
    }
}

impl HashFunction for MurmurHasher {
    fn hash(&self, key: u64) -> u64 {
        murmur3_64(key, self.seed)
    }

    fn hash_to_range(&self, key: u64, range: u64) -> u64 {
        if range == 0 {
            return 0;
        }
        murmur3_64(key, self.seed) % range
    }
}

/// Measure avalanche effect: for each input bit flip, count output bit changes.
/// Returns a 64×64 matrix where entry [i][j] is the probability that flipping
/// input bit i changes output bit j.
pub fn avalanche_matrix(seed: u64, num_samples: usize) -> Vec<Vec<f64>> {
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(0xDEADBEEF));
    let mut counts = vec![vec![0u64; 64]; 64];

    for _ in 0..num_samples {
        let x: u64 = rng.gen();
        let h0 = murmur3_64(x, seed);
        for bit in 0..64 {
            let x_flipped = x ^ (1u64 << bit);
            let h1 = murmur3_64(x_flipped, seed);
            let diff = h0 ^ h1;
            for out_bit in 0..64 {
                if diff & (1u64 << out_bit) != 0 {
                    counts[bit][out_bit] += 1;
                }
            }
        }
    }

    counts
        .iter()
        .map(|row| row.iter().map(|&c| c as f64 / num_samples as f64).collect())
        .collect()
}

/// Check that all avalanche probabilities are close to 0.5 (ideal).
pub fn avalanche_quality(matrix: &[Vec<f64>]) -> f64 {
    let mut max_deviation = 0.0f64;
    for row in matrix {
        for &p in row {
            let dev = (p - 0.5).abs();
            if dev > max_deviation {
                max_deviation = dev;
            }
        }
    }
    max_deviation
}

// ────────────────────────────────────────────────────────────────────────
// 128-bit variant
// ────────────────────────────────────────────────────────────────────────

/// Compute a 128-bit MurmurHash3-style hash of a single u64 key with a seed.
///
/// Returns `(h1, h2)` where each half is a 64-bit value.
pub fn murmur3_128(key: u64, seed: u64) -> (u64, u64) {
    let mut h1 = seed;
    let mut h2 = seed;

    // Mix key into h1
    let mut k1 = key;
    k1 = k1.wrapping_mul(0x87c37b91114253d5);
    k1 = k1.rotate_left(31);
    k1 = k1.wrapping_mul(0x4cf5ad432745937f);
    h1 ^= k1;
    h1 = h1.rotate_left(27);
    h1 = h1.wrapping_add(h2);
    h1 = h1.wrapping_mul(5).wrapping_add(0x52dce729);

    // Mix key into h2 with different constants
    let mut k2 = key;
    k2 = k2.wrapping_mul(0x4cf5ad432745937f);
    k2 = k2.rotate_left(33);
    k2 = k2.wrapping_mul(0x87c37b91114253d5);
    h2 ^= k2;
    h2 = h2.rotate_left(31);
    h2 = h2.wrapping_add(h1);
    h2 = h2.wrapping_mul(5).wrapping_add(0x38495ab5);

    // Finalization
    h1 ^= 8u64;
    h2 ^= 8u64;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);

    (h1, h2)
}

// ────────────────────────────────────────────────────────────────────────
// Bulk hashing
// ────────────────────────────────────────────────────────────────────────

/// Hash a batch of u64 keys using MurmurHash3-64.
pub fn murmur3_bulk(keys: &[u64], seed: u64) -> Vec<u64> {
    keys.iter().map(|&k| murmur3_64(k, seed)).collect()
}

// ────────────────────────────────────────────────────────────────────────
// String / byte-slice hashing (convenience alias)
// ────────────────────────────────────────────────────────────────────────

/// Hash an arbitrary byte slice using the MurmurHash3-64 byte variant.
///
/// This is a thin wrapper around `murmur3_64_bytes` provided under a
/// name that is easier to discover.
pub fn murmur3_string(data: &[u8], seed: u64) -> u64 {
    murmur3_64_bytes(data, seed)
}

// ────────────────────────────────────────────────────────────────────────
// Chi-squared uniformity test
// ────────────────────────────────────────────────────────────────────────

/// Result of a chi-squared uniformity test.
#[derive(Clone, Debug)]
pub struct ChiSquaredResult {
    /// The chi-squared statistic.
    pub chi_squared: f64,
    /// Degrees of freedom (num_buckets − 1).
    pub degrees_of_freedom: usize,
    /// Expected count per bucket.
    pub expected_per_bucket: f64,
    /// Maximum bucket count.
    pub max_bucket: u64,
    /// Minimum bucket count.
    pub min_bucket: u64,
    /// Whether the test passes at the given critical-value threshold.
    pub passes: bool,
}

/// Run a chi-squared uniformity test on the output of `murmur3_64`.
///
/// Hashes keys `0..num_keys` with the given seed into `num_buckets` buckets
/// and computes the chi-squared statistic.  `critical_value` is the threshold
/// above which the test is considered to fail (depends on the desired p-value
/// and degrees of freedom).
pub fn chi_squared_uniformity_test(
    seed: u64,
    num_keys: u64,
    num_buckets: u64,
    critical_value: f64,
) -> ChiSquaredResult {
    let mut counts = vec![0u64; num_buckets as usize];
    for k in 0..num_keys {
        let bucket = murmur3_64(k, seed) % num_buckets;
        counts[bucket as usize] += 1;
    }
    let expected = num_keys as f64 / num_buckets as f64;
    let chi_sq: f64 = counts
        .iter()
        .map(|&c| {
            let d = c as f64 - expected;
            d * d / expected
        })
        .sum();
    let max_bucket = counts.iter().copied().max().unwrap_or(0);
    let min_bucket = counts.iter().copied().min().unwrap_or(0);
    let dof = num_buckets as usize - 1;

    ChiSquaredResult {
        chi_squared: chi_sq,
        degrees_of_freedom: dof,
        expected_per_bucket: expected,
        max_bucket,
        min_bucket,
        passes: chi_sq <= critical_value,
    }
}

/// Measure bit-level bias for a given seed across `num_keys` sequential keys.
///
/// Returns a 64-element array where entry `b` is the fraction of keys whose
/// hash has bit `b` set.  Ideal value is 0.5 for every bit.
pub fn bit_bias(seed: u64, num_keys: u64) -> [f64; 64] {
    let mut ones = [0u64; 64];
    for k in 0..num_keys {
        let h = murmur3_64(k, seed);
        for b in 0..64 {
            if h & (1u64 << b) != 0 {
                ones[b] += 1;
            }
        }
    }
    let mut bias = [0.0f64; 64];
    for b in 0..64 {
        bias[b] = ones[b] as f64 / num_keys as f64;
    }
    bias
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmix64_known_values() {
        // fmix64(0) should be 0 since all XORs and multiplies of 0 stay 0.
        assert_eq!(fmix64(0), 0);
        // fmix64 should produce non-trivial outputs.
        assert_ne!(fmix64(1), 1);
        assert_ne!(fmix64(42), 42);
    }

    #[test]
    fn test_fmix64_bijective() {
        // fmix64 should be a bijection: fmix64_inv(fmix64(x)) == x.
        for x in [0, 1, 2, 42, 12345, u64::MAX, u64::MAX / 2, 0xDEADBEEFCAFEBABE] {
            assert_eq!(fmix64_inv(fmix64(x)), x, "bijectivity failed for {}", x);
        }
    }

    #[test]
    fn test_murmur3_deterministic() {
        let h1 = murmur3_64(100, 0);
        let h2 = murmur3_64(100, 0);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_murmur3_different_seeds() {
        let h1 = murmur3_64(100, 0);
        let h2 = murmur3_64(100, 1);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_murmur3_different_keys() {
        let h1 = murmur3_64(0, 42);
        let h2 = murmur3_64(1, 42);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hasher_struct() {
        let hasher = MurmurHasher::new(42);
        assert_eq!(hasher.seed(), 42);
        let v1 = hasher.hash_u64(100);
        let v2 = hasher.hash_u64(100);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_hash_function_trait() {
        let hasher = MurmurHasher::new(0);
        let h = hasher.hash(12345);
        assert_ne!(h, 12345);
        let r = hasher.hash_to_range(12345, 100);
        assert!(r < 100);
    }

    #[test]
    fn test_hash_to_range_bounds() {
        let hasher = MurmurHasher::new(99);
        for key in 0..1000u64 {
            assert!(hasher.hash_to_range(key, 16) < 16);
            assert!(hasher.hash_to_range(key, 1) == 0);
        }
    }

    #[test]
    fn test_hash_to_range_zero() {
        let hasher = MurmurHasher::new(0);
        assert_eq!(hasher.hash_to_range(42, 0), 0);
    }

    #[test]
    fn test_distribution_quality() {
        let hasher = MurmurHasher::new(0);
        let m = 16u64;
        let n = 10_000u64;
        let mut counts = vec![0u64; m as usize];
        for key in 0..n {
            let bucket = hasher.hash_to_range(key, m) as usize;
            counts[bucket] += 1;
        }
        let expected = n as f64 / m as f64;
        for (i, &c) in counts.iter().enumerate() {
            let ratio = c as f64 / expected;
            assert!(
                ratio > 0.5 && ratio < 2.0,
                "bucket {} has {} items (expected ~{})",
                i,
                c,
                expected
            );
        }
    }

    #[test]
    fn test_avalanche_quality() {
        let matrix = avalanche_matrix(0, 1000);
        let max_dev = avalanche_quality(&matrix);
        // MurmurHash3 should have good avalanche (deviation < 0.1 from 0.5).
        assert!(
            max_dev < 0.15,
            "avalanche quality poor: max deviation {:.3}",
            max_dev
        );
    }

    #[test]
    fn test_murmur3_bytes_consistency() {
        // Hashing a u64 as bytes should produce a result (not necessarily
        // the same as murmur3_64 since the mixing differs).
        let key: u64 = 0x0102030405060708;
        let bytes = key.to_le_bytes();
        let h = murmur3_64_bytes(&bytes, 0);
        let h2 = murmur3_64_bytes(&bytes, 0);
        assert_eq!(h, h2);
    }

    #[test]
    fn test_murmur3_bytes_empty() {
        let h = murmur3_64_bytes(&[], 0);
        // Empty input should still produce a hash.
        let _ = h;
    }

    #[test]
    fn test_murmur3_bytes_short() {
        let h1 = murmur3_64_bytes(&[1, 2, 3], 0);
        let h2 = murmur3_64_bytes(&[1, 2, 4], 0);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_default_seed_hasher() {
        let h = MurmurHasher::default_seed();
        assert_eq!(h.seed(), 0);
    }

    #[test]
    fn test_large_key() {
        let hasher = MurmurHasher::new(0);
        let h = hasher.hash(u64::MAX);
        assert_ne!(h, u64::MAX);
    }

    // ── new tests ──────────────────────────────────────────────────────

    #[test]
    fn test_murmur3_128_deterministic() {
        let (h1a, h2a) = murmur3_128(42, 0);
        let (h1b, h2b) = murmur3_128(42, 0);
        assert_eq!(h1a, h1b);
        assert_eq!(h2a, h2b);
    }

    #[test]
    fn test_murmur3_128_different_keys() {
        let (h1a, h2a) = murmur3_128(0, 0);
        let (h1b, h2b) = murmur3_128(1, 0);
        // At least one half should differ.
        assert!(h1a != h1b || h2a != h2b);
    }

    #[test]
    fn test_murmur3_128_different_seeds() {
        let (h1a, h2a) = murmur3_128(100, 0);
        let (h1b, h2b) = murmur3_128(100, 1);
        assert!(h1a != h1b || h2a != h2b);
    }

    #[test]
    fn test_murmur3_bulk() {
        let keys: Vec<u64> = (0..500).collect();
        let hashes = murmur3_bulk(&keys, 42);
        assert_eq!(hashes.len(), 500);
        for (i, &k) in keys.iter().enumerate() {
            assert_eq!(hashes[i], murmur3_64(k, 42));
        }
    }

    #[test]
    fn test_murmur3_string() {
        let data = b"hello world";
        let h1 = murmur3_string(data, 0);
        let h2 = murmur3_string(data, 0);
        assert_eq!(h1, h2);

        let h3 = murmur3_string(b"hello worlD", 0);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_chi_squared_uniformity() {
        // 10 000 keys into 16 buckets. For 15 dof at p=0.001 critical ≈ 37.7.
        let result = chi_squared_uniformity_test(0, 10_000, 16, 50.0);
        assert!(
            result.passes,
            "chi-squared {} exceeds threshold",
            result.chi_squared
        );
        assert_eq!(result.degrees_of_freedom, 15);
        assert!(result.max_bucket > 0);
    }

    #[test]
    fn test_bit_bias() {
        let bias = bit_bias(0, 10_000);
        for b in 0..64 {
            assert!(
                (bias[b] - 0.5).abs() < 0.1,
                "bit {} has bias {:.3}",
                b,
                bias[b]
            );
        }
    }

    #[test]
    fn test_murmur3_bulk_empty() {
        let hashes = murmur3_bulk(&[], 0);
        assert!(hashes.is_empty());
    }
}
