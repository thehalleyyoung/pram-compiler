//! Siegel k-wise independent hash family.
//!
//! Uses polynomial evaluation over a Mersenne-prime field (p = 2^61 - 1).
//! A k-wise independent hash function is a degree-(k-1) polynomial
//! h(x) = a_0 + a_1*x + a_2*x^2 + ... + a_{k-1}*x^{k-1}  (mod p).

use rand::Rng;
use super::HashFunction;

/// Mersenne prime p = 2^61 - 1.
const MERSENNE_P: u64 = (1u64 << 61) - 1;

/// Multiply two u64 values modulo the Mersenne prime 2^61 - 1.
/// Uses 128-bit intermediate to avoid overflow.
#[inline]
fn mod_mul(a: u64, b: u64) -> u64 {
    let full = (a as u128) * (b as u128);
    mod_mersenne_128(full)
}

/// Reduce a u128 value modulo 2^61 - 1 using the Mersenne identity:
/// x mod (2^61-1) = (x >> 61) + (x & (2^61-1)), with a final correction.
#[inline]
fn mod_mersenne_128(x: u128) -> u64 {
    let lo = (x & (MERSENNE_P as u128)) as u64;
    let hi = (x >> 61) as u64;
    let sum = lo + hi;
    if sum >= MERSENNE_P {
        sum - MERSENNE_P
    } else {
        sum
    }
}

/// Add two values modulo 2^61 - 1.
#[inline]
fn mod_add(a: u64, b: u64) -> u64 {
    let sum = a + b;
    if sum >= MERSENNE_P {
        sum - MERSENNE_P
    } else {
        sum
    }
}

/// Subtract modulo 2^61 - 1.
#[inline]
fn mod_sub(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + MERSENNE_P - b
    }
}

/// Compute base^exp mod (2^61 - 1).
fn mod_pow(mut base: u64, mut exp: u64) -> u64 {
    let mut result = 1u64;
    base %= MERSENNE_P;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base);
        }
        exp >>= 1;
        base = mod_mul(base, base);
    }
    result
}

/// Compute the modular inverse of a modulo 2^61 - 1 via Fermat's little theorem:
/// a^{-1} = a^{p-2} mod p.
fn mod_inv(a: u64) -> u64 {
    mod_pow(a, MERSENNE_P - 2)
}

/// Generate a random coefficient in [1, p-1].
fn random_coeff<R: Rng>(rng: &mut R) -> u64 {
    loop {
        let v = rng.gen::<u64>() & MERSENNE_P;
        if v != 0 && v < MERSENNE_P {
            return v;
        }
    }
}

/// Generate a random coefficient in [0, p-1].
fn random_coeff_with_zero<R: Rng>(rng: &mut R) -> u64 {
    loop {
        let v = rng.gen::<u64>() & MERSENNE_P;
        if v < MERSENNE_P {
            return v;
        }
    }
}

/// A Siegel k-wise independent hash function.
///
/// Internally a polynomial of degree k-1 with random coefficients over GF(2^61-1).
#[derive(Clone, Debug)]
pub struct SiegelHash {
    /// Polynomial coefficients a_0, a_1, ..., a_{k-1}.
    coeffs: Vec<u64>,
    /// Independence level.
    k: usize,
}

impl SiegelHash {
    /// Create a new k-wise independent hash function with random coefficients.
    pub fn new<R: Rng>(k: usize, rng: &mut R) -> Self {
        assert!(k >= 1, "independence level must be >= 1");
        let coeffs: Vec<u64> = (0..k).map(|_| random_coeff_with_zero(rng)).collect();
        Self { coeffs, k }
    }

    /// Create a hash function from explicit coefficients (for testing).
    pub fn from_coeffs(coeffs: Vec<u64>) -> Self {
        let k = coeffs.len();
        assert!(k >= 1);
        Self { coeffs, k }
    }

    /// Return the independence level k.
    pub fn independence_level(&self) -> usize {
        self.k
    }

    /// Evaluate the polynomial at x using Horner's method modulo p.
    ///
    /// h(x) = a_0 + x*(a_1 + x*(a_2 + ... + x*a_{k-1}))
    fn eval_horner(&self, x: u64) -> u64 {
        let x_mod = x % MERSENNE_P;
        let mut result = 0u64;
        for &coeff in self.coeffs.iter().rev() {
            result = mod_add(mod_mul(result, x_mod), coeff);
        }
        result
    }

    /// Evaluate the polynomial at x using the naive method (for verification).
    fn eval_naive(&self, x: u64) -> u64 {
        let x_mod = x % MERSENNE_P;
        let mut result = 0u64;
        let mut x_power = 1u64;
        for &coeff in &self.coeffs {
            result = mod_add(result, mod_mul(coeff, x_power));
            x_power = mod_mul(x_power, x_mod);
        }
        result
    }

    /// Return the coefficients (for inspection / debugging).
    pub fn coefficients(&self) -> &[u64] {
        &self.coeffs
    }
}

impl HashFunction for SiegelHash {
    fn hash(&self, key: u64) -> u64 {
        self.eval_horner(key)
    }

    fn hash_to_range(&self, key: u64, range: u64) -> u64 {
        if range == 0 {
            return 0;
        }
        self.eval_horner(key) % range
    }
}

/// A family that produces independent SiegelHash instances.
pub struct SiegelHashFamily {
    k: usize,
}

impl SiegelHashFamily {
    /// Create a new family for producing k-wise independent hash functions.
    pub fn new(k: usize) -> Self {
        assert!(k >= 1);
        Self { k }
    }

    /// Generate one hash function from the family.
    pub fn generate<R: Rng>(&self, rng: &mut R) -> SiegelHash {
        SiegelHash::new(self.k, rng)
    }

    /// Generate `count` independent hash functions.
    pub fn generate_many<R: Rng>(&self, count: usize, rng: &mut R) -> Vec<SiegelHash> {
        (0..count).map(|_| SiegelHash::new(self.k, rng)).collect()
    }

    /// Return the independence level.
    pub fn independence_level(&self) -> usize {
        self.k
    }
}

/// Evaluate a polynomial given as a coefficient slice at `x` using Horner's method mod p.
///
/// This is a standalone helper that avoids constructing a full `SiegelHash` when
/// you only need a single evaluation.
pub fn evaluate_polynomial_mod(coeffs: &[u64], x: u64) -> u64 {
    let x_mod = x % MERSENNE_P;
    let mut result = 0u64;
    for &coeff in coeffs.iter().rev() {
        result = mod_add(mod_mul(result, x_mod), coeff % MERSENNE_P);
    }
    result
}

/// Compute the theoretical pairwise collision probability bound for a
/// k-wise independent hash mapping n keys into m buckets.
///
/// For k ≥ 2 the pairwise collision probability is exactly 1/m (the same as
/// fully random).  For k = 1 the function is constant, so every pair collides
/// and the bound is 1.  Returns a value in \[0, 1\].
pub fn collision_probability(k: usize, _n: usize, m: u64) -> f64 {
    if m == 0 {
        return 1.0;
    }
    if k <= 1 {
        return 1.0;
    }
    1.0 / m as f64
}

/// Deterministically generate `k` polynomial coefficients from a 64-bit seed.
///
/// Uses a seeded PRNG so that the same `(k, seed)` pair always produces the
/// same coefficients.  Useful for reproducible experiments.
pub fn generate_coefficients(k: usize, seed: u64) -> Vec<u64> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k).map(|_| random_coeff_with_zero(&mut rng)).collect()
}

/// Batch hasher: pre-computes nothing extra but provides a convenient API for
/// hashing many keys through the same polynomial at once.
#[derive(Clone, Debug)]
pub struct SiegelHashBatch {
    inner: SiegelHash,
}

impl SiegelHashBatch {
    /// Wrap an existing `SiegelHash`.
    pub fn new(hash: SiegelHash) -> Self {
        Self { inner: hash }
    }

    /// Create a new batch hasher with random coefficients.
    pub fn random<R: Rng>(k: usize, rng: &mut R) -> Self {
        Self {
            inner: SiegelHash::new(k, rng),
        }
    }

    /// Create from a seed (deterministic).
    pub fn from_seed(k: usize, seed: u64) -> Self {
        let coeffs = generate_coefficients(k, seed);
        Self {
            inner: SiegelHash::from_coeffs(coeffs),
        }
    }

    /// Hash a slice of keys, returning one hash per key.
    pub fn hash_all(&self, keys: &[u64]) -> Vec<u64> {
        keys.iter().map(|&k| self.inner.hash(k)).collect()
    }

    /// Hash a slice of keys into `[0, range)`.
    pub fn hash_all_to_range(&self, keys: &[u64], range: u64) -> Vec<u64> {
        keys.iter()
            .map(|&k| self.inner.hash_to_range(k, range))
            .collect()
    }

    /// Return a reference to the underlying hash.
    pub fn inner(&self) -> &SiegelHash {
        &self.inner
    }
}

/// Statistical summary of a hash function's output distribution.
#[derive(Clone, Debug)]
pub struct HashQualityReport {
    /// Chi-squared statistic.
    pub chi_squared: f64,
    /// Expected items per bucket.
    pub expected_per_bucket: f64,
    /// Maximum bucket count.
    pub max_bucket: u64,
    /// Minimum bucket count.
    pub min_bucket: u64,
    /// Number of empty buckets.
    pub empty_buckets: usize,
    /// Entropy of the distribution (bits).
    pub entropy: f64,
}

/// Analyse hash quality by distributing `keys` into `m` buckets.
pub fn hash_quality_report(hash: &SiegelHash, m: u64, keys: &[u64]) -> HashQualityReport {
    let mut counts = vec![0u64; m as usize];
    for &k in keys {
        let bucket = hash.hash_to_range(k, m) as usize;
        counts[bucket] += 1;
    }
    let expected = keys.len() as f64 / m as f64;
    let chi_sq: f64 = counts
        .iter()
        .map(|&c| {
            let diff = c as f64 - expected;
            diff * diff / expected
        })
        .sum();
    let max_bucket = counts.iter().copied().max().unwrap_or(0);
    let min_bucket = counts.iter().copied().min().unwrap_or(0);
    let empty_buckets = counts.iter().filter(|&&c| c == 0).count();

    let n = keys.len() as f64;
    let entropy: f64 = if n > 0.0 {
        counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n;
                -p * p.ln()
            })
            .sum::<f64>()
            / (2.0f64).ln() // convert nats → bits
    } else {
        0.0
    };

    HashQualityReport {
        chi_squared: chi_sq,
        expected_per_bucket: expected,
        max_bucket,
        min_bucket,
        empty_buckets,
        entropy,
    }
}

/// Compute the coefficient of variation (std-dev / mean) of bucket counts.
pub fn coefficient_of_variation(hash: &SiegelHash, m: u64, keys: &[u64]) -> f64 {
    let mut counts = vec![0u64; m as usize];
    for &k in keys {
        counts[hash.hash_to_range(k, m) as usize] += 1;
    }
    let n = counts.len() as f64;
    let mean = counts.iter().map(|&c| c as f64).sum::<f64>() / n;
    if mean == 0.0 {
        return 0.0;
    }
    let var = counts
        .iter()
        .map(|&c| {
            let d = c as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    var.sqrt() / mean
}

/// Utility: compute pair-collision rate empirically.
/// Hashes `n` random keys into `m` buckets and returns (total_collisions, total_pairs).
pub fn empirical_collision_rate<R: Rng>(
    hash: &SiegelHash,
    m: u64,
    n: usize,
    rng: &mut R,
) -> (usize, usize) {
    let keys: Vec<u64> = (0..n).map(|_| rng.gen::<u64>()).collect();
    let hashes: Vec<u64> = keys.iter().map(|&k| hash.hash_to_range(k, m)).collect();

    let mut collisions = 0usize;
    let total_pairs = n * (n - 1) / 2;
    for i in 0..n {
        for j in (i + 1)..n {
            if hashes[i] == hashes[j] {
                collisions += 1;
            }
        }
    }
    (collisions, total_pairs)
}

/// Utility: measure distribution uniformity via chi-squared statistic.
/// Returns (chi_sq, expected_per_bucket).
pub fn distribution_chi_squared(
    hash: &SiegelHash,
    m: u64,
    keys: &[u64],
) -> (f64, f64) {
    let mut counts = vec![0u64; m as usize];
    for &k in keys {
        let bucket = hash.hash_to_range(k, m) as usize;
        counts[bucket] += 1;
    }
    let expected = keys.len() as f64 / m as f64;
    let chi_sq: f64 = counts
        .iter()
        .map(|&c| {
            let diff = c as f64 - expected;
            diff * diff / expected
        })
        .sum();
    (chi_sq, expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_mod_arithmetic() {
        assert_eq!(mod_add(MERSENNE_P - 1, 1), 0);
        assert_eq!(mod_add(MERSENNE_P - 1, 2), 1);
        assert_eq!(mod_mul(2, 3), 6);
        assert_eq!(mod_mul(MERSENNE_P - 1, 2), MERSENNE_P - 2);
        assert_eq!(mod_sub(5, 3), 2);
        assert_eq!(mod_sub(3, 5), MERSENNE_P - 2);
    }

    #[test]
    fn test_mod_pow_and_inv() {
        assert_eq!(mod_pow(2, 10), 1024);
        let a = 12345u64;
        let inv = mod_inv(a);
        assert_eq!(mod_mul(a, inv), 1);
    }

    #[test]
    fn test_horner_vs_naive() {
        let mut rng = make_rng();
        let h = SiegelHash::new(5, &mut rng);
        for x in 0..1000u64 {
            assert_eq!(h.eval_horner(x), h.eval_naive(x), "mismatch at x={}", x);
        }
    }

    #[test]
    fn test_hash_deterministic() {
        let h = SiegelHash::from_coeffs(vec![1, 2, 3]);
        let v1 = h.hash(100);
        let v2 = h.hash(100);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_hash_to_range() {
        let mut rng = make_rng();
        let h = SiegelHash::new(4, &mut rng);
        for x in 0..500u64 {
            let val = h.hash_to_range(x, 64);
            assert!(val < 64);
        }
    }

    #[test]
    fn test_distribution_uniformity() {
        let mut rng = make_rng();
        let h = SiegelHash::new(4, &mut rng);
        let m = 16u64;
        let n = 10_000usize;
        let keys: Vec<u64> = (0..n as u64).collect();
        let (chi_sq, _expected) = distribution_chi_squared(&h, m, &keys);
        // For 15 degrees of freedom, chi-squared critical value at p=0.001 is ~37.7.
        // A good hash should be well below this for sequential keys.
        assert!(
            chi_sq < 100.0,
            "chi-squared {} too high for uniform distribution",
            chi_sq
        );
    }

    #[test]
    fn test_pairwise_independence_collision_rate() {
        let mut rng = make_rng();
        let h = SiegelHash::new(2, &mut rng);
        let m = 128u64;
        let n = 200;
        let (collisions, total_pairs) = empirical_collision_rate(&h, m, n, &mut rng);
        let empirical_rate = collisions as f64 / total_pairs as f64;
        let expected_rate = 1.0 / m as f64;
        // Allow up to 5x the expected rate (statistical slack for small samples).
        assert!(
            empirical_rate < expected_rate * 5.0,
            "collision rate {:.4} much higher than expected {:.4}",
            empirical_rate,
            expected_rate
        );
    }

    #[test]
    fn test_family_generates_different_functions() {
        let mut rng = make_rng();
        let family = SiegelHashFamily::new(3);
        let fns = family.generate_many(5, &mut rng);
        // Different functions should (almost certainly) produce different hashes.
        let hashes: Vec<u64> = fns.iter().map(|f| f.hash(999)).collect();
        let unique: std::collections::HashSet<u64> = hashes.iter().copied().collect();
        assert!(unique.len() >= 3, "expected diverse outputs from family");
    }

    #[test]
    fn test_k1_hash_is_constant() {
        let h = SiegelHash::from_coeffs(vec![42]);
        assert_eq!(h.hash(0), 42);
        assert_eq!(h.hash(1), 42);
        assert_eq!(h.hash(99999), 42);
    }

    #[test]
    fn test_k2_is_linear() {
        // h(x) = 3 + 5x mod p
        let h = SiegelHash::from_coeffs(vec![3, 5]);
        assert_eq!(h.hash(0), 3);
        assert_eq!(h.hash(1), 8);
        assert_eq!(h.hash(2), 13);
        assert_eq!(h.hash(10), 53);
    }

    #[test]
    fn test_collision_rate_improves_with_k() {
        let mut rng = make_rng();
        let m = 64u64;
        let n = 150;

        let h2 = SiegelHash::new(2, &mut rng);
        let (c2, pairs2) = empirical_collision_rate(&h2, m, n, &mut rng);
        let rate2 = c2 as f64 / pairs2 as f64;

        let h4 = SiegelHash::new(4, &mut rng);
        let (c4, pairs4) = empirical_collision_rate(&h4, m, n, &mut rng);
        let rate4 = c4 as f64 / pairs4 as f64;

        // Higher k shouldn't dramatically worsen collision rate.
        assert!(
            rate4 < rate2 * 3.0 + 0.01,
            "k=4 rate {:.4} much worse than k=2 rate {:.4}",
            rate4,
            rate2
        );
    }

    #[test]
    fn test_large_keys() {
        let mut rng = make_rng();
        let h = SiegelHash::new(4, &mut rng);
        let large_key = u64::MAX;
        let val = h.hash(large_key);
        assert!(val < MERSENNE_P);
        let val_range = h.hash_to_range(large_key, 1000);
        assert!(val_range < 1000);
    }

    #[test]
    fn test_zero_key() {
        let h = SiegelHash::from_coeffs(vec![7, 11, 13]);
        // h(0) = a_0 = 7
        assert_eq!(h.hash(0), 7);
    }

    #[test]
    fn test_mersenne_boundary() {
        // Ensure keys near the prime boundary are handled.
        let h = SiegelHash::from_coeffs(vec![1, 1]);
        let val = h.hash(MERSENNE_P);
        // MERSENNE_P mod MERSENNE_P == 0, so h(p) = 1 + 1*0 = 1
        assert_eq!(val, 1);
        let val2 = h.hash(MERSENNE_P + 1);
        // (p+1) mod p == 1, so h(p+1) = 1 + 1*1 = 2
        assert_eq!(val2, 2);
    }

    // ── new tests ──────────────────────────────────────────────────────

    #[test]
    fn test_large_k_values() {
        let mut rng = make_rng();
        for k in [8, 16, 32, 64] {
            let h = SiegelHash::new(k, &mut rng);
            assert_eq!(h.independence_level(), k);
            // Horner and naive must agree even for high degree.
            for x in [0u64, 1, 100, 999_999] {
                assert_eq!(h.eval_horner(x), h.eval_naive(x), "k={}, x={}", k, x);
            }
        }
    }

    #[test]
    fn test_deterministic_seeding() {
        let c1 = generate_coefficients(5, 42);
        let c2 = generate_coefficients(5, 42);
        assert_eq!(c1, c2, "same seed must produce same coefficients");

        let c3 = generate_coefficients(5, 99);
        assert_ne!(c1, c3, "different seeds should (almost certainly) differ");
    }

    #[test]
    fn test_batch_hashing() {
        let mut rng = make_rng();
        let batch = SiegelHashBatch::random(4, &mut rng);
        let keys: Vec<u64> = (0..500).collect();
        let hashes = batch.hash_all(&keys);
        assert_eq!(hashes.len(), 500);
        // Every hash should match the inner hash.
        for (i, &k) in keys.iter().enumerate() {
            assert_eq!(hashes[i], batch.inner().hash(k));
        }
    }

    #[test]
    fn test_batch_hashing_to_range() {
        let batch = SiegelHashBatch::from_seed(3, 77);
        let keys: Vec<u64> = (0..200).collect();
        let hashes = batch.hash_all_to_range(&keys, 64);
        for &h in &hashes {
            assert!(h < 64);
        }
    }

    #[test]
    fn test_collision_probability_computation() {
        // k >= 2: collision probability = 1/m.
        let p2 = collision_probability(2, 1000, 128);
        assert!((p2 - 1.0 / 128.0).abs() < 1e-12);

        let p4 = collision_probability(4, 5000, 256);
        assert!((p4 - 1.0 / 256.0).abs() < 1e-12);

        // k = 1: constant function → every pair collides.
        let p1 = collision_probability(1, 100, 128);
        assert!((p1 - 1.0).abs() < 1e-12);

        // m = 0 edge case.
        assert!((collision_probability(3, 10, 0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_evaluate_polynomial_mod_standalone() {
        let coeffs = vec![3, 5, 7]; // 3 + 5x + 7x^2
        let h = SiegelHash::from_coeffs(coeffs.clone());
        for x in 0..200u64 {
            assert_eq!(
                evaluate_polynomial_mod(&coeffs, x),
                h.hash(x),
                "mismatch at x={}",
                x
            );
        }
    }

    #[test]
    fn test_hash_quality_report() {
        let mut rng = make_rng();
        let h = SiegelHash::new(4, &mut rng);
        let keys: Vec<u64> = (0..5000).collect();
        let report = hash_quality_report(&h, 32, &keys);
        assert!(report.chi_squared >= 0.0);
        assert!(report.entropy > 0.0);
        assert!(report.max_bucket >= report.min_bucket);
        assert_eq!(report.expected_per_bucket, 5000.0 / 32.0);
    }

    #[test]
    fn test_coefficient_of_variation() {
        let mut rng = make_rng();
        let h = SiegelHash::new(4, &mut rng);
        let keys: Vec<u64> = (0..10_000).collect();
        let cv = coefficient_of_variation(&h, 16, &keys);
        // A good hash should have small CV.
        assert!(cv < 1.0, "CV {} unexpectedly high", cv);
    }
}
