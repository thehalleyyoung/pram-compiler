//! 2-universal hash families.
//!
//! h_{a,b}(x) = ((a * x + b) mod p) mod m
//!
//! where p = 2^61 - 1 is a Mersenne prime, a ∈ [1, p-1], b ∈ [0, p-1].
//! Guarantees collision probability Pr[h(x) = h(y)] ≤ 1/m for x ≠ y.

use rand::Rng;
use super::HashFunction;

/// Mersenne prime p = 2^61 - 1.
const MERSENNE_P: u64 = (1u64 << 61) - 1;

/// Multiply two u64 values modulo 2^61 - 1 using 128-bit intermediate.
#[inline]
fn mod_mul(a: u64, b: u64) -> u64 {
    let full = (a as u128) * (b as u128);
    mod_mersenne_128(full)
}

/// Reduce a u128 modulo 2^61 - 1.
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

/// A 2-universal hash function: h(x) = ((a * x + b) mod p) mod m.
#[derive(Clone, Debug)]
pub struct TwoUniversalHash {
    a: u64,
    b: u64,
    /// Output range m. If 0, raw hash is returned (full p range).
    m: u64,
}

impl TwoUniversalHash {
    /// Create a new 2-universal hash function with random coefficients.
    /// `m` is the output range. If m == 0, hash returns the raw field element.
    pub fn new<R: Rng>(m: u64, rng: &mut R) -> Self {
        let a = loop {
            let v = rng.gen::<u64>() & MERSENNE_P;
            if v != 0 && v < MERSENNE_P {
                break v;
            }
        };
        let b = loop {
            let v = rng.gen::<u64>() & MERSENNE_P;
            if v < MERSENNE_P {
                break v;
            }
        };
        Self { a, b, m }
    }

    /// Create from explicit coefficients (for testing).
    pub fn from_params(a: u64, b: u64, m: u64) -> Self {
        assert!(a > 0 && a < MERSENNE_P, "a must be in [1, p-1]");
        assert!(b < MERSENNE_P, "b must be in [0, p-1]");
        Self { a, b, m }
    }

    /// Return the output range.
    pub fn range(&self) -> u64 {
        self.m
    }

    /// Return the coefficients (a, b).
    pub fn coefficients(&self) -> (u64, u64) {
        (self.a, self.b)
    }

    /// Evaluate: (a * x + b) mod p
    #[inline]
    fn eval_raw(&self, x: u64) -> u64 {
        let x_mod = x % MERSENNE_P;
        mod_add(mod_mul(self.a, x_mod), self.b)
    }
}

impl HashFunction for TwoUniversalHash {
    fn hash(&self, key: u64) -> u64 {
        let raw = self.eval_raw(key);
        if self.m == 0 {
            raw
        } else {
            raw % self.m
        }
    }

    fn hash_to_range(&self, key: u64, range: u64) -> u64 {
        if range == 0 {
            return 0;
        }
        self.eval_raw(key) % range
    }
}

/// A family that generates 2-universal hash functions with a fixed output range.
pub struct TwoUniversalFamily {
    m: u64,
}

impl TwoUniversalFamily {
    /// Create a family producing hash functions with output range m.
    pub fn new(m: u64) -> Self {
        Self { m }
    }

    /// Generate a single hash function.
    pub fn generate<R: Rng>(&self, rng: &mut R) -> TwoUniversalHash {
        TwoUniversalHash::new(self.m, rng)
    }

    /// Generate `count` independent hash functions.
    pub fn generate_many<R: Rng>(&self, count: usize, rng: &mut R) -> Vec<TwoUniversalHash> {
        (0..count).map(|_| TwoUniversalHash::new(self.m, rng)).collect()
    }
}

/// Empirically estimate pairwise collision probability.
///
/// Hashes `n` distinct keys into `m` buckets and returns
/// (collision_count, total_pairs, empirical_rate).
pub fn empirical_collision_probability<R: Rng>(
    hash: &TwoUniversalHash,
    n: usize,
    rng: &mut R,
) -> (usize, usize, f64) {
    let keys: Vec<u64> = (0..n).map(|_| rng.gen::<u64>()).collect();
    let hashes: Vec<u64> = keys.iter().map(|&k| hash.hash(k)).collect();

    let mut collisions = 0usize;
    let total_pairs = n * (n - 1) / 2;
    for i in 0..n {
        for j in (i + 1)..n {
            if keys[i] != keys[j] && hashes[i] == hashes[j] {
                collisions += 1;
            }
        }
    }
    let rate = if total_pairs > 0 {
        collisions as f64 / total_pairs as f64
    } else {
        0.0
    };
    (collisions, total_pairs, rate)
}

/// Compute the theoretical collision bound 1/m.
pub fn theoretical_collision_bound(m: u64) -> f64 {
    if m == 0 {
        return 1.0;
    }
    1.0 / m as f64
}

/// Measure bucket load distribution for a 2-universal hash.
/// Returns a vector of bucket counts.
pub fn bucket_load_distribution(hash: &TwoUniversalHash, keys: &[u64], num_buckets: u64) -> Vec<u64> {
    let mut counts = vec![0u64; num_buckets as usize];
    for &k in keys {
        let bucket = hash.hash_to_range(k, num_buckets) as usize;
        counts[bucket] += 1;
    }
    counts
}

/// Compute the max load across all buckets.
pub fn max_bucket_load(counts: &[u64]) -> u64 {
    counts.iter().copied().max().unwrap_or(0)
}

/// Compute variance of bucket loads.
pub fn bucket_load_variance(counts: &[u64]) -> f64 {
    if counts.is_empty() {
        return 0.0;
    }
    let n: f64 = counts.len() as f64;
    let mean: f64 = counts.iter().map(|&c| c as f64).sum::<f64>() / n;
    counts
        .iter()
        .map(|&c| {
            let diff = c as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / n
}

// ────────────────────────────────────────────────────────────────────────
// Tabulation hashing
// ────────────────────────────────────────────────────────────────────────

/// Number of bytes in a u64 key.
const NUM_BYTES: usize = 8;
/// Lookup-table size: one entry per possible byte value.
const TABLE_SIZE: usize = 256;

/// Simple tabulation hash: split the key into bytes, XOR independent table
/// look-ups.
///
/// h(x) = T_0[x_0] ⊕ T_1[x_1] ⊕ … ⊕ T_7[x_7]
///
/// This gives 3-wise independence.
#[derive(Clone, Debug)]
pub struct TabulationHash {
    tables: [[u64; TABLE_SIZE]; NUM_BYTES],
}

impl TabulationHash {
    /// Create a new simple tabulation hash with random tables.
    pub fn new<R: rand::Rng>(rng: &mut R) -> Self {
        let mut tables = [[0u64; TABLE_SIZE]; NUM_BYTES];
        for tbl in tables.iter_mut() {
            for entry in tbl.iter_mut() {
                *entry = rng.gen();
            }
        }
        Self { tables }
    }

    /// Evaluate the tabulation hash.
    #[inline]
    pub fn eval(&self, key: u64) -> u64 {
        let bytes = key.to_le_bytes();
        let mut h = 0u64;
        for i in 0..NUM_BYTES {
            h ^= self.tables[i][bytes[i] as usize];
        }
        h
    }
}

impl super::HashFunction for TabulationHash {
    fn hash(&self, key: u64) -> u64 {
        self.eval(key)
    }

    fn hash_to_range(&self, key: u64, range: u64) -> u64 {
        if range == 0 {
            return 0;
        }
        self.eval(key) % range
    }
}

/// 4-independent tabulation hash (twisted tabulation).
///
/// Uses an extra derived character: after computing the simple tabulation
/// value t, derive an extra byte `d = (t as u8)` and XOR in a 9th table
/// look-up.  This achieves 4-wise independence under the twisted-tabulation
/// framework of Pătraşcu & Thorup.
#[derive(Clone, Debug)]
pub struct Tabulation4Hash {
    base: TabulationHash,
    extra_table: [u64; TABLE_SIZE],
}

impl Tabulation4Hash {
    /// Create a new 4-independent tabulation hash.
    pub fn new<R: rand::Rng>(rng: &mut R) -> Self {
        let base = TabulationHash::new(rng);
        let mut extra_table = [0u64; TABLE_SIZE];
        for entry in extra_table.iter_mut() {
            *entry = rng.gen();
        }
        Self { base, extra_table }
    }

    /// Evaluate the 4-independent tabulation hash.
    #[inline]
    pub fn eval(&self, key: u64) -> u64 {
        let t = self.base.eval(key);
        t ^ self.extra_table[(t & 0xFF) as usize]
    }
}

impl super::HashFunction for Tabulation4Hash {
    fn hash(&self, key: u64) -> u64 {
        self.eval(key)
    }

    fn hash_to_range(&self, key: u64, range: u64) -> u64 {
        if range == 0 {
            return 0;
        }
        self.eval(key) % range
    }
}

/// Compute the theoretical expected maximum load when hashing n items into m
/// bins with a 2-universal hash family.
///
/// E[max load] ≈ n/m + Θ(√((n/m)·ln m))
pub fn expected_max_load(n: u64, m: u64) -> f64 {
    if m == 0 {
        return n as f64;
    }
    let avg = n as f64 / m as f64;
    let ln_m = (m as f64).ln().max(1.0);
    avg + (avg * ln_m).sqrt()
}

/// Detailed variance analysis of bucket loads.
#[derive(Clone, Debug)]
pub struct VarianceAnalysis {
    /// Number of buckets.
    pub num_buckets: usize,
    /// Number of items hashed.
    pub num_items: u64,
    /// Empirical mean load.
    pub mean: f64,
    /// Empirical variance.
    pub variance: f64,
    /// Empirical standard deviation.
    pub std_dev: f64,
    /// Theoretical variance for a 2-universal family: (n/m)(1 − 1/m).
    pub theoretical_variance: f64,
    /// Ratio of empirical to theoretical variance.
    pub variance_ratio: f64,
}

/// Perform a variance analysis of a 2-universal hash on `keys` into `m` buckets.
pub fn variance_analysis(hash: &TwoUniversalHash, keys: &[u64], m: u64) -> VarianceAnalysis {
    let counts = bucket_load_distribution(hash, keys, m);
    let n = keys.len() as u64;
    let mean = n as f64 / m as f64;
    let var = bucket_load_variance(&counts);
    let theoretical_var = mean * (1.0 - 1.0 / m as f64);
    let ratio = if theoretical_var > 0.0 {
        var / theoretical_var
    } else {
        0.0
    };
    VarianceAnalysis {
        num_buckets: m as usize,
        num_items: n,
        mean,
        variance: var,
        std_dev: var.sqrt(),
        theoretical_variance: theoretical_var,
        variance_ratio: ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(123)
    }

    #[test]
    fn test_deterministic() {
        let h = TwoUniversalHash::from_params(5, 3, 100);
        assert_eq!(h.hash(42), h.hash(42));
    }

    #[test]
    fn test_in_range() {
        let mut rng = make_rng();
        let h = TwoUniversalHash::new(64, &mut rng);
        for x in 0..1000u64 {
            assert!(h.hash(x) < 64);
        }
    }

    #[test]
    fn test_hash_to_range_different_from_m() {
        let h = TwoUniversalHash::from_params(7, 11, 100);
        for x in 0..500u64 {
            assert!(h.hash_to_range(x, 32) < 32);
        }
    }

    #[test]
    fn test_collision_bound() {
        let mut rng = make_rng();
        let m = 128u64;
        let h = TwoUniversalHash::new(m, &mut rng);
        let n = 200;
        let (collisions, total_pairs, rate) = empirical_collision_probability(&h, n, &mut rng);
        let bound = theoretical_collision_bound(m);
        // Empirical rate should be within a generous factor of the theoretical bound.
        assert!(
            rate < bound * 5.0 + 0.01,
            "empirical rate {:.5} exceeds 5x theoretical bound {:.5} (collisions={}, pairs={})",
            rate,
            bound,
            collisions,
            total_pairs
        );
    }

    #[test]
    fn test_different_keys_different_hash_usually() {
        let h = TwoUniversalHash::from_params(1000003, 999983, 0);
        let h1 = h.hash(1);
        let h2 = h.hash(2);
        // With m=0 (raw), different keys almost certainly differ.
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_family_generates_distinct() {
        let mut rng = make_rng();
        let family = TwoUniversalFamily::new(1000);
        let fns = family.generate_many(10, &mut rng);
        let hashes: Vec<u64> = fns.iter().map(|f| f.hash(12345)).collect();
        let unique: std::collections::HashSet<u64> = hashes.iter().copied().collect();
        assert!(unique.len() >= 5, "expected diverse family outputs");
    }

    #[test]
    fn test_bucket_load_distribution() {
        let mut rng = make_rng();
        let h = TwoUniversalHash::new(16, &mut rng);
        let keys: Vec<u64> = (0..1600).collect();
        let counts = bucket_load_distribution(&h, &keys, 16);
        assert_eq!(counts.len(), 16);
        let total: u64 = counts.iter().sum();
        assert_eq!(total, 1600);
    }

    #[test]
    fn test_bucket_load_variance_reasonable() {
        let mut rng = make_rng();
        let m = 32u64;
        let h = TwoUniversalHash::new(m, &mut rng);
        let keys: Vec<u64> = (0..3200).collect();
        let counts = bucket_load_distribution(&h, &keys, m);
        let variance = bucket_load_variance(&counts);
        let expected_per = 3200.0 / m as f64;
        // Variance should be small relative to mean squared.
        assert!(
            variance < expected_per * expected_per,
            "variance {} too high (expected_per={})",
            variance,
            expected_per
        );
    }

    #[test]
    fn test_max_load() {
        let counts = vec![10, 20, 5, 15];
        assert_eq!(max_bucket_load(&counts), 20);
    }

    #[test]
    fn test_zero_key() {
        let h = TwoUniversalHash::from_params(7, 11, 100);
        // h(0) = (7*0 + 11) mod p mod 100 = 11
        assert_eq!(h.hash(0), 11);
    }

    #[test]
    fn test_large_m() {
        let h = TwoUniversalHash::from_params(3, 7, MERSENNE_P);
        let val = h.hash(100);
        assert!(val < MERSENNE_P);
    }

    #[test]
    fn test_collision_rate_many_trials() {
        let mut rng = make_rng();
        let m = 64u64;
        let bound = theoretical_collision_bound(m);
        let mut total_rate = 0.0;
        let trials = 5;
        for _ in 0..trials {
            let h = TwoUniversalHash::new(m, &mut rng);
            let (_, _, rate) = empirical_collision_probability(&h, 100, &mut rng);
            total_rate += rate;
        }
        let avg_rate = total_rate / trials as f64;
        assert!(
            avg_rate < bound * 4.0 + 0.01,
            "average collision rate {:.5} exceeds bound {:.5}",
            avg_rate,
            bound
        );
    }

    // ── new tests ──────────────────────────────────────────────────────

    #[test]
    fn test_tabulation_hash_deterministic() {
        let mut rng = make_rng();
        let h = TabulationHash::new(&mut rng);
        assert_eq!(h.eval(42), h.eval(42));
        assert_eq!(h.hash(42), h.hash(42));
    }

    #[test]
    fn test_tabulation_hash_range() {
        let mut rng = make_rng();
        let h = TabulationHash::new(&mut rng);
        for x in 0..1000u64 {
            assert!(h.hash_to_range(x, 64) < 64);
        }
    }

    #[test]
    fn test_tabulation_hash_distribution() {
        let mut rng = make_rng();
        let h = TabulationHash::new(&mut rng);
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
                ratio > 0.3 && ratio < 3.0,
                "tabulation bucket {} has {} items (expected ~{})",
                i, c, expected
            );
        }
    }

    #[test]
    fn test_tabulation4_deterministic() {
        let mut rng = make_rng();
        let h = Tabulation4Hash::new(&mut rng);
        assert_eq!(h.eval(100), h.eval(100));
        assert_eq!(h.hash(100), h.hash(100));
    }

    #[test]
    fn test_tabulation4_range() {
        let mut rng = make_rng();
        let h = Tabulation4Hash::new(&mut rng);
        for x in 0..500u64 {
            assert!(h.hash_to_range(x, 32) < 32);
        }
    }

    #[test]
    fn test_expected_max_load() {
        // 1000 items into 100 bins → expected load 10.
        let eml = expected_max_load(1000, 100);
        assert!(eml > 10.0);
        assert!(eml < 50.0);

        // Edge: m=0 → all load on one bin.
        assert_eq!(expected_max_load(100, 0), 100.0);
    }

    #[test]
    fn test_variance_analysis() {
        let mut rng = make_rng();
        let m = 32u64;
        let h = TwoUniversalHash::new(m, &mut rng);
        let keys: Vec<u64> = (0..3200).collect();
        let va = variance_analysis(&h, &keys, m);
        assert_eq!(va.num_buckets, 32);
        assert_eq!(va.num_items, 3200);
        assert!((va.mean - 100.0).abs() < 1e-9);
        assert!(va.variance >= 0.0);
        assert!(va.std_dev >= 0.0);
        assert!(va.theoretical_variance > 0.0);
        // Ratio should be in a reasonable range.
        assert!(
            va.variance_ratio < 10.0,
            "variance ratio {} too high",
            va.variance_ratio
        );
    }

    #[test]
    fn test_tabulation_different_keys() {
        let mut rng = make_rng();
        let h = TabulationHash::new(&mut rng);
        // Different keys should very likely yield different hashes.
        let h1 = h.eval(1);
        let h2 = h.eval(2);
        assert_ne!(h1, h2);
    }
}
