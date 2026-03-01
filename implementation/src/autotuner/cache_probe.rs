//! Hardware cache hierarchy detection and probing.
//!
//! Detects L1/L2/L3 cache sizes, line sizes, and associativity
//! by probing memory access latencies and parsing system info.

use std::time::Instant;

/// Detected cache level parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheLevelInfo {
    pub level: usize,
    pub size_bytes: usize,
    pub line_size: usize,
    pub associativity: usize,
    pub latency_ns: f64,
}

/// Complete cache hierarchy description.
#[derive(Debug, Clone)]
pub struct CacheHierarchy {
    pub levels: Vec<CacheLevelInfo>,
    pub page_size: usize,
    pub detection_method: DetectionMethod,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DetectionMethod {
    /// Parsed from sysctl (macOS) or /proc/cpuinfo (Linux)
    SystemInfo,
    /// Measured via latency probing
    LatencyProbe,
    /// Defaults based on common hardware
    Default,
}

impl CacheHierarchy {
    /// Detect cache hierarchy on the current platform.
    pub fn detect() -> Self {
        // Try system info first
        if let Some(h) = Self::detect_from_system() {
            return h;
        }
        // Fall back to latency probing
        if let Some(h) = Self::detect_from_probing() {
            return h;
        }
        // Use sensible defaults
        Self::default_hierarchy()
    }

    /// Get the L1 data cache info.
    pub fn l1d(&self) -> &CacheLevelInfo {
        self.levels.first().expect("at least L1 should exist")
    }

    /// Get cache level by number (1-indexed).
    pub fn level(&self, n: usize) -> Option<&CacheLevelInfo> {
        self.levels.iter().find(|l| l.level == n)
    }

    /// Total cache capacity across all levels.
    pub fn total_capacity(&self) -> usize {
        self.levels.iter().map(|l| l.size_bytes).sum()
    }

    /// Optimal tile size for a given cache level, targeting fraction of capacity.
    pub fn optimal_tile_size(&self, level: usize, element_size: usize, fraction: f64) -> usize {
        let cache = self.level(level).unwrap_or(self.l1d());
        let usable = (cache.size_bytes as f64 * fraction) as usize;
        let elements = usable / element_size;
        // Round down to power of 2 for efficient loop tiling
        if elements == 0 { 1 } else { elements.next_power_of_two() / 2 }
    }

    /// Optimal block size for hash partition based on cache line size.
    pub fn optimal_block_size(&self) -> usize {
        self.l1d().line_size
    }

    /// Number of cache blocks that fit in L1.
    pub fn l1_block_count(&self) -> usize {
        let l1 = self.l1d();
        l1.size_bytes / l1.line_size
    }

    /// Number of cache blocks that fit in L2.
    pub fn l2_block_count(&self) -> Option<usize> {
        self.level(2).map(|l2| l2.size_bytes / l2.line_size)
    }

    #[cfg(target_os = "macos")]
    fn detect_from_system() -> Option<Self> {
        use std::process::Command;

        let sysctl = |key: &str| -> Option<usize> {
            let out = Command::new("sysctl").arg("-n").arg(key).output().ok()?;
            let s = String::from_utf8_lossy(&out.stdout);
            s.trim().parse().ok()
        };

        let l1d_size = sysctl("hw.l1dcachesize")?;
        let l1_line = sysctl("hw.cachelinesize").unwrap_or(64);
        let l2_size = sysctl("hw.l2cachesize").unwrap_or(256 * 1024);
        let page_size = sysctl("hw.pagesize").unwrap_or(4096);

        // L3 may not exist on all Macs
        let l3_size = sysctl("hw.l3cachesize");

        let mut levels = vec![
            CacheLevelInfo {
                level: 1,
                size_bytes: l1d_size,
                line_size: l1_line,
                associativity: 8,
                latency_ns: 1.0,
            },
            CacheLevelInfo {
                level: 2,
                size_bytes: l2_size,
                line_size: l1_line,
                associativity: 8,
                latency_ns: 4.0,
            },
        ];

        if let Some(l3) = l3_size {
            levels.push(CacheLevelInfo {
                level: 3,
                size_bytes: l3,
                line_size: l1_line,
                associativity: 16,
                latency_ns: 12.0,
            });
        }

        Some(CacheHierarchy {
            levels,
            page_size,
            detection_method: DetectionMethod::SystemInfo,
        })
    }

    #[cfg(target_os = "linux")]
    fn detect_from_system() -> Option<Self> {
        use std::fs;

        let read_cache_info = |level: usize, field: &str| -> Option<String> {
            let path = format!(
                "/sys/devices/system/cpu/cpu0/cache/index{}/{}",
                level, field
            );
            fs::read_to_string(&path).ok().map(|s| s.trim().to_string())
        };

        let parse_size = |s: &str| -> Option<usize> {
            if s.ends_with('K') {
                s[..s.len() - 1].parse::<usize>().ok().map(|v| v * 1024)
            } else if s.ends_with('M') {
                s[..s.len() - 1].parse::<usize>().ok().map(|v| v * 1024 * 1024)
            } else {
                s.parse().ok()
            }
        };

        let mut levels = Vec::new();
        // index0 = L1 instruction, index1 = L1 data, index2 = L2, index3 = L3
        for (idx, level) in [(1, 1), (2, 2), (3, 3)] {
            if let Some(size_str) = read_cache_info(idx, "size") {
                if let Some(size) = parse_size(&size_str) {
                    let line_size = read_cache_info(idx, "coherency_line_size")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(64);
                    let assoc = read_cache_info(idx, "ways_of_associativity")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(8);
                    levels.push(CacheLevelInfo {
                        level,
                        size_bytes: size,
                        line_size,
                        associativity: assoc,
                        latency_ns: match level {
                            1 => 1.0,
                            2 => 4.0,
                            3 => 12.0,
                            _ => 40.0,
                        },
                    });
                }
            }
        }

        if levels.is_empty() {
            return None;
        }

        Some(CacheHierarchy {
            levels,
            page_size: 4096,
            detection_method: DetectionMethod::SystemInfo,
        })
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    fn detect_from_system() -> Option<Self> {
        None
    }

    /// Probe cache hierarchy via memory access latency measurement.
    fn detect_from_probing() -> Option<Self> {
        let sizes_to_test: Vec<usize> = vec![
            8 * 1024,       // 8 KB
            16 * 1024,      // 16 KB
            32 * 1024,      // 32 KB
            64 * 1024,      // 64 KB
            128 * 1024,     // 128 KB
            256 * 1024,     // 256 KB
            512 * 1024,     // 512 KB
            1024 * 1024,    // 1 MB
            2 * 1024 * 1024,  // 2 MB
            4 * 1024 * 1024,  // 4 MB
            8 * 1024 * 1024,  // 8 MB
            16 * 1024 * 1024, // 16 MB
            32 * 1024 * 1024, // 32 MB
        ];

        let mut latencies = Vec::new();
        for &size in &sizes_to_test {
            let lat = measure_access_latency(size, 1_000_000);
            latencies.push((size, lat));
        }

        // Find jumps in latency to identify cache boundaries
        let mut boundaries = Vec::new();
        for i in 1..latencies.len() {
            let ratio = latencies[i].1 / latencies[i - 1].1;
            if ratio > 1.5 {
                boundaries.push(latencies[i - 1].0);
            }
        }

        if boundaries.is_empty() {
            return None;
        }

        let mut levels = Vec::new();
        for (i, &boundary) in boundaries.iter().enumerate() {
            levels.push(CacheLevelInfo {
                level: i + 1,
                size_bytes: boundary,
                line_size: 64,
                associativity: 8,
                latency_ns: latencies.iter()
                    .find(|(s, _)| *s <= boundary)
                    .map(|(_, l)| *l)
                    .unwrap_or(1.0),
            });
        }

        Some(CacheHierarchy {
            levels,
            page_size: 4096,
            detection_method: DetectionMethod::LatencyProbe,
        })
    }

    /// Sensible defaults for modern x86-64 CPUs.
    pub fn default_hierarchy() -> Self {
        CacheHierarchy {
            levels: vec![
                CacheLevelInfo {
                    level: 1,
                    size_bytes: 32 * 1024,     // 32 KB
                    line_size: 64,
                    associativity: 8,
                    latency_ns: 1.0,
                },
                CacheLevelInfo {
                    level: 2,
                    size_bytes: 256 * 1024,    // 256 KB
                    line_size: 64,
                    associativity: 8,
                    latency_ns: 4.0,
                },
                CacheLevelInfo {
                    level: 3,
                    size_bytes: 8 * 1024 * 1024, // 8 MB
                    line_size: 64,
                    associativity: 16,
                    latency_ns: 12.0,
                },
            ],
            page_size: 4096,
            detection_method: DetectionMethod::Default,
        }
    }
}

/// Measure average memory access latency for a given working set size.
fn measure_access_latency(working_set_bytes: usize, iterations: usize) -> f64 {
    let n = working_set_bytes / std::mem::size_of::<usize>();
    if n < 2 {
        return 0.0;
    }

    // Create pointer-chasing chain
    let mut chain: Vec<usize> = (0..n).collect();
    // Fisher-Yates shuffle for random access pattern
    let mut rng_state: u64 = 0xdeadbeef12345678;
    for i in (1..n).rev() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (rng_state >> 33) as usize % (i + 1);
        chain.swap(i, j);
    }

    // Warmup
    let mut idx = 0;
    for _ in 0..iterations / 10 {
        idx = chain[idx % n];
    }

    // Measure
    let start = Instant::now();
    for _ in 0..iterations {
        idx = chain[idx % n];
    }
    let elapsed = start.elapsed();

    // Prevent optimization
    if idx == usize::MAX { println!("{}", idx); }

    elapsed.as_nanos() as f64 / iterations as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_hierarchy() {
        let h = CacheHierarchy::detect();
        assert!(!h.levels.is_empty());
        assert!(h.l1d().size_bytes > 0);
        assert!(h.l1d().line_size >= 16);
    }

    #[test]
    fn test_default_hierarchy() {
        let h = CacheHierarchy::default_hierarchy();
        assert_eq!(h.levels.len(), 3);
        assert_eq!(h.l1d().size_bytes, 32 * 1024);
        assert_eq!(h.l1d().line_size, 64);
    }

    #[test]
    fn test_optimal_tile_size() {
        let h = CacheHierarchy::default_hierarchy();
        let tile = h.optimal_tile_size(1, 8, 0.5);
        assert!(tile > 0);
        assert!(tile <= h.l1d().size_bytes / 8);
    }

    #[test]
    fn test_optimal_block_size() {
        let h = CacheHierarchy::default_hierarchy();
        assert_eq!(h.optimal_block_size(), 64);
    }

    #[test]
    fn test_l1_block_count() {
        let h = CacheHierarchy::default_hierarchy();
        assert_eq!(h.l1_block_count(), 512); // 32KB / 64
    }

    #[test]
    fn test_total_capacity() {
        let h = CacheHierarchy::default_hierarchy();
        assert_eq!(
            h.total_capacity(),
            32 * 1024 + 256 * 1024 + 8 * 1024 * 1024
        );
    }

    #[test]
    fn test_measure_latency() {
        let lat = measure_access_latency(32 * 1024, 100_000);
        assert!(lat > 0.0);
    }
}
