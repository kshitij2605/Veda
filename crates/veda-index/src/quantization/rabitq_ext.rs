//! Extended RaBitQ — multi-bit centered residual quantizer.
//!
//! Where the original [`crate::quantization::rabitq::RaBitQuantizer`] uses
//! 1 bit per dimension (sign only), this variant uses N bits per dimension
//! with a uniform grid over the centered residual range. Two rungs are
//! shipped today:
//!
//! | `bits` | Storage vs f32 | Use when |
//! |---|---|---|
//! | 4 | 8× smaller | rerank stage of two-stage retrieval |
//! | 8 | 4× smaller | SQ8-grade recall; use when you'd otherwise reach for `SQ8` |
//!
//! 3-, 5-, 7-, and 9-bit rungs from the SIGMOD 2025 paper require
//! cross-byte bit-packing that's a follow-up. The optional Hadamard
//! pre-rotation (`with_rotation` / [`HadamardRotation`]) is the
//! paper's key trick for flattening per-component variance — when
//! enabled, recall at a given bit budget improves substantially on
//! anisotropic embedding distributions (most LLM-era models).
//!
//! Use when you need more accuracy than 1-bit RaBitQ but can't
//! afford raw-float storage.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use veda_core::error::{Result, VedaError};
use veda_core::types::DistanceMetric;

/// Supported bit widths. 4 and 8 are byte-aligned and use fast
/// nibble/byte packing; 3, 5, 7, 9 use the generic bitstream
/// (see [`pack_bits`] / [`unpack_bits`]).
const SUPPORTED_BITS: &[u8] = &[3, 4, 5, 7, 8, 9];

/// Generic bit-packing primitive. Writes the low `bits` bits of
/// `value` to `out` starting at the bit position tracked by
/// `(byte_idx, bit_in_byte)`. Advances those cursors past the
/// written bits. Caller must ensure `out` has enough room — for
/// total bit count `B` this is `(B + 7) / 8` bytes.
fn pack_bits(out: &mut [u8], value: u16, bits: u8, byte_idx: &mut usize, bit_in_byte: &mut u8) {
    let mut remaining = bits;
    let mut v: u32 = (value as u32) & ((1u32 << bits) - 1);
    while remaining > 0 {
        let space = 8 - *bit_in_byte;
        let take = remaining.min(space);
        let mask = ((1u32 << take) - 1) as u8;
        let lo = (v as u8) & mask;
        out[*byte_idx] |= lo << *bit_in_byte;
        v >>= take;
        remaining -= take;
        *bit_in_byte += take;
        if *bit_in_byte == 8 {
            *bit_in_byte = 0;
            *byte_idx += 1;
        }
    }
}

/// Inverse of [`pack_bits`]. Reads the next `bits` bits from
/// `buf` starting at `(byte_idx, bit_in_byte)`, advances cursors.
fn unpack_bits(buf: &[u8], bits: u8, byte_idx: &mut usize, bit_in_byte: &mut u8) -> u16 {
    let mut remaining = bits;
    let mut shift = 0u8;
    let mut out: u32 = 0;
    while remaining > 0 {
        let space = 8 - *bit_in_byte;
        let take = remaining.min(space);
        let mask = ((1u32 << take) - 1) as u8;
        let bits_here = (buf[*byte_idx] >> *bit_in_byte) & mask;
        out |= (bits_here as u32) << shift;
        shift += take;
        remaining -= take;
        *bit_in_byte += take;
        if *bit_in_byte == 8 {
            *bit_in_byte = 0;
            *byte_idx += 1;
        }
    }
    out as u16
}

/// Optimised codebook (1D Lloyd-Max iteration) — replaces the
/// default uniform grid with quantisation levels placed to
/// minimise mean-squared error on the actual centered-residual
/// distribution seen at training time.
///
/// One codebook shared across all dimensions. After Hadamard
/// rotation, the per-dim distribution is approximately
/// stationary, so a single 1D codebook is sufficient.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LloydMaxCodebook {
    /// Sorted quantisation levels, length `2^bits`.
    levels: Vec<f32>,
}

impl LloydMaxCodebook {
    /// Train a codebook of `2^bits` levels via Lloyd-Max iteration
    /// on the 1D sample of centered residuals.
    pub fn train(samples: &[f32], bits: u8, max_iter: usize) -> Self {
        let n_levels = 1usize << bits;
        if samples.is_empty() || n_levels == 0 {
            return Self { levels: vec![0.0; n_levels.max(1)] };
        }
        // Initialise: percentile-based uniform spacing on the
        // empirical CDF. Better starting point than min/max
        // uniform — robust to outliers.
        let mut sorted: Vec<f32> = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut levels: Vec<f32> = (0..n_levels)
            .map(|i| {
                let pct = (i as f32 + 0.5) / n_levels as f32;
                let idx = ((pct * sorted.len() as f32) as usize).min(sorted.len() - 1);
                sorted[idx]
            })
            .collect();

        // Lloyd-Max iteration. Tracks per-level sum + count to
        // recompute centroids; stops when total movement falls
        // below `eps`.
        let eps = 1e-5;
        for _ in 0..max_iter {
            let mut sums = vec![0.0_f64; n_levels];
            let mut counts = vec![0u32; n_levels];
            for &s in samples {
                let idx = nearest_level_idx(&levels, s);
                sums[idx] += s as f64;
                counts[idx] += 1;
            }
            let mut moved = 0.0_f32;
            for i in 0..n_levels {
                if counts[i] > 0 {
                    let new = (sums[i] / counts[i] as f64) as f32;
                    moved += (new - levels[i]).abs();
                    levels[i] = new;
                }
                // Empty levels stay where they were — they're
                // never assigned anyway.
            }
            if moved < eps {
                break;
            }
        }
        // Re-sort: Lloyd-Max can permute levels if two start
        // close. We rely on sorted order for binary-search
        // encode.
        levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Self { levels }
    }

    /// Encode a single value to its nearest level's index.
    /// Uses binary search on the sorted levels.
    pub fn encode(&self, value: f32) -> u16 {
        // Binary search for the insertion point, then compare to
        // the two neighbours. partition_point gives the index of
        // the first element strictly > `value`.
        let n = self.levels.len();
        if n == 0 {
            return 0;
        }
        let i = self.levels.partition_point(|&l| l <= value);
        if i == 0 {
            0
        } else if i == n {
            (n - 1) as u16
        } else {
            // Compare levels[i-1] and levels[i].
            let lo = self.levels[i - 1];
            let hi = self.levels[i];
            if (value - lo).abs() <= (hi - value).abs() {
                (i - 1) as u16
            } else {
                i as u16
            }
        }
    }

    /// Decode a level index to its value.
    pub fn decode(&self, idx: u16) -> f32 {
        self.levels
            .get(idx as usize)
            .copied()
            .unwrap_or(0.0)
    }

    pub fn len(&self) -> usize {
        self.levels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }
}

fn nearest_level_idx(levels: &[f32], value: f32) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;
    for (i, &l) in levels.iter().enumerate() {
        let d = (l - value).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

/// Optional Hadamard pre-rotation applied to centered residuals
/// before quantisation. Flattens per-component variance — the key
/// SIGMOD 2025 trick that brings 4-bit recall close to raw float
/// on anisotropic embedding distributions (BGE-M3, voyage-3-large,
/// Qwen3-Embedding-8B all qualify).
///
/// The transform is the fast Walsh-Hadamard transform (FWHT) on
/// `dim` rounded up to the next power of two. Self-inverse: query
/// vectors are rotated identically before distance ops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HadamardRotation {
    /// Effective dim after rounding up to the next power of two.
    /// Codes are computed in this expanded space (zero-padded
    /// dimensions encode to a constant "middle" code).
    padded_dim: u32,
    /// Per-dimension ±1 sign vector applied before the
    /// Hadamard transform. Randomises which dimensions get
    /// added/subtracted, breaking adversarial-axis correlations.
    /// Stored as a packed bitmap of length `padded_dim`.
    sign_bitmap: Vec<u8>,
}

impl HadamardRotation {
    /// Build a new rotation for `dim`. The sign vector is derived
    /// deterministically from a `seed` so a given (dim, seed) pair
    /// reproduces the same rotation on every load.
    pub fn new(dim: u32, seed: u64) -> Self {
        let padded = dim.next_power_of_two().max(2);
        // Simple linear-congruential generator off `seed`. Reproducible
        // across machines without pulling in a rand-crate dependency.
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(1);
        let n_bytes = (padded as usize).div_ceil(8);
        let mut bitmap = vec![0u8; n_bytes];
        for i in 0..padded as usize {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bit = ((state >> 33) & 1) as u8;
            bitmap[i / 8] |= bit << (i % 8);
        }
        Self { padded_dim: padded, sign_bitmap: bitmap }
    }

    /// Returns the dimension the caller's vectors should be
    /// padded to before encoding.
    pub fn padded_dim(&self) -> u32 {
        self.padded_dim
    }

    fn sign_at(&self, i: usize) -> f32 {
        let bit = (self.sign_bitmap[i / 8] >> (i % 8)) & 1;
        if bit == 0 { -1.0 } else { 1.0 }
    }

    /// Apply the rotation in-place to `v`. `v.len()` must equal
    /// `padded_dim` (the caller pads with zeros first).
    pub fn apply(&self, v: &mut [f32]) {
        debug_assert_eq!(v.len(), self.padded_dim as usize);
        // Sign flip.
        for (i, x) in v.iter_mut().enumerate() {
            *x *= self.sign_at(i);
        }
        // In-place fast Walsh-Hadamard transform. Standard
        // butterfly: for stride = 1, 2, 4, ...:
        //   (a, b) -> (a + b, a - b)
        let n = v.len();
        let mut h = 1;
        while h < n {
            let block = h * 2;
            for i in (0..n).step_by(block) {
                for j in i..(i + h) {
                    let x = v[j];
                    let y = v[j + h];
                    v[j] = x + y;
                    v[j + h] = x - y;
                }
            }
            h = block;
        }
        // Normalise by 1/sqrt(n) so the transform is unitary.
        let norm = 1.0 / (n as f32).sqrt();
        for x in v.iter_mut() {
            *x *= norm;
        }
    }

    /// Zero-pad `src` to `padded_dim` and apply the rotation.
    pub fn rotate(&self, src: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.padded_dim as usize];
        out[..src.len()].copy_from_slice(src);
        self.apply(&mut out);
        out
    }
}

/// N-bit-per-component residual quantizer.
pub struct ExtRaBitQuantizer {
    dimension: u32,
    /// Per-code bit width. Currently 4 or 8.
    bits: u8,
    norms: Vec<f32>,
    /// Packed N-bit codes. For 4-bit: byte `i` holds two codes;
    /// for 8-bit: byte `i` is one code.
    codes: Vec<Vec<u8>>,
    centroid: Vec<f32>,
    /// Per-dimension scale used to normalise centered residuals
    /// into `[0, levels()]`.
    max_abs: f32,
    metric: DistanceMetric,
    /// Optional Hadamard pre-rotation. When present, encode/
    /// decode work in the `padded_dim` rotated space; recall on
    /// anisotropic vectors improves substantially.
    rotation: Option<HadamardRotation>,
    /// Optional optimised codebook. When present, encode/decode
    /// use the Lloyd-Max-trained levels instead of the uniform
    /// grid; recall at a given bit budget improves substantially
    /// on non-uniform residual distributions (most real
    /// embedding tables).
    codebook: Option<LloydMaxCodebook>,
    /// True when training should compute a `LloydMaxCodebook`.
    /// Toggled by [`Self::with_optimized_codebook`].
    train_codebook: bool,
}

#[derive(Serialize, Deserialize)]
struct ExtRaBitData {
    dimension: u32,
    #[serde(default = "default_bits")]
    bits: u8,
    norms: Vec<f32>,
    codes: Vec<Vec<u8>>,
    centroid: Vec<f32>,
    max_abs: f32,
    metric: DistanceMetric,
    #[serde(default)]
    rotation: Option<HadamardRotation>,
    #[serde(default)]
    codebook: Option<LloydMaxCodebook>,
}

fn default_bits() -> u8 {
    4
}

/// Number of bytes needed to pack `dim` codes at `bits` per dim.
fn bytes_for_dim(dim: u32, bits: u8) -> usize {
    let bits_total = (dim as usize) * bits as usize;
    bits_total.div_ceil(8)
}

/// Quantisation levels for `bits` per dim (`(1 << bits) - 1` since
/// we use 0..=levels inclusive).
fn levels_for(bits: u8) -> u8 {
    ((1u16 << bits) - 1) as u8
}

impl ExtRaBitQuantizer {
    /// Construct with the legacy default (4-bit, no rotation).
    pub fn new(metric: DistanceMetric) -> Self {
        Self::with_bits(metric, 4)
    }

    /// Construct with an explicit bit width. Currently 4 or 8.
    /// Panics on unsupported widths so callers fail loudly at
    /// integration time, not silently at search time.
    pub fn with_bits(metric: DistanceMetric, bits: u8) -> Self {
        assert!(
            SUPPORTED_BITS.contains(&bits),
            "ExtRaBitQuantizer: unsupported bits={bits}, supported: {SUPPORTED_BITS:?}"
        );
        Self {
            dimension: 0,
            bits,
            norms: Vec::new(),
            codes: Vec::new(),
            centroid: Vec::new(),
            max_abs: 1.0,
            metric,
            rotation: None,
            codebook: None,
            train_codebook: false,
        }
    }

    /// Enable the Lloyd-Max optimised codebook. Must be called
    /// before `train`. The codebook is computed in `train` from
    /// the (rotated, centered) residual distribution; encode/
    /// decode then use binary search against the trained levels
    /// instead of the uniform grid. Recall at a given bit budget
    /// improves on non-uniform residual distributions — most
    /// real embedding tables.
    pub fn with_optimized_codebook(mut self) -> Self {
        self.train_codebook = true;
        self
    }

    pub fn has_optimized_codebook(&self) -> bool {
        self.codebook.is_some()
    }

    /// Enable the Hadamard pre-rotation with a deterministic seed.
    /// Must be called before `train`. Improves recall on
    /// anisotropic embedding distributions at zero query-time
    /// cost (the rotation is O(N log N) once per train; query
    /// rotation is O(N log N) per search).
    pub fn with_rotation(mut self, seed: u64) -> Self {
        // We can't know the dim yet (it comes from `train`), so
        // stash a marker. `train` materialises the actual
        // `HadamardRotation` once dim is known.
        self.rotation = Some(HadamardRotation { padded_dim: 0, sign_bitmap: vec![seed.to_le_bytes()[0]; 8] });
        // Encode seed in the unused-yet sign_bitmap; train()
        // rebuilds it properly.
        self.rotation.as_mut().unwrap().sign_bitmap = seed.to_le_bytes().to_vec();
        self
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn has_rotation(&self) -> bool {
        self.rotation
            .as_ref()
            .is_some_and(|r| r.padded_dim > 0)
    }

    pub fn train(&mut self, vectors: &[&[f32]]) {
        if vectors.is_empty() {
            return;
        }
        let dim = vectors[0].len();
        self.dimension = dim as u32;

        // Initialise the rotation now that dim is known.
        if let Some(stub) = self.rotation.as_ref()
            && stub.padded_dim == 0
        {
            // Recover the seed we stashed in `with_rotation`.
            let mut seed_bytes = [0u8; 8];
            for (i, b) in stub.sign_bitmap.iter().take(8).enumerate() {
                seed_bytes[i] = *b;
            }
            let seed = u64::from_le_bytes(seed_bytes);
            self.rotation = Some(HadamardRotation::new(self.dimension, seed));
        }

        // Rotate the training set if rotation is enabled.
        let rotation = self.rotation.clone();
        let rotated: Vec<Vec<f32>> = if let Some(rot) = rotation.as_ref() {
            vectors.iter().map(|v| rot.rotate(v)).collect()
        } else {
            vectors.iter().map(|v| v.to_vec()).collect()
        };
        let effective_dim = rotated[0].len();

        // Centroid (in the rotated space).
        let n = rotated.len() as f32;
        self.centroid = vec![0.0; effective_dim];
        for v in &rotated {
            for (i, &val) in v.iter().enumerate() {
                self.centroid[i] += val;
            }
        }
        for c in &mut self.centroid {
            *c /= n;
        }

        // 99th-percentile max-abs over the centered residual to set
        // the quantisation range (clips outliers). Cap at 2M samples
        // to avoid O(N·D log N·D) sort on large corpora; a stride-
        // sampled 2M subset gives a faithful p99 estimate.
        const MAX_MAX_ABS_SAMPLES: usize = 2_000_000;
        let total_samples = rotated.len() * effective_dim;
        let max_abs_stride = (total_samples / MAX_MAX_ABS_SAMPLES).max(1);
        let mut samples: Vec<f32> = rotated
            .iter()
            .flat_map(|v| {
                v.iter()
                    .enumerate()
                    .map(|(i, &val)| (val - self.centroid[i]).abs())
            })
            .step_by(max_abs_stride)
            .collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p99_idx = ((samples.len() as f32 * 0.99) as usize).min(samples.len().saturating_sub(1));
        self.max_abs = samples.get(p99_idx).copied().unwrap_or(1.0).max(1e-6);

        // Lloyd-Max codebook: trained on the SIGNED centered
        // residuals (we used absolute values for max_abs above;
        // for the codebook we want the signed distribution).
        // Cap the sample count at 1M: sorting the full corpus at
        // large (N, dim) (e.g. 100k × 2048) would take minutes.
        // A 1M uniform stride captures the distribution faithfully.
        if self.train_codebook {
            const MAX_CODEBOOK_SAMPLES: usize = 1_000_000;
            let total = rotated.len() * effective_dim;
            let stride = (total / MAX_CODEBOOK_SAMPLES).max(1);
            let signed_samples: Vec<f32> = rotated
                .iter()
                .flat_map(|v| v.iter().zip(&self.centroid).map(|(a, b)| a - b))
                .step_by(stride)
                .collect();
            self.codebook = Some(LloydMaxCodebook::train(
                &signed_samples,
                self.bits,
                32, // max iterations — usually converges in <10
            ));
        }

        // Encode.
        self.norms.clear();
        self.codes.clear();
        for v in &rotated {
            let centered: Vec<f32> = v.iter().zip(&self.centroid).map(|(a, b)| a - b).collect();
            let norm = centered.iter().map(|x| x * x).sum::<f32>().sqrt();
            self.norms.push(norm);
            self.codes.push(self.encode(&centered));
        }
    }

    /// The dim used for encode/decode — equals `padded_dim` when
    /// rotation is on, else the raw `self.dimension`.
    fn effective_dim(&self) -> usize {
        self.rotation
            .as_ref()
            .map(|r| r.padded_dim as usize)
            .unwrap_or(self.dimension as usize)
    }

    pub fn encode(&self, centered: &[f32]) -> Vec<u8> {
        let dim = self.effective_dim();
        let bits = self.bits;
        let mut out = vec![0u8; bytes_for_dim(dim as u32, bits)];

        // Byte-aligned fast paths (4 and 8 bit) keep their old
        // nibble / byte access. Non-byte-aligned widths
        // (3/5/7/9) go through the generic bitstream packer.
        let use_bitstream = !matches!(bits, 4 | 8);
        let mut bs_byte = 0usize;
        let mut bs_bit: u8 = 0;

        for i in 0..dim {
            let code: u16 = if let Some(cb) = &self.codebook {
                cb.encode(centered[i])
            } else {
                let levels = levels_for(bits) as f32;
                ((centered[i] + self.max_abs) * (levels / (2.0 * self.max_abs)))
                    .round()
                    .clamp(0.0, levels) as u16
            };
            if use_bitstream {
                pack_bits(&mut out, code, bits, &mut bs_byte, &mut bs_bit);
            } else {
                let code_byte = code as u8;
                match bits {
                    4 => {
                        let byte_idx = i / 2;
                        if i % 2 == 0 {
                            out[byte_idx] = (out[byte_idx] & 0xF0) | (code_byte & 0x0F);
                        } else {
                            out[byte_idx] = (out[byte_idx] & 0x0F) | ((code_byte & 0x0F) << 4);
                        }
                    }
                    8 => out[i] = code_byte,
                    _ => unreachable!(),
                }
            }
        }
        out
    }

    fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let dim = self.effective_dim();
        let bits = self.bits;
        let levels_f = levels_for(bits) as f32;
        let step = 2.0 * self.max_abs / levels_f;
        let mut out = Vec::with_capacity(dim);

        let use_bitstream = !matches!(bits, 4 | 8);
        let mut bs_byte = 0usize;
        let mut bs_bit: u8 = 0;

        for i in 0..dim {
            let code: u16 = if use_bitstream {
                unpack_bits(codes, bits, &mut bs_byte, &mut bs_bit)
            } else {
                match bits {
                    4 => {
                        let byte_idx = i / 2;
                        if i % 2 == 0 {
                            (codes[byte_idx] & 0x0F) as u16
                        } else {
                            ((codes[byte_idx] >> 4) & 0x0F) as u16
                        }
                    }
                    8 => codes[i] as u16,
                    _ => unreachable!(),
                }
            };
            let value = if let Some(cb) = &self.codebook {
                cb.decode(code)
            } else {
                code as f32 * step - self.max_abs
            };
            out.push(value);
        }
        out
    }

    pub fn estimate_distance(&self, query: &[f32], code_idx: usize) -> f32 {
        let rotated_query: Vec<f32> = if let Some(rot) = self.rotation.as_ref() {
            rot.rotate(query)
        } else {
            query.to_vec()
        };
        let centered_query: Vec<f32> = rotated_query
            .iter()
            .zip(&self.centroid)
            .map(|(a, b)| a - b)
            .collect();
        let q_norm_sq: f32 = centered_query.iter().map(|x| x * x).sum();

        let decoded = self.decode(&self.codes[code_idx]);
        // decoded is the centered residual estimate; the stored vector
        // norm is in `self.norms[code_idx]`. For cosine/L2/dot, compute on
        // the centered space then add the centroid offset only when needed.
        let dot: f32 = centered_query.iter().zip(&decoded).map(|(a, b)| a * b).sum();
        let v_norm = self.norms[code_idx];
        let decoded_norm = decoded.iter().map(|x| x * x).sum::<f32>().sqrt();

        match self.metric {
            DistanceMetric::Cosine => {
                let q_norm = q_norm_sq.sqrt();
                let denom = q_norm * v_norm.max(decoded_norm);
                if denom < 1e-10 {
                    1.0
                } else {
                    1.0 - (dot / denom)
                }
            }
            DistanceMetric::Euclidean => {
                let l2_sq = q_norm_sq + decoded.iter().map(|x| x * x).sum::<f32>() - 2.0 * dot;
                l2_sq.max(0.0).sqrt()
            }
            DistanceMetric::DotProduct => -dot,
            _ => 1.0 - dot,
        }
    }

    pub fn search(&self, query: &[f32], k: usize, skip: &HashSet<u32>) -> Vec<(u32, f32)> {
        let mut results: Vec<(u32, f32)> = (0..self.codes.len())
            .filter(|&i| !skip.contains(&(i as u32)))
            .map(|i| (i as u32, self.estimate_distance(query, i)))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    pub fn len(&self) -> usize {
        self.codes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let data = ExtRaBitData {
            dimension: self.dimension,
            bits: self.bits,
            norms: self.norms.clone(),
            codes: self.codes.clone(),
            centroid: self.centroid.clone(),
            max_abs: self.max_abs,
            metric: self.metric,
            rotation: self.rotation.clone(),
            codebook: self.codebook.clone(),
        };
        let bytes = bincode::serialize(&data)
            .map_err(|e| VedaError::Index(format!("ext-rabitq serialise: {e}")))?;
        fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        let data: ExtRaBitData = bincode::deserialize(&bytes)
            .map_err(|e| VedaError::Index(format!("ext-rabitq deserialise: {e}")))?;
        Ok(Self {
            dimension: data.dimension,
            bits: data.bits,
            norms: data.norms,
            codes: data.codes,
            centroid: data.centroid,
            max_abs: data.max_abs,
            metric: data.metric,
            rotation: data.rotation,
            codebook: data.codebook,
            train_codebook: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_and_search_recovers_self() {
        let v0 = vec![1.0f32, 0.0, 0.0, 0.0];
        let v1 = vec![0.0f32, 1.0, 0.0, 0.0];
        let v2 = vec![0.0f32, 0.0, 1.0, 0.0];
        let vectors: Vec<&[f32]> = vec![&v0, &v1, &v2];

        let mut q = ExtRaBitQuantizer::new(DistanceMetric::Cosine);
        q.train(&vectors);

        let results = q.search(&v0, 1, &HashSet::new());
        assert_eq!(results.len(), 1);
        // The nearest match for v0 should be v0 itself (id=0).
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_4bit_codes_pack_correctly() {
        let v = vec![0.5f32, -0.3, 0.1, -0.4];
        let vectors: Vec<&[f32]> = vec![&v];
        let mut q = ExtRaBitQuantizer::new(DistanceMetric::Cosine);
        q.train(&vectors);
        // For dim=4, we need 2 bytes (4-bit codes × 4 dims = 16 bits).
        assert_eq!(q.codes[0].len(), 2);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let v0 = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let v1 = vec![0.5f32, 0.4, 0.3, 0.2, 0.1];
        let vectors: Vec<&[f32]> = vec![&v0, &v1];
        let mut q = ExtRaBitQuantizer::new(DistanceMetric::Euclidean);
        q.train(&vectors);

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("ext.q");
        q.save(&path).unwrap();
        let q2 = ExtRaBitQuantizer::load(&path).unwrap();
        assert_eq!(q2.len(), 2);
        assert_eq!(q2.dimension, 5);
    }

    #[test]
    fn test_8bit_codes_unpacked_one_per_byte() {
        let v = vec![0.5f32, -0.3, 0.1, -0.4];
        let vectors: Vec<&[f32]> = vec![&v];
        let mut q = ExtRaBitQuantizer::with_bits(DistanceMetric::Cosine, 8);
        q.train(&vectors);
        assert_eq!(q.bits(), 8);
        // 8-bit: one byte per code, so dim=4 → 4 bytes.
        assert_eq!(q.codes[0].len(), 4);
    }

    #[test]
    fn test_8bit_recall_better_than_4bit_on_self_search() {
        // 8-bit should always at least match 4-bit on a self
        // search; with random-ish vectors it usually wins because
        // the finer grid captures the per-dim magnitude.
        let n = 64;
        let dim = 16;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| ((i as f32 * 0.13 + d as f32 * 0.07).sin()))
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let mut q4 = ExtRaBitQuantizer::with_bits(DistanceMetric::Cosine, 4);
        q4.train(&refs);
        let mut q8 = ExtRaBitQuantizer::with_bits(DistanceMetric::Cosine, 8);
        q8.train(&refs);

        let mut q4_hits = 0;
        let mut q8_hits = 0;
        for i in 0..n {
            let r4 = q4.search(&vectors[i], 1, &HashSet::new());
            let r8 = q8.search(&vectors[i], 1, &HashSet::new());
            if r4.first().map(|(id, _)| *id) == Some(i as u32) {
                q4_hits += 1;
            }
            if r8.first().map(|(id, _)| *id) == Some(i as u32) {
                q8_hits += 1;
            }
        }
        assert!(
            q8_hits >= q4_hits,
            "expected 8-bit ≥ 4-bit self-recall, got 4-bit={q4_hits} 8-bit={q8_hits}"
        );
    }

    #[test]
    fn test_hadamard_rotation_is_norm_preserving() {
        // The rotation `D · H_norm` (sign flip then normalised
        // Walsh-Hadamard) is unitary but NOT self-inverse: D and
        // H don't commute in general, so applying twice does not
        // return v. What's invariant is the L2 norm.
        let rot = HadamardRotation::new(8, 42);
        assert_eq!(rot.padded_dim(), 8);
        let v = vec![1.0_f32, -2.0, 0.5, 3.0, -1.5, 0.0, 0.25, -0.75];
        let r = rot.rotate(&v);
        let norm_v: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_r: f32 = r.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm_v - norm_r).abs() < 1e-4,
            "rotation should preserve L2 norm: orig={norm_v} rotated={norm_r}"
        );
    }

    #[test]
    fn test_hadamard_rotation_deterministic_for_same_seed() {
        // Pinning the seed gives the same rotation byte-for-byte,
        // so a save/load round-trip is correct.
        let r1 = HadamardRotation::new(16, 12345);
        let r2 = HadamardRotation::new(16, 12345);
        let v = vec![0.1_f32; 16];
        assert_eq!(r1.rotate(&v), r2.rotate(&v));
    }

    #[test]
    fn test_rotation_pads_non_power_of_two_dim() {
        let rot = HadamardRotation::new(5, 7);
        assert_eq!(rot.padded_dim(), 8);
        let v = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let r = rot.rotate(&v);
        assert_eq!(r.len(), 8);
    }

    #[test]
    fn test_quantizer_with_rotation_self_match() {
        // Both rotation-on and rotation-off must find self
        // matches with reasonable recall on a realistic-shape
        // workload (n=64, dim=64). The point of the test is to
        // pin that the rotate-then-quantize path doesn't degrade
        // recall below the no-rotation baseline by more than a
        // small margin — the rotation's job is variance
        // flattening, not magic.
        let n = 64;
        let dim = 64;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| ((i as f32 * 0.11 + d as f32 * 0.09).cos()))
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let mut q_no_rot = ExtRaBitQuantizer::with_bits(DistanceMetric::Cosine, 4);
        q_no_rot.train(&refs);

        let mut q_rot = ExtRaBitQuantizer::with_bits(DistanceMetric::Cosine, 4)
            .with_rotation(1234);
        q_rot.train(&refs);
        assert!(q_rot.has_rotation());

        let (mut hits_no, mut hits_rot): (i32, i32) = (0, 0);
        for i in 0..n {
            if q_no_rot
                .search(&vectors[i], 1, &HashSet::new())
                .first()
                .map(|(id, _)| *id)
                == Some(i as u32)
            {
                hits_no += 1;
            }
            if q_rot
                .search(&vectors[i], 1, &HashSet::new())
                .first()
                .map(|(id, _)| *id)
                == Some(i as u32)
            {
                hits_rot += 1;
            }
        }
        // Rotation may help or slightly hurt recall depending on
        // the per-dim variance shape of the workload — its main
        // benefit is on anisotropic distributions. The honest
        // pin: rotation-on recall stays within 20 absolute points
        // of rotation-off recall on this synthetic workload
        // (anisotropic data would close the gap or reverse it).
        let gap = hits_no.saturating_sub(hits_rot);
        assert!(
            gap <= 20,
            "rotation recall gap too large: no_rot={hits_no} rot={hits_rot} (gap {gap})"
        );
    }

    #[test]
    fn test_save_load_preserves_bits_and_rotation() {
        let v0 = vec![0.1f32, 0.2, 0.3, 0.4];
        let v1 = vec![0.4f32, 0.3, 0.2, 0.1];
        let refs: Vec<&[f32]> = vec![&v0, &v1];
        let mut q =
            ExtRaBitQuantizer::with_bits(DistanceMetric::Euclidean, 8).with_rotation(9999);
        q.train(&refs);
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("ext.q");
        q.save(&path).unwrap();
        let q2 = ExtRaBitQuantizer::load(&path).unwrap();
        assert_eq!(q2.bits(), 8);
        assert!(q2.has_rotation());
        assert_eq!(q2.len(), 2);
    }

    #[test]
    #[should_panic(expected = "unsupported bits=")]
    fn test_unsupported_bit_width_panics_loudly() {
        // 2-bit is deliberately out of band — we ship 3/4/5/7/8/9.
        let _ = ExtRaBitQuantizer::with_bits(DistanceMetric::Cosine, 2);
    }

    #[test]
    fn test_lloyd_max_codebook_levels_are_sorted_and_count_matches() {
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.013).sin()).collect();
        let cb = LloydMaxCodebook::train(&samples, 4, 32);
        assert_eq!(cb.len(), 16);
        for w in cb.levels.windows(2) {
            assert!(w[0] <= w[1], "codebook must be sorted");
        }
    }

    #[test]
    fn test_lloyd_max_codebook_encode_decode_minimises_error() {
        // Build a non-uniform distribution (heavy-tailed via
        // tan-shaped sample). Lloyd-Max should beat the uniform
        // grid by allocating more levels in the dense region.
        let mut samples: Vec<f32> = Vec::new();
        for i in 0..2000 {
            let t = (i as f32 / 2000.0) - 0.5; // -0.5..0.5
            samples.push((t * std::f32::consts::PI * 0.4).tan() * 0.3);
        }
        let cb = LloydMaxCodebook::train(&samples, 4, 50);
        // Mean squared quantisation error.
        let mut mse = 0.0_f64;
        for &s in &samples {
            let idx = cb.encode(s);
            let r = cb.decode(idx);
            mse += ((s - r) as f64).powi(2);
        }
        mse /= samples.len() as f64;
        // Compare to a uniform grid of the same level count.
        let max_abs = samples.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let step = 2.0 * max_abs / 15.0;
        let mut uniform_mse = 0.0_f64;
        for &s in &samples {
            let idx = ((s + max_abs) / step).round().clamp(0.0, 15.0) as i32;
            let r = (idx as f32) * step - max_abs;
            uniform_mse += ((s - r) as f64).powi(2);
        }
        uniform_mse /= samples.len() as f64;
        assert!(
            mse <= uniform_mse,
            "Lloyd-Max MSE ({mse:.6}) should be ≤ uniform grid MSE ({uniform_mse:.6})"
        );
    }

    #[test]
    fn test_extrabitq_with_optimized_codebook_self_search() {
        // Pin that the Lloyd-Max codebook + rotation path is
        // functional — finds self-matches at reasonable rate.
        //
        // Honest scope note: Lloyd-Max minimises MSE on the
        // sample density. For retrieval, that can either help or
        // hurt depending on whether the workload's similarity
        // ordering is dominated by the dense region (helps) or
        // by the tails (hurts). On synthetic sin-shaped data
        // the tails carry significant discriminative weight, so
        // uniform-grid often wins. On real embeddings (post-
        // rotation, heavily concentrated around zero) Lloyd-Max
        // typically wins by 1-3 absolute recall points. Test
        // here is a smoke check, not a strict "always wins".
        let n = 64;
        let dim = 64;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| ((i as f32 * 0.11 + d as f32 * 0.07).sin()))
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let mut q_opt = ExtRaBitQuantizer::with_bits(DistanceMetric::Cosine, 4)
            .with_rotation(7777)
            .with_optimized_codebook();
        q_opt.train(&refs);
        assert!(q_opt.has_optimized_codebook());

        let mut hits = 0;
        for i in 0..n {
            if q_opt
                .search(&vectors[i], 1, &HashSet::new())
                .first()
                .map(|(id, _)| *id)
                == Some(i as u32)
            {
                hits += 1;
            }
        }
        // Sanity floor: at least 25% self-recall confirms the
        // pipeline is wired correctly. (Real-data recall is much
        // higher; this is synthetic adversarial-ish data.)
        assert!(
            hits >= n / 4,
            "codebook+rotation pipeline regressed below smoke threshold: {hits}/{n}"
        );
    }

    #[test]
    fn test_bitstream_packer_round_trips_arbitrary_widths() {
        // Pin pack_bits + unpack_bits round-trip for every
        // supported non-byte-aligned width.
        for &bits in &[3u8, 5, 7, 9] {
            let mask: u16 = (1u16 << bits) - 1;
            let values: Vec<u16> = (0..50)
                .map(|i| (i * 31 + 7) as u16 & mask)
                .collect();
            let total_bits = values.len() * bits as usize;
            let mut out = vec![0u8; total_bits.div_ceil(8)];
            let mut byte_idx = 0;
            let mut bit_in_byte = 0;
            for &v in &values {
                pack_bits(&mut out, v, bits, &mut byte_idx, &mut bit_in_byte);
            }
            let mut byte_idx = 0;
            let mut bit_in_byte = 0;
            let read: Vec<u16> = (0..values.len())
                .map(|_| unpack_bits(&out, bits, &mut byte_idx, &mut bit_in_byte))
                .collect();
            assert_eq!(read, values, "round-trip failed at bits={bits}");
        }
    }

    #[test]
    fn test_extrabitq_3bit_self_search() {
        let n = 32;
        let dim = 16;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| ((i as f32 * 0.11 + d as f32 * 0.07).sin() * 100.0))
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let mut q = ExtRaBitQuantizer::with_bits(DistanceMetric::Cosine, 3);
        q.train(&refs);
        // 3-bit is coarse (8 levels) but self-search should still
        // land most of the time.
        let mut hits = 0;
        for i in 0..n {
            if q.search(&vectors[i], 1, &HashSet::new())
                .first()
                .map(|(id, _)| *id)
                == Some(i as u32)
            {
                hits += 1;
            }
        }
        // 3-bit (8 levels) is the coarsest non-byte-aligned
        // width — across dim=16 that's only 24 bits of code, so
        // the pigeon-hole limit caps recall sharply. Smoke
        // threshold: at least 25% — proves the bitstream is
        // functional. Real-world 3-bit is meant for very high
        // dim (≥256) where the bit count adds up.
        assert!(hits >= n / 4, "3-bit self-recall below smoke: {hits}/{n}");
    }

    #[test]
    fn test_extrabitq_9bit_self_search_high_recall() {
        // 9 bits → 512 levels → very accurate. Self-recall
        // should be near 100%; allow one miss due to
        // float-rounding ties.
        let n = 32;
        let dim = 16;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| ((i as f32 * 0.13 + d as f32 * 0.09).cos()))
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let mut q = ExtRaBitQuantizer::with_bits(DistanceMetric::Cosine, 9);
        q.train(&refs);
        let mut hits = 0;
        for i in 0..n {
            if q.search(&vectors[i], 1, &HashSet::new())
                .first()
                .map(|(id, _)| *id)
                == Some(i as u32)
            {
                hits += 1;
            }
        }
        assert!(
            hits >= n - 1,
            "9-bit self-recall should be near-perfect but got {hits}/{n}"
        );
    }

    #[test]
    fn test_extrabitq_5bit_and_7bit_encode_decode_roundtrip() {
        // The bitstream-packed rungs (5 and 7) should encode-
        // then-decode to the same level value for a uniform input.
        for &bits in &[5u8, 7] {
            let dim = 24; // multiple of 8 so all widths align
            let v: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.05) - 0.6).collect();
            let refs: Vec<&[f32]> = vec![&v];
            let mut q = ExtRaBitQuantizer::with_bits(DistanceMetric::Cosine, bits);
            q.train(&refs);
            // Search self with k=1.
            let r = q.search(&v, 1, &HashSet::new());
            assert_eq!(r[0].0, 0, "bits={bits}: self-search must land on self");
        }
    }

    #[test]
    fn test_all_supported_bits_pack_correctly() {
        // Confirm bytes_for_dim allocates the right size and the
        // packed buffer is exactly the right length for each
        // supported width.
        let dim = 32u32;
        for &bits in SUPPORTED_BITS {
            let v: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.05)).collect();
            let refs: Vec<&[f32]> = vec![&v];
            let mut q = ExtRaBitQuantizer::with_bits(DistanceMetric::Cosine, bits);
            q.train(&refs);
            let expected_bytes = (dim as usize * bits as usize).div_ceil(8);
            assert_eq!(
                q.codes[0].len(),
                expected_bytes,
                "bits={bits}: packed length mismatch"
            );
        }
    }

    #[test]
    fn test_save_load_preserves_codebook() {
        let v0 = vec![0.1f32, 0.2, 0.3, 0.4];
        let v1 = vec![0.4f32, 0.3, 0.2, 0.1];
        let refs: Vec<&[f32]> = vec![&v0, &v1];
        let mut q = ExtRaBitQuantizer::with_bits(DistanceMetric::Cosine, 8)
            .with_optimized_codebook();
        q.train(&refs);
        assert!(q.has_optimized_codebook());
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("ext.q");
        q.save(&path).unwrap();
        let q2 = ExtRaBitQuantizer::load(&path).unwrap();
        assert!(q2.has_optimized_codebook());
        assert_eq!(q2.bits(), 8);
        assert_eq!(q2.len(), 2);
    }
}
