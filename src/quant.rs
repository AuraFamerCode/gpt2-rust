//! int8 weight quantization and integer dot-product kernels.
//!
//! Quantization is block-wise with 32 values per block (`Q8_0` style): each
//! block of 32 weights gets its own symmetric scale, and the activations of a
//! token are quantized the same way at call time. Block scales matter because
//! transformer activations contain a few extreme outlier channels; with one
//! scale per row those outliers would consume almost all 127 levels and crush
//! everything else. With per-block scales an outlier only affects its own block.
//!
//! The inner product is then integer: `sum(a_q * w_q)` in int32 (no overflow,
//! since 127 * 127 * 32 fits comfortably), rescaled once per block. That lets
//! the kernel use NEON `SDOT`, 16 multiply-adds per instruction instead of the
//! 4 of an f32 FMA, and it reads a quarter of the bytes.
//!
//! The `SDOT` kernel keeps a four-lane float accumulator: each block's int32
//! partial sums are converted and scaled into those lanes, so the cross-lane
//! reduction happens once per output row instead of once per block.

use rayon::prelude::*;

/// Values per quantization block.
pub const BLOCK: usize = 32;

/// int8 matrix with one scale per 32-value block, row-major `[rows, cols]`.
pub struct QTensor {
    pub rows: usize,
    pub cols: usize,
    blocks: usize,
    data: Vec<i8>,
    scales: Vec<f32>, // [rows * blocks]
}

impl QTensor {
    pub fn from_f32(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols, "quantize shape mismatch");
        let blocks = cols.div_ceil(BLOCK);
        let mut q = vec![0i8; rows * cols];
        let mut scales = vec![0.0f32; rows * blocks];
        for r in 0..rows {
            for b in 0..blocks {
                let off = b * BLOCK;
                let end = (off + BLOCK).min(cols);
                let base = r * cols;
                scales[r * blocks + b] = quantize_block(
                    &data[base + off..base + end],
                    &mut q[base + off..base + end],
                );
            }
        }
        QTensor {
            rows,
            cols,
            blocks,
            data: q,
            scales,
        }
    }

    #[inline]
    fn block(&self, row: usize, b: usize) -> &[i8] {
        let off = row * self.cols + b * BLOCK;
        let len = BLOCK.min(self.cols - b * BLOCK);
        &self.data[off..off + len]
    }

    #[inline]
    fn block_scale(&self, row: usize, b: usize) -> f32 {
        self.scales[row * self.blocks + b]
    }

    pub fn dequant_row(&self, row: usize, out: &mut [f32]) {
        for b in 0..self.blocks {
            let s = self.block_scale(row, b);
            let off = b * BLOCK;
            for (o, &v) in out[off..].iter_mut().zip(self.block(row, b)) {
                *o = v as f32 * s;
            }
        }
    }

    pub fn nbytes(&self) -> usize {
        self.data.len() + self.scales.len() * std::mem::size_of::<f32>()
    }
}

/// Symmetric quantization of one block, returning the scale to undo it.
#[inline]
pub fn quantize_block(a: &[f32], out: &mut [i8]) -> f32 {
    let mut max = 0.0f32;
    for &v in a {
        let m = v.abs();
        if m > max {
            max = m;
        }
    }
    let scale = if max > 0.0 { max / 127.0 } else { 1.0 };
    let inv = 1.0 / scale;
    for (o, &v) in out.iter_mut().zip(a.iter()) {
        *o = (v * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

fn quantize_blocks(a: &[f32], out: &mut [i8], scales: &mut [f32]) {
    for (b, s) in scales.iter_mut().enumerate() {
        let off = b * BLOCK;
        let end = (off + BLOCK).min(a.len());
        *s = quantize_block(&a[off..end], &mut out[off..end]);
    }
}

/// Active integer kernel, for reporting.
pub fn kernel_name() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    if dotprod_available() {
        return "neon-sdot";
    }
    "scalar"
}

#[cfg(target_arch = "aarch64")]
fn dotprod_available() -> bool {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| std::arch::is_aarch64_feature_detected!("dotprod"))
}

type DotRow = fn(&[i8], &[f32], &QTensor, usize) -> f32;

/// Resolved once per matmul, so the hot loop has no feature check.
#[inline]
fn dot_row_dispatch() -> DotRow {
    #[cfg(target_arch = "aarch64")]
    if dotprod_available() {
        return dot_row_sdot;
    }
    dot_row_generic
}

/// Integer dot product over all blocks of one quantized row, rescaled per block.
fn dot_row_generic(aq: &[i8], a_scales: &[f32], w: &QTensor, row: usize) -> f32 {
    let mut acc = 0.0f32;
    for (b, &sa) in a_scales.iter().enumerate() {
        let off = b * BLOCK;
        let end = (off + BLOCK).min(w.cols);
        let d = dot_i8_scalar(&aq[off..end], w.block(row, b));
        acc += d as f32 * (sa * w.block_scale(row, b));
    }
    acc
}

#[cfg(target_arch = "aarch64")]
fn dot_row_sdot(aq: &[i8], a_scales: &[f32], w: &QTensor, row: usize) -> f32 {
    // SAFETY: this wrapper is only reached after `dotprod_available()`.
    unsafe { dot_row_sdot_impl(aq, a_scales, w, row) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_row_sdot_impl(aq: &[i8], a_scales: &[f32], w: &QTensor, row: usize) -> f32 {
    use std::arch::aarch64::*;
    let cols = w.cols;
    let mut acc = vdupq_n_f32(0.0);
    let mut tail = 0.0f32;
    for (b, &sa) in a_scales.iter().enumerate() {
        let off = b * BLOCK;
        let len = BLOCK.min(cols - off);
        let scale = vdupq_n_f32(sa * w.block_scale(row, b));
        if len == BLOCK {
            let aa = aq.as_ptr().add(off);
            let wb = w.block(row, b).as_ptr();
            let prod = vdotq_s32(vdupq_n_s32(0), vld1q_s8(aa), vld1q_s8(wb));
            let prod = vdotq_s32(prod, vld1q_s8(aa.add(16)), vld1q_s8(wb.add(16)));
            // Fold the block into four float lanes; the lanes are only summed
            // once, after the loop.
            acc = vfmaq_f32(acc, vcvtq_f32_s32(prod), scale);
        } else {
            let d = dot_i8_scalar(&aq[off..off + len], w.block(row, b));
            tail += d as f32 * sa * w.block_scale(row, b);
        }
    }
    vaddvq_f32(acc) + tail
}

/// Portable fallback for the 16-wide block sum.
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    let n = a.len();
    let chunks = n / 4;
    let mut acc = [0i32; 4];
    for c in 0..chunks {
        let i = c * 4;
        acc[0] += a[i] as i32 * b[i] as i32;
        acc[1] += a[i + 1] as i32 * b[i + 1] as i32;
        acc[2] += a[i + 2] as i32 * b[i + 2] as i32;
        acc[3] += a[i + 3] as i32 * b[i + 3] as i32;
    }
    let mut tail = 0i32;
    for i in chunks * 4..n {
        tail += a[i] as i32 * b[i] as i32;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

/// `out[m, n] = a[m, k] @ w[n, k]^T + bias[n]` with int8 weights.
pub fn matmul_i8(
    a: &[f32],
    m: usize,
    k: usize,
    w: &QTensor,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    assert_eq!(a.len(), m * k, "activation shape mismatch");
    assert_eq!(w.cols, k, "weight shape mismatch");
    assert_eq!(out.len(), m * w.rows, "output shape mismatch");
    let n = w.rows;
    let blocks = k.div_ceil(BLOCK);
    let dot = dot_row_dispatch();

    // Single token: quantize once, then run every output row in parallel.
    if m == 1 {
        let mut aq = vec![0i8; k];
        let mut sa = vec![0.0f32; blocks];
        quantize_blocks(&a[..k], &mut aq, &mut sa);
        out.par_iter_mut().enumerate().for_each(|(j, o)| {
            let mut v = dot(&aq, &sa, w, j);
            if let Some(b) = bias {
                v += b[j];
            }
            *o = v;
        });
        return;
    }

    // Prompt prefill: parallelize over activation rows.
    out.par_chunks_mut(n).enumerate().for_each(|(i, orow)| {
        let mut aq = vec![0i8; k];
        let mut sa = vec![0.0f32; blocks];
        quantize_blocks(&a[i * k..(i + 1) * k], &mut aq, &mut sa);
        for (j, o) in orow.iter_mut().enumerate() {
            let mut v = dot(&aq, &sa, w, j);
            if let Some(b) = bias {
                v += b[j];
            }
            *o = v;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random values in [-1, 1].
    fn values(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
            })
            .collect()
    }

    /// The kernel must compute exactly the integer math it promises: integer
    /// products of the quantized values, rescaled per block.
    #[test]
    fn int8_matmul_matches_quantized_reference() {
        let (m, k, n) = (3, 96, 7);
        let a = values(m * k, 12345);
        let w = values(n * k, 999);
        let q = QTensor::from_f32(&w, n, k);

        let mut out8 = vec![0.0f32; m * n];
        matmul_i8(&a, m, k, &q, None, &mut out8);

        for i in 0..m {
            let mut aq = vec![0i8; k];
            let mut sa = vec![0.0f32; k.div_ceil(BLOCK)];
            quantize_blocks(&a[i * k..(i + 1) * k], &mut aq, &mut sa);
            let mut wrow = vec![0.0f32; k];
            for j in 0..n {
                q.dequant_row(j, &mut wrow);
                let mut want = 0.0f64;
                for t in 0..k {
                    // quantized activation value times dequantized weight
                    let av = aq[t] as f64 * sa[t / BLOCK] as f64;
                    want += av * wrow[t] as f64;
                }
                let got = out8[i * n + j] as f64;
                assert!(
                    (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                    "kernel drifted from its own math at {i},{j}: {got} vs {want}"
                );
            }
        }
    }

    /// Quantization itself is lossy but must stay small: block scales keep the
    /// error near the int8 step rather than the row range.
    #[test]
    fn int8_matmul_tracks_f32() {
        let (m, k, n) = (3, 96, 7);
        let a = values(m * k, 12345);
        let w = values(n * k, 999);
        let q = QTensor::from_f32(&w, n, k);

        let mut out8 = vec![0.0f32; m * n];
        matmul_i8(&a, m, k, &q, None, &mut out8);
        let mut out32 = vec![0.0f32; m * n];
        crate::tensor::matmul(&a, m, k, &w, n, None, &mut out32);

        let mut sq_err = 0.0f64;
        let mut sq_sig = 0.0f64;
        for i in 0..m * n {
            let err = (out8[i] - out32[i]) as f64;
            assert!(err.abs() < 0.05, "abs err {err} at {i}: {} vs {}", out8[i], out32[i]);
            sq_err += err * err;
            sq_sig += (out32[i] as f64).powi(2);
        }
        let rel = (sq_err / sq_sig).sqrt();
        assert!(rel < 0.02, "RMS relative error {rel} too large");
    }

    #[test]
    fn quantize_roundtrip_is_close() {
        let w = values(128, 7);
        let q = QTensor::from_f32(&w, 2, 64);
        let mut back = vec![0.0f32; 64];
        q.dequant_row(1, &mut back);
        for i in 0..64 {
            assert!((back[i] - w[64 + i]).abs() < 0.02, "at {i}");
        }
    }
}
