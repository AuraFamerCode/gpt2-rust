//! Minimal dense tensor container and the matrix-multiply kernel.
//!
//! Everything is row-major `f32`. A linear layer weight is stored as
//! `[out_dim, in_dim]` (i.e. one contiguous row per output unit), so a forward
//! pass is a dot product per output element.

use rayon::prelude::*;

#[derive(Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "shape/data mismatch");
        Tensor { shape, data }
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    // Four independent accumulators so the fma's latency is hidden.
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let chunks = a.len() / 4;
    for c in 0..chunks {
        let i = c * 4;
        acc0 += a[i] * b[i];
        acc1 += a[i + 1] * b[i + 1];
        acc2 += a[i + 2] * b[i + 2];
        acc3 += a[i + 3] * b[i + 3];
    }
    let mut tail = 0.0f32;
    for i in chunks * 4..a.len() {
        tail += a[i] * b[i];
    }
    (acc0 + acc1) + (acc2 + acc3) + tail
}

/// `out[m, n] = a[m, k] @ w[n, k]^T + bias[n]`
///
/// `w` is row-major `[n, k]`. Threaded across output rows (and across columns
/// when there is a single row, which is the token-at-a-time decode case).
pub fn matmul(
    a: &[f32],
    m: usize,
    k: usize,
    w: &[f32],
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    assert_eq!(a.len(), m * k, "activation shape mismatch");
    assert_eq!(w.len(), n * k, "weight shape mismatch");
    assert_eq!(out.len(), m * n, "output shape mismatch");

    if m == 1 {
        let arow = &a[..k];
        out.par_iter_mut().enumerate().for_each(|(j, o)| {
            let mut v = dot(arow, &w[j * k..(j + 1) * k]);
            if let Some(b) = bias {
                v += b[j];
            }
            *o = v;
        });
        return;
    }

    out.par_chunks_mut(n).enumerate().for_each(|(i, orow)| {
        let arow = &a[i * k..(i + 1) * k];
        for (j, o) in orow.iter_mut().enumerate() {
            let mut v = dot(arow, &w[j * k..(j + 1) * k]);
            if let Some(b) = bias {
                v += b[j];
            }
            *o = v;
        }
    });
}

/// In-place softmax over a slice.
pub fn softmax(x: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        let e = (*v - max).exp();
        *v = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// LayerNorm with the biased variance, exactly as in the original model.
pub fn layernorm(x: &[f32], w: &[f32], b: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..x.len() {
        out[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

// GPT-2 uses the tanh approximation of GELU ("gelu_new").
const GELU_C: f32 = 0.797_884_6; // sqrt(2/pi)

#[inline]
pub fn gelu(x: f32) -> f32 {
    let x3 = x * x * x;
    0.5 * x * (1.0 + (GELU_C * (x + 0.044_715 * x3)).tanh())
}
