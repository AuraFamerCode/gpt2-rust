//! GPT-2 decoder-only transformer forward pass.
//!
//! Weights come from the HuggingFace safetensors file. Conv1D weights in the
//! original checkpoint are `[in, out]`; the safetensors conversion and some
//! re-uploads differ, so `take_weight` normalises everything to `[out, in]`.
//!
//! Projections run either in fp32 or as int8 weights with integer dot products
//! (`Precision`). All activations, LayerNorm and the KV cache stay fp32.

use crate::quant::{matmul_i8, QTensor};
use crate::tensor::{gelu, layernorm, matmul, softmax, Tensor};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub n_vocab: usize,
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub n_inner: usize,
    pub eps: f32,
}

impl Config {
    pub fn gpt2_small() -> Self {
        Config {
            n_vocab: 50257,
            n_ctx: 1024,
            n_embd: 768,
            n_head: 12,
            n_layer: 12,
            n_inner: 3072,
            eps: 1e-5,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Precision {
    /// Exact fp32 weights.
    F32,
    /// int8 weights, int8 activations, int32 accumulation.
    I8,
}

impl Precision {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "f32" | "fp32" | "float32" => Ok(Precision::F32),
            "int8" | "i8" | "q8" => Ok(Precision::I8),
            other => Err(format!("unknown precision {other} (expected f32 or int8)")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::I8 => "int8",
        }
    }
}

/// A projection matrix stored as `[rows, cols]` with `rows` = output features.
pub struct Weight {
    data: WeightData,
    rows: usize,
    cols: usize,
}

enum WeightData {
    Dense(Vec<f32>),
    Int8(QTensor),
}

impl Weight {
    fn new(
        data: Vec<f32>,
        rows: usize,
        cols: usize,
        precision: Precision,
    ) -> Result<Self, String> {
        if data.len() != rows * cols {
            return Err(format!(
                "weight has {} values, expected {}",
                data.len(),
                rows * cols
            ));
        }
        let data = match precision {
            Precision::F32 => WeightData::Dense(data),
            Precision::I8 => WeightData::Int8(QTensor::from_f32(&data, rows, cols)),
        };
        Ok(Weight { data, rows, cols })
    }

    /// `out[m, rows] = a[m, cols] @ self^T + bias`
    fn linear(&self, a: &[f32], m: usize, k: usize, bias: Option<&[f32]>, out: &mut [f32]) {
        debug_assert_eq!(k, self.cols);
        debug_assert_eq!(out.len(), m * self.rows);
        match &self.data {
            WeightData::Dense(w) => matmul(a, m, k, w, self.rows, bias, out),
            WeightData::Int8(q) => matmul_i8(a, m, k, q, bias, out),
        }
    }

    /// Copy (or dequantize) one row, for embedding lookups.
    fn gather(&self, idx: usize, out: &mut [f32]) {
        match &self.data {
            WeightData::Dense(w) => out.copy_from_slice(&w[idx * self.cols..(idx + 1) * self.cols]),
            WeightData::Int8(q) => q.dequant_row(idx, out),
        }
    }

    fn numel(&self) -> usize {
        self.rows * self.cols
    }

    fn nbytes(&self) -> usize {
        match &self.data {
            WeightData::Dense(w) => w.len() * std::mem::size_of::<f32>(),
            WeightData::Int8(q) => q.nbytes(),
        }
    }
}

struct Layer {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    attn_w: Weight, // [3 * n_embd, n_embd]
    attn_b: Vec<f32>,
    attn_proj_w: Weight, // [n_embd, n_embd]
    attn_proj_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    fc_w: Weight, // [n_inner, n_embd]
    fc_b: Vec<f32>,
    fc_proj_w: Weight, // [n_embd, n_inner]
    fc_proj_b: Vec<f32>,
}

/// Keys/values of every layer at every position seen so far.
pub struct KvCache {
    k: Vec<Vec<f32>>, // per layer, [n_ctx, n_embd]
    v: Vec<Vec<f32>>,
    pub len: usize,
}

impl KvCache {
    pub fn new(cfg: &Config) -> Self {
        let n = cfg.n_ctx * cfg.n_embd;
        KvCache {
            k: vec![vec![0.0; n]; cfg.n_layer],
            v: vec![vec![0.0; n]; cfg.n_layer],
            len: 0,
        }
    }
}

pub struct Gpt2 {
    pub cfg: Config,
    wte: Weight, // [n_vocab, n_embd], also the LM head (weights are tied)
    wpe: Vec<f32>, // [n_ctx, n_embd]
    ln_f_w: Vec<f32>,
    ln_f_b: Vec<f32>,
    layers: Vec<Layer>,
}

impl Gpt2 {
    pub fn load(
        mut tensors: HashMap<String, Tensor>,
        mut cfg: Config,
        precision: Precision,
    ) -> Result<Self, String> {
        let d = cfg.n_embd;

        let wte_t = tensors.remove("wte.weight").ok_or("missing wte.weight")?;
        cfg.n_vocab = wte_t.shape[0];
        cfg.n_embd = wte_t.shape[1];
        cfg.n_ctx = tensors
            .get("wpe.weight")
            .map(|t| t.shape[0])
            .unwrap_or(cfg.n_ctx);
        if cfg.n_embd != d {
            return Err(format!(
                "embedding dim {} does not match config {d}",
                cfg.n_embd
            ));
        }
        let wte = Weight::new(wte_t.data, cfg.n_vocab, cfg.n_embd, precision)?;

        // Infer the feed-forward width from the checkpoint rather than trusting
        // the config, so other GPT-2 sizes load without edits.
        if let Some(t) = tensors.get("h.0.mlp.c_fc.weight") {
            let inner = match t.shape.as_slice() {
                [d0, inner] if *d0 == cfg.n_embd => *inner,
                [inner, d0] if *d0 == cfg.n_embd => *inner,
                _ => cfg.n_inner,
            };
            cfg.n_inner = inner;
        }
        let inner = cfg.n_inner;

        let wpe = tensors.remove("wpe.weight").ok_or("missing wpe.weight")?;
        let ln_f_w = take_bias(&mut tensors, "ln_f.weight", d)?;
        let ln_f_b = take_bias(&mut tensors, "ln_f.bias", d)?;

        // Infer the layer count from the checkpoint.
        let mut n_layer = 0usize;
        while tensors.contains_key(&format!("h.{n_layer}.ln_1.weight")) {
            n_layer += 1;
        }
        if n_layer == 0 {
            return Err("no transformer blocks found in checkpoint".into());
        }
        cfg.n_layer = n_layer;

        // Conv1D projections may be stored as [in, out] or [out, in]; square
        // ones are indistinguishable by shape, so settle the convention once
        // using c_attn, whose shape is never square.
        let conv1d = {
            let t = tensors
                .get("h.0.attn.c_attn.weight")
                .ok_or("missing h.0.attn.c_attn.weight")?;
            if t.shape == vec![d, 3 * d] {
                WeightLayout::InOut
            } else if t.shape == vec![3 * d, d] {
                WeightLayout::OutIn
            } else {
                return Err(format!("c_attn weight has unexpected shape {:?}", t.shape));
            }
        };

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = |s: &str| format!("h.{i}.{s}");
            layers.push(Layer {
                ln1_w: take_bias(&mut tensors, &p("ln_1.weight"), d)?,
                ln1_b: take_bias(&mut tensors, &p("ln_1.bias"), d)?,
                attn_w: take_weight_proj(&mut tensors, &p("attn.c_attn.weight"), 3 * d, d, conv1d, precision)?,
                attn_b: take_bias(&mut tensors, &p("attn.c_attn.bias"), 3 * d)?,
                attn_proj_w: take_weight_proj(&mut tensors, &p("attn.c_proj.weight"), d, d, conv1d, precision)?,
                attn_proj_b: take_bias(&mut tensors, &p("attn.c_proj.bias"), d)?,
                ln2_w: take_bias(&mut tensors, &p("ln_2.weight"), d)?,
                ln2_b: take_bias(&mut tensors, &p("ln_2.bias"), d)?,
                fc_w: take_weight_proj(&mut tensors, &p("mlp.c_fc.weight"), inner, d, conv1d, precision)?,
                fc_b: take_bias(&mut tensors, &p("mlp.c_fc.bias"), inner)?,
                fc_proj_w: take_weight_proj(&mut tensors, &p("mlp.c_proj.weight"), d, inner, conv1d, precision)?,
                fc_proj_b: take_bias(&mut tensors, &p("mlp.c_proj.bias"), d)?,
            });
        }

        let model = Gpt2 {
            cfg,
            wte,
            wpe: wpe.data,
            ln_f_w,
            ln_f_b,
            layers,
        };
        model.check_sizes()?;
        Ok(model)
    }

    fn check_sizes(&self) -> Result<(), String> {
        let cfg = &self.cfg;
        if self.wte.numel() != cfg.n_vocab * cfg.n_embd {
            return Err("wte size mismatch".into());
        }
        if self.wpe.len() < cfg.n_ctx * cfg.n_embd {
            return Err("wpe size mismatch".into());
        }
        Ok(())
    }

    pub fn n_params(&self) -> usize {
        let mut n = self.wte.numel() + self.wpe.len() + self.ln_f_w.len() + self.ln_f_b.len();
        for l in &self.layers {
            n += l.attn_w.numel()
                + l.attn_proj_w.numel()
                + l.fc_w.numel()
                + l.fc_proj_w.numel()
                + l.ln1_w.len()
                + l.ln1_b.len()
                + l.attn_b.len()
                + l.attn_proj_b.len()
                + l.ln2_w.len()
                + l.ln2_b.len()
                + l.fc_b.len()
                + l.fc_proj_b.len();
        }
        n
    }

    /// Bytes actually held for the weights (fp32 vs int8 + scales).
    pub fn weight_bytes(&self) -> usize {
        let mut n = self.wte.nbytes()
            + self.wpe.len() * std::mem::size_of::<f32>()
            + (self.ln_f_w.len() + self.ln_f_b.len()) * std::mem::size_of::<f32>();
        for l in &self.layers {
            n += l.attn_w.nbytes() + l.attn_proj_w.nbytes() + l.fc_w.nbytes() + l.fc_proj_w.nbytes();
            n += (l.ln1_w.len()
                + l.ln1_b.len()
                + l.attn_b.len()
                + l.attn_proj_b.len()
                + l.ln2_w.len()
                + l.ln2_b.len()
                + l.fc_b.len()
                + l.fc_proj_b.len())
                * std::mem::size_of::<f32>();
        }
        n
    }

    /// Run `tokens` through the network starting at `start_pos`, writing keys
    /// and values into `cache`. Returns the logits for the final token only.
    pub fn forward(&self, cache: &mut KvCache, tokens: &[u32], start_pos: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.n_embd;
        let nh = cfg.n_head;
        let hd = cfg.head_dim();
        let t = tokens.len();
        assert!(start_pos + t <= cfg.n_ctx, "sequence longer than context");
        assert!(t > 0, "empty token batch");

        // Token + position embeddings.
        let mut x = vec![0.0f32; t * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = start_pos + i;
            let (tok, pos) = (tok as usize, pos as usize);
            let row = &mut x[i * d..(i + 1) * d];
            self.wte.gather(tok, row);
            let p = &self.wpe[pos * d..(pos + 1) * d];
            for j in 0..d {
                row[j] += p[j];
            }
        }

        let mut xa = vec![0.0f32; t * d]; // layer-normed activations
        let mut qkv = vec![0.0f32; t * 3 * d];
        let mut attn = vec![0.0f32; t * d];
        let mut proj = vec![0.0f32; t * d];
        let mut h = vec![0.0f32; t * cfg.n_inner];
        let mut hout = vec![0.0f32; t * d];
        let mut scores = vec![0.0f32; start_pos + t];
        let scale = 1.0 / (hd as f32).sqrt();

        for li in 0..cfg.n_layer {
            let layer = &self.layers[li];

            // --- self-attention ---
            for i in 0..t {
                layernorm(&x[i * d..(i + 1) * d], &layer.ln1_w, &layer.ln1_b, cfg.eps, &mut xa[i * d..(i + 1) * d]);
            }
            layer.attn_w.linear(&xa, t, d, Some(&layer.attn_b), &mut qkv);

            // Append this chunk's keys/values, then read them back as slices.
            {
                let kc = &mut cache.k[li];
                let vc = &mut cache.v[li];
                for i in 0..t {
                    let pos = start_pos + i;
                    let row = &qkv[i * 3 * d..(i + 1) * 3 * d];
                    kc[pos * d..(pos + 1) * d].copy_from_slice(&row[d..2 * d]);
                    vc[pos * d..(pos + 1) * d].copy_from_slice(&row[2 * d..3 * d]);
                }
            }
            let kc: &[f32] = &cache.k[li];
            let vc: &[f32] = &cache.v[li];

            attn.fill(0.0);
            for head in 0..nh {
                let off = head * hd;
                for i in 0..t {
                    let pos = start_pos + i; // attends to 0..=pos
                    let q = &qkv[i * 3 * d + off..i * 3 * d + off + hd];
                    for j in 0..=pos {
                        let kb = j * d + off;
                        scores[j] = crate::tensor::dot(q, &kc[kb..kb + hd]) * scale;
                    }
                    softmax(&mut scores[..=pos]);

                    let ob = i * d + off;
                    for j in 0..=pos {
                        let w = scores[j];
                        if w == 0.0 {
                            continue;
                        }
                        let vb = j * d + off;
                        let orow = &mut attn[ob..ob + hd];
                        let vrow = &vc[vb..vb + hd];
                        for e in 0..hd {
                            orow[e] += w * vrow[e];
                        }
                    }
                }
            }

            layer.attn_proj_w.linear(&attn, t, d, Some(&layer.attn_proj_b), &mut proj);
            for i in 0..t * d {
                x[i] += proj[i];
            }

            // --- feed-forward ---
            for i in 0..t {
                layernorm(&x[i * d..(i + 1) * d], &layer.ln2_w, &layer.ln2_b, cfg.eps, &mut xa[i * d..(i + 1) * d]);
            }
            layer.fc_w.linear(&xa, t, d, Some(&layer.fc_b), &mut h);
            for v in h.iter_mut() {
                *v = gelu(*v);
            }
            layer.fc_proj_w.linear(&h, t, cfg.n_inner, Some(&layer.fc_proj_b), &mut hout);
            for i in 0..t * d {
                x[i] += hout[i];
            }
        }

        cache.len = start_pos + t;

        // Final layer norm on the last position, then the tied LM head.
        let last = &x[(t - 1) * d..t * d];
        let mut normed = vec![0.0f32; d];
        layernorm(last, &self.ln_f_w, &self.ln_f_b, cfg.eps, &mut normed);
        let mut logits = vec![0.0f32; cfg.n_vocab];
        self.wte.linear(&normed, 1, d, None, &mut logits);
        logits
    }
}

fn take_bias(
    tensors: &mut HashMap<String, Tensor>,
    name: &str,
    n: usize,
) -> Result<Vec<f32>, String> {
    let t = tensors
        .remove(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    if t.data.len() != n {
        return Err(format!(
            "{name}: expected {n} values, found {} (shape {:?})",
            t.data.len(),
            t.shape
        ));
    }
    Ok(t.data)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WeightLayout {
    /// `[in_dim, out_dim]`, the original TensorFlow Conv1D layout.
    InOut,
    /// `[out_dim, in_dim]`, the usual PyTorch linear layout.
    OutIn,
}

fn take_weight_proj(
    tensors: &mut HashMap<String, Tensor>,
    name: &str,
    out_dim: usize,
    in_dim: usize,
    layout: WeightLayout,
    precision: Precision,
) -> Result<Weight, String> {
    let data = take_weight(tensors, name, out_dim, in_dim, layout)?;
    Weight::new(data, out_dim, in_dim, precision)
}

/// Load a projection weight as `[out_dim, in_dim]`, transposing when the
/// checkpoint uses the Conv1D layout.
fn take_weight(
    tensors: &mut HashMap<String, Tensor>,
    name: &str,
    out_dim: usize,
    in_dim: usize,
    layout: WeightLayout,
) -> Result<Vec<f32>, String> {
    let t = tensors
        .remove(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    let expected = match layout {
        WeightLayout::OutIn => vec![out_dim, in_dim],
        WeightLayout::InOut => vec![in_dim, out_dim],
    };
    if t.shape != expected {
        return Err(format!("{name}: shape {:?}, expected {expected:?}", t.shape));
    }
    if layout == WeightLayout::OutIn {
        return Ok(t.data);
    }
    let mut tr = vec![0.0f32; t.data.len()];
    for i in 0..in_dim {
        for o in 0..out_dim {
            tr[o * in_dim + i] = t.data[i * out_dim + o];
        }
    }
    Ok(tr)
}
