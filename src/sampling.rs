//! Temperature / top-k sampling with a small xorshift PRNG.

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the zero state.
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// `temp <= 0` means greedy decoding.
pub fn sample(logits: &[f32], temp: f32, top_k: usize, rng: &mut Rng) -> u32 {
    if temp <= 0.0 {
        return argmax(logits);
    }

    let mut scored: Vec<(f32, u32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (l / temp, i as u32))
        .collect();

    if top_k > 0 && top_k < scored.len() {
        scored.select_nth_unstable_by(top_k - 1, |a, b| b.0.total_cmp(&a.0));
        scored.truncate(top_k);
    }

    let max = scored.iter().fold(f32::NEG_INFINITY, |m, s| m.max(s.0));
    let mut sum = 0.0f32;
    for s in scored.iter_mut() {
        s.0 = (s.0 - max).exp();
        sum += s.0;
    }

    let mut r = rng.next_f32() * sum;
    for &(p, id) in &scored {
        r -= p;
        if r <= 0.0 {
            return id;
        }
    }
    scored.last().map(|s| s.1).unwrap_or(0)
}
