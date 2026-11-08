//! Decoding: temperature, top-k, nucleus (top-p) sampling and a repetition
//! penalty, plus the xorshift PRNG that drives them.
//!
//! Order of operations matches the usual reference pipeline: repetition penalty
//! on the raw logits, then temperature, then top-k, then top-p, then a sampled
//! draw from the surviving tokens.

use std::collections::HashSet;

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

#[derive(Clone, Copy, Debug)]
pub struct Sampler {
    /// Sampling temperature; `<= 0` means greedy decoding.
    pub temp: f32,
    /// Keep only the `k` most likely tokens; `0` disables.
    pub top_k: usize,
    /// Keep the smallest set of tokens whose probabilities sum to `top_p`;
    /// `1.0` disables.
    pub top_p: f32,
    /// CTRL-style penalty applied once per distinct token already in the
    /// context: positive logits are divided by it, negative ones multiplied;
    /// `1.0` disables.
    pub repetition_penalty: f32,
}

impl Sampler {
    fn penalize(&self, logits: &mut [f32], history: &[u32]) {
        if (self.repetition_penalty - 1.0).abs() < 1e-6 || history.is_empty() {
            return;
        }
        let mut seen: HashSet<u32> = HashSet::with_capacity(history.len());
        for &id in history {
            // Apply once per distinct token, not once per occurrence.
            if !seen.insert(id) {
                continue;
            }
            if let Some(logit) = logits.get_mut(id as usize) {
                *logit = if *logit > 0.0 {
                    *logit / self.repetition_penalty
                } else {
                    *logit * self.repetition_penalty
                };
            }
        }
    }

    /// `logits` is modified in place (penalty), so callers must own a fresh
    /// copy per step — which the forward pass produces anyway.
    pub fn sample(&self, logits: &mut [f32], history: &[u32], rng: &mut Rng) -> u32 {
        self.penalize(logits, history);
        if self.temp <= 0.0 {
            return argmax(logits);
        }

        let mut scored: Vec<(f32, u32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (l / self.temp, i as u32))
            .collect();

        if self.top_k > 0 && self.top_k < scored.len() {
            scored.select_nth_unstable_by(self.top_k - 1, |a, b| b.0.total_cmp(&a.0));
            scored.truncate(self.top_k);
        }

        // Softmax over the surviving candidates.
        let max = scored.iter().fold(f32::NEG_INFINITY, |m, s| m.max(s.0));
        let mut sum = 0.0f32;
        for s in scored.iter_mut() {
            s.0 = (s.0 - max).exp();
            sum += s.0;
        }

        // Nucleus: smallest descending prefix whose mass reaches top_p. Always
        // keeps at least one token, so a very small top_p degrades to greedy.
        if self.top_p < 1.0 && scored.len() > 1 {
            scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            let mut cum = 0.0f32;
            let mut keep = 0usize;
            for (i, s) in scored.iter().enumerate() {
                cum += s.0;
                keep = i + 1;
                if cum / sum >= self.top_p {
                    break;
                }
            }
            scored.truncate(keep.max(1));
        }

        let total: f32 = scored.iter().map(|s| s.0).sum();
        let mut r = rng.next_f32() * total;
        for &(p, id) in &scored {
            r -= p;
            if r <= 0.0 {
                return id;
            }
        }
        scored.last().map(|s| s.1).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampler(temp: f32, top_k: usize, top_p: f32, penalty: f32) -> Sampler {
        Sampler {
            temp,
            top_k,
            top_p,
            repetition_penalty: penalty,
        }
    }

    #[test]
    fn greedy_picks_argmax() {
        let mut logits = vec![0.5, 3.0, -1.0, 2.0];
        let mut rng = Rng::new(1);
        assert_eq!(sampler(0.0, 0, 1.0, 1.0).sample(&mut logits, &[], &mut rng), 1);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut rng = Rng::new(7);
        for seed in 0..20 {
            let mut logits = vec![0.5, 3.0, 2.9, 2.8];
            let id = sampler(1.0, 1, 1.0, 1.0).sample(&mut logits, &[], &mut rng);
            assert_eq!(id, 1, "seed {seed}");
        }
    }

    #[test]
    fn tiny_top_p_keeps_only_the_best_token() {
        let mut rng = Rng::new(3);
        for _ in 0..50 {
            let mut logits = vec![1.0, 4.0, 3.9, 0.1];
            let id = sampler(1.0, 0, 1e-6, 1.0).sample(&mut logits, &[], &mut rng);
            assert_eq!(id, 1);
        }
    }

    #[test]
    fn top_p_excludes_the_tail() {
        // 0.7 / 0.2 / 0.1 -> nucleus at 0.8 is the first two tokens only.
        let logits = vec![0.7f32.ln(), 0.2f32.ln(), 0.1f32.ln()];
        let s = sampler(1.0, 0, 0.8, 1.0);
        let mut rng = Rng::new(11);
        for _ in 0..200 {
            let mut l = logits.clone();
            let id = s.sample(&mut l, &[], &mut rng);
            assert!(id != 2, "tail token survived top-p");
        }
    }

    #[test]
    fn repetition_penalty_flips_the_argmax() {
        let mut logits = vec![3.0, 2.9];
        let mut rng = Rng::new(5);
        // Without penalty token 0 wins; once it is in the context it loses.
        assert_eq!(sampler(0.0, 0, 1.0, 1.0).sample(&mut logits.clone(), &[], &mut rng), 0);
        assert_eq!(
            sampler(0.0, 0, 1.0, 2.0).sample(&mut logits.clone(), &[0], &mut rng),
            1
        );
    }

    #[test]
    fn repetition_penalty_applies_once_per_distinct_token() {
        let mut a = vec![3.0, 2.0];
        let mut b = vec![3.0, 2.0];
        let mut rng = Rng::new(9);
        let s = sampler(0.0, 0, 1.0, 1.5);
        s.sample(&mut a, &[0], &mut rng);
        s.sample(&mut b, &[0, 0, 0], &mut rng);
        assert_eq!(a, b, "duplicate history entries compounded the penalty");
    }
}
