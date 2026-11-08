//! GPT-2's byte-level BPE tokenizer, written from scratch.
//!
//! Three stages, matching the reference implementation:
//!   1. split the text with GPT-2's pre-tokenizer pattern,
//!   2. map the UTF-8 bytes of each piece onto printable unicode stand-ins,
//!   3. greedily apply byte-pair merges in rank order.

use serde_json::Value;
use std::collections::HashMap;
use std::fs;

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    inv: Vec<String>,
    byte_encoder: Vec<char>,
    byte_decoder: HashMap<char, u8>,
    /// Interning tables so merge ranks can be keyed by `(u32, u32)`.
    sym_ids: HashMap<String, u32>,
    sym_strs: Vec<String>,
    ranks: HashMap<(u32, u32), usize>,
    eos: Option<u32>,
}

impl Tokenizer {
    pub fn load(vocab_path: &str, merges_path: &str) -> Result<Self, String> {
        let raw = fs::read_to_string(vocab_path)
            .map_err(|e| format!("read {vocab_path}: {e}"))?;
        let json: Value =
            serde_json::from_str(&raw).map_err(|e| format!("parse {vocab_path}: {e}"))?;
        let obj = json
            .as_object()
            .ok_or("vocab.json must be a JSON object")?;

        let mut vocab = HashMap::with_capacity(obj.len());
        let mut max_id = 0u32;
        for (token, id) in obj {
            let id = id.as_u64().ok_or("vocab ids must be integers")? as u32;
            max_id = max_id.max(id);
            vocab.insert(token.clone(), id);
        }
        let mut inv = vec![String::new(); max_id as usize + 1];
        for (token, &id) in &vocab {
            inv[id as usize] = token.clone();
        }

        let mut sym_ids: HashMap<String, u32> = HashMap::new();
        let mut sym_strs: Vec<String> = Vec::new();
        let mut ranks: HashMap<(u32, u32), usize> = HashMap::new();

        let merges = fs::read_to_string(merges_path)
            .map_err(|e| format!("read {merges_path}: {e}"))?;
        let mut rank = 0usize;
        for line in merges.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split(' ');
            let (Some(a), Some(b)) = (parts.next(), parts.next()) else {
                continue;
            };
            let ia = intern(&mut sym_ids, &mut sym_strs, a);
            let ib = intern(&mut sym_ids, &mut sym_strs, b);
            ranks.insert((ia, ib), rank);
            rank += 1;
        }

        let byte_encoder = bytes_to_unicode();
        let byte_decoder = byte_encoder
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();

        let eos = vocab.get("<|endoftext|>").copied();

        Ok(Tokenizer {
            vocab,
            inv,
            byte_encoder,
            byte_decoder,
            sym_ids,
            sym_strs,
            ranks,
            eos,
        })
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos
    }

    pub fn len(&self) -> usize {
        self.inv.len()
    }

    fn intern(&mut self, s: String) -> u32 {
        if let Some(&id) = self.sym_ids.get(&s) {
            return id;
        }
        let id = self.sym_strs.len() as u32;
        self.sym_ids.insert(s.clone(), id);
        self.sym_strs.push(s);
        id
    }

    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut syms: Vec<u32> = Vec::new();

        for piece in pretokenize(text) {
            syms.clear();
            for byte in piece.as_bytes() {
                let c = self.byte_encoder[*byte as usize].to_string();
                syms.push(self.intern(c));
            }

            // Repeatedly merge the adjacent pair with the lowest rank.
            loop {
                let mut best: Option<(usize, u32, u32)> = None;
                for w in syms.windows(2) {
                    if let Some(&r) = self.ranks.get(&(w[0], w[1])) {
                        if best.map_or(true, |(br, _, _)| r < br) {
                            best = Some((r, w[0], w[1]));
                        }
                    }
                }
                let Some((_, a, b)) = best else { break };

                let merged_str =
                    format!("{}{}", self.sym_strs[a as usize], self.sym_strs[b as usize]);
                let merged = self.intern(merged_str);

                let mut next = Vec::with_capacity(syms.len());
                let mut i = 0;
                while i < syms.len() {
                    if i + 1 < syms.len() && syms[i] == a && syms[i + 1] == b {
                        next.push(merged);
                        i += 2;
                    } else {
                        next.push(syms[i]);
                        i += 1;
                    }
                }
                syms = next;
            }

            for &sym in &syms {
                let word = &self.sym_strs[sym as usize];
                match self.vocab.get(word) {
                    Some(&id) => ids.push(id),
                    None => {
                        // Not reachable for a well-formed vocab; fall back to bytes.
                        let singles: Vec<u32> = word
                            .chars()
                            .filter_map(|c| self.vocab.get(&c.to_string()).copied())
                            .collect();
                        ids.extend(singles);
                    }
                }
            }
        }
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(&self.token_bytes(id));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Raw bytes represented by a token.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        match self.inv.get(id as usize) {
            Some(s) => s
                .chars()
                .filter_map(|c| self.byte_decoder.get(&c).copied())
                .collect(),
            None => Vec::new(),
        }
    }
}

fn intern(map: &mut HashMap<String, u32>, strs: &mut Vec<String>, s: &str) -> u32 {
    if let Some(&id) = map.get(s) {
        return id;
    }
    let id = strs.len() as u32;
    map.insert(s.to_string(), id);
    strs.push(s.to_string());
    id
}

/// Reversible byte <-> printable unicode table used by GPT-2.
fn bytes_to_unicode() -> Vec<char> {
    let mut bs: Vec<u32> = Vec::new();
    for b in 33u32..=126 {
        bs.push(b);
    }
    for b in 161u32..=172 {
        bs.push(b);
    }
    for b in 174u32..=255 {
        bs.push(b);
    }
    let mut cs: Vec<u32> = bs.clone();
    let mut extra = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + extra);
            extra += 1;
        }
    }
    let mut table = vec!['\0'; 256];
    for (&b, &c) in bs.iter().zip(cs.iter()) {
        table[b as usize] = char::from_u32(c).unwrap();
    }
    table
}

/// GPT-2's pre-tokenizer:
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
///
/// Hand-rolled because the pattern uses a negative lookahead. Unicode letter and
/// number classes are approximated with `char::is_alphabetic` / `is_numeric`,
/// which agree with `\p{L}` / `\p{N}` on ordinary text.
fn pretokenize(text: &str) -> Vec<&str> {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

    let ch: Vec<char> = text.chars().collect();
    let idx: Vec<usize> = text.char_indices().map(|(b, _)| b).collect();
    let n = ch.len();
    let byte = |i: usize| if i < n { idx[i] } else { text.len() };

    let mut out: Vec<&str> = Vec::new();
    let mut i = 0usize;

    while i < n {
        let start = i;
        let c = ch[i];

        // 1. English contractions.
        if c == '\'' {
            let mut hit = 0;
            for pat in CONTRACTIONS {
                let pl = pat.chars().count();
                if i + pl <= n && ch[i..i + pl].iter().copied().eq(pat.chars()) {
                    hit = pl;
                    break;
                }
            }
            if hit > 0 {
                i += hit;
                out.push(&text[byte(start)..byte(i)]);
                continue;
            }
        }

        // 2. Optional single space glued to letters, numbers or punctuation.
        if c == ' ' && i + 1 < n {
            let d = ch[i + 1];
            let class = if d.is_alphabetic() {
                Some(0)
            } else if d.is_numeric() {
                Some(1)
            } else if !d.is_whitespace() {
                Some(2)
            } else {
                None
            };
            if let Some(k) = class {
                i += 1;
                while i < n && matches_class(ch[i], k) {
                    i += 1;
                }
                out.push(&text[byte(start)..byte(i)]);
                continue;
            }
        }

        // 3. Bare letters, numbers, punctuation.
        if c.is_alphabetic() || c.is_numeric() {
            let k = if c.is_alphabetic() { 0 } else { 1 };
            while i < n && matches_class(ch[i], k) {
                i += 1;
            }
            out.push(&text[byte(start)..byte(i)]);
            continue;
        }

        if !c.is_whitespace() {
            while i < n && matches_class(ch[i], 2) {
                i += 1;
            }
            out.push(&text[byte(start)..byte(i)]);
            continue;
        }

        // 4. Whitespace: all but the last char of a run, so the final space can
        //    attach to the following word (`\s+(?!\S)`).
        let mut j = i;
        while j < n && ch[j].is_whitespace() {
            j += 1;
        }
        let run = j - i;
        let take = if run >= 2 { run - 1 } else { run };
        i += take;
        out.push(&text[byte(start)..byte(i)]);
    }

    out
}

fn matches_class(c: char, class: u8) -> bool {
    match class {
        0 => c.is_alphabetic(),
        1 => c.is_numeric(),
        _ => !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric(),
    }
}
