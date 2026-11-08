//! Interactive streaming chat, keeping the conversation in the KV cache.
//!
//! GPT-2 is not instruction tuned, so the session is presented to it as a
//! written transcript (`User:` / `Assistant:`) that it continues. Every turn
//! reuses the keys and values already in the cache, so only the new text is
//! prefilled; when a turn no longer fits in the context window the oldest
//! tokens are dropped and the retained transcript is prefilled again.

use std::io::{BufRead, Write};

use crate::bpe::{StreamDecoder, Tokenizer};
use crate::model::{Config, Gpt2, KvCache};
use crate::sampling::{Rng, Sampler};

/// Opening context. GPT-2 is not instruction tuned, so it needs a transcript to
/// continue rather than raw prose to finish — and one worked example turn
/// anchors the format far better than a bare preamble (without it the model
/// answers "French is our capital.").
const PREAMBLE: &str = "The following is a conversation between a user and a helpful AI assistant.\n\n\
User: What is the capital of Italy?\nAssistant: The capital of Italy is Rome.\n";
/// Assistant label the model continues from.
const ASSISTANT: &str = "Assistant:";
/// Generation stops when the model starts writing the next turn itself.
const STOP: &str = "\nUser:";
/// Context tokens kept for the repetition penalty.
const RECENT: usize = 256;
/// Positions always left free for conversation history.
const HISTORY_FLOOR: usize = 64;

pub struct Session {
    cache: KvCache,
    /// Text whose tokens are in the cache, used to rebuild after trimming.
    transcript: String,
    /// Length of the cached prefix, in tokens.
    pos: usize,
    /// Most recent token ids, for the repetition penalty.
    recent: Vec<u32>,
}

impl Session {
    pub fn new(cfg: &Config) -> Self {
        Session {
            cache: KvCache::new(cfg),
            transcript: String::new(),
            pos: 0,
            recent: Vec::new(),
        }
    }

    pub fn reset(&mut self, cfg: &Config) {
        self.cache = KvCache::new(cfg);
        self.transcript.clear();
        self.pos = 0;
        self.recent.clear();
    }

    pub fn context_tokens(&self) -> usize {
        self.pos
    }

    /// Drop the oldest history when `need` more tokens would not fit, then
    /// rebuild the cache from the retained tail of the transcript.
    fn ensure_capacity(
        &mut self,
        model: &Gpt2,
        tok: &mut Tokenizer,
        need: usize,
    ) -> Result<(), String> {
        let cfg = model.cfg;
        if self.pos + need <= cfg.n_ctx {
            return Ok(());
        }
        let budget = cfg
            .n_ctx
            .checked_sub(need)
            .filter(|b| *b > 0)
            .ok_or("input is too long for the context window")?;

        let mut ids = tok.encode(&self.transcript);
        if ids.len() > budget {
            let drop = ids.len() - budget;
            ids.drain(..drop);
        }

        self.cache = KvCache::new(&cfg);
        if !ids.is_empty() {
            let logits = model.forward(&mut self.cache, &ids, 0);
            std::hint::black_box(&logits);
        }
        self.transcript = tok.decode(&ids);
        self.pos = ids.len();
        self.recent = ids;
        trim_recent(&mut self.recent);
        Ok(())
    }

    /// Feed one user message, stream the assistant's reply, and leave the cache
    /// positioned right after the reply.
    pub fn turn(
        &mut self,
        model: &Gpt2,
        tok: &mut Tokenizer,
        sampler: &Sampler,
        rng: &mut Rng,
        user: &str,
        max_new: usize,
        out: &mut impl Write,
    ) -> Result<(), String> {
        let cfg = model.cfg;
        let max_new = max_new
            .min(cfg.n_ctx.saturating_sub(HISTORY_FLOOR))
            .max(1);

        let prefix = if self.pos == 0 {
            format!("{PREAMBLE}User: {user}\n{ASSISTANT}")
        } else if self.transcript.ends_with('\n') {
            format!("User: {user}\n{ASSISTANT}")
        } else {
            format!("\nUser: {user}\n{ASSISTANT}")
        };
        let tokens = tok.encode(&prefix);
        self.ensure_capacity(model, tok, tokens.len() + max_new)?;

        let turn_start = self.pos;
        let mut logits = model.forward(&mut self.cache, &tokens, turn_start);
        let base = turn_start + tokens.len(); // position of the first new token
        self.pos = base;
        self.transcript.push_str(&prefix);
        self.recent.extend_from_slice(&tokens);
        trim_recent(&mut self.recent);

        let eos = tok.eos_id();
        let mut decoder = StreamDecoder::new();
        let mut text = String::new(); // decoded reply so far
        let mut starts: Vec<usize> = Vec::with_capacity(max_new); // byte start per token
        let mut produced: Vec<u32> = Vec::with_capacity(max_new);
        let mut printed = 0usize;
        let mut first = true;
        let mut stop_at: Option<usize> = None;
        let mut gen_pos = base;

        for _ in 0..max_new {
            let next = sampler.sample(&mut logits, &self.recent, rng);
            if Some(next) == eos {
                break;
            }
            let piece = decoder.push(tok, next);
            starts.push(text.len());
            text.push_str(&piece);
            produced.push(next);
            self.recent.push(next);

            if let Some(found) = find_stop(&text, STOP, printed.saturating_sub(STOP.len())) {
                write_chunk(out, &text[printed..found], &mut first);
                printed = found;
                stop_at = Some(found);
                break;
            }

            // Print everything that can no longer become part of a stop marker.
            let safe = safe_prefix(&text, printed, STOP.len());
            if safe > printed {
                write_chunk(out, &text[printed..safe], &mut first);
                printed = safe;
            }
            let _ = out.flush();

            if produced.len() == max_new {
                break;
            }
            logits = model.forward(&mut self.cache, &[next], gen_pos);
            gen_pos += 1;
        }

        if stop_at.is_none() && text.len() > printed {
            write_chunk(out, &text[printed..], &mut first);
        }
        let _ = out.flush();

        // Rewind the cache to just before the stop marker: the token containing
        // its first byte is dropped so the next turn can supply its own header.
        let cut = stop_at.unwrap_or(text.len());
        let keep = match stop_at {
            Some(b) => token_index_containing(&starts, b).min(produced.len()),
            None => produced.len(),
        };
        self.pos = base + keep;
        self.cache.len = self.pos;
        self.transcript.push_str(&text[..cut]);
        self.recent.truncate(self.recent.len() - (produced.len() - keep));
        Ok(())
    }
}

pub fn run(
    model: &Gpt2,
    tok: &mut Tokenizer,
    sampler: Sampler,
    rng: &mut Rng,
    max_new: usize,
    out: &mut impl Write,
) -> Result<(), String> {
    let cfg = model.cfg;
    let mut session = Session::new(&cfg);
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();

    let _ = writeln!(
        out,
        "gpt2 chat: {} layers, {} heads, d_model {} (context {} tokens)",
        cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.n_ctx
    );
    let _ = writeln!(
        out,
        "commands: /help, /reset, /quit   (Ctrl-D to exit)\n"
    );

    loop {
        let _ = write!(out, "you> ");
        let _ = out.flush();
        let Some(line) = lines.next() else { break };
        let line = line.map_err(|e| format!("read stdin: {e}"))?;
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        match text {
            "/quit" | "/exit" => break,
            "/help" => {
                let _ = writeln!(
                    out,
                    "  /help   this text\n  /reset  forget the conversation (clears the KV cache)\n  /quit   exit"
                );
                continue;
            }
            "/reset" => {
                session.reset(&cfg);
                let _ = writeln!(out, "(context cleared)");
                continue;
            }
            _ => {}
        }

        let _ = write!(out, "gpt2> ");
        let _ = out.flush();
        session.turn(model, tok, &sampler, rng, text, max_new, out)?;
        let _ = writeln!(out, "\n      [{} context tokens]", session.context_tokens());
    }
    Ok(())
}

/// Writes a reply chunk, dropping leading whitespace on the first one: the
/// model's first token after `Assistant:` almost always carries a space.
fn write_chunk(out: &mut impl Write, chunk: &str, first: &mut bool) {
    if *first {
        let trimmed = chunk.trim_start();
        if trimmed.is_empty() {
            return;
        }
        let _ = write!(out, "{trimmed}");
        *first = false;
    } else {
        let _ = write!(out, "{chunk}");
    }
}

fn trim_recent(recent: &mut Vec<u32>) {
    if recent.len() > RECENT {
        let drop = recent.len() - RECENT;
        recent.drain(..drop);
    }
}

/// Byte offset of `stop` at or after `from`, rounded down to a char boundary.
fn find_stop(text: &str, stop: &str, from: usize) -> Option<usize> {
    let mut start = from.min(text.len());
    while start > 0 && !text.is_char_boundary(start) {
        start -= 1;
    }
    text[start..].find(stop).map(|i| i + start)
}

/// How much of `text` is safe to print without risking a partial stop marker.
fn safe_prefix(text: &str, printed: usize, stop_len: usize) -> usize {
    let mut safe = text
        .len()
        .saturating_sub(stop_len.saturating_sub(1))
        .max(printed);
    while safe > printed && !text.is_char_boundary(safe) {
        safe -= 1;
    }
    safe
}

/// Index of the token whose byte span contains `byte`.
fn token_index_containing(starts: &[usize], byte: usize) -> usize {
    starts.partition_point(|&s| s <= byte).saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_stop_marker() {
        assert_eq!(find_stop("hi\nUser: there", STOP, 0), Some(2));
        assert_eq!(find_stop("no marker here", STOP, 0), None);
        assert_eq!(find_stop("a\nUser: b\nUser: c", STOP, 0), Some(1));
        assert_eq!(find_stop("a\nUser: b\nUser: c", STOP, 3), Some(9));
    }

    #[test]
    fn holds_back_a_partial_marker() {
        // "abc\nUse" may still become "\nUser:", so only "ab" is safe.
        assert_eq!(safe_prefix("abc\nUse", 0, STOP.len()), 2);
        // Printing lags by stop_len - 1 bytes; the end of the turn flushes the
        // remainder, so nothing is lost, only delayed.
        assert_eq!(safe_prefix("abcdefghij", 0, STOP.len()), 5);
        // Never re-prints bytes already emitted.
        assert_eq!(safe_prefix("abcdefghij", 8, STOP.len()), 8);
    }

    #[test]
    fn maps_marker_to_token_index() {
        let starts = [0, 4, 9, 15];
        assert_eq!(token_index_containing(&starts, 0), 0);
        assert_eq!(token_index_containing(&starts, 4), 1);
        assert_eq!(token_index_containing(&starts, 8), 1);
        assert_eq!(token_index_containing(&starts, 20), 3);
    }
}
