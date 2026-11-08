//! Interactive streaming chat, keeping the conversation in the KV cache.
//!
//! GPT-2 is not instruction tuned, so the session is presented to it as a
//! written transcript (`User:` / `Assistant:`) that it continues. Every turn
//! reuses the keys and values already in the cache, so only the new text is
//! prefilled; when a turn no longer fits in the context window the oldest
//! tokens are dropped and the retained transcript is prefilled again.
//!
//! Each turn is timed: prefill (the prompt of this turn), decode (one forward
//! pass per generated token) and, with `--cache-report`, a counterfactual full
//! re-prefill of the whole transcript, which is what the cache saves.

use std::io::{BufRead, Write};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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

/// Timing for one turn.
#[derive(Clone, Copy, Debug)]
pub struct TurnStats {
    pub turn: usize,
    pub prefill_tokens: usize,
    pub prefill_ms: f64,
    /// Time spent rebuilding the cache after trimming the context, if it happened.
    pub rebuild_ms: Option<f64>,
    pub decode_tokens: usize,
    pub decode_forwards: usize,
    pub decode_ms: f64,
    pub context_tokens: usize,
    /// Tokens that would have been prefilled had the cache been discarded.
    pub saved_tokens: usize,
    /// Measured cost of prefilling the whole transcript from scratch.
    pub full_prefill_ms: Option<f64>,
}

impl TurnStats {
    fn decode_ms_per_token(&self) -> f64 {
        if self.decode_forwards == 0 {
            0.0
        } else {
            self.decode_ms / self.decode_forwards as f64
        }
    }
}

pub struct Session {
    cache: KvCache,
    /// Text whose tokens are in the cache, used to rebuild after trimming.
    transcript: String,
    /// Length of the cached prefix, in tokens.
    pos: usize,
    /// Most recent token ids, for the repetition penalty.
    recent: Vec<u32>,
    turns: usize,
}

impl Session {
    pub fn new(cfg: &Config) -> Self {
        Session {
            cache: KvCache::new(cfg),
            transcript: String::new(),
            pos: 0,
            recent: Vec::new(),
            turns: 0,
        }
    }

    pub fn reset(&mut self, cfg: &Config) {
        self.cache = KvCache::new(cfg);
        self.transcript.clear();
        self.pos = 0;
        self.recent.clear();
        self.turns = 0;
    }

    /// Write the conversation to `path` as a plain transcript. The preamble is
    /// implied and re-added on load, so the file holds only the exchange and
    /// stays readable (and editable). Returns (turns, cached tokens).
    pub fn save(&self, path: &str) -> Result<(usize, usize), String> {
        let body = conversation_body(&self.transcript);
        if body.trim().is_empty() {
            return Err("nothing to save yet".to_string());
        }
        std::fs::write(path, body).map_err(|e| format!("cannot write {path}: {e}"))?;
        Ok((self.turns, self.pos))
    }

    /// Replace the conversation with the transcript in `path`, prefilled into a
    /// fresh cache. Returns (context tokens, whether the head was trimmed).
    pub fn load(
        &mut self,
        model: &Gpt2,
        tok: &mut Tokenizer,
        path: &str,
    ) -> Result<(usize, bool), String> {
        let body =
            std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        if body.trim().is_empty() {
            return Err(format!("{path} is empty"));
        }

        let mut text = String::with_capacity(PREAMBLE.len() + body.len());
        text.push_str(PREAMBLE);
        text.push_str(&body);

        let mut ids = tok.encode(&text);
        let budget = model.cfg.n_ctx.saturating_sub(HISTORY_FLOOR);
        let trimmed = ids.len() > budget;
        if trimmed {
            ids.drain(..ids.len() - budget);
        }

        self.cache = KvCache::new(&model.cfg);
        if !ids.is_empty() {
            let logits = model.forward(&mut self.cache, &ids, 0);
            std::hint::black_box(&logits);
        }
        self.pos = ids.len();
        self.cache.len = self.pos;
        // Re-decoding the ids keeps text and cache position consistent, exactly
        // as the sliding-window rebuild does.
        self.transcript = tok.decode(&ids);
        self.recent = ids;
        trim_recent(&mut self.recent);
        self.turns = count_turns(&self.transcript);
        Ok((self.pos, trimmed))
    }

    /// Drop the oldest history when `need` more tokens would not fit, then
    /// rebuild the cache from the retained tail of the transcript. Returns how
    /// long the rebuild took, when one happened.
    fn ensure_capacity(
        &mut self,
        model: &Gpt2,
        tok: &mut Tokenizer,
        need: usize,
    ) -> Result<Option<f64>, String> {
        let cfg = model.cfg;
        if self.pos + need <= cfg.n_ctx {
            return Ok(None);
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
        let started = Instant::now();
        if !ids.is_empty() {
            let logits = model.forward(&mut self.cache, &ids, 0);
            std::hint::black_box(&logits);
        }
        let rebuild_ms = elapsed_ms(started);

        self.transcript = tok.decode(&ids);
        self.pos = ids.len();
        self.recent = ids;
        trim_recent(&mut self.recent);
        Ok(Some(rebuild_ms))
    }

    /// Feed one user message, stream the assistant's reply, and leave the cache
    /// positioned right after the reply.
    #[allow(clippy::too_many_arguments)]
    pub fn turn(
        &mut self,
        model: &Gpt2,
        tok: &mut Tokenizer,
        sampler: &Sampler,
        rng: &mut Rng,
        user: &str,
        max_new: usize,
        measure_full: bool,
        out: &mut impl Write,
    ) -> Result<TurnStats, String> {
        let cfg = model.cfg;
        let max_new = max_new.min(cfg.n_ctx.saturating_sub(HISTORY_FLOOR)).max(1);

        // Tokens already cached: what a cacheless run would have to redo.
        let saved_tokens = self.pos;
        self.turns += 1;

        let prefix = if self.pos == 0 {
            format!("{PREAMBLE}User: {user}\n{ASSISTANT}")
        } else if self.transcript.ends_with('\n') {
            format!("User: {user}\n{ASSISTANT}")
        } else {
            format!("\nUser: {user}\n{ASSISTANT}")
        };
        let tokens = tok.encode(&prefix);
        let rebuild_ms = self.ensure_capacity(model, tok, tokens.len() + max_new)?;

        let turn_start = self.pos;
        let started = Instant::now();
        let mut logits = model.forward(&mut self.cache, &tokens, turn_start);
        let prefill_ms = elapsed_ms(started);

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
        let mut decode_ms = 0.0;
        let mut decode_forwards = 0usize;

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
            let started = Instant::now();
            logits = model.forward(&mut self.cache, &[next], gen_pos);
            decode_ms += elapsed_ms(started);
            decode_forwards += 1;
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
        self.recent
            .truncate(self.recent.len() - (produced.len() - keep));

        // Counterfactual: what prefilling the whole transcript from scratch costs
        // now, i.e. the work the cache avoided this turn.
        let full_prefill_ms = if measure_full {
            let all = tok.encode(&self.transcript);
            let mut fresh = KvCache::new(&cfg);
            let started = Instant::now();
            let logits = model.forward(&mut fresh, &all, 0);
            let ms = elapsed_ms(started);
            std::hint::black_box(&logits);
            Some(ms)
        } else {
            None
        };

        Ok(TurnStats {
            turn: self.turns,
            prefill_tokens: tokens.len(),
            prefill_ms,
            rebuild_ms,
            decode_tokens: produced.len(),
            decode_forwards,
            decode_ms,
            context_tokens: self.pos,
            saved_tokens,
            full_prefill_ms,
        })
    }
}

pub fn run(
    model: &Gpt2,
    tok: &mut Tokenizer,
    sampler: Sampler,
    rng: &mut Rng,
    max_new: usize,
    cache_report: bool,
    out: &mut impl Write,
) -> Result<(), String> {
    let cfg = model.cfg;
    let mut session = Session::new(&cfg);
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let mut rows: Vec<(usize, f64, f64)> = Vec::new(); // context, cached, full

    let _ = writeln!(
        out,
        "gpt2 chat: {} layers, {} heads, d_model {} (context {} tokens)",
        cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.n_ctx
    );
    let _ = writeln!(
        out,
        "commands: /help, /reset, /save, /load, /list, /quit   (Ctrl-D to exit)\n"
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
        // Slash commands never reach the model.
        if let Some(rest) = text.strip_prefix('/') {
            let (cmd, arg) = match rest.split_once(char::is_whitespace) {
                Some((cmd, arg)) => (cmd, arg.trim()),
                None => (rest, ""),
            };
            match cmd {
                "quit" | "exit" => break,
                "help" => {
                    let _ = writeln!(
                        out,
                        "  /help        this text\n  /reset       forget the conversation (clears the KV cache)\n  /save <file> write the transcript to <file>\n  /load [file] replace the conversation with <file>, default: newest *.txt here\n  /list        saved transcripts here, newest first, with times and sizes\n  /quit        exit"
                    );
                }
                "reset" => {
                    session.reset(&cfg);
                    rows.clear();
                    let _ = writeln!(out, "(context cleared)");
                }
                "save" => {
                    if arg.is_empty() {
                        let _ = writeln!(out, "usage: /save <file>");
                    } else {
                        match session.save(arg) {
                            Ok((turns, tokens)) => {
                                let noun = if turns == 1 { "turn" } else { "turns" };
                                let _ = writeln!(
                                    out,
                                    "(saved {turns} {noun}, {tokens} context tokens to {arg})"
                                );
                            }
                            Err(e) => {
                                let _ = writeln!(out, "(not saved: {e})");
                            }
                        }
                    }
                }
                "load" => {
                    // No argument: resume the transcript saved most recently here.
                    let path = if arg.is_empty() {
                        match newest_transcript() {
                            Ok(path) => path,
                            Err(e) => {
                                let _ = writeln!(out, "(not loaded: {e})");
                                continue;
                            }
                        }
                    } else {
                        arg.to_string()
                    };
                    match session.load(model, tok, &path) {
                        Ok((tokens, trimmed)) => {
                            rows.clear();
                            let note = if trimmed {
                                " (trimmed to fit the context window)"
                            } else {
                                ""
                            };
                            let _ = writeln!(
                                out,
                                "(loaded {tokens} context tokens from {path}{note})"
                            );
                        }
                        Err(e) => {
                            let _ = writeln!(out, "(not loaded: {e})");
                        }
                    }
                }
                "list" | "ls" => match saved_transcripts() {
                    Ok(files) if files.is_empty() => {
                        let _ = writeln!(
                            out,
                            "(no saved transcripts here; /save <file> writes one)"
                        );
                    }
                    Ok(files) => {
                        let width = files.iter().map(|f| f.name.len()).max().unwrap_or(0).max(8);
                        let _ = writeln!(out, "saved transcripts in . (* = what a bare /load picks)");
                        for (i, f) in files.iter().enumerate() {
                            let mark = if i == 0 { '*' } else { ' ' };
                            let _ = writeln!(
                                out,
                                "  {mark} {:<width$}  {:>9}  {} UTC",
                                f.name,
                                format_size(f.size),
                                format_utc(f.modified),
                                width = width
                            );
                        }
                    }
                    Err(e) => {
                        let _ = writeln!(out, "(cannot list: {e})");
                    }
                },
                other => {
                    let _ = writeln!(out, "(unknown command /{other}, try /help)");
                }
            }
            continue;
        }

        let _ = write!(out, "gpt2> ");
        let _ = out.flush();
        let stats = session.turn(model, tok, &sampler, rng, text, max_new, cache_report, out)?;
        let _ = writeln!(out, "\n");
        report_turn(&stats);
        if let Some(full) = stats.full_prefill_ms {
            rows.push((stats.context_tokens, stats.prefill_ms, full));
        }
    }

    if !rows.is_empty() {
        print_cache_report(&rows);
    }
    Ok(())
}

/// Per-turn timings go to stderr, like the rest of the CLI's measurements, so
/// stdout stays pure conversation text.
fn report_turn(s: &TurnStats) {
    let mut line = format!(
        "[turn {}] prefill {} tok in {:.1} ms | decode {} tok in {:.0} ms ({:.1} ms/tok) | context {} tok | cache saved {} tok (~{:.1} ms at this rate)",
        s.turn,
        s.prefill_tokens,
        s.prefill_ms,
        s.decode_tokens,
        s.decode_ms,
        s.decode_ms_per_token(),
        s.context_tokens,
        s.saved_tokens,
        s.saved_tokens as f64 * ms_per_token(s.prefill_ms, s.prefill_tokens),
    );
    if let Some(rb) = s.rebuild_ms {
        line.push_str(&format!(" | rebuilt window in {:.0} ms", rb));
    }
    eprintln!("{line}");

    if let Some(full) = s.full_prefill_ms {
        let saved = (full - s.prefill_ms).max(0.0);
        let ratio = if s.prefill_ms > 0.0 {
            full / s.prefill_ms
        } else {
            0.0
        };
        eprintln!(
            "[turn {}] counterfactual: full prefill {:.1} ms vs cached {:.1} ms -> saved {:.1} ms ({:.1}x)",
            s.turn, full, s.prefill_ms, saved, ratio
        );
    }
}

fn print_cache_report(rows: &[(usize, f64, f64)]) {
    let first_ctx = rows.first().map(|r| r.0).unwrap_or(0);
    let last_ctx = rows.last().map(|r| r.0).unwrap_or(0);
    let total_cached: f64 = rows.iter().map(|r| r.1).sum();
    let total_full: f64 = rows.iter().map(|r| r.2).sum();

    eprintln!(
        "\nKV cache report: {} turns, context {} -> {} tok",
        rows.len(),
        first_ctx,
        last_ctx
    );
    eprintln!("  turn   context   cached prefill   full prefill   saving");
    for (i, (ctx, cached, full)) in rows.iter().enumerate() {
        let ratio = if *cached > 0.0 { full / cached } else { 0.0 };
        eprintln!(
            "  {:>4}   {:>7}   {:>11.1} ms  {:>11.1} ms   {:>5.1}x",
            i + 1,
            ctx,
            cached,
            full,
            ratio
        );
    }
    let ratio = if total_cached > 0.0 {
        total_full / total_cached
    } else {
        0.0
    };
    eprintln!(
        "  total   {:>10}   {:>11.1} ms  {:>11.1} ms   {:>5.1}x",
        "", total_cached, total_full, ratio
    );
}

fn ms_per_token(ms: f64, tokens: usize) -> f64 {
    if tokens == 0 {
        0.0
    } else {
        ms / tokens as f64
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
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

/// A transcript written by `/save` and still sitting in the working directory.
#[derive(Clone, Debug)]
struct SavedTranscript {
    name: String,
    modified: SystemTime,
    size: u64,
}

/// Every saved transcript in the working directory, newest first. Both `/list`
/// and a bare `/load` use this, so they can never disagree about what is there.
fn saved_transcripts() -> Result<Vec<SavedTranscript>, String> {
    let entries =
        std::fs::read_dir(".").map_err(|e| format!("cannot list current directory: {e}"))?;
    let mut files = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !is_txt(&path) {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        let Ok(head) = read_head(&path) else { continue };
        if !looks_like_transcript(&head) {
            continue;
        }
        files.push(SavedTranscript {
            name: entry.file_name().to_string_lossy().into_owned(),
            modified: meta.modified().unwrap_or(UNIX_EPOCH),
            size: meta.len(),
        });
    }
    sort_transcripts(&mut files);
    Ok(files)
}

fn sort_transcripts(files: &mut [SavedTranscript]) {
    files.sort_by(|a, b| {
        b.modified
            .cmp(&a.modified)
            .then_with(|| a.name.cmp(&b.name))
    });
}

/// The most recently modified transcript, for a bare `/load`.
fn newest_transcript() -> Result<String, String> {
    saved_transcripts()?
        .first()
        .map(|f| f.name.clone())
        .ok_or_else(|| {
            "no saved transcript here (expected a *.txt file starting with 'User: ')".to_string()
        })
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Modification time as `YYYY-MM-DD HH:MM` in UTC: formats without a date crate
/// and without guessing at the local offset.
fn format_utc(t: SystemTime) -> String {
    let secs = t
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    let rem = secs.rem_euclid(86_400);
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}",
        rem / 3600,
        (rem % 3600) / 60
    )
}

/// Howard Hinnant's days-to-civil algorithm (proleptic Gregorian).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn is_txt(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("txt"))
        .unwrap_or(false)
}

/// A `/save` transcript opens with a user line; ordinary text files do not.
fn looks_like_transcript(head: &str) -> bool {
    head.trim_start().starts_with("User: ")
}

fn read_head(path: &std::path::Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut buf = [0u8; 512];
    let mut file = std::fs::File::open(path)?;
    let n = file.read(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf[..n]).into_owned())
}

/// The conversation without the implied preamble, which is what `/save` writes.
fn conversation_body(transcript: &str) -> &str {
    transcript.strip_prefix(PREAMBLE).unwrap_or(transcript)
}

/// Turn count implied by a transcript, so timing labels stay continuous after
/// a `/load`.
fn count_turns(transcript: &str) -> usize {
    let body = conversation_body(transcript);
    if body.trim().is_empty() {
        0
    } else {
        1 + body.matches("\nUser: ").count()
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

    #[test]
    fn save_round_trip_keeps_the_conversation() {
        let transcript = format!("{PREAMBLE}User: hi\nAssistant: hello there\n");
        let body = conversation_body(&transcript);
        assert_eq!(body, "User: hi\nAssistant: hello there\n");
        // Reloading rebuilds exactly the text the session had.
        assert_eq!(format!("{PREAMBLE}{body}"), transcript);
        assert_eq!(count_turns(&transcript), 1);
        // A transcript that lost its preamble is left as it is.
        assert_eq!(conversation_body("User: hi\n"), "User: hi\n");
        assert_eq!(count_turns(""), 0);
    }

    #[test]
    fn counts_turns_across_a_conversation() {
        let transcript = format!(
            "{PREAMBLE}User: a\nAssistant: b\nUser: c\nAssistant: d\n"
        );
        assert_eq!(count_turns(&transcript), 2);
    }

    #[test]
    fn recognises_saved_transcripts() {
        assert!(looks_like_transcript("User: hi\nAssistant: hello\n"));
        assert!(looks_like_transcript("\n  User: hi\n"));
        assert!(!looks_like_transcript("plain prose\nUser: hi\n"));
        assert!(!looks_like_transcript(""));
        assert!(is_txt(std::path::Path::new("chat.txt")));
        assert!(is_txt(std::path::Path::new("chat.TXT")));
        assert!(!is_txt(std::path::Path::new("chat.md")));
    }

    #[test]
    fn formats_sizes_readably() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(1023), "1023 B");
        assert_eq!(format_size(1024), "1.0 KiB");
        assert_eq!(format_size(1536), "1.5 KiB");
        assert_eq!(format_size(2 * 1024 * 1024), "2.0 MiB");
    }

    #[test]
    fn formats_utc_timestamps() {
        let at = |secs: u64| UNIX_EPOCH + std::time::Duration::from_secs(secs);
        assert_eq!(format_utc(at(0)), "1970-01-01 00:00");
        assert_eq!(format_utc(at(1_700_000_000)), "2023-11-14 22:13");
        assert_eq!(format_utc(at(1_767_225_600)), "2026-01-01 00:00");
    }

    #[test]
    fn sorts_transcripts_newest_first() {
        let at = |secs: u64| UNIX_EPOCH + std::time::Duration::from_secs(secs);
        let make = |name: &str, secs: u64| SavedTranscript {
            name: name.to_string(),
            modified: at(secs),
            size: 1,
        };
        let mut files = vec![make("old.txt", 10), make("new.txt", 20), make("b.txt", 20)];
        sort_transcripts(&mut files);
        // Newest first, ties broken by name so the order is stable.
        assert_eq!(files[0].name, "b.txt");
        assert_eq!(files[1].name, "new.txt");
        assert_eq!(files[2].name, "old.txt");
    }

    #[test]
    fn stats_report_per_token_average() {
        let s = TurnStats {
            turn: 1,
            prefill_tokens: 12,
            prefill_ms: 6.0,
            rebuild_ms: None,
            decode_tokens: 21,
            decode_forwards: 20,
            decode_ms: 100.0,
            context_tokens: 40,
            saved_tokens: 0,
            full_prefill_ms: None,
        };
        assert!((s.decode_ms_per_token() - 5.0).abs() < 1e-9);
        // One forward per generated token, except the first (from prefill).
        assert_eq!(s.decode_forwards + 1, s.decode_tokens);
        assert!((ms_per_token(6.0, 12) - 0.5).abs() < 1e-9);
        assert_eq!(ms_per_token(6.0, 0), 0.0);
    }
}
