# gpt2 — GPT-2 from scratch in pure Rust

A complete GPT-2 inference implementation with no machine-learning frameworks.
Every piece is written here: the byte-level BPE tokenizer, the `.safetensors`
reader, the transformer forward pass, int8 quantization with SIMD integer
kernels, and the sampler. Only three small pure-Rust crates are used
(`serde_json` for the safetensors header, `rayon` for threading the matmuls,
`memmap2` to map the checkpoint) — no `tch`, no `candle`, no `onnxruntime`, no
Python.

## Getting the weights

GPT-2 small (124M) from the original OpenAI release, repackaged by HuggingFace:

```sh
mkdir -p models/gpt2 && cd models/gpt2
base=https://huggingface.co/gpt2/resolve/main
curl -LO $base/model.safetensors   # 548 MB, fp32
curl -LO $base/vocab.json
curl -LO $base/merges.txt
curl -LO $base/config.json
```

## Building and running

```sh
cargo build --release
./target/release/gpt2 "The meaning of life is"
```

```
usage: gpt2 [OPTIONS] [PROMPT]

options:
  --model <DIR>            directory with model.safetensors, vocab.json, merges.txt
                           (default: models/gpt2)
  -n, --max-new-tokens <N> tokens to generate (default: 40, chat: 120)
      --chat               interactive REPL; keeps the conversation in the KV cache
  -t, --temp <F>           sampling temperature, 0 = greedy (default: 0.8)
      --top-k <N>          keep only the N most likely tokens (default: 40, 0 = off)
      --top-p <F>          nucleus sampling: keep the smallest set of tokens whose
                           probability sums to F (default: 1.0 = off)
      --repetition-penalty <F>
                           penalize tokens already in the context (default: 1.0 = off)
      --seed <N>           RNG seed (default: 42)
      --precision <P>      f32 (exact) or int8 (quantized, ~1.8x faster decode)
                           (default: int8)
      --tokens             print the prompt's token ids and exit
      --logits             print the top next-token logits before sampling
      --bench <N>          time N decode steps with a fixed token and exit
  -h, --help               show this help
```

Examples:

```sh
./target/release/gpt2 -n 80 --temp 0.7 --seed 7 "In the beginning there was"
./target/release/gpt2 --precision f32 --temp 0 "The capital of France is"   # exact, greedy
./target/release/gpt2 --precision int8 --logits "Once upon a time"          # inspect candidates
./target/release/gpt2 --tokens "Hello, world!"                              # [15496, 11, 995, 0]
./target/release/gpt2 --chat                                                # REPL, cached context
./target/release/gpt2 --top-k 0 --top-p 0.9 --temp 0.9 "The ocean at night is"  # nucleus
./target/release/gpt2 --temp 0 --repetition-penalty 1.3 "Alan Turing was a"      # no repeat loop
```

Generation streams to stdout; model info and timings go to stderr, so
`./target/release/gpt2 ... 2>/dev/null` gives clean text.

## Chat mode

Run with no prompt (or with `--chat`) on a terminal for an interactive REPL:

```
$ ./target/release/gpt2

gpt2 chat: 12 layers, 12 heads, d_model 768 (context 1024 tokens)
commands: /help, /reset, /quit   (Ctrl-D to exit)

you> What is the capital of France?
gpt2> The capital of France is Paris.
      [56 context tokens]

you> And which river runs through it?
gpt2> The river of Brescia.
      [76 context tokens]
```

That second answer is wrong, and it is worth being clear why: GPT-2 small is a
2019 base model, not an instruction-tuned assistant. The streaming, the
stop-marker handling and the cached-context plumbing are all real, but factual
answers from 124M parameters are hit-and-miss — the first answer carries over
correctly, the specific recall does not.

How the session works:

- The conversation lives in the KV cache. Each turn prefills only that turn's
  `User:` / `Assistant:` text, so a reply costs the same whether the history is
  100 or 900 tokens long (the bracketed number is the cached token count).
- Generation stops when the model starts writing the next turn itself (`\nUser:`).
  The cache is then rewound to the token before that marker, so the next turn
  appends cleanly instead of nesting one transcript inside another. Bytes that
  could still become part of the marker are held back, so no partial `\nUser:`
  is ever printed.
- When a turn no longer fits in the context window, the oldest tokens are dropped
  and the retained transcript is prefilled again — a sliding window, rebuilt only
  when needed.
- `/reset` clears the conversation and cache, `/help` lists commands, `/quit` (or
  Ctrl-D) exits. `-n` sets the reply budget (120 by default in chat), and the
  sampling flags apply per turn.

## Performance

Decode is weight-bandwidth bound: every generated token reads the entire
network, so cutting weight bytes 4x matters more than peak FLOPs. With int8
weights the inner products also become integer and run on NEON `SDOT`
(16 int8 multiply-adds per instruction, versus 4 for an f32 FMA).

Measured on an Apple M4 (10 cores, `dotprod` available): 200 decode steps after
a 5-token prompt, median of 5 interleaved runs (`scripts/bench.sh 200 5`; the
runs alternate fp32/int8 so thermal drift cannot favour one of them):

| weights | resident weights | decode | throughput | prefill (5 tokens) |
| --- | --- | --- | --- | --- |
| fp32 | 498 MB | 8.6 ms/token | ~116 tok/s | ~34 ms |
| int8 | 143 MB | 4.7 ms/token | ~215 tok/s | ~14 ms |

That is 1.85x on decode and 2.4x on prefill. Timing uses `--bench`, which runs a
fixed number of decode steps with a constant input token. Timing *sampled*
generation instead is unreliable: runs stop early when they sample EOS and the
sequence length changes the per-token cost (that method reported 2.3x here,
overstating the gain).

Peak RSS is about 1.1 GB either way, because loading still streams the fp32
checkpoint and quantizes on the fly. What shrinks is the resident weight
footprint, 498 MB -> 143 MB (the 15 MB above one byte per parameter is the block
scales).

### How the quantization works

- **Block-wise scales.** Weights and activations are quantized symmetrically to
  int8 in blocks of 32 values, each block carrying its own f32 scale (`Q8_0`
  style). One scale per row is not enough: transformer activations have a few
  outlier channels, and with per-row scales those outliers eat most of the 127
  levels and turn the output into word salad — which is exactly what happened
  before this was fixed.
- **Integer accumulation.** A dot product becomes `sum(a_q * w_q)` in int32
  (127 × 127 × 32 cannot overflow), multiplied by both block scales once per
  block. The `SDOT` kernel keeps a four-lane float accumulator so the
  cross-lane reduction happens once per output row, not per block.
- **Fallback.** Without `dotprod` the same math runs through a portable scalar
  loop that auto-vectorizes; the CLI reports which kernel is active
  (`kernel neon-sdot`).
- **What stays fp32.** LayerNorm, GELU, softmax, attention and the KV cache.
  `--precision f32` remains bit-for-bit the reference path.

Model sizes load the same way in both precisions; quantization happens at load
time from the fp32 checkpoint, so no separate quantized checkpoint is needed.

## Layout

| file | what it does |
| --- | --- |
| `src/bpe.rs` | byte↔unicode table, GPT-2 pre-tokenizer, rank-ordered BPE merges |
| `src/safetensors.rs` | header parsing and f32/f16/bf16 decoding from a mapped file |
| `src/tensor.rs` | row-major f32 `matmul`, softmax, LayerNorm, tanh-approximation GELU |
| `src/quant.rs` | block-wise int8 quantization, `SDOT`/scalar integer kernels |
| `src/model.rs` | embeddings, 12 pre-LN blocks, causal attention, tied LM head, KV cache |
| `src/sampling.rs` | xorshift PRNG, temperature, top-k, top-p and repetition penalty |
| `src/chat.rs` | REPL session that keeps the conversation in the KV cache |
| `src/main.rs` | CLI, streaming UTF-8 decoder, timing |

Notes on the implementation:

- **Weight layout.** The checkpoint stores TensorFlow-style Conv1D weights as
  `[in, out]`, while square projections are ambiguous by shape alone. The loader
  detects the convention once from `c_attn` (never square) and applies it to
  every projection, transposing into `[out, in]` for the kernels.
- **KV cache.** The prompt is prefilled in one batched pass; each new token then
  attends only to cached keys and values, so decoding costs one position of work
  instead of re-running the whole prefix. Prefill also uses the int8 path, with
  activations quantized per token.
- **GELU.** GPT-2 uses the tanh approximation (`gelu_new`), and LayerNorm uses
  biased variance with `eps = 1e-5`, matching the reference exactly.
- **Sampling.** Repetition penalty (CTRL-style, applied once per distinct
  context token), then temperature, then top-k, then nucleus (top-p), then the
  draw. Both new knobs default to `1.0` (off), so existing seeds and outputs are
  unchanged; top-p is what stops low-temperature sampling from admitting the
  junk tail that top-k alone lets through.

## Verification

- Tokenizer: `"Hello, world!"` → `[15496, 11, 995, 0]` and contractions split as
  `["It", "'s", " 2024", ",", " isn", "'t", ...]`, matching the reference ids.
- Numerics (fp32): `scripts/reference.py` is an independent numpy implementation
  of the same forward pass. For `"The meaning of life is"` both agree to four
  decimals on the whole top-5, and the same holds after a prefill + cached
  decode.
- int8 kernels: `cargo test` checks that the integer kernels reproduce their own
  quantized math exactly, and that the quantized matmul stays within 2% RMS of
  fp32 on randomized inputs.
- int8 quality: the top-5 next tokens for the reference prompt stay in the same
  order with nearly the same probabilities (`" not"` 0.141 vs 0.109 in fp32), and
  generations remain fluent; greedy decoding still reproduces GPT-2's
  repetition loop.

```sh
python3 scripts/reference.py 464 3616 286 1204 318   # numpy top-5 (fp32)
./target/release/gpt2 --precision f32 --logits -n 1 --temp 0 "The meaning of life is"
cargo test --release
```

## Limitations

int8 quantization is weight-and-activation, block-wise: it is fast and stays
close to fp32, but it is not lossless, so use `--precision f32` when exactness
matters. There is no GPU path, no batching across sequences, and `n_head` is a
setting (the loader infers vocabulary, embedding width, layer count and
feed-forward width from the checkpoint). It is an inference engine, not a
training implementation.
