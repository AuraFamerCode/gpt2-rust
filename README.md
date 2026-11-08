# gpt2 — GPT-2 from scratch in pure Rust

A complete GPT-2 inference implementation with no machine-learning frameworks.
Every piece is written here: the byte-level BPE tokenizer, the `.safetensors`
reader, the transformer forward pass, and the sampler. Only three small
pure-Rust crates are used (`serde_json` for the safetensors header, `rayon` for
threading the matmuls, `memmap2` to map the checkpoint) — no `tch`, no `candle`,
no `onnxruntime`, no Python.

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
  -n, --max-new-tokens <N> tokens to generate (default: 40)
  -t, --temp <F>           sampling temperature, 0 = greedy (default: 0.8)
      --top-k <N>          keep only the N most likely tokens (default: 40, 0 = off)
      --seed <N>           RNG seed (default: 42)
      --tokens             print the prompt's token ids and exit
      --logits             print the top next-token logits before sampling
```

Examples:

```sh
./target/release/gpt2 -n 80 --temp 0.7 --seed 7 "In the beginning there was"
./target/release/gpt2 --temp 0 --top-k 0 "The capital of France is"   # greedy
./target/release/gpt2 --tokens "Hello, world!"                        # [15496, 11, 995, 0]
```

Generation streams to stdout; model info and timings go to stderr, so
`./target/release/gpt2 ... 2>/dev/null` gives clean text.

## Layout

| file | what it does |
| --- | --- |
| `src/bpe.rs` | byte↔unicode table, GPT-2 pre-tokenizer, rank-ordered BPE merges |
| `src/safetensors.rs` | header parsing and f32/f16/bf16 decoding from a mapped file |
| `src/tensor.rs` | row-major `matmul`, softmax, LayerNorm, tanh-approximation GELU |
| `src/model.rs` | embeddings, 12 pre-LN blocks, causal attention, tied LM head, KV cache |
| `src/sampling.rs` | xorshift PRNG, temperature and top-k sampling |
| `src/main.rs` | CLI, streaming UTF-8 decoder, timing |

Notes on the implementation:

- **Weight layout.** The checkpoint stores TensorFlow-style Conv1D weights as
  `[in, out]`, while square projections are ambiguous by shape alone. The loader
  detects the convention once from `c_attn` (never square) and applies it to
  every projection, transposing into `[out, in]` for the kernels.
- **KV cache.** The prompt is prefilled in one batched pass; each new token then
  attends only to cached keys and values, so decoding costs one position of work
  instead of re-running the whole prefix.
- **GELU.** GPT-2 uses the tanh approximation (`gelu_new`), and LayerNorm uses
  biased variance with `eps = 1e-5`, matching the reference exactly.

## Verification

- Tokenizer: `"Hello, world!"` → `[15496, 11, 995, 0]` and contractions split as
  `["It", "'s", " 2024", ",", " isn", "'t", ...]`, matching the reference ids.
- Numerics: `scripts/reference.py` is an independent numpy implementation of the
  same forward pass. For the prompt `"The meaning of life is"` both agree to four
  decimals on the whole top-5, and the same holds after a prefill + cached decode.
- Output: sampled generations are fluent and on-topic; greedy decoding reproduces
  GPT-2's well-known repetition loop.

```sh
python3 scripts/reference.py 464 3616 286 1204 318   # numpy top-5
./target/release/gpt2 --logits -n 1 --temp 0 "The meaning of life is"
```

On an M-series laptop this runs at roughly 8–10 ms per token with all 12 layers
in fp32; the 50257-wide LM head is the single largest matmul per step.

## Limitations

fp32 only, GPT-2 small's head count is the default config (the loader infers
vocabulary, embedding width, layer count and feed-forward width from the
checkpoint, but `n_head` is a setting), and there is no batching across
sequences. It is an inference engine, not a training implementation.
