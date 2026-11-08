#!/usr/bin/env bash
# What does the KV cache save as a conversation grows?
#
# Feeds a scripted multi-turn conversation into the chat REPL with
# --cache-report, which times each turn's cached prefill and, for comparison, a
# full re-prefill of the whole transcript (what a cacheless run would pay).
# Per-turn timings and the summary table go to stderr, so only those are shown.
#
# usage: scripts/chat_cache_bench.sh [turns] [max_new_tokens]
set -euo pipefail
cd "$(dirname "$0")/.."

TURNS="${1:-6}"
MAX_NEW="${2:-60}"
BIN=./target/release/gpt2

if [ ! -x "$BIN" ]; then
  echo "build first: cargo build --release" >&2
  exit 1
fi

prompts=(
  "What is the capital of France?"
  "Tell me about the ocean."
  "What is the capital of Italy?"
  "Tell me about the mountains."
  "What is a lighthouse for?"
  "Tell me about the desert."
  "What is the capital of Spain?"
  "Tell me about the forest."
)

tmp=$(mktemp)
for ((i = 0; i < TURNS && i < ${#prompts[@]}; i++)); do
  printf '%s\n' "${prompts[i]}"
done >"$tmp"
printf '/quit\n' >>"$tmp"

"$BIN" --chat --cache-report -n "$MAX_NEW" --seed 5 <"$tmp" 2>&1 >/dev/null | grep -v '^$'
rm -f "$tmp"
