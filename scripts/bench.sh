#!/usr/bin/env bash
# Decode throughput comparison: fp32 vs int8 weights.
#
# Times `--bench`, which runs a fixed number of decode steps with a constant
# input token, so both precisions do identical work and early EOS cannot skew
# the measurement. Runs are interleaved (f32, int8, f32, int8, ...) so thermal
# drift or background load does not favour one precision.
#
# usage: scripts/bench.sh [decode_steps] [reps]
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${1:-200}"
REPS="${2:-5}"
BIN=./target/release/gpt2
PROMPT="The meaning of life is"

if [ ! -x "$BIN" ]; then
  echo "build first: cargo build --release" >&2
  exit 1
fi

f32_runs=()
int8_runs=()
for _ in $(seq "$REPS"); do
  for prec in f32 int8; do
    ms=$("$BIN" --precision "$prec" --bench "$STEPS" "$PROMPT" 2>&1 >/dev/null \
      | grep -oE '[0-9.]+ ms/token' | grep -oE '[0-9.]+')
    if [ -z "$ms" ]; then
      echo "no timing from $prec run" >&2
      exit 1
    fi
    if [ "$prec" = f32 ]; then
      f32_runs+=("$ms")
    else
      int8_runs+=("$ms")
    fi
  done
done

median() {
  printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'
}

f32_med=$(median "${f32_runs[@]}")
int8_med=$(median "${int8_runs[@]}")
printf 'f32   median %6s ms/token  (%6.1f tok/s)  runs: %s\n' \
  "$f32_med" "$(awk -v ms="$f32_med" 'BEGIN {print 1000/ms}')" "${f32_runs[*]}"
printf 'int8  median %6s ms/token  (%6.1f tok/s)  runs: %s\n' \
  "$int8_med" "$(awk -v ms="$int8_med" 'BEGIN {print 1000/ms}')" "${int8_runs[*]}"
awk -v a="$f32_med" -v b="$int8_med" 'BEGIN {printf "speedup: %.2fx\n", a/b}'
