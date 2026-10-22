#!/usr/bin/env bash
# bench one sglang endpoint at concurrency 1 and 32, print summary blocks only
set -u
URL=$1
TAG=$2
V=/home/mattia-bosi/venv/sglang-fork
for C in 1 32; do
  echo "########## $TAG conc=$C"
  $V/bin/python -m sglang.bench_serving --backend sglang --base-url "$URL" \
    --model Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B \
    --dataset-name random --num-prompts 64 \
    --random-input-len 512 --random-output-len 256 \
    --max-concurrency "$C" 2>/dev/null \
    | grep -E "Request throughput|Output token throughput|Total token throughput|Mean TTFT|Mean TPOT|Median TPOT|Mean E2E|Benchmark duration"
done
