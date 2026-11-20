#!/usr/bin/env bash
# LLaMA-Berry 风格 Self-Refine MCTS 基线，表 5。
# usage: bash scripts/run_berry.sh <out_dir> [n_shards] [seed] [budget]
# env: PYTHON / GEN_MODEL / NGPU / SPLIT
set -euo pipefail
OUT=${1:?usage: run_berry.sh <out_dir> [n_shards] [seed] [budget]}
SHARDS=${2:-4}
SEED=${3:-1}
BUDGET=${4:-8}
PYTHON=${PYTHON:-python3}
SPLIT=${SPLIT:-test}
GEN_MODEL=${GEN_MODEL:-Qwen/Qwen3.5-9B}
NGPU=${NGPU:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=${PYTHONPATH:-.}
mkdir -p "$OUT"
for ((i=0; i<SHARDS; i++)); do
  g=$(( i % NGPU ))
  d="$OUT/gen_s$i"; mkdir -p "$d"
  CUDA_VISIBLE_DEVICES=$g $PYTHON scripts/methods/berry.py --split "$SPLIT" --budget "$BUDGET" \
      --out "$d" --seed "$SEED" --shard-index $i --shard-count $SHARDS \
      --gen-model "$GEN_MODEL" > "$d/gen.log" 2>&1 &
done
wait
INPUTS=""
for ((i=0; i<SHARDS; i++)); do INPUTS="$INPUTS,$OUT/gen_s$i/preds.jsonl"; done
$PYTHON scripts/merge_shards.py --split "$SPLIT" --inputs "${INPUTS#,}" --out "$OUT/preds.jsonl"
$PYTHON scripts/eval_preds.py --preds "$OUT/preds.jsonl" --gold "data/$SPLIT.jsonl" \
    --out "$OUT/metrics.json" --tax-dir vocab
echo "[run] done -> $OUT/preds.jsonl  $OUT/metrics.json"