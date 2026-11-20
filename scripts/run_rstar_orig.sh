#!/usr/bin/env bash
# 原始 rStar 基线（整题一次 MCTS、开放作答），表 5。
# usage: bash scripts/run_rstar_orig.sh <out_dir> [n_shards] [seed]
# env: PYTHON / GEN_MODEL / DISC_MODEL / NGPU / SPLIT / K
set -euo pipefail
OUT=${1:?usage: run_rstar_orig.sh <out_dir> [n_shards] [seed]}
SHARDS=${2:-8}
SEED=${3:-1}
PYTHON=${PYTHON:-python3}
SPLIT=${SPLIT:-test}
K=${K:-8}
GEN_MODEL=${GEN_MODEL:-Qwen/Qwen3.5-9B}
DISC_MODEL=${DISC_MODEL:-Qwen/Qwen3.5-4B}
NGPU=${NGPU:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=${PYTHONPATH:-.}
mkdir -p "$OUT"
for ((i=0; i<SHARDS; i++)); do
  g=$(( i % NGPU ))
  d="$OUT/gen_s$i"; mkdir -p "$d"
  CUDA_VISIBLE_DEVICES=$g $PYTHON scripts/methods/rstar_orig.py --stage gen --split "$SPLIT" \
      --k "$K" --out "$d" --seed "$SEED" --shard-index $i --shard-count $SHARDS \
      --gen-model "$GEN_MODEL" --disc-model "$DISC_MODEL" > "$d/gen.log" 2>&1 &
done
wait
INPUTS=""
for ((i=0; i<SHARDS; i++)); do INPUTS="$INPUTS,$OUT/gen_s$i/paths.jsonl"; done
$PYTHON scripts/merge_shards.py --split "$SPLIT" --inputs "${INPUTS#,}" --out "$OUT/paths.jsonl"
# 鉴别/互一致性阶段必须单进程：全局随机数语义
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/methods/rstar_orig.py --stage disc --split "$SPLIT" \
    --k "$K" --out "$OUT" --seed "$SEED" --gen-model "$GEN_MODEL" --disc-model "$DISC_MODEL"
$PYTHON scripts/eval_preds.py --preds "$OUT/preds.jsonl" --gold "data/$SPLIT.jsonl" \
    --out "$OUT/metrics.json" --tax-dir vocab
echo "[run] done -> $OUT/preds.jsonl  $OUT/metrics.json"