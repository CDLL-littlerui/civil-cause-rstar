#!/usr/bin/env bash
# Run the released method (method_v3, frozen config) on one split, end to end.
#
# usage: bash scripts/run_method_v3.sh <out_dir> [n_shards] [seed] [extra method args...]
#   e.g. bash scripts/run_method_v3.sh runs/main_s1 8 1
#        bash scripts/run_method_v3.sh runs/abl_no_mcts 8 1 --no_mcts
#
# env: PYTHON (default python3), GEN_MODEL / DISC_MODEL / BERT (HF ids or local dirs),
#      GEN_SCRIPT (default scripts/methods/method_v3.py), SPLIT (default test),
#      TAU / LAM (default 2.0 / 0.3, the frozen configuration), NGPU (default 8)
set -euo pipefail
OUT=${1:?usage: run_method_v3.sh <out_dir> [n_shards] [seed] [extra args...]}
SHARDS=${2:-8}
SEED=${3:-1}
shift 3 2>/dev/null || shift $#
EXTRA="$@"
PYTHON=${PYTHON:-python3}
GEN_SCRIPT=${GEN_SCRIPT:-scripts/methods/method_v3.py}
SPLIT=${SPLIT:-test}
TAU=${TAU:-2.0}
LAM=${LAM:-0.3}
GEN_MODEL=${GEN_MODEL:-Qwen/Qwen3.5-9B}
DISC_MODEL=${DISC_MODEL:-Qwen/Qwen3.5-4B}
BERT=${BERT:-dienstag/chinese-bert-wwm-ext}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=${PYTHONPATH:-.}
NGPU=${NGPU:-8}

mkdir -p "$OUT"
echo "[run] split=$SPLIT shards=$SHARDS seed=$SEED tau=$TAU lam=$LAM gen=$GEN_MODEL disc=$DISC_MODEL"
for ((i=0; i<SHARDS; i++)); do
  g=$(( i % NGPU ))
  d="$OUT/gen_s$i"
  mkdir -p "$d"
  CUDA_VISIBLE_DEVICES=$g $PYTHON $GEN_SCRIPT --stage gen --split "$SPLIT" \
      --out "$d" --seed "$SEED" --n-sim 16 --n-sim-lo 8 --tau "$TAU" --lam "$LAM" \
      --gen-model "$GEN_MODEL" --disc-model "$DISC_MODEL" --bert "$BERT" \
      --shard-index $i --shard-count $SHARDS $EXTRA \
      > "$d/gen.log" 2>&1 &
done
wait
echo "[run] gen done"

INPUTS=""
for ((i=0; i<SHARDS; i++)); do
  INPUTS="$INPUTS,$OUT/gen_s$i/paths_t${TAU%%.*}_l${LAM}.jsonl"
done
$PYTHON scripts/merge_shards.py --split "$SPLIT" --inputs "${INPUTS#,}" --out "$OUT/paths_all.jsonl"

CUDA_VISIBLE_DEVICES=0 $PYTHON $GEN_SCRIPT --stage disc --split "$SPLIT" \
    --out "$OUT" --inputs paths_all.jsonl --seed "$SEED" \
    --gen-model "$GEN_MODEL" --disc-model "$DISC_MODEL" --bert "$BERT" $EXTRA

$PYTHON scripts/eval_preds.py --preds "$OUT/preds_all.jsonl" --gold "data/$SPLIT.jsonl" \
    --out "$OUT/metrics.json" --tax-dir vocab
echo "[run] done -> $OUT/preds_all.jsonl  $OUT/metrics.json"

