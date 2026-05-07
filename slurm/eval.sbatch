#!/bin/bash -l
#SBATCH --job-name=3dgs-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o $SCRATCH/logs/3dgs-eval_%j.out
#SBATCH -e $SCRATCH/logs/3dgs-eval_%j.err

# Render views from a trained 3DGS model, side-by-side with the source frames,
# and re-export the .ply. Auto-discovers the latest run unless MODEL_DIR is set.
#
#   MODEL_DIR=runs/3dgs_j12345 sbatch slurm/eval.sbatch
#   VIDEO=./data/foo.mov sbatch slurm/eval.sbatch    # picks ./cache/foo_sfm.pkl
#   NUM_VIEWS=24 sbatch slurm/eval.sbatch

set -euo pipefail
source $SCRATCH/venvs/3dgs/bin/activate

export REPO=~/3d-gaussian-splatting
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
unset DISPLAY
cd "$REPO"

# W&B credentials (mirror train.sbatch: file -> env -> offline fallback)
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$SCRATCH/.wandb_api_key" ]; then
  export WANDB_API_KEY=$(cat "$SCRATCH/.wandb_api_key")
fi
export WANDB_DIR="${WANDB_DIR:-$SCRATCH/wandb}"
mkdir -p "$WANDB_DIR"
WANDB_MODE="${WANDB_MODE:-online}"
[ -z "${WANDB_API_KEY:-}" ] && WANDB_MODE=offline && echo "[wandb] no API key -> offline mode"
WANDB_PROJECT="${WANDB_PROJECT:-3d-gaussian-splatting}"

# Auto-discover latest run (override: MODEL_DIR=<path> sbatch ...)
MODEL_DIR="${MODEL_DIR:-}"
if [ -z "$MODEL_DIR" ]; then
  MODEL_DIR=$(ls -td runs/3dgs_j*/final_model.pth output/final_model.pth 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
  [ -z "$MODEL_DIR" ] && echo "ERROR: No trained 3DGS run found." && exit 1
  echo "[auto] Using model dir: $MODEL_DIR"
fi
[ ! -f "$MODEL_DIR/final_model.pth" ] && echo "ERROR: $MODEL_DIR/final_model.pth not found." && exit 1

# Pick SfM cache from the video stem (default: input -> cache/input_sfm.pkl)
VIDEO="${VIDEO:-./data/input.mp4}"
VIDEO_STEM=$(basename "$VIDEO")
VIDEO_STEM="${VIDEO_STEM%.*}"
SFM_CACHE="${SFM_CACHE:-./cache/${VIDEO_STEM}_sfm.pkl}"
OUTPUT="${OUTPUT:-$MODEL_DIR/eval}"
NUM_VIEWS="${NUM_VIEWS:-12}"

[ ! -f "$SFM_CACHE" ] && echo "ERROR: SfM cache not found: $SFM_CACHE (run train first)" && exit 1
mkdir -p "$OUTPUT"

WANDB_RUN_NAME="${WANDB_RUN_NAME:-3dgs_eval_j${SLURM_JOB_ID}}"

python -u -m src.eval \
    --model "$MODEL_DIR/final_model.pth" \
    --sfm-cache "$SFM_CACHE" \
    --output "$OUTPUT" \
    --num-views "$NUM_VIEWS" \
    --device cuda \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_mode "$WANDB_MODE" \
    --wandb_run_name "$WANDB_RUN_NAME"

echo "[DONE] 3DGS eval complete. Renders + PLY: $OUTPUT  Model: $MODEL_DIR"
