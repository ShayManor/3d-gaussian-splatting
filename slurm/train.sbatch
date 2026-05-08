#!/bin/bash -l
#SBATCH --job-name=3dgs-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o $SCRATCH/logs/3dgs-train_%j.out
#SBATCH -e $SCRATCH/logs/3dgs-train_%j.err

# Train a 3D Gaussian Splat from a single video. Override defaults via env:
#
#   VIDEO=./data/foo.mov STRIDE=3 ITERATIONS=150000 sbatch slurm/train.sbatch
#   MATCHER=opencv sbatch slurm/train.sbatch     # fallback matcher
#
# Output goes to runs/3dgs_j<JOBID>/ unless OUTPUT is set.

set -euo pipefail
source $SCRATCH/venvs/3dgs/bin/activate

export REPO=~/3d-gaussian-splatting
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
unset DISPLAY
cd "$REPO"

export WANDB_DIR="${WANDB_DIR:-$SCRATCH/wandb}"
mkdir -p "$WANDB_DIR"
export WANDB_MODE="${WANDB_MODE:-online}"

VIDEO="${VIDEO:-./data/input.mp4}"
OUTPUT="${OUTPUT:-runs/3dgs_j${SLURM_JOB_ID}}"
CACHE_DIR="${CACHE_DIR:-./cache}"
STRIDE="${STRIDE:-5}"
ITERATIONS="${ITERATIONS:-300000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MATCHER="${MATCHER:-loftr}"
WANDB_PROJECT="${WANDB_PROJECT:-3d-gaussian-splatting}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-3dgs_j${SLURM_JOB_ID}}"

[ ! -f "$VIDEO" ] && echo "ERROR: video not found: $VIDEO" && exit 1
mkdir -p "$OUTPUT" "$CACHE_DIR"

python -u -m src.train \
    --videos "$VIDEO" \
    --output "$OUTPUT" \
    --cache_dir "$CACHE_DIR" \
    --stride "$STRIDE" \
    --device cuda \
    --iterations "$ITERATIONS" \
    --batch_size "$BATCH_SIZE" \
    --matcher "$MATCHER" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_mode "$WANDB_MODE" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --wandb_tags train slurm "j${SLURM_JOB_ID}"

echo "[DONE] 3DGS train complete. Output: $OUTPUT  Video: $VIDEO  Stride: $STRIDE"
