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
#
# IMPORTANT — submission-env propagation: `FOCAL_35MM=24 sbatch ...` relies on
# slurm exporting the caller's environment to the job. Some clusters default to
# --export=NONE, and `#!/bin/bash -l` re-runs login profiles that may clobber
# custom vars. Both modes have bit us before, so the focal-length defaults
# below are hard-coded to the iPhone 17 main camera (the actual capture device)
# — overriding via env still works, but the job never silently falls back to
# the 1.2*max(W,H) heuristic just because env didn't propagate.

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
# Big artifacts (checkpoints, PLYs) live on $SCRATCH — $HOME has tight quota.
OUTPUT="${OUTPUT:-$SCRATCH/3dgs/runs/3dgs_j${SLURM_JOB_ID}}"
CACHE_DIR="${CACHE_DIR:-$SCRATCH/3dgs/cache}"
# Stride between sampled frames. At 60fps phone capture, stride=30 gives one
# sample every ~0.5s — enough hand-motion parallax (~30-80 px) for SfM to
# triangulate. Stride=5 was producing 83ms-apart pairs with ~3-8 px parallax
# (sub-triangulation threshold), causing 98% pair rejection and 24-pose SfM.
# nerfstudio's ns-process-data targets ~300 total frames; for a 6184-frame
# video that maps to stride ~20. We use 30 for safer margin on slow motion.
STRIDE="${STRIDE:-30}"
ITERATIONS="${ITERATIONS:-300000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
# Matcher choice. Default: sift (deterministic, reproducible, CPU-only).
# Override to loftr (neural, faster, CUDA non-deterministic — different SfM
# each run) or opencv (ORB; weak on small images, kept for tests).
MATCHER="${MATCHER:-sift}"
INITIAL_GAUSSIANS="${INITIAL_GAUSSIANS:-10000}"
DENSIFY_CLONE_RATIO="${DENSIFY_CLONE_RATIO:-0.1}"
DENSIFY_PRUNE_RATIO="${DENSIFY_PRUNE_RATIO:-2.0}"
SCALE_CLAMP_RATIO="${SCALE_CLAMP_RATIO:-0.2}"
# Focal override. Default to iPhone 17 main (24mm 35mm-equivalent). To use a
# different camera, set FOCAL_35MM=X. To use the f=1.2*max(W,H) heuristic
# instead, set FOCAL_35MM=0 explicitly. The `${VAR-default}` form (no colon)
# means "default only if unset" — an empty env value is NOT overwritten — so
# `FOCAL_35MM= sbatch ...` would also fall through to the heuristic.
FOCAL_PX="${FOCAL_PX-}"
FOCAL_35MM="${FOCAL_35MM-24}"
# 0 is a sentinel meaning "no override, use heuristic".
[ "$FOCAL_35MM" = "0" ] && FOCAL_35MM=""
WANDB_PROJECT="${WANDB_PROJECT:-3d-gaussian-splatting}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-3dgs_j${SLURM_JOB_ID}}"

# Loud startup log so future runs are diagnosable from the slurm output alone.
echo "[ENV] VIDEO='$VIDEO' STRIDE=$STRIDE BATCH_SIZE=$BATCH_SIZE MATCHER=$MATCHER" >&2
echo "[ENV] FOCAL_PX='${FOCAL_PX}'" >&2
echo "[ENV] FOCAL_35MM='${FOCAL_35MM}'" >&2
echo "[ENV] DENSIFY_CLONE_RATIO=$DENSIFY_CLONE_RATIO DENSIFY_PRUNE_RATIO=$DENSIFY_PRUNE_RATIO" >&2

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
    --initial_gaussians "$INITIAL_GAUSSIANS" \
    --densify_clone_extent_ratio "$DENSIFY_CLONE_RATIO" \
    --densify_prune_extent_ratio "$DENSIFY_PRUNE_RATIO" \
    --scale_clamp_ratio "$SCALE_CLAMP_RATIO" \
    ${FOCAL_PX:+--focal_px "$FOCAL_PX"} \
    ${FOCAL_35MM:+--focal_35mm "$FOCAL_35MM"} \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_mode "$WANDB_MODE" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --wandb_tags train slurm "j${SLURM_JOB_ID}"

echo "[DONE] 3DGS train complete. Output: $OUTPUT  Video: $VIDEO  Stride: $STRIDE"
