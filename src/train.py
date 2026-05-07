import argparse
import logging
import os
from dataclasses import asdict
from logging import log, INFO, WARNING

import wandb

from src.gaussians.training_config import TrainingConfig
from src.training.export import export
from src.training.multi_video_processor import MultiVideoProcessor
from src.training.trainer import GaussianTrainer

logging.basicConfig(level=logging.INFO)


def _init_wandb(config: TrainingConfig, args) -> "wandb.sdk.wandb_run.Run":
    """Initialize a W&B run, falling back to a disabled run if unavailable."""
    try:
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            mode=config.wandb_mode,
            name=config.wandb_run_name,
            tags=config.wandb_tags,
            config={**vars(args), **asdict(config)},
            settings=wandb.Settings(start_method="thread"),
        )
        log(INFO, f"W&B run initialized: mode={config.wandb_mode} url={run.url}")
        return run
    except Exception as e:
        log(WARNING, f"wandb.init failed ({e}); continuing with mode=disabled")
        return wandb.init(mode="disabled")


def main():
    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splatting from videos")
    parser.add_argument("--videos", nargs="+", required=True, help="Paths to video files")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory")
    parser.add_argument("--stride", type=int, default=10, help="Frame sampling stride")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--iterations", type=int, default=300000, help="Training iterations")
    parser.add_argument("--distributed", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--matcher", type=str, default="opencv", help="Matcher: opencv | loftr")

    # W&B flags
    parser.add_argument("--wandb_project", default="3d-gaussian-splatting")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument(
        "--wandb_mode",
        default=os.environ.get("WANDB_MODE", "online"),
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=None)

    args = parser.parse_args()

    config = TrainingConfig()
    config.iterations_per_video = args.iterations
    config.frame_stride = args.stride
    config.cache_dir = args.cache_dir
    config.distributed = args.distributed
    config.batch_size = args.batch_size
    config.matcher = args.matcher
    config.wandb_project = args.wandb_project
    config.wandb_entity = args.wandb_entity
    config.wandb_mode = args.wandb_mode
    config.wandb_run_name = args.wandb_run_name
    config.wandb_tags = args.wandb_tags

    run = _init_wandb(config, args)

    log(INFO, "Processing videos...")
    processor = MultiVideoProcessor(
        cache=args.cache_dir,
        device=args.device,
        matcher=config.matcher,
    )
    merged_data = processor.process_videos(args.videos, stride=args.stride, use_cache=True)

    log(INFO, f"Processed {len(args.videos)} videos")
    log(INFO, f"Total 3D points: {len(merged_data['points_3d'])}")

    try:
        run.log(
            {
                "sfm/n_points_total": int(len(merged_data["points_3d"])),
                "sfm/n_videos": len(args.videos),
                "sfm/n_views_total": int(sum(len(p) for p in merged_data["all_poses"])),
            },
            step=0,
        )
    except Exception as e:
        log(WARNING, f"wandb sfm summary log failed: {e}")

    log(INFO, "Starting training...")
    trainer = GaussianTrainer(config, device=args.device, wandb_run=run)
    gaussians = trainer.train(merged_data, args.output)

    output_model = os.path.join(args.output, "final_model.pth")
    ply_output = output_model.replace("pth", "ply")
    export(output_model, ply_output)

    log(INFO, "\n" + "=" * 60)
    log(INFO, f"Training complete! Model: {output_model}, Splat: {ply_output}")
    log(INFO, "=" * 60)

    # Upload final model + PLY as a W&B artifact
    try:
        artifact = wandb.Artifact(
            name=f"3dgs-model-{run.id}",
            type="model",
            metadata={
                "n_gaussians": int(gaussians.xyz.shape[0]),
                "iterations": int(config.iterations_per_video),
            },
        )
        artifact.add_file(output_model)
        artifact.add_file(ply_output)
        run.log_artifact(artifact)
    except Exception as e:
        log(WARNING, f"wandb artifact upload failed: {e}")

    run.finish()


if __name__ == "__main__":
    main()
