import argparse
import logging
from logging import log, INFO

from src.gaussians.training_config import TrainingConfig
from src.training.export import export
from src.training.multi_video_processor import MultiVideoProcessor
from src.training.trainer import GaussianTrainer

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Train 3D Gaussian Splatting from videos')
    parser.add_argument('--videos', nargs='+', required=True, help='Paths to video files')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Cache directory')
    parser.add_argument('--stride', type=int, default=10, help='Frame sampling stride')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--iterations', type=int, default=30000, help='Training iterations')
    parser.add_argument('--distributed', type=bool, default=False, help='Distributed training among multiple GPUs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')

    args = parser.parse_args()

    config = TrainingConfig()
    config.iterations_per_video = args.iterations
    config.frame_stride = args.stride
    config.cache_dir = args.cache_dir
    config.distributed = args.distributed
    config.batch_size = args.batch_size

    log(INFO, "Processing videos...")
    processor = MultiVideoProcessor(
        cache=args.cache_dir,
        device=args.device
    )

    merged_data = processor.process_videos(
        args.videos,
        stride=args.stride,
        use_cache=True
    )

    print(f"Processed {len(args.videos)} videos")
    print(f"Total 3D points: {len(merged_data['points_3d'])}")
    print("Starting training...")

    trainer = GaussianTrainer(config, device=args.device)
    gaussians = trainer.train(merged_data, args.output)
    ply_output = args.output.replace('pth', 'ply')
    print("\n" + "=" * 60)
    print(f"Training complete! Model saved to {args.output}, Splat saved to {ply_output}")
    print("=" * 60)

    export(args.output, ply_output)


if __name__ == "__main__":
    main()
