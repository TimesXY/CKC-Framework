import argparse
from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="Validate DINO model on a custom dataset")

    parser.add_argument(
        "--config_file",
        type=str,
        default="work_dir/dino-5scale_swin-l_8xb2-12e.py",
        help="Path to the DINO config file (default: work_dino/dino-5scale_swin-l_8xb2-12e.py)"
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="work_dir/best_coco_bbox_mAP_50.pth",
        help="Path to the checkpoint file to use for validation"
    )
    parser.add_argument(
        "--test_image_dir",
        type=str,
        default="dataset/test",
        help="Directory containing test images"
    )
    parser.add_argument(
        "--ann_file",
        type=str,
        default="dataset/annotations/instances_test.json",
        help="Path to the COCO-format annotation file for validation"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1,
        help="Number of foreground classes (default: 1)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for data loading"
    )

    return parser.parse_args()


def validate():
    args = parse_args()

    # Load configuration
    cfg = Config.fromfile(args.config_file)

    # Update number of classes
    cfg.model.bbox_head.num_classes = args.num_classes

    # Configure validation dataset
    cfg.val_dataloader.dataset.ann_file = args.ann_file
    cfg.val_dataloader.dataset.data_prefix = {'img': args.test_image_dir}

    # Configure evaluation metrics
    cfg.val_evaluator = dict(
        type='CocoMetric',
        ann_file=args.ann_file,
        metric=['bbox'],
        format_only=False,
        backend_args=None
    )

    # Configure test dataset and evaluator
    cfg.test_dataloader = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset=dict(
            type='CocoDataset',
            ann_file=args.ann_file,
            data_prefix=dict(img=args.test_image_dir),
            metainfo=dict(classes=('foreground',)),
            pipeline=cfg.val_dataloader.dataset.pipeline,
            test_mode=True
        )
    )
    cfg.test_evaluator = cfg.val_evaluator  # Use same evaluator for test

    # Remove EMAHook if it exists
    if hasattr(cfg, 'custom_hooks'):
        cfg.custom_hooks = [hook for hook in cfg.custom_hooks if hook.get('type') != 'EMAHook']

    # Initialize runner
    runner = Runner.from_cfg(cfg)

    # Load checkpoint
    runner.load_checkpoint(args.checkpoint_file)

    # Execute validation
    results = runner.test()

    # Print COCO metrics
    print("Validation results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")


if __name__ == '__main__':
    print("Starting validation...")
    validate()
