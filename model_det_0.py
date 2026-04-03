import argparse
import torch
import os

from mmengine.config import Config
from mmengine.runner import Runner
from utils.utils_down import download_checkpoint
from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description="Train DINO on a custom dataset")

    # Config file path
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/dino/dino-5scale_swin-l_8xb2-12e.py",
        help="Path to DINO config file (default: configs/dino/dino-5scale_swin-l_8xb2-12e.py)"
    )

    # Working directory (logs, checkpoints, and pretrained weights)
    parser.add_argument(
        "--work_dir",
        type=str,
        default="work_dir",
        help="Directory to save logs, checkpoints, and pretrained weights"
    )

    # Training and validation annotation JSON
    parser.add_argument(
        "--train_ann",
        type=str,
        default="dataset/annotations/instances_train.json",
        help="Path to training annotation JSON file"
    )
    parser.add_argument(
        "--val_ann",
        type=str,
        default="dataset/annotations/instances_val.json",
        help="Path to validation annotation JSON file"
    )

    # Training and validation image directories
    parser.add_argument(
        "--train_img_dir",
        type=str,
        default="dataset/train",
        help="Directory containing training images"
    )
    parser.add_argument(
        "--val_img_dir",
        type=str,
        default="dataset/val",
        help="Directory containing validation images"
    )

    # Pretrained weights URL
    parser.add_argument(
        "--pretrained_url",
        type=str,
        default=(
            "https://download.openmmlab.com/mmdetection/v3.0/dino/"
            "dino-5scale_swin-l_8xb2-12e_coco/"
            "dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth"
        ),
        help="URL of pretrained DINO weights"
    )

    # Training batch size and number of workers
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training and validation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create working directory if it does not exist
    os.makedirs(args.work_dir, exist_ok=True)

    # Load configuration
    cfg = Config.fromfile(args.config_file)

    # Define dataset classes
    classes = ('foreground',)  # Adjust according to your dataset
    num_classes = len(classes)

    # Set working directory
    cfg.work_dir = args.work_dir

    # Set dataset type
    cfg.dataset_type = 'CocoDataset'
    cfg.data_root = ''

    # Configure training dataset
    cfg.train_dataloader.dataset = dict(
        type=cfg.dataset_type,
        data_root='',
        ann_file=args.train_ann,
        data_prefix=dict(img=args.train_img_dir),
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        metainfo=dict(classes=classes),
        pipeline=cfg.train_dataloader.dataset.pipeline
    )

    # Configure validation dataset
    cfg.val_dataloader.dataset = dict(
        type=cfg.dataset_type,
        data_root='',
        ann_file=args.val_ann,
        data_prefix=dict(img=args.val_img_dir),
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        test_mode=True,
        metainfo=dict(classes=classes),
        pipeline=cfg.val_dataloader.dataset.pipeline
    )

    # Use validation dataset as test dataset
    cfg.test_dataloader = cfg.val_dataloader

    # Set batch size and number of workers
    cfg.train_dataloader.batch_size = args.batch_size
    cfg.val_dataloader.batch_size = args.batch_size
    cfg.train_dataloader.num_workers = args.num_workers
    cfg.val_dataloader.num_workers = args.num_workers

    # Update the model's number of classes
    cfg.model.bbox_head.num_classes = num_classes

    # Configure checkpoint hook
    cfg.default_hooks = cfg.get('default_hooks', {})
    cfg.default_hooks.checkpoint = dict(
        type='CheckpointHook',
        interval=1,  # Save every epoch
        save_best='coco/bbox_mAP_50',  # Monitor best bbox mAP
        rule='greater',
        max_keep_ckpts=3,
        save_last=True
    )

    # Download pretrained weights into work_dir
    local_checkpoint = download_checkpoint(args.pretrained_url, args.work_dir)

    # Configure evaluators
    cfg.val_evaluator = dict(
        type='CocoMetric',
        ann_file=args.val_ann,
        metric=['bbox'],
        format_only=False,
        backend_args=None
    )
    cfg.test_evaluator = cfg.val_evaluator

    # Add meta information for classes
    cfg.metainfo = dict(classes=classes)

    # Optimize Dynamo cache
    setup_cache_size_limit_of_dynamo()

    # Build Runner instance
    runner = Runner.from_cfg(cfg)

    # Load pretrained weights
    checkpoint = torch.load(local_checkpoint, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)

    # Remove 'module.' prefix if present
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

    # Exclude certain layers (classification/regression heads)
    excluded_layers = [
        'bbox_head.cls_branches',
        'bbox_head.reg_branches',
        'dn_query_generator.label_embedding'
    ]

    # Filter state_dict to remove excluded layers
    filtered_state_dict = {k: v for k, v in new_state_dict.items()
                           if not any(k.startswith(layer) for layer in excluded_layers)}

    # Load weights into the model
    missing_keys, unexpected_keys = runner.model.load_state_dict(filtered_state_dict, strict=False)

    # Print missing/unexpected keys
    if missing_keys:
        print(f"Missing keys ({len(missing_keys)}): {missing_keys}")
    else:
        print("No missing keys.")
    if unexpected_keys:
        print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")
    else:
        print("No unexpected keys.")

    # Start training
    runner.train()


if __name__ == '__main__':
    main()
