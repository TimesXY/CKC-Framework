import os
import torch
import argparse
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader

from utils.utils_model import CKC
from utils.utils_train_cls import train
from utils.usv_dataloader import BUSV_Dataset
from utils.utils_transform import VideoTransform


def parse_args():
    parser = argparse.ArgumentParser(description="Train CKC model on Breast_USV dataset")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/Breast_USV",
        help="Root path for the dataset (default: dataset/Breast_USV)"
    )
    parser.add_argument(
        "--excel_path",
        type=str,
        default="dataset/Breast_USV/labels.xlsx",
        help="Path to Excel file containing labels (default: dataset/Breast_USV/labels.xlsx)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of frames sampled per video"
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=224,
        help="Spatial size of output frames"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and validation"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help=(
            "Optional path to DINO detection checkpoint for backbone initialization. "
            "If not provided, SharedBackBone will automatically use timm official pretrained weights."
        )
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data augmentation
    train_transform = VideoTransform(
        output_size=args.output_size,
        flip_prob=0.5,
        crop_prob=0.5,
        scale=(0.8, 1.0),
        ratio=(3 / 4, 4 / 3)
    )
    valid_transform = VideoTransform(
        output_size=args.output_size,
        flip_prob=0.0,
        crop_prob=0.0,
        scale=(1.0, 1.0),
        ratio=(1.0, 1.0)
    )

    # Datasets
    train_dataset = BUSV_Dataset(
        os.path.join(args.dataset_path, "train.txt"),
        args.excel_path,
        base_dir="breast",
        transform=train_transform,
        num_samples=args.num_samples
    )
    valid_dataset = BUSV_Dataset(
        os.path.join(args.dataset_path, "valid.txt"),
        args.excel_path,
        base_dir="breast",
        transform=valid_transform,
        num_samples=args.num_samples
    )

    # Dataloaders
    data_train = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers
    )
    data_valid = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers
    )

    # Initialize CKC model
    model = CKC(
        backbone_name='swin_large',
        num_class=2,
        channel=1536,
        num_clinical=8,
        out_channel=512,
        num_frame=args.num_samples,
        weights_path=args.weights_path
    )

    model = model.to(device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Start training
    model = train(
        model,
        data_train,
        data_valid,
        optimizer,
        args.epochs,
        device=device
    )
