import os
import torch
import argparse

from torch import nn
from torch.utils.data import DataLoader

from utils.utils_model import CKC
from utils.utils_sr import validate_model
from utils.usv_dataloader import BUSV_Dataset
from utils.utils_transform import VideoTransform


def parse_args():
    parser = argparse.ArgumentParser(description="Validate CKC model on Breast_USV dataset")

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
        "--txt_file",
        type=str,
        default="test.txt",
        help="Text file listing test video folders (default: test.txt)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="save_weights/best_model.pth",
        help="Path to the trained CKC model weights"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for validation"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of frames per video"
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=224,
        help="Spatial size of input frames"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )

    return parser.parse_args()


def load_model_weights(model, model_path, device):
    """
    Load model weights safely for both single-GPU and multi-GPU checkpoints.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # If checkpoint is wrapped in a dict, try to extract the real state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    return model


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validation transform
    valid_transform = VideoTransform(
        output_size=args.output_size,
        flip_prob=0.0,
        crop_prob=0.0,
        scale=(1.0, 1.0),
        ratio=(1.0, 1.0)
    )

    # Validation dataset
    valid_dataset = BUSV_Dataset(
        os.path.join(args.dataset_path, args.txt_file),
        args.excel_path,
        base_dir="breast",
        transform=valid_transform,
        num_samples=args.num_samples
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
        weights_path=None
    )

    # Load full trained CKC weights first
    model = load_model_weights(model, args.model_path, device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)
    model.eval()

    print("Model loaded successfully and set to evaluation mode.")

    # Run validation
    avg_loss = validate_model(model, data_valid, device=device)
    print(f"Average validation loss: {avg_loss:.4f}")
