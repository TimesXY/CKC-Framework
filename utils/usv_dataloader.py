import os
import math
import torch
import logging
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from typing import List, Tuple, Callable, Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _uniform_sample_with_keyframe(start: int, end: int, key_frame: int,
                                  num_samples: int, oversample: bool = False) -> List[int]:
    """
    Uniformly sample `num_samples` frames between [start, end] and ensure key_frame is included.
    If oversample=True and enough frames are available, sample num_samples + 2 frames then remove the first and last.
    If not enough frames, duplicate the key frame and its neighbors to meet the required num_samples while keeping temporal order.

    Args:
        start (int): Start frame index (inclusive, predicted from first stage)
        end (int): End frame index (inclusive, predicted from first stage)
        key_frame (int): Index of the key frame
        num_samples (int): Total number of frames to sample
        oversample (bool): Whether to perform oversampling
    Returns:
        List[int]: List of sampled frame indices
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if start > end:
        raise ValueError("start must be less than or equal to end")
    if not (start <= key_frame <= end):
        raise ValueError("key_frame must be between start and end")

    total_available = end - start + 1
    frames = list(range(start, end + 1))

    if total_available >= num_samples:
        target_num = num_samples + 2 if oversample else num_samples
        target_num = min(target_num, total_available)

        if target_num == 1:
            sampled = [key_frame]
        else:
            step = (end - start) / (target_num - 1)
            sampled = [min(int(math.floor(start + i * step + 0.5)), end) for i in range(target_num)]

            # Ensure key frame is included
            if key_frame not in sampled:
                distances = [abs(f - key_frame) for f in sampled]
                min_distance = min(distances)
                candidates = [f for f, d in zip(sampled, distances) if d == min_distance]
                frame_to_replace = max(candidates)
                replace_index = sampled.index(frame_to_replace)
                sampled[replace_index] = key_frame

            sampled = sorted(sampled)

        if oversample and target_num > num_samples:
            # Remove first and last frame
            sampled = sampled[1:-1]

        return sampled
    else:
        # Not enough frames, duplicate key frame and neighbors to meet num_samples
        frame_counts = {frame: 1 for frame in frames}  # each frame appears at least once
        additional = num_samples - total_available

        if additional > 0:
            # Compute distance to key frame
            frame_distances = {frame: abs(frame - key_frame) for frame in frames}
            # Sort frames by distance to key frame, then by frame index
            sorted_frames = sorted(frames, key=lambda x: (frame_distances[x], x))
            for i in range(additional):
                frame_to_duplicate = sorted_frames[i % len(sorted_frames)]
                frame_counts[frame_to_duplicate] += 1

        # Generate final sampled frame indices in temporal order
        frame_indices = []
        for frame in sorted(frames):
            frame_indices.extend([frame] * frame_counts[frame])

        return frame_indices[:num_samples]


class BUSV_Dataset(Dataset):
    """
    Each sample contains a sequence of images (video frames), a label, key frame index,
    diagnostic attributes, and Gaussian distances.

    Args:
        txt_path (str): Path to a text file containing image folder names and labels
        excel_path (str): Path to an Excel file containing additional information
        base_dir (str): Base directory where image folders are located
        transform (callable, optional): Optional image preprocessing function, applied per image
        num_samples (int): Number of frames to sample (must include key frame)
        sigma (float): Standard deviation for Gaussian distance calculation
    """

    def __init__(self, txt_path: str, excel_path: str, base_dir: str, transform: Optional[Callable] = None,
                 num_samples: int = 32, sigma: float = 0.25):

        self.base_dir = base_dir
        self.transform = transform
        self.num_samples = num_samples
        self.sigma = sigma
        self.image_paths_labels: List[Tuple[str, int]] = []  # List of (image folder path, label)
        self.additional_info: Dict[str, Dict[str, Any]] = {}  # Dict storing extra info for each folder

        # Load image paths and labels
        self._load_txt(txt_path)

        # Load additional info and cache image files
        self._load_excel(excel_path)

    def _load_txt(self, txt_path: str):
        """Load subfolder names and labels from text file and generate full image folder paths"""
        txt_path = Path(txt_path).resolve()
        if not txt_path.is_file():
            raise FileNotFoundError(f"Text file not found: {txt_path}")

        base_dir = txt_path.parent / self.base_dir
        if not base_dir.is_dir():
            raise FileNotFoundError(f"Base directory not found: {base_dir}")

        with txt_path.open('r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    logger.warning(f"Invalid line in text file {line_num}: {line}")
                    continue
                subfolder_name, label_str = parts[0], parts[1]
                try:
                    label = int(label_str)
                except ValueError:
                    logger.warning(f"Cannot convert label to int at line {line_num}: {label_str}")
                    continue

                images_path = base_dir / subfolder_name
                if not images_path.is_dir():
                    logger.warning(f"Image folder not found: {images_path} (line {line_num})")
                    continue
                self.image_paths_labels.append((str(images_path.resolve()), label))

    def _load_excel(self, excel_path: str):
        """Load additional info from Excel and cache image files"""
        excel_path = Path(excel_path).resolve()
        if not excel_path.is_file():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        try:
            excel_data = pd.read_excel(excel_path)
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}")

        expected_columns = 13
        if excel_data.shape[1] < expected_columns:
            raise ValueError(f"Excel must have at least {expected_columns} columns, got {excel_data.shape[1]}")

        for idx, row in excel_data.iterrows():
            existing_folder_names = set(Path(path).name for path, _ in self.image_paths_labels)
            folder_name = str(row.iloc[0]).strip()
            if folder_name not in existing_folder_names:
                continue

            try:
                folder_name = str(row.iloc[0]).strip()
                key_frame_index = int(row.iloc[2])
                diagnostic_attributes = row.iloc[3:11].to_numpy(dtype=np.float32)
                start_frame = int(row.iloc[11])
                end_frame = int(row.iloc[12])
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid data in Excel at row {idx + 1}: {e}")
                continue

            matching_paths = [path for path, _ in self.image_paths_labels if Path(path).name == folder_name]
            if not matching_paths:
                logger.warning(f"Folder '{folder_name}' in Excel not found in txt file")
                continue

            images_path = Path(matching_paths[0])
            image_files = natsorted([
                str(images_path / f) for f in os.listdir(images_path)
                if (images_path / f).is_file() and Path(f).suffix.lower() == '.jpg'])

            if not image_files:
                logger.warning(f"No image files found in folder: {images_path} (Excel folder_name '{folder_name}')")
                continue

            self.additional_info[folder_name] = {
                'key_frame_index': key_frame_index,
                'diagnostic_attributes': diagnostic_attributes,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'image_files': image_files
            }

    def __len__(self) -> int:
        return len(self.image_paths_labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int, torch.Tensor, torch.Tensor, str]:
        """
        Args:
            index (int): Index of the sample
        Returns:
            tuple: (imgs_batch, label, new_key_frame_index, diagnostic_attributes, gaussian_distances, images_path)
                - imgs_batch (torch.Tensor): shape (channels, frames, height, width)
                - label (int): sample label
                - new_key_frame_index (int): key frame index after sampling
                - diagnostic_attributes (torch.Tensor): diagnostic attributes tensor
                - gaussian_distances (torch.Tensor): Gaussian distances to key frame (shape: num_samples)
                - images_path (str): path to the image folder
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range (0, {len(self) - 1})")
        images_path, label = self.image_paths_labels[index]
        folder_name = Path(images_path).name

        if folder_name not in self.additional_info:
            raise ValueError(f"Missing additional info for folder {folder_name}")
        info = self.additional_info[folder_name]
        key_frame_index = info['key_frame_index']
        diagnostic_attributes = info['diagnostic_attributes']
        start_frame = info['start_frame']
        end_frame = info['end_frame']
        image_files = info.get('image_files', [])

        if not image_files:
            raise ValueError(f"No images found in folder: {images_path}")

        total_frames = len(image_files)
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, total_frames - 1)

        if start_frame > end_frame:
            raise ValueError(f"Start frame {start_frame} > end frame {end_frame} in folder {folder_name}")

        can_oversample = (end_frame - start_frame + 1) >= (self.num_samples + 2)

        frame_indices_relative = _uniform_sample_with_keyframe(
            start=start_frame,
            end=end_frame,
            key_frame=key_frame_index,
            num_samples=self.num_samples,
            oversample=can_oversample)

        selected_image_files = [image_files[i] for i in frame_indices_relative]

        images = []
        for img_path in selected_image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                raise RuntimeError(f"Error loading image {img_path}: {e}")

        if self.transform:
            transformed_images = self.transform(images)
        else:
            transformed_images = [F.to_tensor(img) for img in images]

        frame_sizes = [img.shape for img in transformed_images]
        if not all(size == frame_sizes[0] for size in frame_sizes):
            raise RuntimeError(
                f"All frames must have the same size, but found different sizes in video at index {index}: {frame_sizes}")

        imgs_batch = torch.stack(transformed_images)  # (frames, channels, height, width)
        imgs_batch = imgs_batch.permute(1, 0, 2, 3)  # (channels, frames, height, width)

        diagnostic_attributes_tensor = torch.from_numpy(diagnostic_attributes)

        frame_indices_tensor = torch.tensor(frame_indices_relative, dtype=torch.float32)
        frame_indices_normalized = frame_indices_tensor / total_frames
        key_frame_index_normalized = torch.tensor(key_frame_index, dtype=torch.float32) / total_frames
        squared_diff = (frame_indices_normalized - key_frame_index_normalized) ** 2
        gaussian_distances = torch.exp(-squared_diff / (2 * self.sigma ** 2))

        try:
            new_key_frame_index = frame_indices_relative.index(key_frame_index)
        except ValueError:
            closest_idx = min(range(len(frame_indices_relative)),
                              key=lambda i: abs(frame_indices_relative[i] - key_frame_index))
            new_key_frame_index = closest_idx

        return imgs_batch, label, new_key_frame_index, diagnostic_attributes_tensor, gaussian_distances, images_path
