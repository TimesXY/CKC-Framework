# CKC-Framework

Official implementation of the paper:

**Clinical Knowledge Constrained Multi-Task Learning Framework for Breast Cancer Diagnosis Using Ultrasound Videos**

---

## Overview

This repository provides the implementation of a framework for breast ultrasound video analysis, including:

1. **Detection**: tumor localization and redundant frame filtering.
2. **Classification**: diagnostic attribute modeling and benign/malignant classification on ultrasound videos.

---

## Data Access

The de-identified **Breast-USV** dataset is publicly available for **non-commercial academic research only**, subject to the associated data usage terms.  
Detailed dataset access information is provided in the **Data Availability** section of the paper.

---

## Dataset Organization

The classification scripts expect the dataset to be organized as follows:

```text
dataset/Breast_USV/
├── train.txt
├── test.txt
├── labels.xlsx
└── breast/
    ├── BR001/
    │   ├── BR001_frame_0000.jpg
    │   ├── BR001_frame_0001.jpg
    │   └── ...
    ├── BR003/
    │   ├── BR003_frame_0000.jpg
    │   ├── BR003_frame_0001.jpg
    │   └── ...
    └── ...
````

Each line in `train.txt` and `test.txt` should contain:

```text
folder_name label
```

Example:

```text
BR001 0
BR003 1
```

Here, `0` denotes benign and `1` denotes malignant.

---

## Installation

### Requirements

* Linux
* Python >= 3.8
* CUDA >= 11.3
* PyTorch >= 1.11.0
* torchvision >= 0.12.0

Install all dependencies with:

```bash
pip install -r requirements.txt
```

If OpenMMLab-related packages fail to install in your environment, please follow the official installation instructions for **MMDetection**, **MMCV**, and **MMEngine**.

---

## Detection Training

Train the DINO-based detector:

```bash
python model_det_0.py \
    --config_file configs/dino/dino-5scale_swin-l_8xb2-12e.py \
    --work_dir work_dir \
    --train_ann dataset/annotations/instances_train.json \
    --val_ann dataset/annotations/instances_val.json \
    --train_img_dir dataset/train \
    --val_img_dir dataset/val \
    --batch_size 4 \
    --num_workers 4
```

## Detection Testing

```bash
python model_det_1.py \
    --config_file work_dir/dino-5scale_swin-l_8xb2-12e.py \
    --checkpoint_file work_dir/best_coco_bbox_mAP_50.pth \
    --test_image_dir dataset/test \
    --ann_file dataset/annotations/instances_test.json \
    --num_classes 1 \
    --batch_size 1 \
    --num_workers 1
```

### Detection Notes

* The training script automatically downloads the OpenMMLab pretrained DINO checkpoint.
* Detection checkpoints and logs are saved in `work_dir/`.
* The best detection checkpoint is typically:
  `work_dir/best_coco_bbox_mAP_50.pth`

---

## Classification Training

Train the CKC classification framework:

```bash
python model_cls_0.py \
    --dataset_path dataset/Breast_USV \
    --excel_path dataset/Breast_USV/labels.xlsx \
    --num_samples 16 \
    --output_size 224 \
    --batch_size 16 \
    --epochs 150 \
    --learning_rate 2e-5 \
    --weight_decay 1e-5 \
    --num_workers 4
```

If you want to initialize the classification backbone from the detection checkpoint, add:

```bash
--weights_path work_dir/best_coco_bbox_mAP_50.pth
```

Example:

```bash
python model_cls_0.py \
    --dataset_path dataset/Breast_USV \
    --excel_path dataset/Breast_USV/labels.xlsx \
    --num_samples 16 \
    --output_size 224 \
    --batch_size 16 \
    --epochs 150 \
    --learning_rate 2e-5 \
    --weight_decay 1e-5 \
    --num_workers 4 \
    --weights_path work_dir/best_coco_bbox_mAP_50.pth
```

## Classification Testing

```bash
python model_cls_1.py \
    --dataset_path dataset/Breast_USV \
    --excel_path dataset/Breast_USV/labels.xlsx \
    --txt_file test.txt \
    --model_path save_weights/best_model.pth \
    --batch_size 16 \
    --num_samples 16 \
    --output_size 224 \
    --num_workers 4
```

---

## Classification Arguments

### `model_cls_0.py`

| Argument   | Default                          | Description                                               |
| ---------- | -------------------------------- | --------------------------------------------------------- |
| `dataset_path` | `dataset/Breast_USV`             | Root path of the dataset                                  |
| `excel_path` | `dataset/Breast_USV/labels.xlsx` | Excel file containing labels and metadata                 |
| `num_samples` | `16`                             | Number of frames sampled per video                        |
| `output_size` | `224`                            | Spatial size of the input frames                          |
| `batch_size` | `16`                             | Batch size for training and validation                    |
| `epochs`   | `150`                            | Number of training epochs                                 |
| `learning_rate` | `2e-5`                           | Learning rate                                             |
| `weight_decay` | `1e-5`                           | Weight decay                                              |
| `weights_path` | `None`                           | Optional detection checkpoint for backbone initialization |
| `num_workers` | `4`                              | Number of dataloader workers                              |

### `model_cls_1.py`

| Argument   | Default                          | Description                               |
| ---------- | -------------------------------- | ----------------------------------------- |
| `dataset_path` | `dataset/Breast_USV`             | Root path of the dataset                  |
| `excel_path` | `dataset/Breast_USV/labels.xlsx` | Excel file containing labels and metadata |
| `txt_file` | `test.txt`                       | Text file listing test video folders      |
| `model_path` | `save_weights/best_model.pth`    | Path to the trained CKC checkpoint        |
| `batch_size` | `16`                             | Batch size for validation/testing         |
| `num_samples` | `16`                             | Number of frames per video                |
| `output_size` | `224`                            | Spatial size of the input frames          |
| `num_workers` | `4`                              | Number of dataloader workers              |

---

## Backbone Initialization Strategy

The CKC framework supports two initialization modes:

1. **Detection-based initialization**
   If `--weights_path` is provided, the shared backbone is initialized from the DINO detection checkpoint.

2. **timm pretrained initialization**
   If `--weights_path` is not provided, the backbone falls back to the official timm pretrained Swin weights.

---

## Citation

If you find this repository useful, please cite the corresponding paper.

---

## Acknowledgements

This repository builds upon the following open-source projects:

* [MMDetection](https://github.com/open-mmlab/mmdetection)
* [timm](https://github.com/huggingface/pytorch-image-models)