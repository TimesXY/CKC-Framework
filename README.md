# CKC-Framework
This repository is an official implementation of the paper "Clinical Knowledge Constrained Multi-Task Learning Framework for Breast Cancer Diagnosis Using Ultrasound Videos."

## Breast-USV Dataset
We collected the largest known publicly available breast ultrasound video dataset, Breast-USV, for breast cancer classification, 
including 211 benign and 207 malignant cases. This dataset is available exclusively for non-commercial use in research or educational purposes. 
As long as it is used within these scopes, users are allowed to edit or process the images in the dataset. 

The full breast ultrasound dataset will be released after the paper is accepted for publication.

## Related Dataset
We also collected a multimodal ultrasound image dataset for classification in breast cancer, including 145 benign and 103 malignant cases. 
You can access it here: https://www.kaggle.com/datasets/timesxy/multimodal-breast-ultrasound-dataset-us3m

Pengfei Yan, Wushuang Gong, Minglei Li, Jiusi Zhang, Xiang Li, Yuchen Jiang, Hao Luo, and Hang Zhou. (2024). TDF-Net: Trusted Dynamic Feature Fusion Network for breast cancer diagnosis using incomplete multimodal ultrasound. Information Fusion, 102592.

## Code Usage

## Installation

### Requirements

* Linux, CUDA>=11.3, GCC>=7.5.0
  
* Python>=3.8

* PyTorch>=1.11.0, torchvision>=0.12.0 (following instructions [here](https://pytorch.org/))

* Other requirements
    ```bash
    pip install -r requirements.txt
    ```
  
### Dataset preparation

Please organize the dataset as follows:

```
code_root/
└── BR001/
      ├── BR001_frame_0000.jpg
      ├── BR001_frame_0001.jpg
      └── BR001_frame_0002.jpg
└── BR003/
      ├── BR003_frame_0000.jpg
      ├── BR003_frame_0001.jpg
      └── BR003_frame_0002.jpg
```

### Training

For example, the command for the training CKC-Framework is as follows:

```bash
python model_stage_0.py
```
The configs in model_stage_0.py or other files can be changed.

### Evaluation

After obtaining the trained CKC-Framework, then run the following command to evaluate it on the test set:

```bash
python model_stage_1.py
```

## Notes
The code of this repository is built on
https://github.com/TimesXY/CKC-Framework.
