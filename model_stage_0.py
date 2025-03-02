import os
import torch
import torch.optim as optim

from utils.utils_model import CKC
from utils.utils_train_cls import train
from torch.utils.data import DataLoader
from utils.utils_transform import VideoTransform
from process.busv_dataloader import BUSV_Dataset

if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据增强 采样数
    num_samples = 16
    output_size = 224
    train_transform = VideoTransform(output_size=output_size, flip_prob=0.5, crop_prob=0.5,
                                     scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3))
    valid_transform = VideoTransform(output_size=output_size, flip_prob=0.0, crop_prob=0.0,
                                     scale=(1.0, 1.0), ratio=(1.0, 1.0))

    # 加载数据集
    dataset_path = r'E:\MyDataSet\Breast_USV'
    excel_path = r'E:\MyDataSet\Breast_USV\labels.xlsx'
    Data_train = BUSV_Dataset(os.path.join(dataset_path, "train.txt"), excel_path, base_dir="breast",
                              transform=train_transform, num_samples=num_samples)
    Data_valid = BUSV_Dataset(os.path.join(dataset_path, "valid.txt"), excel_path, base_dir="breast",
                              transform=valid_transform, num_samples=num_samples)

    # 划分数据集
    batch_size = 16
    data_train = DataLoader(Data_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    data_valid = DataLoader(Data_valid, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 设置参数
    epochs = 150
    weight_decay = 1e-5
    learning_rate = 2e-5

    # 建立模型
    model = CKC(backbone_name='swin_large', num_class=2, channel=1536, num_clinical=8, out_channel=512,
                num_frame=num_samples).cuda()

    # 建立优化器
    optimizer_class = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 模型训练
    model = train(model, data_train, data_valid, optimizer_class, epochs, device=device)
