import os
import torch

from utils.utils_model import CKC
from torch.utils.data import DataLoader
from utils.utils_sr import validate_model
from utils.utils_transform import VideoTransform
from process.busv_dataloader import BUSV_Dataset

if __name__ == '__main__':

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据增强 采样数
    batch_size = 16
    num_samples = 16
    output_size = 224
    valid_transform = VideoTransform(output_size=output_size, flip_prob=0.0, crop_prob=0.0,
                                     scale=(1.0, 1.0), ratio=(1.0, 1.0))

    # 加载测试集
    dataset_path = r'E:\MyDataSet\Breast_USV'
    excel_path = r'E:\MyDataSet\Breast_USV\labels.xlsx'
    Data_valid = BUSV_Dataset(os.path.join(dataset_path, "test.txt"), excel_path, base_dir="breast",
                              transform=valid_transform, num_samples=num_samples)
    data_valid = DataLoader(Data_valid, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 建立模型
    model = CKC(backbone_name='swin_large', num_class=2, channel=1536, num_clinical=8, out_channel=512,
                num_frame=num_samples).cuda()

    # 加载训练好的模型权重
    model_path = 'save_weights/best_model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式
    print("模型加载成功并设置为评估模式。")

    # 验证模型
    avg_loss = validate_model(model, data_valid, device=device)
