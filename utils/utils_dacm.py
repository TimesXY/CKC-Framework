import torch
import torch.nn as nn


class DACM(nn.Module):
    """
    Diagnostic Attribute Constraints Module
    Args:
        num_clinical (int): 需要预测的诊断属性数量。
        feature_dim (int, optional): 提取特征的维度。默认值为 1024。
        output_dim (int, optional): 输出特征的维度。默认值为 256。
    """

    def __init__(self, num_clinical, feature_dim=1024, output_dim=256, num_class=2):
        super(DACM, self).__init__()
        self.feature_dim = feature_dim

        # 显式诊断特征
        self.explicit_path = nn.Sequential(
            nn.Conv2d(feature_dim, output_dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True))

        # 显式诊断特征约束
        self.explicit_class = nn.Sequential(
            nn.Conv2d(output_dim, num_class, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_class),
            nn.ReLU(inplace=True))

        # 隐式诊断特征
        self.implicit_path = nn.Sequential(
            nn.Conv2d(feature_dim, output_dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True))

        # 全局平均池化 - 压缩到指定维度
        self.global_avg_pool = nn.AdaptiveAvgPool2d((num_clinical, 1))

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * output_dim, output_dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, channels, height, width)。
        Returns:
            tuple:
                - fused_features (torch.Tensor): 融合后的特征，形状为 (batch_size, feature_dim, H, W)。
                - clinical_output (torch.Tensor): 预测的诊断属性，形状为 (batch_size, num_class, num_clinical)。
        """

        # 显式诊断特征约束
        explicit_features = self.explicit_path(x)
        explicit_output = self.explicit_class(explicit_features)

        # 全局平均池化并通过全连接层得到诊断属性输出
        clinical_output = self.global_avg_pool(explicit_output).squeeze(-1)

        # 隐式诊断特征
        implicit_features = self.implicit_path(x)

        # 在通道维度上合并显式和隐式诊断特征
        fused_embeddings = torch.cat([explicit_features, implicit_features], dim=1)

        # 融合特征信息
        fused_features = self.fusion(fused_embeddings)

        return fused_features, clinical_output
