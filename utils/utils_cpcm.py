import torch.nn as nn


class CPCM(nn.Module):
    """
    Clinical Prior Constraints Module
    参数：
        num_frames (int): 输入视频的帧数。
        feature_dim (int, 可选): 提取特征的维度，默认为 1024。
        output_dim (int, 可选): 经过 1x1 卷积后的特征维度，默认为 256。
    """

    def __init__(self, num_frames=32, feature_dim=1024, output_dim=256):
        super(CPCM, self).__init__()

        self.num_frames = num_frames

        # 1x1 卷积进行通道降维
        self.dim_reduce = nn.Sequential(
            nn.Conv2d(feature_dim, output_dim, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True))

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 用于生成时间注意力权重的全连接层
        self.attention_fc = nn.Sequential(
            nn.Linear(num_frames, num_frames // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_frames // 2, num_frames, bias=False),
            nn.Softmax(dim=1))

    def forward(self, x):
        """
        参数：
            x (torch.Tensor): 输入张量，形状为 (batch_size * num_frames, channels, height, width)。
        返回：
            元组：
                - weighted_features (torch.Tensor): 应用时间注意力后的加权特征，形状为 (batch_size, output_dim, num_frames, H', W')。
                - time_attention (torch.Tensor): 每一帧的时间注意力权重，形状为 (batch_size, num_frames)。
        """
        batch_size = x.size(0) // self.num_frames

        # 1x1 卷积进行维度降维
        x = self.dim_reduce(x)  # 形状：(batch_size * num_frames, output_dim, H', W')

        # 重塑张量形状并调整维度顺序
        x = x.view(batch_size, self.num_frames, -1, x.size(-2), x.size(-1)).permute(0, 2, 1, 3, 4)

        # 全局平均池化，得到 (batch_size, output_dim, num_frames, 1, 1)
        pooled_features = self.global_avg_pool(x)

        # 压缩空间维度，得到 (batch_size, output_dim, num_frames)
        pooled_features = pooled_features.view(batch_size, -1, self.num_frames)

        # 在特征维度上取平均，得到 (batch_size, num_frames)
        pooled_features_mean = pooled_features.mean(dim=1)

        # 生成时间注意力权重，形状为 (batch_size, num_frames)
        time_attention = self.attention_fc(pooled_features_mean)

        # 扩展注意力权重以匹配特征维度，形状为 (batch_size, 1, num_frames, 1, 1)
        time_attention_x = time_attention.view(batch_size, 1, self.num_frames, 1, 1)

        # 应用时间注意力权重，形状为 (batch_size, output_dim, num_frames, H', W')
        weighted_features = x * time_attention_x

        return weighted_features, time_attention
