import timm
import torch

from torch import nn
from .utils_dacm import DACM
from .utils_cpcm import CPCM
import torch.nn.functional as F
from .utils_mamba import ResidualMamba
from .utils_residual import ResidualBlock


class SharedBackBone(nn.Module):
    """
    共享的骨干网络，支持 Swin Transformer
    参数：
        backbone_name (str, optional): 骨干网络的名称，支持以下选项：
            - 'swin_base'
            - 'swin_large'
          默认值为 'swin_base'。
        pretrain (bool, optional): 是否使用预训练权重。默认值为 True。
    """

    def __init__(self, backbone_name='swin_base', pretrain=True):
        super(SharedBackBone, self).__init__()

        self.backbone_name = backbone_name  # 保存骨干网络名称

        # 根据骨干网络名称创建对应的模型
        if backbone_name == 'swin_base':
            self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrain)
            self.feature_dim = self.backbone.num_features  # Swin Transformer 的输出维度
            self.backbone.head = nn.Identity()  # 移除分类头

        elif backbone_name == 'swin_large':
            self.backbone = timm.create_model('swin_large_patch4_window7_224', pretrained=pretrain)
            self.feature_dim = self.backbone.num_features
            self.backbone.head = nn.Identity()

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x):
        """
        参数：
            x (torch.Tensor): 输入张量，形状为 (batch_size, channels, height, width)。
        返回：
            torch.Tensor: 提取的特征，形状为 (batch_size, feature_dim, H', W')。
        """

        if self.backbone_name in ['swin_base', 'swin_large']:

            # Swin Transformer 提取特征图
            x = self.backbone.patch_embed(x)
            for layer in self.backbone.layers:
                x = layer(x)
            x = self.backbone.norm(x)  # [B, H, W, C]

            # 将序列重塑回二维特征图
            x = x.permute((0, 3, 1, 2))  # [B, C, H, W]

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        return x


class CKC(nn.Module):
    """
    CKC 模型，将 DACM 和 CPCM 模块结合 Mamba 网络进行分类。
    参数：
        num_class (int, optional): 分类任务的类别数。默认值为 2。
        channel (int, optional): 输入特征维度。默认值为 1024。
        num_clinical (int, optional): 临床属性数量。默认值为 8。
        out_channel (int, optional): 中间输出通道数。默认值为 256。
        num_frame (int, optional): 输入视频的帧数。默认值为 32。
    """

    def __init__(self, backbone_name='swin_base', num_class=2, channel=1024, num_clinical=8, out_channel=256,
                 num_frame=4):
        super(CKC, self).__init__()

        # 建立共享骨干网络
        self.backbone = SharedBackBone(backbone_name=backbone_name, pretrain=True)

        # 诊断属性约束模块
        self.frame_dacm = DACM(num_clinical=num_clinical, feature_dim=channel, output_dim=out_channel)

        # 语义信息转换的残差模块
        self.residual_block_1 = ResidualBlock(out_channel, out_channel // 4, stride=1, use_gap=True)
        self.residual_block_2 = ResidualBlock(out_channel, out_channel // 4, stride=1, use_gap=True)

        # 临床先验约束模块
        self.video_cpcm = CPCM(num_frames=num_frame, feature_dim=out_channel, output_dim=out_channel)

        # Mamba 网络结构
        self.key_mamba = ResidualMamba(d_model=out_channel, expand=2, len_size=num_frame, dt_rank='auto', d_state=16,
                                       d_conv=4, conv_bias=True, bias=False)

        # 全局平均池化层
        self.gap_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类层
        self.classification = nn.Linear(out_channel, num_class)

    def forward(self, video, key_index):
        """
        参数：
            video (torch.Tensor): 输入视频张量，形状为 (batch_size, channels, num_frames, height, width)。
            key_index (torch.Tensor): 关键帧索引，形状为 (batch_size,)。
        返回：
            tuple:
                - out_class (torch.Tensor): 分类输出，形状为 (batch_size, num_class)。
                - time_attention (torch.Tensor): 时间注意力权重，形状为 (batch_size, num_frames)。
                - clinical_output (torch.Tensor): 临床属性输出，形状为 (batch_size, num_clinical)。
        """

        batch_size, channels, num_frames, height, width = video.size()

        # 重塑输入以并行处理所有帧
        video = video.permute(0, 2, 1, 3, 4).contiguous()  # (batch_size, num_frames, channels, height, width)
        video = video.view(batch_size * num_frames, channels, height, width)

        # 共享特征提取器提取特征
        features = self.backbone(video)  # (batch_size * num_frames, feature_dim, H', W')

        # 通过 DACM 模块处理视频帧
        features_f_video, _ = self.frame_dacm(features)
        # frame_feature: (batch_size, out_channel, H', W'), clinical_output: (batch_size, num_clinical)

        # 通过语义转换模块, 转换为语义特征
        features_out = self.residual_block_1(features_f_video)

        # 恢复帧维度 (batch_size, output_dim, num_frames)
        features_out = F.normalize(features_out, p=2, dim=1)
        features_out = features_out.view(batch_size, num_frames, -1)

        # 视频诊断的预测结果
        video_feature, time_attention = self.video_cpcm(features_f_video)
        # video_feature: (batch_size, out_channel, num_frames, H', W'), time_attention: (batch_size, num_frames)

        # 使用高级索引获取关键帧 恢复帧维度 (batch_size, output_dim, num_frames, H', W')
        features = features.view(batch_size, num_frames, features.size(1), features.size(2), features.size(3))
        features = features.permute(0, 2, 1, 3, 4).contiguous()

        # 获取关键帧特征
        frame_key = features[torch.arange(features.size(0), device=features.device), :, key_index, :, :]

        # 通过 DACM 模块处理关键帧
        frame_feature, clinical_output = self.frame_dacm(frame_key)
        # frame_feature: (batch_size, out_channel, H', W'), clinical_output: (batch_size, num_clinical)

        # 通过语义转换模块, 转换为语义特征
        frame_feature_out = self.residual_block_2(frame_feature)
        frame_feature_out = F.normalize(frame_feature_out, p=2, dim=1)

        # 利用全局平均池化层消除空间维度，保留通道维度
        video_feature_pool = self.gap_pool(video_feature).view(batch_size, -1, num_frames)
        frame_feature_pool = self.gap_pool(frame_feature).view(batch_size, -1, 1)

        video_feature_seq = video_feature_pool.permute(0, 2, 1)
        frame_feature_seq = frame_feature_pool.permute(0, 2, 1)

        # 通过 Mamba 网络提取时间信息
        time_feature = self.key_mamba(video_feature_seq, frame_feature_seq)  # (batch_size, 1, out_channel)

        # 在时间维度上聚合加权特征
        last_feature = time_feature[:, -1, :]  # 选择最后一个时间步 (B, out_channel)
        mean_feature = time_feature.mean(dim=1)  # 平均 (B, out_channel)
        time_feature = last_feature + mean_feature

        # 通过分类层
        out_class = self.classification(time_feature)  # (batch_size, num_class)

        return out_class, time_attention, clinical_output, frame_feature_out, features_out
