import os
import timm
import torch
import warnings
import torch.nn.functional as F

from torch import nn
from mmengine.logging import MMLogger
from mmdet.models.backbones.swin import SwinTransformer

from .utils_dacm import DACM
from .utils_cpcm import CPCM
from .utils_mamba import ResidualMamba
from .utils_residual import ResidualBlock


class SharedBackBone(nn.Module):
    """
    Shared backbone with two modes:
    1. If `weights_path` is provided and the file exists:
       use OpenMMLab Swin-L and load the DINO detection checkpoint.
    2. Otherwise:
       use timm official pretrained Swin backbone.
    Notes:
        - The OpenMMLab path is only supported for `swin_large`, because the
          provided DINO checkpoint is matched to the Swin-L configuration.
        - The timm path supports both `swin_base` and `swin_large`.
    """

    def __init__(self, backbone_name='swin_large', weights_path=None, with_cp=False):
        super().__init__()

        self.backbone_name = backbone_name
        self.weights_path = weights_path
        self.with_cp = with_cp
        self.backend = None

        use_openmmlab = weights_path is not None and os.path.exists(weights_path)

        # Mode 1: OpenMMLab Swin-L + DINO detection checkpoint
        if use_openmmlab:
            if backbone_name != 'swin_large':
                raise ValueError(
                    "OpenMMLab/DINO checkpoint mode currently supports only "
                    "'swin_large', because the provided detection checkpoint "
                    "is matched to the Swin-L configuration."
                )

            self.backend = 'openmmlab'
            init_cfg = dict(type='Pretrained', checkpoint=weights_path)

            self.backbone = SwinTransformer(
                pretrain_img_size=384,
                embed_dims=192,
                depths=(2, 2, 18, 2),
                num_heads=(6, 12, 24, 48),
                window_size=12,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.2,
                patch_norm=True,
                out_indices=(3,),
                with_cp=with_cp,
                convert_weights=True,
                init_cfg=init_cfg
            )

            MMLogger.get_instance('ckc_openmmlab')
            self.backbone.init_weights()

            self.feature_dim = 1536
            print(f"[SharedBackBone] Using OpenMMLab Swin-L with checkpoint: {weights_path}")

        # Mode 2: timm official pretrained Swin
        else:
            self.backend = 'timm'

            if weights_path is not None and not os.path.exists(weights_path):
                warnings.warn(
                    f"[SharedBackBone] Checkpoint not found: {weights_path}. "
                    f"Falling back to timm official pretrained weights."
                )

            if backbone_name == 'swin_base':
                model_name = 'swin_base_patch4_window7_224'
            elif backbone_name == 'swin_large':
                model_name = 'swin_large_patch4_window7_224'
            else:
                raise ValueError(f"Unsupported backbone: {backbone_name}")

            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0
            )
            self.feature_dim = self.backbone.num_features
            print(f"[SharedBackBone] Using timm official pretrained backbone: {model_name}")

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W)
        Returns:
            torch.Tensor: Feature map with shape (B, C, H', W')
        """
        if self.backend == 'openmmlab':
            outs = self.backbone(x)
            x = outs[-1]
            return x

        if self.backend == 'timm':
            x = self.backbone.forward_features(x)

            # timm Swin may return either:
            # - (B, H, W, C)
            # - (B, L, C)
            # Convert both to (B, C, H, W)
            if x.dim() == 4:
                x = x.permute(0, 3, 1, 2).contiguous()
            elif x.dim() == 3:
                b, l, c = x.shape
                h = w = int(l ** 0.5)
                if h * w != l:
                    raise RuntimeError(
                        f"Cannot reshape timm output of shape {(b, l, c)} "
                        f"into a 2D feature map."
                    )
                x = x.transpose(1, 2).contiguous().reshape(b, c, h, w)
            else:
                raise RuntimeError(f"Unexpected timm backbone output dimension: {x.dim()}")

            return x

        raise RuntimeError(f"Unknown backend: {self.backend}")


class CKC(nn.Module):
    """
    CKC model integrating DACM, CPCM, Residual blocks and Mamba
    for breast ultrasound video classification.

    Args:
        backbone_name (str): Backbone name. Supported:
            - 'swin_base'  (timm mode only)
            - 'swin_large' (timm mode or OpenMMLab+DINO mode)
        num_class (int): Number of classes.
        channel (int): Input backbone feature dimension.
        num_clinical (int): Number of clinical attributes.
        out_channel (int): Intermediate feature dimension.
        num_frame (int): Number of sampled video frames.
        weights_path (str or None): Optional DINO detection checkpoint path.
            If provided and exists, OpenMMLab Swin-L will be used.
            Otherwise timm official pretrained weights will be used.
    """

    def __init__(
            self,
            backbone_name='swin_large',
            num_class=2,
            channel=1536,
            num_clinical=8,
            out_channel=256,
            num_frame=4,
            weights_path=None
    ):
        super(CKC, self).__init__()

        # Shared backbone
        self.backbone = SharedBackBone(
            backbone_name=backbone_name,
            weights_path=weights_path
        )

        # Diagnostic Attribute Constraint Module
        self.frame_dacm = DACM(
            num_clinical=num_clinical,
            feature_dim=channel,
            output_dim=out_channel
        )

        # Residual blocks for semantic transformation
        self.residual_block_1 = ResidualBlock(
            out_channel, out_channel // 4, stride=1, use_gap=True
        )
        self.residual_block_2 = ResidualBlock(
            out_channel, out_channel // 4, stride=1, use_gap=True
        )

        # Clinical Prior Constraint Module
        self.video_cpcm = CPCM(
            num_frames=num_frame,
            feature_dim=out_channel,
            output_dim=out_channel
        )

        # Temporal modeling module
        self.key_mamba = ResidualMamba(
            d_model=out_channel,
            expand=2,
            len_size=num_frame,
            dt_rank='auto',
            d_state=16,
            d_conv=4,
            conv_bias=True,
            bias=False
        )

        # Pooling and classification
        self.gap_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classification = nn.Linear(out_channel, num_class)

    def forward(self, video, key_index):
        """
        Args:
            video (torch.Tensor): Input video tensor of shape
                (batch_size, channels, num_frames, height, width)
            key_index (torch.Tensor): Key frame indices of shape (batch_size,)
        Returns:
            tuple:
                - out_class (torch.Tensor): (batch_size, num_class)
                - time_attention (torch.Tensor): (batch_size, num_frames)
                - clinical_output (torch.Tensor): (batch_size, num_clinical)
                - frame_feature_out (torch.Tensor)
                - features_out (torch.Tensor)
        """
        batch_size, channels, num_frames, height, width = video.size()

        # Merge frame dimension into batch dimension
        video = video.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        video = video.view(batch_size * num_frames, channels, height, width)

        # Backbone feature extraction
        features = self.backbone(video)  # (B*T, C, H', W')

        # Frame-level DACM on all frames
        features_f_video, _ = self.frame_dacm(features)

        # Semantic transformation
        features_out = self.residual_block_1(features_f_video)
        features_out = F.normalize(features_out, p=2, dim=1)
        features_out = features_out.view(batch_size, num_frames, -1)

        # Video-level prior modeling
        video_feature, time_attention = self.video_cpcm(features_f_video)

        # Restore frame dimension for key-frame indexing
        features = features.view(
            batch_size,
            num_frames,
            features.size(1),
            features.size(2),
            features.size(3)
        )
        features = features.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H', W')

        # Key-frame feature selection
        frame_key = features[
                    torch.arange(features.size(0), device=features.device),
                    :,
                    key_index,
                    :,
                    :
                    ]

        # Key-frame DACM
        frame_feature, clinical_output = self.frame_dacm(frame_key)

        # Semantic transformation on key-frame
        frame_feature_out = self.residual_block_2(frame_feature)
        frame_feature_out = F.normalize(frame_feature_out, p=2, dim=1)

        # Global average pooling
        video_feature_pool = self.gap_pool(video_feature).view(batch_size, -1, num_frames)
        frame_feature_pool = self.gap_pool(frame_feature).view(batch_size, -1, 1)

        video_feature_seq = video_feature_pool.permute(0, 2, 1)  # (B, T, C)
        frame_feature_seq = frame_feature_pool.permute(0, 2, 1)  # (B, 1, C)

        # Temporal modeling
        time_feature = self.key_mamba(video_feature_seq, frame_feature_seq)

        # Aggregate temporal features
        last_feature = time_feature[:, -1, :]
        mean_feature = time_feature.mean(dim=1)
        time_feature = last_feature + mean_feature

        # Classification
        out_class = self.classification(time_feature)

        return out_class, time_attention, clinical_output, frame_feature_out, features_out
