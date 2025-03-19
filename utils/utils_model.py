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
    Shared backbone network, supporting Swin Transformer.
    Parameters:
        backbone_name (str, optional): Name of the backbone network, supporting the following options:
            - 'swin_base'
            - 'swin_large'
          Default is 'swin_base'.
        pretrain (bool, optional): Whether to use pretrained weights. Default is True.
        weights_path: The path of the pretrained model from the object detection model's backbone.
    """

    def __init__(self, backbone_name='swin_base', pretrain=True, weights_path="pretrain_model.pth"):
        super(SharedBackBone, self).__init__()

        self.backbone_name = backbone_name  # Save the backbone network name

        # Create the corresponding model based on the backbone network name
        if backbone_name == 'swin_base':
            self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrain)

            # If a trained weight path is provided, load the trained weights
            if weights_path is not None:
                state_dict = torch.load(weights_path, map_location="cpu")
                self.backbone.load_state_dict(state_dict)

            self.feature_dim = self.backbone.num_features  # Output dimension of Swin Transformer
            self.backbone.head = nn.Identity()  # Remove classification head

        elif backbone_name == 'swin_large':
            self.backbone = timm.create_model('swin_large_patch4_window7_224', pretrained=pretrain)
            if weights_path is not None:
                state_dict = torch.load(weights_path, map_location="cpu")
                self.backbone.load_state_dict(state_dict)
            self.feature_dim = self.backbone.num_features
            self.backbone.head = nn.Identity()

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x):
        """
        Parameters:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).
        Returns:
            torch.Tensor: Extracted features with shape (batch_size, feature_dim, H', W').
        """

        if self.backbone_name in ['swin_base', 'swin_large']:

            # Extract feature maps using Swin Transformer
            x = self.backbone.patch_embed(x)
            for layer in self.backbone.layers:
                x = layer(x)
            x = self.backbone.norm(x)  # [B, H, W, C]

            # Reshape the sequence back to a 2D feature map
            x = x.permute((0, 3, 1, 2))  # [B, C, H, W]

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        return x


class CKC(nn.Module):
    """
    CKC model, integrating DACM and CPCM modules with the Mamba network for classification.
    Parameters:
        num_class (int, optional): Number of classes for classification. Default is 2.
        channel (int, optional): Input feature dimension. Default is 1024.
        num_clinical (int, optional): Number of clinical attributes. Default is 8.
        out_channel (int, optional): Intermediate output channels. Default is 256.
        num_frame (int, optional): Number of frames in the input video. Default is 32.
    """

    def __init__(self, backbone_name='swin_base', num_class=2, channel=1024, num_clinical=8, out_channel=256,
                 num_frame=4):
        super(CKC, self).__init__()

        # Build the shared backbone network
        self.backbone = SharedBackBone(backbone_name=backbone_name, pretrain=True)

        # Diagnostic Attribute Constraint Module
        self.frame_dacm = DACM(num_clinical=num_clinical, feature_dim=channel, output_dim=out_channel)

        # Residual module for semantic information transformation
        self.residual_block_1 = ResidualBlock(out_channel, out_channel // 4, stride=1, use_gap=True)
        self.residual_block_2 = ResidualBlock(out_channel, out_channel // 4, stride=1, use_gap=True)

        # Clinical Prior Constraint Module
        self.video_cpcm = CPCM(num_frames=num_frame, feature_dim=out_channel, output_dim=out_channel)

        # Mamba network structure
        self.key_mamba = ResidualMamba(d_model=out_channel, expand=2, len_size=num_frame, dt_rank='auto', d_state=16,
                                       d_conv=4, conv_bias=True, bias=False)

        # Global average pooling layer
        self.gap_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification layer
        self.classification = nn.Linear(out_channel, num_class)

    def forward(self, video, key_index):
        """
        Parameters:
            video (torch.Tensor): Input video tensor with shape (batch_size, channels, num_frames, height, width).
            key_index (torch.Tensor): Keyframe index with shape (batch_size,).
        Returns:
            tuple:
                - out_class (torch.Tensor): Classification output with shape (batch_size, num_class).
                - time_attention (torch.Tensor): Temporal attention weights with shape (batch_size, num_frames).
                - clinical_output (torch.Tensor): Clinical attribute output with shape (batch_size, num_clinical).
        """

        batch_size, channels, num_frames, height, width = video.size()

        # Reshape input to process all frames in parallel
        video = video.permute(0, 2, 1, 3, 4).contiguous()  # (batch_size, num_frames, channels, height, width)
        video = video.view(batch_size * num_frames, channels, height, width)

        # Extract features using the shared feature extractor
        features = self.backbone(video)  # (batch_size * num_frames, feature_dim, H', W')

        # Process video frames through the DACM module
        features_f_video, _ = self.frame_dacm(features)
        # frame_feature: (batch_size, out_channel, H', W'), clinical_output: (batch_size, num_clinical)

        # Transform into semantic features through the semantic transformation module
        features_out = self.residual_block_1(features_f_video)

        # Restore frame dimension (batch_size, output_dim, num_frames)
        features_out = F.normalize(features_out, p=2, dim=1)
        features_out = features_out.view(batch_size, num_frames, -1)

        # Prediction result for video diagnosis
        video_feature, time_attention = self.video_cpcm(features_f_video)
        # video_feature: (batch_size, out_channel, num_frames, H', W'), time_attention: (batch_size, num_frames)

        # Retrieve keyframes using advanced indexing, restoring frame dimensions
        # (batch_size, output_dim, num_frames, H', W')
        features = features.view(batch_size, num_frames, features.size(1), features.size(2), features.size(3))
        features = features.permute(0, 2, 1, 3, 4).contiguous()

        # Get keyframe features
        frame_key = features[torch.arange(features.size(0), device=features.device), :, key_index, :, :]

        # Process keyframes through the DACM module
        frame_feature, clinical_output = self.frame_dacm(frame_key)
        # frame_feature: (batch_size, out_channel, H', W'), clinical_output: (batch_size, num_clinical)

        # Transform into semantic features through the semantic transformation module
        frame_feature_out = self.residual_block_2(frame_feature)
        frame_feature_out = F.normalize(frame_feature_out, p=2, dim=1)

        # Use global average pooling to eliminate spatial dimensions, retaining only channel dimensions
        video_feature_pool = self.gap_pool(video_feature).view(batch_size, -1, num_frames)
        frame_feature_pool = self.gap_pool(frame_feature).view(batch_size, -1, 1)

        video_feature_seq = video_feature_pool.permute(0, 2, 1)
        frame_feature_seq = frame_feature_pool.permute(0, 2, 1)

        # Extract temporal information using the Mamba network
        time_feature = self.key_mamba(video_feature_seq, frame_feature_seq)  # (batch_size, 1, out_channel)

        # Aggregate weighted features along the time dimension
        last_feature = time_feature[:, -1, :]  # Select the last time step (B, out_channel)
        mean_feature = time_feature.mean(dim=1)  # Compute mean (B, out_channel)
        time_feature = last_feature + mean_feature

        # Classification layer
        out_class = self.classification(time_feature)  # (batch_size, num_class)

        return out_class, time_attention, clinical_output, frame_feature_out, features_out
