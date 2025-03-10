import torch.nn as nn


class CPCM(nn.Module):
    """
    Clinical Prior Constraints Module
    Parameters:
        num_frames (int): Number of frames in the input video.
        feature_dim (int, optional): Dimension of extracted features, default is 1024.
        output_dim (int, optional): Feature dimension after 1x1 convolution, default is 256.
    """

    def __init__(self, num_frames=32, feature_dim=1024, output_dim=256):
        super(CPCM, self).__init__()

        self.num_frames = num_frames

        # 1x1 convolution for channel reduction
        self.dim_reduce = nn.Sequential(
            nn.Conv2d(feature_dim, output_dim, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True))

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers for generating temporal attention weights
        self.attention_fc = nn.Sequential(
            nn.Linear(num_frames, num_frames // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_frames // 2, num_frames, bias=False),
            nn.Softmax(dim=1))

    def forward(self, x):
        """
        Parameters:
            x (torch.Tensor): Input tensor with shape (batch_size * num_frames, channels, height, width).
        Returns:
            Tuple:
                - weighted_features (torch.Tensor): Weighted features after applying temporal attention,
                  shape (batch_size, output_dim, num_frames, H', W').
                - time_attention (torch.Tensor): Temporal attention weights for each frame,
                  shape (batch_size, num_frames).
        """
        batch_size = x.size(0) // self.num_frames

        # 1x1 convolution for dimension reduction
        x = self.dim_reduce(x)  # Shape: (batch_size * num_frames, output_dim, H', W')

        # Reshape tensor and adjust dimension order
        x = x.view(batch_size, self.num_frames, -1, x.size(-2), x.size(-1)).permute(0, 2, 1, 3, 4)

        # Global average pooling, resulting in (batch_size, output_dim, num_frames, 1, 1)
        pooled_features = self.global_avg_pool(x)

        # Compress spatial dimensions, resulting in (batch_size, output_dim, num_frames)
        pooled_features = pooled_features.view(batch_size, -1, self.num_frames)

        # Compute mean across feature dimensions, resulting in (batch_size, num_frames)
        pooled_features_mean = pooled_features.mean(dim=1)

        # Generate temporal attention weights, shape (batch_size, num_frames)
        time_attention = self.attention_fc(pooled_features_mean)

        # Expand attention weights to match feature dimensions, shape (batch_size, 1, num_frames, 1, 1)
        time_attention_x = time_attention.view(batch_size, 1, self.num_frames, 1, 1)

        # Apply temporal attention weights, shape (batch_size, output_dim, num_frames, H', W')
        weighted_features = x * time_attention_x

        return weighted_features, time_attention
