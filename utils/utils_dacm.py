import torch
import torch.nn as nn


class DACM(nn.Module):
    """
    Diagnostic Attribute Constraints Module
    Args:
        num_clinical (int): Number of diagnostic attributes to predict.
        feature_dim (int, optional): Dimension of extracted features. Default is 1024.
        output_dim (int, optional): Dimension of output features. Default is 256.
    """

    def __init__(self, num_clinical, feature_dim=1024, output_dim=256, num_class=2):
        super(DACM, self).__init__()
        self.feature_dim = feature_dim

        # Explicit diagnostic features
        self.explicit_path = nn.Sequential(
            nn.Conv2d(feature_dim, output_dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True))

        # Explicit diagnostic feature constraints
        self.explicit_class = nn.Sequential(
            nn.Conv2d(output_dim, num_class, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_class),
            nn.ReLU(inplace=True))

        # Implicit diagnostic features
        self.implicit_path = nn.Sequential(
            nn.Conv2d(feature_dim, output_dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True))

        # Global average pooling - compress to the specified dimension
        self.global_avg_pool = nn.AdaptiveAvgPool2d((num_clinical, 1))

        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * output_dim, output_dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).
        Returns:
            tuple:
                - fused_features (torch.Tensor): Fused features with shape (batch_size, feature_dim, H, W).
                - clinical_output (torch.Tensor): Predicted diagnostic attributes
                  with shape (batch_size, num_class, num_clinical).
        """

        # Explicit diagnostic feature constraints
        explicit_features = self.explicit_path(x)
        explicit_output = self.explicit_class(explicit_features)

        # Global average pooling and fully connected layer to obtain diagnostic attribute predictions
        clinical_output = self.global_avg_pool(explicit_output).squeeze(-1)

        # Implicit diagnostic features
        implicit_features = self.implicit_path(x)

        # Concatenate explicit and implicit diagnostic features along the channel dimension
        fused_embeddings = torch.cat([explicit_features, implicit_features], dim=1)

        # Fuse feature information
        fused_features = self.fusion(fused_embeddings)

        return fused_features, clinical_output
