import torch
import random
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank:
    """Memory Bank for storing and sampling features across batches."""

    def __init__(self, feature_dim, memory_size):
        """
        Initialize the Memory Bank.

        Args:
            feature_dim (int): Dimension of the features.
            memory_size (int): Total size of the Memory Bank.
        """
        self.memory_size = memory_size
        self.features = torch.zeros(memory_size, feature_dim).cuda()  # Initialize stored features as all zeros
        self.ptr = 0  # Initialize pointer position

    def update(self, new_features):
        """
        Update the features in the Memory Bank.

        Args:
            new_features (torch.Tensor): New features with shape [batch_size, feature_dim]
        """
        batch_size = new_features.shape[0]
        new_ptr = (self.ptr + batch_size) % self.memory_size  # Compute new pointer position

        if new_ptr > self.ptr:
            # No circular overwrite needed
            self.features[self.ptr:new_ptr, :] = new_features
        else:
            # Circular overwrite needed
            self.features[self.ptr:, :] = new_features[:self.memory_size - self.ptr]
            self.features[:new_ptr, :] = new_features[self.memory_size - self.ptr:]
        self.ptr = new_ptr  # Update pointer position

    def sample(self, batch_size):
        """
        Randomly sample features from the Memory Bank.

        Args:
            batch_size (int): Number of features to sample.

        Returns:
            torch.Tensor: Sampled features with shape [batch_size, feature_dim]
        """
        idx = random.sample(range(self.memory_size), batch_size)
        return self.features[idx]


class SupConLossWithMemoryBank(nn.Module):
    """Supervised contrastive learning loss with Memory Bank."""

    def __init__(self, temperature=1.0, contrast_mode='all', base_temperature=1.0, memory_size=1024):
        """
        Initialize the supervised contrastive learning loss.

        Args:
            temperature (float): Temperature parameter.
            contrast_mode (str): Contrast mode, default is 'all'.
            base_temperature (float): Base temperature parameter.
            memory_size (int): Size of the Memory Bank.
        """
        super(SupConLossWithMemoryBank, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.memory_bank = None  # Initially, Memory Bank is empty
        self.memory_bank_size = memory_size

    def forward(self, feature1, feature2, labels=None):
        """
        Forward pass to compute supervised contrastive loss.

        Args:
            feature1 (torch.Tensor): First feature with shape [batch_size, len]
            feature2 (torch.Tensor): Second feature with shape [batch_size, num_frame, len]
            labels (torch.Tensor): Labels with shape [batch_size]

        Returns:
            torch.Tensor: Computed loss value
        """
        device = feature1.device
        batch_size = feature1.shape[0]

        # Randomly sample one frame from feature2
        num_frame = feature2.shape[1]  # feature2 has shape [batch_size, num_frame, len]
        sampled_idx = random.randint(0, num_frame - 1)
        sampled_feature2 = feature2[:, sampled_idx, :]  # [batch_size, len]

        # Concatenate feature1 and sampled_feature2
        # Both feature1 and sampled_feature2 have shape [batch_size, len]
        features = torch.stack([feature1, sampled_feature2], dim=1)  # [batch_size, 2, len]
        features = features.view(batch_size * 2, -1)  # [batch_size * 2, len]

        # Initialize Memory Bank
        feature_dim = features.shape[1]
        if self.memory_bank is None:
            self.memory_bank = MemoryBank(feature_dim, self.memory_bank_size)

        # Sample historical features from Memory Bank
        memory_features = self.memory_bank.sample(batch_size).to(device)  # [batch_size, len]

        # Concatenate sampled Memory Bank features with current batch features
        contrast_features = torch.cat([features, memory_features], dim=0)  # [batch_size * 2 + batch_size, len]

        # Expand labels to match the number of feature samples
        labels = labels.contiguous().view(-1, 1)  # [batch_size, 1]
        labels = labels.repeat(2, 1)  # [batch_size * 2, 1]

        # Create a mask indicating which sample pairs are positive pairs
        mask = torch.eq(labels, labels.T).float().to(device)  # [batch_size * 2, batch_size * 2]

        # Create a zero mask for Memory Bank features, indicating they have no positive pairs
        zeros_mask = torch.zeros(batch_size * 2, memory_features.shape[0]).to(device)  # [batch_size * 2, batch_size]

        # Concatenate masks to match the contrast_features dimension
        mask = torch.cat([mask, zeros_mask], dim=1)  # [batch_size * 2, batch_size * 3]

        # Normalize features for numerical stability
        features = F.normalize(features, p=2, dim=1)  # [batch_size * 2, len]
        contrast_features = F.normalize(contrast_features, p=2, dim=1)  # [batch_size * 3, len]

        # Compute similarity logits
        anchor_dot_contrast = torch.div(torch.matmul(features, contrast_features.T),
                                        self.temperature)  # [batch_size * 2, batch_size * 3]
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # [batch_size * 2, 1]
        logits = anchor_dot_contrast - logits_max.detach()  # [batch_size * 2, batch_size * 3]

        # Compute log-probability
        exp_logits = torch.exp(logits)  # [batch_size * 2, batch_size * 3]
        exp_logits = torch.clamp(exp_logits, min=1e-12)  # Prevent numerical instability

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # [batch_size * 2, batch_size * 3]

        # Compute mean log-likelihood of positive pairs
        mask_pos_pairs = mask.sum(1)  # [batch_size * 2]
        mask_pos_pairs = torch.clamp(mask_pos_pairs, min=1.0)  # Prevent division by zero

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs  # [batch_size * 2]

        # Compute loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos  # [batch_size * 2]
        loss = loss.mean()  # [1]

        # Update Memory Bank
        self.memory_bank.update(features.detach())  # Add current batch features to Memory Bank

        return loss
