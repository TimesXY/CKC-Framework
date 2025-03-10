import torch
import torch.nn.functional as F


def similarity_frame(videos, key_index):
    """
    Compute the content similarity between the key frame and other frames in each batch,
    as well as the Gaussian distance based on frame index, and normalize the results.

    Args:
        videos (torch.Tensor): Input video tensor with shape [batch, channel, len, width, height].
        key_index (torch.Tensor): Indices of key frames with shape [batch], where 0 <= key_index[i] < len.

    Returns:
        combined_normalized (torch.Tensor): A tensor combining the normalized similarity and distance,
                                            with shape [batch, len].
    """
    batch_size, channels, seq_len, width, height = videos.shape

    if key_index.dim() != 1 or key_index.size(0) != batch_size:
        raise ValueError(f"key_index must be a 1D tensor with length {batch_size}, but got shape {key_index.shape}")

    # Reshape the video tensor, flattening each frame into a vector [batch, len, channels * width * height]
    videos_flat = videos.view(batch_size, channels, seq_len, -1)  # [batch, channels, len, width * height]
    videos_flat = videos_flat.permute(0, 2, 1, 3).contiguous()  # [batch, len, channels, width * height]
    videos_flat = videos_flat.view(batch_size, seq_len, -1)  # [batch, len, feature_dim]

    # Extract key frame features key_features: [batch, feature_dim]
    batch_indices = torch.arange(batch_size, device=videos.device)
    key_features = videos_flat[batch_indices, key_index, :]  # [batch, feature_dim]

    # Compute cosine similarity similarity: [batch, len]
    similarity = F.cosine_similarity(videos_flat, key_features.unsqueeze(1), dim=2)

    # Normalize similarity to [0, 1]
    similarity_min = similarity.min(dim=1, keepdim=True)[0]
    similarity_max = similarity.max(dim=1, keepdim=True)[0]
    similarity_normalized = (similarity - similarity_min) / (similarity_max - similarity_min + 1e-8)

    return similarity_normalized
