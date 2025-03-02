import torch
import torch.nn.functional as F


def similarity_frame(videos, key_index):
    """
    计算每个批次中关键帧与其他帧之间的内容相似性和基于帧索引的高斯距离，并进行归一化。
    Args:
        videos (torch.Tensor): 输入视频张量，形状为 [batch, channel, len, width, height]。
        key_index (torch.Tensor): 关键帧的索引，形状为 [batch]，0 <= key_index[i] < len
    Returns:
        combined_normalized (torch.Tensor): 归一化后的相似性和距离的结合张量，形状为 [batch, len]。
    """
    batch_size, channels, seq_len, width, height = videos.shape

    if key_index.dim() != 1 or key_index.size(0) != batch_size:
        raise ValueError(f"key_index 必须是一维张量，长度为 {batch_size}，但获得了形状 {key_index.shape}")

    # 重塑视频张量，将每帧展平成向量 [batch, len, channels * width * height]
    videos_flat = videos.view(batch_size, channels, seq_len, -1)  # [batch, channels, len, width * height]
    videos_flat = videos_flat.permute(0, 2, 1, 3).contiguous()  # [batch, len, channels, width * height]
    videos_flat = videos_flat.view(batch_size, seq_len, -1)  # [batch, len, feature_dim]

    # 提取关键帧特征 key_features: [batch, feature_dim]
    batch_indices = torch.arange(batch_size, device=videos.device)
    key_features = videos_flat[batch_indices, key_index, :]  # [batch, feature_dim]

    # 计算余弦相似度 similarity: [batch, len]
    similarity = F.cosine_similarity(videos_flat, key_features.unsqueeze(1), dim=2)

    # 归一化相似性到 [0, 1]
    similarity_min = similarity.min(dim=1, keepdim=True)[0]
    similarity_max = similarity.max(dim=1, keepdim=True)[0]
    similarity_normalized = (similarity - similarity_min) / (similarity_max - similarity_min + 1e-8)

    return similarity_normalized
