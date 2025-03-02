import torch
import random
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank:
    """Memory Bank 用于跨 batch 存储和采样特征."""

    def __init__(self, feature_dim, memory_size):
        """
        初始化 Memory Bank.

        参数:
            feature_dim (int): 特征的维度大小.
            memory_size (int): Memory Bank 的总大小.
        """
        self.memory_size = memory_size
        self.features = torch.zeros(memory_size, feature_dim).cuda()  # 初始化存储特征为全零
        self.ptr = 0  # 初始化指针位置

    def update(self, new_features):
        """
        更新 Memory Bank 中的特征.

        参数:
            new_features (torch.Tensor): 新的特征，形状为 [batch_size, feature_dim]
        """
        batch_size = new_features.shape[0]
        new_ptr = (self.ptr + batch_size) % self.memory_size  # 计算新的指针位置

        if new_ptr > self.ptr:
            # 不需要循环覆盖
            self.features[self.ptr:new_ptr, :] = new_features
        else:
            # 需要循环覆盖
            self.features[self.ptr:, :] = new_features[:self.memory_size - self.ptr]
            self.features[:new_ptr, :] = new_features[self.memory_size - self.ptr:]
        self.ptr = new_ptr  # 更新指针位置

    def sample(self, batch_size):
        """
        从 Memory Bank 中随机采样特征.

        参数:
            batch_size (int): 采样的特征数量.

        返回:
            torch.Tensor: 采样的特征，形状为 [batch_size, feature_dim]
        """
        idx = random.sample(range(self.memory_size), batch_size)
        return self.features[idx]


class SupConLossWithMemoryBank(nn.Module):
    """带有 Memory Bank 的监督对比学习损失."""

    def __init__(self, temperature=1.0, contrast_mode='all', base_temperature=1.0, memory_size=1024):
        """
        初始化监督对比学习损失.

        参数:
            temperature (float): 温度参数.
            contrast_mode (str): 对比模式，默认为 'all'.
            base_temperature (float): 基础温度参数.
            memory_size (int): Memory Bank 的大小.
        """
        super(SupConLossWithMemoryBank, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.memory_bank = None  # 初始时 Memory Bank 为空
        self.memory_bank_size = memory_size

    def forward(self, feature1, feature2, labels=None):
        """
        前向传播，计算监督对比损失.

        参数:
            feature1 (torch.Tensor): 第一个特征，形状为 [batch_size, len]
            feature2 (torch.Tensor): 第二个特征，形状为 [batch_size, num_frame, len]
            labels (torch.Tensor): 标签，形状为 [batch_size]

        返回:
            torch.Tensor: 计算得到的损失值
        """
        device = feature1.device
        batch_size = feature1.shape[0]

        # 从 feature2 中随机采样一个帧
        num_frame = feature2.shape[1]  # feature2 的形状为 [batch_size, num_frame, len]
        sampled_idx = random.randint(0, num_frame - 1)
        sampled_feature2 = feature2[:, sampled_idx, :]  # [batch_size, len]

        # 拼接 feature1 和 sampled_feature2
        # feature1 和 sampled_feature2 均为 [batch_size, len]
        features = torch.stack([feature1, sampled_feature2], dim=1)  # [batch_size, 2, len]
        features = features.view(batch_size * 2, -1)  # [batch_size * 2, len]

        # 初始化 Memory Bank
        feature_dim = features.shape[1]
        if self.memory_bank is None:
            self.memory_bank = MemoryBank(feature_dim, self.memory_bank_size)

        # 从 Memory Bank 中采样历史特征
        memory_features = self.memory_bank.sample(batch_size).to(device)  # [batch_size, len]

        # 将 Memory Bank 中采样的特征与当前 batch 的特征拼接
        contrast_features = torch.cat([features, memory_features], dim=0)  # [batch_size * 2 + batch_size, len]

        # 扩展标签以匹配 features 的样本数
        labels = labels.contiguous().view(-1, 1)  # [batch_size, 1]
        labels = labels.repeat(2, 1)  # [batch_size * 2, 1]

        # 生成掩码，表示哪些样本对是正样本对
        mask = torch.eq(labels, labels.T).float().to(device)  # [batch_size * 2, batch_size * 2]

        # 为 Memory Bank 中的特征创建零掩码，表示这些特征没有正样本对
        zeros_mask = torch.zeros(batch_size * 2, memory_features.shape[0]).to(device)  # [batch_size * 2, batch_size]

        # 拼接掩码，使其与 contrast_features 的维度匹配
        mask = torch.cat([mask, zeros_mask], dim=1)  # [batch_size * 2, batch_size * 3]

        # 对特征进行归一化，确保数值稳定性
        features = F.normalize(features, p=2, dim=1)  # [batch_size * 2, len]
        contrast_features = F.normalize(contrast_features, p=2, dim=1)  # [batch_size * 3, len]

        # 计算相似度 logit
        anchor_dot_contrast = torch.div(torch.matmul(features, contrast_features.T),
                                        self.temperature)  # [batch_size * 2, batch_size * 3]
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # [batch_size * 2, 1]
        logits = anchor_dot_contrast - logits_max.detach()  # [batch_size * 2, batch_size * 3]

        # 计算 log-概率
        exp_logits = torch.exp(logits)  # [batch_size * 2, batch_size * 3]
        exp_logits = torch.clamp(exp_logits, min=1e-12)  # 防止数值过大或过小

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # [batch_size * 2, batch_size * 3]

        # 计算正样本的 log-似然均值
        mask_pos_pairs = mask.sum(1)  # [batch_size * 2]
        mask_pos_pairs = torch.clamp(mask_pos_pairs, min=1.0)  # 防止分母为 0

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs  # [batch_size * 2]

        # 计算损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos  # [batch_size * 2]
        loss = loss.mean()  # [1]

        # 更新 Memory Bank
        self.memory_bank.update(features.detach())  # 将当前 batch 的特征加入 Memory Bank

        return loss
