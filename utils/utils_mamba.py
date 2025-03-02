import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum


class RMSNorm(nn.Module):
    """
    参数：
        d_model (int): 输入特征的维度。
        eps (float, optional): 防止除零的微小常数。默认值为 1e-8。
    """

    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # 计算均方根
        rms = x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        x_norm = x / (rms + self.eps)
        return self.weight * x_norm


# 建立 Mamba 网络模块
class MambaBlock(nn.Module):
    """
    MambaBlock 模块，结合特征提取和状态空间模型
    Args:
        d_model (int): 输入隐藏层维度。
        expand (int, optional): 扩展比例。默认值为 2。
        dt_rank (int or str, optional): 数据依赖的秩。默认值为 'auto'，自动计算。
        d_state (int, optional): 状态维度。默认值为 16。
        d_conv (int, optional): 卷积核大小。默认值为 4。
        conv_bias (bool, optional): 卷积层是否包含偏置。默认值为 True。
        bias (bool, optional): 线性层是否包含偏置。默认值为 False。
    """

    def __init__(self, d_model, expand=2, len_size=32, dt_rank='auto', d_state=16, d_conv=4, conv_bias=True,
                 bias=False):
        super(MambaBlock, self).__init__()

        # 参数赋值
        self.d_model = d_model
        self.expand = expand
        self.len_size = len_size
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.bias = bias

        # 计算内部维度
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = max(1, math.ceil(self.d_model / 16))

        # 初始线性映射层，输出维度为 d_inner * 2
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=self.bias)

        # 一维卷积层，提取局部时间信息
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=self.d_conv,
                                groups=self.d_inner, padding=self.d_conv - 1, bias=self.conv_bias)

        # 状态空间模型的线性映射
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.x_proj_s = nn.Linear(self.d_inner, self.dt_rank + (self.d_model * self.expand) * 2, bias=False)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.dt_proj_s = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 关键帧特征映射为初始状态
        self.key_proj_1 = nn.Linear(d_model, self.d_inner, bias=False)
        self.key_proj_2 = nn.Linear(1, self.d_state, bias=False)
        self.key_proj_s = nn.Linear(d_model, self.len_size, bias=False)

        # 初始化状态转移矩阵 A
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0)  # (1, d_state)
        self.A_log = nn.Parameter(torch.log(A))  # 可学习参数

        A_b = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0)  # (1, d_state)
        self.A_b_log = nn.Parameter(torch.log(A_b))  # 可学习参数

        A_s = torch.arange(1, self.d_model * self.expand + 1, dtype=torch.float32).unsqueeze(0)  # (1, d_state)
        self.A_s_log = nn.Parameter(torch.log(A_s))  # 可学习参数

        # 初始化可学习参数 D
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D_b = nn.Parameter(torch.ones(self.d_inner))
        self.D_s = nn.Parameter(torch.ones(self.d_inner))

        # 输出线性映射层
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=self.bias)

    def forward(self, x, x_key):
        """
        参数：
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)。
            x_key (torch.Tensor): 关键帧特征，形状为 (batch_size, d_model)。
        返回：
            torch.Tensor: 输出张量，形状为 (batch_size, seq_len, d_model)。
        """
        batch_size, seq_len, d = x.shape

        # 初始线性映射并分割为 x 和 res
        x_proj = self.in_proj(x)  # (batch_size, seq_len, d_inner * 2)
        x, res = x_proj.chunk(2, dim=-1)  # (batch_size, seq_len, d_inner)

        # 一维卷积处理
        x = rearrange(x, 'b l d -> b d l')  # (batch_size, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]  # (batch_size, d_inner, seq_len)
        x = rearrange(x, 'b d l -> b l d')  # (batch_size, seq_len, d_inner)

        # 激活函数
        x = F.silu(x)

        # 状态空间模型处理
        y = self.ssm(x, x_key)

        # 加上残差并通过输出映射
        y = y + res
        y = self.out_proj(y)

        return y

    def ssm(self, x, x_key):
        """
        状态空间模型处理
        参数：
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_inner)。
            x_key (torch.Tensor): 关键帧特征，形状为 (batch_size, d_model)。
        返回：
            torch.Tensor: 状态空间模型的输出，形状为 (batch_size, seq_len, d_inner)。
        """
        A = -torch.exp(self.A_log).to(x.device)  # (1, d_state)
        A_b = -torch.exp(self.A_b_log).to(x.device)  # (1, d_state)
        A_s = -torch.exp(self.A_s_log).to(x.device)  # (1, d_state)

        # 数据依赖的 delta, B, C
        x_proj = self.x_proj(x)  # (batch_size, seq_len, dt_rank + 2 * d_state)
        x_proj_b = self.x_proj_b(x)  # (batch_size, seq_len, dt_rank + 2 * d_state)
        x_proj_s = self.x_proj_s(x)  # (batch_size, seq_len, dt_rank + 2 * d_state)

        delta_raw, B_raw, C_raw = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta_raw_b, B_raw_b, C_raw_b = torch.split(x_proj_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta_raw_s, B_raw_s, C_raw_s = torch.split(x_proj_s, [self.dt_rank, self.d_model * self.expand,
                                                               self.d_model * self.expand], dim=-1)

        # delta 的线性映射并激活
        delta = F.softplus(self.dt_proj(delta_raw))  # (batch_size, seq_len, d_inner)
        delta_b = F.softplus(self.dt_proj_b(delta_raw_b))  # (batch_size, seq_len, d_inner)
        delta_s = F.softplus(self.dt_proj_s(delta_raw_s))  # (batch_size, seq_len, d_inner)

        # 双向扫描
        y_forward = self.selective_scan(x, x_key, delta, A, B_raw, C_raw, self.D, direction='forward')
        y_backward = self.selective_scan(x, x_key, delta_b, A_b, B_raw_b, C_raw_b, self.D_b, direction='backward')
        y_spatial = self.selective_scan_s(x, x_key, delta_s, A_s, B_raw_s, C_raw_s, self.D_s)

        # 双向结果相加
        y = y_forward + y_backward + y_spatial

        return y

    def selective_scan(self, u, x_key, delta, A, B, C, D, direction='forward'):
        """
        选择性扫描，执行前向或后向时间序列处理。
        参数：
            u (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_inner)。
            delta (torch.Tensor): delta 权重，形状为 (batch_size, seq_len, d_inner)。
            A (torch.Tensor): 状态转移矩阵，形状为 (1, d_state)。
            B_raw (torch.Tensor): 输入矩阵，形状为 (batch_size, seq_len, d_state)。
            C_raw (torch.Tensor): 输出矩阵，形状为 (batch_size, seq_len, d_state)。
            x0 (torch.Tensor): 初始状态，形状为 (batch_size, d_state)。
            direction (str, optional): 扫描方向，'forward' 或 'backward'。
        返回：
            torch.Tensor: 扫描后的输出，形状为 (batch_size, seq_len, d_inner)。
        """
        b, seq_len, d_in = u.shape

        # 离散化 A 和 B
        delta_A = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        delta_B_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # 初始状态
        x_key = self.key_proj_1(x_key).view(b, d_in, 1)  # (batch_size, d_inner, 1)
        x = self.key_proj_2(x_key).squeeze(-1)  # (batch_size, d_state)

        if direction == 'forward':
            time_steps = range(seq_len)
        elif direction == 'backward':
            time_steps = reversed(range(seq_len))
        else:
            raise ValueError("direction must be 'forward' or 'backward'")

        ys = []
        for i in time_steps:
            x = delta_A[:, i] * x + delta_B_u[:, i]  # [8 10 32 16] [8 32 16] [8 10 32 16]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')  # [8 10 16]
            ys.append(y)

        if direction == 'backward':
            ys = ys[::-1]

        y = torch.stack(ys, dim=1)  # (batch_size, seq_len)
        y = y + u * D.unsqueeze(0).unsqueeze(1)  # 广播 D

        return y

    def selective_scan_s(self, u, x_key, delta, A, B, C, D):
        """
        选择性扫描，执行前向或后向时间序列处理。
        参数：
            u (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_inner)。
            delta (torch.Tensor): delta 权重，形状为 (batch_size, seq_len, d_inner)。
            A (torch.Tensor): 状态转移矩阵，形状为 (1, d_state)。
            B_raw (torch.Tensor): 输入矩阵，形状为 (batch_size, seq_len, d_state)。
            C_raw (torch.Tensor): 输出矩阵，形状为 (batch_size, seq_len, d_state)。
            x0 (torch.Tensor): 初始状态，形状为 (batch_size, d_state)。
            direction (str, optional): 扫描方向，'forward' 或 'backward'。
        返回：
            torch.Tensor: 扫描后的输出，形状为 (batch_size, seq_len, d_inner)。
        """
        b, seq_len, d_in = u.shape

        # 离散化 A 和 B
        delta_A = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l n'))
        delta_B_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l n')

        # 初始状态
        x = self.key_proj_s(x_key).view(b, seq_len)

        time_steps = range(delta_A.size(2))
        ys = []
        for i in time_steps:
            x = delta_A[:, :, i] * x + delta_B_u[:, :, i]  # [8 10 32*16] [8 10] [8 10 32*16]
            y = einsum(x, C[:, :, i], 'b n, b n -> b n')  # [8 10 16]
            ys.append(y)

        y = torch.stack(ys, dim=2)  # (batch_size, seq_len)
        y = y + u * D.unsqueeze(0).unsqueeze(1)  # 广播 D

        return y


class ResidualMamba(nn.Module):
    """
    ResidualMamba 模块，结合 MambaBlock 和 RMSNorm 进行残差连接。
    参数：
        d_model (int): 输入隐藏层维度。
        expand (int, optional): MambaBlock 的扩展比例。默认值为 2。
        dt_rank (int or str, optional): MambaBlock 的数据依赖秩。默认值为 'auto'。
        d_state (int, optional): MambaBlock 的状态维度。默认值为 16。
        d_conv (int, optional): MambaBlock 的卷积核大小。默认值为 4。
        conv_bias (bool, optional): MambaBlock 的卷积层是否包含偏置。默认值为 True。
        bias (bool, optional): MambaBlock 的线性层是否包含偏置。默认值为 False。
    """

    def __init__(self, d_model, expand=2, len_size=32, dt_rank='auto', d_state=16, d_conv=4,
                 conv_bias=True, bias=False):
        super(ResidualMamba, self).__init__()
        self.mamba = MambaBlock(d_model=d_model, expand=expand, len_size=len_size, dt_rank=dt_rank,
                                d_state=d_state, d_conv=d_conv, conv_bias=conv_bias, bias=bias)
        self.norm = RMSNorm(d_model=d_model)

    def forward(self, x, x_key):
        """
        参数：
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)。
            x_key (torch.Tensor): 关键帧特征，形状为 (batch_size, d_model)。
        返回：
            torch.Tensor: 输出张量，形状为 (batch_size, seq_len, d_model)。
        """
        return self.norm(x + self.mamba(x, x_key))
