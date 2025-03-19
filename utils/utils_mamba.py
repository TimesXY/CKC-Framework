import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum


class RMSNorm(nn.Module):
    """
    Parameters:
        d_model (int): Dimension of the input features.
        eps (float, optional): A small constant to prevent division by zero. Default is 1e-8.
    """

    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Calculate root mean square
        rms = x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        x_norm = x / (rms + self.eps)
        return self.weight * x_norm


# Build Mamba network module
class MambaBlock(nn.Module):
    """
    MambaBlock module, combining feature extraction and state space model.
    Args:
        d_model (int): Dimension of the input hidden layer.
        expand (int, optional): Expansion ratio. Default is 2.
        dt_rank (int or str, optional): Data-dependent rank. Default is 'auto', automatically computed.
        d_state (int, optional): State dimension. Default is 16.
        d_conv (int, optional): Convolution kernel size. Default is 4.
        conv_bias (bool, optional): Whether the convolution layer includes bias. Default is True.
        bias (bool, optional): Whether the linear layer includes bias. Default is False.
    """

    def __init__(self, d_model, expand=2, len_size=32, dt_rank='auto', d_state=16, d_conv=4, conv_bias=True,
                 bias=False):
        super(MambaBlock, self).__init__()

        # Assign parameters
        self.d_model = d_model
        self.expand = expand
        self.len_size = len_size
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.bias = bias

        # Calculate internal dimension
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = max(1, math.ceil(self.d_model / 16))

        # Initial linear mapping layer, output dimension is d_inner * 2
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=self.bias)

        # 1D convolution layer to extract local temporal information
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=self.d_conv,
                                groups=self.d_inner, padding=self.d_conv - 1, bias=self.conv_bias)

        # Linear mappings for the state space model
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.x_proj_s = nn.Linear(self.d_inner, self.dt_rank + (self.d_model * self.expand) * 2, bias=False)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.dt_proj_s = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Map key frame features to the initial state
        self.key_proj_1 = nn.Linear(d_model, self.d_inner, bias=False)
        self.key_proj_2 = nn.Linear(1, self.d_state, bias=False)
        self.key_proj_s = nn.Linear(d_model, self.len_size, bias=False)

        # Initialize state transition matrix A
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0)  # (1, d_state)
        self.A_log = nn.Parameter(torch.log(A))  # Learnable parameter

        A_b = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0)  # (1, d_state)
        self.A_b_log = nn.Parameter(torch.log(A_b))  # Learnable parameter

        A_s = torch.arange(1, self.d_model * self.expand + 1, dtype=torch.float32).unsqueeze(0)  # (1, d_state)
        self.A_s_log = nn.Parameter(torch.log(A_s))  # Learnable parameter

        # Initialize learnable parameter D
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D_b = nn.Parameter(torch.ones(self.d_inner))
        self.D_s = nn.Parameter(torch.ones(self.d_inner))

        # Output linear mapping layer
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=self.bias)

    def forward(self, x, x_key):
        """
        Parameters:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_model).
            x_key (torch.Tensor): Key frame features, shape (batch_size, d_model).
        Returns:
            torch.Tensor: Output tensor, shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, d = x.shape

        # Initial linear mapping and split into x and res
        x_proj = self.in_proj(x)  # (batch_size, seq_len, d_inner * 2)
        x, res = x_proj.chunk(2, dim=-1)  # (batch_size, seq_len, d_inner)

        # 1D convolution processing
        x = rearrange(x, 'b l d -> b d l')  # (batch_size, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]  # (batch_size, d_inner, seq_len)
        x = rearrange(x, 'b d l -> b l d')  # (batch_size, seq_len, d_inner)

        # Activation function
        x = F.silu(x)

        # State space model processing
        y = self.ssm(x, x_key)

        # Add residual and pass through output mapping
        y = y + res
        y = self.out_proj(y)

        return y

    def ssm(self, x, x_key):
        """
        State space model processing.
        Parameters:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_inner).
            x_key (torch.Tensor): Key frame features, shape (batch_size, d_model).
        Returns:
            torch.Tensor: Output of the state space model, shape (batch_size, seq_len, d_inner).
        """
        A = -torch.exp(self.A_log).to(x.device)  # (1, d_state)
        A_b = -torch.exp(self.A_b_log).to(x.device)  # (1, d_state)
        A_s = -torch.exp(self.A_s_log).to(x.device)  # (1, d_state)

        # Data-dependent delta, B, C
        x_proj = self.x_proj(x)  # (batch_size, seq_len, dt_rank + 2 * d_state)
        x_proj_b = self.x_proj_b(x)  # (batch_size, seq_len, dt_rank + 2 * d_state)
        x_proj_s = self.x_proj_s(x)  # (batch_size, seq_len, dt_rank + 2 * d_state)

        delta_raw, B_raw, C_raw = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta_raw_b, B_raw_b, C_raw_b = torch.split(x_proj_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta_raw_s, B_raw_s, C_raw_s = torch.split(x_proj_s, [self.dt_rank, self.d_model * self.expand,
                                                               self.d_model * self.expand], dim=-1)

        # Linear mapping and activation for delta
        delta = F.softplus(self.dt_proj(delta_raw))  # (batch_size, seq_len, d_inner)
        delta_b = F.softplus(self.dt_proj_b(delta_raw_b))  # (batch_size, seq_len, d_inner)
        delta_s = F.softplus(self.dt_proj_s(delta_raw_s))  # (batch_size, seq_len, d_inner)

        # Bidirectional scanning
        y_forward = self.selective_scan(x, x_key, delta, A, B_raw, C_raw, self.D, direction='forward')
        y_backward = self.selective_scan(x, x_key, delta_b, A_b, B_raw_b, C_raw_b, self.D_b, direction='backward')
        y_spatial = self.selective_scan_s(x, x_key, delta_s, A_s, B_raw_s, C_raw_s, self.D_s)

        # Add bidirectional results
        y = y_forward + y_backward + y_spatial

        return y

    def selective_scan(self, u, x_key, delta, A, B, C, D, direction='forward'):
        """
        Selective scan, performing forward or backward time series processing.
        Parameters:
            u (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_inner).
            delta (torch.Tensor): Delta weights, shape (batch_size, seq_len, d_inner).
            A (torch.Tensor): State transition matrix, shape (1, d_state).
            B_raw (torch.Tensor): Input matrix, shape (batch_size, seq_len, d_state).
            C_raw (torch.Tensor): Output matrix, shape (batch_size, seq_len, d_state).
            x0 (torch.Tensor): Initial state, shape (batch_size, d_state).
            direction (str, optional): Scan direction, 'forward' or 'backward'.
        Returns:
            torch.Tensor: Output after scanning, shape (batch_size, seq_len, d_inner).
        """
        b, seq_len, d_in = u.shape

        # Discretize A and B
        delta_A = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        delta_B_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # Initial state
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
        y = y + u * D.unsqueeze(0).unsqueeze(1)  # Broadcast D

        return y

    def selective_scan_s(self, u, x_key, delta, A, B, C, D):
        """
        Selective scan, performing forward or backward time series processing.
        Parameters:
            u (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_inner).
            delta (torch.Tensor): Delta weights, shape (batch_size, seq_len, d_inner).
            A (torch.Tensor): State transition matrix, shape (1, d_state).
            B_raw (torch.Tensor): Input matrix, shape (batch_size, seq_len, d_state).
            C_raw (torch.Tensor): Output matrix, shape (batch_size, seq_len, d_state).
            x0 (torch.Tensor): Initial state, shape (batch_size, d_state).
            direction (str, optional): Scan direction, 'forward' or 'backward'.
        Returns:
            torch.Tensor: Output after scanning, shape (batch_size, seq_len, d_inner).
        """
        b, seq_len, d_in = u.shape

        # Discretize A and B
        delta_A = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l n'))
        delta_B_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l n')

        # Initial state
        x = self.key_proj_s(x_key).view(b, seq_len)

        time_steps = range(delta_A.size(2))
        ys = []
        for i in time_steps:
            x = delta_A[:, :, i] * x + delta_B_u[:, :, i]  # [8 10 32*16] [8 10] [8 10 32*16]
            y = einsum(x, C[:, :, i], 'b n, b n -> b n')  # [8 10 16]
            ys.append(y)

        y = torch.stack(ys, dim=2)  # (batch_size, seq_len)
        y = y + u * D.unsqueeze(0).unsqueeze(1)  # Broadcast D

        return y


class ResidualMamba(nn.Module):
    """
    ResidualMamba module, combining MambaBlock and RMSNorm for residual connections.
    Parameters:
        d_model (int): Dimension of the input hidden layer.
        expand (int, optional): Expansion ratio for MambaBlock. Default is 2.
        dt_rank (int or str, optional): Data-dependent rank for MambaBlock. Default is 'auto'.
        d_state (int, optional): State dimension for MambaBlock. Default is 16.
        d_conv (int, optional): Convolution kernel size for MambaBlock. Default is 4.
        conv_bias (bool, optional): Whether MambaBlock's convolution layer includes bias. Default is True.
        bias (bool, optional): Whether MambaBlock's linear layer includes bias. Default is False.
    """

    def __init__(self, d_model, expand=2, len_size=32, dt_rank='auto', d_state=16, d_conv=4,
                 conv_bias=True, bias=False):
        super(ResidualMamba, self).__init__()
        self.mamba = MambaBlock(d_model=d_model, expand=expand, len_size=len_size, dt_rank=dt_rank,
                                d_state=d_state, d_conv=d_conv, conv_bias=conv_bias, bias=bias)
        self.norm = RMSNorm(d_model=d_model)

    def forward(self, x, x_key):
        """
        Parameters:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_model).
            x_key (torch.Tensor): Key frame features, shape (batch_size, d_model).
        Returns:
            torch.Tensor: Output tensor, shape (batch_size, seq_len, d_model).
        """
        return self.norm(x + self.mamba(x, x_key))
