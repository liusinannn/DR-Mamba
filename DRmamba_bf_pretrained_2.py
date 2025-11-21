import os
import time
import math
import numpy as np
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
# from mpl_toolkits.mplot3d.proj3d import transform
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import selective scan ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    print(e, flush=True)


# cross selective scan ===============================
class SelectiveScanCore(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        # print(1)
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        # print(2)
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


# =============
def antidiagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)


def diagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)


def diagonal_scatter(tensor_flat, original_shape):
    # 把斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor_flat.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 创建一个空的张量来存储反向散布的结果
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, H, W]，考虑到需要使用transpose将H和W调换
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_根据expanded_index将元素放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


def antidiagonal_scatter(tensor_flat, original_shape):
    # 把反斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor_flat.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 初始化一个与原始张量形状相同、元素全为0的张量
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, W, H]，因为操作是沿最后一个维度收集的，需要调整形状并交换维度
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_将元素根据索引放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor

###窗口扫描
def windowed_scan(x: torch.Tensor, window_size: int):
    """
    执行窗口化扫描。
    将 (B, C, H, W) 的张量划分为 (window_size, window_size) 的窗口，
    在每个窗口内进行行优先扫描，然后将结果串联起来。
    """
    B, C, H, W = x.shape
    # 确保图像尺寸可以被窗口大小整除
    assert H % window_size == 0 and W % window_size == 0, \
        f"Image dimensions ({H}, {W}) must be divisible by window_size ({window_size})."

    # 1. 将图像划分为窗口
    # (B, C, H, W) -> (B, C, H/ws, ws, W/ws, ws)
    x_windows = x.view(B, C, H // window_size, window_size, W // window_size, window_size)

    # 2. 交换维度，将窗口内的像素组合在一起
    # (B, C, H/ws, ws, W/ws, ws) -> (B, C, H/ws, W/ws, ws, ws)
    scanned_windows = x_windows.permute(0, 1, 2, 4, 3, 5).contiguous()

    # 3. 展平为最终序列
    # (B, C, H/ws, W/ws, ws, ws) -> (B, C, H * W)
    return scanned_windows.view(B, C, H * W)


def windowed_scatter(ys: torch.Tensor, window_size: int, original_shape: tuple):
    """
    windowed_scan 的逆操作。
    将扁平化的梯度序列还原为原始的图像网格格式。
    """
    B, C, H, W = original_shape
    assert H % window_size == 0 and W % window_size == 0, \
        f"Image dimensions ({H}, {W}) must be divisible by window_size ({window_size})."

    # 1. 还原窗口化的展平视图
    # (B, C, H * W) -> (B, C, H/ws, W/ws, ws, ws)
    ys_windows = ys.view(B, C, H // window_size, W // window_size, window_size, window_size)

    # 2. 执行逆向的维度交换，将像素放回其空间位置
    # (B, C, H/ws, W/ws, ws, ws) -> (B, C, H/ws, ws, W/ws, ws)
    scattered_windows = ys_windows.permute(0, 1, 2, 4, 3, 5).contiguous()

    # 3. 还原为原始图像形状
    # (B, C, H/ws, ws, W/ws, ws) -> (B, C, H, W)
    return scattered_windows.view(B, C, H, W)


class CrossScan(torch.autograd.Function):
    """
    CrossScan Function
    将输入张量 x (B, C, H, W) 按照四种方式进行扫描：
    1. 标准行优先扫描 (全局)
    2. 2x2 窗口化行优先扫描
    3. 4x4 窗口化行优先扫描
    4. 8x8 窗口化行优先扫描
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)

        # 新的输出张量，包含4种扫描方式
        xs = x.new_empty((B, 4, C, H * W))

        # 扫描方式 1: 标准行优先扫描 (整张图)
        xs[:, 0] = x.flatten(2, 3)

        # 扫描方式 2: 2x2 窗口化扫描
        xs[:, 1] = windowed_scan(x, 2)

        # 扫描方式 3: 4x4 窗口化扫描
        xs[:, 2] = windowed_scan(x, 4)

        # 扫描方式 4: 8x8 窗口化扫描
        xs[:, 3] = windowed_scan(x, 8)

        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # ys 是从后续层传来的梯度，形状为 (B, 4, C, H * W)
        B, C, H, W = ctx.shape

        # 梯度是四条路径的梯度之和

        # 路径 1 的梯度: 标准行优先扫描的逆操作
        y_row_major = ys[:, 0].view(B, C, H, W)

        # 路径 2 的梯度: 2x2 窗口化扫描的逆操作
        y_win2 = windowed_scatter(ys[:, 1], 2, (B, C, H, W))

        # 路径 3 的梯度: 4x4 窗口化扫描的逆操作
        y_win4 = windowed_scatter(ys[:, 2], 4, (B, C, H, W))

        # 路径 4 的梯度: 8x8 窗口化扫描的逆操作
        y_win8 = windowed_scatter(ys[:, 3], 8, (B, C, H, W))

        # 将来自四条路径的梯度相加，得到最终传给前一层的梯度
        y_res = y_row_major + y_win2 + y_win4 + y_win8

        return y_res


class CrossMerge(torch.autograd.Function):
    """
    CrossMerge Function
    Merges 4 processed scan sequences back into a single tensor.
    This is the inverse operation of the modified CrossScan.
    """

    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        # Input ys has shape (B, K, D, H, W), where K=4 for our 4 scan types
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)

        # Flatten the spatial dimensions for processing
        # (B, 4, D, H, W) -> (B, 4, D, L) where L = H * W
        ys = ys.view(B, K, D, -1)

        # "Un-scan" or "scatter" each of the 4 paths back to the image format

        # Path 1: Standard row-major. Inverse is a simple view().
        y1 = ys[:, 0].view(B, D, H, W)

        # Path 2: 2x2 windowed scan. Use windowed_scatter to invert.
        y2 = windowed_scatter(ys[:, 1], 2, (B, D, H, W))

        # Path 3: 4x4 windowed scan. Use windowed_scatter to invert.
        y3 = windowed_scatter(ys[:, 2], 4, (B, D, H, W))

        # Path 4: 8x8 windowed scan. Use windowed_scatter to invert.
        y4 = windowed_scatter(ys[:, 3], 8, (B, D, H, W))

        # The final result is the sum of the un-scanned paths
        y_res = y1 + y2 + y3 + y4

        # Return in the flattened format (B, D, L) as in the original code
        return y_res.view(B, D, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # The backward pass of the merge operation is the scan operation itself.
        # Input x is the upstream gradient, shape (B, D, L)

        H, W = ctx.shape
        B, D, L = x.shape

        # Reshape gradient to image format to perform scans
        x_img = x.view(B, D, H, W)

        # Create output tensor for gradients, shape (B, 4, D, L)
        xs = x.new_empty((B, 4, D, L))

        # Apply the 4 scanning patterns to the gradient, creating 4 gradient paths

        # Path 1: Standard row-major scan
        xs[:, 0] = x_img.flatten(2, 3)

        # Path 2: 2x2 windowed scan
        xs[:, 1] = windowed_scan(x_img, 2)

        # Path 3: 4x4 windowed scan
        xs[:, 2] = windowed_scan(x_img, 4)

        # Path 4: 8x8 windowed scan
        xs[:, 3] = windowed_scan(x_img, 8)

        # Return in the 5D format (B, K, D, H, W) expected by the previous layer
        return xs.view(B, 4, D, H, W)

# =============
# ZSJ 这里是mamba的具体内容，要增加扫描方向就在这里改

def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        delta_softplus=True,
        out_norm: torch.nn.Module = None,
        out_norm_shape="v0",
        # ==============================
        to_dtype=True,  # True: final out to dtype
        force_fp32=False,  # True: input fp32
        # ==============================
        nrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows == 0:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    if backnrows == 0:
        if D % 4 == 0:
            backnrows = 4
        elif D % 3 == 0:
            backnrows = 3
        elif D % 2 == 0:
            backnrows = 2
        else:
            backnrows = 1

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    xs = CrossScan.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)
    # ZSJ 这里把矩阵拆分成不同方向的序列，并进行扫描
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)
    # ZSJ 这里把处理之后的序列融合起来，并还原回原来的矩阵形式
    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]:  # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
    else:  # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)



# =====================================================
class OSSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        if forward_type[-len("none"):] == "none":
            forward_type = forward_type[:-len("none")]
            self.out_norm = nn.Identity()
        elif forward_type[-len("dwconv3"):] == "dwconv3":
            forward_type = forward_type[:-len("dwconv3")]
            self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False)
            self.out_norm_shape = "v1"
        elif forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanCore),
        )

        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        # ZSJ k_group 指的是扫描的方向
        # k_group = 4 if forward_type not in ["debugscan_sharessm"] else 1
        k_group = 4 if forward_type not in ["debugscan_sharessm"] else 1

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = nn.Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
            del self.dt_projs

            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanOflex,
                       cross_selective_scan=cross_selective_scan, force_fp32=None):
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        # ZSJ V2版本使用的mamba，要改扫描方向在这里改
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, delta_softplus=True,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            force_fp32=force_fp32,
            SelectiveScan=SelectiveScan,
        )
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        # print (x.shape)
        with_dconv = (self.d_conv > 1)
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=-1)  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if with_dconv:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x, channel_first=with_dconv)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class OSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        _OSSM = OSSM

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = _OSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channels_first=False)

    def _forward(self, input: torch.Tensor):
        if self.ssm_branch:
            if self.post_norm:
                x = input + self.drop_path(self.norm(self.op(input)))
            else:
                x = input + self.drop_path(self.op(self.norm(input)))

        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN

        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


###Upernet_decoder
class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear',
                                                align_corners=True)
            out_puts.append(ppm_out)
        return out_puts


class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], num_classes=8):
        super(PPMHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes) * self.out_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out


class FPNHEAD(nn.Module):
    def __init__(self, channels=512, out_channels=64):
        super(FPNHEAD, self).__init__()
        self.PPMHead = PPMHEAD(in_channels=channels, out_channels=out_channels)

        self.Conv_fuse1 = nn.Sequential(
            nn.Conv2d(channels // 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv2d(channels // 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.Conv_fuse3 = nn.Sequential(
            nn.Conv2d(channels // 8, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.fuse_all = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv_x1 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, input_fpn):
        # b, 512, 7, 7
        x1 = self.PPMHead(input_fpn[-1])

        x = nn.functional.interpolate(x1, size=(x1.size(2) * 2, x1.size(3) * 2), mode='bilinear', align_corners=True)
        x = self.conv_x1(x) + self.Conv_fuse1(input_fpn[-2])
        x2 = self.Conv_fuse1_(x)

        x = nn.functional.interpolate(x2, size=(x2.size(2) * 2, x2.size(3) * 2), mode='bilinear', align_corners=True)
        x = x + self.Conv_fuse2(input_fpn[-3])
        x3 = self.Conv_fuse2_(x)

        x = nn.functional.interpolate(x3, size=(x3.size(2) * 2, x3.size(3) * 2), mode='bilinear', align_corners=True)
        x = x + self.Conv_fuse3(input_fpn[-4])
        x4 = self.Conv_fuse3_(x)

        x1 = F.interpolate(x1, x4.size()[-2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, x4.size()[-2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, x4.size()[-2:], mode='bilinear', align_corners=True)

        x = self.fuse_all(torch.cat([x1, x2, x3, x4], 1))

        return x


###Upernet_decoder

class RSM_SS(nn.Module):
    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=8,
            depths=[2, 2, 9, 2],
            # dims=[64, 128, 256, 512],
            dims=[96, 192, 384, 768],
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",
            use_checkpoint=False,
            **kwargs,
    ):
        super().__init__()
        ###transformer
        ###transformer
        ###gate
        ###gate
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims

        ###Upernet
        self.decoder = FPNHEAD()
        self.seg = nn.Sequential(
            nn.Conv2d(96, self.num_classes, kernel_size=3, padding=1),
        )
        ###Upernet

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        if isinstance(norm_layer, str) and norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if isinstance(ssm_act_layer, str) and ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]

        if isinstance(mlp_act_layer, str) and mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]

        _make_patch_embed = self._make_patch_embed_v2
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer)

        _make_downsample = self._make_downsample_v3

        # self.encoder_layers = [nn.ModuleList()] * self.num_layers
        self.encoder_layers = []

        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer - 1],
                self.dims[i_layer],
                norm_layer=norm_layer,
            ) if (i_layer != 0) else nn.Identity()  # ZSJ i_layer != 0，也就是第一层不下采样，和论文的图保持一致，也方便我取出每个尺度处理好的特征

            self.encoder_layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            ))

        self.encoder_block1, self.encoder_block2, self.encoder_block3, self.encoder_block4 = self.encoder_layers
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        assert patch_size == 4
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            (Permute(0, 2, 3, 1) if patch_norm else nn.Identity()),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (Permute(0, 3, 1, 2) if patch_norm else nn.Identity()),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(OSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))

        return nn.Sequential(OrderedDict(
            # ZSJ 把downsample放到前面来，方便我取出encoder中每个尺度处理好的图像，而不是刚刚下采样完的图像
            downsample=downsample,
            blocks=nn.Sequential(*blocks, ),
        ))

    def forward(self, x1: torch.Tensor):
        x1 = self.patch_embed(x1)

        x1_1 = self.encoder_block1(x1)
        x1_2 = self.encoder_block2(x1_1)
        x1_3 = self.encoder_block3(x1_2)
        x1_4 = self.encoder_block4(x1_3)  # b,h,w,c

        e1_mamba = rearrange(x1_1, "b h w c -> b c h w").contiguous()
        e2_mamba = rearrange(x1_2, "b h w c -> b c h w").contiguous()
        e3_mamba = rearrange(x1_3, "b h w c -> b c h w").contiguous()
        e4_mamba = rearrange(x1_4, "b h w c -> b c h w").contiguous()

        return [e1_mamba, e2_mamba, e3_mamba, e4_mamba]

###xiaobo
class HaarWaveletLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 用于融合小波系数的1x1卷积
        self.fusion_conv = nn.Conv2d(
            in_channels * 4,  # 4个子带(LL, LH, HL, HH)
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, x):
        batch_size = x.shape[0]
        channels = x.shape[1]

        # 对每个通道执行小波变换
        coeffs_list = []
        for c in range(channels):
            single_channel = x[:, c:c + 1, :, :]

            # 执行二维哈尔小波变换
            LL, (LH, HL, HH) = pywt.dwt2(single_channel.cpu().numpy(), 'haar')

            # 转换为Tensor并拼接
            coeffs = torch.cat([
                torch.from_numpy(LL).to(x.device),
                torch.from_numpy(LH).to(x.device),
                torch.from_numpy(HL).to(x.device),
                torch.from_numpy(HH).to(x.device)
            ], dim=1)  # [batch, 4, h/2, w/2]

            coeffs_list.append(coeffs)

        # 合并所有通道 [batch, in_ch*4, h/2, w/2]
        combined = torch.cat(coeffs_list, dim=1)

        # 通道融合 [batch, out_channels, h/2, w/2]
        out = self.fusion_conv(combined)
        return out


###xiaobo
###resnet34
class BasicBlock(nn.Module):
    expansion = 1  # 扩展系数，保持通道数不变

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 下采样连接
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = self.relu(out)

        return out


# 定义完整的ResNet-34
class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 初始卷积层
        # self.conv1 = nn.Conv2d(
        #     12,
        #     64,
        #     kernel_size=7,
        #     stride=2,
        #     padding=3,
        #     bias=False
        # )

        # 替换前部卷积层
        self.haar_layer = HaarWaveletLayer(in_channels=3, out_channels=64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(12)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个残差阶段
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # # 分类头
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                downsample
            )
        )
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始层

        ###小波
        x = self.haar_layer(x)

        ###小波
        # x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.maxpool(x)

        # 残差阶段
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # 分类头
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return [x1, x2, x3, x4]


###resnet34

###MDAF
import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class LightweightCrossAttentionFusion(nn.Module):
    def __init__(self, dim_vssm, dim_cnn, fuse_dim, num_heads=4, downsample_ratio=4):
        """
        轻量化交叉注意力融合 (Lightweight Cross-Attention Fusion, LCAF)

        Args:
            dim_cnn (int): CNN分支的输入通道数。
            dim_vssm (int): VSSM分支的输入通道数。
            fuse_dim (int): 模块内部及输出的特征通道数。
            num_heads (int): 注意力头的数量。
            downsample_ratio (int): 对K和V进行降采样的比率。
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = fuse_dim // num_heads
        inner_dim = self.num_heads * self.dim_head
        self.scale = self.dim_head ** -0.5

        # 1. 通道对齐层 (1x1 卷积)
        self.proj_cnn = nn.Conv2d(dim_cnn, fuse_dim, 1)
        self.proj_vssm = nn.Conv2d(dim_vssm, fuse_dim, 1)

        # 2. 轻量化的K, V降采样层
        # 使用带步长的卷积实现降采样，这比池化更灵活，是可学习的
        self.downsample_cnn = nn.Sequential(
            nn.Conv2d(fuse_dim, fuse_dim, kernel_size=downsample_ratio, stride=downsample_ratio),
            nn.BatchNorm2d(fuse_dim)
        )
        self.downsample_vssm = nn.Sequential(
            nn.Conv2d(fuse_dim, fuse_dim, kernel_size=downsample_ratio, stride=downsample_ratio),
            nn.BatchNorm2d(fuse_dim)
        )

        # 3. Q, K, V 生成层 (nn.Linear)
        self.to_q_cnn = nn.Linear(fuse_dim, inner_dim, bias=False)
        self.to_kv_cnn = nn.Linear(fuse_dim, inner_dim * 2, bias=False)

        self.to_q_vssm = nn.Linear(fuse_dim, inner_dim, bias=False)
        self.to_kv_vssm = nn.Linear(fuse_dim, inner_dim * 2, bias=False)

        # 4. 输出投影层
        self.to_out_cnn = nn.Linear(inner_dim, fuse_dim)
        self.to_out_vssm = nn.Linear(inner_dim, fuse_dim)

        # 5. 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Conv2d(fuse_dim * 2, fuse_dim, 1, bias=False),
            nn.BatchNorm2d(fuse_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_vssm, x_cnn):
        # 确保空间尺寸一致
        if x_cnn.shape[2:] != x_vssm.shape[2:]:
            x_vssm = F.interpolate(x_vssm, size=x_cnn.shape[2:], mode='bilinear', align_corners=False)

        b, _, h, w = x_cnn.shape

        # --- 通道对齐 ---
        cnn_p = self.proj_cnn(x_cnn)
        vssm_p = self.proj_vssm(x_vssm)

        # --- 路径1: 用VSSM的低分辨率上下文，来增强CNN的高分辨率细节 ---
        q_cnn = rearrange(cnn_p, 'b c h w -> b (h w) c')
        q_cnn = self.to_q_cnn(q_cnn)

        vssm_down = self.downsample_vssm(vssm_p)
        vssm_down_flat = rearrange(vssm_down, 'b c h w -> b (h w) c')
        k_vssm, v_vssm = self.to_kv_vssm(vssm_down_flat).chunk(2, dim=-1)

        q_cnn, k_vssm, v_vssm = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            (q_cnn, k_vssm, v_vssm)
        )

        attn_out_cnn = torch.einsum('b h i d, b h j d -> b h i j', q_cnn, k_vssm) * self.scale
        attn_out_cnn = attn_out_cnn.softmax(dim=-1)
        out1 = torch.einsum('b h i j, b h j d -> b h i d', attn_out_cnn, v_vssm)

        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        enhanced_cnn = cnn_p + rearrange(self.to_out_cnn(out1), 'b (h w) c -> b c h w', h=h, w=w)

        # --- 路径2: 用CNN的低分辨率上下文，来增强VSSM的高分辨率细节 ---
        q_vssm = rearrange(vssm_p, 'b c h w -> b (h w) c')
        q_vssm = self.to_q_vssm(q_vssm)

        cnn_down = self.downsample_cnn(cnn_p)
        cnn_down_flat = rearrange(cnn_down, 'b c h w -> b (h w) c')
        k_cnn, v_cnn = self.to_kv_cnn(cnn_down_flat).chunk(2, dim=-1)

        q_vssm, k_cnn, v_cnn = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            (q_vssm, k_cnn, v_cnn)
        )

        attn_out_vssm = torch.einsum('b h i d, b h j d -> b h i j', q_vssm, k_cnn) * self.scale
        attn_out_vssm = attn_out_vssm.softmax(dim=-1)
        out2 = torch.einsum('b h i j, b h j d -> b h i d', attn_out_vssm, v_cnn)

        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        enhanced_vssm = vssm_p + rearrange(self.to_out_vssm(out2), 'b (h w) c -> b c h w', h=h, w=w)

        # --- 最终融合 ---
        output = self.final_fusion(torch.cat([enhanced_cnn, enhanced_vssm], dim=1))

        return output

class AxialCrossAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads=8, dim_head=32):
        """
        改良版的DAF，采用清晰的双向轴向交叉注意力。
        dim1: x1的通道数 (e.g., from VMamba)
        dim2: x2的通道数 (e.g., from ResNet)
        """
        super().__init__()
        self.num_heads = num_heads
        inner_dim = dim_head * num_heads
        self.scale = dim_head ** -0.5

        # --- 路径1: 用 x2 的信息来增强 x1 ---
        self.norm1_for_q = nn.LayerNorm(dim1)
        self.norm2_for_kv = nn.LayerNorm(dim2)
        self.to_q1 = nn.Linear(dim1, inner_dim, bias=False)
        self.to_kv2 = nn.Linear(dim2, inner_dim * 2, bias=False)
        self.to_out1 = nn.Linear(inner_dim, dim2)  # 输出维度回到dim2

        # --- 路径2: 用 x1 的信息来增强 x2 ---
        self.norm2_for_q = nn.LayerNorm(dim2)
        self.norm1_for_kv = nn.LayerNorm(dim1)
        self.to_q2 = nn.Linear(dim2, inner_dim, bias=False)
        self.to_kv1 = nn.Linear(dim1, inner_dim * 2, bias=False)
        self.to_out2 = nn.Linear(inner_dim, dim2)  # 输出维度回到dim2
        # 2. 输出投影层，直接将attention结果投射到target_dim
        self.shortcut = nn.Sequential(
            nn.Conv2d(dim1, dim2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim2)
        )

    def forward(self, x1, x2):
        # x1: (B, C1, H, W), C1=dim1, from VMamba
        # x2: (B, C2, H, W), C2=dim2, from ResNet
        b, c1, h, w = x1.shape
        _, c2, _, _ = x2.shape

        # === 路径1: 计算 y1 (用x2增强x1) ===
        x1_flat = rearrange(x1, 'b c h w -> b (h w) c')
        x2_flat = rearrange(x2, 'b c h w -> b (h w) c')

        q1 = self.to_q1(self.norm1_for_q(x1_flat))
        k2, v2 = self.to_kv2(self.norm2_for_kv(x2_flat)).chunk(2, dim=-1)

        # 高度轴注意力
        q1_h = rearrange(q1, 'b (h w) (head d) -> (b w) head h d', h=h, w=w, head=self.num_heads)
        k2_h = rearrange(k2, 'b (h w) (head d) -> (b w) head h d', h=h, w=w, head=self.num_heads)
        v2_h = rearrange(v2, 'b (h w) (head d) -> (b w) head h d', h=h, w=w, head=self.num_heads)

        dots_h = torch.einsum('b h i d, b h j d -> b h i j', q1_h, k2_h) * self.scale
        attn_h = (dots_h).softmax(dim=-1)
        out_h = torch.einsum('b h i j, b h j d -> b h i d', attn_h, v2_h)
        out_h = rearrange(out_h, '(b w) head h d -> b (h w) (head d)', h=h, w=w)

        # 宽度轴注意力
        q1_w = rearrange(q1, 'b (h w) (head d) -> (b h) head w d', h=h, w=w, head=self.num_heads)
        k2_w = rearrange(k2, 'b (h w) (head d) -> (b h) head w d', h=h, w=w, head=self.num_heads)
        v2_w = rearrange(v2, 'b (h w) (head d) -> (b h) head w d', h=h, w=w, head=self.num_heads)

        dots_w = torch.einsum('b h i d, b h j d -> b h i j', q1_w, k2_w) * self.scale
        attn_w = (dots_w).softmax(dim=-1)
        out_w = torch.einsum('b h i j, b h j d -> b h i d', attn_w, v2_w)
        out_w = rearrange(out_w, '(b h) head w d -> b (h w) (head d)', h=h, w=w)

        # 融合轴向结果并通过干净的残差连接
        fused_info1 = self.to_out1(out_h + out_w)
        fused_info1 = rearrange(fused_info1, 'b (h w) c -> b c h w', h=h, w=w)
        y1 = self.shortcut(x1) + fused_info1  # 增强后的x1

        # === 路径2: 计算 y2 (用x1增强x2) ===
        q2 = self.to_q2(self.norm2_for_q(x2_flat))
        k1, v1 = self.to_kv1(self.norm1_for_kv(x1_flat)).chunk(2, dim=-1)

        # 高度轴注意力 (Path 2)
        q2_h = rearrange(q2, 'b (h w) (head d) -> (b w) head h d', h=h, w=w, head=self.num_heads)
        k1_h = rearrange(k1, 'b (h w) (head d) -> (b w) head h d', h=h, w=w, head=self.num_heads)
        v1_h = rearrange(v1, 'b (h w) (head d) -> (b w) head h d', h=h, w=w, head=self.num_heads)

        dots_h2 = torch.einsum('b h i d, b h j d -> b h i j', q2_h, k1_h) * self.scale
        attn_h2 = (dots_h2).softmax(dim=-1)
        out_h2 = torch.einsum('b h i j, b h j d -> b h i d', attn_h2, v1_h)
        out_h2 = rearrange(out_h2, '(b w) head h d -> b (h w) (head d)', h=h, w=w)

        # 宽度轴注意力 (Path 2)
        q2_w = rearrange(q2, 'b (h w) (head d) -> (b h) head w d', h=h, w=w, head=self.num_heads)
        k1_w = rearrange(k1, 'b (h w) (head d) -> (b h) head w d', h=h, w=w, head=self.num_heads)
        v1_w = rearrange(v1, 'b (h w) (head d) -> (b h) head w d', h=h, w=w, head=self.num_heads)
        dots_w2 = torch.einsum('b h i d, b h j d -> b h i j', q2_w, k1_w) * self.scale
        attn_w2 = (dots_w2).softmax(dim=-1)
        out_w2 = torch.einsum('b h i j, b h j d -> b h i d', attn_w2, v1_w)
        out_w2 = rearrange(out_w2, '(b h) head w d -> b (h w) (head d)', h=h, w=w)

        fused_info2 = self.to_out2(out_h2 + out_w2)
        fused_info2 = rearrange(fused_info2, 'b (h w) c -> b c h w', h=h, w=w)
        y2 = x2 + fused_info2  # 增强后的x2，维度为dim2

        return y1 + y2

class FUSE(nn.Module):
    def __init__(self, res_dim, vma_dim):
        super().__init__()
        self.res_dim = res_dim
        self.vma_dim = vma_dim

        self.conv1 = nn.Conv2d(res_dim + vma_dim, res_dim, kernel_size=1)


    def forward(self, res_feat, vma_feat):
        # 通道融合与校准（保持原逻辑）
        fused = torch.cat([res_feat, vma_feat], dim=1)
        fused = self.conv1(fused)
        return fused



###MADF

class DualEncoderHIF(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 子模块定义 --------------------------------
        # ResNet34编码器
        self.resnet = load_backbone_weights(ResNet(block=BasicBlock, layers=[2, 2, 2, 2]), "D:\GIS\Pretrain\ResNet18\DRmamba_epoch_25_ResNet18.pth")

        # Vmamba编码器（自定义初始化）
        self.vmamba = load_backbone_weights(RSM_SS(), "D:\GIS\Pretrain\ResNet18\DRmamba_epoch_4_mamba.pth")
        # self.vmamba = RSM_SS()

        self.MDAF1 = AxialCrossAttention(96, 64)
        self.MDAF2 = AxialCrossAttention(192, 128)
        self.MDAF3 = AxialCrossAttention(384, 256)
        self.MDAF4 = AxialCrossAttention(768, 512)

        self.decoder = FPNHEAD()
        self.seg = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # 双路特征提取
        res_features = self.resnet(x)  # List[Tensor]
        vma_features = self.vmamba(x)  # List[Tensor]


        h1 = self.MDAF1(vma_features[0],res_features[0])
        h2 = self.MDAF2(vma_features[1],res_features[1])
        h3 = self.MDAF3(vma_features[2],res_features[2])
        h4 = self.MDAF4(vma_features[3],res_features[3])

        fused_features = [h1, h2, h3, h4]

        x = self.decoder(fused_features)
        x = nn.functional.interpolate(x, size=(x.size(2) * 4, x.size(3) * 4), mode='bilinear', align_corners=True)
        x = self.seg(x)
        # 层次化特征融合

        # 解码器输出
        return x



def load_backbone_weights(model_backbone, checkpoint_path):
    """
    加载一个完整的预训练模型（含分类头）的权重到
    一个只有主干网络（backbone）的模型中。

    Args:
        model_backbone (nn.Module): 目标模型，只包含主干网络结构。
        checkpoint_path (str): 您自己预训练好的完整模型的权重文件路径。

    Returns:
        nn.Module: 加载了权重的 model_backbone。
    """
    print(f"Loading custom pretrained weights from: {checkpoint_path}")

    # 1. 加载您完整的预训练模型检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 2. 提取 state_dict。根据您的保存习惯，可能需要从 'model' 或 'state_dict' 键中提取
    if 'model' in checkpoint:
        pretrained_state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        pretrained_state_dict = checkpoint['state_dict']
    else:
        pretrained_state_dict = checkpoint

    # 3. 使用 strict=False 加载权重
    # 这会自动加载所有名称匹配的层（conv1, layer1...），并忽略不匹配的层（如 fc）
    msg = model_backbone.load_state_dict(pretrained_state_dict, strict=False)

    # 4. 打印加载信息，这对于调试非常重要
    print("\n--- Weight Loading Summary ---")
    print("Missing keys (should be empty if your backbone is a subset):", msg.missing_keys)
    print("Unexpected keys (should list your classifier's weights, e.g., 'fc.weight'):", msg.unexpected_keys)
    print("----------------------------\n")

    if not msg.missing_keys and msg.unexpected_keys:
        print("Pretrained backbone weights loaded successfully!")
    else:
        print("Warning: Check the loading summary. There might be unexpected mismatches.")

    return model_backbone


if __name__ == '__main__':
    from thop import profile
    x = torch.randn(1, 3, 512, 512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确保输入数据在GPU上
    x = x.to(device)  # 假设 x 是输入张量
    net = DualEncoderHIF(8).to(device)
    out = net(x)
    print(net)
    print(out.shape)
    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)
