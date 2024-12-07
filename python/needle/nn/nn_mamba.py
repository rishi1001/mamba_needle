import math
from dataclasses import dataclass
from typing import List, Union

import needle.backend_ndarray.ndarray as ndarray
import needle.init as init
import numpy as np

# from mambapy.pscan import pscan
from needle import ops
from needle.autograd import Tensor
from needle.ops import PScan

from .nn_basic import Dropout, LayerNorm1d, Linear, Module, Parameter, ReLU, Sequential
from .nn_conv import Conv1d
from .nn_sequence import Embedding

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison. Also, it is possible to use the official Mamba implementation.

This is the structure of the torch modules :
- A Mamba model is composed of several layers, which are ResidualBlockMamba.
- A ResidualBlockMamba is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlockMamba(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""


@dataclass
class MambaConfig:
    d_model: int  # D
    n_layers: int
    dt_rank: Union[int, str] = "auto"
    d_state: int = 16  # N in paper/comments
    expand_factor: int = 2  # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False  # apply layernorms to internal activations

    mup: bool = False
    mup_base_width: float = 128  # width=d_model

    pscan: bool = True  # use parallel scan mode or sequential mode when training
    use_cuda: bool = (
        False  # use official CUDA implementation when training (not compatible with (b)float16)
    )

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width


class Mamba(Module):
    def __init__(self, config: MambaConfig, device=None):
        super().__init__()

        self.config = config

        # self.layers = nn.ModuleList(
        #     [ResidualBlockMamba(config) for _ in range(config.n_layers)]
        # )
        # TODO verify if nn.ModuleList is needed, ie are the weights changing?
        self.layers = []
        for _ in range(config.n_layers):
            self.layers.append(ResidualBlockMamba(config, device=device))

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        return x

    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class ResidualBlockMamba(Module):
    def __init__(self, config: MambaConfig, device=None):
        super().__init__()

        self.mixer = MambaBlock(config, device=device)
        self.norm = RMSNorm(
            config.d_model, config.rms_norm_eps, config.mup, device=device
        )

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)
        output = self.mixer(self.norm(x)) + x
        return output

    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
        # h : (B, ED, N)
        # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class MambaBlock(Module):
    def __init__(self, config: MambaConfig, device=None):
        super().__init__()

        self.config = config
        self.device = device

        # projects block input from D to 2*ED (two branches)
        self.in_proj = Linear(
            config.d_model, 2 * config.d_inner, bias=config.bias, device=device
        )

        self.conv1d = Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
            device=device,
        )

        # projects x to input-dependent delta, B, C
        self.x_proj = Linear(
            config.d_inner,
            config.dt_rank + 2 * config.d_state,
            bias=False,
            device=device,
        )

        # projects delta from dt_rank to d_inner
        self.dt_proj = Linear(config.dt_rank, config.d_inner, bias=True, device=device)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        # TODO verify it
        if config.dt_init == "constant":
            self.dt_proj.weight = init.constant(
                *self.dt_proj.weight.shape,
                c=dt_init_std,
                device=device,
                dtype=self.dt_proj.weight.dtype,
                requires_grad=self.dt_proj.weight.requires_grad,
            )
        elif config.dt_init == "random":
            self.dt_proj.weight = init.rand(
                *self.dt_proj.weight.shape,
                low=-dt_init_std,
                high=dt_init_std,
                device=device,
                dtype=self.dt_proj.weight.dtype,
                requires_grad=self.dt_proj.weight.requires_grad,
            )
        else:
            raise NotImplementedError

        # delta bias
        dt = ops.exp(
            init.rand(config.d_inner, device=device)
            * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        )
        # .clamp(min=config.dt_init_floor)
        dt = ops.clamp(dt, minimum=config.dt_init_floor)
        inv_dt = dt + ops.log(
            -ops.exp(-dt) + 1
        )  # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759

        # TODO weird hack to avoid reinitialization, check if needed
        # with torch.no_grad():
        #     self.dt_proj.bias.copy_(inv_dt)
        original_requires_grad = self.dt_proj.bias.requires_grad
        self.dt_proj.bias.requires_grad = False
        self.dt_proj.bias.data = inv_dt  # or however you update the underlying data
        self.dt_proj.bias.requires_grad = original_requires_grad
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        # A = torch.arange(1, config.d_state + 1, dtype="float32").repeat(
        #     config.d_inner, 1
        # )
        # TODO verify arange and repeat
        seq = Tensor(
            ndarray.NDArray(list(range(1, config.d_state + 1))),
            device=device,
            dtype=self.dt_proj.bias.dtype,
        )

        # Repeat the sequence along the first dimension (similar to torch.repeat)
        A = seq.reshape((1, config.d_state)).broadcast_to(
            (config.d_inner, config.d_state)
        )
        self.A_log = Parameter(
            ops.log(A)
        )  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = Parameter(init.ones(config.d_inner, device=device))
        self.D._no_weight_decay = True

        # projects block output from ED back to D
        self.out_proj = Linear(
            config.d_inner, config.d_model, bias=config.bias, device=device
        )

        # used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(
                self.config.dt_rank, config.rms_norm_eps, config.mup, device=device
            )
            self.B_layernorm = RMSNorm(
                self.config.d_state, config.rms_norm_eps, config.mup, device=device
            )
            self.C_layernorm = RMSNorm(
                self.config.d_state, config.rms_norm_eps, config.mup, device=device
            )
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

        self.silu = SiLu()
        self.softplus = Softplus()

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        _, L, _ = x.shape

        x = x.reshape((-1, self.config.d_model))  # (B*L, D)
        xz = self.in_proj(x)  # (B, L, 2*ED)
        xz = xz.reshape((-1, L, 2 * self.config.d_inner))  # (B, L, 2*ED)
        x, z = chunk(xz, 2, dim=-1)  # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose((1, 2))  # (B, ED, L)
        x = self.conv1d(x)[
            :, :, :L
        ]  # depthwise convolution over time, with a short filter
        x = x.transpose((1, 2))  # (B, L, ED)

        x = self.silu(x)
        y = self.ssm(x, z)

        if self.config.use_cuda:
            output = self.out_proj(y)  # (B, L, D)
            return output  # the rest of the operations are done in the ssm function (fused with the CUDA pscan)

        # z branch
        z = self.silu(z)

        output = y * z
        output = self.out_proj(output.reshape((-1, self.config.d_inner))).reshape(
            (-1, L, self.config.d_model)
        )  # (B, L, D)

        return output

    def ssm(self, x, z):
        # x : (B, L, ED)

        # y : (B, L, ED)

        (B_, L, ED) = x.shape

        A = -ops.exp(self.A_log)  # (ED, N)
        D = self.D

        x = x.reshape((-1, self.config.d_inner))  # (B*L, ED)
        deltaBC = self.x_proj(x)  # (B, L, dt_rank+2*N)
        deltaBC = deltaBC.reshape((B_, L, -1))
        x = x.reshape((B_, L, -1))

        # Get the sizes to split
        sizes = [self.config.dt_rank, self.config.d_state, self.config.d_state]

        # Compute cumulative indices for slicing manually
        cumsum_sizes = [0]
        for size in sizes:
            cumsum_sizes.append(cumsum_sizes[-1] + size)

        # Slice the tensor manually
        delta = deltaBC[:, :, cumsum_sizes[0] : cumsum_sizes[1]]  # (B, L, dt_rank)
        B = deltaBC[:, :, cumsum_sizes[1] : cumsum_sizes[2]]  # (B, L, N)
        C = deltaBC[:, :, cumsum_sizes[2] : cumsum_sizes[3]]  # (B, L, N)

        # Result: delta, B, C
        delta, B, C = self._apply_layernorms(delta, B, C)
        # delta = self.dt_proj.weight @ delta.transpose((1, 2))  # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)

        # Assuming delta has shape (B, L, dt_rank) and self.dt_proj.weight has shape (dt_rank, ED)
        B_, L, dt_rank = delta.shape
        _, ED = self.dt_proj.weight.shape

        # Initialize the output tensor with zeros
        delta_result = init.zeros(B_, ED, L, device=delta.device)
        delta_result = []

        # Perform batch-wise matrix multiplication
        for b in range(B_):
            # Multiply weight (dt_rank, ED) with delta[b].T (dt_rank, L)
            delta_result.append(
                self.dt_proj.weight.transpose((0, 1)) @ delta[b].transpose((1, 0))
            )  # (ED, dt_rank) @ (dt_rank, L))
            # delta_result[b] = self.dt_proj.weight.transpose((0, 1)) @ delta[b].transpose((1, 0))  # (ED, dt_rank) @ (dt_rank, L)

        delta = ops.stack(delta_result, axis=0)  # (B, ED, L)
        # Assign the result back to delta
        # delta = delta_result

        delta = delta.transpose((1, 2))
        delta = self.softplus(
            delta + self.dt_proj.bias.reshape((1, 1, -1)).broadcast_to(delta.shape)
        )

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)


        return y

    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)
        (B_, L, ED) = x.shape
        N = A.shape[1]
        # deltaA = ops.exp(ops.unsqueeze(delta, -1) * A.unsqueeze(0).unsqueeze(1).broadcast_to(delta.shape))
        deltaA = ops.exp(
            ops.unsqueeze(delta, -1).broadcast_to((B_, L, ED, N))
            * ops.unsqueeze(ops.unsqueeze(A, 0), 1).broadcast_to((B_, L, ED, N))
        )  # (B, L, ED, N)
        # deltaB = ops.unsqueeze(delta, -1) * B.unsqueeze(2)  # (B, L, ED, N)
        deltaB = ops.unsqueeze(delta, -1).broadcast_to((B_, L, ED, N)) * ops.unsqueeze(
            B, 2
        ).broadcast_to((B_, L, ED, N))

        BX = deltaB * ops.unsqueeze(x, -1).broadcast_to((B_, L, ED, N))  # (B, L, ED, N)
        # hs = ops.pscan(deltaA, BX, use_cuda=self.config.use_cuda)  # (B, L, ED, N)
        hs = ops.pscan(deltaA, BX, use_cuda=True)  # (B, L, ED, N)

        # y = (hs @ C.unsqueeze(-1)).squeeze(
        #     3
        # )  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)
        y = ops.squeeze(
            self.batched_matmul(hs, ops.unsqueeze(C, -1).transpose()), 3
        )  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D.reshape((1, 1, -1)).broadcast_to(x.shape) * x

        return y


    def selective_scan_seq(self, x, dt, A, B, C, D):
        """
        Forward pass through the selective scan
        Args:
        x: Input tensor of shape (B, L, E*D)
        dt: dt tensor of shape (B, L, E*D)
        A: A tensor of shape (E*D, N)
        B: B tensor of shape (B, L, N)
        C: C tensor of shape (B, L, N)
        D: D tensor of shape (E*D)
        Returns:
        Output tensor of shape (B, L, E*D)
        """
        batch_size, seq_length, d_inner = x.shape
        _, d_state = A.shape
        assert d_inner == self.config.d_inner
        assert d_state == self.config.d_state

        # Calculate dA (B, L, E*D, N)
        dA = ops.exp(ops.broadcast_to(ops.unsqueeze(dt, 3), dt.shape + (d_state,))
                     * ops.broadcast_to(ops.reshape(A, (1, 1) + A.shape), (batch_size, seq_length) + A.shape))
        assert dA.shape == (batch_size, seq_length, d_inner, d_state)

        # Calculate dB (B, L, E*D, N)
        dB = (ops.broadcast_to(ops.unsqueeze(dt, 3), dt.shape + (d_state,))
              * ops.broadcast_to(ops.unsqueeze(B, 2), B.shape[:2] + (d_inner,) + B.shape[2:]))
        assert dB.shape == (batch_size, seq_length, d_inner, d_state)

        # Calculate dB*x (B, L, E*D, N)
        dB_x = dB * ops.broadcast_to(ops.unsqueeze(x, 3), x.shape + (d_state,))
        assert dB_x.shape == (batch_size, seq_length, d_inner, d_state)

        # Initialize h (B, E*D, N)
        h = init.zeros(batch_size, d_inner, d_state,
                       device=self.device, requires_grad=True)
        hs = []

        # Sequential (RNN) approach involves iteration over length of sequence
        dA_split = ops.split(dA, 1)
        dB_x_split = ops.split(dB_x, 1)
        for t in range(seq_length):
            # SSM equation 1a/2a (Section 2 of paper)
            h = dA_split[t] * h + dB_x_split[t]
            hs.append(h)

        # Combine into (B, L, E*D, N) Tensor
        hs = ops.stack(hs, 1)
        assert hs.shape == (batch_size, seq_length, d_inner, d_state)

        # SSM equation 1b/2b (Section 2 of paper)
        # y = (hs @ C.unsqueeze(-1)).squeeze(3)
        hs_split = [ops.split(s, 0) for s in ops.split(hs, 0)]
        C_split = [ops.split(s, 0) for s in ops.split(ops.unsqueeze(C, 3), 0)]
        y_stack = [ops.stack([hs_split[i][j] @ C_split[i][j] for j in range(seq_length)], 0) for i in range(batch_size)]
        y = ops.stack(y_stack, 0)
        y = ops.reshape(y, (batch_size, seq_length, d_inner))
        assert y.shape == (batch_size, seq_length, d_inner)

        # Skip connection using D
        y = y + ops.broadcast_to(ops.reshape(D, (1, 1, d_inner)), y.shape) * x

        return y

    def batched_matmul(self, a: Tensor, b_transpose: Tensor) -> Tensor:
        """
        batched matrix multiplication;
        """
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)

        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_transpose_shape[-2]
        a = a.broadcast_to(broadcast_shape)

        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)

        return (a * b_transpose).sum(len(a.shape) - 1)


    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """

    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
        # h : (B, ED, N)
        # inputs : (B, ED, d_conv-1)

        # y : (B, D)
        # cache : (h, inputs)

        h, inputs = cache

        xz = self.in_proj(x)  # (B, 2*ED)
        # x, z = xz.chunk(2, dim=1)  # (B, ED), (B, ED)
        x, z = chunk(xz, 2, dim=1)  # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(ops.concat([inputs, x_cache], dim=2))[
            :, :, self.config.d_conv - 1
        ]  # (B, ED)

        x = self.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = self.silu(z)

        output = y * z
        output = self.out_proj(output)  # (B, D)

        # prepare cache for next call
        inputs = ops.concat([inputs[:, :, 1:], x_cache], dim=2)  # (B, ED, d_conv-1)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -ops.exp(
            self.A_log.float()
        )  # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, dt_rank+2*N)

        # TODO check split
        delta, B, C = ops.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )  # (B, dt_rank), (B, N), (B, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.softplus(self.dt_proj(delta))  # (B, ED)

        deltaA = ops.exp(delta.unsqueeze(-1) * A)  # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, ED, N)

        if h is None:
            h = init.zeros(
                x.size(0),
                self.config.d_inner,
                self.config.d_state,
                device=deltaA.device,
            )  # (B, ED, N)

        h = deltaA * h + BX  # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2)  # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h


class RMSNorm(Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        use_mup: bool = False,
        device=None,
        dtype="float32",
    ):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        # https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = Parameter(init.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        # output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # output = x * ops.power_scalar(
        #     ops.summation(ops.power_scalar(x, 2), axes=(1,)), -0.5
        # )
        tot_dim = len(x.shape) - 1

        # Step 1: Compute squared values
        squared = ops.power_scalar(x, 2)

        # Step 2: Sum along axis and compute mean
        summed = ops.summation(squared, axes=(tot_dim,))
        mean = ops.divide_scalar(summed, x.shape[tot_dim])  # Compute mean

        # Step 3: Add epsilon
        adjusted_mean = ops.add_scalar(mean, self.eps)

        # Step 4: Compute reciprocal square root
        inv_sqrt = ops.power_scalar(adjusted_mean, -0.5)

        # TODO check if need unsqueeze?
        # Step 5: Multiply with input
        # output = x * inv_sqrt.unsqueeze(-1)
        output = x * ops.unsqueeze(inv_sqrt, -1).broadcast_to(x.shape)

        if not self.use_mup:
            return output * self.weight.reshape((1, -1)).broadcast_to(x.shape)
        else:
            return output


class SiLu(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x / (1 + ops.exp(-x))


class Softplus(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.log(1 + ops.exp(x))


# TODO check chunk
def chunk(tensor, num_chunks, dim):
    # Calculate size of each chunk
    chunk_size = tensor.shape[dim] // num_chunks

    # Ensure the dimension is divisible by the number of chunks
    assert tensor.shape[dim] % num_chunks == 0, "Tensor cannot be evenly split"

    # # Use split to divide the tensor
    # chunks = ops.split(tensor, axis=dim, size=chunk_size)

    # Use slicing to create chunks
    chunks = []
    for i in range(num_chunks):
        # Create a slice for each chunk
        chunk_slice = [slice(None)] * len(tensor.shape)
        chunk_slice[dim] = slice(i * chunk_size, (i + 1) * chunk_size)

        # Select the chunk using the slice
        chunks.append(tensor[tuple(chunk_slice)])

    return chunks
