"""The module.
"""

import math
from typing import Any, Callable, List

import needle.init as init
import numpy as np
from needle import ops
from needle.autograd import Tensor

from .nn_basic import Module, Parameter


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        k = self.kernel_size
        i = self.in_channels
        o = self.out_channels

        self.weight = Parameter(
            init.kaiming_uniform(
                i * (k**2), o * (k**2), shape=(k, k, i, o), device=device
            )
        )
        if bias:
            bound = 1 / (math.sqrt(i * (k**2)))
            self.bias = Parameter(
                init.rand(
                    o,
                    low=-1 * bound,
                    high=bound,
                    device=device,
                )
            )
        else:
            self.bias = init.zeros(o, device=device)

    def forward(self, x: Tensor) -> Tensor:
        K = self.kernel_size

        # NCHW -> NHWC
        x = ops.transpose(
            ops.transpose(x, axes=(1, 2)),
            axes=(2, 3),
        )

        # H_out = ((H+2P-K)//self.stride) + 1
        # W_out = ((W+2P-K)//self.stride) + 1

        out = ops.conv(x, self.weight, stride=self.stride, padding=(K - 1) // 2)
        bias = ops.broadcast_to(
            ops.reshape(self.bias, (1, 1, 1, self.out_channels)),
            out.shape,
        )

        if bias is not None:
            out += bias

        # NHWC -> NCHW
        return ops.transpose(
            ops.transpose(out, axes=(2, 3)),
            axes=(1, 2),
        )


class Conv1d(Module):
    """
    Multi-channel 1D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(
                in_channels * kernel_size, out_channels * kernel_size, shape=(kernel_size, in_channels, out_channels), device=device
            )
        )
        if bias:
            bound = 1 / (math.sqrt(in_channels * kernel_size))
            self.bias = Parameter(
                init.rand(
                    out_channels,
                    low=-1 * bound,
                    high=bound,
                    device=device,
                )
            )
        else:
            self.bias = init.zeros(out_channels, device=device)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        K = self.kernel_size

        # NCH -> NHC
        x = ops.transpose(x, axes=(1, 2))

        # H_out = ((H+2P-K)//self.stride) + 1
        # W_out = ((W+2P-K)//self.stride) + 1

        out = ops.conv1d(x, self.weight, stride=self.stride, padding=(K - 1) // 2)
        bias = ops.broadcast_to(
            ops.reshape(self.bias, (1, 1, self.out_channels)),
            out.shape,
        )

        if bias is not None:
            out += bias

        # NHC -> NCH
        return ops.transpose(out, axes=(1, 2))

        ### END YOURSOLUTION
