"""Operator implementations."""

import operator
from functools import reduce
from numbers import Number
from typing import List, Optional, Tuple, Union

from ..autograd import NDArray, Op, Tensor, TensorOp, TensorTuple, TensorTupleOp, Value
from ..backend_selection import BACKEND, array_api
from .ops_tuple import *

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return (out_grad * rhs * (lhs ** (rhs - 1)), out_grad * (lhs**rhs) * log(lhs))


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a**self.scalar

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        return (out_grad * self.scalar * (lhs ** (self.scalar - 1)),)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return (
            power_scalar(rhs, -1) * out_grad,
            -1 * lhs * power_scalar(rhs, -2) * out_grad,
        )


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return (out_grad * (1 / self.scalar),)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        dim = len(a.shape)

        dims = [i for i in range(dim)]
        if self.axes is None:
            dims[-1], dims[-2] = dims[-2], dims[-1]
            return a.permute(dims)
        else:
            dims[self.axes[0]], dims[self.axes[1]] = (
                dims[self.axes[1]],
                dims[self.axes[0]],
            )
            return a.permute(dims)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (transpose(out_grad, self.axes),)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        if -1 in self.shape:
            new_shape = list(self.shape)
            new_shape[new_shape.index(-1)] = a.size // -reduce(operator.mul, new_shape)
            return array_api.reshape(a, new_shape)
        return array_api.reshape(a.compact(), self.shape)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (reshape(out_grad, a.shape),)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (reverse_broadcast(out_grad, a.shape),)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = axes

    def compute(self, a):
        # TODO: multiple axis support
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        a = node.inputs[0]

        if self.axes is None:
            return broadcast_to(out_grad, a.shape)

        out_shape = list(out_grad.shape)
        for axis in self.axes:
            out_shape.insert(axis, 1)

        return broadcast_to(reshape(out_grad, out_shape), a.shape)


def summation(a, axes=None):
    return Summation(axes)(a)


def reverse_broadcast(out, a_shape):
    out_shape = list(out.shape)

    a_ind = 0
    out_ind = 0

    diff = len(out_shape) - len(a_shape)
    a_ind -= diff

    axis = 0
    while a_ind < len(a_shape) and out_ind < len(out_shape):
        if a_ind < 0:
            n = out_shape[0]
            out = summation(out, axes=(axis,))
            out_shape.pop(0)

            a_ind += 1
            continue

        if a_shape[a_ind] == out_shape[out_ind]:
            a_ind += 1
            out_ind += 1
            axis += 1
            continue

        n = out_shape[out_ind]
        out_shape[out_ind] = 1
        out = reshape(summation(out, axes=(axis,)), out_shape)
        axis += 1
        a_ind += 1
        out_ind += 1

    return out


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs

        if lhs.shape == rhs.shape:
            return (matmul(out_grad, transpose(rhs)), matmul(transpose(lhs), out_grad))

        l = reduce(operator.mul, lhs.shape[:-2], 1)
        r = reduce(operator.mul, rhs.shape[:-2], 1)

        if l < r:
            return reverse_broadcast(
                matmul(out_grad, transpose(rhs)), lhs.shape
            ), matmul(transpose(lhs), out_grad)
        else:
            return matmul(out_grad, transpose(rhs)), reverse_broadcast(
                matmul(transpose(lhs), out_grad), rhs.shape
            )


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -1 * a

    def gradient(self, out_grad, node):
        return -1 * out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (power_scalar(a, -1) * out_grad,)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return exp(a) * out_grad


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        a = node.inputs[0]

        out = out_grad.realize_cached_data() * (a.realize_cached_data() >= 0)
        return Tensor(out, device=a.device)


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (out_grad * negate((power_scalar(tanh(a), 2) - 1)),)


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        arrays = list(args)
        n = len(arrays)

        if not arrays:
            raise ValueError

        new_shape = list(arrays[0].shape)
        new_shape.insert(self.axis, n)

        out = array_api.empty(new_shape, device=arrays[0].device)

        for i in range(n):
            idxs = [slice(None)] * len(new_shape)
            idxs[self.axis] = i
            out[tuple(idxs)] = arrays[i]
        return out

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return split(out_grad, self.axis)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        shape = A.shape
        n = shape[self.axis]

        new_shape = list(shape)
        new_shape.pop(self.axis)

        result = [array_api.empty(new_shape, device=A.device) for _ in range(n)]
        for i in range(n):
            idxs = [slice(None)] * len(shape)
            idxs[self.axis] = i
            result[i] = A[tuple(idxs)].compact().reshape(new_shape)

        return tuple(result)

    def gradient(self, out_grad, node):
        return stack(out_grad, self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.flip(a, self.axes)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        if self.dilation == 0:
            return a

        dim = len(a.shape)
        new_shape = list(a.shape)

        idxes = [slice(None) for _ in range(dim)]
        for ax in self.axes:
            if ax < 0 or ax >= dim:
                continue

            new_shape[ax] *= self.dilation + 1
            idxes[ax] = slice(0, new_shape[ax], self.dilation + 1)

        out = array_api.full(new_shape, 0, device=a.device)
        out[tuple(idxes)] = a
        return out.compact()

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        if self.dilation == 0:
            return a

        dim = len(a.shape)
        new_shape = list(a.shape)

        idxes = [slice(None) for _ in range(dim)]
        for ax in self.axes:
            if ax < 0 or ax >= dim:
                continue

            idxes[ax] = slice(0, a.shape[ax], self.dilation + 1)
            new_shape[ax] //= self.dilation + 1

        out = array_api.full(new_shape, 0, device=a.device)
        out = a[tuple(idxes)]
        return out

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        if stride is None:
            self.stride = 1
        else:
            self.stride = stride

        if padding is None:
            self.padding = 0
        else:
            self.padding = padding

    def compute(self, A, B):
        if self.padding:
            p = self.padding
            A = A.pad(((0, 0), (p, p), (p, p), (0, 0)))

        N, H, W, C_in = A.shape
        Ns, Hs, Ws, C_ins = A.strides
        K, _, _, C_out = B.shape

        H_out = ((H - K) // self.stride) + 1
        W_out = ((W - K) // self.stride) + 1

        Z = NDArray.make(
            shape=(N, H_out, W_out, K, K, C_in),
            strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, C_ins),
            device=A.device,
            handle=A._handle,
        )

        Z = Z.compact().reshape((N * H_out * W_out, K * K * C_in))

        out = Z @ B.compact().reshape((K * K * C_in, C_out))
        return out.compact().reshape((N, H_out, W_out, C_out))

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        k = rhs.shape[0]

        # out_hw = ((lhs.shape[1]-k)//self.stride) + 1
        # as_is = out_grad.shape[1]*self.stride - k + 1
        # want = lhs.shape[1]
        l = conv(
            dilate(out_grad, axes=(1, 2), dilation=self.stride - 1),
            transpose(flip(rhs, (0, 1)), axes=(2, 3)),
            padding=k - self.padding - 1,
        )

        # as_is = lhs.shape[1] - (out_grad.shape[1] * (self.stride)) + 1
        # want = rhs.shape[0]

        r = conv(
            transpose(lhs, axes=(0, 3)),
            dilate(
                transpose(
                    transpose(out_grad, axes=(0, 1)),
                    axes=(1, 2),
                ),
                axes=(0, 1),
                dilation=self.stride - 1,
            ),
            padding=self.padding,
        )
        r = transpose(
            transpose(r, axes=(0, 1)),
            axes=(1, 2),
        )
        return l, r


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


# TODO check this
class Conv1d(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        if stride is None:
            self.stride = 1
        else:
            self.stride = stride

        if padding is None:
            self.padding = 0
        else:
            self.padding = padding

    def compute(self, A, B):
        if self.padding:
            p = self.padding
            A = A.pad(((0, 0), (p, p), (0, 0)))

        N, H, C_in = A.shape
        Ns, Hs, C_ins = A.strides
        K, _, C_out = B.shape

        H_out = ((H - K) // self.stride) + 1

        Z = NDArray.make(
            shape=(N, H_out, K, C_in),
            strides=(Ns, Hs * self.stride, Hs, C_ins),
            device=A.device,
            handle=A._handle,
        )

        Z = Z.compact().reshape((N * H_out, K * C_in))

        out = Z @ B.compact().reshape((K * C_in, C_out))
        return out.compact().reshape((N, H_out, C_out))

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        k = rhs.shape[0]

        l = conv1d(
            dilate(out_grad, axes=(1,), dilation=self.stride - 1),
            transpose(flip(rhs, (0,))),
            padding=k - self.padding - 1,
        )

        r = conv1d(
            transpose(lhs, axes=(0, 2)),
            dilate(
                transpose(out_grad, axes=(0, 1)),
                axes=(0),
                dilation=self.stride - 1,
            ),
            padding=self.padding,
        )
        r = transpose(r, axes=(0, 1))
        return l, r


def conv1d(arr, k, stride=None, padding=None):
    return Conv1d(stride=stride, padding=padding)(arr, k)

# class Conv1d(TensorOp):
#     def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0, groups: Optional[int] = 1):
#         if stride is None:
#             self.stride = 1
#         else:
#             self.stride = stride

#         if padding is None:
#             self.padding = 0
#         else:
#             self.padding = padding

#         if groups is None:
#             self.groups = 1
#         else:
#             self.groups = groups

#     def compute(self, A, B):
#         if self.padding:
#             p = self.padding
#             A = A.pad(((0, 0), (p, p), (0, 0)))

#         N, H, C_in = A.shape
#         Ns, Hs, C_ins = A.strides
#         K, _, C_out = B.shape

#         # Ensure input and output channels are divisible by groups
#         assert C_in % self.groups == 0, "Input channels must be divisible by groups"
#         assert C_out % self.groups == 0, "Output channels must be divisible by groups"

#         # Calculate output dimensions
#         H_out = ((H - K) // self.stride) + 1

#         # Split inputs and weights by groups
#         group_in_channels = C_in // self.groups
#         group_out_channels = C_out // self.groups

#         # Collect outputs for each group
#         group_outputs = []

#         for g in range(self.groups):
#             # Select input channels for this group
#             A_group = A[:, :, g * group_in_channels : (g + 1) * group_in_channels]
            
#             # Select output channels for this group
#             B_group = B[:, :, g * group_out_channels : (g + 1) * group_out_channels]

#             # Perform convolution for this group
#             Ns, Hs, C_ins = A_group.strides

#             Z = NDArray.make(
#                 shape=(N, H_out, K, group_in_channels),
#                 strides=(Ns, Hs * self.stride, Hs, C_ins),
#                 device=A.device,
#                 handle=A._handle,
#             )

#             Z = Z.compact().reshape((N * H_out, K * group_in_channels))

#             out_group = Z @ B_group.compact().reshape((K * group_in_channels, group_out_channels))
#             group_outputs.append(out_group.compact().reshape((N, H_out, group_out_channels)))

#         # Concatenate group outputs
#         return NDArray.concatenate(group_outputs, axis=2)

#     def gradient(self, out_grad, node):
#         lhs, rhs = node.inputs
#         k = rhs.shape[0]

#         # Ensure input and output channels are divisible by groups
#         N, H, C_in = lhs.shape
#         _, _, C_out = rhs.shape
        
#         group_in_channels = C_in // self.groups
#         group_out_channels = C_out // self.groups

#         # Collect gradients for each group
#         lhs_grad_groups = []
#         rhs_grad_groups = []

#         for g in range(self.groups):
#             # Select input and output channels for this group
#             lhs_group = lhs[:, :, g * group_in_channels : (g + 1) * group_in_channels]
#             out_grad_group = out_grad[:, :, g * group_out_channels : (g + 1) * group_out_channels]
#             rhs_group = rhs[:, :, g * group_out_channels : (g + 1) * group_out_channels]

#             # Compute gradient for input
#             lhs_grad_g = conv1d(
#                 dilate(out_grad_group, axes=(1), dilation=self.stride - 1),
#                 transpose(flip(rhs_group, (0)), axes=(1)),
#                 padding=k - self.padding - 1,
#                 groups=1  # Gradient computation for each group is standard convolution
#             )

#             # Compute gradient for weights
#             rhs_grad_g = conv1d(
#                 transpose(lhs_group, axes=(0, 2)),
#                 dilate(
#                     transpose(out_grad_group, axes=(0, 1)),
#                     axes=(0),
#                     dilation=self.stride - 1,
#                 ),
#                 padding=self.padding,
#                 groups=1  # Gradient computation for each group is standard convolution
#             )
#             rhs_grad_g = transpose(rhs_grad_g, axes=(0, 1))

#             lhs_grad_groups.append(lhs_grad_g)
#             rhs_grad_groups.append(rhs_grad_g)

#         # TODO check this
#         # Concatenate gradients across groups
#         lhs_grad = concat(lhs_grad_groups, axis=2)
#         rhs_grad = concat(rhs_grad_groups, axis=2)

#         return lhs_grad, rhs_grad
    

# def conv1d(a, b, stride=1, padding=1, groups=1):
#     return Conv1d(stride, padding, groups)(a, b)

# TODO check concat
class Concat(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along an existing dimension.
        Parameters:
        axis - dimension to concatenate along.
        All arrays need to have matching sizes except along this axis.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        arrays = list(args)

        if not arrays:
            raise ValueError("No tensors to concatenate.")

        # Check shape compatibility
        for i in range(1, len(arrays)):
            if any(s != t for j, (s, t) in enumerate(zip(arrays[0].shape, arrays[i].shape)) if j != self.axis):
                raise ValueError("Shape mismatch: Tensors must have the same shape except along the concatenation axis.")

        # Compute the shape of the output tensor
        concat_size = sum(array.shape[self.axis] for array in arrays)
        new_shape = list(arrays[0].shape)
        new_shape[self.axis] = concat_size

        out = array_api.empty(new_shape, device=arrays[0].device)

        # Concatenate along the specified axis
        start_idx = 0
        for array in arrays:
            end_idx = start_idx + array.shape[self.axis]
            idxs = [slice(None)] * len(new_shape)
            idxs[self.axis] = slice(start_idx, end_idx)
            out[tuple(idxs)] = array
            start_idx = end_idx

        return out

    def gradient(self, out_grad, node):
        arrays = node.inputs
        splits = [array.shape[self.axis] for array in arrays]
        return split(out_grad, self.axis, splits)


def concat(args, axis):
    return Concat(axis)(make_tuple(*args))

class Softplus(TensorOp):
    def __init__(self, beta: Optional[float] = 1.0, threshold: Optional[float] = 20.0):
        self.beta = beta
        self.threshold = threshold

    def compute(self, A: NDArray):
        ### BEGIN YOUR SOLUTION
        below = (A * self.beta <= self.threshold) * array_api.log(1 + array_api.exp(self.beta * A)) / self.beta
        above = (A * self.beta > self.threshold) * A
        return below + above
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        A = node.inputs[0]
        below = Tensor((A.realize_cached_data() * self.beta <= self.threshold), device=A.device, dtype=A.dtype, requires_grad=False) * exp(self.beta * A) / (1 + exp(self.beta * A))
        above = Tensor((A.realize_cached_data() * self.beta > self.threshold), device=A.device, dtype=A.dtype, requires_grad=False)
        return (below + above) * out_grad
        ### END YOUR SOLUTION


def softplus(a, beta=1.0, threshold=20.0):
    return Softplus(beta, threshold)(a)


class Clamp(TensorOp):
    def __init__(self, minimum: Optional[float] = None, maximum: Optional[float] = None):
        self.minimum = minimum
        self.maximum = maximum

    def compute(self, A: NDArray):
        ### BEGIN YOUR SOLUTION
        if self.minimum is not None:
            new_A = A * (A >= self.minimum)
            A = new_A + ((A < self.minimum) * self.minimum)
        if self.maximum is not None:
            new_A = A * (A <= self.maximum)
            A = new_A + ((A > self.maximum) * self.maximum)
        return A
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        A = node.inputs[0]
        if self.minimum is not None:
            out_grad.data = out_grad.data * Tensor((A.realize_cached_data() >= self.minimum), device=A.device, dtype=A.dtype, requires_grad=False)
        if self.maximum is not None:
            out_grad.data = out_grad.data * Tensor((A.realize_cached_data() <= self.maximum), device=A.device, dtype=A.dtype, requires_grad=False)
        return out_grad
        ### END YOUR SOLUTION


def clamp(a, minimum=None, maximum=None):
    return Clamp(minimum, maximum)(a)



class StridedSlice(TensorOp):
    # takes index slice(1, None, None) as input
    def __init__(self, start: int, end: Optional[int], stride: int, axis: int):
        self.start = start
        self.end = end
        self.stride = stride
        self.axis = axis

    def compute(self, a):
        idx = [slice(None)] * len(a.shape)
        idx[self.axis] = slice(self.start, self.end, self.stride)
        return a[tuple(idx)]
    
    def gradient(self, out_grad, node):
        a = node.inputs[0]
        out = array_api.full(a.shape, 0, device=a.device)
        out = Tensor(out, device=a.device, requires_grad=True)
        # out = array_api.full(a.shape, 0, device=a.device)
        idx = [slice(None)] * len(a.shape)
        idx[self.axis] = slice(self.start, self.end, self.stride)
        out[tuple(idx)] = out_grad
        return out
    
class Squeeze(TensorOp):
    def __init__(self, axis: int):
        self.axis = axis

    def compute(self, a):
        # return array_api.squeeze(a, axis=self.axis)
        if self.axis < 0:
            self.axis = len(a.shape) + self.axis
        shape = list(a.shape)
        assert shape[self.axis] == 1, f"Can't squeeze axis {self.axis} as it's not of size 1"
        shape.pop(self.axis)    # TODO check for negative axis
        return array_api.reshape(a, shape)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        # Get the original shape before squeeze
        shape = list(a.shape)
        # Insert 1 at the squeezed axis position
        # For example, if shape was (3,1,4) and axis=1, we want to go back to (3,1,4)
        return array_api.reshape(out_grad, shape)

def squeeze(a, axis):
    return Squeeze(axis)(a)
    
class Unsqueeze(TensorOp):
    def __init__(self, axis: int):
        self.axis = axis

    def compute(self, a):
        if self.axis < 0:
            self.axis = len(a.shape) + self.axis + 1
        shape = list(a.shape)
        shape.insert(self.axis, 1)
        return array_api.reshape(a, shape)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        shape = list(a.shape)
        return array_api.reshape(out_grad, shape)

def unsqueeze(a, axis):
    return Unsqueeze(axis)(a)

# class Pad(TensorOp):
#     def __init__(self, pad_width: Tuple[Tuple[int, int], ...], mode: str = "constant", constant_values: Optional[Union[int, float]] = 0):
#         self.pad_width = pad_width
#         self.mode = mode
#         self.constant_values = constant_values

#     def compute(self, a):
#         return array_api.pad(a, pad_width=self.pad_width, mode=self.mode, constant_values=self.constant_values)

#     def gradient(self, out_grad, node):
#         return array_api.slice(out_grad, [slice(self.pad_width[i][0], -self.pad_width[i][1]) for i in range(len(self.pad_width))])
    
# def pad(a, pad_width, mode="constant", constant_values=0):
#     return Pad(pad_width, mode, constant_values)(a)

