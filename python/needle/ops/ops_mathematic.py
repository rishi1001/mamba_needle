"""Operator implementations."""

from functools import reduce
from numbers import Number
from typing import Optional, List, Tuple, Union
import operator

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


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
            dims[self.axes[0]], dims[self.axes[1]] = dims[self.axes[1]], dims[self.axes[0]]
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
            self.axes = (axes, )
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
        return (out_grad * negate((power_scalar(tanh(a), 2) - 1)), )


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

            new_shape[ax] *= (self.dilation + 1)
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
            new_shape[ax] //= (self.dilation + 1)
        
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

        H_out = ((H-K)//self.stride) + 1
        W_out = ((W-K)//self.stride) + 1

        
        Z = NDArray.make(
            shape=(N, H_out, W_out, K, K, C_in),
            strides=(Ns, Hs*self.stride, Ws*self.stride, Hs, Ws, C_ins),
            device=A.device,
            handle=A._handle,
        )
        
        Z = Z.compact().reshape((N*H_out*W_out,K*K*C_in))
        
        out = Z @ B.compact().reshape((K*K*C_in, C_out))
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


