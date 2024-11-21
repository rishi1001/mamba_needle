from typing import Optional

from ..autograd import NDArray, Op, Tensor, TensorOp, TensorTuple, TensorTupleOp, Value
from ..backend_selection import BACKEND, array_api
from .ops_mathematic import *


class LogSoftmax(TensorOp):
    def compute(self, Z):
        m, n = Z.shape

        ma = Z.max(axis=1, keepdims=True).broadcast_to(Z.shape)
        lse = array_api.log(array_api.exp(Z - ma).sum(axis=1)) + Z.max(axis=1)
        result = Z - array_api.reshape(lse, (m, 1))
        return result

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        m, n = a.shape

        row = broadcast_to(reshape(summation(out_grad, axes=(1,)), (m, 1)), (m, n))
        g = exp(node)

        return out_grad - (row * g)


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = axes

    def compute(self, Z):
        m = Z.max(axis=self.axes, keepdims=True).broadcast_to(Z.shape)
        return array_api.log(array_api.exp(Z - m).sum(axis=self.axes)) + Z.max(
            axis=self.axes
        )

    def gradient(self, out_grad, node):
        a = node.inputs[0]

        if self.axes is None:
            return exp(a) / summation(exp(a)) * broadcast_to(out_grad, a.shape)

        out_shape = list(out_grad.shape)
        for axis in self.axes:
            out_shape.insert(axis, 1)

        return exp(a - broadcast_to(reshape(node, out_shape), a.shape)) * broadcast_to(
            reshape(out_grad, out_shape), a.shape
        )


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
