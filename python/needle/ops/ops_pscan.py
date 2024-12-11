"""Operator implementations."""

from typing import Tuple

from ..autograd import NDArray, TensorOp, Value
from .ops_mathematic import *
from .ops_tuple import *


class PScan(TensorOp):

    def compute(self, A: NDArray, X: NDArray):  # type: ignore
        # A : (B, L, D, N)
        # X : (B, L, D, N)

        # the pscan operation requires the input to be in the form of (B, D, L, N)
        A = A.compact().permute((0, 2, 1, 3)).compact()  # (B, D, L, N)
        X = X.compact().permute((0, 2, 1, 3)).compact()  # (B, D, L, N)
        result = A.pscan(X)

        return result.permute((0, 2, 1, 3))

    def gradient(self, out_grad: Value, node: Value):
        # A_in : (B, L, D, N)
        # X_in : (B, L, D, N)
        # out_grad : (B, L, D, N)
        # node: (B, L, D, N)
        A_in, X_in = node.inputs

        out_grad = transpose(out_grad, axes=(1, 2))  # (B, D, L, N)
        A_in = transpose(A_in, axes=(1, 2))  # (B, D, L, N)
        result = transpose(node, axes=(1, 2))  # (B, D, L, N)

        out_grad = reverse_pscan(A_in, out_grad)

        Q = init.zeros_like(A_in, device=A_in.device)
        Q[:, :, 1:, :] = Q[:, :, 1:, :] + (result[:, :, :-1, :] * out_grad[:, :, 1:, :])

        return Q.transpose((2, 1)), out_grad.transpose((2, 1))


def pscan(A, X):
    return PScan()(A, X)


class ReversePScan(TensorOp):

    def compute(self, A: NDArray, X: NDArray):  # type: ignore
        # A : (B, D, L, N)
        # X : (B, D, L, N)
        A = A[:, :, 1:, :].pad(((0, 0), (0, 0), (0, 1), (0, 0)))
        return A.compact().reverse_pscan(X.compact())

    def gradient(self, out_grad: Value, node: Value) -> Tuple[Value, ...]:
        # not needed for our purposes
        raise NotImplementedError


def reverse_pscan(A, X):
    return ReversePScan()(A, X)
