"""Operator implementations."""

import math
import operator
from functools import reduce
from numbers import Number
from typing import List, Optional, Tuple, Union

from ..autograd import NDArray, Op, Tensor, TensorOp, TensorTuple, TensorTupleOp, Value
from ..backend_selection import BACKEND, array_api
from .ops_mathematic import *
from .ops_tuple import *


class PScan(TensorOp):

    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda

    def compute(self, A: NDArray, X: NDArray):  # type: ignore
        # A : (B, L, D, N)
        # X : (B, L, D, N)

        A = A.permute((0, 2, 1, 3))  # (B, D, L, N)
        X = X.permute((0, 2, 1, 3))  # (B, D, L, N)

        if self.use_cuda:
            result = A.pscan(X)
        else:
            result = self.cpu_pscan(A, X)

        return result.permute((0, 2, 1, 3))

    def gradient(self, out_grad: Value, node: Value) -> Tuple[Value, ...]:
        # A_in : (B, L, D, N)
        # X_in : (B, L, D, N)
        # out_grad : (B, L, D, N)
        # node: (B, L, D, N)
        A_in, X_in = node.inputs

        out_grad = transpose(out_grad, axes=(1, 2))  # (B, D, L, N)
        A_in = transpose(A_in, axes=(1, 2))  # (B, D, L, N)
        result = transpose(node, axes=(1, 2))  # (B, D, L, N)

        out_grad = reverse_pscan(A_in, out_grad, self.use_cuda)

        Q = init.zeros_like(X_in, device=X_in.device)
        Q[:, :, 1:, :] = Q[:, :, 1:, :] + (result[:, :, :-1, :] * out_grad[:, :, 1:, :])

        return Q.transpose(2, 1), out_grad.transpose(2, 1)

    @staticmethod
    def cpu_pscan(A: NDArray, X: NDArray) -> NDArray:  # type: ignore
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values :
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

        # only supports L that is a power of two (mainly for a clearer code)

        B, D, L, N = A.shape
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.shape[2]
            Aa = Aa.reshape((B, D, T // 2, 2, N))
            Xa = Xa.reshape((B, D, T // 2, 2, N))

            # Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Xa[:, :, :, 1, :] = Xa[:, :, :, 1, :] + (
                Aa[:, :, :, 1, :] * Xa[:, :, :, 0, :]
            )
            # Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])
            Aa[:, :, :, 1, :] = Aa[:, :, :, 1, :] * Aa[:, :, :, 0, :]

            Aa = Aa[:, :, :, 1, :]
            Xa = Xa[:, :, :, 1, :]

        # we have only 4, 2 or 1 nodes left
        if Xa.shape[2] == 4:
            # Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Xa[:, :, 1, :, :] = Xa[:, :, 1, :, :] + (
                Aa[:, :, 1, :, :] * Xa[:, :, 0, :, :]
            )
            # Aa[:, :, 1].mul_(Aa[:, :, 0])
            Aa[:, :, 1, :, :] = Aa[:, :, 1, :, :] * Aa[:, :, 0, :, :]

            # Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
            Xa[:, :, 3, :, :] = Xa[:, :, 3, :, :] + (
                Aa[:, :, 3, :, :]
                * (Xa[:, :, 2, :, :] + Aa[:, :, 2, :, :] * Xa[:, :, 1, :, :])
            )
        elif Xa.shape[2] == 2:
            # Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Xa[:, :, 1] = Xa[:, :, 1] + (Aa[:, :, 1] * Xa[:, :, 0])
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        Aa = A[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        Xa = X[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        # Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Xa[:, :, 2] = Xa[:, :, 2] + (Aa[:, :, 2] * Xa[:, :, 1])
        # Aa[:, :, 2].mul_(Aa[:, :, 1])
        Aa[:, :, 2] = Aa[:, :, 2] * Aa[:, :, 1]

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 2**k - 1 : L : 2**k]
            Xa = X[:, :, 2**k - 1 : L : 2**k]

            T = Xa.shape[2]
            Aa = Aa.reshape((B, D, T // 2, 2, N))
            Xa = Xa.reshape((B, D, T // 2, 2, N))

            # Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Xa[:, :, 1:, 0] = Xa[:, :, 1:, 0] + (Aa[:, :, 1:, 0] * Xa[:, :, :-1, 1])
            # Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])
            Aa[:, :, 1:, 0] = Aa[:, :, 1:, 0] * Aa[:, :, :-1, 1]


def pscan(A, X, use_cuda: bool):
    return PScan(use_cuda)(A, X)


class ReversePScan(TensorOp):

    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda

    def compute(self, A: NDArray, X: NDArray):  # type: ignore
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        A = A[:, :, 1:, :].pad(((0, 0), (0, 0), (0, 1), (0, 0)))
        if self.use_cuda:
            return A.reverse_pscan(X)
        else:
            return self.cpu_pscan_rev(A, X)

    def gradient(self, out_grad: Value, node: Value) -> Tuple[Value, ...]:
        raise NotImplementedError

    @staticmethod
    def cpu_pscan_rev(A: NDArray, X: NDArray) -> NDArray:  # type: ignore
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # the same function as above, but in reverse
        # (if you flip the input, call pscan, then flip the output, you get what this function outputs)
        # it is used in the backward pass

        # only supports L that is a power of two (mainly for a clearer code)

        B, D, L, N = A.shape
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.shape[2]
            Aa = Aa.reshape((B, D, T // 2, 2, N))
            Xa = Xa.reshape((B, D, T // 2, 2, N))

            # Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Xa[:, :, :, 0] = Xa[:, :, :, 0] + (Aa[:, :, :, 0] * Xa[:, :, :, 1])
            # Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])
            Aa[:, :, :, 0] = Aa[:, :, :, 0] * Aa[:, :, :, 1]

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        # we have only 4, 2 or 1 nodes left
        if Xa.shape[2] == 4:
            # Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Xa[:, :, 2] = Xa[:, :, 2] + (Aa[:, :, 2] * Xa[:, :, 3])
            # Aa[:, :, 2].mul_(Aa[:, :, 3])
            Aa[:, :, 2] = Aa[:, :, 2] * Aa[:, :, 3]

            # Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
            Xa[:, :, 0] = Xa[:, :, 0] + (
                Aa[:, :, 0] * (Xa[:, :, 1] + Aa[:, :, 1] * Xa[:, :, 2])
            )
        elif Xa.shape[2] == 2:
            # Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            Xa[:, :, 0] = Xa[:, :, 0] + (Aa[:, :, 0] * Xa[:, :, 1])
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        Aa = A[:, :, 0 : L : 2 ** (num_steps - 2)]
        Xa = X[:, :, 0 : L : 2 ** (num_steps - 2)]
        # Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Xa[:, :, 1] = Xa[:, :, 1] + (Aa[:, :, 1] * Xa[:, :, 2])
        # Aa[:, :, 1].mul_(Aa[:, :, 2])
        Aa[:, :, 1] = Aa[:, :, 1] * Aa[:, :, 2]

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 0 : L : 2**k]
            Xa = X[:, :, 0 : L : 2**k]

            T = Xa.shape[2]
            Aa = Aa.reshape((B, D, T // 2, 2, -1))
            Xa = Xa.reshape((B, D, T // 2, 2, -1))

            # Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Xa[:, :, :-1, 1] = Xa[:, :, :-1, 1] + (Aa[:, :, :-1, 1] * Xa[:, :, 1:, 0])
            # Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])
            Aa[:, :, :-1, 1] = Aa[:, :, :-1, 1] * Aa[:, :, 1:, 0]


def reverse_pscan(A, X, use_cuda: bool):
    return ReversePScan(use_cuda)(A, X)


class CPUPscan:
    """Obsolete. Use Pscan instead with use_cuda=False."""

    def compute(self, A: NDArray, X: NDArray) -> NDArray:  # type: ignore
        pass

    def gradient(self, out_grad: Value, node: Value) -> Tuple[Value, ...]:
        raise NotImplementedError

    @staticmethod
    def pscan(A: NDArray, X: NDArray):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values :
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

        # only supports L that is a power of two (mainly for a clearer code)

        B, D, L, N = A.shape
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.shape[2]
            Aa = Aa.reshape((B, D, T // 2, 2, N))
            Xa = Xa.reshape((B, D, T // 2, 2, N))

            breakpoint()
            # Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Xa[:, :, :, 1, :] = Xa[:, :, :, 1, :] + (
                Aa[:, :, :, 1, :] * Xa[:, :, :, 0, :]
            )
            # Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])
            Aa[:, :, :, 1, :] = Aa[:, :, :, 1, :] * Aa[:, :, :, 0, :]

            Aa = Aa[:, :, :, 1, :]
            Xa = Xa[:, :, :, 1, :]

        breakpoint()
        # we have only 4, 2 or 1 nodes left
        if Xa.shape[2] == 4:
            # Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Xa[:, :, 1, :, :] = Xa[:, :, 1, :, :] + (
                Aa[:, :, 1, :, :] * Xa[:, :, 0, :, :]
            )
            # Aa[:, :, 1].mul_(Aa[:, :, 0])
            Aa[:, :, 1, :, :] = Aa[:, :, 1, :, :] * Aa[:, :, 0, :, :]

            # Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
            Xa[:, :, 3, :, :] = Xa[:, :, 3, :, :] + (
                Aa[:, :, 3, :, :]
                * (Xa[:, :, 2, :, :] + Aa[:, :, 2, :, :] * Xa[:, :, 1, :, :])
            )
        elif Xa.shape[2] == 2:
            # Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Xa[:, :, 1] = Xa[:, :, 1] + (Aa[:, :, 1] * Xa[:, :, 0])
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        Aa = A[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        Xa = X[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        # Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Xa[:, :, 2] = Xa[:, :, 2] + (Aa[:, :, 2] * Xa[:, :, 1])
        # Aa[:, :, 2].mul_(Aa[:, :, 1])
        Aa[:, :, 2] = Aa[:, :, 2] * Aa[:, :, 1]

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 2**k - 1 : L : 2**k]
            Xa = X[:, :, 2**k - 1 : L : 2**k]

            T = Xa.shape[2]
            Aa = Aa.reshape((B, D, T // 2, 2, N))
            Xa = Xa.reshape((B, D, T // 2, 2, N))

            # Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Xa[:, :, 1:, 0] = Xa[:, :, 1:, 0] + (Aa[:, :, 1:, 0] * Xa[:, :, :-1, 1])
            # Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])
            Aa[:, :, 1:, 0] = Aa[:, :, 1:, 0] * Aa[:, :, :-1, 1]

    @staticmethod
    def pscan_rev(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # the same function as above, but in reverse
        # (if you flip the input, call pscan, then flip the output, you get what this function outputs)
        # it is used in the backward pass

        # only supports L that is a power of two (mainly for a clearer code)

        B, D, L, N = A.shape
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.shape[2]
            Aa = Aa.reshape((B, D, T // 2, 2, N))
            Xa = Xa.reshape((B, D, T // 2, 2, N))

            # Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Xa[:, :, :, 0] = Xa[:, :, :, 0] + (Aa[:, :, :, 0] * Xa[:, :, :, 1])
            # Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])
            Aa[:, :, :, 0] = Aa[:, :, :, 0] * Aa[:, :, :, 1]

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        # we have only 4, 2 or 1 nodes left
        if Xa.shape[2] == 4:
            # Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Xa[:, :, 2] = Xa[:, :, 2] + (Aa[:, :, 2] * Xa[:, :, 3])
            # Aa[:, :, 2].mul_(Aa[:, :, 3])
            Aa[:, :, 2] = Aa[:, :, 2] * Aa[:, :, 3]

            # Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
            Xa[:, :, 0] = Xa[:, :, 0] + (
                Aa[:, :, 0] * (Xa[:, :, 1] + Aa[:, :, 1] * Xa[:, :, 2])
            )
        elif Xa.shape[2] == 2:
            # Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            Xa[:, :, 0] = Xa[:, :, 0] + (Aa[:, :, 0] * Xa[:, :, 1])
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        Aa = A[:, :, 0 : L : 2 ** (num_steps - 2)]
        Xa = X[:, :, 0 : L : 2 ** (num_steps - 2)]
        # Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Xa[:, :, 1] = Xa[:, :, 1] + (Aa[:, :, 1] * Xa[:, :, 2])
        # Aa[:, :, 1].mul_(Aa[:, :, 2])
        Aa[:, :, 1] = Aa[:, :, 1] * Aa[:, :, 2]

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 0 : L : 2**k]
            Xa = X[:, :, 0 : L : 2**k]

            T = Xa.shape[2]
            Aa = Aa.reshape((B, D, T // 2, 2, -1))
            Xa = Xa.reshape((B, D, T // 2, 2, -1))

            # Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Xa[:, :, :-1, 1] = Xa[:, :, :-1, 1] + (Aa[:, :, :-1, 1] * Xa[:, :, 1:, 0])
            # Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])
            Aa[:, :, :-1, 1] = Aa[:, :, :-1, 1] * Aa[:, :, 1:, 0]

    @staticmethod
    def forward(A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.
        If you can, privilege sequence lengths that are powers of two.

        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """

        L = X_in.shape[1]

        # cloning is requiered because of the in-place ops
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            # pad tensors (and clone btw)
            A = pad_npo2(A_in)  # (B, npo2(L), D, N)
            X = pad_npo2(X_in)  # (B, npo2(L), D, N)

        # prepare tensors
        A = A.transpose((2, 1))  # (B, D, npo2(L), N)
        X = X.transpose((2, 1))  # (B, D, npo2(L), N)

        # parallel scan (modifies X in-place)
        PScan.pscan(A, X)

        # ctx.save_for_backward(A_in, X)
        PScan.saved_tensors["A_in"] = A_in
        PScan.saved_tensors["X"] = X

        # slice [:, :L] (cut if there was padding)
        return X.transpose((2, 1))[:, :L]

    @staticmethod
    def backward(grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx : A_in : (B, L, D, N), X : (B, D, L, N)
            grad_output_in : (B, L, D, N)

        Returns:
            gradA : (B, L, D, N), gradX : (B, L, D, N)
        """

        # A_in, X = ctx.saved_tensors
        A_in = PScan.saved_tensors["A_in"]
        X = PScan.saved_tensors["X"]

        L = grad_output_in.shape[1]

        # cloning is requiered because of the in-place ops
        grad_output = grad_output_in.clone()
        # the next padding will clone A_in

        # prepare tensors
        grad_output = grad_output.transpose((2, 1))
        A_in = A_in.transpose((2, 1))  # (B, D, npo2(L), N)

        # A = torch.nn.functional.pad(A_in[:, :, 1:], (0, 0, 0, 1)) # (B, D, npo2(L), N) shift 1 to the left (see hand derivation)
        custom_pad_tuple = convert_to_custom_pad_format((0, 0, 0, 1), 4)
        A = A_in[:, :, 1:].pad(custom_pad_tuple)  # DONE

        # reverse parallel scan (modifies grad_output in-place)
        PScan.pscan_rev(A, grad_output)

        # TODO new function can test it if required
        Q = init.zeros_like(X)
        # Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])
        Q[:, :, 1:] = Q[:, :, 1:] + (X[:, :, :-1] * grad_output[:, :, 1:])

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]
