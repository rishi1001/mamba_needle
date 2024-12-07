import sys
from time import perf_counter

sys.path.append("./python")
sys.path.append("./apps")
sys.path.append(".")


import needle as ndl
import needle.nn as nn
import numpy as np

np.random.seed(3)

# B, D, L, N = 10, 28, 64, 10

B, D, N = 10, 28, 10


def main():
    # device = ndl.cpu()
    device = ndl.cuda()

    for L in [16, 32, 64]:
        # A_n = np.ones((B, D, L, N))
        A_n = np.random.rand(B, D, L, N)
        X_n = np.random.rand(B, D, L, N)

        # A = nn.init.ones(B, D, L, N, device=device)
        # X = nn.init.ones(B, D, L, N, device=device)

        A = ndl.Tensor(A_n, device=device)
        X = ndl.Tensor(X_n, device=device)

        start = perf_counter()
        for _ in range(1_000):
            y1 = A.cached_data.pscan(X.cached_data).numpy()
        print(L, perf_counter() - start)

        # print(A.cached_data[0, 0, :, 1])
        # print(y[0, 0, :, 0])
        # print(y[0, 0, :, 1])

        A = A.transpose((1, 2))
        X = X.transpose((1, 2))

        start = perf_counter()
        for _ in range(1_000):
            y2 = ndl.ops.pscan(A, X, use_cuda=False).cached_data.numpy()
        print(L, perf_counter() - start)

        start = perf_counter()
        for _ in range(1_000):
            result = np.zeros((B, D, L, N))
            result[:, :, 0, :] = X_n[:, :, 0, :]
            for i in range(1, L):
                result[:, :, i, :] = (result[:, :, i - 1, :] * A_n[:, :, i, :]) + X_n[
                    :, :, i, :
                ]
        print(L, perf_counter() - start)

        print("testing correctness")
        np.testing.assert_allclose(y1, result, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.swapaxes(y2, 1, 2), result, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    main()

# main_character <- ("sruti durbha")
