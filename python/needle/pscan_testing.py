from time import perf_counter

import needle as ndl
import needle.nn as nn
import numpy as np

# sys.path.append("./python")
# sys.path.append("./apps")
# sys.path.append(".")


np.random.seed(3)

B, D, L, N = 1, 28, 128, 1


def main():
    device = ndl.cpu()
    # device = ndl.cuda()

    # A_n = np.ones((B, D, L, N))
    A_n = np.random.rand(B, D, L, N)
    X_n = np.ones((B, D, L, N))

    # A = nn.init.ones(B, D, L, N, device=device)
    # X = nn.init.ones(B, D, L, N, device=device)

    A = ndl.Tensor(A_n, device=device)
    X = ndl.Tensor(X_n, device=device)

    start = perf_counter()
    for _ in range(1000):
        y = A.cached_data.pscan(X.cached_data)
    print(perf_counter() - start)

    # print(A.cached_data[0, 0, :, 1])
    # print(y[0, 0, :, 0])
    # print(y[0, 0, :, 1])

    start = perf_counter()
    for _ in range(1000):
        result = np.zeros((B, D, L, N))
        result[:, :, 0, :] = X_n[:, :, 0, :]
        for i in range(1, L):
            result[:, :, i, :] = (result[:, :, i - 1, :] * A_n[:, :, i, :]) + X_n[
                :, :, i, :
            ]
    print(perf_counter() - start)

    print("testing correctness")
    np.testing.assert_allclose(y.numpy(), result, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    main()

# main_character <- ("sruti durbha")
