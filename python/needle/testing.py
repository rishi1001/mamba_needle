import itertools
import os
import sys

import needle as ndl
import needle.nn as nn
import numpy as np
import torch
from models import LanguageModel
from simple_ml import *

np.random.seed(3)


if __name__ == "__main__":
    # Test basic functionality

    # x -> 2d tensor
    # x = ndl.Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    # # test slicing
    # y = x[1:, 0::2]

    # print(y)
    device = ndl.cpu()

    # Create a 2D tensor
    x = ndl.Tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        device=device,
        requires_grad=True,
    )

    # create a 3d tensor
    x = ndl.Tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
            ],
        ],
        device=device,
        requires_grad=True,
    )

    print("x.shape:", x.shape)

    # breakpoint()

    # Perform slicing
    y = x[1:, 2, 1]  # Slice rows 1 to end, and columns 0 to end with step 2

    print(y)

    # Define a simple loss function
    loss = y.sum()  # Scalar output

    # Backward pass
    loss.backward()

    # Check gradients
    print("Gradient of x:\n", x.grad.numpy())


# Create a tensor
# x = ndl.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# # Assign a scalar value to a slice
# x[1:, 1:] = 42
# print("After scalar assignment:\n", x)

# # Assign another tensor to a slice
# y = ndl.Tensor([[10, 11], [12, 13]])
# x[1:, 1:] = y
# print("After tensor assignment:\n", x)
