"""Optimization module"""

import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            u = (1 - self.momentum) * (
                param.grad.detach() + self.weight_decay * param.data
            )
            if param in self.u:
                u += self.momentum * self.u[param].detach()
            self.u[param] = u

            param.data = ndl.Tensor(param.data - self.lr * u, dtype=param.dtype)

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.u = {}
        self.v = {}

    def step(self):
        self.t += 1
        for param in self.params:
            u = param.grad.detach() + self.weight_decay * param.data
            v = (1 - self.beta2) * ndl.ops.power_scalar(u, 2)
            u = (1 - self.beta1) * u

            if param in self.u:
                u = u + self.beta1 * self.u[param]
            if param in self.v:
                v = v + self.beta2 * self.v[param]
            self.u[param] = u.detach()
            self.v[param] = v.detach()

            u_hat = u / (1 - (self.beta1**self.t))
            v_hat = v / (1 - (self.beta2**self.t))

            param.data = ndl.Tensor(
                param.data
                - self.lr
                * u_hat.detach()
                / (ndl.ops.power_scalar(v_hat.detach(), 1 / 2) + self.eps),
                dtype=param.dtype,
            ).detach()
