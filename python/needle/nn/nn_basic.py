"""The module.
"""
import math
from typing import List, Callable, Any

from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(ops.reshape(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype), (1, out_features)).detach())
        else:
            self.bias = init.zeros(1, out_features, device=device, dtype=dtype)

    def forward(self, X: Tensor) -> Tensor:
        b = ops.broadcast_to(self.bias, (X.shape[0], self.weight.shape[1]))
        return X @ self.weight + b


class Flatten(Module):
    def forward(self, X):
        shape = [X.shape[0], math.prod(X.shape[1:])]
        return ops.reshape(X, shape)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module.forward(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        m, n = logits.shape
        
        y_one_hot = init.one_hot(n, y, device=y.device)
        h = logits * y_one_hot # element-wise mult
        h = ops.broadcast_to(ops.reshape(ops.summation(h, axes=(1,)), (m, 1)), (m, n))
    
        return ops.summation(ops.logsumexp(logits - h, axes=(1,))) / m


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        m, n = x.shape

        sm = ops.summation(x, axes=(0,)) / m
        if self.training:
            self.running_mean = self.momentum*sm + (self.running_mean * (1 - self.momentum))
        sm = ops.broadcast_to(ops.reshape(sm, (1, n)), (m, n))

        sv = ops.summation(ops.power_scalar(x - sm, 2), axes=(0,)) / m
        if self.training:
            self.running_var = self.momentum*(sv) + (self.running_var * (1 - self.momentum))
        sv = ops.broadcast_to(ops.reshape(sv, (1, n)), (m, n))


        w = ops.broadcast_to(self.weight, (m, self.dim))
        b = ops.broadcast_to(self.bias, (m, self.dim))
        
        if self.training:
            return (w * ((x - sm) / ops.power_scalar(sv + self.eps, 1/2))) + b
        else:
            return (w * ((x - ops.broadcast_to(self.running_mean, (m, self.dim))) / ops.power_scalar(ops.broadcast_to(self.running_var, (m, self.dim)) + self.eps, 1/2))) + b

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        m, n = x.shape
        sm = ops.broadcast_to(ops.reshape(ops.summation(x, axes=(1,)), (m, 1)), (m, n)) / n
        sv = ops.broadcast_to(ops.reshape(ops.summation(ops.power_scalar(x - sm, 2), axes=(1,)), (m, 1)), (m, n)) / n

        w = ops.broadcast_to(self.weight, (m, self.dim))
        b = ops.broadcast_to(self.bias, (m, self.dim))

        return (w * ((x - sm) / ops.power_scalar(sv + self.eps, 1/2))) + b


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        
        b = init.randb(*x.shape, p=1 - self.p, device=x.device, dtype=x.dtype) * (1 / (1-self.p))
        return x * b


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
