import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-1 * a, high=a, **kwargs)


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out, std=std, **kwargs)



def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    bound = math.sqrt(2) * math.sqrt(3 / fan_in)
    
    if shape is None:
        return rand(fan_in, fan_out, low=-1 * bound, high=bound, **kwargs)
    else:
        return rand(*shape, low=-1 * bound, high=bound, **kwargs)

def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    std = math.sqrt(2) / math.sqrt(fan_in)
    return randn(fan_in, fan_out, std=std, **kwargs)