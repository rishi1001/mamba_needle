import math
from typing import Union, Optional
import needle as ndl
#from needle import Device, Tensor, ops, init
import needle.init as init
from needle import ops
from .nn_basic import Module, Parameter, Linear, Sequential
from .nn_conv import Conv1d
from .nn_mamba import MambaConfig

"""
class MambaConfig:
    Configuration class for the Mamba model as defined in
    https://arxiv.org/pdf/2312.00752

    # Model architecture configuration
    d_model: int                          # Dimensionality of the model (D)
    n_layers: int                         # Number of layers in the model
    d_state: int = 16                     # Dimensionality of the state (N)
    d_conv: int = 4                       # Dimensionality of convolution layers
    expand_factor: int = 2                # Expansion factor for intermediate layers (E)

    # Time step configuration
    dt_rank: Union[int, str] = 'auto'     # Rank of the time-step dimension (can be int or 'auto')
    dt_min: float = 0.001                 # Minimum value for the time-step
    dt_max: float = 0.1                   # Maximum value for the time-step
    dt_init: str = "random"               # Initialization method for time-step: "random" or "constant"
    dt_scale: float = 1.0                 # Scaling factor for time-step adjustments
    dt_init_floor: float = 1e-4           # Floor value for time-step initialization to prevent it from becoming too small

    # Regularization and initialization
    rms_norm_eps: float = 1e-5            # Epsilon value for normalization
    base_std: float = 0.02                # Base standard deviation for weight initialization

    # Bias and normalization settings
    bias: bool = False                    # Whether to use bias terms in layers
    conv_bias: bool = True                # Whether to use bias in convolution layers
    inner_layernorms: bool = False        # Whether to apply layer normalization to internal activations

    # Mup for RMSNorm
    mup: bool = False                     # Whether to use mup
    mup_base_width: float = 128           # Base width


    pscan: bool = False  # use parallel scan mode or sequential mode when training
    use_cuda: bool = (
        False  # use official CUDA implementation when training (not compatible with (b)float16)
    )

    # Device and datatype
    device = None
    dtype = "float32"

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        # Values without default must be initiated in kwargs
        if self.d_model is None:
            raise ValueError("d_model must be provided")
        if self.n_layers is None:
            raise ValueError("n_layers must be provided")

        # Calculate d_inner as the product of expand_factor and d_model (E*D)
        self.d_inner = self.expand_factor * self.d_model

        # Set dt_rank to an automatically calculated value if 'auto' is provided
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
"""

class SelectiveScanSeq(Module):
    """
    Sequential implementation of selective scan
    """

    def __init__(self, config: MambaConfig, device=None):
        super().__init__()
        self.config = config
        self.device = device

    def forward(self, x, dt, A, B, C, D):
        """
        Forward pass through the selective scan
        Args:
        x: Input tensor of shape (B, L, E*D)
        dt: dt tensor of shape (B, L, E*D)
        A: A tensor of shape (E*D, N)
        B: B tensor of shape (B, L, N)
        C: C tensor of shape (B, L, N)
        D: D tensor of shape (E*D)
        Returns:
        Output tensor of shape (B, L, E*D)
        """

        batch_size, seq_length, d_inner = x.shape
        _, d_state = A.shape
        assert d_inner == self.config.d_inner
        assert d_state == self.config.d_state

        # Calculate dA (B, L, E*D, N)
        dA = ops.exp(ops.broadcast_to(ops.unsqueeze(dt, 3), dt.shape + (d_state,))
                     * ops.broadcast_to(ops.reshape(A, (1, 1) + A.shape), (batch_size, seq_length) + A.shape))
        assert dA.shape == (batch_size, seq_length, d_inner, d_state)

        # Calculate dB (B, L, E*D, N)
        dB = (ops.broadcast_to(ops.unsqueeze(dt, 3), dt.shape + (d_state,))
              * ops.broadcast_to(ops.unsqueeze(B, 2), B.shape[:2] + (d_inner,) + B.shape[2:]))
        assert dB.shape == (batch_size, seq_length, d_inner, d_state)

        # Calculate dB*x (B, L, E*D, N)
        dB_x = dB * ops.broadcast_to(ops.unsqueeze(x, 3), x.shape + (d_state,))
        assert dB_x.shape == (batch_size, seq_length, d_inner, d_state)

        # Initialize h (B, E*D, N)
        h = init.zeros(batch_size, d_inner, d_state,
                       device=self.device, requires_grad=True)
        hs = []

        # Sequential (RNN) approach involves iteration over length of sequence
        dA_split = ops.split(dA, 1)
        dB_x_split = ops.split(dB_x, 1)
        for t in range(seq_length):
            # SSM equation 1a/2a (Section 2 of paper)
            h = dA_split[t] * h + dB_x_split[t]
            hs.append(h)

        # Combine into (B, L, E*D, N) Tensor
        hs = ops.stack(hs, 1)
        assert hs.shape == (batch_size, seq_length, d_inner, d_state)

        # SSM equation 1b/2b (Section 2 of paper)
        # y = (hs @ C.unsqueeze(-1)).squeeze(3)
        hs_split = [ops.split(s, 0) for s in ops.split(hs, 0)]
        C_split = [ops.split(s, 0) for s in ops.split(ops.unsqueeze(C, 3), 0)]
        y_stack = [ops.stack([hs_split[i][j] @ C_split[i][j] for j in range(seq_length)], 0) for i in range(batch_size)]
        y = ops.stack(y_stack, 0)
        y = ops.reshape(y, (batch_size, seq_length, d_inner))
        assert y.shape == (batch_size, seq_length, d_inner)

        # Skip connection using D
        y = y + ops.broadcast_to(ops.reshape(D, (1, 1, d_inner)), y.shape) * x

        return y

    def step(self, x, h, dt, A, B, C, D):
        """
        Perform a single time step of the selective scan
        Args:
        x: Input tensor of shape (B, E*D)
        h: Hidden state tensor of shape (B, E*D, N)
        dt: dt tensor of shape (B, E*D)
        A: A tensor of shape (E*D, N)
        B: B tensor of shape (B, N)
        C: C tensor of shape (B, N)
        D: D tensor of shape (E*D)
        Returns:
        Output tensor of shape (B, E*D)
        Updated hidden state tensor of shape (B, E*D, N)
        """

        batch_size, d_inner = x.shape
        _, d_state = A.shape
        assert d_inner == self.config.d_inner
        assert d_state == self.config.d_state

        # Calculate dA (B, E*D, N)
        dA = ops.exp(ops.broadcast_to(ops.unsqueeze(dt, 2), dt.shape + (d_state,))
                     * ops.broadcast_to(ops.unsqueeze(A, 0), (batch_size,) + A.shape))
        assert dA.shape == (batch_size, d_inner, d_state)

        # Calculate dB (B, E*D, N)
        dB = (ops.broadcast_to(ops.unsqueeze(dt, 2), dt.shape + (d_state,))
              * ops.broadcast_to(ops.unsqueeze(B, 1), (batch_size, d_inner, d_state)))
        assert dB.shape == (batch_size, d_inner, d_state)

        # Calculate dB*x (B, E*D, N)
        dB_x = dB * ops.broadcast_to(ops.unsqueeze(x, 2), x.shape + (d_state,))
        assert dB_x.shape == (batch_size, d_inner, d_state)

        # initialize h
        if h is None:
            h = init.zeros(batch_size, d_inner, d_state, 
                           device=self.device, requires_grad=True) # (B, E*D, N)

        # SSM equation 1a/2a (Section 2 of paper)
        h = dA * h + dB_x # (B, E*D, N)
        assert h.shape == (batch_size, d_inner, d_state)

        # SSM equation 1b/2b (Section 2 of paper)
        # y = (h @ C.unsqueeze(-1)).squeeze(2) # (B, E*D, N) @ (B, N, 1) -> (B, E*D, 1)
        h_split = ops.split(h, 0)
        C_split = ops.split(ops.unsqueeze(C, 2), 0)
        y = ops.stack([h_split[i] @ C_split[i] for i in range(batch_size)], 0)
        y = ops.reshape(y, (batch_size, d_inner))
        assert y.shape == (batch_size, d_inner)

        # Skip connection using D
        y = y + ops.broadcast_to(ops.unsqueeze(0, D), y.shape) * x

        return y, h


class SSMBlockSeq(Module):
    """
    SSM block for the Mamba model
    """

    def __init__(self, config: MambaConfig, device=None):
        super().__init__()
        self.config = config
        self.device = device

        # Projects x from d_inner (E*D) to dt (dt_rank), B (d_state), and C (d_state)
        self.x_proj = Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False,
                             device=device)

        # Projects dt from dt_rank to d_inner (E*D)
        self.dt_proj = Linear(config.dt_rank, config.d_inner, bias=True,
                              device=device)
        # Softplus activation
        self.activation = SoftplusSeq()

        # Initialize dt_proj weights to preserve variance
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            self.dt_proj.weight.data = init.constant(*self.dt_proj.weight.shape, c=dt_init_std, 
                                                     device=device, requires_grad=True)
        elif config.dt_init == "random":
            self.dt_proj.weight.data = init.rand(*self.dt_proj.weight.shape, low=-dt_init_std, high=dt_init_std, 
                                                 device=device, requires_grad=True)
        else:
            raise NotImplementedError

        # Initialize dt_proj bias so dt_min <= F.softplus(dt_proj.bias) <= dt_max
        dt = ops.exp(init.rand(config.d_inner, low=math.log(config.dt_min), high=math.log(config.dt_max), 
                            device=device, requires_grad=True))
        dt = ops.clamp(dt, minimum=config.dt_init_floor)
        inv_dt = dt + ops.log(-ops.exp(-dt) + init.ones_like(dt, requires_grad=True))
        self.dt_proj.bias.data = inv_dt
        # self.dt_proj.bias._no_reinit = True # Set dt_proj bias to not be re-initialized, not needed since we initialized

        # S4D real initialization for A (d_inner, d_state)
        A = ops.broadcast_to(
            ops.unsqueeze(init.arange(1, config.d_state+1, device=device, requires_grad=True), 0), 
            (config.d_inner, config.d_state))
        A_log = ops.log(A)
        self.A_log = Parameter(A_log)
        # self.A_log._no_weight_decay = True # no decay anyways

        # D "skip" parameter (d_inner,)
        self.D = Parameter(init.ones(config.d_inner, device=device, requires_grad=True))
        # self.D._no_weight_decay = True # no decay anyways

        self.selective_scan = SelectiveScanSeq(config)

        # Optional Layernorms
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNormSeq(config, config.dt_rank)
            self.B_layernorm = RMSNormSeq(config, config.d_state)
            self.C_layernorm = RMSNormSeq(config, config.d_state)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

    def forward(self, x):
        """
        Forward pass through the SSM block.
        Args:
        x: Input tensor of shape (B, L, E*D)
        Returns:
        Output tensor of shape (B, L, E*D)
        """

        batch_size, seq_length, d_inner = x.shape
        assert d_inner == self.config.d_inner

        # Define A
        A = -ops.exp(self.A_log)

        # Define D
        D = self.D

        # Define dt (B, L, dt_rank), B (B, L, N), and C (B, L, N) from x
        dt_B_C = ops.stack([self.x_proj(s) for s in ops.split(x, 0)], 0)
        dt_B_C_list = list(ops.split(dt_B_C, 2))
        dt = ops.stack(dt_B_C_list[:self.config.dt_rank], 2)
        B = ops.stack(dt_B_C_list[self.config.dt_rank:self.config.dt_rank+self.config.d_state], 2)
        C = ops.stack(dt_B_C_list[self.config.dt_rank+self.config.d_state:], 2)
        assert dt.shape == (batch_size, seq_length, self.config.dt_rank)
        assert B.shape == (batch_size, seq_length, self.config.d_state)
        assert C.shape == (batch_size, seq_length, self.config.d_state)

        # Apply layer norms
        dt, B, C = self._apply_layernorms(dt, B, C)

        # Projects dt to (B, L, d_inner)
        dt = ops.stack([self.dt_proj(s) for s in ops.split(dt, 0)], 0)
        assert dt.shape == (batch_size, seq_length, d_inner)

        # Apply softplus activation to dt
        dt = self.activation(dt)

        # Selective scan applied (B, L, E*D)
        y = self.selective_scan(x, dt, A, B, C, D)
        assert y.shape == (batch_size, seq_length, d_inner)

        return y

    def step(self, x, h):
        """
        Perform a single time step of the SSM block.
        Args:
        x: Input tensor of shape (B, E*D)
        h: Hidden state tensor of shape (B, E*D, N)
        Returns:
        Output tensor of shape (B, E*D)
        Updated hidden state tensor of shape (B, E*D, N)
        """

        batch_size, d_inner, d_state = h.shape
        assert d_inner == self.config.d_inner
        assert d_state == self.config.d_state

        # Define A
        A = -ops.exp(self.A_log)

        # Define D
        D = self.D

        # Define dt (B, dt_rank), B (B, d_state), and C (B, d_state) from x
        dt_B_C = self.x_proj(x)
        dt_B_C_list = list(ops.split(dt_B_C, 1))
        dt = ops.stack(dt_B_C_list[:self.config.dt_rank], 1)
        B = ops.stack(dt_B_C_list[self.config.dt_rank:self.config.dt_rank+d_state], 1)
        C = ops.stack(dt_B_C_list[self.config.dt_rank+d_state:], 1)
        assert dt.shape == (batch_size, self.config.dt_rank)
        assert B.shape == (batch_size, d_state)
        assert C.shape == (batch_size, d_state)

        # Apply layer norms
        dt, B, C = self._apply_layernorms(dt, B, C)

        # Project dt to (B, E*D)
        dt = self.dt_proj(dt)
        assert dt.shape == (batch_size, d_inner)

        # Apply softplus activation to dt
        dt = self.activation(dt)

        # Selective scan step to get y (B, E*D) and h (B, E*D, N)
        y, h = self.selective_scan.step(x, h, dt, A, B, C, D)
        assert y.shape == (batch_size, d_inner)
        assert h.shape == (batch_size, d_inner, d_state)

        return y, h
    
    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C


class MambaBlockSeq(Module):
    """
    Mamba block for the Mamba model as described in Figure 3 of
    https://arxiv.org/pdf/2312.00752
    """

    def __init__(self, config: MambaConfig, device=None):
        super().__init__()
        self.config = config
        self.device = device

        # Projects input from d_model (D) to 2*d_inner (2*E*D) for skip
        self.input_proj = Linear(config.d_model, 2 * config.d_inner,
                                 bias=config.bias, device=device)

        # Convolution block with both in_channels and out_channels as d_inner (E*D)
        # groups??
        """
        self.conv = Conv1d(config.d_inner, config.d_inner, config.d_conv,
                           padding=config.d_conv-1, groups=config.d_inner,
                           bias=config.conv_bias, device=config.device, dtype=config.dtype)
        """
        self.conv = Conv1d(config.d_inner, config.d_inner, config.d_conv, padding=config.d_conv-1, groups=config.d_inner,
                           bias=config.conv_bias, device=device) 

        # Activation function is SiLU/swish activation
        self.activation = SiLUSeq()

        # SSM block
        self.ssm = SSMBlockSeq(config)

        # Projects output from d_inner (E*D) to d_model (D)
        self.output_proj = Linear(config.d_inner, config.d_model, bias=config.bias, device=device)

    def forward(self, x):
        """
        Forward pass through the Mamba block.
        Args:
        x: Input tensor of shape (B, L, D)
        Returns:
        Output tensor of shape (B, L, D)
        """

        batch_size, seq_length, d_model = x.shape
        assert d_model == self.config.d_model

        # Splits input into 2 (B, L, E*D) Tensors for skip connection
        x_skip = ops.stack([self.input_proj(s) for s in ops.split(x, 0)], 0)
        x_skip_list = list(ops.split(x_skip, 2))
        x = ops.stack(x_skip_list[:self.config.d_inner], 2)
        skip = ops.stack(x_skip_list[self.config.d_inner:], 2)
        assert x.shape == (batch_size, seq_length, self.config.d_inner)
        assert skip.shape == (batch_size, seq_length, self.config.d_inner)

        # Convolution of x
        # x = self.conv(ops.transpose(x))[:,:,:seq_length].transpose(1, 2)
        x = ops.transpose(
            ops.stack(
                list(ops.split(
                    self.conv(ops.transpose(x)),
                    2
                ))[:seq_length],
                2
            )
        )
        assert x.shape == (batch_size, seq_length, self.config.d_inner)

        # Activation for x
        x = self.activation(x)

        # SSM Block (B, L, E*D)
        x = self.ssm(x)
        assert x.shape == (batch_size, seq_length, self.config.d_inner)

        # Activation for skip
        skip = self.activation(x)

        # Nonlinearity
        out = x * skip

        # Project output to (B, L, D)
        out = ops.stack([self.output_proj(s) for s in ops.split(out, 0)], 0)
        assert out.shape == (batch_size, seq_length, self.config.d_model)

        return out

    def step(self, x, cache):
        """
        Perform a single time step of the Mamba block.
        Args:
        x: Input tensor of shape (B, D)
        cache: Tuple of (h, inputs) where h is a tensor of shape (B, E*D, N) and
                inputs is a tensor of shape (B, E*D, d_conv-1)
        Returns:
        Output tensor of shape (B, D)
        Updated cache
        """

        batch_size, d_model = x.shape
        h, inputs = cache
        _, d_inner, d_state = h.shape
        _, _, d_conv_minus_1 = inputs.shape
        assert d_model == self.config.d_model
        assert d_inner == self.config.d_inner
        assert d_state == self.config.d_state
        assert d_conv_minus_1 == self.config.d_conv - 1

        # Splits input into 2 (B, E*D) Tensors for skip connection
        x_skip = self.input_proj(x)
        x_skip_list = list(ops.split(x_skip, 1))
        x = ops.stack(x_skip_list[:d_inner], 1)
        skip = ops.stack(x_skip_list[d_inner:], 1)
        assert x.shape == (batch_size, d_inner)
        assert skip.shape == (batch_size, d_inner)

        # Convolution step (B, E*D)
        #x = self.conv(torch.cat([inputs, x_cache], dim=2))[:,:,self.config.d_conv-1]
        x_cache = x
        inputs_x = ops.stack(ops.split(inputs, 2) + [x_cache], 2)
        x = ops.split(
                ops.stack(
                    [self.conv_grouped[i](ops.unsqueeze(s, 1)) 
                     for i, s in enumerate(ops.split(inputs_x, 1))],
                    1
                ), 
                2
            )[d_conv_minus_1]
        assert x.shape == (batch_size, d_inner)

        # Activation for x
        x = self.activation(x)

        # SSM step to get y (B, E*D) and h (B, E*D, N)
        y, h = self.ssm.step(x, h)
        assert y.shape == (batch_size, d_inner)
        assert h.shape == (batch_size, d_inner, d_state)

        # Activation for skip
        skip = self.activation(skip)

        # Nonlinearity step
        out = y * skip

        # Project output to (B, D)
        out = self.output_proj(out)
        assert out.shape == (batch_size, d_model)

        # Update cache
        inputs = ops.stack(list(ops.split(inputs, 2))[1:] + [x_cache], 2)
        assert inputs.shape == (batch_size, d_inner, d_conv_minus_1)
        cache = (h, inputs)

        return out, cache


class ResidualBlockSeq(Module):
    """
    Residual block for the Mamba model.
    """

    def __init__(self, config: MambaConfig, device=None):
        super().__init__()
        self.config = config
        self.device = device
        self.layer = MambaBlockSeq(config)
        self.norm = RMSNormSeq(config, config.d_model)

    def forward(self, x):
        """
        Forward pass through the Residual block.
        Args:
        x: Input tensor of shape (B, L, D)
        Returns:
        Output tensor of shape (B, L, D)
        """

        batch_size, seq_length, d_model = x.shape
        assert d_model == self.config.d_model

        out = x + self.layer(self.norm(x))
        assert out.shape == (batch_size, seq_length, d_model)
        return out

    def step(self, x, cache):
        """
        Perform a single time step of the Residual block.
        Args:
        x: Input tensor of shape (B, L, D)
        cache: Tuple of (h, inputs) where h is a tensor of shape (B, E*D, N) and
                inputs is a tensor of shape (B, E*D, d_conv-1)
        Returns:
        Output tensor of shape (B, L, D)
        Updated cache
        """

        batch_size, seq_length, d_model = x.shape
        h, inputs = cache
        _, d_inner, d_state = h.shape
        _, _, d_conv_minus_1 = inputs.shape
        assert d_model == self.config.d_model
        assert d_inner == self.config.d_inner
        assert d_state == self.config.d_state
        assert d_conv_minus_1 == self.config.d_conv - 1

        out, cache = self.layer.step(self.norm(x), cache)
        assert out.shape == (batch_size, seq_length, d_model)
        out = x + out
        return out, cache


class MambaSeq(Module):
    """
    Main class for the Mamba model.
    """

    def __init__(self, config: MambaConfig, device=None):
        super().__init__()
        self.config = config
        self.device = device
        self.layers = Sequential(*[ResidualBlockSeq(config) for _ in range(config.n_layers)])

    def forward(self, x):
        """
        Forward pass through the Mamba model.
        Args:
        x: Input tensor of shape (B, L, D)
        Returns:
        Output tensor of shape (B, L, D)
        """

        batch_size, seq_length, d_model = x.shape
        assert d_model == self.config.d_model

        out = self.layers(x)
        return out

    def step(self, x, caches):
        """
        Perform a single time step of the Mamba model.
        Args:
        x: Input tensor of shape (B, L, D)
        caches: Input list of cached layers (see ResidualBlock.step())
        Returns:
        Output tensor of shape (B, L, D)
        Updated caches list
        """

        batch_size, seq_length, d_model = x.shape
        assert d_model == self.config.d_model

        for i, layer in enumerate(self.layers.modules):
            x, caches[i] = layer.step(x, caches[i])
            assert x.shape == (batch_size, seq_length, d_model)
        return x, caches


class RMSNormSeq(Module):
    def __init__(self, config: MambaConfig, last_dim: int, device=None):
        super().__init__()
        self.device = device
        self.use_mup = config.mup
        self.eps = config.rms_norm_eps

        # https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not self.use_mup:
            self.weight = Parameter(init.ones(last_dim, device=device, requires_grad=True))

    def forward(self, x):
        # output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        last_dim = x.shape[-1]
        output = x * ops.power_scalar(
            ops.broadcast_to(
                ops.reshape(
                    ops.summation(
                        ops.power_scalar(x, 2), 
                        len(x.shape) - 1
                    ), 
                    x.shape[:-1] + (1,)
                ),
                x.shape
            ) + self.eps, 
            -0.5
        )

        if not self.use_mup:
            return output * ops.broadcast_to(
                ops.reshape(
                    self.weight, 
                    tuple([1 for _ in x.shape[:-1]])+ (last_dim,)
                ), 
                x.shape
            )
        else:
            return output


class SoftplusSeq(Module):
    def __init__(self, beta=1.0, threshold=20.0):
        self.beta = beta
        self.threshold = threshold
    
    def forward(self, x):
        return ops.softplus(x, beta=self.beta, threshold=self.threshold)


class SiLUSeq(Module):
    def forward(self, x):
        return x / (1 + ops.exp(-x))