"""The module.
"""

import math
from typing import List

import needle.init as init
import numpy as np
from needle import ops
from needle.autograd import Tensor

from .nn_basic import Module, Parameter


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return ops.power_scalar(ops.exp(ops.negate(x)) + 1, -1)


class RNNCell(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        bound = 1 / math.sqrt(hidden_size)
        self.W_ih = Parameter(
            init.rand(
                input_size,
                hidden_size,
                low=-1 * bound,
                high=bound,
                device=device,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                hidden_size,
                low=-1 * bound,
                high=bound,
                device=device,
            )
        )

        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    hidden_size,
                    low=-1 * bound,
                    high=bound,
                    device=device,
                )
            )

            self.bias_hh = Parameter(
                init.rand(
                    hidden_size,
                    low=-1 * bound,
                    high=bound,
                    device=device,
                )
            )
        else:
            self.bias_ih = init.zeros(hidden_size, device=device)
            self.bias_hh = init.zeros(hidden_size, device=device)

        self.nonlinearity = {
            "tanh": ops.tanh,
            "relu": ops.relu,
        }[nonlinearity]

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=self.device)

        bias_ih = ops.broadcast_to(
            ops.reshape(self.bias_ih, (1, self.hidden_size)),
            h.shape,
        )
        bias_hh = ops.broadcast_to(
            ops.reshape(self.bias_hh, (1, self.hidden_size)),
            h.shape,
        )
        return self.nonlinearity((X @ self.W_ih) + bias_ih + (h @ self.W_hh) + bias_hh)


class RNN(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()

        self.rnn_cells = [
            RNNCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                device=device,
            )
            for i in range(num_layers)
        ]

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        num_layers = len(self.rnn_cells)
        Xs = ops.split(X, 0)

        if h0 is not None:
            hs = ops.split(h0, 0)
        else:
            hs = [None] * num_layers

        ys = []
        for i in range(len(Xs)):
            x_i = Xs[i]

            new_hs = []
            for j, layer in enumerate(self.rnn_cells):
                h = layer(x_i if j == 0 else new_hs[-1], hs[j])
                new_hs.append(h)

            hs = new_hs
            ys.append(h)

        return ops.stack(ys, axis=0), ops.stack(hs, axis=0)


class LSTMCell(Module):
    def __init__(
        self, input_size, hidden_size, bias=True, device=None, dtype="float32"
    ):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        bound = 1 / math.sqrt(hidden_size)
        self.W_ih = Parameter(
            init.rand(
                input_size,
                4 * hidden_size,
                low=-1 * bound,
                high=bound,
                device=device,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                4 * hidden_size,
                low=-1 * bound,
                high=bound,
                device=device,
            )
        )

        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    4 * hidden_size,
                    low=-1 * bound,
                    high=bound,
                    device=device,
                )
            )

            self.bias_hh = Parameter(
                init.rand(
                    4 * hidden_size,
                    low=-1 * bound,
                    high=bound,
                    device=device,
                )
            )
        else:
            self.bias_ih = init.zeros(4 * hidden_size, device=device)
            self.bias_hh = init.zeros(4 * hidden_size, device=device)

        self.sig = Sigmoid()

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        bs = X.shape[0]

        if h is None or h[0] is None or h[1] is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=self.device)
            c = init.zeros(X.shape[0], self.hidden_size, device=self.device)
        else:
            h, c = h

        bias_ih = ops.broadcast_to(
            ops.reshape(self.bias_ih, (1, 4 * self.hidden_size)),
            (bs, 4 * self.hidden_size),
        )
        bias_hh = ops.broadcast_to(
            ops.reshape(self.bias_hh, (1, 4 * self.hidden_size)),
            (bs, 4 * self.hidden_size),
        )

        i, f, g, o = ops.split(
            ops.reshape(
                (X @ self.W_ih) + bias_ih + (h @ self.W_hh) + bias_hh,
                (bs, 4, self.hidden_size),
            ),
            axis=1,
            # axis=2,
        )
        i, f, g, o = self.sig(i), self.sig(f), ops.tanh(g), self.sig(o)

        c_out = f * c + i * g
        h_out = o * ops.tanh(c_out)
        return h_out, c_out


class LSTM(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        self.lstm_cells = [
            LSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                bias=bias,
                device=device,
            )
            for i in range(num_layers)
        ]

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        num_layers = len(self.lstm_cells)
        Xs = ops.split(X, 0)

        if h is not None:
            hs = ops.split(h[0], 0)
            cs = ops.split(h[1], 0)
        else:
            hs = [None] * num_layers
            cs = [None] * num_layers

        ys = []
        for i in range(len(Xs)):
            x_i = Xs[i]

            new_hs = []
            new_cs = []
            for j, layer in enumerate(self.lstm_cells):
                h, c = layer(x_i if j == 0 else new_hs[-1], (hs[j], cs[j]))
                new_hs.append(h)
                new_cs.append(c)

            hs = new_hs
            cs = new_cs
            ys.append(h)

        return ops.stack(ys, axis=0), (ops.stack(hs, axis=0), ops.stack(cs, axis=0))


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Parameter(
            init.randn(
                num_embeddings,
                embedding_dim,
                device=device,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        seq_len, bs = x.shape

        one_hot = (
            Tensor(
                init.one_hot(
                    self.num_embeddings,
                    ops.reshape(x, (seq_len * bs,)),
                    device=x.device,
                ),
                device=x.device,
            )
            # (seq_len, bs, self.num_embeddings)
        )
        # weight = ops.broadcast_to(
        #     ops.reshape(
        #         self.weight,
        #         (1, self.num_embeddings, self.embedding_dim),
        #     ),
        #     (seq_len, self.num_embeddings, self.embedding_dim)
        # )
        return ops.reshape(one_hot @ self.weight, (seq_len, bs, self.embedding_dim))
