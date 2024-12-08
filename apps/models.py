import sys
from typing import ParamSpecKwargs

sys.path.append("./python")
import json
import math
from dataclasses import asdict, dataclass, fields

import needle as ndl
import needle.init as init
import needle.nn as nn
import numpy as np
from needle.nn import Mamba, MambaConfig, RMSNorm

np.random.seed(0)


class ConvBN(ndl.nn.Module):

    def __init__(self, a, b, k, s, device=None, dtype="float32"):
        super().__init__()

        self.stack = nn.Sequential(
            nn.Conv(
                a,
                b,
                k,
                stride=s,
                device=device,
                dtype=dtype,
            ),
            nn.BatchNorm2d(
                dim=b,
                device=device,
                dtype=dtype,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.stack(x)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()

        self.stack = nn.Sequential(
            ConvBN(3, 16, 7, 4, device=device),
            ConvBN(16, 32, 3, 2, device=device),
            nn.Residual(
                nn.Sequential(
                    ConvBN(32, 32, 3, 1, device=device),
                    ConvBN(32, 32, 3, 1, device=device),
                )
            ),
            ConvBN(32, 64, 3, 2, device=device),
            ConvBN(64, 128, 3, 2, device=device),
            nn.Residual(
                nn.Sequential(
                    ConvBN(128, 128, 3, 1, device=device),
                    ConvBN(128, 128, 3, 1, device=device),
                )
            ),
            nn.Flatten(),
            nn.Linear(128, 128, device=device),
            nn.ReLU(),
            nn.Linear(128, 10, device=device),
        )

    def forward(self, x):
        return self.stack(x)


class LanguageModel(nn.Module):
    def __init__(
        self,
        embedding_size,
        output_size,
        hidden_size,
        num_layers=1,
        seq_model="rnn",
        seq_len=40,
        device=None,
        dtype="float32",
    ):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()

        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            output_size, embedding_size, device=device, dtype=dtype
        )

        if seq_model == "rnn":
            self.model = nn.RNN(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                device=device,
                dtype=dtype,
            )
        elif seq_model == "lstm":
            self.model = nn.LSTM(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError

        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        seq_len, bs = x.shape

        embeddings = self.embedding(x)  # seq_ln, bs, embedding_size
        out, h = self.model(embeddings, h)

        out = nn.ops.reshape(
            out,
            (seq_len * bs, self.hidden_size),
        )
        out = self.linear(out)
        return out, h


"""

Encapsulates a Mamba model as language model. It has an embedding layer, and a LM head which maps the model output to logits.

"""

# TODO generate function : batch size != 1 ? (for now B=1)
# TODO generate function : top-p sampling


@dataclass
class MambaLMConfig(MambaConfig):
    vocab_size: int = 32000
    pad_vocab_size_multiple: int = 8

    # def __post_init__(self):
    #     super().__post_init__()

    #     if self.vocab_size % self.pad_vocab_size_multiple != 0:
    #         self.vocab_size += (
    #             self.pad_vocab_size_multiple
    #             - self.vocab_size % self.pad_vocab_size_multiple
    #         )

    def to_mamba_config(self) -> MambaConfig:
        mamba_config_fields = {field.name for field in fields(MambaConfig)}
        filtered_dict = {
            k: v for k, v in asdict(self).items() if k in mamba_config_fields
        }
        return MambaConfig(**filtered_dict)



class MambaLM(nn.Module):
    def __init__(self, lm_config: MambaLMConfig, device=None, dtype="float32", sequential=False):
        super().__init__()
        self.device = device
        self.lm_config = lm_config
        self.config = lm_config.to_mamba_config()
        self.embedding = nn.Embedding(
            self.lm_config.vocab_size, self.config.d_model, device=device, dtype=dtype
        )
        self.mamba = Mamba(self.config, device=device)
        self.norm_f = RMSNorm(self.config.d_model, device=device)


    def init_caches(self):
        # hs will be initialized to zeros, so do inputs
        hs = init.zeros(
            self.config.n_layers,
            1,
            self.config.d_inner,
            self.config.d_state,
            device=self.device,
        )
        # inputs size would be like this
        inputs = init.zeros(
            self.config.n_layers,
            1,
            self.config.d_inner,
            self.config.d_conv - 1,
            device=self.device,
        )

        return hs, inputs

    def forward(self, x):

        x = x.transpose((0, 1))
        x = self.embedding(x)
        x = self.mamba(x)
        x = self.norm_f(x)
        logits = (
            x.reshape((-1, self.config.d_model)) @ self.embedding.weight.transpose()
        )
        return logits, None


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset(
        "data/cifar-10-batches-py", train=True
    )
    train_loader = ndl.data.DataLoader(
        cifar10_train_dataset, 128, ndl.cpu(), dtype="float32"
    )
    print(cifar10_train_dataset[1][0].shape)
