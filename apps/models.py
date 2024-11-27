import sys
from typing import ParamSpecKwargs

sys.path.append("./python")
import math

import needle as ndl
import needle.nn as nn
import numpy as np
from needle.nn import MambaConfig, Mamba, RMSNorm
import needle.init as init

from dataclasses import dataclass, fields, asdict
import json


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

    def __post_init__(self):
        super().__post_init__()

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple)

    def to_mamba_config(self) -> MambaConfig:
        mamba_config_fields = {field.name for field in fields(MambaConfig)}
        filtered_dict = {k: v for k, v in asdict(self).items() if k in mamba_config_fields}
        return MambaConfig(**filtered_dict)

# adapted from https://github.com/johnma2006/mamba-minimal
# def from_pretrained(name: str):
#     """
#     Returns a model loaded with pretrained weights pulled from HuggingFace.

#     Note :
#     This only work with the state-spaces/mamba-XXX model family, because there is a pytorch_model.bin file in the HF repo.
#     This is not the case of typical model saved on HF (like the state-spaces/mamba-XXX-hf model family).
#     To load the state dict of such models, I think the only way is to load the model into a AutoModelForCausalLM, and then
#     pass the state_dict to a MambaLM. I see no other way around unfrortunately (this is how it's done in jamba.py)

#     Args:
#         name: As of now, supports
#             * 'state-spaces/mamba-2.8b-slimpj'
#             * 'state-spaces/mamba-2.8b'
#             * 'state-spaces/mamba-1.4b'
#             * 'state-spaces/mamba-790m'
#             * 'state-spaces/mamba-370m'
#             * 'state-spaces/mamba-130m'

#     Returns:
#         model: a Mamba model configured with the proper parameters and initialized with the proper weights
#     """   

#     try:
#         from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
#         from transformers.utils.hub import cached_file
#     except ImportError:
#         print("The from_pretrained function pulls weights from HuggingFace and thus needs transformers to be installed (pip install transformers)")
#         return

#     def load_config_hf(model_name):
#         resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
#         return json.load(open(resolved_archive_file))
                
#     def load_state_dict_hf(model_name):
#         resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
#         return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
#     # copy config data
#     config_data = load_config_hf(name)
#     config = MambaLMConfig(d_model=config_data['d_model'], n_layers=config_data['n_layer'], vocab_size=config_data['vocab_size'])

#     model = MambaLM(config)

#     # copy weights
#     state_dict = load_state_dict_hf(name)

#     new_state_dict = {}
#     for key in state_dict:
#         if key == 'backbone.embedding.weight' or key == 'backbone.norm_f.weight':
#             new_key = key.replace('backbone.', '')
#         else:
#             new_key = key.replace('backbone', 'mamba')

#         new_state_dict[new_key] = state_dict[key]

#     model.load_state_dict(new_state_dict)

#     return model #, config

# class MambaLM(nn.Module):
#     def __init__(self, lm_config: MambaLMConfig, device=None, dtype="float32"):
#         super().__init__()
#         self.lm_config = lm_config
#         self.config = lm_config.to_mamba_config()

#         self.embedding = nn.Embedding(self.lm_config.vocab_size, self.config.d_model, device=device, dtype=dtype)
#         self.mamba = Mamba(self.config)
#         self.norm_f = RMSNorm(self.config.d_model)

#         self.lm_head = nn.Linear(self.config.d_model, self.lm_config.vocab_size, bias=False, device=device)
#         self.lm_head.weight = self.embedding.weight
        
#     def init_caches(self):
#         # hs will be initialized to zeros, so do inputs
#         hs = init.zeros(self.config.n_layers, 1, self.config.d_inner, self.config.d_state, device=next(self.parameters()).device)
#         # inputs size would be like this
#         inputs = init.zeros(self.config.n_layers, 1, self.config.d_inner, self.config.d_conv-1, device=next(self.parameters()).device)
        
#         return hs, inputs
        
#     def forward(self, token, hs, inputs):
#         # TODO figure this out?
#         breakpoint()
#         # token : (B)
#         # caches : [cache(layer) for all layers], cache : (h, inputs)

#         # logits : (B, vocab_size)
#         # caches : [cache(layer) for all layers], cache : (h, inputs)

#         x = self.embedding(token)

#         x, hs, inputs = self.mamba.step(x, hs, inputs)
#         x = self.norm_f(x)

#         logits = self.lm_head(x)

#         return logits, hs, inputs
    
class MambaLM(nn.Module):
    def __init__(self, lm_config: MambaLMConfig, device=None, dtype="float32"):
        super().__init__()
        breakpoint()
        self.lm_config = lm_config
        self.config = lm_config.to_mamba_config()

        self.embedding = nn.Embedding(self.lm_config.vocab_size, self.config.d_model, device=device, dtype=dtype)
        self.mamba = Mamba(self.config)
        self.norm_f = RMSNorm(self.config.d_model)

        self.lm_head = nn.Linear(self.config.d_model, self.lm_config.vocab_size, bias=False, device=device)
        self.lm_head.weight = self.embedding.weight
        
    def init_caches(self):
        # hs will be initialized to zeros, so do inputs
        hs = init.zeros(self.config.n_layers, 1, self.config.d_inner, self.config.d_state, device=next(self.parameters()).device)
        # inputs size would be like this
        inputs = init.zeros(self.config.n_layers, 1, self.config.d_inner, self.config.d_conv-1, device=next(self.parameters()).device)
        
        return hs, inputs
        
    def forward(self, x):
        # TODO figure this out?
        # token : (B)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # logits : (B, vocab_size)
        # caches : [cache(layer) for all layers], cache : (h, inputs)
        # TODO add embedding?
        # breakpoint()        # shape ? (B, T, D) or what?
        x = self.embedding(x)           # TODO check if we need a reshape? or transpose?
        x = self.mamba(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits

        
    


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
