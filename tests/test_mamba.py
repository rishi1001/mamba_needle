import sys

sys.path.append("./python")
sys.path.append("./apps")
sys.path.append("./")
import itertools
import os

import mugrade
import needle as ndl
import needle.nn as nn
import numpy as np
import pytest
import torch
from models import LanguageModel
from simple_ml import *

# np.random.seed(3)

# mamba_config = nn.MambaConfig()

# mamba_model = nn.Mamba(mamba_config)

# batch_size = 2
# seq_len = 3
# input_dim = 4
# hidden_size = 5
# num_layers = 6
# num_head = 7
# dim_head = 8
# dropout = 0.1
# causal = True

# device = torch.device("cpu")


# def test_mamba_model():
#     model = mamba_model
#     assert isinstance(model, nn.Module)
#     assert isinstance(model, nn.Mamba)
#     assert model.config == mamba_config
#     assert model.device == torch.device("cpu")

# def test_mamba_model_forward():
#     model = mamba_model
#     x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
#     y = model(x)
#     assert y.shape == (2, 3)

#     np.random.seed(87745)

#     x = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
#     ndl_x = ndl.Tensor(x, device=device)

#     model = nn.Transformer(
#         input_dim,
#         hidden_size,
#         num_layers,
#         num_head=num_head,
#         dim_head=dim_head,
#         dropout=dropout,
#         causal=causal,
#         device=device,
#         batch_first=True,
#     )

#     result = model(ndl_x)[0]

#     result = result.numpy()

# def test_mamba_model_backward():
#     model = mamba_model
#     x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
#     y = model(x)
#     y.sum().backward()
#     assert model.weight.grad is not None
#     assert model.bias.grad is not None

# def test_mamba_model_train():
#     model = mamba_model
#     x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
#     y = model(x)
#     y.sum().backward()
#     model.train()
#     assert model.training

# def test_mamba_model_eval():
#     model = mamba_model
#     x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
#     y = model(x)
#     y.sum().backward()
#     model.eval()
#     assert not model.training


sys.path.append("./apps")
from models import MambaLM, MambaLMConfig
from simple_ml import evaluate_ptb, train_ptb

# device = ndl.cuda()
device = ndl.cpu()
corpus = ndl.data.Corpus("data/ptb", max_lines=100)
train_data = ndl.data.batchify(
    corpus.train, batch_size=8, device=device, dtype="float32"
)
# breakpoint()

config = MambaLMConfig(d_model=16, n_layers=4, vocab_size=len(corpus.dictionary))
model = MambaLM(config, device=device)

# model = LanguageModel(20, len(corpus.dictionary), hidden_size=32, num_layers=1, seq_model='transformer', seq_len=20, device=device)

train_ptb(
    model,
    train_data,
    seq_len=16,
    n_epochs=10,
    device=device,
    lr=0.003,
    optimizer=ndl.optim.Adam,
)
evaluate_ptb(model, train_data, seq_len=16, device=device)
