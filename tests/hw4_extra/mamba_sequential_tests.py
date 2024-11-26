import sys
sys.path.append("./python")
sys.path.append("./apps")
sys.path.append("./")
import numpy as np
import needle as ndl
import needle.nn as nn
from needle.nn import MambaConfigSeq, MambaSeq
from simple_ml import train_ptb, evaluate_ptb

device = ndl.cpu()
corpus = ndl.data.Corpus("data/ptb")
train_data = ndl.data.batchify(corpus.train, batch_size=256, device=device, dtype="float32")
config = MambaConfigSeq(dim_model=28, num_layers=3, dim_state=12, device=device)
model = MambaSeq(config)
train_ptb(model, train_data, seq_len=20, n_epochs=10, device=device, lr=0.003, optimizer=ndl.optim.Adam)
evaluate_ptb(model, train_data, seq_len=20, device=device)