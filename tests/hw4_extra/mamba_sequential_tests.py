import sys

sys.path.append("./python")
sys.path.append("./apps")
sys.path.append("./")
import gc

import matplotlib.pyplot as plt
import needle as ndl
import numpy as np
from models import MambaLM, MambaLMConfig
from needle.data.datasets.ptb_dataset import get_batch
from needle.nn import (
    MambaBlockSeq,
    MambaConfig,
    MambaSeq,
    ResidualBlockSeq,
    SelectiveScanSeq,
    SoftmaxLoss,
    SSMBlockSeq,
)
from simple_ml import evaluate_ptb, train_ptb


def seq_main():
    device = ndl.cpu()
    corpus = ndl.data.Corpus("data/ptb", max_lines=100)
    train_data = ndl.data.batchify(
        corpus.train, batch_size=8, device=device, dtype="float32"
    )

    config = MambaLMConfig(d_model=16, n_layers=4, vocab_size=len(corpus.dictionary))
    model = MambaLM(config, device=device, sequential=True)

    train_errors, train_losses = train_ptb(
        model,
        train_data,
        seq_len=16,
        n_epochs=10,
        device=device,
        lr=0.003,
        optimizer=ndl.optim.Adam,
    )
    evaluate_ptb(model, train_data, seq_len=16, device=device)

    with open("seq_train_losses.txt", "w") as f:
        print("\n".join([str(x) for x in train_losses]), file=f)


def pscan_main():
    # device = ndl.cpu()
    device = ndl.cuda()
    corpus = ndl.data.Corpus("data/ptb", max_lines=100)
    train_data = ndl.data.batchify(
        corpus.train, batch_size=8, device=device, dtype="float32"
    )

    config = MambaLMConfig(d_model=16, n_layers=4, vocab_size=len(corpus.dictionary))
    model = MambaLM(config, device=device, sequential=False)

    train_errors, train_losses = train_ptb(
        model,
        train_data,
        seq_len=16,
        n_epochs=10,
        device=device,
        lr=0.003,
        optimizer=ndl.optim.Adam,
    )
    evaluate_ptb(model, train_data, seq_len=16, device=device)

    with open("pscan_train_losses.txt", "w") as f:
        print("\n".join([str(x) for x in train_losses]), file=f)


if __name__ == "__main__":
    seq_main()
    pscan_main()
