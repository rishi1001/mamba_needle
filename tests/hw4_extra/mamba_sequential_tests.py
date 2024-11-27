import sys
sys.path.append("./python")
sys.path.append("./apps")
sys.path.append("./")
import numpy as np
import needle as ndl
from needle.nn import MambaConfigSeq, MambaSeq, SoftmaxLoss
import gc
from needle.data.datasets.ptb_dataset import get_batch
#from simple_ml import train_ptb, evaluate_ptb

def epoch_general_ptb(
    data,
    model,
    seq_len=40,
    loss_fn=SoftmaxLoss(),
    opt=None,
    clip=None,
    device=None,
    dtype="float32",
):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    if opt:
        model.train()
        opt.reset_grad()
    else:
        model.eval()

    nbatch, batch_size = data.shape
    tot_num_batches = 0
    tot_data = 0
    tot_batch_loss = 0
    tot_correct = 0

    for i in range(0, nbatch - 1, seq_len):
        if opt is not None:
            opt.reset_grad()

        batch_x, batch_y = get_batch(data, i, seq_len, device=device, dtype=dtype)

        out, _ = model(batch_x)
        logits = out.numpy()

        l = loss_fn(out, batch_y)
        correct = np.sum(logits.argmax(axis=1) != batch_y.numpy())

        if opt is not None:
            l.backward()
            opt.step()
            opt.reset_grad()

        tot_batch_loss += l.numpy().item()
        tot_num_batches += 1

        tot_correct += correct
        tot_data += batch_x.shape[0]

        del out
        del l
        gc.collect()

    avg_batch_loss = tot_batch_loss / tot_num_batches
    avg_batch_acc = tot_correct / tot_data
    return avg_batch_acc, avg_batch_loss

def train_ptb(
    model,
    data,
    seq_len=40,
    n_epochs=1,
    optimizer=ndl.optim.SGD,
    lr=4.0,
    weight_decay=0.0,
    loss_fn=SoftmaxLoss,
    clip=None,
    device=None,
    dtype="float32",
):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)

    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for e in range(n_epochs):
        train_err, train_loss = epoch_general_ptb(
            data,
            model,
            seq_len,
            loss_fn(),
            opt=opt,
            clip=clip,
            device=device,
            dtype=dtype,
        )

    return train_err, train_loss


def evaluate_ptb(
    model, data, seq_len=40, loss_fn=SoftmaxLoss, device=None, dtype="float32"
):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    return epoch_general_ptb(
        data, model, seq_len, loss_fn=loss_fn(), opt=None, device=device, dtype=dtype
    )



device = ndl.cpu()
clamped = ndl.ops.clamp(ndl.Tensor(np.random.rand(4, 4), device=device, dtype="float32", requires_grad=True), minimum=0.2, maximum=0.8)
print(clamped)
clamped.backward()
print(clamped.grad)
corpus = ndl.data.Corpus("data/ptb")
train_data = ndl.data.batchify(corpus.train, batch_size=12, device=device, dtype="float32")
config = MambaConfigSeq(dim_model=28, num_layers=3, dim_state=12, device=device)
model = MambaSeq(config)
optim = ndl.optim.Adam(model.parameters())
optim.reset_grad()
out = model(ndl.Tensor(np.random.rand(12, 8, 28), device=device, dtype="float32", requires_grad=True))
#loss_fn = SoftmaxLoss()
#loss = loss_fn(out, np.random.randint(0, high=28, size=(12, 8)))
out.backward()
optim.step()

#train_ptb(model, train_data, seq_len=8, n_epochs=10, device=device, lr=0.003, optimizer=ndl.optim.Adam)
#evaluate_ptb(model, train_data, seq_len=8, device=device)