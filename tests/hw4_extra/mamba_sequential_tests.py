import sys
sys.path.append("./python")
sys.path.append("./apps")
sys.path.append("./")
import numpy as np
import needle as ndl
from needle.nn import MambaConfig, MambaSeq, SoftmaxLoss, SelectiveScanSeq, SSMBlockSeq, MambaBlockSeq, ResidualBlockSeq
import gc
from needle.data.datasets.ptb_dataset import get_batch
#from simple_ml import train_ptb, evaluate_ptb

from models import MambaLM, MambaLMConfig
from simple_ml import evaluate_ptb, train_ptb

# device = ndl.cuda()
device = ndl.cpu()
corpus = ndl.data.Corpus("data/ptb",max_lines=100)
train_data = ndl.data.batchify(
    corpus.train, batch_size=8, device=device, dtype="float32"
)
# breakpoint()

config = MambaLMConfig(d_model=16, n_layers=4, vocab_size=len(corpus.dictionary))
model = MambaLM(config, device=device, sequential=True)

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
evaluate_ptb(model, train_data, seq_len=20, device=device)



"""
device = ndl.cpu()
corpus = ndl.data.Corpus("data/ptb")
train_data = ndl.data.batchify(corpus.train, batch_size=12, device=device, dtype="float32")
config = MambaConfig(d_model=28, n_layers=3, d_state=12)

selectiveScan = SelectiveScanSeq(config)
optim = ndl.optim.Adam(selectiveScan.parameters())
x = ndl.Tensor(np.random.rand(12, 8, 2*28), device=device, dtype="float32", requires_grad=True)
dt = ndl.Tensor(np.random.rand(12, 8, 2*28), device=device, dtype="float32", requires_grad=True)
A = ndl.Tensor(np.random.rand(2*28, 12), device=device, dtype="float32", requires_grad=True)
B = ndl.Tensor(np.random.rand(12, 8, 12), device=device, dtype="float32", requires_grad=True)
C = ndl.Tensor(np.random.rand(12, 8, 12), device=device, dtype="float32", requires_grad=True)
D = ndl.Tensor(np.random.rand(2*28), device=device, dtype="float32", requires_grad=True)
out = selectiveScan(x, dt, A, B, C, D)
#print(out)
optim.reset_grad()
out.backward()
#print(out.grad)
optim.step()

ssmBlock = SSMBlockSeq(config)
optim = ndl.optim.Adam(ssmBlock.parameters())
x = ndl.Tensor(np.random.rand(12, 8, 2*28), device=device, dtype="float32", requires_grad=True)
out = ssmBlock(x)
#print(out)
optim.reset_grad()
out.backward()
#print(out.grad)
optim.step()

mambaBlock = MambaBlockSeq(config)
optim = ndl.optim.Adam(mambaBlock.parameters())
x = ndl.Tensor(np.random.rand(12, 8, 28), device=device, dtype="float32", requires_grad=True)
out = mambaBlock(x)
#print(out)
optim.reset_grad()
out.backward()
#print(out.grad)
optim.step()

resBlock = ResidualBlockSeq(config)
optim = ndl.optim.Adam(resBlock.parameters())
x = ndl.Tensor(np.random.rand(12, 8, 28), device=device, dtype="float32", requires_grad=True)
out = resBlock(x)
#print(out)
optim.reset_grad()
out.backward()
#print(out.grad)
optim.step()

mamba = MambaSeq(config)
optim = ndl.optim.Adam(mamba.parameters())
print(mamba.parameters()[0])
for i in range(10):
    x = ndl.Tensor(np.random.rand(12, 8, 28), device=device, dtype="float32", requires_grad=True)
    out = mamba(x)
    #print(out)
    optim.reset_grad()
    out.backward()
    #print(out.grad)
    optim.step()
print(mamba.parameters()[0])
"""
#train_ptb(model, train_data, seq_len=8, n_epochs=10, device=device, lr=0.003, optimizer=ndl.optim.Adam)
#evaluate_ptb(model, train_data, seq_len=8, device=device)

