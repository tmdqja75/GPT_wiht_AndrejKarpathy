import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'mps' # macos
eval_iters = 200
n_embed = 32


torch.manual_seed(1337)

# read in shakespeare data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all unique characters that occur in text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping from char to integers
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # takes string, and outputs a list of integer
decode = lambda l: ''.join([itos[i] for i in l])# takes list of integer, and outputs a string

# Train and Test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data Loading
def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data)-batch_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# averages loss over multiple batches
@torch.no_grad() # tells pytorch that we will not call backward calc (more efficient)
def estimate_loss():
    out = {}
    # Set model to evaluation phase
    model.eval() 
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # set model back to train phase
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for next tokn from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        token_embed = self.token_embedding_table(idx) # (B, T, C)
        logits = self.lm_head(token_embed) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # reshape logits
            B, T, C =  logits.shape
            logits = logits.view(B*T, C)
            targets  = targets.view(B*T)
            # negative log loss
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indicies in current context
        for _ in range(max_new_tokens):
            # get prediction
            logits, loss = self(idx) # goes to def forward() function
            # focus only on last time step
            logits = logits[:, -1, :] # shape: (B. C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

# Create Optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3) # usually 1e-4 works well in bigger models


for iter in range(max_iters):

    # in every eval_interval, evaluate loss on train, val set
    if iter % eval_interval == 0:
        losses = estimate_loss()
        loss_train = losses['train']
        loss_val = losses['val']
        print(f'step {iter}: train loss {loss_train:.4f}, val loss {loss_val:.4f}')

    # sample a batch of data    
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the odel
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))