import torch
import torch.nn as nn
from torch.nn import functional as F


# Hyperparameters
batch_size = 32     # number of independent sequences to be processed in parallel
block_size = 8      # maximum context length for prediction
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
new_tokens = 500
# -------------------------------------

torch.manual_seed(1337)

# Load the dataset
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#All the unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

#create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]           # encoder: takes a string, outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: takes a list of integers, outputs as string

# split into train and validation dataset
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
  """ generate a small batch of data of inputs x and targets y"""

  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size]for i in ix])
  y = torch.stack([data[i+1: i+block_size+1] for i in ix])
  x,y = x.to(device), y.to(device)
  return x,y

# Simplest Baseline: Bigram Model
class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
  
  def forward(self, idx, targets=None):

    #idx and targets are both (B,T) tensor of integers
    logits = self.token_embedding_table(idx) #(B,T,C)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)   # ref documentation... target needs to be of shape (B*T) and logits of shape (B*T, C)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is (B,T) array of indices in the current context
    for _ in range(max_new_tokens):
      #get the predictions
      logits, loss = self(idx)
      #focus only on the last time step
      logits = logits[:, -1, :] #becomes (B,C)
      #apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1) #(B,C)
      #sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
      #append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) #(B,T+1)
    return idx

model = BigramLanguageModel(vocab_size).to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)



@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

# Training loop
for iter in range(max_iters):
  #every once in a while evaluate the loss on train and val sets
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f'step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')

  # sample a batch of data
  xb, yb = get_batch('train')

  # evaluate the loss
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none = True)
  loss.backward()
  optimizer.step()


# Generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens = new_tokens)[0].tolist()))