import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from safetensors.torch import load_file
import string
import gc

# -----------------------------------------------------------------------------
# HYPERPARAMETERS (Must match training)
batch_size = 1 # Benchmark 1 sequence at a time
block_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.0 # No dropout during inference
# -----------------------------------------------------------------------------

# Tokenizer (Same as training)
chars = sorted(list(string.printable))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s if c in stoi] 
decode = lambda l: ''.join([itos[i] for i in l])

# -----------------------------------------------------------------------------
# MODEL DEFINITION (Copy of Anexa Architecture)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -----------------------------------------------------------------------------
# BENCHMARK
# -----------------------------------------------------------------------------

print(f"Benchmarking Anexa on {device.upper()}...")

model = GPTLanguageModel().to(device)

# Optimization: Compile the model
try:
    print("Compiling model with torch.compile...")
    model = torch.compile(model)
except Exception as e:
    print(f"Compare failed: {e}")

model.eval() # Set to eval mode

# Load weights
if os.path.exists('model.safetensors'):
    print("Loading model weights...")
    state_dict = load_file('model.safetensors')
    model.load_state_dict(state_dict)
    del state_dict
    gc.collect()
else:
    print("Warning: No model.safetensors found. Benchmarking initialized weights.")

print("Warming up GPU...")
dummy_input = torch.zeros((1, 1), dtype=torch.long, device=device)
model.generate(dummy_input, max_new_tokens=10) # Warmup run

print("Starting Speed Test (Generating 1000 tokens)...")
start_time = time.time()
num_tokens = 1000

# Do generation
context = torch.zeros((1, 1), dtype=torch.long, device=device)
model.generate(context, max_new_tokens=num_tokens)

end_time = time.time()
duration = end_time - start_time
tps = num_tokens / duration

print("-" * 30)
print(f"Time Taken: {duration:.2f} seconds")
print(f"Speed:      {tps:.2f} Tokens/Sec")
print("-" * 30)
