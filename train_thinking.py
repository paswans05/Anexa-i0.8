import torch
import torch.nn as nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
import random
import time
import os
from anexagpt import GPTLanguageModel, encode, decode, device, batch_size, block_size, learning_rate

# -----------------------------------------------------------------------------
# SYNTHETIC REASONING DATASET
# -----------------------------------------------------------------------------
def generate_reasoning_sample():
    """
    Generates a simple math problem with a 'Thinking' step.
    Format: Q: 2+3=? <think> I need to add 2 and 3. The sum is 5. </think> A: 5
    """
    ops = ['+', '-']
    op = random.choice(ops)
    a = random.randint(0, 20)
    b = random.randint(0, 20)
    
    if op == '+':
        res = a + b
        thought = f"I need to add {a} and {b}. The sum is {res}."
    else:
        # Keep it positive for simplicity
        if a < b: a, b = b, a
        res = a - b
        thought = f"I need to subtract {b} from {a}. The difference is {res}."
        
    text = f"Q: {a}{op}{b}=? <think> {thought} </think> A: {res}\n"
    return text

print("Generating 10,000 reasoning samples...")
# Create a robust dataset of random math problems
data_block = ""
target_len = 500000 # 500k chars ~ 500k tokens
while len(data_block) < target_len:
    data_block += generate_reasoning_sample()

# Convert to tensor
print("Tokenizing data...")
data = torch.tensor(encode(data_block), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# TRAINING LOOP (Fine-Tuning)
# -----------------------------------------------------------------------------
print(f"Loading Anexa for Fine-Tuning on {device}...")
model = GPTLanguageModel()
model.to(device)

import gc
if os.path.exists('model.safetensors'):
    state_dict = load_file('model.safetensors')
    model.load_state_dict(state_dict)
    print("Loaded existing weights.")
    del state_dict
    gc.collect()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # Lower LR for fine-tuning

max_iters = 1000 # Quick fine-tune
eval_interval = 100

print("Starting Reasoning Training (Thinking Mode)...")
start_time = time.time()

for iter in range(max_iters):
    if iter % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            xb, yb = get_batch('val')
            logits, loss = model(xb, yb)
            print(f"step {iter}: val loss {loss.item():.4f}")
        model.train()

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Training finished in {time.time() - start_time:.2f}s")
save_file(model.state_dict(), 'model.safetensors')
print("Model saved to model.safetensors")

print("\n--- Test ---")
test_q = "Q: 5+5=? <think>"
context = torch.tensor(encode(test_q), dtype=torch.long, device=device).unsqueeze(0)
# Generate using stream
print(f"Prompt: {test_q}")
out_tokens = []
for tok in model.generate_stream(context, max_new_tokens=50):
    out_tokens.append(tok.item())
out = decode(out_tokens)
print(f"Output: {out}")
