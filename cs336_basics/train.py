import time
import torch
import numpy as np
import pickle
import os
from cs336_basics.nn import TransformerLM, cross_entropy, clip_gradient_norm
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.tokenizer import Tokenizer
from tests.adapters import run_save_checkpoint

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Training on: {device}")

VOCAB_SIZE = 10000
CONTEXT_LEN = 256
BATCH_SIZE = 32 
EVAL_ITERS = 20  
EVAL_INTERVAL = 500
SAVE_INTERVAL = 1000
CHECKPOINT_DIR = "cs336_basics/checkpoints"
RESUME_FROM = None 

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
with open("cs336_basics/vocab.pkl", "rb") as f: vocab = pickle.load(f)
with open("cs336_basics/merges.pkl", "rb") as f: merges = pickle.load(f)
tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
train_data = np.memmap('cs336_basics/data_bin/train.bin', dtype=np.uint16, mode='r')
valid_data = np.memmap('cs336_basics/data_bin/valid.bin', dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else valid_data
    ix = np.random.randint(0, len(data) - CONTEXT_LEN, BATCH_SIZE)
    x = torch.stack([torch.from_numpy(data[i:i+CONTEXT_LEN].astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(data[i+1:i+CONTEXT_LEN+1].astype(np.int64)) for i in ix]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            logits = model(x)
            loss = cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def generate_sample(model, max_new_tokens=100):
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(context)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token), dim=1)
    decoded_text = tokenizer.decode(context[0].tolist())
    print(f"\n--- Model Story Sample ---\n{decoded_text}\n" + "-"*30)
    model.train()

model = TransformerLM(vocab_size=VOCAB_SIZE, context_length=CONTEXT_LEN, d_model=512, num_layers=12, num_heads=8, d_ff=2048).to(device)
optimizer = AdamW(model.parameters(), lr=6e-4)
start_iter = 0
if RESUME_FROM and os.path.exists(RESUME_FROM):
    print(f"Resuming from checkpoint: {RESUME_FROM}")
    checkpoint = torch.load(RESUME_FROM, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint['iteration'] + 1
    print(f"Resuming from iteration {start_iter}")
print("Starting training loop...")
start_time = time.time()
total_steps = 10000
for it in range(start_iter, 10001):
    batch_start = time.time()
    if it % EVAL_INTERVAL == 0:
        losses = estimate_loss(model)
        print(f"Step {it}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")
        generate_sample(model)
    if it > 0 and it % SAVE_INTERVAL == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{it}.pt")
        run_save_checkpoint(model, optimizer, it, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")
    lr = get_lr_cosine_schedule(it, 6e-4, 6e-5, 500, 10000)
    for pg in optimizer.param_groups: pg['lr'] = lr
    x, y = get_batch('train')
    logits = model(x)
    loss = cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    clip_gradient_norm(model.parameters(), 1.0)
    optimizer.step()
    if it % 100 == 0:
        elapsed = time.time() - start_time
        steps_done = it - start_iter
        if steps_done > 0:
            avg_time_per_step = elapsed / steps_done
            tokens_per_sec = (BATCH_SIZE * CONTEXT_LEN) / avg_time_per_step
            steps_left = total_steps - it
            remaining_seconds = steps_left * avg_time_per_step
            hours_left = remaining_seconds / 3600
            print(f"Iter {it} | Loss: {loss.item():.4f} | {tokens_per_sec:.0f} tok/s | ETA: {hours_left:.2f} hours")