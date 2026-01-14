#!/usr/bin/env python3
"""
Simple DeepSpeed training benchmark for GPU metrics testing.
Trains a small GPT-like model on synthetic data to stress the GPUs.
"""

import argparse
import time
import torch
import torch.nn as nn
import deepspeed

class SimpleGPT(nn.Module):
    """Minimal GPT-like model for benchmarking."""
    
    def __init__(self, vocab_size=50257, n_embd=768, n_head=12, n_layer=12, block_size=1024):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4 * n_embd,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_emb.weight
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params / 1e6:.2f}M")

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def get_synthetic_batch(batch_size, seq_len, vocab_size, device):
    """Generate random token sequences for benchmarking."""
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=100)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    deepspeed.init_distributed()
    
    model = SimpleGPT(n_embd=args.n_embd, n_layer=args.n_layer, block_size=args.seq_len)
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )
    
    device = model_engine.local_rank
    rank = model_engine.global_rank
    
    if rank == 0:
        print(f"\nStarting training benchmark:")
        print(f"  Layers: {args.n_layer}, Embedding: {args.n_embd}")
        print(f"  Batch size: {args.batch_size}, Seq len: {args.seq_len}")
        print(f"  Steps: {args.steps}")
        print()
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Warmup
    for _ in range(3):
        x, y = get_synthetic_batch(args.batch_size, args.seq_len, 50257, device)
        logits = model_engine(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        model_engine.backward(loss)
        model_engine.step()
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    total_tokens = 0
    for step in range(args.steps):
        x, y = get_synthetic_batch(args.batch_size, args.seq_len, 50257, device)
        
        logits = model_engine(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        
        model_engine.backward(loss)
        model_engine.step()
        
        total_tokens += args.batch_size * args.seq_len
        
        if rank == 0 and (step + 1) % 10 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            print(f"Step {step + 1}/{args.steps} | Loss: {loss.item():.4f} | "
                  f"Tokens/sec: {tokens_per_sec:.0f}")
    
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    if rank == 0:
        print(f"\nBenchmark complete:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Tokens/sec: {total_tokens / total_time:.0f}")


if __name__ == '__main__':
    main()

