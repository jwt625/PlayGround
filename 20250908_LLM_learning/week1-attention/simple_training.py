"""
Simple training example for the Annotated Transformer
Demonstrates training on a copy task - learning to copy input sequences
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import time
import numpy as np
from annotated_transformer import make_model, subsequent_mask


class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


def data_gen(V, batch, nbatches):
    """Generate random data for a src-tgt copy task."""
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1  # Start token
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    """A simple loss compute and train function."""
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                            y.contiguous().view(-1))
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss.data.item()


def run_epoch(data_iter, model, loss_compute):
    """Standard Training and Logging Function"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, 
                          batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        
        if i % 50 == 1:
            elapsed = time.time() - start
            print(f"Epoch Step: {i} Loss: {loss / batch.ntokens:.4f} "
                  f"Tokens per Sec: {tokens / elapsed:.1f}")
            start = time.time()
            tokens = 0
            
    return total_loss / total_tokens


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """Greedy decoding for inference"""
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                          ys, 
                          subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                       torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def train_copy_task():
    """Train the model on a simple copy task"""
    print("Training Transformer on copy task...")

    # Model parameters
    V = 11  # Vocabulary size
    criterion = nn.CrossEntropyLoss()
    
    # Create model - using smaller size for quick training
    model = make_model(V, V, N=2, d_model=64, d_ff=128, h=4)
    
    # Move to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on device: {device}")
    
    # Optimizer
    model_opt = Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.98), eps=1e-9)
    
    # Training loop
    model.train()
    for epoch in range(10):
        print(f"\nEpoch {epoch + 1}/10")
        
        # Move data to device
        data_iter = []
        for batch in data_gen(V, 30, 20):  # 30 batch size, 20 batches
            batch.src = batch.src.to(device)
            batch.tgt = batch.tgt.to(device)
            batch.tgt_y = batch.tgt_y.to(device)
            batch.src_mask = batch.src_mask.to(device)
            batch.tgt_mask = batch.tgt_mask.to(device)
            data_iter.append(batch)
        
        loss_compute = SimpleLossCompute(model.generator, criterion, model_opt)
        avg_loss = run_epoch(data_iter, model, loss_compute)
        print(f"Average loss: {avg_loss:.4f}")
    
    # Test the model
    print("\nTesting the trained model...")
    model.eval()
    
    # Create a test sequence
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(device)
    src_mask = torch.ones(1, 1, 10).to(device)
    
    print(f"Source: {src}")
    
    # Decode
    result = greedy_decode(model, src, src_mask, max_len=10, start_symbol=1)
    print(f"Result: {result}")
    
    # Check if it learned to copy
    if torch.equal(src, result):
        print("✅ Perfect copy! Model learned the task.")
    else:
        print("❌ Not perfect, but that's expected with limited training.")
        print(f"Accuracy: {(src == result).float().mean().item():.2%}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_copy_task()
