"""
Simple training example for the Annotated Transformer
Demonstrates training on a copy task - learning to copy input sequences
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Tuple
from annotated_transformer import make_model, subsequent_mask, visualize_attention, visualize_multi_head_attention


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


def data_gen(V, batch, nbatches, seq_len=32):
    """Generate random data for a src-tgt copy task with configurable sequence length."""
    for i in range(nbatches):
        # Generate longer, more varied sequences to prevent trivial memorization
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, seq_len)))
        data[:, 0] = 1  # Start token
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


class TrainingMonitor:
    """Monitor and visualize training progress"""
    def __init__(self):
        self.metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)

    def log_batch(self, loss: float, perplexity: float, tokens_per_sec: float, step: int):
        """Log metrics for a single batch"""
        self.metrics['loss'].append(loss)
        self.metrics['perplexity'].append(perplexity)
        self.metrics['tokens_per_sec'].append(tokens_per_sec)
        self.metrics['step'].append(step)

    def log_epoch(self, epoch: int, avg_loss: float, avg_perplexity: float):
        """Log metrics for an epoch"""
        self.epoch_metrics['epoch'].append(epoch)
        self.epoch_metrics['avg_loss'].append(avg_loss)
        self.epoch_metrics['avg_perplexity'].append(avg_perplexity)

    def plot_training_curves(self, save_path: str = None):
        """Plot training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curve
        ax1.plot(self.metrics['step'], self.metrics['loss'], alpha=0.7, label='Batch Loss')
        ax1.plot(self.epoch_metrics['epoch'], self.epoch_metrics['avg_loss'],
                'ro-', linewidth=2, label='Epoch Avg Loss')
        ax1.set_xlabel('Step/Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Perplexity curve
        ax2.plot(self.metrics['step'], self.metrics['perplexity'], alpha=0.7, label='Batch Perplexity')
        ax2.plot(self.epoch_metrics['epoch'], self.epoch_metrics['avg_perplexity'],
                'go-', linewidth=2, label='Epoch Avg Perplexity')
        ax2.set_xlabel('Step/Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('Training Perplexity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Tokens per second
        ax3.plot(self.metrics['step'], self.metrics['tokens_per_sec'], 'b-', alpha=0.7)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Tokens/sec')
        ax3.set_title('Training Throughput')
        ax3.grid(True, alpha=0.3)

        # Loss distribution
        ax4.hist(self.metrics['loss'], bins=30, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Loss')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Loss Distribution')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class SimpleLossCompute:
    """A simple loss compute and train function."""
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        # Calculate loss - CrossEntropyLoss with default reduction='mean'
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                            y.contiguous().view(-1))

        # For gradient computation, normalize by number of tokens
        # This helps with gradient scaling for different batch sizes
        normalized_loss = loss / norm
        normalized_loss.backward()

        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()

        # Return the original loss (already averaged by CrossEntropyLoss)
        return loss.data.item()


def run_epoch(data_iter, model, loss_compute, monitor: TrainingMonitor = None, epoch: int = 0):
    """Enhanced Training and Logging Function with monitoring"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    step_count = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt,
                          batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)

        # Debug: Check if ntokens is reasonable
        if i == 0:  # First batch of epoch
            print(f"  Debug: batch.ntokens = {batch.ntokens}, loss = {loss:.6f}")

        # Calculate metrics - loss is the average loss per token in the batch
        batch_loss = loss  # Average loss per token (from CrossEntropyLoss)
        perplexity = torch.exp(torch.tensor(batch_loss).detach()).item()

        # Accumulate for epoch average - weight by number of tokens
        total_loss += loss * batch.ntokens  # Weight loss by number of tokens
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        step_count += 1

        if i % 20 == 1:  # More frequent logging
            elapsed = time.time() - start
            tokens_per_sec = tokens / elapsed if elapsed > 0 else 0

            print(f"Epoch {epoch} Step: {i:3d} | "
                  f"Loss: {batch_loss:.4f} | "
                  f"PPL: {perplexity:.2f} | "
                  f"Tokens/sec: {tokens_per_sec:.1f}")

            # Log to monitor
            if monitor:
                monitor.log_batch(batch_loss, perplexity, tokens_per_sec,
                                epoch * len(data_iter) + i)

            start = time.time()
            tokens = 0

    # Calculate final averages - normalize by total tokens for proper loss
    avg_loss = total_loss / total_tokens  # Average loss per token
    avg_perplexity = torch.exp(torch.tensor(avg_loss).detach()).item()

    return avg_loss, avg_perplexity


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


def train_copy_task(visualize: bool = True, save_plots: bool = True):
    """Train the model on a simple copy task with comprehensive monitoring"""
    print("Training Transformer on copy task...")
    print("=" * 60)

    # Model parameters for copy task
    V = 20  # Larger vocabulary size to make task more challenging
    criterion = nn.CrossEntropyLoss()

    # Create appropriately sized model for copy task
    model = make_model(V, V, N=2, d_model=64, d_ff=128, h=4, dropout=0.1)

    # Move to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize training monitor
    monitor = TrainingMonitor()

    # Optimizer with lower learning rate for copy task
    model_opt = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    # Training loop
    model.train()
    num_epochs = 10  # Reduced epochs to prevent overfitting

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        # Generate data with longer sequences to make task more challenging
        data_iter = []
        seq_length = 25  # Longer sequences to make copying harder
        batch_size = 12  # Smaller batch size for longer sequences
        num_batches = 40  # More batches for better training

        for batch in data_gen(V, batch_size, num_batches, seq_len=seq_length):
            batch.src = batch.src.to(device)
            batch.tgt = batch.tgt.to(device)
            batch.tgt_y = batch.tgt_y.to(device)
            batch.src_mask = batch.src_mask.to(device)
            batch.tgt_mask = batch.tgt_mask.to(device)
            data_iter.append(batch)

        loss_compute = SimpleLossCompute(model.generator, criterion, model_opt)
        avg_loss, avg_perplexity = run_epoch(data_iter, model, loss_compute, monitor, epoch)

        # Log epoch metrics
        monitor.log_epoch(epoch, avg_loss, avg_perplexity)

        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Perplexity: {avg_perplexity:.2f}")
        print(f"  Batch size: {batch_size}, Seq length: {seq_length}, Batches: {num_batches}")

        # Monitor for overfitting in copy task
        if avg_loss < 0.1:
            print(f"  WARNING: Very low loss ({avg_loss:.6f}) - possible overfitting!")
        if avg_perplexity < 1.5:
            print(f"  WARNING: Very low perplexity ({avg_perplexity:.2f}) - model may be memorizing!")

    # Plot training curves
    if visualize:
        print("\nGenerating training visualizations...")
        save_path = "training_curves.png" if save_plots else None
        monitor.plot_training_curves(save_path)

    # Test the model
    print("\nTesting the trained model...")
    print("=" * 60)
    model.eval()

    # Create test sequences
    test_cases = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
        [1, 10, 9, 8, 7, 6, 5, 4, 3, 2]
    ]

    total_accuracy = 0
    for i, test_seq in enumerate(test_cases):
        src = torch.LongTensor([test_seq]).to(device)
        src_mask = torch.ones(1, 1, len(test_seq)).to(device)

        print(f"\nTest Case {i+1}:")
        print(f"Source: {test_seq}")

        # Decode
        result = greedy_decode(model, src, src_mask, max_len=len(test_seq), start_symbol=1)
        result_list = result[0].cpu().tolist()
        print(f"Result: {result_list}")

        # Calculate accuracy
        accuracy = (src.cpu() == result.cpu()).float().mean().item()
        total_accuracy += accuracy
        print(f"Accuracy: {accuracy:.2%}")

        # Visualize attention for first test case
        if i == 0 and visualize:
            print("Generating attention visualization...")
            src_tokens = [f"tok_{x}" for x in test_seq]
            tgt_tokens = [f"tok_{x}" for x in result_list]

            # Create target input for attention visualization
            tgt_input = torch.LongTensor([result_list[:-1]]).to(device)
            tgt_mask = torch.ones(1, 1, len(result_list)-1).to(device)

            try:
                visualize_multi_head_attention(
                    model, src, tgt_input, src_mask, tgt_mask,
                    src_tokens, tgt_tokens[:-1], layer_idx=0
                )
            except Exception as e:
                print(f"Attention visualization failed: {e}")

    avg_accuracy = total_accuracy / len(test_cases)
    print(f"\nOverall Test Accuracy: {avg_accuracy:.2%}")

    if avg_accuracy > 0.8:
        print("✅ Excellent! Model learned the copy task well.")
    elif avg_accuracy > 0.5:
        print("✅ Good! Model partially learned the copy task.")
    else:
        print("❌ Model needs more training or different hyperparameters.")

    return model, monitor


def create_shakespeare_data(file_path: str = "tinyshakespeare.txt"):
    """
    Create Shakespeare dataset for character-level language modeling
    Downloads and preprocesses Shakespeare text if not available
    """
    import urllib.request
    import os

    if not os.path.exists(file_path):
        print("Downloading Tiny Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, file_path)
        print("Download complete!")

    # Read text
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Character-level tokenization
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # Convert text to indices
    data = [char_to_idx[ch] for ch in text]

    print(f"Dataset loaded: {len(text)} characters, {vocab_size} unique")
    print(f"Sample text: {text[:100]}")

    return chars, char_to_idx, idx_to_char, data, vocab_size


def shakespeare_data_gen(data, batch_size=16, seq_len=64, num_batches=50, vocab_size=65):
    """Generate batches from Shakespeare character data with improved sampling"""
    data_len = len(data)
    pad_token = vocab_size  # Use vocab_size as padding token (not in actual vocab)

    for _ in range(num_batches):
        batch_data = []
        for _ in range(batch_size):
            # Ensure we don't go out of bounds
            max_start = data_len - seq_len - 2
            if max_start <= 0:
                # Fallback for very short sequences
                start_idx = 0
                seq = data[:seq_len + 1] + [pad_token] * max(0, seq_len + 1 - len(data))
            else:
                start_idx = np.random.randint(0, max_start)
                seq = data[start_idx:start_idx + seq_len + 1]

            batch_data.append(seq)

        # Convert to tensors
        batch_tensor = torch.LongTensor(batch_data)
        src = batch_tensor[:, :-1]  # Input sequence
        tgt = batch_tensor[:, 1:]   # Target sequence (shifted by 1)

        yield Batch(src, tgt, pad_token)


def train_shakespeare_task(chars, char_to_idx, idx_to_char, data, vocab_size, visualize: bool = True, save_plots: bool = True):
    """Train the model on Shakespeare character-level language modeling"""
    print("Training Transformer on Shakespeare text...")
    print("=" * 60)

    # Enhanced model parameters for Shakespeare with regularization
    criterion = nn.CrossEntropyLoss()  # Standard cross-entropy loss

    # Create appropriately sized model for character-level text generation
    # Add 1 to vocab_size for padding token
    model_vocab_size = vocab_size + 1
    model = make_model(model_vocab_size, model_vocab_size, N=3, d_model=128, d_ff=512, h=4, dropout=0.3)

    # Move to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Calculate model size
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32

    print(f"Training on device: {device}")
    print(f"Model parameters: {param_count:,} ({model_size_mb:.1f} MB)")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Dataset size: {len(data):,} characters")

    # Memory optimization for larger model
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()  # Clear cache before training

    # Initialize training monitor
    monitor = TrainingMonitor()

    # Enhanced optimizer with weight decay for regularization
    model_opt = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)

    # Training loop with regularization focus
    model.train()
    num_epochs = 12  # Moderate epochs to prevent overfitting
    print(f"Training for {num_epochs} epochs with regularization to prevent overfitting...")
    print("Focus on generalization rather than memorization.")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        # Better data generation to prevent overfitting
        data_iter = []
        # Use consistent parameters - match seq_len in data generation with training expectations
        batch_size = 8   # Smaller batch size for longer sequences
        seq_len = 128    # Reasonable sequence length for character-level modeling
        num_batches = 40  # Moderate number of batches

        for batch in shakespeare_data_gen(data, batch_size=batch_size, seq_len=seq_len, num_batches=num_batches, vocab_size=vocab_size):
            batch.src = batch.src.to(device)
            batch.tgt = batch.tgt.to(device)
            batch.tgt_y = batch.tgt_y.to(device)
            batch.src_mask = batch.src_mask.to(device)
            batch.tgt_mask = batch.tgt_mask.to(device)
            data_iter.append(batch)

        # Gentle learning rate decay to maintain learning
        if epoch == 7:
            for param_group in model_opt.param_groups:
                param_group['lr'] *= 0.8
                print(f"  Learning rate reduced to: {param_group['lr']:.6f}")
        elif epoch == 12:
            for param_group in model_opt.param_groups:
                param_group['lr'] *= 0.8
                print(f"  Learning rate reduced to: {param_group['lr']:.6f}")

        loss_compute = SimpleLossCompute(model.generator, criterion, model_opt)
        avg_loss, avg_perplexity = run_epoch(data_iter, model, loss_compute, monitor, epoch)

        # Log epoch metrics
        monitor.log_epoch(epoch, avg_loss, avg_perplexity)

        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Perplexity: {avg_perplexity:.2f}")
        print(f"  Batch size: {batch_size}, Seq length: {seq_len}, Batches: {num_batches}")

        # Monitor for overfitting
        if avg_loss < 0.001:
            print(f"  WARNING: Very low loss ({avg_loss:.6f}) - possible overfitting!")

        # Generate sample text every 3 epochs to monitor progress more frequently
        if (epoch + 1) % 3 == 0:
            print(f"  Sample generation after epoch {epoch + 1}:")
            model.eval()
            sample_prompt = "ROMEO:"
            prompt_indices = [char_to_idx.get(c, 0) for c in sample_prompt]
            src = torch.LongTensor([prompt_indices]).to(device)
            src_mask = torch.ones(1, 1, len(prompt_indices)).to(device)

            with torch.no_grad():
                result = greedy_decode(model, src, src_mask, max_len=len(prompt_indices) + 40, start_symbol=prompt_indices[0])
                generated_indices = result[0].cpu().tolist()
                generated_text = ''.join([idx_to_char.get(idx, '?') for idx in generated_indices])
                print(f"  '{generated_text}'")
            model.train()

    # Plot training curves
    if visualize:
        print("\nGenerating training visualizations...")
        save_path = "shakespeare_training_curves.png" if save_plots else None
        monitor.plot_training_curves(save_path)

    # Save the trained model
    model_path = "shakespeare_transformer_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size,
        'model_config': {'N': 4, 'd_model': 256, 'd_ff': 1024, 'h': 8}
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # Test the model with enhanced text generation
    print("\nTesting the trained model...")
    print("=" * 60)
    model.eval()

    # Enhanced test prompts
    test_prompts = [
        "ROMEO:",
        "JULIET:",
        "To be or not to be",
        "HAMLET:",
        "What light through",
        "Fair is foul and"
    ]

    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            print(f"\nTest Case {i+1}:")
            print(f"Prompt: '{prompt}'")

            # Convert prompt to indices
            prompt_indices = [char_to_idx.get(c, 0) for c in prompt]
            src = torch.LongTensor([prompt_indices]).to(device)
            src_mask = torch.ones(1, 1, len(prompt_indices)).to(device)

            # Generate longer continuation
            max_gen_length = len(prompt_indices) + 80
            result = greedy_decode(model, src, src_mask, max_len=max_gen_length, start_symbol=prompt_indices[0])

            # Convert back to text
            generated_indices = result[0].cpu().tolist()
            generated_text = ''.join([idx_to_char.get(idx, '?') for idx in generated_indices])

            print(f"Generated: '{generated_text}'")
            print(f"Length: {len(generated_text)} characters")

            # Calculate some basic metrics
            unique_chars = len(set(generated_text))
            print(f"Unique characters: {unique_chars}/{vocab_size}")

    # Final model statistics
    print(f"\nFinal Training Statistics:")
    print(f"  Total epochs: {num_epochs}")
    print(f"  Final loss: {avg_loss:.4f}")
    print(f"  Final perplexity: {avg_perplexity:.2f}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, monitor


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("Choose training mode:")
    print("1. Copy task (simple, fast)")
    print("2. Shakespeare text (more realistic)")

    choice = input("Enter choice (1 or 2, default=1): ").strip()

    if choice == "2":
        print("Training on Shakespeare text...")
        try:
            chars, char_to_idx, idx_to_char, data, vocab_size = create_shakespeare_data()
            train_shakespeare_task(chars, char_to_idx, idx_to_char, data, vocab_size)
        except Exception as e:
            print(f"Error setting up Shakespeare data: {e}")
            print("Falling back to copy task...")
            train_copy_task()
    else:
        print("Training on copy task...")
        train_copy_task()
