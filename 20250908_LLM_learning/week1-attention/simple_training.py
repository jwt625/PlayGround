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
    """Enhanced loss compute and train function with gradient clipping."""
    def __init__(self, generator, criterion, opt=None, clip_grad=1.0):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        self.clip_grad = clip_grad

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
            # Apply gradient clipping to prevent exploding gradients
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.opt.param_groups[0]['params'], self.clip_grad)

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
    """Greedy decoding for inference - simplified for language modeling"""
    device = src.device
    ys = torch.ones(1, 1, dtype=torch.long, device=device).fill_(start_symbol)

    for i in range(max_len - 1):
        # Create target mask
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src).to(device)

        # Forward pass
        out = model.forward(src, ys, src_mask, tgt_mask)

        # Get probabilities for the last token
        prob = model.generator(out[:, -1])

        # Get the most likely next token
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]

        # Append to sequence
        ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long, device=device).fill_(next_word)], dim=1)

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
    """Generate batches from Shakespeare character data for language modeling"""
    data_len = len(data)

    for batch_idx in range(num_batches):
        batch_data = []
        for _ in range(batch_size):
            # Ensure we don't go out of bounds
            max_start = data_len - seq_len - 1
            if max_start <= 0:
                # If data is too short, just use what we have
                start_idx = 0
                seq = data[:min(seq_len + 1, len(data))]
                # Pad if necessary
                while len(seq) < seq_len + 1:
                    seq.append(0)  # Use 0 as padding (should be rare)
            else:
                start_idx = np.random.randint(0, max_start)
                seq = data[start_idx:start_idx + seq_len + 1]

            batch_data.append(seq)

        # Convert to tensors
        batch_tensor = torch.LongTensor(batch_data)

        # For language modeling: input is seq[:-1], target is seq[1:]
        src = batch_tensor[:, :-1]  # Input sequence [0, 1, 2, ..., seq_len-1]
        tgt = batch_tensor[:, 1:]   # Target sequence [1, 2, 3, ..., seq_len]

        # Debug: Print first batch to verify data
        if batch_idx == 0:
            print(f"  Data sample - src shape: {src.shape}, tgt shape: {tgt.shape}")
            print(f"  First sequence src: {src[0][:10].tolist()}")
            print(f"  First sequence tgt: {tgt[0][:10].tolist()}")

        yield Batch(src, tgt, pad=0)  # Use 0 as padding token


def nucleus_sample_decode(model, src, src_mask, max_len, start_symbol, temperature=0.8, top_p=0.9, repetition_penalty=1.1):
    """Nucleus (top-p) sampling with repetition penalty for better text generation"""
    device = src.device
    ys = torch.ones(1, 1, dtype=torch.long, device=device).fill_(start_symbol)

    for i in range(max_len - 1):
        # Create target mask
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src).to(device)

        # Forward pass
        out = model.forward(src, ys, src_mask, tgt_mask)

        # Get logits for the last token
        logits = model.generator(out[:, -1])

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(ys[0].tolist()):
                logits[0, token_id] /= repetition_penalty

        # Apply temperature scaling
        logits = logits / temperature

        # Nucleus (top-p) sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Set logits to -inf for tokens to remove
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = float('-inf')

        # Sample from the filtered distribution
        probs = torch.softmax(logits, dim=-1)
        next_word = torch.multinomial(probs, 1)
        next_word = next_word.data[0, 0]

        # Append to sequence
        ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long, device=device).fill_(next_word)], dim=1)

    return ys


def sample_decode(model, src, src_mask, max_len, start_symbol, temperature=1.0, top_k=None):
    """Improved sampling-based decoding with repetition penalty"""
    device = src.device
    ys = torch.ones(1, 1, dtype=torch.long, device=device).fill_(start_symbol)

    for i in range(max_len - 1):
        # Create target mask
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src).to(device)

        # Forward pass
        out = model.forward(src, ys, src_mask, tgt_mask)

        # Get probabilities for the last token
        logits = model.generator(out[:, -1])

        # Apply repetition penalty for recent tokens
        if ys.size(1) > 1:
            recent_tokens = ys[0, -min(10, ys.size(1)):].tolist()  # Last 10 tokens
            for token_id in set(recent_tokens):
                logits[0, token_id] *= 0.85  # Reduce probability of recent tokens

        # Apply temperature scaling
        logits = logits / temperature

        # Apply top-k filtering if specified
        if top_k is not None:
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(1, top_k_indices, top_k_logits)

        # Sample from the distribution
        probs = torch.softmax(logits, dim=-1)
        next_word = torch.multinomial(probs, 1)
        next_word = next_word.data[0, 0]

        # Append to sequence
        ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long, device=device).fill_(next_word)], dim=1)

    return ys


def train_shakespeare_task(chars, char_to_idx, idx_to_char, data, vocab_size, visualize: bool = True, save_plots: bool = True):
    """Train the model on Shakespeare character-level language modeling"""
    print("Training Transformer on Shakespeare text...")
    print("=" * 60)

    # Enhanced model parameters for better Shakespeare generation
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens (using 0 as pad)

    # Create better-sized model for character-level text generation
    model_vocab_size = vocab_size  # No need to add 1, using existing vocab for padding
    model = make_model(model_vocab_size, model_vocab_size, N=4, d_model=256, d_ff=512, h=8, dropout=0.1)

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

    # Enhanced optimizer with better learning rate scheduling
    model_opt = Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)

    # Cosine annealing scheduler for better convergence
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(model_opt, T_max=25, eta_min=1e-6)

    # Extended training for better results
    model.train()
    num_epochs = 25
    print(f"Training for {num_epochs} epochs with regularization to prevent overfitting...")
    print("Focus on generalization rather than memorization.")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        # Improved data generation parameters
        data_iter = []
        # Better parameters for quality learning
        batch_size = 12   # Balanced batch size
        seq_len = 64     # Longer sequences for better context
        num_batches = 75  # More batches for better coverage

        for batch in shakespeare_data_gen(data, batch_size=batch_size, seq_len=seq_len, num_batches=num_batches, vocab_size=vocab_size):
            batch.src = batch.src.to(device)
            batch.tgt = batch.tgt.to(device)
            batch.tgt_y = batch.tgt_y.to(device)
            batch.src_mask = batch.src_mask.to(device)
            batch.tgt_mask = batch.tgt_mask.to(device)
            data_iter.append(batch)

        loss_compute = SimpleLossCompute(model.generator, criterion, model_opt, clip_grad=1.0)
        avg_loss, avg_perplexity = run_epoch(data_iter, model, loss_compute, monitor, epoch)

        # Update learning rate
        scheduler.step()

        # Log epoch metrics
        monitor.log_epoch(epoch, avg_loss, avg_perplexity)

        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Perplexity: {avg_perplexity:.2f}")
        print(f"  Batch size: {batch_size}, Seq length: {seq_len}, Batches: {num_batches}")

        # Monitor for overfitting
        if avg_loss < 0.5:
            print(f"  WARNING: Very low loss ({avg_loss:.6f}) - possible overfitting!")

        # Generate sample text every 3 epochs to monitor progress
        if (epoch + 1) % 3 == 0:
            print(f"  Sample generation after epoch {epoch + 1}:")
            model.eval()
            sample_prompt = "ROMEO:"
            prompt_indices = [char_to_idx.get(c, 0) for c in sample_prompt]
            src = torch.LongTensor([prompt_indices]).to(device)
            src_mask = torch.ones(1, 1, len(prompt_indices)).to(device)

            with torch.no_grad():
                # Use nucleus sampling for better generation
                result = nucleus_sample_decode(model, src, src_mask, max_len=len(prompt_indices) + 60,
                                             start_symbol=prompt_indices[0], temperature=0.7, top_p=0.9, repetition_penalty=1.2)
                generated_indices = result[0].cpu().tolist()
                generated_text = ''.join([idx_to_char.get(idx, '?') for idx in generated_indices])
                print(f"  Nucleus: '{generated_text[:80]}...'")

                # Also try regular sampling for comparison
                result2 = sample_decode(model, src, src_mask, max_len=len(prompt_indices) + 60,
                                      start_symbol=prompt_indices[0], temperature=0.8, top_k=15)
                generated_indices2 = result2[0].cpu().tolist()
                generated_text2 = ''.join([idx_to_char.get(idx, '?') for idx in generated_indices2])
                print(f"  Regular: '{generated_text2[:80]}...'")
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

            # Generate longer continuation with multiple methods
            max_gen_length = len(prompt_indices) + 120

            print("  Greedy decoding:")
            result_greedy = greedy_decode(model, src, src_mask, max_len=max_gen_length, start_symbol=prompt_indices[0])
            generated_indices_greedy = result_greedy[0].cpu().tolist()
            generated_text_greedy = ''.join([idx_to_char.get(idx, '?') for idx in generated_indices_greedy])
            print(f"    '{generated_text_greedy[:100]}...'")

            print("  Nucleus sampling (top-p=0.9):")
            result_nucleus = nucleus_sample_decode(model, src, src_mask, max_len=max_gen_length,
                                                 start_symbol=prompt_indices[0], temperature=0.7, top_p=0.9, repetition_penalty=1.2)
            generated_indices_nucleus = result_nucleus[0].cpu().tolist()
            generated_text_nucleus = ''.join([idx_to_char.get(idx, '?') for idx in generated_indices_nucleus])
            print(f"    '{generated_text_nucleus[:100]}...'")

            print("  Top-k sampling (k=20):")
            result_topk = sample_decode(model, src, src_mask, max_len=max_gen_length,
                                      start_symbol=prompt_indices[0], temperature=0.8, top_k=20)
            generated_indices_topk = result_topk[0].cpu().tolist()
            generated_text_topk = ''.join([idx_to_char.get(idx, '?') for idx in generated_indices_topk])
            print(f"    '{generated_text_topk[:100]}...'")

            # Calculate metrics for nucleus sampling result (usually best)
            unique_chars = len(set(generated_text_nucleus))
            print(f"  Unique characters (nucleus): {unique_chars}/{vocab_size}")
            print(f"  Length: {len(generated_text_nucleus)} characters")

            # Check for repetition patterns
            words = generated_text_nucleus.split()
            if len(words) > 1:
                unique_words = len(set(words))
                print(f"  Word diversity: {unique_words}/{len(words)} unique words")

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
        chars, char_to_idx, idx_to_char, data, vocab_size = create_shakespeare_data()
        train_shakespeare_task(chars, char_to_idx, idx_to_char, data, vocab_size)
    else:
        print("Training on copy task...")
        train_copy_task()
