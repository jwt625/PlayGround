"""
Training Data Utilities for Week 1 Attention Implementation
Provides various datasets for testing the Transformer model
"""

import torch
import numpy as np
import urllib.request
import os
from collections import Counter
from typing import List, Tuple, Dict, Iterator
from simple_training import Batch


class CopyTaskDataset:
    """Simple copy task dataset for basic testing"""
    
    def __init__(self, vocab_size: int = 11, seq_len: int = 10):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
    
    def generate_batch(self, batch_size: int) -> Batch:
        """Generate a single batch of copy task data"""
        data = torch.from_numpy(np.random.randint(1, self.vocab_size, size=(batch_size, self.seq_len)))
        data[:, 0] = 1  # Start token
        src = data.clone().detach()
        tgt = data.clone().detach()
        return Batch(src, tgt, 0)
    
    def generate_batches(self, batch_size: int, num_batches: int) -> Iterator[Batch]:
        """Generate multiple batches"""
        for _ in range(num_batches):
            yield self.generate_batch(batch_size)


class ArithmeticDataset:
    """Simple arithmetic dataset: learn to add two numbers"""
    
    def __init__(self, max_num: int = 99):
        self.max_num = max_num
        self.vocab_size = 15  # 0-9, +, =, <pad>, <start>, <end>
        self.char_to_idx = {
            '<pad>': 0, '<start>': 1, '<end>': 2,
            '0': 3, '1': 4, '2': 5, '3': 6, '4': 7,
            '5': 8, '6': 9, '7': 10, '8': 11, '9': 12,
            '+': 13, '=': 14
        }
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
    
    def number_to_tokens(self, num: int) -> List[int]:
        """Convert number to token indices"""
        return [self.char_to_idx[d] for d in str(num)]
    
    def generate_example(self) -> Tuple[List[int], List[int]]:
        """Generate a single arithmetic example"""
        a = np.random.randint(1, self.max_num)
        b = np.random.randint(1, self.max_num)
        result = a + b
        
        # Create input: <start> a + b =
        src = [self.char_to_idx['<start>']]
        src.extend(self.number_to_tokens(a))
        src.append(self.char_to_idx['+'])
        src.extend(self.number_to_tokens(b))
        src.append(self.char_to_idx['='])
        
        # Create target: <start> result <end>
        tgt = [self.char_to_idx['<start>']]
        tgt.extend(self.number_to_tokens(result))
        tgt.append(self.char_to_idx['<end>'])
        
        return src, tgt
    
    def generate_batch(self, batch_size: int, max_len: int = 15) -> Batch:
        """Generate a batch of arithmetic examples"""
        src_batch = []
        tgt_batch = []
        
        for _ in range(batch_size):
            src, tgt = self.generate_example()
            
            # Pad sequences
            src_padded = src + [0] * (max_len - len(src))
            tgt_padded = tgt + [0] * (max_len - len(tgt))
            
            src_batch.append(src_padded[:max_len])
            tgt_batch.append(tgt_padded[:max_len])
        
        src_tensor = torch.LongTensor(src_batch)
        tgt_tensor = torch.LongTensor(tgt_batch)
        
        return Batch(src_tensor, tgt_tensor, 0)
    
    def generate_batches(self, batch_size: int, num_batches: int) -> Iterator[Batch]:
        """Generate multiple batches"""
        for _ in range(num_batches):
            yield self.generate_batch(batch_size)


class TinyShakespeareDataset:
    """Tiny Shakespeare dataset for character-level language modeling"""
    
    def __init__(self, file_path: str = "tinyshakespeare.txt", seq_len: int = 64):
        self.file_path = file_path
        self.seq_len = seq_len
        self.text = self._download_and_load()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.data = [self.char_to_idx[ch] for ch in self.text]
        
        print(f"Dataset loaded: {len(self.text)} characters, {self.vocab_size} unique")
    
    def _download_and_load(self) -> str:
        """Download and load the dataset"""
        if not os.path.exists(self.file_path):
            print("Downloading Tiny Shakespeare dataset...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, self.file_path)
            print("Download complete!")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def generate_batch(self, batch_size: int) -> Batch:
        """Generate a batch of sequences"""
        batch_data = []
        
        for _ in range(batch_size):
            # Random starting position
            start_idx = np.random.randint(0, len(self.data) - self.seq_len - 1)
            seq = self.data[start_idx:start_idx + self.seq_len + 1]
            batch_data.append(seq)
        
        # Convert to tensors
        batch_tensor = torch.LongTensor(batch_data)
        src = batch_tensor[:, :-1]  # Input sequence
        tgt = batch_tensor[:, 1:]   # Target sequence (shifted by 1)
        
        return Batch(src, tgt, 0)
    
    def generate_batches(self, batch_size: int, num_batches: int) -> Iterator[Batch]:
        """Generate multiple batches"""
        for _ in range(num_batches):
            yield self.generate_batch(batch_size)
    
    def decode_sequence(self, indices: List[int]) -> str:
        """Convert indices back to text"""
        return ''.join([self.idx_to_char[i] for i in indices])


class SortingDataset:
    """Dataset for learning to sort sequences"""
    
    def __init__(self, vocab_size: int = 10, seq_len: int = 8):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
    
    def generate_batch(self, batch_size: int) -> Batch:
        """Generate a batch of sorting examples"""
        src_batch = []
        tgt_batch = []
        
        for _ in range(batch_size):
            # Generate random sequence
            seq = np.random.randint(1, self.vocab_size, size=self.seq_len)
            sorted_seq = np.sort(seq)
            
            src_batch.append(seq.tolist())
            tgt_batch.append(sorted_seq.tolist())
        
        src_tensor = torch.LongTensor(src_batch)
        tgt_tensor = torch.LongTensor(tgt_batch)
        
        return Batch(src_tensor, tgt_tensor, 0)
    
    def generate_batches(self, batch_size: int, num_batches: int) -> Iterator[Batch]:
        """Generate multiple batches"""
        for _ in range(num_batches):
            yield self.generate_batch(batch_size)


def get_dataset(dataset_name: str, **kwargs):
    """Factory function to get datasets"""
    datasets = {
        'copy': CopyTaskDataset,
        'arithmetic': ArithmeticDataset,
        'shakespeare': TinyShakespeareDataset,
        'sorting': SortingDataset
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}")
    
    return datasets[dataset_name](**kwargs)


if __name__ == "__main__":
    # Test different datasets
    print("Testing datasets...")
    
    # Test copy task
    print("\n1. Copy Task Dataset:")
    copy_dataset = get_dataset('copy', vocab_size=11, seq_len=10)
    batch = copy_dataset.generate_batch(2)
    print(f"Source: {batch.src}")
    print(f"Target: {batch.tgt}")
    
    # Test arithmetic
    print("\n2. Arithmetic Dataset:")
    arith_dataset = get_dataset('arithmetic', max_num=50)
    batch = arith_dataset.generate_batch(2)
    print(f"Source: {batch.src}")
    print(f"Target: {batch.tgt}")
    
    # Test sorting
    print("\n3. Sorting Dataset:")
    sort_dataset = get_dataset('sorting', vocab_size=10, seq_len=6)
    batch = sort_dataset.generate_batch(2)
    print(f"Source: {batch.src}")
    print(f"Target: {batch.tgt}")
    
    print("\nAll datasets working correctly!")
