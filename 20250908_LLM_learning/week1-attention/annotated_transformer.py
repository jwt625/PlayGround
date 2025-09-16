"""
The Annotated Transformer
Implementation based on "Attention is All You Need" (Vaswani et al., 2017)

This is a clean, educational implementation of the Transformer model
optimized for learning and experimentation on dual H100 setup.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Callable, Tuple, List, Dict


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute 'Scaled Dot Product Attention'

    Args:
        query: [batch_size, h, seq_len, d_k]
        key: [batch_size, h, seq_len, d_k]
        value: [batch_size, h, seq_len, d_v]
        mask: [batch_size, 1, seq_len, seq_len] or [batch_size, 1, 1, seq_len]
        dropout: dropout layer

    Returns:
        Tuple of (attention output, attention weights)
    """
    d_k = query.size(-1)
    
    # Compute attention scores: QK^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided (set masked positions to large negative value)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax to get attention weights
    p_attn = F.softmax(scores, dim=-1)
    
    # Apply dropout if provided
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    # Apply attention weights to values
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism
    """
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        """
        Args:
            h: number of attention heads
            d_model: model dimension
            dropout: dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k: int = d_model // h
        self.h: int = h

        # Linear projections for Q, K, V and output
        self.linears: nn.ModuleList = clones(nn.Linear(d_model, d_model), 4)
        self.attn: Optional[torch.Tensor] = None
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or [batch_size, 1, seq_len]

        Returns:
            Multi-head attention output: [batch_size, seq_len, d_model]
        """
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        
        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1: nn.Linear = nn.Linear(d_model, d_ff)
        self.w_2: nn.Linear = nn.Linear(d_ff, d_model)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    """
    Layer Normalization
    """
    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2: nn.Parameter = nn.Parameter(torch.ones(features))
        self.b_2: nn.Parameter = nn.Parameter(torch.zeros(features))
        self.eps: float = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm: LayerNorm = LayerNorm(size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attention and feed forward
    """
    def __init__(
        self,
        size: int,
        self_attn: MultiHeadAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float
    ):
        super(EncoderLayer, self).__init__()
        self.self_attn: MultiHeadAttention = self_attn
        self.feed_forward: PositionwiseFeedForward = feed_forward
        self.sublayer: nn.ModuleList = clones(SublayerConnection(size, dropout), 2)
        self.size: int = size

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """
    def __init__(self, layer: EncoderLayer, N: int):
        """
        Args:
            layer: A single EncoderLayer instance to be cloned N times
            N: Number of encoder layers in the stack
        """
        super(Encoder, self).__init__()
        self.layers: nn.ModuleList = clones(layer, N)
        self.norm: LayerNorm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward
    """
    def __init__(
        self,
        size: int,
        self_attn: MultiHeadAttention,
        src_attn: MultiHeadAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float
    ):
        super(DecoderLayer, self).__init__()
        self.size: int = size
        self.self_attn: MultiHeadAttention = self_attn
        self.src_attn: MultiHeadAttention = src_attn
        self.feed_forward: PositionwiseFeedForward = feed_forward
        self.sublayer: nn.ModuleList = clones(SublayerConnection(size, dropout), 3)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking
    """
    def __init__(self, layer: DecoderLayer, N: int):
        """
        Args:
            layer: A single DecoderLayer instance to be cloned N times
            N: Number of decoder layers in the stack
        """
        super(Decoder, self).__init__()
        self.layers: nn.ModuleList = clones(layer, N)
        self.norm: LayerNorm = LayerNorm(layer.size)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Embeddings(nn.Module):
    """
    Standard embedding layer
    """
    def __init__(self, d_model: int, vocab: int):
        super(Embeddings, self).__init__()
        self.lut: nn.Embedding = nn.Embedding(vocab, d_model)
        self.d_model: int = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Implement the PE function using sine and cosine functions
    """
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step
    """
    def __init__(self, d_model: int, vocab: int):
        super(Generator, self).__init__()
        self.proj: nn.Linear = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture
    """
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Sequential,
        tgt_embed: nn.Sequential,
        generator: Generator
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.src_embed: nn.Sequential = src_embed
        self.tgt_embed: nn.Sequential = tgt_embed
        self.generator: Generator = generator

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model(
    src_vocab: int,
    tgt_vocab: int,
    N: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1
) -> EncoderDecoder:
    """
    Helper: Construct a model from hyperparameters

    Args:
        src_vocab: source vocabulary size
        tgt_vocab: target vocabulary size
        N: number of encoder/decoder layers
        d_model: model dimension
        d_ff: feed-forward dimension
        h: number of attention heads
        dropout: dropout rate

    Returns:
        Complete EncoderDecoder model
    """
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # Initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def subsequent_mask(size: int) -> torch.Tensor:
    """
    Mask out subsequent positions for decoder self-attention

    Args:
        size: sequence length

    Returns:
        Lower triangular mask tensor
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def visualize_attention(
    attention_weights: torch.Tensor,
    input_tokens: List[str],
    output_tokens: List[str],
    layer: int = 0,
    head: int = 0,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize attention weights as a heatmap

    Args:
        attention_weights: [batch, heads, seq_len, seq_len] attention weights
        input_tokens: List of input token strings
        output_tokens: List of output token strings
        layer: Which layer's attention to visualize
        head: Which attention head to visualize
        save_path: Optional path to save the plot
    """
    # Extract attention for specific head
    attn = attention_weights[0, head].detach().cpu().numpy()

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn,
        xticklabels=input_tokens,
        yticklabels=output_tokens,
        cmap='Blues',
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Attention Weight'}
    )

    plt.title(f'Attention Weights - Layer {layer}, Head {head}')
    plt.xlabel('Input Tokens (Keys)')
    plt.ylabel('Output Tokens (Queries)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_multi_head_attention(
    model: 'EncoderDecoder',
    src: torch.Tensor,
    tgt: torch.Tensor,
    src_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
    src_tokens: List[str],
    tgt_tokens: List[str],
    layer_idx: int = 0
) -> None:
    """
    Visualize all attention heads for a specific layer

    Args:
        model: The transformer model
        src: Source tensor
        tgt: Target tensor
        src_mask: Source mask
        tgt_mask: Target mask
        src_tokens: Source token strings
        tgt_tokens: Target token strings
        layer_idx: Which encoder/decoder layer to visualize
    """
    model.eval()
    with torch.no_grad():
        # Forward pass to get attention weights
        memory = model.encode(src, src_mask)
        _ = model.decode(memory, src_mask, tgt, tgt_mask)

        # Get encoder self-attention from specified layer
        encoder_attn = model.encoder.layers[layer_idx].self_attn.attn

        if encoder_attn is not None:
            num_heads = encoder_attn.size(1)
            fig, axes = plt.subplots(2, num_heads//2, figsize=(15, 8))
            axes = axes.flatten()

            for head in range(num_heads):
                attn = encoder_attn[0, head].cpu().numpy()

                sns.heatmap(
                    attn,
                    xticklabels=src_tokens,
                    yticklabels=src_tokens,
                    cmap='Blues',
                    ax=axes[head],
                    cbar=False,
                    square=True
                )
                axes[head].set_title(f'Head {head}')

            plt.suptitle(f'Encoder Self-Attention - Layer {layer_idx}')
            plt.tight_layout()
            plt.show()


def make_std_mask(tgt: torch.Tensor, pad: int) -> torch.Tensor:
    """
    Create a mask to hide padding and future words.

    Args:
        tgt: target tensor
        pad: padding token id

    Returns:
        Combined padding and subsequent mask
    """
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask


if __name__ == "__main__":
    # Example usage and testing
    print("Creating a small Transformer model...")

    # Small model for testing
    model = make_model(src_vocab=11, tgt_vocab=11, N=2, d_model=64, d_ff=128, h=4)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    tgt = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])

    src_mask = torch.ones(1, 1, 10)
    tgt_mask = make_std_mask(tgt, 0)

    print("Running forward pass...")
    out = model.forward(src, tgt, src_mask, tgt_mask)
    print(f"Output shape: {out.shape}")
    print("Forward pass successful!")

    # Test on GPU if available
    if torch.cuda.is_available():
        print(f"CUDA available with {torch.cuda.device_count()} GPUs")
        device = torch.device("cuda:0")
        model = model.to(device)
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        print("Running forward pass on GPU...")
        out = model.forward(src, tgt, src_mask, tgt_mask)
        print(f"GPU output shape: {out.shape}")
        print("GPU forward pass successful!")
