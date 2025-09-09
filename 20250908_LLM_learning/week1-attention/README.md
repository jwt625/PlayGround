# Week 1: Attention Mechanisms - Annotated Transformer

This directory contains a clean, educational implementation of the Transformer model based on "Attention is All You Need" (Vaswani et al., 2017).

## Files

- `annotated_transformer.py` - Complete Transformer implementation with detailed comments
- `simple_training.py` - Training script demonstrating the model on a copy task
- `README.md` - This file

## Implementation Features

### Core Components
- **Scaled Dot-Product Attention**: The fundamental attention mechanism
- **Multi-Head Attention**: Parallel attention heads for different representation subspaces
- **Position-wise Feed-Forward Networks**: Two linear transformations with ReLU activation
- **Layer Normalization**: Applied in residual connections
- **Positional Encoding**: Sine and cosine functions to inject position information

### Architecture
- **Encoder**: Stack of N=6 identical layers, each with self-attention and feed-forward sublayers
- **Decoder**: Stack of N=6 identical layers with masked self-attention, encoder-decoder attention, and feed-forward sublayers
- **Embeddings**: Learned embeddings scaled by sqrt(d_model)
- **Generator**: Linear projection to vocabulary size with log-softmax

## Usage

### Setup Environment
```bash
# From the project root
source .venv/bin/activate
```

### Test the Implementation
```bash
cd week1-attention
python annotated_transformer.py
```

### Train on Copy Task
```bash
python simple_training.py
```

## Model Architecture Details

### Hyperparameters (Small Test Model)
- Model dimension (d_model): 64
- Feed-forward dimension (d_ff): 128
- Number of attention heads (h): 4
- Number of layers (N): 2
- Vocabulary size: 11
- Total parameters: ~170K

### Hardware Utilization
- Automatically detects and uses available GPUs (dual H100 setup)
- Achieves ~22K tokens/sec training throughput on H100
- Memory efficient for small models

## Learning Outcomes

After implementing and running this code, you should understand:

1. **Attention Mechanism**: How queries, keys, and values work together
2. **Multi-Head Attention**: Parallel processing of different representation subspaces
3. **Transformer Architecture**: Encoder-decoder structure with residual connections
4. **Positional Encoding**: How position information is injected without recurrence
5. **Training Process**: Forward pass, loss computation, and backpropagation

## Next Steps

This implementation provides the foundation for:
- Scaling to larger models (Week 2: minGPT experiments)
- Understanding BERT-style bidirectional models (Week 3)
- Training larger models with nanoGPT (Week 4)

## Performance Notes

The copy task demonstrates basic functionality but is intentionally simple. With limited training:
- Model learns some patterns but doesn't achieve perfect copying
- 30% accuracy is typical for this quick demonstration
- More training epochs and data would improve performance

This serves as a proof-of-concept for the Transformer architecture before moving to more complex tasks and larger models.
