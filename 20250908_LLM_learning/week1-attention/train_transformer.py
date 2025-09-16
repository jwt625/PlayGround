#!/usr/bin/env python3
"""
Comprehensive Training Script for Week 1 Transformer Implementation
Supports multiple datasets and comprehensive monitoring
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import argparse
import os
from typing import Dict, Any
import json

from annotated_transformer import make_model
from simple_training import TrainingMonitor, SimpleLossCompute, run_epoch, greedy_decode
from training_data import get_dataset


def create_config(dataset_name: str) -> Dict[str, Any]:
    """Create training configuration for different datasets"""
    
    base_config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_epochs': 15,
        'batch_size': 32,
        'learning_rate': 0.0003,
        'betas': (0.9, 0.98),
        'eps': 1e-9,
        'save_model': True,
        'visualize': True,
        'save_plots': True
    }
    
    dataset_configs = {
        'copy': {
            'vocab_size': 11,
            'seq_len': 10,
            'model_config': {'N': 2, 'd_model': 64, 'd_ff': 128, 'h': 4},
            'num_batches': 25,
            'description': 'Simple copy task - learn to copy input sequences'
        },
        'arithmetic': {
            'vocab_size': 15,
            'seq_len': 15,
            'model_config': {'N': 3, 'd_model': 128, 'd_ff': 256, 'h': 8},
            'num_batches': 50,
            'num_epochs': 25,
            'description': 'Arithmetic task - learn to add two numbers'
        },
        'sorting': {
            'vocab_size': 10,
            'seq_len': 8,
            'model_config': {'N': 2, 'd_model': 64, 'd_ff': 128, 'h': 4},
            'num_batches': 30,
            'description': 'Sorting task - learn to sort sequences'
        },
        'shakespeare': {
            'vocab_size': None,  # Will be set by dataset
            'seq_len': 64,
            'model_config': {'N': 4, 'd_model': 256, 'd_ff': 512, 'h': 8},
            'num_batches': 100,
            'num_epochs': 20,
            'batch_size': 16,
            'description': 'Character-level Shakespeare text generation'
        }
    }
    
    config = base_config.copy()
    config.update(dataset_configs.get(dataset_name, {}))
    config['dataset_name'] = dataset_name
    
    return config


def train_model(config: Dict[str, Any]) -> tuple:
    """Train transformer model with given configuration"""
    
    print(f"Training Configuration:")
    print(f"Dataset: {config['dataset_name']} - {config['description']}")
    print(f"Device: {config['device']}")
    print("=" * 60)
    
    # Create dataset
    dataset_kwargs = {}
    if config['dataset_name'] == 'copy':
        dataset_kwargs = {'vocab_size': config['vocab_size'], 'seq_len': config['seq_len']}
    elif config['dataset_name'] == 'arithmetic':
        dataset_kwargs = {'max_num': 99}
    elif config['dataset_name'] == 'sorting':
        dataset_kwargs = {'vocab_size': config['vocab_size'], 'seq_len': config['seq_len']}
    elif config['dataset_name'] == 'shakespeare':
        dataset_kwargs = {'seq_len': config['seq_len']}
    
    dataset = get_dataset(config['dataset_name'], **dataset_kwargs)
    
    # Update vocab size for datasets that determine it dynamically
    if hasattr(dataset, 'vocab_size'):
        config['vocab_size'] = dataset.vocab_size
    
    # Create model
    model_config = config['model_config']
    model = make_model(
        src_vocab=config['vocab_size'],
        tgt_vocab=config['vocab_size'],
        **model_config
    )
    
    # Move to device
    device = torch.device(config['device'])
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Vocabulary size: {config['vocab_size']}")
    
    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=config['learning_rate'],
        betas=config['betas'],
        eps=config['eps']
    )
    monitor = TrainingMonitor()
    
    # Training loop
    model.train()
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 40)
        
        # Generate data for this epoch
        data_iter = []
        for batch in dataset.generate_batches(config['batch_size'], config['num_batches']):
            batch.src = batch.src.to(device)
            batch.tgt = batch.tgt.to(device)
            batch.tgt_y = batch.tgt_y.to(device)
            batch.src_mask = batch.src_mask.to(device)
            batch.tgt_mask = batch.tgt_mask.to(device)
            data_iter.append(batch)
        
        # Train epoch
        loss_compute = SimpleLossCompute(model.generator, criterion, optimizer)
        avg_loss, avg_perplexity = run_epoch(data_iter, model, loss_compute, monitor, epoch)
        
        # Log metrics
        monitor.log_epoch(epoch, avg_loss, avg_perplexity)
        
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Perplexity: {avg_perplexity:.2f}")
    
    # Generate visualizations
    if config.get('visualize', True):
        print("\nGenerating training visualizations...")
        save_path = f"{config['dataset_name']}_training_curves.png" if config.get('save_plots') else None
        monitor.plot_training_curves(save_path)
    
    # Save model
    if config.get('save_model', True):
        model_path = f"{config['dataset_name']}_transformer_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'vocab_size': config['vocab_size']
        }, model_path)
        print(f"Model saved to {model_path}")
    
    return model, monitor, dataset


def test_model(model, dataset, config: Dict[str, Any], num_tests: int = 3):
    """Test the trained model"""
    print("\nTesting the trained model...")
    print("=" * 60)
    
    device = torch.device(config['device'])
    model.eval()
    
    if config['dataset_name'] == 'copy':
        test_sequences = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
            [1, 10, 9, 8, 7, 6, 5, 4, 3, 2]
        ]
        
        total_accuracy = 0
        for i, test_seq in enumerate(test_sequences[:num_tests]):
            src = torch.LongTensor([test_seq]).to(device)
            src_mask = torch.ones(1, 1, len(test_seq)).to(device)
            
            print(f"\nTest Case {i+1}:")
            print(f"Source: {test_seq}")
            
            result = greedy_decode(model, src, src_mask, max_len=len(test_seq), start_symbol=1)
            result_list = result[0].cpu().tolist()
            print(f"Result: {result_list}")
            
            accuracy = (src.cpu() == result.cpu()).float().mean().item()
            total_accuracy += accuracy
            print(f"Accuracy: {accuracy:.2%}")
        
        avg_accuracy = total_accuracy / num_tests
        print(f"\nOverall Test Accuracy: {avg_accuracy:.2%}")
        
    elif config['dataset_name'] == 'shakespeare':
        # Generate some text
        print("Generating text samples...")
        start_text = "ROMEO:"
        if hasattr(dataset, 'char_to_idx'):
            start_indices = [dataset.char_to_idx.get(c, 0) for c in start_text]
            src = torch.LongTensor([start_indices]).to(device)
            src_mask = torch.ones(1, 1, len(start_indices)).to(device)
            
            result = greedy_decode(model, src, src_mask, max_len=100, start_symbol=start_indices[0])
            generated_text = dataset.decode_sequence(result[0].cpu().tolist())
            print(f"Generated: {generated_text}")
    
    else:
        # Generic testing for other datasets
        print("Running generic tests...")
        test_batch = dataset.generate_batch(num_tests)
        test_batch.src = test_batch.src.to(device)
        test_batch.tgt = test_batch.tgt.to(device)
        
        for i in range(num_tests):
            src = test_batch.src[i:i+1]
            src_mask = torch.ones(1, 1, src.size(-1)).to(device)
            
            print(f"\nTest Case {i+1}:")
            print(f"Source: {src[0].cpu().tolist()}")
            print(f"Target: {test_batch.tgt[i].cpu().tolist()}")
            
            result = greedy_decode(model, src, src_mask, max_len=src.size(-1), start_symbol=1)
            print(f"Result: {result[0].cpu().tolist()}")


def main():
    parser = argparse.ArgumentParser(description='Train Transformer on various tasks')
    parser.add_argument('--dataset', type=str, default='copy',
                       choices=['copy', 'arithmetic', 'sorting', 'shakespeare'],
                       help='Dataset to train on')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--no-visualize', action='store_true', help='Disable visualizations')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save model')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_config(args.dataset)
    
    # Override with command line arguments
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.no_visualize:
        config['visualize'] = False
    if args.no_save:
        config['save_model'] = False
    
    # Train model
    model, monitor, dataset = train_model(config)
    
    # Test model
    test_model(model, dataset, config)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
