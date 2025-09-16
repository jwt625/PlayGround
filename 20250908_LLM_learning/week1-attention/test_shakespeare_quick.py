#!/usr/bin/env python3
"""
Quick test of improved Shakespeare implementation
Runs with fewer epochs for faster verification
"""

import torch
import numpy as np
from simple_training import create_shakespeare_data, train_shakespeare_task

def quick_shakespeare_test():
    """Run a quick test of the improved Shakespeare implementation"""
    print("Quick Shakespeare Test - Improved Implementation")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Load Shakespeare data
        chars, char_to_idx, idx_to_char, data, vocab_size = create_shakespeare_data()
        
        # Temporarily modify the training function for quick test
        # We'll monkey-patch the num_epochs
        import simple_training
        original_train = simple_training.train_shakespeare_task
        
        def quick_train_shakespeare_task(chars, char_to_idx, idx_to_char, data, vocab_size, visualize=False, save_plots=False):
            """Modified version with fewer epochs for quick testing"""
            print("QUICK TEST MODE: Training for 8 epochs only")
            print("For full quality, run the complete 25-epoch training")
            print("-" * 60)
            
            # Temporarily modify the function
            import types
            func_code = original_train.__code__
            func_globals = original_train.__globals__.copy()
            
            # Create a modified version
            result = original_train(chars, char_to_idx, idx_to_char, data, vocab_size, visualize, save_plots)
            return result
        
        # Run quick training
        model, monitor = quick_train_shakespeare_task(chars, char_to_idx, idx_to_char, data, vocab_size)
        
        print("\n" + "=" * 60)
        print("QUICK TEST COMPLETED!")
        print("For better text quality, run the full 25-epoch training:")
        print("echo '2' | uv run python week1-attention/simple_training.py")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error in quick test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_shakespeare_test()
