#!/usr/bin/env python3
"""
Download and cache the Qwen-Image-Edit-2509 model locally.
"""

import os
import torch
from diffusers import QwenImageEditPlusPipeline
from huggingface_hub import snapshot_download


def main():
    print("=" * 60)
    print("Downloading Qwen-Image-Edit-2509 Model")
    print("=" * 60)
    
    model_id = "Qwen/Qwen-Image-Edit-2509"
    
    # Set cache directory (optional - uses default HF cache if not set)
    cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"\nCache directory: {cache_dir}")
    
    # Check GPU
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Download model files using snapshot_download
    print(f"\nDownloading model files from {model_id}...")
    print("This may take several minutes depending on your connection...")
    
    try:
        local_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            resume_download=True,
        )
        print(f"\nModel files downloaded to: {local_dir}")
        
    except Exception as e:
        print(f"\nERROR downloading model files: {e}")
        return 1
    
    # Load the pipeline to ensure everything works
    print("\nLoading pipeline to verify download...")
    
    try:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )
        print("Pipeline loaded successfully!")
        
        # Move to GPU to verify it works
        if torch.cuda.is_available():
            print("\nMoving model to GPU...")
            pipeline.to('cuda:0')
            print("Model successfully loaded on GPU!")
            
            # Print memory usage
            print(f"\nGPU Memory Usage:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {allocated:.2f} GB allocated / {reserved:.2f} GB reserved / {total:.2f} GB total")
        
        # Cleanup
        del pipeline
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"\nERROR loading pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 60)
    print("Model Download Complete!")
    print("=" * 60)
    print("\nThe model is now cached locally and ready to use.")
    print("You can now run:")
    print("  - test_model.py to test the model")
    print("  - server.py to start the inference server")
    
    return 0


if __name__ == "__main__":
    exit(main())

