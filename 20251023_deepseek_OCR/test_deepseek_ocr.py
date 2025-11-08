#!/usr/bin/env python3
"""
DeepSeek-OCR Test Script using vLLM

This script tests the DeepSeek-OCR model with various OCR tasks.
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from PIL import Image
import torch

def test_vllm_import():
    """Test if vLLM is properly installed."""
    print("=" * 80)
    print("TEST 1: Checking vLLM Installation")
    print("=" * 80)
    try:
        import vllm
        print(f"✓ vLLM version: {vllm.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import vLLM: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability."""
    print("\n" + "=" * 80)
    print("TEST 2: GPU Availability")
    print("=" * 80)
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  - GPU Count: {torch.cuda.device_count()}")
        print(f"  - Current GPU: {torch.cuda.current_device()}")
        print(f"  - GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("✗ CUDA is not available")
        return False

def test_model_loading():
    """Test loading the DeepSeek-OCR model."""
    print("\n" + "=" * 80)
    print("TEST 3: Loading DeepSeek-OCR Model")
    print("=" * 80)
    try:
        from vllm import LLM

        print("Loading model: deepseek-ai/DeepSeek-OCR...")
        start_time = time.time()

        llm = LLM(
            model="deepseek-ai/DeepSeek-OCR",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.7,
            trust_remote_code=True,
        )

        load_time = time.time() - start_time
        print(f"✓ Model loaded successfully in {load_time:.2f} seconds")
        return llm
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_ocr_inference(llm):
    """Test OCR inference on test images."""
    print("\n" + "=" * 80)
    print("TEST 4: OCR Inference")
    print("=" * 80)

    test_images_dir = Path("test_images")
    if not test_images_dir.exists():
        print(f"✗ Test images directory not found: {test_images_dir}")
        return False

    test_images = list(test_images_dir.glob("*.png"))
    if not test_images:
        print(f"✗ No test images found in {test_images_dir}")
        return False

    print(f"Found {len(test_images)} test images")

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("test_results") / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # Track success/failure for each image
    successful_inferences = 0
    failed_inferences = 0

    # Store all results for summary
    all_results = []

    try:
        from vllm import SamplingParams

        for img_path in test_images:
            print(f"\nProcessing: {img_path.name}")

            # Initialize result record
            result_record = {
                "image_name": img_path.name,
                "image_path": str(img_path),
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": None,
                "ocr_text": None,
                "inference_time_seconds": None,
                "image_size": None,
                "image_mode": None,
            }

            # Verify image exists and is readable
            if not img_path.exists():
                print(f"  ✗ Image file not found")
                result_record["error"] = "Image file not found"
                failed_inferences += 1
                all_results.append(result_record)
                continue

            try:
                img = Image.open(img_path)
                result_record["image_size"] = list(img.size)
                result_record["image_mode"] = img.mode
                print(f"  - Image size: {img.size}")
                print(f"  - Image mode: {img.mode}")
            except Exception as e:
                print(f"  ✗ Failed to open image: {e}")
                result_record["error"] = f"Failed to open image: {str(e)}"
                failed_inferences += 1
                all_results.append(result_record)
                continue

            # Test basic OCR prompt
            prompt_text = "<image>\nFree OCR."
            print(f"  - Prompt: {prompt_text}")

            try:
                sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=512,
                )

                # Create prompt with multi-modal data (correct vLLM API)
                prompt_dict = {
                    "prompt": prompt_text,
                    "multi_modal_data": {"image": img}
                }

                start_time = time.time()
                outputs = llm.generate(
                    [prompt_dict],
                    sampling_params=sampling_params,
                )
                inference_time = time.time() - start_time

                if outputs and len(outputs) > 0:
                    result = outputs[0].outputs[0].text
                    result_record["status"] = "success"
                    result_record["ocr_text"] = result
                    result_record["inference_time_seconds"] = inference_time

                    print(f"  ✓ OCR completed in {inference_time:.2f} seconds")
                    print(f"  - Result preview: {result[:100]}...")

                    # Save individual result to text file
                    output_file = results_dir / f"{img_path.stem}_ocr.txt"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(f"Image: {img_path.name}\n")
                        f.write(f"Size: {img.size}\n")
                        f.write(f"Inference Time: {inference_time:.2f}s\n")
                        f.write(f"Timestamp: {result_record['timestamp']}\n")
                        f.write("=" * 80 + "\n")
                        f.write(result)
                    print(f"  - Saved to: {output_file}")

                    successful_inferences += 1
                else:
                    print(f"  ✗ No output from model")
                    result_record["error"] = "No output from model"
                    failed_inferences += 1
            except Exception as e:
                print(f"  ✗ Inference failed: {e}")
                result_record["error"] = str(e)
                import traceback
                traceback.print_exc()
                failed_inferences += 1

            all_results.append(result_record)

        # Print summary
        print(f"\n  Summary: {successful_inferences} successful, {failed_inferences} failed out of {len(test_images)} total")

        # Save summary JSON
        summary = {
            "test_run_timestamp": timestamp,
            "total_images": len(test_images),
            "successful": successful_inferences,
            "failed": failed_inferences,
            "results": all_results
        }

        summary_file = results_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  Summary saved to: {summary_file}")

        # Save summary markdown
        markdown_file = results_dir / "summary.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(f"# OCR Test Results\n\n")
            f.write(f"**Test Run**: {timestamp}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- **Total Images**: {len(test_images)}\n")
            f.write(f"- **Successful**: {successful_inferences}\n")
            f.write(f"- **Failed**: {failed_inferences}\n\n")
            f.write(f"## Individual Results\n\n")

            for result in all_results:
                f.write(f"### {result['image_name']}\n\n")
                f.write(f"- **Status**: {result['status']}\n")
                if result['image_size']:
                    f.write(f"- **Size**: {result['image_size'][0]}x{result['image_size'][1]}\n")
                if result['inference_time_seconds']:
                    f.write(f"- **Inference Time**: {result['inference_time_seconds']:.2f}s\n")
                if result['error']:
                    f.write(f"- **Error**: {result['error']}\n")
                if result['ocr_text']:
                    f.write(f"\n**OCR Output**:\n```\n{result['ocr_text']}\n```\n")
                f.write("\n---\n\n")

        print(f"  Markdown summary saved to: {markdown_file}")

        # Return True only if at least one inference succeeded
        return successful_inferences > 0
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n")
    print("*" * 80)
    print("DeepSeek-OCR Test Suite")
    print("*" * 80)
    
    results = {}
    
    # Test 1: vLLM import
    results['vllm_import'] = test_vllm_import()
    
    # Test 2: GPU availability
    results['gpu_available'] = test_gpu_availability()
    
    # Test 3: Model loading
    llm = test_model_loading()
    results['model_loading'] = llm is not None
    
    # Test 4: OCR inference
    if llm:
        results['ocr_inference'] = test_ocr_inference(llm)
    else:
        results['ocr_inference'] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. Please review the output above.")
    print("=" * 80 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

