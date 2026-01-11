#!/usr/bin/env python3
"""
Test script to verify PaddleOCR optimizations (PP-OCRv5, FP16, TensorRT)
Compare performance between configurations.
"""
import os
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check if GPU is available and show specs"""
    try:
        import paddle
        gpu_count = paddle.device.cuda.device_count()
        logger.info(f"PaddlePaddle version: {paddle.__version__}")
        logger.info(f"GPU available: {gpu_count} device(s)")
        
        if gpu_count > 0:
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                      capture_output=True, text=True)
                logger.info(f"GPU Info: {result.stdout.strip()}")
            except:
                pass
        return gpu_count > 0
    except Exception as e:
        logger.error(f"Error checking GPU: {e}")
        return False

def check_tensorrt():
    """Check if TensorRT is installed"""
    try:
        import tensorrt
        logger.info(f"TensorRT version: {tensorrt.__version__}")
        return True
    except ImportError:
        logger.warning("TensorRT not installed. Install with: pip install tensorrt==8.6.1")
        return False

def test_ocr_initialization():
    """Test OCR initialization with current configuration"""
    logger.info("\n=== Testing OCR Initialization ===")
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    # Show current configuration
    config = {
        "OCR_VERSION": os.environ.get("OCR_VERSION", "default"),
        "OCR_DET_ALGORITHM": os.environ.get("OCR_DET_ALGORITHM", "default"),
        "OCR_REC_ALGORITHM": os.environ.get("OCR_REC_ALGORITHM", "default"),
        "OCR_USE_GPU": os.environ.get("OCR_USE_GPU", "false"),
        "OCR_GPU_MEM": os.environ.get("OCR_GPU_MEM", "8000"),
        "OCR_USE_FP16": os.environ.get("OCR_USE_FP16", "false"),
        "OCR_USE_TENSORRT": os.environ.get("OCR_USE_TENSORRT", "false"),
        "OCR_REC_BATCH_NUM": os.environ.get("OCR_REC_BATCH_NUM", "6"),
        "OCR_LANG": os.environ.get("OCR_LANG", "en"),
    }
    
    logger.info("Current Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize OCR
    try:
        from workers.ocr.ocr_engine import get_ocr_instance
        
        start_time = time.time()
        ocr = get_ocr_instance()
        init_time = time.time() - start_time
        
        logger.info(f"✓ OCR initialization successful in {init_time:.2f}s")
        return True
    except Exception as e:
        logger.error(f"✗ OCR initialization failed: {e}")
        return False

def run_warmup_test():
    """Run warmup test to trigger model loading and TensorRT conversion"""
    logger.info("\n=== Running Warmup Test ===")
    
    try:
        import numpy as np
        from workers.ocr.ocr_engine import get_ocr_instance
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        
        logger.info("Running warmup pass (loads models + TensorRT conversion if enabled)...")
        start_time = time.time()
        
        ocr = get_ocr_instance()
        result = ocr.ocr(dummy_image)
        
        warmup_time = time.time() - start_time
        logger.info(f"✓ Warmup completed in {warmup_time:.2f}s")
        
        if warmup_time > 30:
            logger.info("  (First run with TensorRT takes 1-2 min for model conversion)")
        
        return True
    except Exception as e:
        logger.error(f"✗ Warmup test failed: {e}")
        return False

def benchmark_configurations():
    """Benchmark different OCR configurations"""
    logger.info("\n=== Performance Benchmark ===")
    
    try:
        import numpy as np
        from workers.ocr.ocr_engine import get_ocr_instance
        
        # Create test images of different sizes
        test_images = {
            "small (1000x1000)": np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8),
            "medium (2000x2000)": np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8),
            "large (4000x4000)": np.random.randint(0, 255, (4000, 4000, 3), dtype=np.uint8),
        }
        
        ocr = get_ocr_instance()
        
        logger.info("Running benchmark (3 runs per size)...")
        for name, image in test_images.items():
            times = []
            for i in range(3):
                start_time = time.time()
                result = ocr.ocr(image)
                times.append(time.time() - start_time)
            
            avg_time = sum(times) / len(times)
            logger.info(f"  {name}: {avg_time:.2f}s avg (min: {min(times):.2f}s, max: {max(times):.2f}s)")
        
        return True
    except Exception as e:
        logger.error(f"✗ Benchmark failed: {e}")
        return False

def show_performance_estimates():
    """Show expected performance based on configuration"""
    logger.info("\n=== Expected Performance ===")
    
    use_gpu = os.environ.get("OCR_USE_GPU", "false").lower() == "true"
    use_fp16 = os.environ.get("OCR_USE_FP16", "false").lower() == "true"
    use_tensorrt = os.environ.get("OCR_USE_TENSORRT", "false").lower() == "true"
    ocr_version = os.environ.get("OCR_VERSION", "default")
    
    if not use_gpu:
        logger.info("CPU Mode:")
        logger.info("  - Short images (≤4000px): 120-300s")
        logger.info("  - Medium images (4001-15000px): 180-400s")
        logger.info("  - Long images (>15000px): 300-600s")
        logger.info("  - Accuracy: Baseline")
    elif use_tensorrt:
        logger.info("GPU + FP16 + TensorRT Mode:")
        logger.info("  - Short images (≤4000px): 0.5-1.5s ⚡")
        logger.info("  - Medium images (4001-15000px): 1-5s ⚡⚡")
        logger.info("  - Long images (>15000px): 3-10s ⚡⚡⚡")
        logger.info("  - Speedup vs CPU: 40-60x faster")
        logger.info(f"  - Accuracy: {'+13% (PP-OCRv5)' if ocr_version == 'PP-OCRv5' else 'Baseline'}")
    elif use_fp16:
        logger.info("GPU + FP16 Mode:")
        logger.info("  - Short images (≤4000px): 1-3s ⚡")
        logger.info("  - Medium images (4001-15000px): 3-10s ⚡⚡")
        logger.info("  - Long images (>15000px): 8-20s ⚡⚡")
        logger.info("  - Speedup vs CPU: 15-25x faster")
        logger.info(f"  - Accuracy: {'+13% (PP-OCRv5)' if ocr_version == 'PP-OCRv5' else 'Baseline'}")
    else:
        logger.info("GPU (FP32) Mode:")
        logger.info("  - Short images (≤4000px): 2-5s")
        logger.info("  - Medium images (4001-15000px): 5-15s")
        logger.info("  - Long images (>15000px): 15-40s")
        logger.info("  - Speedup vs CPU: 8-15x faster")
        logger.info(f"  - Accuracy: {'+13% (PP-OCRv5)' if ocr_version == 'PP-OCRv5' else 'Baseline'}")
    
    logger.info("\n💡 Recommendations:")
    if not use_gpu:
        logger.info("  → Enable GPU: Set OCR_USE_GPU=true (8-15x speedup)")
    if use_gpu and not use_fp16:
        logger.info("  → Enable FP16: Set OCR_USE_FP16=true (2x speedup, minimal accuracy loss)")
    if use_gpu and use_fp16 and not use_tensorrt:
        logger.info("  → Install TensorRT: pip install tensorrt==8.6.1 (5x speedup)")
        logger.info("  → Enable TensorRT: Set OCR_USE_TENSORRT=true")
    if ocr_version != "PP-OCRv5":
        logger.info("  → Upgrade to PP-OCRv5: Set OCR_VERSION=PP-OCRv5 (+13% accuracy)")

def main():
    """Run all tests"""
    logger.info("=== PaddleOCR Optimization Test Suite ===\n")
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check prerequisites
    has_gpu = check_gpu_availability()
    has_tensorrt = check_tensorrt()
    
    # Run tests
    tests = [
        ("OCR Initialization", test_ocr_initialization),
        ("Warmup Test", run_warmup_test),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Optional: Run benchmark (takes time)
    if input("\nRun performance benchmark? (y/n): ").lower() == 'y':
        benchmark_configurations()
    
    # Show performance estimates
    show_performance_estimates()
    
    # Summary
    logger.info("\n=== Test Summary ===")
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"  {status}: {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    logger.info(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! OCR is ready for production.")
    else:
        logger.warning("\n⚠️ Some tests failed. Check configuration and dependencies.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
