#!/usr/bin/env python3
"""
Demo script to test tiling OCR on a local image file.
Shows the difference between standard OCR and tiling+two-pass OCR.

Usage:
    python test_tiling_demo.py path/to/image.jpg
"""
import sys
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Set debug mode
os.environ["OCR_DEBUG_DIR"] = "./debug_output"

from workers.ocr.ocr_engine import run_ocr, run_ocr_with_tiling


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_tiling_demo.py <image_path>")
        print("\nExample: python test_tiling_demo.py sample_webtoon.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)
    
    print(f"Testing OCR on: {image_path}")
    print("=" * 60)
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # Test 1: Standard OCR
    print("\n[1] Standard OCR (no tiling)...")
    start = time.time()
    lines_standard = run_ocr(image_bytes)
    elapsed_standard = time.time() - start
    
    print(f"   Detected: {len(lines_standard)} lines")
    print(f"   Time: {elapsed_standard:.2f}s")
    
    if lines_standard:
        print(f"   Sample texts:")
        for line in lines_standard[:5]:
            print(f"      - '{line['text']}' (conf: {line['confidence']:.2f})")
    
    # Test 2: Tiling OCR with two-pass
    print("\n[2] Tiling OCR with two-pass enhancement...")
    start = time.time()
    lines_tiling = run_ocr_with_tiling(
        image_bytes,
        tile_height=1400,
        overlap=200,
        use_two_pass=True,
        debug_dir="./debug_output"
    )
    elapsed_tiling = time.time() - start
    
    print(f"   Detected: {len(lines_tiling)} lines")
    print(f"   Time: {elapsed_tiling:.2f}s")
    
    if lines_tiling:
        print(f"   Sample texts:")
        for line in lines_tiling[:5]:
            print(f"      - '{line['text']}' (conf: {line['confidence']:.2f})")
    
    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON:")
    print(f"  Standard OCR:  {len(lines_standard)} lines in {elapsed_standard:.2f}s")
    print(f"  Tiling OCR:    {len(lines_tiling)} lines in {elapsed_tiling:.2f}s")
    
    improvement = len(lines_tiling) - len(lines_standard)
    improvement_pct = (improvement / len(lines_standard) * 100) if len(lines_standard) > 0 else 0
    
    print(f"  Improvement:   +{improvement} lines ({improvement_pct:+.1f}%)")
    print(f"  Time overhead: {elapsed_tiling - elapsed_standard:.2f}s ({elapsed_tiling/elapsed_standard:.1f}x)")
    
    print("\nDebug images saved to: ./debug_output/")
    print("  - tile_000.jpg, tile_001.jpg, ... (individual tiles with boxes)")
    print("  - full_page.jpg (final result with all boxes)")
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
