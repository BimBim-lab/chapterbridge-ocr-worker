# OCR Worker - Tiling & Two-Pass Enhancement

## Overview

OCR worker telah di-upgrade dengan fitur **tiling** dan **two-pass OCR** untuk meningkatkan recall detection, khususnya pada gambar webtoon/manhwa yang sangat panjang (vertikal).

### Fitur Baru

1. **Tiling (Image Slicing)**
   - Gambar panjang dipotong menjadi tile-tile horizontal dengan overlap
   - Setiap tile diproses terpisah untuk menghindari limitation PaddleOCR pada gambar besar
   - Koordinat bbox otomatis disesuaikan ke posisi global
   - Default: tile_height=1400px, overlap=200px

2. **Two-Pass OCR**
   - Pass 1: OCR pada gambar original
   - Pass 2: OCR pada gambar dengan CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Meningkatkan deteksi text tipis/putih yang sulit terbaca
   - Hasil dari kedua pass digabung dan dideduplikasi

3. **Deduplication**
   - Menghilangkan deteksi duplikat berdasarkan IoU (Intersection over Union)
   - Mempertimbangkan text similarity untuk memastikan duplikasi
   - Memilih deteksi dengan confidence lebih tinggi
   - Default: IoU threshold=0.5, text similarity=0.7

4. **Debug Visualization**
   - Opsional: simpan gambar dengan bounding boxes untuk debugging
   - Per-tile dan full-page visualization
   - Berguna untuk menganalisis missed detections

5. **Auto-Detection Mode**
   - Worker secara otomatis memilih mode standard vs tiling berdasarkan ukuran gambar
   - Default: tiling untuk gambar dengan height > 2000px

## Configuration

Semua parameter dapat dikonfigurasi via `.env`:

```env
# Detection thresholds
OCR_DET_THRESH=0.15              # Lower = more sensitive
OCR_DET_BOX_THRESH=0.35          # Lower = more boxes
OCR_DET_UNCLIP_RATIO=2.5         # Higher = larger boxes
OCR_REC_SCORE_THRESH=0.3         # Min confidence threshold

# Tiling configuration
OCR_TILE_HEIGHT=1400             # Height per tile (pixels)
OCR_TILE_OVERLAP=200             # Overlap between tiles (pixels)
OCR_TWO_PASS=true                # Enable/disable two-pass OCR
OCR_USE_TILING=auto              # auto | always | never

# Deduplication
OCR_IOU_THRESHOLD=0.5            # IoU threshold for duplicate detection
OCR_TEXT_SIM_THRESHOLD=0.7       # Text similarity threshold

# Debug (uncomment to enable)
# OCR_DEBUG_DIR=./debug_output
```

## Usage

### 1. Standard Worker (Auto Mode)

Worker akan otomatis memilih tiling untuk gambar tinggi:

```bash
python workers/ocr/main.py --poll-seconds 3
```

### 2. Force Tiling Mode

Selalu gunakan tiling untuk semua gambar:

```bash
OCR_USE_TILING=always python workers/ocr/main.py --poll-seconds 3
```

### 3. Disable Tiling

Gunakan standard OCR tanpa tiling:

```bash
OCR_USE_TILING=never python workers/ocr/main.py --poll-seconds 3
```

### 4. Test with Demo Script

Test tiling pada image lokal dan lihat perbandingannya:

```bash
python test_tiling_demo.py path/to/webtoon_image.jpg
```

Output akan menunjukkan:
- Jumlah deteksi standard vs tiling
- Processing time comparison
- Sample text detections
- Debug images di `./debug_output/`

## Expected Results

### Performance Impact

- **Processing Time**: 1.5-3x lebih lambat (tergantung jumlah tile dan two-pass)
- **Detection Improvement**: 30-60% lebih banyak text terdeteksi
- **Recall**: Signifikan lebih baik pada:
  - Text tipis/putih
  - Small text di speech bubbles
  - Text di bagian atas/bawah gambar panjang

### Example Output

```
Standard OCR:  46 lines in 12.3s
Tiling OCR:    73 lines in 28.7s
Improvement:   +27 lines (+58.7%)
Time overhead: 16.4s (2.3x)
```

## Tuning Tips

### Untuk Recall Lebih Tinggi

```env
OCR_DET_THRESH=0.10              # Lower threshold
OCR_DET_BOX_THRESH=0.30          # More sensitive box detection
OCR_REC_SCORE_THRESH=0.2         # Accept lower confidence
OCR_TILE_HEIGHT=1200             # Smaller tiles (more overlap coverage)
OCR_TILE_OVERLAP=250             # More overlap
```

### Untuk Speed (trade-off)

```env
OCR_TWO_PASS=false               # Disable second pass
OCR_TILE_HEIGHT=1600             # Larger tiles (fewer tiles)
OCR_TILE_OVERLAP=150             # Less overlap
OCR_USE_TILING=never             # Disable tiling entirely
```

### Untuk Quality Balance (recommended)

```env
OCR_DET_THRESH=0.15
OCR_DET_BOX_THRESH=0.35
OCR_REC_SCORE_THRESH=0.3
OCR_TILE_HEIGHT=1400
OCR_TILE_OVERLAP=200
OCR_TWO_PASS=true
OCR_USE_TILING=auto
```

## Debugging

### Enable Debug Output

```env
OCR_DEBUG_DIR=./debug_output
```

Worker akan menyimpan:
- `tile_000.jpg`, `tile_001.jpg`, ... - Setiap tile dengan bounding boxes
- `full_page.jpg` - Full page dengan semua deteksi

### Check Logs

Worker akan log:
```
INFO Processing image 800x5000, tiling=1400px, overlap=200px, two_pass=True
DEBUG Image 800x5000 tiled into 4 tiles
DEBUG Processing tile 1/4 (y=0-1400)
DEBUG OCR original pass: 12 detections
DEBUG OCR enhanced pass: 8 detections
...
INFO Total raw detections: 65 from 4 tiles
DEBUG Deduplication: 65 -> 58 lines (removed 7 duplicates)
INFO OCR complete: 58 final detections in 24.3s (6.1s/tile)
```

## Architecture

### Function Flow

```
run_ocr_with_tiling()
  ├─ tile_image()                    # Slice to overlapping tiles
  │
  ├─ For each tile:
  │   ├─ _run_ocr_on_image()         # Pass 1: Original
  │   └─ enhance_contrast()          # Pass 2: CLAHE
  │       └─ _run_ocr_on_image()
  │
  ├─ deduplicate_boxes()             # Remove duplicates (IoU + text sim)
  ├─ Sort by (y, x)                  # Reading order
  └─ draw_debug_boxes()              # Optional visualization
```

### Backward Compatibility

Original `run_ocr()` function tetap ada dan tidak berubah:

```python
# Old code still works
lines = run_ocr(image_bytes)

# New enhanced version
lines = run_ocr_with_tiling(image_bytes)
```

Worker `main.py` menggunakan auto-detection mode untuk memilih fungsi yang tepat.

## Dependencies

New dependency added:

```
opencv-python>=4.8.0  # For CLAHE contrast enhancement
```

Install via:

```bash
pip install opencv-python>=4.8.0
```

## Known Limitations

1. **Memory Usage**: Large tiles consume more memory. Adjust `OCR_TILE_HEIGHT` jika OOM.
2. **Boundary Effects**: Text yang terpotong di tile boundary bisa miss → solved by overlap.
3. **Processing Time**: Two-pass dengan banyak tile bisa lambat → configurable.
4. **IoU Deduplication**: Tidak 100% perfect, bisa ada false negative/positive.

## Migration Guide

### For Existing Deployments

1. Update `.env` dengan parameter baru (atau gunakan default)
2. Install `opencv-python` dependency
3. Test dengan demo script
4. Deploy worker (backward compatible)

No breaking changes. Existing code akan tetap bekerja dengan auto-mode.

---

**Questions?** Check logs untuk detailed debug info atau enable `OCR_DEBUG_DIR` untuk visual inspection.
