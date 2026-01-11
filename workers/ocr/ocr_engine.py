"""
PaddleOCR engine wrapper with adaptive tiling and selective two-pass OCR for optimal speed/recall.
"""
import os
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

_ocr_instance = None

# Adaptive OCR configuration
DEBUG_MODE = os.environ.get("OCR_DEBUG", "0") == "1"
DEBUG_DIR = os.environ.get("OCR_DEBUG_DIR", "debug")

def get_ocr_instance():
    """
    Get or create singleton PaddleOCR instance with configurable parameters.
    All detection thresholds can be overridden via environment variables.
    """
    global _ocr_instance
    
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        
        lang = os.environ.get("OCR_LANG", "en")
        use_angle_cls = os.environ.get("OCR_USE_ANGLE_CLS", "true").lower() == "true"
        det_limit_side_len = int(os.environ.get("OCR_DET_LIMIT_SIDE_LEN", "2560"))
        det_thresh = float(os.environ.get("OCR_DET_THRESH", "0.15"))
        det_box_thresh = float(os.environ.get("OCR_DET_BOX_THRESH", "0.35"))
        det_unclip_ratio = float(os.environ.get("OCR_DET_UNCLIP_RATIO", "2.5"))
        
        _ocr_instance = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            text_det_limit_side_len=det_limit_side_len,
            text_det_limit_type='max',
            text_det_thresh=det_thresh,
            text_det_box_thresh=det_box_thresh,
            text_det_unclip_ratio=det_unclip_ratio
        )
        
        logger.info(f"PaddleOCR initialized: lang={lang}, det_thresh={det_thresh}, "
                   f"box_thresh={det_box_thresh}, unclip={det_unclip_ratio}")
    
    return _ocr_instance


def tile_image(image: Image.Image, tile_height: int = 1400, overlap: int = 200) -> List[Tuple[Image.Image, int, int]]:
    """
    Slice a tall image into overlapping horizontal tiles.
    
    Args:
        image: PIL Image to tile
        tile_height: Height of each tile in pixels
        overlap: Overlap between tiles in pixels
        
    Returns:
        List of (tile_image, y_start, y_end) tuples
    """
    width, height = image.size
    
    if height <= tile_height:
        return [(image, 0, height)]
    
    tiles = []
    y_start = 0
    
    while y_start < height:
        y_end = min(y_start + tile_height, height)
        tile = image.crop((0, y_start, width, y_end))
        tiles.append((tile, y_start, y_end))
        
        # If this is the last tile, break
        if y_end >= height:
            break
            
        # Move to next tile with overlap
        y_start = y_end - overlap
    
    logger.debug(f"Image {width}x{height} tiled into {len(tiles)} tiles "
                f"(tile_height={tile_height}, overlap={overlap})")
    return tiles


def enhance_contrast(image: Image.Image) -> Image.Image:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve text visibility.
    Particularly helpful for light/thin text on light backgrounds.
    
    Args:
        image: PIL Image (RGB)
        
    Returns:
        Enhanced PIL Image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Convert back to RGB for PaddleOCR
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(enhanced_rgb)


def calculate_iou(box1: List[List[float]], box2: List[List[float]]) -> float:
    """
    Calculate Intersection over Union for two bounding boxes.
    
    Args:
        box1, box2: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
    Returns:
        IoU score (0.0 to 1.0)
    """
    # Convert to min-max format
    x1_min = min(p[0] for p in box1)
    y1_min = min(p[1] for p in box1)
    x1_max = max(p[0] for p in box1)
    y1_max = max(p[1] for p in box1)
    
    x2_min = min(p[0] for p in box2)
    y2_min = min(p[1] for p in box2)
    x2_max = max(p[0] for p in box2)
    y2_max = max(p[1] for p in box2)
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def text_similarity(text1: str, text2: str) -> float:
    """
    Simple text similarity based on character overlap.
    
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if text1 == text2:
        return 1.0
    
    # Simple character overlap ratio
    set1 = set(text1.lower())
    set2 = set(text2.lower())
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def deduplicate_boxes(
    lines: List[Dict[str, Any]], 
    iou_threshold: float = 0.5,
    text_sim_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Remove duplicate detections based on IoU and text similarity.
    Keep the detection with higher confidence.
    
    Args:
        lines: List of OCR results
        iou_threshold: IoU threshold for considering boxes duplicate
        text_sim_threshold: Text similarity threshold
        
    Returns:
        Deduplicated list of lines
    """
    if len(lines) <= 1:
        return lines
    
    # Sort by confidence descending
    sorted_lines = sorted(lines, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    skip_indices = set()
    
    for i, line1 in enumerate(sorted_lines):
        if i in skip_indices:
            continue
            
        keep.append(line1)
        
        # Check for duplicates with lower confidence
        for j in range(i + 1, len(sorted_lines)):
            if j in skip_indices:
                continue
                
            line2 = sorted_lines[j]
            
            iou = calculate_iou(line1['bbox'], line2['bbox'])
            text_sim = text_similarity(line1['text'], line2['text'])
            
            if iou > iou_threshold and text_sim > text_sim_threshold:
                skip_indices.add(j)
    
    logger.debug(f"Deduplication: {len(lines)} -> {len(keep)} lines "
                f"(removed {len(lines) - len(keep)} duplicates)")
    
    return keep


def draw_debug_boxes(
    image: Image.Image, 
    lines: List[Dict[str, Any]], 
    output_path: str,
    title: str = ""
):
    """
    Draw bounding boxes on image for debugging.
    
    Args:
        image: PIL Image
        lines: List of OCR results with bbox
        output_path: Path to save debug image
        title: Optional title text
    """
    try:
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        
        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        for idx, line in enumerate(lines):
            bbox = line['bbox']
            text = line['text']
            conf = line['confidence']
            
            # Draw polygon
            points = [(p[0], p[1]) for p in bbox]
            draw.polygon(points, outline='red', width=2)
            
            # Draw text label
            label = f"{text[:20]}... ({conf:.2f})" if len(text) > 20 else f"{text} ({conf:.2f})"
            draw.text((bbox[0][0], bbox[0][1] - 25), label, fill='blue', font=font)
        
        # Draw title
        if title:
            draw.text((10, 10), title, fill='green', font=font)
        
        img_draw.save(output_path)
        logger.debug(f"Debug image saved: {output_path}")
        
    except Exception as e:
        logger.warning(f"Failed to save debug image: {e}")


def run_ocr(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Run OCR on image bytes and return normalized results.
    Original function for backward compatibility.
    
    Returns list of line objects:
    [
        {
            "text": "detected text",
            "confidence": 0.98,
            "bbox": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        },
        ...
    ]
    """
    ocr = get_ocr_instance()
    
    image = Image.open(BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    img_array = np.array(image)
    
    result = ocr.ocr(img_array)
    
    lines = []
    
    # PaddleOCR 3.x returns OCRResult object
    if result and len(result) > 0:
        ocr_result = result[0]
        
        # Access as dict-like object
        if hasattr(ocr_result, '__getitem__'):
            try:
                rec_texts = ocr_result.get('rec_texts', []) if hasattr(ocr_result, 'get') else (ocr_result['rec_texts'] if 'rec_texts' in ocr_result else [])
                rec_scores = ocr_result.get('rec_scores', []) if hasattr(ocr_result, 'get') else (ocr_result['rec_scores'] if 'rec_scores' in ocr_result else [])
                rec_polys = ocr_result.get('rec_polys', []) if hasattr(ocr_result, 'get') else (ocr_result['rec_polys'] if 'rec_polys' in ocr_result else [])
                
                # Combine texts, scores, and bboxes
                for i in range(len(rec_texts)):
                    text = rec_texts[i]
                    confidence = rec_scores[i] if i < len(rec_scores) else 0.0
                    bbox = rec_polys[i] if i < len(rec_polys) else None
                    
                    # Format bbox as [[x,y], [x,y], [x,y], [x,y]]
                    bbox_formatted = []
                    try:
                        if bbox is not None and hasattr(bbox, '__len__') and len(bbox) >= 4:
                            for pt in bbox[:4]:
                                if hasattr(pt, '__getitem__') and len(pt) >= 2:
                                    bbox_formatted.append([float(pt[0]), float(pt[1])])
                            
                            if len(bbox_formatted) == 4:
                                lines.append({
                                    "text": str(text),
                                    "confidence": round(float(confidence), 4),
                                    "bbox": bbox_formatted
                                })
                    except (ValueError, TypeError, IndexError):
                        continue
                        
            except (KeyError, TypeError):
                pass
    
    return lines


def merge_nearby_lines(lines: List[Dict[str, Any]], 
                       vertical_threshold: Optional[float] = None,
                       horizontal_overlap_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Merge text lines that are close together (likely in same speech bubble).
    
    This is critical for webtoon/manhwa where one speech bubble often contains
    multiple detected lines that should be merged into one logical text block.
    
    Args:
        lines: List of OCR results with bbox and text
        vertical_threshold: Max vertical distance to merge (default: 1.5x box height)
        horizontal_overlap_threshold: Min horizontal overlap ratio to merge (0-1)
        
    Returns:
        List of merged lines with combined text
    """
    if len(lines) <= 1:
        return lines
    
    # Check if merging is enabled
    merge_enabled = os.environ.get("OCR_MERGE_NEARBY", "true").lower() == "true"
    if not merge_enabled:
        return lines
    
    # Sort by Y position (top to bottom)
    sorted_lines = sorted(lines, key=lambda x: min(p[1] for p in x['bbox']))
    
    merged = []
    skip_indices = set()
    
    for i, line1 in enumerate(sorted_lines):
        if i in skip_indices:
            continue
        
        # Get bbox bounds for line1
        x1_min = min(p[0] for p in line1['bbox'])
        y1_min = min(p[1] for p in line1['bbox'])
        x1_max = max(p[0] for p in line1['bbox'])
        y1_max = max(p[1] for p in line1['bbox'])
        box1_height = y1_max - y1_min
        box1_width = x1_max - x1_min
        
        # Default vertical threshold: 1.5x box height
        if vertical_threshold is None:
            v_thresh = float(os.environ.get("OCR_MERGE_DISTANCE_THRESHOLD", "1.5")) * box1_height
        else:
            v_thresh = vertical_threshold
        
        # Start group with current line
        group = [line1]
        group_indices = [i]
        
        # Look for nearby lines to merge
        for j in range(i + 1, len(sorted_lines)):
            if j in skip_indices:
                continue
            
            line2 = sorted_lines[j]
            
            # Get bbox bounds for line2
            x2_min = min(p[0] for p in line2['bbox'])
            y2_min = min(p[1] for p in line2['bbox'])
            x2_max = max(p[0] for p in line2['bbox'])
            y2_max = max(p[1] for p in line2['bbox'])
            
            # Check vertical distance (from bottom of line1 to top of line2)
            vertical_distance = y2_min - y1_max
            
            # Stop if too far below
            if vertical_distance > v_thresh:
                break
            
            # Check horizontal overlap
            overlap_start = max(x1_min, x2_min)
            overlap_end = min(x1_max, x2_max)
            overlap_width = max(0, overlap_end - overlap_start)
            
            # Calculate overlap ratio (relative to smaller box width)
            min_width = min(box1_width, x2_max - x2_min)
            overlap_ratio = overlap_width / min_width if min_width > 0 else 0
            
            # Merge if close vertically AND horizontally overlapping
            if vertical_distance <= v_thresh and overlap_ratio >= horizontal_overlap_threshold:
                group.append(line2)
                group_indices.append(j)
                
                # Update bounds for next comparison
                y1_max = max(y1_max, y2_max)
                x1_min = min(x1_min, x2_min)
                x1_max = max(x1_max, x2_max)
        
        # Mark group members as processed
        for idx in group_indices:
            skip_indices.add(idx)
        
        # Merge group into single line
        if len(group) == 1:
            # No merge needed
            merged.append(group[0])
        else:
            # Merge multiple lines
            merged_text = " ".join(line['text'] for line in group)
            avg_confidence = sum(line['confidence'] for line in group) / len(group)
            
            # Create merged bbox (bounding rectangle of all boxes)
            all_points = []
            for line in group:
                all_points.extend(line['bbox'])
            
            merged_x_min = min(p[0] for p in all_points)
            merged_y_min = min(p[1] for p in all_points)
            merged_x_max = max(p[0] for p in all_points)
            merged_y_max = max(p[1] for p in all_points)
            
            merged_bbox = [
                [merged_x_min, merged_y_min],
                [merged_x_max, merged_y_min],
                [merged_x_max, merged_y_max],
                [merged_x_min, merged_y_max]
            ]
            
            merged_line = {
                "text": merged_text,
                "confidence": round(avg_confidence, 4),
                "bbox": merged_bbox,
                "merged_from": len(group)  # Metadata: how many lines were merged
            }
            
            # Preserve any additional metadata from first line
            for key in group[0]:
                if key not in ['text', 'confidence', 'bbox']:
                    merged_line[key] = group[0][key]
            
            merged.append(merged_line)
    
    if DEBUG_MODE and len(merged) < len(lines):
        logger.info(f"[MERGE] Merged {len(lines)} lines → {len(merged)} bubbles "
                   f"({len(lines) - len(merged)} merges)")
    
    return merged


def preprocess_clahe(image: Image.Image) -> Image.Image:
    """
    Apply CLAHE preprocessing for better text detection on low-contrast images.
    This is used as a fallback for tiles with poor initial results.
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Convert back to RGB for PaddleOCR
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(enhanced_rgb)


def bbox_iou(box1: List[List[float]], box2: List[List[float]]) -> float:
    """
    Calculate IoU between two bounding boxes (4-point polygons).
    Uses bounding rectangle approximation for speed.
    """
    # Get min-max bounds for each box
    x1_min = min(p[0] for p in box1)
    y1_min = min(p[1] for p in box1)
    x1_max = max(p[0] for p in box1)
    y1_max = max(p[1] for p in box1)
    
    x2_min = min(p[0] for p in box2)
    y2_min = min(p[1] for p in box2)
    x2_max = max(p[0] for p in box2)
    y2_max = max(p[1] for p in box2)
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using SequenceMatcher."""
    return SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()


def deduplicate_lines(lines: List[Dict[str, Any]], iou_threshold: float = 0.6, text_sim_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Remove duplicate detections from overlapping tiles using IoU and text similarity.
    Optimized with spatial binning to avoid O(n^2) comparisons.
    """
    if len(lines) <= 1:
        return lines
    
    # Sort by y-coordinate for spatial binning
    sorted_lines = sorted(lines, key=lambda x: min(p[1] for p in x['bbox']))
    
    keep = []
    skip_indices = set()
    
    for i, line1 in enumerate(sorted_lines):
        if i in skip_indices:
            continue
        
        keep.append(line1)
        y1_min = min(p[1] for p in line1['bbox'])
        y1_max = max(p[1] for p in line1['bbox'])
        box_height = y1_max - y1_min
        
        # Only compare with nearby lines (within 3x box height)
        for j in range(i + 1, len(sorted_lines)):
            if j in skip_indices:
                continue
            
            line2 = sorted_lines[j]
            y2_min = min(p[1] for p in line2['bbox'])
            
            # Stop if too far below
            if y2_min > y1_max + box_height * 3:
                break
            
            # Check IoU and text similarity
            iou = bbox_iou(line1['bbox'], line2['bbox'])
            if iou > iou_threshold:
                text_sim = text_similarity(line1['text'], line2['text'])
                if text_sim > text_sim_threshold:
                    # Mark lower confidence one for removal
                    if line2.get('confidence', 0) > line1.get('confidence', 0):
                        # Current line is worse, remove it and stop comparing
                        keep.pop()
                        skip_indices.add(i)
                        break
                    else:
                        skip_indices.add(j)
    
    logger.debug(f"Deduplication: {len(lines)} -> {len(keep)} lines (removed {len(lines) - len(keep)} duplicates)")
    return keep


def choose_plan(width: int, height: int) -> Dict[str, Any]:
    """
    Choose OCR strategy based on image dimensions.
    Returns plan dict with strategy and parameters.
    """
    # Read thresholds from environment
    h1 = int(os.environ.get("OCR_ADAPTIVE_H1", "3500"))
    h2 = int(os.environ.get("OCR_ADAPTIVE_H2", "12000"))
    tile_height_med = int(os.environ.get("OCR_TILE_HEIGHT_MED", "2200"))
    tile_height_long = int(os.environ.get("OCR_TILE_HEIGHT_LONG", "1800"))
    overlap = int(os.environ.get("OCR_TILE_OVERLAP", "250"))
    
    if height <= h1:
        return {
            "strategy": "NO_TILE",
            "tile_height": None,
            "overlap": None,
            "reason": f"Image height {height}px <= {h1}px threshold"
        }
    elif height <= h2:
        return {
            "strategy": "TILE_MED",
            "tile_height": tile_height_med,
            "overlap": overlap,
            "reason": f"Image height {height}px in medium range ({h1}-{h2}px)"
        }
    else:
        return {
            "strategy": "TILE_LONG",
            "tile_height": tile_height_long,
            "overlap": overlap,
            "reason": f"Image height {height}px > {h2}px, using smaller tiles"
        }


def run_ocr_adaptive(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Adaptive OCR pipeline that chooses optimal strategy based on image size.
    
    Strategy:
    1. Analyze image dimensions
    2. Choose tiling strategy (or no tiling for small images)
    3. Run OCR with pass A (normal) on all tiles
    4. Identify poor-performing tiles
    5. Re-run only poor tiles with pass B (CLAHE preprocessing)
    6. Deduplicate results from overlapping tiles
    7. Fallback to smaller tiles if entire image has poor results
    
    Returns:
        List of OCR results with global coordinates, deduplicated and sorted
    """
    start_time = time.time()
    
    # Load image and get dimensions
    image = Image.open(BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    width, height = image.size
    
    # Choose strategy
    plan = choose_plan(width, height)
    
    if DEBUG_MODE:
        logger.info(f"[ADAPTIVE] Image: {width}x{height} | Plan: {plan['strategy']} | {plan['reason']}")
    
    # Create debug directory if needed
    if DEBUG_MODE and not os.path.exists(DEBUG_DIR):
        os.makedirs(DEBUG_DIR, exist_ok=True)
    
    # NO_TILE strategy: direct OCR
    if plan['strategy'] == 'NO_TILE':
        lines = _run_ocr_on_image(image)
        
        # Sort by position
        lines.sort(key=lambda x: (min(p[1] for p in x['bbox']), min(p[0] for p in x['bbox'])))
        
        # Merge nearby lines (combine text in same bubble)
        merged = merge_nearby_lines(lines)
        
        if DEBUG_MODE:
            logger.info(f"[ADAPTIVE] NO_TILE: {len(lines)} lines → {len(merged)} bubbles "
                       f"in {time.time() - start_time:.1f}s")
            if merged:
                debug_path = os.path.join(DEBUG_DIR, f"notile_{int(time.time())}.jpg")
                draw_debug_boxes(image, merged, debug_path, f"NO_TILE: {len(merged)} bubbles")
        
        return merged
    
    # TILE strategy
    tile_height = plan['tile_height']
    overlap = plan['overlap']
    
    # Create tiles
    tiles = tile_image(image, tile_height, overlap)
    
    if DEBUG_MODE:
        logger.info(f"[ADAPTIVE] Created {len(tiles)} tiles (height={tile_height}, overlap={overlap})")
    
    all_lines = []
    tile_stats = []
    
    # Pass A: Normal OCR on all tiles
    for tile_idx, (tile_img, y_start, y_end) in enumerate(tiles):
        tile_start = time.time()
        
        lines_passA = _run_ocr_on_image(tile_img, pass_name="passA")
        
        # Adjust bbox to global coordinates
        for line in lines_passA:
            line['bbox'] = [[p[0], p[1] + y_start] for p in line['bbox']]
            line['tile_index'] = tile_idx
            line['pass'] = 'A'
        
        # Calculate tile stats
        line_count = len(lines_passA)
        avg_conf = sum(l['confidence'] for l in lines_passA) / line_count if line_count > 0 else 0.0
        
        tile_stats.append({
            'index': tile_idx,
            'line_count': line_count,
            'avg_conf': avg_conf,
            'y_start': y_start,
            'y_end': y_end,
            'time': time.time() - tile_start
        })
        
        all_lines.extend(lines_passA)
        
        if DEBUG_MODE:
            logger.info(f"[ADAPTIVE] Tile {tile_idx + 1}/{len(tiles)} passA: {line_count} lines, "
                       f"avg_conf={avg_conf:.3f}, time={tile_stats[-1]['time']:.1f}s")
    
    # Identify bad tiles for pass B
    bad_linecount_threshold = int(os.environ.get("OCR_TILE_BAD_LINECOUNT", "2"))
    bad_avgconf_threshold = float(os.environ.get("OCR_TILE_BAD_AVGCONF", "0.45"))
    
    bad_tiles = [
        stat for stat in tile_stats
        if stat['line_count'] < bad_linecount_threshold or stat['avg_conf'] < bad_avgconf_threshold
    ]
    
    # Pass B: CLAHE preprocessing on bad tiles only
    if bad_tiles:
        if DEBUG_MODE:
            logger.info(f"[ADAPTIVE] Running pass B (CLAHE) on {len(bad_tiles)} poor tiles")
        
        for stat in bad_tiles:
            tile_idx = stat['index']
            tile_img, y_start, y_end = tiles[tile_idx]
            
            tile_start = time.time()
            
            # Preprocess with CLAHE
            enhanced_tile = preprocess_clahe(tile_img)
            lines_passB = _run_ocr_on_image(enhanced_tile, pass_name="passB")
            
            # Adjust bbox to global coordinates
            for line in lines_passB:
                line['bbox'] = [[p[0], p[1] + y_start] for p in line['bbox']]
                line['tile_index'] = tile_idx
                line['pass'] = 'B'
            
            all_lines.extend(lines_passB)
            
            if DEBUG_MODE:
                logger.info(f"[ADAPTIVE] Tile {tile_idx + 1} passB: {len(lines_passB)} additional lines "
                           f"in {time.time() - tile_start:.1f}s")
    
    # Deduplicate overlapping detections
    dedup_start = time.time()
    iou_thresh = float(os.environ.get("OCR_IOU_THRESHOLD", "0.6"))
    text_sim_thresh = float(os.environ.get("OCR_TEXT_SIM_THRESHOLD", "0.7"))
    deduplicated = deduplicate_lines(all_lines, iou_thresh, text_sim_thresh)
    
    if DEBUG_MODE:
        logger.info(f"[ADAPTIVE] Deduplication: {len(all_lines)} -> {len(deduplicated)} lines "
                   f"in {time.time() - dedup_start:.1f}s")
    
    # Check if overall result is poor (fallback to smaller tiles)
    total_lines = len(deduplicated)
    min_lines_threshold = int(os.environ.get("OCR_FALLBACK_MIN_LINES", "5"))
    
    if total_lines < min_lines_threshold and plan['strategy'] != 'TILE_LONG':
        if DEBUG_MODE:
            logger.warning(f"[ADAPTIVE] Poor results ({total_lines} lines < {min_lines_threshold}), "
                          f"attempting fallback with smaller tiles")
        
        # Fallback: use smaller tile height
        fallback_tile_height = int(os.environ.get("OCR_FALLBACK_TILE_HEIGHT_SMALL", "1400"))
        tiles_fallback = tile_image(image, fallback_tile_height, overlap)
        
        all_lines_fallback = []
        
        for tile_idx, (tile_img, y_start, y_end) in enumerate(tiles_fallback):
            # Run with CLAHE directly since we know it's a hard image
            enhanced_tile = preprocess_clahe(tile_img)
            lines = _run_ocr_on_image(enhanced_tile, pass_name="fallback")
            
            for line in lines:
                line['bbox'] = [[p[0], p[1] + y_start] for p in line['bbox']]
                line['tile_index'] = tile_idx
                line['pass'] = 'FALLBACK'
            
            all_lines_fallback.extend(lines)
        
        deduplicated_fallback = deduplicate_lines(all_lines_fallback, iou_thresh, text_sim_thresh)
        
        if DEBUG_MODE:
            logger.info(f"[ADAPTIVE] Fallback result: {len(deduplicated_fallback)} lines")
        
        # Use fallback if better
        if len(deduplicated_fallback) > total_lines:
            deduplicated = deduplicated_fallback
    
    # Sort by position (y then x)
    deduplicated.sort(key=lambda x: (min(p[1] for p in x['bbox']), min(p[0] for p in x['bbox'])))
    
    # Merge nearby lines (combine text in same bubble)
    merge_start = time.time()
    merged = merge_nearby_lines(deduplicated)
    
    if DEBUG_MODE and len(merged) < len(deduplicated):
        logger.info(f"[ADAPTIVE] Line merging: {len(deduplicated)} -> {len(merged)} bubbles "
                   f"in {time.time() - merge_start:.1f}s")
    
    elapsed = time.time() - start_time
    
    if DEBUG_MODE:
        logger.info(f"[ADAPTIVE] Complete: {len(merged)} bubbles in {elapsed:.1f}s | "
                   f"Strategy: {plan['strategy']} | PassB tiles: {len(bad_tiles)}")
        
        # Save debug overlay
        if merged:
            debug_path = os.path.join(DEBUG_DIR, f"adaptive_{int(time.time())}.jpg")
            draw_debug_boxes(image, merged, debug_path, 
                           f"{plan['strategy']}: {len(merged)} bubbles")
    
    return merged


def _run_ocr_on_image(image: Image.Image, pass_name: str = "") -> List[Dict[str, Any]]:
    """
    Internal helper to run OCR on a PIL Image.
    Returns lines with local coordinates (not adjusted for global position).
    """
    try:
        ocr = get_ocr_instance()
        
        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        img_array = np.array(image)
        result = ocr.ocr(img_array)
        
        lines = []
        
        if result and len(result) > 0:
            ocr_result = result[0]
            
            if hasattr(ocr_result, '__getitem__'):
                try:
                    rec_texts = ocr_result.get('rec_texts', []) if hasattr(ocr_result, 'get') else (ocr_result['rec_texts'] if 'rec_texts' in ocr_result else [])
                    rec_scores = ocr_result.get('rec_scores', []) if hasattr(ocr_result, 'get') else (ocr_result['rec_scores'] if 'rec_scores' in ocr_result else [])
                    rec_polys = ocr_result.get('rec_polys', []) if hasattr(ocr_result, 'get') else (ocr_result['rec_polys'] if 'rec_polys' in ocr_result else [])
                    
                    for i in range(len(rec_texts)):
                        text = rec_texts[i]
                        confidence = rec_scores[i] if i < len(rec_scores) else 0.0
                        bbox = rec_polys[i] if i < len(rec_polys) else None
                        
                        bbox_formatted = []
                        try:
                            if bbox is not None and hasattr(bbox, '__len__') and len(bbox) >= 4:
                                for pt in bbox[:4]:
                                    if hasattr(pt, '__getitem__') and len(pt) >= 2:
                                        bbox_formatted.append([float(pt[0]), float(pt[1])])
                                
                                if len(bbox_formatted) == 4:
                                    lines.append({
                                        "text": str(text),
                                        "confidence": round(float(confidence), 4),
                                        "bbox": bbox_formatted
                                    })
                        except (ValueError, TypeError, IndexError):
                            continue
                            
                except (KeyError, TypeError):
                    pass
        
        if pass_name:
            logger.debug(f"OCR {pass_name} pass: {len(lines)} detections")
        
        return lines
        
    except Exception as e:
        logger.error(f"Error in _run_ocr_on_image ({pass_name}): {e}")
        return []


def run_ocr_with_tiling(
    image_bytes: bytes,
    tile_height: Optional[int] = None,
    overlap: Optional[int] = None,
    use_two_pass: Optional[bool] = None,
    debug_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Enhanced OCR with tiling for tall images and optional two-pass processing.
    
    Args:
        image_bytes: Image data as bytes
        tile_height: Height of each tile (default from env OCR_TILE_HEIGHT or 2000)
        overlap: Overlap between tiles (default from env OCR_TILE_OVERLAP or 200)
        use_two_pass: Run OCR on both original and enhanced images (default from env OCR_TWO_PASS)
        debug_dir: If set, save debug images with bounding boxes to this directory
        
    Returns:
        List of deduplicated OCR results sorted by position (y, then x)
    """
    try:
        print("[DEBUG] run_ocr_with_tiling called", flush=True)
        logger.info("=== Starting tiling OCR ===")
        start_time = time.time()
        
        # Get config from env or use defaults
        if tile_height is None:
            tile_height = int(os.environ.get("OCR_TILE_HEIGHT", "2000"))
        if overlap is None:
            overlap = int(os.environ.get("OCR_TILE_OVERLAP", "200"))
        if use_two_pass is None:
            use_two_pass = os.environ.get("OCR_TWO_PASS", "false").lower() == "true"
        
        iou_threshold = float(os.environ.get("OCR_IOU_THRESHOLD", "0.5"))
        text_sim_threshold = float(os.environ.get("OCR_TEXT_SIM_THRESHOLD", "0.7"))
        
        print(f"[DEBUG] Config: tile_height={tile_height}, overlap={overlap}, two_pass={use_two_pass}", flush=True)
        logger.info(f"Config: tile_height={tile_height}, overlap={overlap}, two_pass={use_two_pass}")
        
        # Load and tile image
        image = Image.open(BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        width, height = image.size
        print(f"[DEBUG] Image size: {width}x{height}", flush=True)
        logger.info(f"Image size: {width}x{height}")
        
        tiles = tile_image(image, tile_height, overlap)
        print(f"[DEBUG] Created {len(tiles)} tiles", flush=True)
        logger.info(f"Created {len(tiles)} tiles")
        
        all_lines = []
        
        # Process each tile
        for tile_idx, (tile_img, y_start, y_end) in enumerate(tiles):
            tile_start = time.time()
            logger.info(f"=== Processing tile {tile_idx + 1}/{len(tiles)} (y={y_start}-{y_end}) ===")
            
            try:
                # Pass 1: Original image
                logger.info(f"Running OCR on tile {tile_idx + 1}...")
                lines_pass1 = _run_ocr_on_image(tile_img, pass_name="original")
                logger.info(f"Tile {tile_idx + 1}: Got {len(lines_pass1)} raw detections")
                
                # Adjust bbox coordinates to global position
                for line in lines_pass1:
                    line['bbox'] = [[p[0], p[1] + y_start] for p in line['bbox']]
                    line['tile_index'] = tile_idx
                    line['source_pass'] = 'original'
                
                all_lines.extend(lines_pass1)
                logger.info(f"Tile {tile_idx + 1}: Completed in {time.time() - tile_start:.1f}s")
                
            except Exception as e:
                logger.error(f"Failed to process tile {tile_idx + 1}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        logger.info(f"All tiles processed. Total raw detections: {len(all_lines)}")
        
        # Deduplicate
        logger.info("Deduplicating...")
        try:
            deduplicated = deduplicate_boxes(all_lines, iou_threshold, text_sim_threshold)
            logger.info(f"Deduplication complete: {len(all_lines)} -> {len(deduplicated)}")
        except Exception as e:
            logger.warning(f"Deduplication failed: {e}, using all detections")
            deduplicated = all_lines
        
        # Sort by position
        logger.info("Sorting results...")
        try:
            deduplicated.sort(key=lambda line: (
                min(p[1] for p in line['bbox']),
                min(p[0] for p in line['bbox'])
            ))
        except Exception as e:
            logger.warning(f"Sorting failed: {e}")
        
        elapsed = time.time() - start_time
        logger.info(f"=== Tiling OCR complete: {len(deduplicated)} detections in {elapsed:.2f}s ===")
        
        return deduplicated
        
    except Exception as e:
        logger.error(f"FATAL error in run_ocr_with_tiling: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("Falling back to standard OCR...")
        return run_ocr(image_bytes)


def build_ocr_output(
    lines: List[Dict[str, Any]],
    work_id: str,
    edition_id: str,
    segment_id: str,
    chapter: int = None,
    page: int = None,
    raw_r2_key: str = None,
    raw_asset_id: str = None
) -> Dict[str, Any]:
    """
    Build OCR output JSON structure.
    
    Returns:
    {
        "version": "ocr_v1",
        "engine": "paddleocr",
        "source": {
            "raw_asset_id": "uuid",
            "raw_r2_key": "raw/..."
        },
        "metadata": {
            "work_id": "uuid",
            "edition_id": "uuid",
            "segment_id": "uuid",
            "chapter": 1,
            "page": 1,
            "source_key": "raw/..."
        },
        "stats": {
            "line_count": 46
        },
        "lines": [...]
    }
    """
    return {
        "version": "ocr_v1",
        "engine": "paddleocr",
        "source": {
            "raw_asset_id": raw_asset_id,
            "raw_r2_key": raw_r2_key
        },
        "metadata": {
            "work_id": work_id,
            "edition_id": edition_id,
            "segment_id": segment_id,
            "chapter": chapter,
            "page": page,
            "source_key": raw_r2_key
        },
        "stats": {
            "line_count": len(lines)
        },
        "lines": lines
    }
