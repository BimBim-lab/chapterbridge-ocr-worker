"""
PaddleOCR engine wrapper with tiling and two-pass OCR for improved recall.
"""
import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

logger = logging.getLogger(__name__)

_ocr_instance = None

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
        rec_score_thresh = float(os.environ.get("OCR_REC_SCORE_THRESH", "0.3"))
        
        _ocr_instance = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            text_det_limit_side_len=det_limit_side_len,
            text_det_limit_type='max',
            text_det_thresh=det_thresh,
            text_det_box_thresh=det_box_thresh,
            text_det_unclip_ratio=det_unclip_ratio,
            text_rec_score_thresh=rec_score_thresh
        )
        
        logger.info(f"PaddleOCR initialized: lang={lang}, det_thresh={det_thresh}, "
                   f"box_thresh={det_box_thresh}, unclip={det_unclip_ratio}, "
                   f"rec_thresh={rec_score_thresh}")
    
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


def run_ocr_with_tiling(
) -> List[Dict[str, Any]]:
    """
    Enhanced OCR with tiling for tall images and optional two-pass processing.
    
    Args:
        image_bytes: Image data as bytes
        tile_height: Height of each tile (default from env OCR_TILE_HEIGHT or 1400)
        overlap: Overlap between tiles (default from env OCR_TILE_OVERLAP or 200)
        use_two_pass: Run OCR on both original and enhanced images (default from env OCR_TWO_PASS)
        debug_dir: If set, save debug images with bounding boxes to this directory
        
    Returns:
        List of deduplicated OCR results sorted by position (y, then x)
    """
    import time
    start_time = time.time()
    
    # Get config from env or use defaults
    if tile_height is None:
        tile_height = int(os.environ.get("OCR_TILE_HEIGHT", "1400"))
    if overlap is None:
        overlap = int(os.environ.get("OCR_TILE_OVERLAP", "200"))
    if use_two_pass is True:
        use_two_pass = os.environ.get("OCR_TWO_PASS", "true").lower() == "true"
    
    iou_threshold = float(os.environ.get("OCR_IOU_THRESHOLD", "0.5"))
    text_sim_threshold = float(os.environ.get("OCR_TEXT_SIM_THRESHOLD", "0.7"))
    
    # Load and tile image
    image = Image.open(BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    width, height = image.size
    logger.info(f"Processing image {width}x{height}, tiling={tile_height}px, "
               f"overlap={overlap}px, two_pass={use_two_pass}")
    
    tiles = tile_image(image, tile_height, overlap)
    
    all_lines = []
    
    # Process each tile
    for tile_idx, (tile_img, y_start, y_end) in enumerate(tiles):
        logger.debug(f"Processing tile {tile_idx + 1}/{len(tiles)} "
                    f"(y={y_start}-{y_end})")
        
        # Pass 1: Original image
        lines_pass1 = _run_ocr_on_image(tile_img, pass_name="original")
        
        # Adjust bbox coordinates to global position
        for line in lines_pass1:
            line['bbox'] = [[p[0], p[1] + y_start] for p in line['bbox']]
            line['tile_index'] = tile_idx
            line['source_pass'] = 'original'
        
        all_lines.extend(lines_pass1)
        
        # Pass 2: Enhanced image (optional)
        if use_two_pass:
            enhanced_img = enhance_contrast(tile_img)
            lines_pass2 = _run_ocr_on_image(enhanced_img, pass_name="enhanced")
            
            for line in lines_pass2:
                line['bbox'] = [[p[0], p[1] + y_start] for p in line['bbox']]
                line['tile_index'] = tile_idx
                line['source_pass'] = 'enhanced'
            
            all_lines.extend(lines_pass2)
        
        # Debug visualization per tile
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            
            tile_lines = [l for l in all_lines if l.get('tile_index') == tile_idx]
            debug_path = os.path.join(debug_dir, f"tile_{tile_idx:03d}.jpg")
            draw_debug_boxes(tile_img, tile_lines, debug_path, 
                           f"Tile {tile_idx} (y={y_start}-{y_end})")
    
    logger.info(f"Total raw detections: {len(all_lines)} from {len(tiles)} tiles")
    
    # Deduplicate across tiles and passes
    deduplicated = deduplicate_boxes(all_lines, iou_threshold, text_sim_threshold)
    
    # Sort by position: y first, then x
    deduplicated.sort(key=lambda line: (
        min(p[1] for p in line['bbox']),  # min y
        min(p[0] for p in line['bbox'])   # min x
    ))
    
    elapsed = time.time() - start_time
    logger.info(f"OCR complete: {len(deduplicated)} final detections in {elapsed:.2f}s "
               f"({elapsed/len(tiles):.2f}s/tile)")
    
    # Full page debug visualization
    if debug_dir:
        debug_path = os.path.join(debug_dir, "full_page.jpg")
        draw_debug_boxes(image, deduplicated, debug_path, 
                        f"Full page: {len(deduplicated)} detections")
    
    return deduplicated


def _run_ocr_on_image(image: Image.Image, pass_name: str = "") -> List[Dict[str, Any]]:
    """
    Internal helper to run OCR on a PIL Image.
    Returns lines with local coordinates (not adjusted for global position).
    """
    ocr = get_ocr_instance()
    
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
    Simple text similarity based on Levenshtein-like comparison.
    
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
