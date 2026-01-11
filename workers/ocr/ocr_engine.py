"""
PaddleOCR engine wrapper with singleton pattern.
"""
import os
from typing import List, Dict, Any
from io import BytesIO
from PIL import Image
import numpy as np

_ocr_instance = None

def get_ocr_instance():
    """
    Get or create singleton PaddleOCR instance.
    Avoids re-initialization overhead per job.
    """
    global _ocr_instance
    
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        
        lang = os.environ.get("OCR_LANG", "en")
        use_angle_cls = os.environ.get("OCR_USE_ANGLE_CLS", "true").lower() == "true"
        det_limit_side_len = int(os.environ.get("OCR_DET_LIMIT_SIDE_LEN", "2560"))
        
        _ocr_instance = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            text_det_limit_side_len=det_limit_side_len,
            text_det_limit_type='max',
            text_det_thresh=0.15,
            text_det_box_thresh=0.35,
            text_det_unclip_ratio=2.5
        )
    
    return _ocr_instance

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
