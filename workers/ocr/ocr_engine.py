"""
PaddleOCR engine singleton with result normalization.
"""
import os
from typing import List, Dict, Any, Optional
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
        
        _ocr_instance = PaddleOCR(
            lang=lang,
            use_textline_orientation=use_angle_cls
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
    
    result = ocr.ocr(img_array, cls=True)
    
    lines = []
    
    if result and result[0]:
        for line in result[0]:
            if line and len(line) >= 2:
                bbox = line[0]
                text_info = line[1]
                
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text = str(text_info[0])
                    confidence = float(text_info[1])
                else:
                    text = str(text_info)
                    confidence = 0.0
                
                bbox_formatted = [[float(pt[0]), float(pt[1])] for pt in bbox]
                
                lines.append({
                    "text": text,
                    "confidence": round(confidence, 4),
                    "bbox": bbox_formatted
                })
    
    return lines

def build_ocr_output(
    lines: List[Dict[str, Any]],
    work_id: Optional[str],
    edition_id: Optional[str],
    segment_id: Optional[str],
    chapter: Optional[int],
    page: Optional[int],
    raw_r2_key: str
) -> Dict[str, Any]:
    """
    Build the full OCR JSON output with metadata.
    """
    return {
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
