"""
Utility functions for the OCR worker.
"""
import hashlib
import json
import logging
import sys
from datetime import datetime

def setup_logging(name: str = "ocr_worker") -> logging.Logger:
    """Configure and return a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of bytes and return hex digest."""
    return hashlib.sha256(data).hexdigest()

def json_dumps(obj: dict) -> bytes:
    """Serialize object to JSON bytes with consistent formatting."""
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")

def utc_now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.utcnow().isoformat() + "Z"
