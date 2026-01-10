"""
R2 key parsing utilities.

Parses raw image keys like:
  raw/manhwa/{work_id}/{edition_id}/chapter-0236/page-001.jpg

Builds output keys like:
  derived/manhwa/{work_id}/{edition_id}/chapter-0236/ocr/page-001.json
"""
import re
from dataclasses import dataclass
from typing import Optional

RAW_KEY_PATTERN = re.compile(
    r"^raw/manhwa/(?P<work_id>[^/]+)/(?P<edition_id>[^/]+)/"
    r"(?P<chapter>chapter-\d+)/(?P<page>page-\d+)\.\w+$"
)

@dataclass
class ParsedKey:
    """Parsed components from an R2 key."""
    work_id: Optional[str] = None
    edition_id: Optional[str] = None
    chapter: Optional[str] = None
    page: Optional[str] = None
    is_valid: bool = False

def parse_raw_key(r2_key: str) -> ParsedKey:
    """
    Parse a raw image R2 key into its components.
    Returns ParsedKey with is_valid=False if pattern doesn't match.
    """
    match = RAW_KEY_PATTERN.match(r2_key)
    if not match:
        return ParsedKey(is_valid=False)
    
    return ParsedKey(
        work_id=match.group("work_id"),
        edition_id=match.group("edition_id"),
        chapter=match.group("chapter"),
        page=match.group("page"),
        is_valid=True
    )

def build_output_key(r2_key: str, raw_asset_id: str) -> str:
    """
    Build the output R2 key for OCR JSON based on raw key.
    
    If key can be parsed:
      derived/manhwa/{work_id}/{edition_id}/{chapter}/ocr/{page}.json
    
    Fallback:
      derived/manhwa/unknown/unknown/ocr/{raw_asset_id}.json
    """
    parsed = parse_raw_key(r2_key)
    
    if parsed.is_valid:
        return f"derived/manhwa/{parsed.work_id}/{parsed.edition_id}/{parsed.chapter}/ocr/{parsed.page}.json"
    
    return f"derived/manhwa/unknown/unknown/ocr/{raw_asset_id}.json"

def extract_chapter_number(chapter_str: str) -> Optional[int]:
    """Extract numeric chapter number from string like 'chapter-0236'."""
    match = re.search(r"chapter-(\d+)", chapter_str)
    if match:
        return int(match.group(1))
    return None

def extract_page_number(page_str: str) -> Optional[int]:
    """Extract numeric page number from string like 'page-001'."""
    match = re.search(r"page-(\d+)", page_str)
    if match:
        return int(match.group(1))
    return None
