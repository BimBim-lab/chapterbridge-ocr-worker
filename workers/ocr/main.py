#!/usr/bin/env python3
"""
OCR Worker Daemon

Polls Supabase for queued OCR jobs, processes them using PaddleOCR,
and uploads results to Cloudflare R2.

Usage:
    python workers/ocr/main.py --poll-seconds 3
"""
import os
import sys
import time
import random
import argparse
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from workers.ocr.utils import setup_logging, compute_sha256, json_dumps
from workers.ocr.supabase_client import SupabaseDB
from workers.ocr.r2_client import R2Client
from workers.ocr.ocr_engine import run_ocr, run_ocr_with_tiling, build_ocr_output
from workers.ocr.key_parser import (
    parse_raw_key, 
    build_output_key, 
    extract_chapter_number, 
    extract_page_number
)

logger = setup_logging("ocr_worker")


def elapsed_ms(start_time: datetime) -> float:
    """Calculate elapsed milliseconds since start_time."""
    return (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

def process_job(db: SupabaseDB, r2: R2Client, job: dict) -> None:
    """
    Process a single OCR job.
    
    Steps:
    1. Load and validate raw asset
    2. Resolve segment_id, work_id, edition_id
    3. Check idempotency (skip if output exists)
    4. Download image from R2
    5. Run OCR
    6. Upload result to R2
    7. Insert asset record and link to segment
    8. Mark job success
    
    Note: Job is already claimed as 'running' by poll_ocr_job.
    """
    job_start = datetime.now(timezone.utc)
    job_id = job["id"]
    job_input = job.get("input", {})
    raw_asset_id = job_input.get("raw_asset_id")
    force = job_input.get("force", False)
    
    logger.info(f"[job={job_id}] Starting OCR job | raw_asset_id={raw_asset_id}")
    
    raw_asset = db.get_asset(raw_asset_id)
    if not raw_asset:
        raise ValueError(f"Raw asset not found: {raw_asset_id}")
    
    if raw_asset.get("asset_type") != "raw_image":
        raise ValueError(f"Asset is not raw_image: {raw_asset.get('asset_type')}")
    
    raw_r2_key = raw_asset.get("r2_key")
    if not raw_r2_key:
        raise ValueError(f"Asset has no r2_key: {raw_asset_id}")
    
    logger.info(f"[job={job_id}] raw_r2_key={raw_r2_key}")
    
    segment_id = job.get("segment_id")
    if not segment_id:
        segment_id = db.get_segment_id_for_asset(raw_asset_id)
    
    if not segment_id:
        raise ValueError(f"Cannot resolve segment_id for asset {raw_asset_id}")
    
    logger.info(f"[job={job_id}] segment_id={segment_id}")
    
    work_id = job.get("work_id")
    edition_id = job.get("edition_id")
    
    parsed = parse_raw_key(raw_r2_key)
    
    if not work_id or not edition_id:
        if parsed.is_valid:
            work_id = work_id or parsed.work_id
            edition_id = edition_id or parsed.edition_id
        else:
            seg_info = db.get_segment_edition_work(segment_id)
            if seg_info:
                work_id = work_id or seg_info.get("work_id")
                edition_id = edition_id or seg_info.get("edition_id")
    
    logger.info(f"[job={job_id}] work_id={work_id}, edition_id={edition_id}")
    
    output_r2_key = build_output_key(raw_r2_key, raw_asset_id)
    logger.info(f"[job={job_id}] output_r2_key={output_r2_key}")
    
    if not force:
        existing_ocr_id = db.ocr_asset_exists(output_r2_key)
        if existing_ocr_id:
            db.link_segment_asset(segment_id, existing_ocr_id, "ocr_json")
            
            db.set_job_success(job_id, {
                "task": "ocr_page",
                "skipped": True,
                "reason": "already_exists",
                "ocr_asset_id": existing_ocr_id,
                "ocr_r2_key": output_r2_key
            })
            logger.info(f"[job={job_id}] Skipped (already exists) | ocr_asset_id={existing_ocr_id} | elapsed={elapsed_ms(job_start):.0f}ms")
            return
    
    download_start = datetime.now(timezone.utc)
    image_bytes = r2.get_object(raw_r2_key)
    download_ms = elapsed_ms(download_start)
    logger.info(f"[job={job_id}] Downloaded {len(image_bytes)} bytes in {download_ms:.0f}ms")
    
    # OCR execution
    ocr_start = datetime.now(timezone.utc)
    debug_dir = os.environ.get("OCR_DEBUG_DIR")
    
    # Check if adaptive mode is enabled
    use_adaptive = os.environ.get("OCR_ADAPTIVE", "false").lower() == "true"
    
    try:
        if use_adaptive:
            # Adaptive OCR: automatically chooses best strategy based on image dimensions
            logger.info(f"[job={job_id}] Using adaptive OCR mode")
            from workers.ocr.ocr_engine import run_ocr_adaptive
            lines = run_ocr_adaptive(image_bytes)
        else:
            # Legacy mode: use tiling configuration
            use_tiling = os.environ.get("OCR_USE_TILING", "auto").lower()
            
            if use_tiling == "always":
                logger.info(f"[job={job_id}] Force tiling mode")
                from workers.ocr.ocr_engine import run_ocr_with_tiling
                lines = run_ocr_with_tiling(image_bytes, debug_dir=debug_dir)
            elif use_tiling == "never":
                logger.info(f"[job={job_id}] Standard OCR mode")
                from workers.ocr.ocr_engine import run_ocr
                lines = run_ocr(image_bytes)
            else:  # auto (default)
                # Quick check of image dimensions
                from PIL import Image
                from io import BytesIO
                from workers.ocr.ocr_engine import run_ocr, run_ocr_with_tiling
                img = Image.open(BytesIO(image_bytes))
                width, height = img.size
                
                # Use tiling for tall images (height > 2000px)
                if height > 2000:
                    logger.info(f"[job={job_id}] Auto-enabling tiling for tall image ({width}x{height})")
                    lines = run_ocr_with_tiling(image_bytes, debug_dir=debug_dir)
                else:
                    logger.info(f"[job={job_id}] Using standard OCR for normal image ({width}x{height})")
                    lines = run_ocr(image_bytes)
    except Exception as e:
        logger.error(f"[job={job_id}] OCR failed: {str(e)}\n{traceback.format_exc()}")
        raise
    
    ocr_ms = elapsed_ms(ocr_start)
    logger.info(f"[job={job_id}] OCR complete: {len(lines)} lines in {ocr_ms:.0f}ms")
    
    chapter_num = extract_chapter_number(parsed.chapter) if parsed.chapter else None
    page_num = extract_page_number(parsed.page) if parsed.page else None
    
    ocr_output = build_ocr_output(
        lines=lines,
        work_id=work_id,
        edition_id=edition_id,
        segment_id=segment_id,
        chapter=chapter_num,
        page=page_num,
        raw_r2_key=raw_r2_key,
        raw_asset_id=raw_asset_id
    )
    
    json_bytes = json_dumps(ocr_output)
    sha256_hash = compute_sha256(json_bytes)
    
    upload_start = datetime.now(timezone.utc)
    r2.put_object(output_r2_key, json_bytes, "application/json")
    upload_ms = elapsed_ms(upload_start)
    logger.info(f"[job={job_id}] Uploaded {len(json_bytes)} bytes in {upload_ms:.0f}ms")
    
    ocr_asset_id = db.insert_asset(
        r2_key=output_r2_key,
        bucket=r2.bucket,
        asset_type="ocr_json",
        content_type="application/json",
        byte_size=len(json_bytes),
        sha256=sha256_hash
    )
    
    db.link_segment_asset(segment_id, ocr_asset_id, "ocr_json")
    
    total_ms = elapsed_ms(job_start)
    db.set_job_success(job_id, {
        "task": "ocr_page",
        "raw_asset_id": raw_asset_id,
        "raw_r2_key": raw_r2_key,
        "ocr_asset_id": ocr_asset_id,
        "ocr_r2_key": output_r2_key,
        "lines": len(lines),
        "processing_time_ms": round(total_ms)
    })
    
    logger.info(f"[job={job_id}] Completed | ocr_asset_id={ocr_asset_id} | lines={len(lines)} | total={total_ms:.0f}ms (download={download_ms:.0f}ms, ocr={ocr_ms:.0f}ms, upload={upload_ms:.0f}ms)")

def run_daemon(poll_seconds: int) -> None:
    """Main daemon loop that polls for and processes jobs."""
    # Add random startup delay to avoid thundering herd
    startup_jitter = random.uniform(0, poll_seconds * 0.5)
    logger.info(f"Starting OCR worker daemon (poll interval: {poll_seconds}s, startup delay: {startup_jitter:.2f}s)")
    time.sleep(startup_jitter)
    
    db = SupabaseDB()
    r2 = R2Client()
    
    logger.info("Initializing PaddleOCR engine (this may take a moment)...")
    from workers.ocr.ocr_engine import get_ocr_instance
    get_ocr_instance()
    logger.info("PaddleOCR engine ready")
    
    while True:
        try:
            job = db.poll_ocr_job()
            
            if job:
                try:
                    process_job(db, r2, job)
                except Exception as e:
                    error_msg = f"{str(e)}\n{traceback.format_exc()}"
                    logger.error(f"Job {job['id']} failed: {error_msg}")
                    db.set_job_failed(job["id"], error_msg)
            else:
                # Add random jitter to prevent all workers polling at same time
                jitter = random.uniform(0, poll_seconds * 0.3)
                sleep_time = poll_seconds + jitter
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Daemon error: {str(e)}\n{traceback.format_exc()}")
            logger.info(f"Retrying in {poll_seconds}s...")
            time.sleep(poll_seconds)
        except Exception as e:
            logger.error(f"Daemon error: {e}")
            time.sleep(poll_seconds)

def main():
    parser = argparse.ArgumentParser(description="OCR Worker Daemon")
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=int(os.environ.get("POLL_SECONDS", 3)),
        help="Seconds between job polls (default: 3)"
    )
    
    args = parser.parse_args()
    run_daemon(args.poll_seconds)

if __name__ == "__main__":
    main()
