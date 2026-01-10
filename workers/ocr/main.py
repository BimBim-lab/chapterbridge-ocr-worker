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
import argparse
import traceback
from dotenv import load_dotenv

load_dotenv()

from workers.ocr.utils import setup_logging, compute_sha256, json_dumps
from workers.ocr.supabase_client import SupabaseDB
from workers.ocr.r2_client import R2Client
from workers.ocr.ocr_engine import run_ocr, build_ocr_output
from workers.ocr.key_parser import (
    parse_raw_key, 
    build_output_key, 
    extract_chapter_number, 
    extract_page_number
)

logger = setup_logging("ocr_worker")

def process_job(db: SupabaseDB, r2: R2Client, job: dict) -> None:
    """
    Process a single OCR job.
    
    Steps:
    1. Mark job running
    2. Load and validate raw asset
    3. Resolve segment_id, work_id, edition_id
    4. Check idempotency (skip if output exists)
    5. Download image from R2
    6. Run OCR
    7. Upload result to R2
    8. Insert asset record and link to segment
    9. Mark job success
    """
    job_id = job["id"]
    job_input = job.get("input", {})
    raw_asset_id = job_input.get("raw_asset_id")
    force = job_input.get("force", False)
    
    logger.info(f"Processing job {job_id} | raw_asset_id={raw_asset_id}")
    
    db.set_job_running(job_id, job.get("attempt", 0))
    
    raw_asset = db.get_asset(raw_asset_id)
    if not raw_asset:
        raise ValueError(f"Raw asset not found: {raw_asset_id}")
    
    if raw_asset.get("asset_type") != "raw_image":
        raise ValueError(f"Asset is not raw_image: {raw_asset.get('asset_type')}")
    
    raw_r2_key = raw_asset.get("r2_key")
    if not raw_r2_key:
        raise ValueError(f"Asset has no r2_key: {raw_asset_id}")
    
    segment_id = job.get("segment_id")
    if not segment_id:
        segment_id = db.get_segment_id_for_asset(raw_asset_id)
    
    if not segment_id:
        raise ValueError(f"Cannot resolve segment_id for asset {raw_asset_id}")
    
    logger.info(f"  segment_id={segment_id}")
    
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
    
    logger.info(f"  work_id={work_id}, edition_id={edition_id}")
    
    output_r2_key = build_output_key(raw_r2_key, raw_asset_id)
    logger.info(f"  output_r2_key={output_r2_key}")
    
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
            logger.info(f"  Skipped - OCR already exists: {existing_ocr_id}")
            return
    
    logger.info(f"  Downloading image from R2: {raw_r2_key}")
    image_bytes = r2.get_object(raw_r2_key)
    logger.info(f"  Downloaded {len(image_bytes)} bytes")
    
    logger.info("  Running OCR...")
    lines = run_ocr(image_bytes)
    logger.info(f"  OCR complete: {len(lines)} lines detected")
    
    chapter_num = extract_chapter_number(parsed.chapter) if parsed.chapter else None
    page_num = extract_page_number(parsed.page) if parsed.page else None
    
    ocr_output = build_ocr_output(
        lines=lines,
        work_id=work_id,
        edition_id=edition_id,
        segment_id=segment_id,
        chapter=chapter_num,
        page=page_num,
        raw_r2_key=raw_r2_key
    )
    
    json_bytes = json_dumps(ocr_output)
    sha256_hash = compute_sha256(json_bytes)
    
    logger.info(f"  Uploading OCR JSON to R2: {output_r2_key}")
    r2.put_object(output_r2_key, json_bytes, "application/json")
    
    logger.info("  Inserting asset record...")
    ocr_asset_id = db.insert_asset(
        r2_key=output_r2_key,
        bucket=r2.bucket,
        asset_type="ocr_json",
        content_type="application/json",
        byte_size=len(json_bytes),
        sha256=sha256_hash
    )
    logger.info(f"  Created asset: {ocr_asset_id}")
    
    db.link_segment_asset(segment_id, ocr_asset_id, "ocr_json")
    logger.info("  Linked asset to segment")
    
    db.set_job_success(job_id, {
        "task": "ocr_page",
        "raw_asset_id": raw_asset_id,
        "raw_r2_key": raw_r2_key,
        "ocr_asset_id": ocr_asset_id,
        "ocr_r2_key": output_r2_key,
        "lines": len(lines)
    })
    
    logger.info(f"Job {job_id} completed successfully")

def run_daemon(poll_seconds: int) -> None:
    """Main daemon loop that polls for and processes jobs."""
    logger.info(f"Starting OCR worker daemon (poll interval: {poll_seconds}s)")
    
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
                time.sleep(poll_seconds)
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
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
