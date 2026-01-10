#!/usr/bin/env python3
"""
OCR Job Enqueue Script

Creates OCR jobs for raw images that don't yet have OCR output.

Usage:
    python workers/ocr/enqueue.py --edition-id <uuid> --limit 500
    python workers/ocr/enqueue.py --prefix raw/manhwa/... --limit 100
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from workers.ocr.utils import setup_logging
from workers.ocr.supabase_client import SupabaseDB
from workers.ocr.key_parser import build_output_key, parse_raw_key

logger = setup_logging("ocr_enqueue")

def enqueue_jobs(
    db: SupabaseDB,
    edition_id: str = None,
    prefix: str = None,
    limit: int = 500,
    force: bool = False
) -> int:
    """
    Find raw_image assets and create OCR jobs for those without output.
    Returns count of jobs created.
    """
    if edition_id:
        logger.info(f"Finding raw images for edition: {edition_id}")
        assets = db.get_raw_images_for_edition(edition_id, limit)
    elif prefix:
        logger.info(f"Finding raw images with prefix: {prefix}")
        assets = db.get_raw_images_by_prefix(prefix, limit)
    else:
        logger.error("Must specify --edition-id or --prefix")
        return 0
    
    logger.info(f"Found {len(assets)} raw image assets")
    
    created = 0
    skipped = 0
    
    for asset in assets:
        asset_id = asset["id"]
        r2_key = asset["r2_key"]
        
        output_key = build_output_key(r2_key, asset_id)
        
        if not force:
            existing = db.ocr_asset_exists(output_key)
            if existing:
                skipped += 1
                continue
        
        parsed = parse_raw_key(r2_key)
        
        segment_id = db.get_segment_id_for_asset(asset_id)
        
        job_edition_id = edition_id
        job_work_id = None
        
        if parsed.is_valid:
            job_edition_id = job_edition_id or parsed.edition_id
            job_work_id = parsed.work_id
        
        try:
            job_id = db.insert_ocr_job(
                raw_asset_id=asset_id,
                edition_id=job_edition_id,
                segment_id=segment_id,
                work_id=job_work_id,
                force=force
            )
            created += 1
            logger.info(f"Created job {job_id} for asset {asset_id}")
        except Exception as e:
            logger.error(f"Failed to create job for {asset_id}: {e}")
    
    logger.info(f"Done: created={created}, skipped={skipped}")
    return created

def main():
    parser = argparse.ArgumentParser(description="Enqueue OCR Jobs")
    parser.add_argument(
        "--edition-id",
        type=str,
        help="Edition UUID to process"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="R2 key prefix to match (e.g., raw/manhwa/...)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum jobs to create (default: 500)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create jobs even if OCR output exists"
    )
    
    args = parser.parse_args()
    
    if not args.edition_id and not args.prefix:
        parser.error("Must specify --edition-id or --prefix")
    
    db = SupabaseDB()
    count = enqueue_jobs(
        db,
        edition_id=args.edition_id,
        prefix=args.prefix,
        limit=args.limit,
        force=args.force
    )
    
    sys.exit(0 if count >= 0 else 1)

if __name__ == "__main__":
    main()
