#!/usr/bin/env python3
"""
Cleanup Duplicate OCR Jobs

Removes duplicate queued OCR jobs, keeping only the oldest job for each asset.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from workers.ocr.utils import setup_logging
from workers.ocr.supabase_client import SupabaseDB

logger = setup_logging("cleanup_duplicates")

def cleanup_duplicate_jobs(db: SupabaseDB, edition_id: str):
    """Remove duplicate queued jobs for an edition."""
    logger.info(f"Finding duplicate jobs for edition: {edition_id}")
    
    # Get all queued jobs for edition with pagination
    all_jobs = []
    batch_size = 1000
    offset = 0
    
    while True:
        response = db.client.table("pipeline_jobs").select(
            "id, input, created_at"
        ).eq(
            "edition_id", edition_id
        ).eq(
            "job_type", "clean"
        ).eq(
            "status", "queued"
        ).filter(
            "input->>task", "eq", "ocr_page"
        ).order("created_at").range(offset, offset + batch_size - 1).execute()
        
        if not response.data:
            break
        
        all_jobs.extend(response.data)
        
        if len(response.data) < batch_size:
            break
        
        offset += batch_size
    
    if not all_jobs:
        logger.info("No queued jobs found")
        return
    
    logger.info(f"Found {len(all_jobs)} queued jobs")
    
    # Group by asset_id
    asset_jobs = {}
    for job in all_jobs:
        asset_id = job["input"].get("raw_asset_id")
        if asset_id:
            if asset_id not in asset_jobs:
                asset_jobs[asset_id] = []
            asset_jobs[asset_id].append(job)
    
    # Find duplicates
    jobs_to_delete = []
    for asset_id, jobs in asset_jobs.items():
        if len(jobs) > 1:
            # Keep the oldest, delete the rest
            jobs_sorted = sorted(jobs, key=lambda x: x["created_at"])
            duplicates = jobs_sorted[1:]  # All except the first
            jobs_to_delete.extend([j["id"] for j in duplicates])
            logger.info(f"Asset {asset_id}: keeping 1 job, deleting {len(duplicates)} duplicates")
    
    if not jobs_to_delete:
        logger.info("No duplicates found!")
        return
    
    logger.info(f"Deleting {len(jobs_to_delete)} duplicate jobs...")
    
    # Delete in batches
    batch_size = 500
    deleted = 0
    for i in range(0, len(jobs_to_delete), batch_size):
        batch = jobs_to_delete[i:i + batch_size]
        db.client.table("pipeline_jobs").delete().in_("id", batch).execute()
        deleted += len(batch)
        logger.info(f"Deleted {deleted}/{len(jobs_to_delete)} jobs...")
    
    logger.info(f"Cleanup complete! Deleted {deleted} duplicate jobs")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cleanup duplicate OCR jobs")
    parser.add_argument("--edition-id", type=str, required=True, help="Edition UUID")
    args = parser.parse_args()
    
    db = SupabaseDB()
    cleanup_duplicate_jobs(db, args.edition_id)

if __name__ == "__main__":
    main()
