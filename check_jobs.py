#!/usr/bin/env python3
"""Quick script to check OCR job status"""
from dotenv import load_dotenv
load_dotenv()

from workers.ocr.supabase_client import SupabaseDB

db = SupabaseDB()

# Get recent jobs
jobs = db.client.table('pipeline_jobs').select('*').order('created_at', desc=True).limit(10).execute()

print(f"\n=== Recent Pipeline Jobs (last 10) ===\n")
for j in jobs.data:
    status = j['status']
    job_id = j['id'][:8]
    created = j['created_at'][:19]
    job_type = j.get('job_type', 'N/A')
    print(f"{status:12} | {job_type:10} | {job_id}... | {created}")

# Count by status
statuses = {}
for j in jobs.data:
    s = j['status']
    statuses[s] = statuses.get(s, 0) + 1

print(f"\n=== Status Counts (of recent 10) ===")
for status, count in statuses.items():
    print(f"  {status}: {count}")
