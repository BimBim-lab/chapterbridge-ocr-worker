"""
Supabase client and database helper functions.
"""
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from supabase import create_client, Client


def utc_now() -> str:
    """Return current UTC time as ISO string for Supabase."""
    return datetime.now(timezone.utc).isoformat()

class SupabaseDB:
    """Database operations wrapper for Supabase."""
    
    def __init__(self):
        """Initialize Supabase client from environment variables."""
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        
        if not url or not key:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY environment variables")
        
        self.client: Client = create_client(url, key)
    
    def claim_ocr_job(self) -> Optional[Dict[str, Any]]:
        """
        Atomically claim one queued OCR job using UPDATE ... RETURNING.
        Uses RPC to avoid race conditions between multiple workers.
        Returns the claimed job dict or None if no jobs available.
        """
        try:
            response = self.client.rpc("claim_ocr_job", {}).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
        except Exception:
            pass
        return self._poll_and_claim_ocr_job()
    
    def _poll_and_claim_ocr_job(self) -> Optional[Dict[str, Any]]:
        """
        Fallback: Poll for one queued OCR job and atomically claim it.
        Uses optimistic locking pattern - update only if still queued.
        """
        poll_response = self.client.table("pipeline_jobs").select("id, attempt").eq(
            "status", "queued"
        ).eq(
            "job_type", "clean"
        ).filter(
            "input->>task", "eq", "ocr_page"
        ).order(
            "created_at", desc=False
        ).limit(1).execute()
        
        if not poll_response.data or len(poll_response.data) == 0:
            return None
        
        job_id = poll_response.data[0]["id"]
        current_attempt = poll_response.data[0].get("attempt") or 0
        
        claim_response = self.client.table("pipeline_jobs").update({
            "status": "running",
            "started_at": utc_now(),
            "attempt": current_attempt + 1
        }).eq(
            "id", job_id
        ).eq(
            "status", "queued"
        ).execute()
        
        if not claim_response.data or len(claim_response.data) == 0:
            return None
        
        job_response = self.client.table("pipeline_jobs").select("*").eq(
            "id", job_id
        ).execute()
        
        if job_response.data and len(job_response.data) > 0:
            return job_response.data[0]
        return None
    
    def poll_ocr_job(self) -> Optional[Dict[str, Any]]:
        """
        Poll for and atomically claim one queued OCR job.
        Returns the claimed job dict or None if no jobs available.
        """
        return self.claim_ocr_job()
    
    def set_job_success(self, job_id: str, output: Dict[str, Any]) -> None:
        """Mark job as success with output data."""
        self.client.table("pipeline_jobs").update({
            "status": "success",
            "finished_at": utc_now(),
            "output": output
        }).eq("id", job_id).execute()
    
    def set_job_failed(self, job_id: str, error: str) -> None:
        """Mark job as failed with error message."""
        self.client.table("pipeline_jobs").update({
            "status": "failed",
            "finished_at": utc_now(),
            "error": error[:10000]
        }).eq("id", job_id).execute()
    
    def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Fetch asset by ID."""
        response = self.client.table("assets").select("*").eq("id", asset_id).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    
    def get_asset_by_r2_key(self, r2_key: str) -> Optional[Dict[str, Any]]:
        """Fetch asset by R2 key."""
        response = self.client.table("assets").select("*").eq("r2_key", r2_key).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    
    def get_segment_id_for_asset(self, asset_id: str) -> Optional[str]:
        """Look up segment_id from segment_assets junction table."""
        response = self.client.table("segment_assets").select("segment_id").eq(
            "asset_id", asset_id
        ).limit(1).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]["segment_id"]
        return None
    
    def get_segment_edition_work(self, segment_id: str) -> Optional[Dict[str, str]]:
        """
        Get edition_id and work_id for a segment.
        Uses two queries to avoid nested join complexities.
        Returns dict with 'edition_id' and 'work_id' keys, or None.
        """
        seg_response = self.client.table("segments").select(
            "edition_id"
        ).eq("id", segment_id).limit(1).execute()
        
        if not seg_response.data or len(seg_response.data) == 0:
            return None
        
        edition_id = seg_response.data[0].get("edition_id")
        if not edition_id:
            return None
        
        ed_response = self.client.table("editions").select(
            "work_id"
        ).eq("id", edition_id).limit(1).execute()
        
        work_id = None
        if ed_response.data and len(ed_response.data) > 0:
            work_id = ed_response.data[0].get("work_id")
        
        return {"edition_id": edition_id, "work_id": work_id}
    
    def insert_asset(
        self,
        r2_key: str,
        bucket: str,
        asset_type: str,
        content_type: str,
        byte_size: int,
        sha256: str
    ) -> str:
        """
        Insert a new asset record and return its ID.
        """
        response = self.client.table("assets").insert({
            "provider": "cloudflare_r2",
            "bucket": bucket,
            "r2_key": r2_key,
            "asset_type": asset_type,
            "content_type": content_type,
            "bytes": byte_size,
            "sha256": sha256,
            "upload_source": "pipeline"
        }).execute()
        
        return response.data[0]["id"]
    
    def link_segment_asset(self, segment_id: str, asset_id: str, role: str) -> None:
        """
        Link an asset to a segment. Ignores conflict if already exists.
        """
        self.client.table("segment_assets").upsert({
            "segment_id": segment_id,
            "asset_id": asset_id,
            "role": role
        }, on_conflict="segment_id,asset_id").execute()
    
    def get_raw_images_for_edition(self, edition_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Get raw_image assets for an edition that don't yet have OCR output.
        Used by enqueue script.
        """
        response = self.client.table("assets").select(
            "id, r2_key"
        ).eq(
            "asset_type", "raw_image"
        ).like(
            "r2_key", f"%/{edition_id}/%"
        ).limit(limit).execute()
        
        return response.data if response.data else []
    
    def get_raw_images_by_prefix(self, prefix: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Get raw_image assets matching a key prefix."""
        response = self.client.table("assets").select(
            "id, r2_key"
        ).eq(
            "asset_type", "raw_image"
        ).like(
            "r2_key", f"{prefix}%"
        ).limit(limit).execute()
        
        return response.data if response.data else []
    
    def ocr_asset_exists(self, output_r2_key: str) -> Optional[str]:
        """Check if OCR asset already exists, return its ID if so."""
        asset = self.get_asset_by_r2_key(output_r2_key)
        if asset:
            return asset["id"]
        return None
    
    def insert_ocr_job(
        self,
        raw_asset_id: str,
        edition_id: Optional[str] = None,
        segment_id: Optional[str] = None,
        work_id: Optional[str] = None,
        force: bool = False
    ) -> str:
        """Insert a new OCR job and return its ID."""
        job_input = {
            "task": "ocr_page",
            "raw_asset_id": raw_asset_id,
            "force": force
        }
        
        insert_data = {
            "job_type": "clean",
            "status": "queued",
            "input": job_input
        }
        
        if edition_id:
            insert_data["edition_id"] = edition_id
        if segment_id:
            insert_data["segment_id"] = segment_id
        if work_id:
            insert_data["work_id"] = work_id
        
        response = self.client.table("pipeline_jobs").insert(insert_data).execute()
        return response.data[0]["id"]
