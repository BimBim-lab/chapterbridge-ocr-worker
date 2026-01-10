"""
Supabase client and database helper functions.
"""
import os
from typing import Optional, Dict, Any, List
from supabase import create_client, Client

class SupabaseDB:
    """Database operations wrapper for Supabase."""
    
    def __init__(self):
        """Initialize Supabase client from environment variables."""
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        
        if not url or not key:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY environment variables")
        
        self.client: Client = create_client(url, key)
    
    def poll_ocr_job(self) -> Optional[Dict[str, Any]]:
        """
        Poll for one queued OCR job.
        Returns the job dict or None if no jobs available.
        """
        response = self.client.table("pipeline_jobs").select("*").eq(
            "status", "queued"
        ).eq(
            "job_type", "clean"
        ).filter(
            "input->>task", "eq", "ocr_page"
        ).order(
            "created_at", desc=False
        ).limit(1).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    
    def set_job_running(self, job_id: str, current_attempt: int) -> None:
        """Mark job as running and increment attempt counter."""
        self.client.table("pipeline_jobs").update({
            "status": "running",
            "started_at": "now()",
            "attempt": current_attempt + 1
        }).eq("id", job_id).execute()
    
    def set_job_success(self, job_id: str, output: Dict[str, Any]) -> None:
        """Mark job as success with output data."""
        self.client.table("pipeline_jobs").update({
            "status": "success",
            "finished_at": "now()",
            "output": output
        }).eq("id", job_id).execute()
    
    def set_job_failed(self, job_id: str, error: str) -> None:
        """Mark job as failed with error message."""
        self.client.table("pipeline_jobs").update({
            "status": "failed",
            "finished_at": "now()",
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
        Returns dict with 'edition_id' and 'work_id' keys, or None.
        """
        response = self.client.table("segments").select(
            "edition_id, editions(work_id)"
        ).eq("id", segment_id).execute()
        
        if response.data and len(response.data) > 0:
            row = response.data[0]
            edition_id = row.get("edition_id")
            work_id = None
            if row.get("editions"):
                work_id = row["editions"].get("work_id")
            return {"edition_id": edition_id, "work_id": work_id}
        return None
    
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
