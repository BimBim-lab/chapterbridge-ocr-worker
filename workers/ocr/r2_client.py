"""
Cloudflare R2 client using boto3 S3 API.
"""
import os
import boto3
from botocore.config import Config

class R2Client:
    """Wrapper around boto3 S3 client for Cloudflare R2 operations."""
    
    def __init__(self):
        """Initialize R2 client from environment variables."""
        self.endpoint = os.environ.get("R2_ENDPOINT")
        self.access_key = os.environ.get("R2_ACCESS_KEY_ID")
        self.secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        self.bucket = os.environ.get("R2_BUCKET", "chapterbridge-data")
        
        if not all([self.endpoint, self.access_key, self.secret_key]):
            raise ValueError("Missing R2 environment variables: R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY")
        
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "adaptive"}
            )
        )
    
    def get_object(self, key: str) -> bytes:
        """
        Download object from R2 and return bytes.
        Raises exception if object doesn't exist.
        """
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()
    
    def put_object(self, key: str, body: bytes, content_type: str = "application/json") -> None:
        """Upload bytes to R2 at specified key."""
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType=content_type
        )
    
    def head_object(self, key: str) -> dict:
        """
        Get object metadata without downloading content.
        Returns response dict or raises ClientError if not found.
        """
        return self.client.head_object(Bucket=self.bucket, Key=key)
    
    def object_exists(self, key: str) -> bool:
        """Check if an object exists in the bucket."""
        try:
            self.head_object(key)
            return True
        except self.client.exceptions.ClientError:
            return False
