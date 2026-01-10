# ChapterBridge OCR Worker

Python worker daemon for processing manhwa page images through PaddleOCR, integrated with Supabase and Cloudflare R2.

## Architecture

This worker is part of the ChapterBridge pipeline. It:
1. Polls Supabase `pipeline_jobs` for queued OCR tasks
2. Downloads raw images from Cloudflare R2
3. Runs OCR using PaddleOCR
4. Uploads JSON results back to R2
5. Updates Supabase with asset records and job status

## Setup

### Environment Variables

Set these in Replit Secrets or a `.env` file:

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

R2_ENDPOINT=https://accountid.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your-access-key
R2_SECRET_ACCESS_KEY=your-secret-key
R2_BUCKET=chapterbridge-data

OCR_LANG=en
OCR_USE_ANGLE_CLS=true
POLL_SECONDS=3
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running the Worker Daemon

```bash
python workers/ocr/main.py --poll-seconds 3
```

The worker will:
- Initialize PaddleOCR once at startup
- Continuously poll for queued jobs
- Process one job at a time
- Handle errors gracefully without crashing

### Creating OCR Jobs

Use the enqueue script to create jobs for raw images:

```bash
# Process all raw images for a specific edition
python workers/ocr/enqueue.py --edition-id <uuid> --limit 500

# Process raw images matching a key prefix
python workers/ocr/enqueue.py --prefix raw/manhwa/work123/ed456 --limit 100

# Force re-processing even if output exists
python workers/ocr/enqueue.py --edition-id <uuid> --force
```

## Job Contract

OCR jobs use `job_type='clean'` with specific input format:

```json
{
  "task": "ocr_page",
  "raw_asset_id": "uuid-of-raw-image-asset",
  "force": false
}
```

## R2 Key Conventions

**Input (raw images):**
```
raw/manhwa/{work_id}/{edition_id}/chapter-0236/page-001.jpg
```

**Output (OCR JSON):**
```
derived/manhwa/{work_id}/{edition_id}/chapter-0236/ocr/page-001.json
```

## OCR Output Format

```json
{
  "metadata": {
    "work_id": "...",
    "edition_id": "...",
    "segment_id": "...",
    "chapter": 236,
    "page": 1,
    "source_key": "raw/manhwa/..."
  },
  "stats": {
    "line_count": 15
  },
  "lines": [
    {
      "text": "Detected text",
      "confidence": 0.98,
      "bbox": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    }
  ]
}
```

## Project Structure

```
workers/ocr/
  main.py              # Daemon loop poller
  enqueue.py           # Job creator script
  supabase_client.py   # Database operations
  r2_client.py         # R2 storage client
  ocr_engine.py        # PaddleOCR wrapper
  key_parser.py        # R2 key parsing utilities
  utils.py             # Logging, hashing utilities
```
