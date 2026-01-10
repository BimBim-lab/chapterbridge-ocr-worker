# ChapterBridge OCR Worker

## Overview
Python OCR worker daemon for the ChapterBridge pipeline. Processes manhwa page images using PaddleOCR and stores results in Cloudflare R2.

## Project Structure
```
workers/ocr/
  main.py              - Main daemon loop
  enqueue.py           - Job creation script
  supabase_client.py   - Supabase database client
  r2_client.py         - Cloudflare R2 client
  ocr_engine.py        - PaddleOCR singleton wrapper
  key_parser.py        - R2 key parsing utilities
  utils.py             - Logging and hashing helpers
```

## Required Secrets
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Service role key
- `R2_ENDPOINT` - Cloudflare R2 endpoint
- `R2_ACCESS_KEY_ID` - R2 access key
- `R2_SECRET_ACCESS_KEY` - R2 secret key
- `R2_BUCKET` - R2 bucket name (default: chapterbridge-data)

## Optional Environment Variables
- `OCR_LANG` - OCR language (default: en)
- `OCR_USE_ANGLE_CLS` - Enable angle classification (default: true)
- `POLL_SECONDS` - Job polling interval (default: 3)

## Database
Uses external Supabase with tables: pipeline_jobs, assets, segment_assets, segments, editions

## Job Contract
OCR jobs use job_type='clean' with input.task='ocr_page'

## Recent Changes
- January 2026: Initial implementation of OCR worker
- January 2026: Fixed PaddleOCR 3.x API compatibility (use_textline_orientation instead of use_angle_cls)
- January 2026: Added required system libraries (libGL, libGLU, libgcc) for OpenCV/PaddlePaddle
