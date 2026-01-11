# 🚀 RunPod Deployment Guide - Quick Start

## Prerequisites

✅ RunPod account dengan credit
✅ Credentials siap:
  - Supabase URL & Service Role Key
  - Cloudflare R2 credentials

---

## Step 1: Create RunPod GPU Instance

### 1.1 Login ke RunPod
https://www.runpod.io/console/pods

### 1.2 Select GPU
- **GPU Type:** RTX A5000 (24GB VRAM) - **$0.34/hour**
- **Container Image:** `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
- **Volume:** 50GB persistent storage (untuk model cache)
- **Ports:** None (worker connects outbound only)

### 1.3 Deploy Pod
Klik **"Deploy"** dan tunggu instance ready (~30 seconds)

---

## Step 2: Connect via SSH

### 2.1 Get SSH Command
Di RunPod console, klik **"Connect"** → copy SSH command seperti:
```bash
ssh root@ssh.runpod.io -p 12345 -i ~/.ssh/id_rsa
```

### 2.2 SSH ke Instance
```bash
ssh root@<your-pod-ssh-address>
```

---

## Step 3: Upload .env File

**Option A - SCP dari local machine:**
```bash
# Di local Windows/WSL, run:
scp .env root@<runpod-ssh>:/workspace/
```

**Option B - Create manual di RunPod:**
```bash
# Di RunPod terminal:
cd /workspace
nano .env
```

Paste content dari .env local Anda, lalu save (Ctrl+O, Enter, Ctrl+X)

---

## Step 4: Run Deployment Script

### 4.1 Download & Run Script
```bash
cd /workspace
wget https://raw.githubusercontent.com/BimBim-lab/chapterbridge-ocr-worker/main/runpod_quick_deploy.sh
chmod +x runpod_quick_deploy.sh
./runpod_quick_deploy.sh
```

Script akan:
- ✅ Install PaddlePaddle GPU
- ✅ Clone repository
- ✅ Install dependencies
- ✅ Verify GPU setup

**Expected output:**
```
PaddlePaddle: 3.0.0
GPU Count: 1
✅ Setup Complete!
```

---

## Step 5: Start Worker

### 5.1 Test Run (foreground)
```bash
cd /workspace/chapterbridge-ocr-worker
python3 workers/ocr/main.py --poll-seconds 3
```

**Expected log:**
```
2026-01-11 22:30:00 | INFO | Starting OCR worker daemon
2026-01-11 22:30:00 | INFO | Initializing PaddleOCR engine...
Creating model: ('PP-OCRv5_server_det', None)
2026-01-11 22:30:15 | INFO | PaddleOCR engine ready
2026-01-11 22:30:15 | INFO | PaddleOCR initialized: GPU (detected 1 device(s))
```

Press **Ctrl+C** to stop

### 5.2 Production Run (background)
```bash
# Run in background dengan nohup
nohup python3 workers/ocr/main.py --poll-seconds 3 > ocr.log 2>&1 &

# Save PID untuk stop nanti
echo $! > worker.pid
```

---

## Step 6: Monitor Performance

### 6.1 GPU Utilization
```bash
watch -n 1 nvidia-smi
```

**Expected GPU usage:**
- GPU Memory: 8-12GB / 24GB
- GPU Utilization: 85-95%
- Power: 200-250W

### 6.2 Worker Logs
```bash
# Real-time logs
tail -f ocr.log

# Last 50 lines
tail -n 50 ocr.log

# Search for errors
grep -i error ocr.log
```

### 6.3 Job Status
```bash
python3 check_jobs.py
```

---

## Step 7: Stop Worker

```bash
# If running in foreground: Ctrl+C

# If running in background:
kill $(cat worker.pid)

# Or find and kill:
ps aux | grep main.py
kill <PID>
```

---

## Performance Expectations

### Speed (per image):
- **Short (≤4000px):** 1-3 seconds ⚡
- **Medium (4001-15000px):** 3-10 seconds ⚡⚡
- **Long (>15000px):** 8-20 seconds ⚡⚡⚡

### Throughput:
- **~600-800 images/hour**
- **~15,000 images/day**
- **6M images in ~17 days** ($136 total cost)

### GPU Metrics:
- Auto-detects RTX A5000
- Uses PP-OCRv5 models
- Batch recognition (6 images)
- Adaptive tiling enabled

---

## Troubleshooting

### GPU Not Detected
```bash
# Check CUDA
nvidia-smi

# Verify PaddlePaddle sees GPU
python3 -c "import paddle; print(paddle.device.cuda.device_count())"
```

### Out of Memory
Edit `.env`:
```env
OCR_REC_BATCH_NUM=4  # Reduce from 6
OCR_TILE_HEIGHT_MED=2200  # Reduce from 2800
```

### Slow Performance
Check GPU usage:
```bash
nvidia-smi dmon -s mu
```

If GPU util < 50%, increase batch size:
```env
OCR_REC_BATCH_NUM=8
```

### Worker Crashes
Check logs:
```bash
tail -n 100 ocr.log
```

Common causes:
- Network timeout (R2/Supabase)
- Invalid image format
- Out of memory

---

## Cost Management

### RTX A5000 Pricing:
- **$0.34/hour** = ~$8.16/day
- Target: **6M images in 17 days** = **$136 total**

### Tips to Reduce Cost:
1. **Stop pod saat idle** (no queue)
2. **Use spot instances** (50% cheaper tapi bisa terminated)
3. **Batch process** (queue banyak jobs sekaligus)
4. **Monitor throughput** (pastikan GPU ≥85% utilized)

### Stop Pod:
```bash
# Di RunPod console:
Click "Stop Pod" → Billing pauses immediately
```

---

## Scaling to Multiple Workers

### Run 2+ workers on same GPU:
```bash
# Terminal 1
nohup python3 workers/ocr/main.py --poll-seconds 3 > ocr1.log 2>&1 &

# Terminal 2  
nohup python3 workers/ocr/main.py --poll-seconds 3 > ocr2.log 2>&1 &
```

**Note:** RTX A5000 can handle 2-3 workers with batch_num=4

### Or Multi-GPU Setup:
Rent multiple pods, each runs 1 worker.

---

## Quick Commands Cheat Sheet

```bash
# Start worker
python3 workers/ocr/main.py --poll-seconds 3

# Background
nohup python3 workers/ocr/main.py --poll-seconds 3 > ocr.log 2>&1 &

# Monitor GPU
watch -n 1 nvidia-smi

# View logs
tail -f ocr.log

# Check jobs
python3 check_jobs.py

# Stop worker
kill $(cat worker.pid)

# Update code
git pull && pip install -r requirements.txt

# Restart worker
kill $(cat worker.pid); nohup python3 workers/ocr/main.py --poll-seconds 3 > ocr.log 2>&1 & echo $! > worker.pid
```

---

## Need Help?

1. Check logs: `tail -f ocr.log`
2. Test GPU: `nvidia-smi`
3. Test PaddleOCR: `python3 -c "from workers.ocr.ocr_engine import get_ocr_instance; ocr = get_ocr_instance()"`
4. Check jobs: `python3 check_jobs.py`

**Ready to process 6 million images!** 🚀
