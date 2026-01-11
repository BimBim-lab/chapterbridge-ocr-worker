#!/bin/bash
# Quick deployment script for RunPod
# Run this after SSH-ing into RunPod instance

set -e  # Exit on error

echo "=== ChapterBridge OCR Worker - RunPod Quick Deploy ==="
echo ""

# 1. Update system
echo "[1/6] Updating system..."
apt-get update -qq
apt-get install -y python3-pip git wget curl -qq

# 2. Check GPU
echo ""
echo "[2/6] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 3. Install PaddlePaddle GPU
echo ""
echo "[3/6] Installing PaddlePaddle GPU (CUDA 11.8)..."
pip install -q paddlepaddle-gpu==3.0.0.b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 4. Clone/Update repository
echo ""
echo "[4/6] Setting up repository..."
if [ ! -d "chapterbridge-ocr-worker" ]; then
    git clone https://github.com/BimBim-lab/chapterbridge-ocr-worker.git
else
    cd chapterbridge-ocr-worker && git pull && cd ..
fi

cd chapterbridge-ocr-worker

# 5. Install dependencies
echo ""
echo "[5/6] Installing dependencies..."
pip install -q -r requirements.txt

# 6. Setup environment
echo ""
echo "[6/6] Setting up environment..."

# Check if .env exists
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  No .env file found!"
    echo ""
    echo "Please create .env file with your credentials:"
    echo ""
    echo "Required variables:"
    echo "  - SUPABASE_URL"
    echo "  - SUPABASE_SERVICE_ROLE_KEY"
    echo "  - R2_ENDPOINT"
    echo "  - R2_ACCESS_KEY_ID"
    echo "  - R2_SECRET_ACCESS_KEY"
    echo "  - R2_BUCKET"
    echo ""
    echo "You can either:"
    echo "  1. Copy from local: scp .env runpod:/workspace/chapterbridge-ocr-worker/"
    echo "  2. Or edit manually: nano .env"
    echo ""
    read -p "Press Enter to open nano editor to create .env..."
    nano .env
fi

# Verify GPU setup
echo ""
echo "Verifying GPU setup..."
python3 -c "import paddle; print(f'PaddlePaddle: {paddle.__version__}'); print(f'GPU Count: {paddle.device.cuda.device_count()}')"

echo ""
echo "=== ✅ Setup Complete! ==="
echo ""
echo "To start the worker:"
echo "  python3 workers/ocr/main.py --poll-seconds 3"
echo ""
echo "Or run in background:"
echo "  nohup python3 workers/ocr/main.py --poll-seconds 3 > ocr.log 2>&1 &"
echo ""
echo "Monitor GPU:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "View logs:"
echo "  tail -f ocr.log"
