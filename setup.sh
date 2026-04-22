#!/usr/bin/env bash
# Maxine Eye Contact Webcam Pipeline - Ubuntu 25 Setup (uv workflow)
set -euo pipefail

echo "============================================"
echo "  Setup: Maxine Eye Contact Webcam Pipeline"
echo "  Ubuntu 25.04+ | RTX 4090 Optimized"
echo "  Using astral.sh uv"
echo "============================================"

# ---------------------------------------------------------------------------
# 0. Ensure uv is installed
# ---------------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo ""
    echo "[0/6] Installing uv (astral.sh)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Ensure uv is on PATH for this shell
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[1/6] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    v4l2loopback-dkms \
    v4l2loopback-utils \
    v4l-utils \
    git \
    wget \
    linux-headers-$(uname -r)

# ---------------------------------------------------------------------------
# 2. v4l2loopback kernel module (auto-load on boot)
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Configuring v4l2loopback virtual webcam..."

if ! lsmod | grep -q v4l2loopback; then
    echo "      Loading v4l2loopback kernel module..."
    sudo modprobe v4l2loopback \
        devices=1 \
        video_nr=10 \
        card_label="MaxineEyeContact" \
        exclusive_caps=1 \
        max_buffers=4
else
    echo "      v4l2loopback already loaded"
fi

V4L2_CONF="/etc/modprobe.d/v4l2loopback.conf"
if [ ! -f "$V4L2_CONF" ]; then
    echo "      Creating persistent config..."
    sudo tee "$V4L2_CONF" > /dev/null <<'EOF'
options v4l2loopback devices=1 video_nr=10 card_label="MaxineEyeContact" exclusive_caps=1 max_buffers=4
EOF
fi

MODULES_LOAD="/etc/modules-load.d/v4l2loopback.conf"
if [ ! -f "$MODULES_LOAD" ]; then
    echo "      Enabling auto-load on boot..."
    echo "v4l2loopback" | sudo tee "$MODULES_LOAD" > /dev/null
fi

if [ -e /dev/video10 ]; then
    echo "      Virtual webcam ready: /dev/video10"
    v4l2-ctl -d /dev/video10 --all 2>/dev/null | head -5
else
    echo "      WARNING: /dev/video10 not found. Available devices:"
    ls -la /dev/video* 2>/dev/null || echo "      (none found)"
fi

# ---------------------------------------------------------------------------
# 3. NVIDIA runtime check
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Checking NVIDIA runtime..."

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "      WARNING: nvidia-smi not found. Install NVIDIA drivers first:"
    echo "      https://developer.nvidia.com/cuda-downloads?target_os=Linux"
fi

if command -v docker &> /dev/null; then
    if docker info 2>/dev/null | grep -q nvidia; then
        echo "      Docker NVIDIA runtime: OK"
    else
        echo "      WARNING: Docker nvidia runtime not configured."
        echo "      Install nvidia-container-toolkit:"
        echo "        https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
else
    echo "      WARNING: Docker not found. You need Docker to run the NIM."
fi

# ---------------------------------------------------------------------------
# 4. Python project setup (uv-managed Python + venv)
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Setting up uv project environment..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure a managed Python version is available and pinned
uv python install 3.12
if [ ! -f ".python-version" ]; then
    uv python pin 3.12
fi

# Create venv and sync dependencies from pyproject.toml + uv.lock
uv sync

# ---------------------------------------------------------------------------
# 5. Build protobuf definitions
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Building gRPC protobuf definitions..."
if [ ! -f "eyecontact_pb2.py" ] || [ ! -f "eyecontact_pb2_grpc.py" ]; then
    bash build_proto.sh
else
    echo "      Protobuf files already exist, skipping build"
    echo "      (Run ./build_proto.sh to force rebuild)"
fi

# ---------------------------------------------------------------------------
# 6. Verify setup
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Verifying setup..."
uv run maxine_webcam_pipeline.py --help > /dev/null 2>&1 && echo "      Pipeline script: OK" || echo "      Pipeline script: MISSING DEPS"

FFMPEG_V=$(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')
if [ -n "$FFMPEG_V" ]; then
    if ffmpeg -encoders 2>/dev/null | grep -q h264_nvenc; then
        echo "      FFmpeg: $FFMPEG_V (NVENC: YES)"
    else
        echo "      FFmpeg: $FFMPEG_V (NVENC: NO - install with --enable-nvenc)"
    fi
else
    echo "      FFmpeg: MISSING"
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Ensure NIM docker is running:"
echo "       docker run -it --rm --name=maxine-eye-contact-nim \\"
echo "         --runtime=nvidia --gpus all --shm-size=6GB \\"
echo "         -e NGC_API_KEY=\$NGC_API_KEY \\"
echo "         -e MAXINE_MAX_CONCURRENCY_PER_GPU=1 \\"
echo "         -p 8002:8000 -p 8003:8001 \\"
echo "         nvcr.io/nim/nvidia/maxine-eye-contact:latest"
echo ""
echo "    2. Run the pipeline with uv:"
echo "       uv run maxine_webcam_pipeline.py --target 127.0.0.1:8003"
echo ""
echo "    3. Select 'MaxineEyeContact' in your video conferencing app"
echo "============================================"