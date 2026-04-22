#!/usr/bin/env bash
# Quick start script - assumes setup.sh has already been run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# Ensure project environment is ready
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found. Run ./setup.sh first."
    exit 1
fi

# Check protobufs
if [ ! -f "eyecontact_pb2.py" ] || [ ! -f "eyecontact_pb2_grpc.py" ]; then
    echo "Protobuf files not found. Running build_proto.sh..."
    bash build_proto.sh
fi

# Check v4l2loopback
if [ ! -e /dev/video10 ]; then
    echo "Loading v4l2loopback kernel module..."
    sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="MaxineEyeContact" exclusive_caps=1 max_buffers=4
fi

# Parse target from args or use default
TARGET="${1:-127.0.0.1:8003}"
CAMERA="${2:-/dev/video0}"

echo "============================================"
echo "  Quick Start: Maxine Eye Contact Webcam"
echo "============================================"
echo "  NIM Target:   $TARGET"
echo "  Camera:       $CAMERA"
echo "  Virtual Cam:  /dev/video10"
echo "  Resolution:   1920x1080 @ 30fps"
echo "  Press Ctrl+C to stop"
echo "============================================"

uv run maxine_webcam_pipeline.py \
    --target "$TARGET" \
    --camera "$CAMERA" \
    --width 1920 \
    --height 1080 \
    --fps 30 \
    --v4l2-device /dev/video10 \
    --output-bitrate 8000000
