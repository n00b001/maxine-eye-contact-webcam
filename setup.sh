#!/usr/bin/env bash
# Maxine Eye Contact Webcam – One-Command Setup
# Pulls the pre-built GHCR image and configures the host for native AR SDK pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GHCR_IMAGE="ghcr.io/n00b001/maxine-eye-contact-webcam/arsdk-gaze:latest"
SERVICE_NAME="maxine-ar-sdk-webcam.service"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo ""; echo "[INFO] $*"; }
ok()    { echo "  ✓ $*"; }
warn()  { echo "  ⚠ $*"; }
fail()  { echo "  ✗ $*"; exit 1; }

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
info "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    ffmpeg \
    v4l2loopback-dkms \
    v4l2loopback-utils \
    v4l-utils \
    curl \
    linux-headers-$(uname -r) \
    || fail "Failed to install system dependencies"
ok "System dependencies installed"

# ---------------------------------------------------------------------------
# 2. Virtual webcam (v4l2loopback) – create & persist
# ---------------------------------------------------------------------------
info "Configuring v4l2loopback virtual webcam..."

# Load the module now (if not already loaded)
if ! lsmod | grep -q "^v4l2loopback"; then
    sudo modprobe v4l2loopback \
        devices=1 \
        video_nr=10 \
        card_label="MaxineEyeContact" \
        exclusive_caps=1 \
        max_buffers=4
    ok "v4l2loopback kernel module loaded"
else
    ok "v4l2loopback already loaded"
fi

# Persistent modprobe config (survives reboots)
V4L2_CONF="/etc/modprobe.d/v4l2loopback.conf"
if [ ! -f "$V4L2_CONF" ]; then
    sudo tee "$V4L2_CONF" > /dev/null <<'EOF'
options v4l2loopback devices=1 video_nr=10 card_label="MaxineEyeContact" exclusive_caps=1 max_buffers=4
EOF
    ok "Created $V4L2_CONF (persistent module options)"
else
    ok "$V4L2_CONF already exists"
fi

# Auto-load on boot
MODULES_LOAD="/etc/modules-load.d/v4l2loopback.conf"
if [ ! -f "$MODULES_LOAD" ]; then
    echo "v4l2loopback" | sudo tee "$MODULES_LOAD" > /dev/null
    ok "Created $MODULES_LOAD (auto-load on boot)"
else
    ok "$MODULES_LOAD already exists"
fi

# Verify device exists
if [ -e /dev/video10 ]; then
    ok "Virtual webcam ready: /dev/video10"
else
    warn "/dev/video10 not found. Available devices:"
    ls -la /dev/video* 2>/dev/null | sed 's/^/        /' || true
fi

# ---------------------------------------------------------------------------
# 3. NVIDIA runtime check
# ---------------------------------------------------------------------------
info "Checking NVIDIA GPU & Docker runtime..."

if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true)
    ok "GPU: $GPU_INFO"
else
    warn "nvidia-smi not found. Install NVIDIA drivers first:"
    warn "  https://developer.nvidia.com/cuda-downloads?target_os=Linux"
fi

if command -v docker &> /dev/null; then
    if docker info 2>/dev/null | grep -q nvidia; then
        ok "Docker NVIDIA runtime: OK"
    else
        warn "Docker NVIDIA runtime not configured."
        warn "  Install nvidia-container-toolkit:"
        warn "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
else
    fail "Docker not found. Install Docker before running this script."
fi

# ---------------------------------------------------------------------------
# 4. Pull pre-built image from GHCR
# ---------------------------------------------------------------------------
info "Pulling pre-built image from GHCR..."
echo "  Image: $GHCR_IMAGE"

# Check if already logged in to ghcr.io; if not, prompt
if ! docker info 2>/dev/null | grep -q "ghcr.io"; then
    warn "You may need to log in to GHCR first:"
    warn "  echo \$GITHUB_TOKEN | docker login ghcr.io -u \$USER --password-stdin"
fi

docker pull "$GHCR_IMAGE" || fail "Failed to pull $GHCR_IMAGE"
ok "Image pulled: $GHCR_IMAGE"

# ---------------------------------------------------------------------------
# 5. Install systemd service
# ---------------------------------------------------------------------------
info "Installing systemd service..."

# Update the service file in-place to use the GHCR image
sed -i "s|arsdk-gaze:latest|$GHCR_IMAGE|g" "$SCRIPT_DIR/$SERVICE_NAME"

sudo cp "$SCRIPT_DIR/$SERVICE_NAME" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
ok "Service installed: $SERVICE_NAME"

# ---------------------------------------------------------------------------
# 6. Start the pipeline
# ---------------------------------------------------------------------------
info "Starting the gaze redirection pipeline..."
sudo systemctl start "$SERVICE_NAME"
ok "Pipeline started"

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Virtual webcam: /dev/video10"
echo "  Image:          $GHCR_IMAGE"
echo "  Service:        $SERVICE_NAME"
echo ""
echo "  Commands:"
echo "    Start:   sudo systemctl start $SERVICE_NAME"
echo "    Stop:    sudo systemctl stop  $SERVICE_NAME"
echo "    Status:  sudo systemctl status $SERVICE_NAME"
echo "    Logs:    sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo "  Select 'MaxineEyeContact' in Zoom, Teams, OBS, Chrome, etc."
echo "============================================"
