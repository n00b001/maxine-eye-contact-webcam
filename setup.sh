#!/usr/bin/env bash
# Maxine Eye Contact Webcam Pipeline – Unified Setup
# Sets up v4l2loopback virtual webcam, system deps, and both pipeline options.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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
    git \
    wget \
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
    v4l2-ctl -d /dev/video10 --all 2>/dev/null | head -3 | sed 's/^/        /'
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
    warn "nvidia-smi not found. Install NVIDIA drivers:"
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
    warn "Docker not found. Required for both pipelines."
fi

# ---------------------------------------------------------------------------
# 4. Native AR SDK Docker pipeline (optional)
# ---------------------------------------------------------------------------
info "Checking for NVIDIA AR SDK..."

if [ -d "/usr/local/ARSDK/lib" ]; then
    ok "ARSDK found at /usr/local/ARSDK"

    if [ -d "$SCRIPT_DIR/docker/arsdk" ]; then
        ok "Build context already has ARSDK copy"
    else
        info "Copying ARSDK into docker build context (~5 GB, hard-linked)..."
        cp -rl /usr/local/ARSDK "$SCRIPT_DIR/docker/arsdk"
        ok "ARSDK copied to docker/arsdk/"
    fi

    info "Building Docker image 'arsdk-gaze:latest'..."
    cd "$SCRIPT_DIR/docker"
    docker build -t arsdk-gaze:latest . || warn "Docker build failed (check ARSDK integrity)"
    cd "$SCRIPT_DIR"
    ok "Docker image 'arsdk-gaze:latest' ready"

    info "Installing systemd service for native pipeline..."
    sudo cp "$SCRIPT_DIR/maxine-ar-sdk-webcam.service" /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable maxine-ar-sdk-webcam.service
    ok "Native pipeline service installed (start with: sudo systemctl start maxine-ar-sdk-webcam)"
else
    warn "ARSDK not found at /usr/local/ARSDK"
    warn "  Skipping native C++ pipeline build."
    warn "  Install ARSDK or use the Python NIM pipeline below."
fi

# ---------------------------------------------------------------------------
# 5. Python NIM pipeline (optional)
# ---------------------------------------------------------------------------
info "Setting up Python NIM pipeline..."

if ! command -v uv &> /dev/null; then
    info "Installing uv (astral.sh)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv: $(uv --version)"

# Pin Python 3.12
uv python install 3.12
if [ ! -f ".python-version" ]; then
    uv python pin 3.12
fi

# Sync deps
uv sync
ok "Python environment ready"

# Build protobufs if missing
if [ ! -f "eyecontact_pb2.py" ] || [ ! -f "eyecontact_pb2_grpc.py" ]; then
    bash build_proto.sh
    ok "Protobuf definitions built"
else
    ok "Protobuf definitions already exist"
fi

# Install systemd service for Python NIM pipeline
sudo cp "$SCRIPT_DIR/maxine-webcam.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable maxine-webcam.service
ok "Python NIM service installed (start with: sudo systemctl start maxine-webcam)"

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Virtual webcam: /dev/video10"
echo "  Persistence:    v4l2loopback will auto-load on boot"
echo ""
echo "  ── Native AR SDK Pipeline (low latency ~33ms) ──"
if [ -d "/usr/local/ARSDK/lib" ]; then
    echo "  Status:         READY"
    echo "  Image:          arsdk-gaze:latest"
    echo "  Service:        sudo systemctl start maxine-ar-sdk-webcam"
    echo "  Manual run:     docker compose up -d"
else
    echo "  Status:         NOT BUILT (ARSDK missing)"
fi
echo ""
echo "  ── Python NIM Pipeline (~1–2s latency) ──"
echo "  Status:         READY"
echo "  Service:        sudo systemctl start maxine-webcam"
echo "  Manual run:     uv run maxine_webcam_pipeline.py --resolution 480p"
echo ""
echo "  ── Logs ──"
echo "  Native:  sudo journalctl -u maxine-ar-sdk-webcam -f"
echo "  Python:  sudo journalctl -u maxine-webcam -f"
echo ""
echo "  Select 'MaxineEyeContact' in Zoom, Teams, OBS, Chrome, etc."
echo "============================================"
