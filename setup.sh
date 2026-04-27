#!/usr/bin/env bash
# Maxine Eye Contact Webcam – One-Command Setup
# Unified GPU pipeline: webcam (v4l2) → LivePortrait head-pose + AR SDK gaze
#   on a single CUDA stream → /dev/video10 (v4l2loopback)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GHCR_IMAGE="ghcr.io/n00b001/maxine-eye-contact-webcam/arsdk-gaze:latest"
SERVICE="maxine-webcam.service"

info()  { echo ""; echo "[INFO] $*"; }
ok()    { echo "  ✓ $*"; }
warn()  { echo "  ⚠ $*"; }
fail()  { echo "  ✗ $*"; exit 1; }

# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------
SKIP_DOCKER_PULL=0
SKIP_WEIGHTS_DOWNLOAD=0
FORCE=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

One-command installer for Maxine Eye Contact Webcam.

Options:
  --skip-docker-pull        Skip pulling the GHCR container image
  --skip-weights-download   Skip downloading FasterLivePortrait ONNX weights
  --force                   Re-do all steps even if already complete
  --help                    Show this help and exit

The script installs apt packages, configures v4l2loopback, pulls the
container image, downloads FLP ONNX weights, and installs the systemd
service. TRT engine compilation happens on first container start.

Requirements:
  - NVIDIA driver R535+ (ubuntu-drivers autoinstall)
  - Docker with NVIDIA runtime (nvidia-container-toolkit)
  - ~40 GB free on /opt (image ~10 GB, weights ~1.5 GB, TRT engines ~8 GB)
EOF
    exit 0
}

for arg in "$@"; do
    case "$arg" in
        --skip-docker-pull)       SKIP_DOCKER_PULL=1 ;;
        --skip-weights-download)  SKIP_WEIGHTS_DOWNLOAD=1 ;;
        --force)                  FORCE=1 ;;
        --help|-h)                usage ;;
        *) echo "Unknown option: $arg" >&2; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# 0. Pre-flight checks
# ---------------------------------------------------------------------------
info "Pre-flight checks..."

# Check NVIDIA driver
if nvidia-smi --query-gpu=name,driver_version --format=csv,noheader &>/dev/null; then
    ok "NVIDIA GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
else
    fail "NVIDIA driver not available. Install NVIDIA proprietary driver (R535+) via:
        sudo ubuntu-drivers autoinstall
  then reboot and re-run setup.sh."
fi

# Check CUDA compute capability >= 7.0 (Volta+)
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
# Compare as a float: strip dot, compare as integer (e.g. "7.5" → 75 >= 70)
COMPUTE_INT="$(echo "$COMPUTE_CAP" | tr -d '.' | sed 's/^0*//')"
if [ -z "$COMPUTE_INT" ] || [ "$COMPUTE_INT" -lt 70 ]; then
    fail "GPU compute capability $COMPUTE_CAP is below the required 7.0 (Volta+).
  AR SDK and TensorRT require a GPU with compute capability >= 7.0.
  Supported GPUs include RTX 20xx, 30xx, 40xx, A-series, and Volta cards."
fi
ok "CUDA compute capability: $COMPUTE_CAP (>= 7.0 required)"

# Check Docker + NVIDIA runtime
if ! command -v docker &>/dev/null; then
    fail "Docker not found. Install Docker: https://docs.docker.com/engine/install/"
fi
if docker info 2>&1 | grep -q "Runtimes:.*nvidia"; then
    ok "Docker NVIDIA runtime: present"
else
    fail "Docker NVIDIA runtime not available. Install nvidia-container-toolkit:
        https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
  Quick install:
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
fi

# Check disk space on /opt (need >= 40 GB = 41943040 KB)
AVAIL_KB="$(df --output=avail /opt | tail -1 | tr -d ' ')"
REQUIRED_KB=41943040
if [ "$AVAIL_KB" -lt "$REQUIRED_KB" ]; then
    AVAIL_GB=$(( AVAIL_KB / 1048576 ))
    fail "/opt has only ~${AVAIL_GB} GB free; at least 40 GB required.
  Free up space or mount a larger volume at /opt."
fi
ok "Disk space on /opt: $(( AVAIL_KB / 1048576 )) GB free (>= 40 GB required)"

# Check webcam (warn only — service can start without one)
if v4l2-ctl --list-devices 2>&1 | grep -q "/dev/video"; then
    ok "Webcam device found"
else
    warn "No webcam detected (/dev/video* absent). The service will start but produce no output until a camera is plugged in."
fi

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
info "Installing system dependencies..."
sudo dpkg --configure -a 2>/dev/null || true
sudo apt-get install -f -y -qq 2>/dev/null || true
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg v4l-utils curl "linux-headers-$(uname -r)" \
    || warn "Some base packages failed to install"
if lsmod | grep -q "^v4l2loopback" || modinfo v4l2loopback &>/dev/null; then
    ok "v4l2loopback already present — skipping dkms install"
else
    sudo apt-get install -y -qq v4l2loopback-dkms v4l2loopback-utils \
        || warn "v4l2loopback-dkms install failed — module may need manual install"
fi
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi
ok "System dependencies installed"

# ---------------------------------------------------------------------------
# 2. v4l2loopback – single output device
#    video10 = final output (exclusive_caps=1: clean Zoom/Chrome detection)
#    The pre-4c38c7b intermediate hop (video11) is no longer needed in the
#    unified GPU pipeline; only video10 is required.
# ---------------------------------------------------------------------------
info "Configuring v4l2loopback (video10=output)..."
V4L2_CONF="/etc/modprobe.d/v4l2loopback.conf"
DESIRED='options v4l2loopback devices=1 video_nr=10 card_label="MaxineEyeContact" exclusive_caps=1 max_buffers=2'
if [ "$(cat "$V4L2_CONF" 2>/dev/null || true)" != "$DESIRED" ]; then
    echo "$DESIRED" | sudo tee "$V4L2_CONF" > /dev/null
    ok "Updated $V4L2_CONF"
    lsmod | grep -q "^v4l2loopback" && { sudo modprobe -r v4l2loopback 2>/dev/null || warn "Could not unload v4l2loopback — reboot may be needed"; }
else
    ok "$V4L2_CONF already correct"
fi
if ! lsmod | grep -q "^v4l2loopback"; then
    sudo modprobe v4l2loopback devices=1 video_nr=10 \
        card_label="MaxineEyeContact" exclusive_caps=1 max_buffers=4
    ok "v4l2loopback loaded"
else
    ok "v4l2loopback already loaded"
fi
MODULES_LOAD="/etc/modules-load.d/v4l2loopback.conf"
[ -f "$MODULES_LOAD" ] || { echo "v4l2loopback" | sudo tee "$MODULES_LOAD" > /dev/null; ok "Created $MODULES_LOAD"; }
if [ -e /dev/video10 ]; then ok "Ready: /dev/video10"; else warn "/dev/video10 not found — check dmesg"; fi

# ---------------------------------------------------------------------------
# 3. Pull GHCR image
# ---------------------------------------------------------------------------
if [ "$SKIP_DOCKER_PULL" -eq 1 ]; then
    ok "Skipping Docker image pull (--skip-docker-pull)"
elif [ "$FORCE" -eq 0 ] && docker image inspect "$GHCR_IMAGE" &>/dev/null; then
    ok "Image already present: $GHCR_IMAGE (use --force to re-pull)"
else
    info "Pulling pre-built image from GHCR..."
    echo "  Image: $GHCR_IMAGE"
    docker info 2>/dev/null | grep -q "ghcr.io" || warn "You may need: echo \$GITHUB_TOKEN | docker login ghcr.io -u \$USER --password-stdin"
    docker pull "$GHCR_IMAGE" || fail "Failed to pull $GHCR_IMAGE"
    ok "Image pulled: $GHCR_IMAGE"
fi

# ---------------------------------------------------------------------------
# 4. Python project install — host side only gets dev tooling (lint + test
# parity with CI). All runtime deps (torch, FLP, tensorrt, maxine_ar_ext)
# live in the Docker image.
# ---------------------------------------------------------------------------
info "Installing Python project dependencies..."
cd "$SCRIPT_DIR"
uv sync --all-groups
ok "Python dev dependencies installed"

# ---------------------------------------------------------------------------
# FasterLivePortrait ONNX weights — downloaded to /opt/flp-checkpoints on
# the host. The container mounts this directory at runtime. TRT engines are
# built on first container start by docker/entrypoint.sh.
# ---------------------------------------------------------------------------
FLP_CHECKPOINTS="/opt/flp-checkpoints"
FLP_SENTINEL="$FLP_CHECKPOINTS/liveportrait_onnx/appearance_feature_extractor.onnx"

_flp_present() {
    # True if sentinel exists AND total size >= 500 MB
    if [ ! -f "$FLP_SENTINEL" ]; then return 1; fi
    local size_kb
    size_kb="$(du -s "$FLP_CHECKPOINTS" 2>/dev/null | cut -f1)"
    [ "${size_kb:-0}" -ge 512000 ]
}

if [ "$SKIP_WEIGHTS_DOWNLOAD" -eq 1 ]; then
    ok "Skipping ONNX weights download (--skip-weights-download)"
elif [ "$FORCE" -eq 0 ] && _flp_present; then
    ok "FasterLivePortrait ONNX weights already present (>= 500 MB)"
else
    info "Downloading FasterLivePortrait ONNX weights to $FLP_CHECKPOINTS (this may take several minutes)..."
    sudo mkdir -p "$FLP_CHECKPOINTS"
    sudo chown "$USER:$USER" "$FLP_CHECKPOINTS"
    uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('warmshao/FasterLivePortrait',
    local_dir='$FLP_CHECKPOINTS')"
    ok "FasterLivePortrait ONNX weights downloaded to $FLP_CHECKPOINTS"
fi

# ---------------------------------------------------------------------------
# 5. Install project to /opt for systemd units
# ---------------------------------------------------------------------------
info "Syncing project to /opt/maxine-eye-contact-webcam/..."
sudo rsync -a --delete \
    --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
    "$SCRIPT_DIR/" /opt/maxine-eye-contact-webcam/
sudo chown -R "$USER:$USER" /opt/maxine-eye-contact-webcam
ok "Synced to /opt/maxine-eye-contact-webcam"
cd /opt/maxine-eye-contact-webcam
uv sync --all-groups
cd "$SCRIPT_DIR"
ok "Python environment built in /opt"

# ---------------------------------------------------------------------------
# 6. Install & start systemd services
# ---------------------------------------------------------------------------
info "Installing single systemd service..."
# Legacy cleanup: remove the pre-merge Stage-1-only service if it still
# exists from an earlier setup.sh run.
sudo systemctl disable --now maxine-ar-sdk-webcam.service 2>/dev/null || true
sudo rm -f /etc/systemd/system/maxine-ar-sdk-webcam.service

# The service runs as the invoking user so the venv, $HOME/.local/bin/uv,
# and the NVIDIA device nodes (owned by `video` / `render` groups) are
# accessible. Substitute the __MAXINE_USER__ placeholder at install time.
TMP="/tmp/$SERVICE"
cp "$SCRIPT_DIR/$SERVICE" "$TMP"
MAXINE_USER="${SUDO_USER:-$USER}"
sed -i "s|__MAXINE_USER__|$MAXINE_USER|g" "$TMP"

# Detect whether installed unit is already up to date
INSTALLED="/etc/systemd/system/$SERVICE"
if [ "$FORCE" -eq 0 ] && [ -f "$INSTALLED" ] && diff -q "$TMP" "$INSTALLED" &>/dev/null; then
    rm -f "$TMP"
    ok "Service already installed and up to date: $SERVICE"
else
    sudo cp "$TMP" "$INSTALLED"
    rm -f "$TMP"
    sudo systemctl daemon-reload
    sudo systemctl enable --now "$SERVICE"
    ok "Enabled+started: $SERVICE (User=$MAXINE_USER)"
fi

# ---------------------------------------------------------------------------
# 7. Portrait placeholder reminder
# ---------------------------------------------------------------------------
if [ ! -f /opt/maxine-portrait.jpg ]; then
    warn "Portrait not found at /opt/maxine-portrait.jpg"
    echo ""
    echo "  ACTION REQUIRED: Copy a neutral frontal portrait of yourself to"
    echo "  /opt/maxine-portrait.jpg before starting the service."
    echo "  Recommended: 1080p or similar, face centred, neutral expression."
    echo "  Example:"
    echo "    cp ~/my-portrait.jpg /opt/maxine-portrait.jpg"
    echo ""
else
    ok "Portrait found: /opt/maxine-portrait.jpg"
fi

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Pipeline:  webcam (v4l2) → LivePortrait head-pose + AR SDK gaze"
echo "             on a single CUDA stream → /dev/video10 (v4l2loopback)"
echo ""
echo "  Select 'MaxineEyeContact' (/dev/video10) in Zoom, Teams, OBS, Chrome, etc."
echo ""
echo "  Logs:"
echo "    journalctl -fu $SERVICE"
echo ""
echo "  NOTE: First start builds TRT engines (~5-15 min, once per GPU)."
echo "        Subsequent starts are fast."
echo "============================================"
