#!/usr/bin/env bash
# Maxine Eye Contact Webcam – One-Command Setup
# Full chained pipeline: AR SDK (Docker) → /dev/video11 → LivePortrait (Python) → /dev/video10
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GHCR_IMAGE="ghcr.io/n00b001/maxine-eye-contact-webcam/arsdk-gaze:latest"
SERVICE_AR="maxine-ar-sdk-webcam.service"
SERVICE_HP="maxine-webcam.service"

info()  { echo ""; echo "[INFO] $*"; }
ok()    { echo "  ✓ $*"; }
warn()  { echo "  ⚠ $*"; }
fail()  { echo "  ✗ $*"; exit 1; }

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
# 2. v4l2loopback – two devices
#    video11 = intermediate (exclusive_caps=0: Python reads and writes)
#    video10 = final output  (exclusive_caps=1: clean Zoom/Chrome detection)
# ---------------------------------------------------------------------------
info "Configuring v4l2loopback (video10=final, video11=intermediate)..."
V4L2_CONF="/etc/modprobe.d/v4l2loopback.conf"
DESIRED='options v4l2loopback devices=2 video_nr=10,11 card_label="MaxineFinal,MaxineIntermediate" exclusive_caps=1,0 max_buffers=4'
if [ "$(cat "$V4L2_CONF" 2>/dev/null || true)" != "$DESIRED" ]; then
    echo "$DESIRED" | sudo tee "$V4L2_CONF" > /dev/null
    ok "Updated $V4L2_CONF"
    lsmod | grep -q "^v4l2loopback" && { sudo modprobe -r v4l2loopback 2>/dev/null || warn "Could not unload v4l2loopback — reboot may be needed"; }
else
    ok "$V4L2_CONF already correct"
fi
if ! lsmod | grep -q "^v4l2loopback"; then
    sudo modprobe v4l2loopback devices=2 video_nr=10,11 \
        card_label="MaxineFinal,MaxineIntermediate" exclusive_caps=1,0 max_buffers=4
    ok "v4l2loopback loaded"
else
    ok "v4l2loopback already loaded"
fi
MODULES_LOAD="/etc/modules-load.d/v4l2loopback.conf"
[ -f "$MODULES_LOAD" ] || { echo "v4l2loopback" | sudo tee "$MODULES_LOAD" > /dev/null; ok "Created $MODULES_LOAD"; }
for DEV in /dev/video10 /dev/video11; do
    [ -e "$DEV" ] && ok "Ready: $DEV" || warn "$DEV not found — check dmesg"
done

# ---------------------------------------------------------------------------
# 3. NVIDIA & Docker check
# ---------------------------------------------------------------------------
info "Checking NVIDIA GPU & Docker runtime..."
if command -v nvidia-smi &>/dev/null; then
    ok "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true)"
else
    warn "nvidia-smi not found — install NVIDIA drivers: https://developer.nvidia.com/cuda-downloads"
fi
command -v docker &>/dev/null || fail "Docker not found."
if docker info 2>/dev/null | grep -q nvidia; then
    ok "Docker NVIDIA runtime: OK"
else
    warn "Docker NVIDIA runtime not configured — see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# ---------------------------------------------------------------------------
# 4. Pull GHCR image
# ---------------------------------------------------------------------------
info "Pulling pre-built image from GHCR..."
echo "  Image: $GHCR_IMAGE"
docker info 2>/dev/null | grep -q "ghcr.io" || warn "You may need: echo \$GITHUB_TOKEN | docker login ghcr.io -u \$USER --password-stdin"
docker pull "$GHCR_IMAGE" || fail "Failed to pull $GHCR_IMAGE"
ok "Image pulled: $GHCR_IMAGE"

# ---------------------------------------------------------------------------
# 5. Python project install (LivePortrait head-pose pipeline)
# ---------------------------------------------------------------------------
info "Installing Python project dependencies..."
cd "$SCRIPT_DIR"
uv sync --extra liveportrait
# mediapipe pins protobuf<5; installed separately per PROJECT_RULES.md
uv pip install 'mediapipe>=0.10,<0.10.20'
ok "Python dependencies installed"

LIVEPORTRAIT_DIR="$SCRIPT_DIR/vendor/LivePortrait"
if [ ! -d "$LIVEPORTRAIT_DIR" ]; then
    info "Cloning LivePortrait..."
    git clone --depth 1 https://github.com/KwaiVGI/LivePortrait.git "$LIVEPORTRAIT_DIR"
    ok "LivePortrait cloned"
else
    ok "LivePortrait already present"
fi

WEIGHTS_SENTINEL="$LIVEPORTRAIT_DIR/pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth"
if [ ! -f "$WEIGHTS_SENTINEL" ]; then
    info "Downloading LivePortrait pretrained weights (this may take several minutes)..."
    uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('KlingTeam/LivePortrait',
    local_dir='$LIVEPORTRAIT_DIR/pretrained_weights',
    allow_patterns=['liveportrait/**', 'insightface/**'])"
    ok "Pretrained weights downloaded"
else
    ok "Pretrained weights already present"
fi

# ---------------------------------------------------------------------------
# 6. Install project to /opt for systemd units
# ---------------------------------------------------------------------------
info "Syncing project to /opt/maxine-eye-contact-webcam/..."
sudo rsync -a --delete \
    --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
    "$SCRIPT_DIR/" /opt/maxine-eye-contact-webcam/
sudo chown -R "$USER:$USER" /opt/maxine-eye-contact-webcam
ok "Synced to /opt/maxine-eye-contact-webcam"
cd /opt/maxine-eye-contact-webcam
uv sync --extra liveportrait
uv pip install 'mediapipe>=0.10,<0.10.20'
cd "$SCRIPT_DIR"
ok "Python environment built in /opt"

# ---------------------------------------------------------------------------
# 7. Install & start systemd services
# ---------------------------------------------------------------------------
info "Installing systemd services..."
# Stage 1 unit file already contains the full GHCR URL — copy verbatim.
sudo cp "$SCRIPT_DIR/$SERVICE_AR" /etc/systemd/system/
# Stage 2 (head-pose) runs as the invoking user so the venv, $HOME/.local/bin/uv,
# and the NVIDIA device nodes (owned by the `video` / `render` groups) are
# accessible. Substitute __MAXINE_USER__ placeholder at install time so the
# repo file doesn't hardcode a username.
TMP_HP="/tmp/$SERVICE_HP"
cp "$SCRIPT_DIR/$SERVICE_HP" "$TMP_HP"
MAXINE_USER="${SUDO_USER:-$USER}"
sed -i "s|__MAXINE_USER__|$MAXINE_USER|g" "$TMP_HP"
sudo cp "$TMP_HP" /etc/systemd/system/
rm -f "$TMP_HP"
sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_AR"
ok "Enabled+started: $SERVICE_AR"
sudo systemctl enable --now "$SERVICE_HP"
ok "Enabled+started: $SERVICE_HP (User=$MAXINE_USER)"

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Pipeline:  Camera → AR SDK → /dev/video11 → LivePortrait → /dev/video10"
echo ""
echo "  Select 'MaxineFinal' (/dev/video10) in Zoom, Teams, OBS, Chrome, etc."
echo ""
echo "  Logs:"
echo "    AR SDK:    journalctl -fu $SERVICE_AR"
echo "    Head-pose: journalctl -fu $SERVICE_HP"
echo ""
echo "  NOTE: First start ~60 s (LivePortrait torch.compile warmup)."
echo "============================================"
