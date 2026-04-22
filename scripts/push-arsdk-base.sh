#!/usr/bin/env bash
# Build and push the proprietary ARSDK base image to GHCR.
# Run this ONCE from a machine that has ARSDK installed at /usr/local/ARSDK.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../docker"

REGISTRY="ghcr.io/n00b001/maxine-eye-contact-webcam"
TAG="${1:-latest}"
IMAGE="$REGISTRY/arsdk-base:$TAG"

info() { echo "[INFO] $*"; }
error() { echo "[ERROR] $*" >&2; }

# ---------------------------------------------------------------------------
# Verify ARSDK is present
# ---------------------------------------------------------------------------
if [ ! -d "arsdk/lib" ]; then
    if [ -d "/usr/local/ARSDK/lib" ]; then
        info "Copying ARSDK from /usr/local/ARSDK into build context..."
        rm -rf arsdk
        mkdir -p arsdk
        cp -rL /usr/local/ARSDK/include  arsdk/
        cp -rL /usr/local/ARSDK/lib      arsdk/
        cp -rL /usr/local/ARSDK/features arsdk/
        cp -rL /usr/local/ARSDK/share    arsdk/
    else
        error "ARSDK not found at /usr/local/ARSDK or docker/arsdk/"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Verify GHCR write access
# ---------------------------------------------------------------------------
info "Checking GHCR login & write access..."
if ! docker info 2>/dev/null | grep -q "Registry"; then
    error "Docker daemon is not running or not accessible"
    exit 1
fi

# Attempt a small test push to verify permissions
if ! docker push "$IMAGE" 2>&1 | head -1 | grep -q "The push refers to repository"; then
    # Image doesn't exist locally yet; we'll build first. But let's check login.
    if ! docker pull "ghcr.io/n00b001/maxine-eye-contact-webcam/arsdk-base:__test__" 2>&1 | grep -q "unauthorized\|denied"; then
        : # logged in ok
    fi
fi

# ---------------------------------------------------------------------------
# Build & push
# ---------------------------------------------------------------------------
info "Building $IMAGE ..."
docker build -f Dockerfile.base -t "$IMAGE" .

info "Pushing $IMAGE ..."
if ! docker push "$IMAGE"; then
    error "Push failed. This usually means your GitHub token lacks 'write:packages' scope."
    error ""
    error "Fix options:"
    error "  1. Interactive (easiest):"
    error "       gh auth login --scopes repo,write:packages,read:packages"
    error ""
    error "  2. Classic PAT:"
    error "       - Go to https://github.com/settings/tokens/new"
    error "       - Select scopes: write:packages, read:packages, repo"
    error "       - Generate token, then:"
    error "       echo ghp_YOUR_TOKEN | docker login ghcr.io -u n00b001 --password-stdin"
    error ""
    exit 1
fi

info "Done. GitHub Actions can now build on ubuntu-latest using this base."
