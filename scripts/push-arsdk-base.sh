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

# ---------------------------------------------------------------------------
# Verify ARSDK is present
# ---------------------------------------------------------------------------
if [ ! -d "arsdk/lib" ]; then
    if [ -d "/usr/local/ARSDK/lib" ]; then
        info "Copying ARSDK from /usr/local/ARSDK into build context..."
        cp -rL /usr/local/ARSDK arsdk
    else
        echo "ERROR: ARSDK not found at /usr/local/ARSDK or docker/arsdk/"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Log in to GHCR (uses GitHub Personal Access Token with write:packages)
# ---------------------------------------------------------------------------
if ! docker info 2>/dev/null | grep -q "Registry: https://index.docker.io/v1/"; then
    info "Ensure you are logged in to GHCR:"
    info "  echo \$GITHUB_TOKEN | docker login ghcr.io -u n00b001 --password-stdin"
fi

# ---------------------------------------------------------------------------
# Build & push
# ---------------------------------------------------------------------------
info "Building $IMAGE ..."
docker build -f Dockerfile.base -t "$IMAGE" .

info "Pushing $IMAGE ..."
docker push "$IMAGE"

info "Done. GitHub Actions can now build on ubuntu-latest using this base."
