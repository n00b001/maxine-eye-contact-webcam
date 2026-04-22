#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  Building Maxine Eye Contact Protobufs"
echo "============================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/4] Cloning NVIDIA nim-clients repository..."
if [ -d "/tmp/nim-clients" ]; then
    echo "      Repository already cloned, pulling latest..."
    cd /tmp/nim-clients && git pull
    cd -
else
    git clone https://github.com/NVIDIA-Maxine/nim-clients.git /tmp/nim-clients
fi

echo "[2/4] Ensuring dependencies (including dev/build tools)..."
# Full sync: both main deps AND dev deps (grpcio-tools). --only-dev would
# strip grpcio/numpy/opencv, breaking grpc_tools.protoc at runtime.
uv sync

echo "[3/4] Compiling protobuf definitions..."
chmod +x /tmp/nim-clients/eye-contact/protos/linux/compile_protos.sh

# Run compile inside the uv-managed env so grpc_tools.protoc is available.
# IMPORTANT: uv run must execute from the project dir (where pyproject.toml
# lives) so it finds the .venv. compile_protos.sh locates its own proto/ and
# output dirs based on $0, so invoking via absolute path works fine.
cd "$SCRIPT_DIR"
uv run bash /tmp/nim-clients/eye-contact/protos/linux/compile_protos.sh

echo "[4/4] Copying generated files to project directory..."
cp /tmp/nim-clients/eye-contact/interfaces/eyecontact_pb2.py "$SCRIPT_DIR/"
cp /tmp/nim-clients/eye-contact/interfaces/eyecontact_pb2_grpc.py "$SCRIPT_DIR/"

cd "$SCRIPT_DIR"
echo ""
echo "============================================"
echo "  Done! Protobuf files ready:"
echo "    - eyecontact_pb2.py"
echo "    - eyecontact_pb2_grpc.py"
echo "============================================"
