# syntax=docker/dockerfile:1
# Multi-stage build:
#   Stage 1 (arsdk-base) – proprietary NVIDIA ARSDK, built locally once and
#   pushed to GHCR.  The public repo never contains the ARSDK binaries.
#   Stage 2 – application code, built on GitHub-hosted runners.
#
# Build the base (one-time, on a machine with /usr/local/ARSDK):
#   docker build -f Dockerfile.base -t ghcr.io/n00b001/maxine-eye-contact-webcam/arsdk-base:latest .
#   docker push ghcr.io/n00b001/maxine-eye-contact-webcam/arsdk-base:latest
#
# Build the app (GitHub Actions or locally):
#   docker build -t arsdk-gaze:latest .
#
# NOTE: TRT engine build is NOT part of image build — it happens on first run
# via docker/entrypoint.sh when engines are missing from /opt/flp-checkpoints/.
# Before starting the service, download ONNX models:
#   huggingface-cli download warmshao/FasterLivePortrait \
#     --local-dir /opt/flp-checkpoints

ARG ARSDK_BASE=ghcr.io/n00b001/maxine-eye-contact-webcam/arsdk-base:latest

# ---------------------------------------------------------------------------
# Stage 1: Import ARSDK from the pre-published base image
# ---------------------------------------------------------------------------
FROM ${ARSDK_BASE} AS arsdk

# ---------------------------------------------------------------------------
# Stage 1b: Build the grid_sample_3d TRT 10 plugin in a -devel image that
# actually has nvcc + CUDA headers. We copy just the resulting .so into the
# runtime stage so the final image stays slim.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS plugin-builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Pull TRT 10 headers + runtime libs from the ARSDK base image so the plugin
# links against the same ABI FLP's ONNX parser uses at runtime.
COPY --from=arsdk /usr/local/ARSDK/external/tensorrt/include /opt/trt/include
COPY --from=arsdk /usr/local/ARSDK/external/tensorrt/lib /opt/trt/lib
ARG GRID_SAMPLE_3D_REPO=https://github.com/SeanWangJS/grid-sample3d-trt-plugin.git
ARG GRID_SAMPLE_3D_COMMIT=main
RUN git clone "$GRID_SAMPLE_3D_REPO" /tmp/grid-sample3d \
    && cd /tmp/grid-sample3d \
    && git checkout "$GRID_SAMPLE_3D_COMMIT" \
    && mkdir build && cd build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="70;80;86;89;90" \
        -DTensorRT_INCLUDE_DIR=/opt/trt/include \
        -DTensorRT_LIB_DIR=/opt/trt/lib \
        -DCMAKE_CXX_FLAGS="-I/opt/trt/include" \
        -DCMAKE_CUDA_FLAGS="-I/opt/trt/include" \
        -DCMAKE_EXE_LINKER_FLAGS="-L/opt/trt/lib -lnvinfer" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/trt/lib -lnvinfer" \
    && cmake --build . --target grid_sample_3d_plugin -- -j$(nproc) \
    && cp libgrid_sample_3d_plugin.so /tmp/grid-sample3d/libgrid_sample_3d_plugin.so

# ---------------------------------------------------------------------------
# Stage 2: Application build (GitHub-hosted runners can do this)
# ---------------------------------------------------------------------------
# Base image: nvidia/cuda:12.8.1-runtime-ubuntu24.04 ships CUDA 12.8.
# Python 3.12 is the default in Ubuntu 24.04; we install python3.12-dev and
# pin torch to the CUDA 12.8-compatible wheel (torch>=2.4 supports cu121/cu124;
# cu124 wheels run fine on CUDA 12.8 due to forward compatibility).
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# (a) System packages: C++ build tools + Python 3.12 dev headers + pybind11
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang \
    cmake \
    pkg-config \
    libopencv-dev \
    libglm-dev \
    v4l-utils \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    pybind11-dev \
    build-essential \
    ffmpeg \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# (b) Python dependencies for FasterLivePortrait + our glue modules
#
# torch/torchvision: CUDA 12.4 wheel — forward-compatible with CUDA 12.8.
# opencv-python-headless: avoids X11/GTK deps (headless server environment).
# omegaconf: required by flp_gpu_adapter.py and FLP configs.
# pycuda: intentionally OMITTED — FLP lists it in requirements.txt but never
#   imports it anywhere in vendor/FasterLivePortrait/. Installing it would
#   require the full CUDA toolkit (nvcc, headers) to compile from source.
# Packages from vendor/FasterLivePortrait/requirements.txt that are relevant
# to inference (audio/speech packages excluded — not needed for webcam use).
# ---------------------------------------------------------------------------
RUN pip3 install --no-cache-dir --break-system-packages \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install --no-cache-dir --break-system-packages \
    ffmpeg-python \
    omegaconf \
    onnx \
    onnxruntime-gpu \
    numpy \
    opencv-python-headless \
    scikit-image \
    scipy \
    insightface \
    "huggingface_hub[cli]" \
    mediapipe \
    torchgeometry \
    munch \
    pillow \
    tqdm \
    transformers \
    # Pin tensorrt to 10.9.x so the engines we serialize match the ABI of
    # the ARSDK-bundled libnvinfer.so.10.9.0 that the running process loads
    # (via LD_LIBRARY_PATH). Mismatched versions → Serialization assertion
    # "stdVersionRead == kSERIALIZATION_VERSION failed" on engine load.
    "tensorrt==10.9.*"

# ---------------------------------------------------------------------------
# Copy ARSDK from the base image (proprietary layer)
# ---------------------------------------------------------------------------
COPY --from=arsdk /usr/local/ARSDK /usr/local/ARSDK

# ---------------------------------------------------------------------------
# Copy application sources (C++ binary)
# ---------------------------------------------------------------------------
COPY docker/src /app/src
COPY docker/utils /app/utils
COPY docker/glm /app/glm

# ---------------------------------------------------------------------------
# (f) Build the headless C++ webcam bridge (kept for debugging / reference)
# We intentionally do NOT link against feature libs (nvARGazeRedirection, etc.)
# because libnvARPose.so contains a built-in fallback that avoids the
# EnableLookAway parameter mismatch in external libnvARGazeRedirection.so.
# ---------------------------------------------------------------------------
RUN clang++ -fpermissive -O2 -std=c++17 \
    -I/usr/local/ARSDK/include \
    -I/usr/local/ARSDK/features/nvargazeredirection/include \
    -I/app/utils \
    -I/app/glm \
    $(pkg-config --cflags opencv4) \
    /app/src/maxine_ar_webcam.cpp \
    /app/src/gazeEngine.cpp \
    /app/utils/renderingUtils.cpp \
    /app/utils/featureVertexName.cpp \
    -o /usr/local/bin/maxine_ar_webcam \
    -L/usr/local/ARSDK/lib -lnvARPose -lNVCVImage \
    $(pkg-config --libs opencv4) \
    -pthread

# ---------------------------------------------------------------------------
# (c) Build and install the pybind11 shim (maxine_ar_ext)
# ARSDK_ROOT must be set so setup.py finds the AR SDK headers + libs.
# ---------------------------------------------------------------------------
COPY docker/src/maxine_ar_ext /app/src/maxine_ar_ext
RUN ARSDK_ROOT=/usr/local/ARSDK \
    pip3 install --no-cache-dir --break-system-packages /app/src/maxine_ar_ext

# ---------------------------------------------------------------------------
# (d) Copy Python orchestration code and FasterLivePortrait vendor tree
#     Weights/checkpoints are NOT embedded — they are mounted at runtime via
#     /opt/flp-checkpoints (see docker-compose.yml).
#
#     Build context must be the repo root (not docker/). CI workflow uses:
#       context: .
#       file: docker/Dockerfile
#     For local builds: docker build -f docker/Dockerfile -t arsdk-gaze:latest .
# ---------------------------------------------------------------------------
COPY flp_gpu_adapter.py /app/flp_gpu_adapter.py
COPY gpu_format_convert.py /app/gpu_format_convert.py
COPY maxine_fused_pipeline.py /app/maxine_fused_pipeline.py
# Copy FLP source tree (no checkpoints — those are runtime-mounted).
# FasterLivePortrait is NOT checked into the public repo (vendor/ is
# gitignored). Clone it fresh at a pinned commit so CI (which sees no
# vendor/ tree) and local builds produce identical images.
ARG FLP_REPO=https://github.com/warmshao/FasterLivePortrait.git
ARG FLP_COMMIT=8aad3602177547aaa5e4beec0c3ef5b7944e7a1f
RUN git clone "$FLP_REPO" /app/vendor/FasterLivePortrait \
    && git -C /app/vendor/FasterLivePortrait checkout "$FLP_COMMIT" \
    && rm -rf /app/vendor/FasterLivePortrait/.git

# Copy in the grid_sample_3d TRT 10 plugin built in stage 1b.
COPY --from=plugin-builder /tmp/grid-sample3d/libgrid_sample_3d_plugin.so /usr/local/lib/libgrid_sample_3d_plugin.so
RUN ldconfig

# Install our TRT 10-compatible ONNX→engine builder alongside FLP's.
# FLP's scripts/onnx2trt.py targets TRT 8.x (max_workspace_size, max_batch_size,
# build_engine — all removed in TRT 10). Our replacement uses the TRT 10 API
# and loads the freshly-built grid_sample_3d plugin before parsing the ONNX.
COPY docker/scripts/onnx2trt_v10.py /app/vendor/FasterLivePortrait/scripts/onnx2trt_v10.py

# Patch FLP's motion_extractor_model.py: the TRT branch hard-codes an output
# ordering (kp first) that our TRT 10 engine doesn't use — our engines
# preserve the ONNX graph order (pitch, yaw, roll, t, exp, scale, kp).
# Replace the TRT-specific unpacking with the same order as the PyTorch
# branch so it works with both back-ends.
RUN sed -i 's|kp, pitch, yaw, roll, t, exp, scale = data|pitch, yaw, roll, t, exp, scale, kp = data  # TRT 10: same order as ONNX / PyTorch|' \
    /app/vendor/FasterLivePortrait/src/models/motion_extractor_model.py

WORKDIR /app

# ---------------------------------------------------------------------------
# (h) Entrypoint script — detects missing TRT engines and builds them on
#     first run before exec-ing the CMD. See docker/entrypoint.sh.
# ---------------------------------------------------------------------------
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Runtime library path: main lib + CUDA/TensorRT externals + feature libs.
ENV LD_LIBRARY_PATH="/usr/local/ARSDK/lib:/usr/local/ARSDK/external/cuda/lib:/usr/local/ARSDK/external/tensorrt/lib:/usr/local/ARSDK/features/nvargazeredirection/lib:/usr/local/ARSDK/features/nvarlandmarkdetection/lib:/usr/local/ARSDK/features/nvarfaceboxdetection/lib"
ENV NVAR_MODEL_DIR=/usr/local/ARSDK/lib/models
# FLP checkpoint dir — overrideable via docker run -e
ENV FLP_CHECKPOINTS_DIR=/opt/flp-checkpoints

ENTRYPOINT ["/entrypoint.sh"]

# (g) Default CMD: Python orchestrator.
# The C++ binary (/usr/local/bin/maxine_ar_webcam) is still available for
# debugging: docker run --rm ... /usr/local/bin/maxine_ar_webcam /dev/video10 640 480
CMD ["python3", "/app/maxine_fused_pipeline.py", \
     "--input-device", "/dev/video0", \
     "--output-device", "/dev/video10", \
     "--width", "1920", "--height", "1080", "--fps", "30", \
     "--src-image", "/app/src-portrait.jpg", \
     "--model-dir", "/usr/local/ARSDK/lib/models", \
     "--cfg", "/app/vendor/FasterLivePortrait/configs/trt_infer.yaml"]
