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

ARG ARSDK_BASE=ghcr.io/n00b001/maxine-eye-contact-webcam/arsdk-base:latest

# ---------------------------------------------------------------------------
# Stage 1: Import ARSDK from the pre-published base image
# ---------------------------------------------------------------------------
FROM ${ARSDK_BASE} AS arsdk

# ---------------------------------------------------------------------------
# Stage 2: Application build (GitHub-hosted runners can do this)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build tools, OpenCV, and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang \
    cmake \
    pkg-config \
    libopencv-dev \
    libglm-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy ARSDK from the base image (proprietary layer)
COPY --from=arsdk /usr/local/ARSDK /usr/local/ARSDK

# Copy application sources
COPY src /app/src
COPY utils /app/utils
COPY glm /app/glm

# Build the headless webcam bridge
# We intentionally do NOT link against feature libs (nvARGazeRedirection, etc.)
# because libnvARPose.so contains a built-in fallback that avoids the
# EnableLookAway parameter mismatch in external libnvARGazeRedirection.so.
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

# Runtime library path: main lib + CUDA/TensorRT externals + feature libs.
ENV LD_LIBRARY_PATH="/usr/local/ARSDK/lib:/usr/local/ARSDK/external/cuda/lib:/usr/local/ARSDK/external/tensorrt/lib:/usr/local/ARSDK/features/nvargazeredirection/lib:/usr/local/ARSDK/features/nvarlandmarkdetection/lib:/usr/local/ARSDK/features/nvarfaceboxdetection/lib"
ENV NVAR_MODEL_DIR=/usr/local/ARSDK/lib/models

# Default args: output to /dev/video10, 640x480
CMD ["/usr/local/bin/maxine_ar_webcam", "/dev/video10", "640", "480"]
