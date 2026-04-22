# syntax=docker/dockerfile:1
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

# Copy ARSDK (hard-linked from host, so it's a full copy in the build context)
COPY arsdk /usr/local/ARSDK

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

# Runtime library path: ONLY main lib + CUDA/TensorRT externals.
# Do NOT add .../features/*/lib subdirectories.
ENV LD_LIBRARY_PATH="/usr/local/ARSDK/lib:/usr/local/ARSDK/external/cuda/lib:/usr/local/ARSDK/external/tensorrt/lib:/usr/local/ARSDK/features/nvargazeredirection/lib:/usr/local/ARSDK/features/nvarlandmarkdetection/lib:/usr/local/ARSDK/features/nvarfaceboxdetection/lib"
ENV NVAR_MODEL_DIR=/usr/local/ARSDK/lib/models

# Default args: output to /dev/video10, 640x480
CMD ["/usr/local/bin/maxine_ar_webcam", "/dev/video10", "640", "480"]
