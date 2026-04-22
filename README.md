# Maxine Eye Contact – Real-time Webcam Pipeline

Two pipeline options for real-time gaze correction from a physical webcam to a
virtual V4L2 webcam (`/dev/video10`).

| Pipeline | Latency | Requirements | Best For |
|----------|---------|--------------|----------|
| **Native AR SDK (Docker)** | **~33 ms** | NVIDIA GPU, AR SDK | Production, low-latency meetings |
| **Python NIM (gRPC)** | **~1–2 s** | NVIDIA GPU, NIM container | Quick start, highest quality |

---

## Native AR SDK Pipeline (Recommended)

A native C++ binary built with the NVIDIA AR SDK (`nvargazeredirection`) runs
inside a Docker container. It captures frame-by-frame, runs gaze-redirection
inference on the GPU, and writes directly to a v4l2loopback device.

**Measured latency (RTX 4090):**

| Resolution | Format | Pipeline FPS | Avg Latency | Max Latency |
|------------|--------|--------------|-------------|-------------|
| 320×240 | YUYV | 30 | **33 ms** | 37 ms |
| 640×480 | YUYV | 30 | **33 ms** | 37 ms |
| 1280×720 | MJPEG | 24 | **42 ms** | 46 ms |
| 1920×1080 | MJPEG | 24 | **42 ms** | 46 ms |

### Prerequisites

- NVIDIA GPU with CUDA 12.x support
- Docker with `nvidia-container-toolkit`
- v4l2loopback kernel module
- Physical webcam at `/dev/video0`
- **NVIDIA AR SDK** installed at `/usr/local/ARSDK` on the host (used as build context)

### Quick Start (Docker)

#### Option A: Pull pre-built image from GHCR (no build required)

A GitHub Actions workflow builds and publishes the image to GitHub Container
Registry. You only need to log in to GHCR and pull:

```bash
# Log in to GHCR (uses your GitHub Personal Access Token with read:packages)
echo $GITHUB_TOKEN | docker login ghcr.io -u n00b001 --password-stdin

# Pull and run
docker run -d --rm --gpus all \
  --device /dev/video0 --device /dev/video10 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  ghcr.io/n00b001/maxine-eye-contact-webcam/arsdk-gaze:latest \
  /usr/local/bin/maxine_ar_webcam --mjpeg /dev/video10 1920 1080
```

#### Option B: Build locally (ARSDK required)

If you prefer to build yourself (or the GHCR image is not yet published):

```bash
cd docker/

# One-time: copy ARSDK into build context
cp -rL /usr/local/ARSDK arsdk

# Build the base image (proprietary layer)
docker build -f Dockerfile.base -t ghcr.io/n00b001/maxine-eye-contact-webcam/arsdk-base:latest .

# Build the app image
docker build -t arsdk-gaze:latest .
```

#### Option C: Docker Compose

```bash
# Uses the GHCR image by default
docker compose up -d
```

See `docker-compose.yml` (single node) or `docker-stack.yml` (Docker Swarm).

#### CLI Options
- `--mjpeg` — request MJPEG from the camera (required for 720p/1080p @ 30 fps)
- `--no-warmup` — skip the default 30-frame warmup
- Arguments: `[v4l2-device] [width] [height]` (default: `/dev/video10 640 480`)

### GitHub Actions CI / GHCR Publishing

The repo includes `.github/workflows/docker-build.yml` which builds the image
on **GitHub-hosted runners** (`ubuntu-latest`) and pushes it to GitHub Container
Registry (GHCR).

Because the NVIDIA ARSDK is proprietary and cannot be committed to a public
repository, the build is split into two images:

1. **`arsdk-base`** — contains only the proprietary ARSDK binaries. Built
   **once locally** on a machine with `/usr/local/ARSDK` and pushed to GHCR.
2. **`arsdk-gaze`** — the application image. Built automatically by GitHub
   Actions on every push to `main`, pulling `arsdk-base` from GHCR.

#### One-time: build & push the ARSDK base image

On the machine that has ARSDK installed:

```bash
# Log in to GHCR (token needs write:packages scope)
echo $GITHUB_TOKEN | docker login ghcr.io -u n00b001 --password-stdin

# Build and push the base image
./scripts/push-arsdk-base.sh latest
```

This copies `/usr/local/ARSDK` into `docker/arsdk/`, builds
`ghcr.io/n00b001/maxine-eye-contact-webcam/arsdk-base:latest`, and pushes it.

#### Automatic app builds (GitHub Actions)

Once the base image is on GHCR, the workflow in `.github/workflows/docker-build.yml`
will build and publish the app image on every push to `main` or version tag:

```
ghcr.io/n00b001/maxine-eye-contact-webcam/arsdk-gaze:latest
```

No self-hosted runner required — it runs on `ubuntu-latest`.

### Systemd Service (Auto-start)

```bash
sudo cp maxine-ar-sdk-webcam.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now maxine-ar-sdk-webcam.service
sudo journalctl -u maxine-ar-sdk-webcam -f
```

---

## Python NIM Pipeline (Fallback)

Captures from a V4L2 webcam, sends H.264 MP4 GOPs to the NVIDIA Maxine Eye
Contact NIM via gRPC, and writes corrected frames to a v4l2loopback virtual
webcam for use in Zoom, Teams, OBS, Chrome, etc.

**Status:** Working end-to-end.  
**Latency:** ~1.0 s at 640×480 @ 30 fps; ~2.0 s at 1280×720 @ 30 fps.

> ⚠️ The NIM processes **discrete MP4 files**, not a live stream.  
> The pipeline works by capturing 1-second GOPs, encoding them to standard MP4,
> sending each GOP to the NIM, and decoding the response.

### Architecture

```
[Webcam /dev/video0] -> OpenCV V4L2 capture (MJPEG) -> BGR24 frames
    |
    v
[Collect 1-second GOP] -> FFmpeg libx264 -> streamable MP4 (-movflags +faststart)
    |
    v
[gRPC stream] -> Maxine Eye Contact NIM (:8003)
    |
    v
[NIM returns MP4] -> OpenCV VideoCapture decode -> BGR24 frames
    |
    v
[FFmpeg rawvideo -> v4l2] -> /dev/video10 (v4l2loopback)
```

**Validated encoding constraints:**
- Container: **standard MP4** (`ftyp` → `moov` → `mdat`) via `-movflags +faststart`
- Video: H.264 8-bit, yuv420p, Baseline or Main profile
- The NIM uses `qtdemux` internally; **fMP4 / MPEG-TS are NOT accepted**.

### Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Ubuntu 24.04+ |
| GPU | NVIDIA RTX 4090 (or any CUDA GPU) |
| Driver | NVIDIA 550+ |
| Docker | With nvidia-container-toolkit |
| RAM | 8 GB+ |
| Camera | Any V4L2 webcam |
| Python | 3.12 (managed by `uv`) |

### Quick Start (Python NIM)

#### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

#### 2. Install system dependencies

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg v4l2loopback-dkms v4l2loopback-utils v4l-utils git
```

#### 3. Configure v4l2loopback

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="MaxineEyeContact" exclusive_caps=0 max_buffers=4
ls -la /dev/video10
```

Make it persistent:

```bash
echo 'options v4l2loopback devices=1 video_nr=10 card_label="MaxineEyeContact" exclusive_caps=0 max_buffers=4' | sudo tee /etc/modprobe.d/v4l2loopback.conf
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf
```

#### 4. Set up the project

```bash
cd maxine-eye-contact-webcam
uv python install 3.12
uv python pin 3.12
uv sync
```

#### 5. Build protobuf definitions

```bash
./build_proto.sh
```

#### 6. Start the Maxine Eye Contact NIM

```bash
export NGC_API_KEY=<your-ngc-api-key>

docker run -it --rm --name=maxine-eye-contact-nim \
  --runtime=nvidia \
  --gpus all \
  --shm-size=6GB \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e MAXINE_MAX_CONCURRENCY_PER_GPU=1 \
  -e NIM_HTTP_API_PORT=8000 \
  -p 8002:8000 \
  -p 8003:8001 \
  nvcr.io/nim/nvidia/maxine-eye-contact:latest
```

Wait for "Maxine GRPC Service: Listening".

#### 7. Run the pipeline

**Recommended – 640×480 (most reliable, ~1 s latency):**

```bash
uv run python maxine_webcam_pipeline.py --resolution 480p
```

**Higher quality – 1280×720 (~2 s latency, occasional frame drops after long runs):**

```bash
uv run python maxine_webcam_pipeline.py --resolution 720p
```

**Dry-run (test capture → output without NIM):**

```bash
uv run python maxine_webcam_pipeline.py --dry-run --resolution 480p
```

#### 8. Use in your app

Select **"MaxineEyeContact"** as your camera in Zoom, Teams, OBS, Chrome, etc.

### Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `/dev/video0` | Input camera device |
| `--output` | `/dev/video10` | v4l2loopback output device |
| `--resolution` | `720p` | Preset: `480p`, `720p`, `1080p` |
| `--width` | (from preset) | Override width |
| `--height` | (from preset) | Override height |
| `--fps` | `30` | Target FPS |
| `--gop` | `30` | GOP size (frames per NIM request) |
| `--nim` | `127.0.0.1:8003` | NIM gRPC endpoint |
| `--nvenc` | (unset) | Use NVENC instead of libx264 |
| `--bitrate` | `8M` | Video bitrate |
| `--temporal` | `0xFFFFFFFF` | Temporal smoothing |
| `--eye-size` | `4` | Eye size sensitivity (2–6) |
| `--no-camera-controls` | (unset) | Skip `v4l2-ctl` setup |
| `--dry-run` | (unset) | Bypass NIM (test only) |

### Systemd Auto-Start (Python NIM)

```bash
sudo mkdir -p /opt/maxine-eye-contact-webcam
sudo cp -r . /opt/maxine-eye-contact-webcam/
sudo cp maxine-webcam.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable maxine-webcam.service
sudo systemctl start maxine-webcam.service
sudo journalctl -u maxine-webcam -f
```

The service uses `uv run` so the environment stays in sync automatically.

---

## Troubleshooting

### "Cannot open /dev/video10"
```bash
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="MaxineEyeContact" exclusive_caps=0 max_buffers=4
```

### Black screen in video app
- Ensure `exclusive_caps=0` is set when loading v4l2loopback
- Some apps need a restart after the virtual cam appears
- Test with: `ffplay -f v4l2 -i /dev/video10`

### Webcam only shows 15 fps
The Logitech C910 (and similar cameras) drop to 15 fps in low light via `exposure_dynamic_framerate`. The pipeline automatically disables this with `v4l2-ctl`. If it doesn't work, run manually:
```bash
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_dynamic_framerate=0
```

### High latency / stuttering at 720p (Python NIM)
Switch to 480p for stable operation, or migrate to the **Native AR SDK (Docker)** pipeline for sub-50 ms latency at 1080p.

### gRPC connection errors (Python NIM)
- Verify NIM container is running: `docker logs maxine-eye-contact-nim`
- Check port mapping: your NIM exposes 8001 internally, mapped to 8003 on host
- Test connectivity: `grpcurl --plaintext localhost:8003 grpc.health.v1.Health/Check`

---

## License

This project follows the MIT license. The Maxine Eye Contact NIM and NVIDIA AR SDK require acceptance of NVIDIA's respective licenses.
