# Maxine Eye Contact + Head Pose – Real-time Webcam Pipeline

A chained real-time pipeline that captures from a physical webcam, applies
**eye-contact gaze redirection** (via the NVIDIA AR SDK) and **head-pose
correction** (via LivePortrait), and exposes the result as a virtual V4L2
webcam (`/dev/video10`) for use in Zoom, Teams, Chrome, OBS, etc.

## Architecture

```
/dev/video0 (physical webcam, MJPEG)
   │
   ▼  ── Stage 1: AR SDK Docker container ───────────────  ~33 ms
   │    • C++ native, NVIDIA AR SDK nvargazeredirection
   │    • Reads /dev/video0, writes /dev/video11
   │
/dev/video11 (intermediate v4l2loopback)
   │
   ▼  ── Stage 2: LivePortrait head-pose (Python) ────────  ~27 ms
   │    • MediaPipe Face Mesh → LivePortrait generator
   │    • fp16 + torch.compile on RTX 4090
   │    • Reads /dev/video11, writes /dev/video10
   │
/dev/video10 (final v4l2loopback, seen by Zoom/Teams/etc.)
```

**End-to-end latency (RTX 4090, 1080p @ 30 fps):** ~65 ms — well inside a
100 ms budget.

## Quick start

```bash
# One-time setup (installs deps, pulls image, downloads weights, starts services)
./setup.sh
```

That runs everything in order: apt packages, v4l2loopback (2 devices),
Docker + GHCR image, LivePortrait weights (~630 MB), both systemd units.
First service start pays a ~60 s `torch.compile` warmup; steady state is
~27 ms/frame in Stage 2.

Then select **MaxineFinal** as your camera in your video app.

---

## Legacy pipelines

Two alternative paths are still in the repo for specific situations:

| Path | Latency | When to use |
|------|---------|-------------|
| **AR SDK only** (no head-pose) | ~33 ms | If you don't care about head pose and want absolute minimum latency |
| **Python NIM** (gRPC) | ~1–2 s | No AR SDK licence? Still works via the Maxine NIM container |

These are documented further down under "Native AR SDK Pipeline" and
"Python NIM Pipeline".

---

## Native AR SDK Pipeline (Stage 1 detail)

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

#### Option A: One-command setup (recommended)

Run `setup.sh` to install all host dependencies, configure the virtual webcam,
pull the pre-built GHCR image, and start the systemd service:

```bash
# 1. Log in to GHCR (one-time)
echo $GITHUB_TOKEN | docker login ghcr.io -u n00b001 --password-stdin

# 2. Run the setup script
./setup.sh
```

That's it. The script will:
- Install `ffmpeg`, `v4l2loopback-dkms`, `v4l-utils`, and kernel headers
- Load and persist the `v4l2loopback` kernel module (`/dev/video10`)
- Pull `ghcr.io/n00b001/maxine-eye-contact-webcam/arsdk-gaze:latest`
- Install and start the systemd service

#### Option B: Manual Docker run

If you prefer not to use the setup script:

```bash
# Log in to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u n00b001 --password-stdin

# Pull and run
docker run -d --rm --gpus all \
  --device /dev/video0 --device /dev/video10 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  ghcr.io/n00b001/maxine-eye-contact-webcam/arsdk-gaze:latest \
  /usr/local/bin/maxine_ar_webcam --mjpeg /dev/video10 1920 1080
```

#### Option C: Build locally (ARSDK required)

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

#### Option D: Docker Compose

```bash
# Uses the GHCR image by default
docker compose up -d
```

See `docker-compose.yml` (single node) or `docker-stack.yml` (Docker Swarm).

#### CLI Options
- `--mjpeg` — request MJPEG from the camera (required for 720p/1080p @ 30 fps)
- `--no-warmup` — skip the default 30-frame warmup
- Arguments: `[v4l2-device] [width] [height]` (default: `/dev/video10 640 480`)

#### Tunable Gaze Parameters

The gaze redirection behavior can be customized via **environment variables** (passed through by the systemd service):

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| `GAZE_EYE_SIZE` | `2` | `2`–`5` | Eye region size for redirection. Larger = bigger eye area modified. |
| `GAZE_LANDMARKS` | `126` | `68`, `126` | Number of facial landmarks tracked. |
| `GAZE_NO_REDIRECT` | `0` | `0`/`1` | Set to `1` to disable gaze redirection (estimation only). |
| `GAZE_NO_STABILIZE` | `0` | `0`/`1` | Set to `1` to disable temporal face stabilization. |
| `GAZE_NO_CUDA_GRAPH` | `0` | `0`/`1` | Set to `1` to disable CUDA Graph optimization. |
| `GAZE_PITCH_LOW` | `10.0` | `10.0`–`35.0` | Lower pitch threshold (°). Inside this angle, gaze is fully redirected to camera. |
| `GAZE_PITCH_HIGH` | `35.0` | `10.0`–`35.0` | Upper pitch threshold (°). Beyond this, no redirection occurs. |
| `GAZE_YAW_LOW` | `10.0` | `10.0`–`35.0` | Lower yaw threshold (°). Same behavior as pitch but for yaw. |
| `GAZE_YAW_HIGH` | `35.0` | `10.0`–`35.0` | Upper yaw threshold (°). |
| `GAZE_HEAD_PITCH_LOW` | `10.0` | `10.0`–`35.0` | Lower head-pose pitch threshold (°). |
| `GAZE_HEAD_PITCH_HIGH` | `35.0` | `10.0`–`35.0` | Upper head-pose pitch threshold (°). |
| `GAZE_HEAD_YAW_LOW` | `10.0` | `10.0`–`35.0` | Lower head-pose yaw threshold (°). |
| `GAZE_HEAD_YAW_HIGH` | `35.0` | `10.0`–`35.0` | Upper head-pose yaw threshold (°). |

#### Head Pose Correction

The Python NIM pipeline supports optional real-time head pose correction that rotates the head to face the camera. This is useful when the user is looking significantly off-center.

| Flag | Default | Description |
|------|---------|-------------|
| `--head-pose` | (unset) | Enable head pose correction |
| `--head-pose-strength` | `1.0` | Correction strength (0.0–1.0) |
| `--head-pose-yaw-limit` | `45.0` | Maximum yaw (°) to correct beyond which the frame is passed through |

**Example:**

```bash
uv run python maxine_webcam_pipeline.py --head-pose --head-pose-strength 0.8 --resolution 480p
```

Two backends are available, selected via `--head-pose-engine` / `HEAD_POSE_ENGINE`:

| Engine | Latency | Effect | Dependencies |
|--------|---------|--------|--------------|
| `geometric` (default) | ~5 ms (CPU) | **Roll-only** — in-plane tilt is corrected cleanly; yaw/pitch can't be rotated by a 2D affine | MediaPipe |
| `liveportrait` | ~37 ms (RTX 4090, fp16) | **Genuine 3D head rotation** — the generative net re-renders the head at the target pose, including occluded pixels | PyTorch + LivePortrait weights (~630 MB) |

**Geometric engine setup.** `mediapipe` pins `protobuf < 5`, which conflicts
with the generated `eyecontact_pb2` stubs, so it's installed outside the
lockfile. The pipeline gracefully disables `--head-pose` when `mediapipe`
is missing.

```bash
uv pip install 'mediapipe>=0.10,<0.10.20'
```

**LivePortrait engine setup.** Clone the repo, download weights, install
the optional extra:

```bash
git clone --depth 1 https://github.com/KwaiVGI/LivePortrait.git vendor/LivePortrait
uv sync --extra liveportrait
uv run python -c "from huggingface_hub import snapshot_download; \
snapshot_download('KlingTeam/LivePortrait', local_dir='vendor/LivePortrait/pretrained_weights', \
allow_patterns=['liveportrait/**', 'insightface/**'])"
```

Set `HEAD_POSE_ENGINE=liveportrait` (default in the shipped systemd unit)
to use it. First frame pays a ~500 ms warmup cost for CUDA kernel
compilation; steady-state is ~37 ms/frame on RTX 4090. Setting
`HEAD_POSE_COMPILE=1` enables `torch.compile` — adds ~60 s first-frame
compile time for 20–30 % steady-state speedup.

**How thresholds work:** Between `*Low` and `*High`, the redirected gaze linearly transitions from full camera-facing to estimated natural gaze. Above the high threshold, no redirection is applied.

**To change parameters:**

```bash
# Edit the service environment variables
sudo systemctl edit maxine-ar-sdk-webcam --full
# Change the Environment= lines, save, then:
sudo systemctl daemon-reload
sudo systemctl restart maxine-ar-sdk-webcam
```

Or test once with `docker run`:

```bash
docker run -d --rm --gpus all --device /dev/video0 --device /dev/video10 \
  -e GAZE_EYE_SIZE=4 \
  -e GAZE_PITCH_LOW=15.0 \
  -e GAZE_PITCH_HIGH=25.0 \
  ghcr.io/n00b001/maxine-eye-contact-webcam/arsdk-gaze:latest \
  /usr/local/bin/maxine_ar_webcam --mjpeg /dev/video10 1920 1080
```

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

> **⚠️ Private base image access**
>
> GHCR packages pushed from a personal account default to **private**. The
> GitHub Actions `GITHUB_TOKEN` cannot access private user-scoped packages.
>
> **Option 1 (recommended):** Make the base image public:
> 1. Visit `https://github.com/users/n00b001/packages/container/package/maxine-eye-contact-webcam%2Farsdk-base/settings`
> 2. Under "Danger Zone", click "Change visibility" → "Make public"
>
> **Option 2:** Use a Personal Access Token (PAT):
> 1. Create a classic PAT with `read:packages` scope at
>    `https://github.com/settings/tokens`
> 2. Add it as a repository secret named `GHCR_PAT` at
>    `https://github.com/n00b001/maxine-eye-contact-webcam/settings/secrets/actions`
> 3. The workflow will automatically use `GHCR_PAT` for GHCR login.

### Systemd Service (Auto-start)

`setup.sh` installs and enables the service automatically. To manage it manually:

```bash
sudo systemctl start  maxine-ar-sdk-webcam   # start
sudo systemctl stop   maxine-ar-sdk-webcam   # stop
sudo systemctl status maxine-ar-sdk-webcam   # status
sudo journalctl -u maxine-ar-sdk-webcam -f   # follow logs
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

# Optional: install mediapipe for head-pose correction (outside the
# lockfile because mediapipe pins protobuf<5 while the generated pb2
# stubs require protobuf>=5).
uv pip install 'mediapipe>=0.10,<0.10.20'
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
| `--head-pose` | (unset) | Enable optional head-pose correction (requires `mediapipe`) |
| `--head-pose-strength` | `1.0` | Head-pose correction strength (0.0–1.0) |
| `--head-pose-yaw-limit` | `45.0` | Disable correction when yaw exceeds this angle (°) |

### Systemd Auto-Start (Python NIM)

The Python NIM pipeline ships as a systemd unit that runs **eye-contact
correction (via NIM) and head-pose correction (via MediaPipe) together** by
default. All tuning knobs are exposed as `Environment=` entries in the
unit — edit them the same way you edit the AR SDK pipeline's `GAZE_*`
variables.

```bash
sudo mkdir -p /opt/maxine-eye-contact-webcam
sudo cp -r . /opt/maxine-eye-contact-webcam/
sudo cp maxine-webcam.service /etc/systemd/system/

# Head-pose requires mediapipe in the venv (see "Head Pose Correction"
# above — mediapipe pins protobuf<5 so it lives outside the uv lockfile).
cd /opt/maxine-eye-contact-webcam
uv sync
uv pip install 'mediapipe>=0.10,<0.10.20'

sudo systemctl daemon-reload
sudo systemctl enable --now maxine-webcam.service
sudo journalctl -u maxine-webcam -f
```

#### Tunable environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `NIM_TARGET` | `127.0.0.1:8003` | gRPC address of the Maxine Eye Contact NIM |
| `RESOLUTION` | `720p` | `480p` / `720p` / `1080p` |
| `FPS` | `30` | Target frame rate |
| `GOP` | `30` | Frames per NIM request |
| `EYE_SIZE` | `4` | NIM `eye_size_sensitivity` (2–6) |
| `TEMPORAL` | `0xFFFFFFFF` | NIM temporal smoothing bitmask |
| `DETECT_CLOSURE` | `0` | NIM `detect_closure` flag |
| `BITRATE` | `8M` | Video bitrate fed to the NIM |
| `NVENC` | `0` | Set `1` to use NVENC instead of libx264 |
| `HEAD_POSE` | `1` | Set `0` to disable head-pose correction |
| `HEAD_POSE_ENGINE` | `liveportrait` | `geometric` (CPU, roll-only) or `liveportrait` (GPU, full 3D) |
| `HEAD_POSE_STRENGTH` | `1.0` | 0.0 = identity, 1.0 = full correction toward frontal (fallback for the per-axis `*_STRENGTH` vars below) |
| `HEAD_POSE_PITCH_STRENGTH` | `1.0` | Per-axis strength. `1.0` = level pitch, `0.0` = disable pitch correction |
| `HEAD_POSE_YAW_STRENGTH` | `1.0` | Per-axis strength. `1.0` = face camera, `0.0` = disable yaw correction |
| `HEAD_POSE_ROLL_STRENGTH` | `0.0` | Per-axis strength. Default `0.0` keeps natural head tilt (forced roll=0 looks uncanny). Set `1.0` to level the head. |
| `HEAD_POSE_PITCH_LIMIT` | `45.0` | Skip correction when `|pitch|` exceeds this (°) |
| `HEAD_POSE_YAW_LIMIT` | `45.0` | Skip correction when `|yaw|` exceeds this (°) |
| `HEAD_POSE_ROLL_LIMIT` | `45.0` | Skip correction when `|roll|` exceeds this (°) |
| `HEAD_POSE_COMPILE` | `0` | Enable `torch.compile` for the LP engine (+60 s warmup, -25 % steady-state) |
| `INPUT_DEVICE` | `/dev/video0` | Physical webcam |
| `OUTPUT_DEVICE` | `/dev/video10` | v4l2loopback sink |

**To change values:**

```bash
sudo systemctl edit maxine-webcam --full
# edit Environment= lines, save, then:
sudo systemctl daemon-reload
sudo systemctl restart maxine-webcam
```

One-off override without editing the unit (the CLI always wins):

```bash
uv run python maxine_webcam_pipeline.py --head-pose-strength 0.5 --eye-size 6
```

The service uses `uv run` so the Python interpreter and deps stay in sync
with the working tree.

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

## Development

### Local setup
```bash
uv sync --all-groups          # install runtime + dev deps
```

This project uses `core.hooksPath=scripts/` so the hooks in `scripts/pre-commit` and `scripts/pre-push` run automatically — no extra install step. Verify they are executable:

```bash
chmod +x scripts/pre-commit scripts/pre-push
```

### Checks that must pass

| Stage | Command | What it runs |
|-------|---------|--------------|
| Local / pre-commit | `scripts/pre-commit` | `ruff check`, `ruff format --check`, `pytest --cov --cov-fail-under=80` |
| Local / pre-push | `scripts/pre-push` | `pytest --cov --cov-fail-under=80` |
| GitHub Actions | `.github/workflows/python-ci.yml` | same ruff + pytest coverage gate |

### Running individual checks

```bash
uv run ruff check .                    # lint
uv run ruff format .                   # auto-format
uv run pytest                          # tests
uv run pytest --cov --cov-fail-under=80  # tests with coverage gate
```

### Adding dependencies

Use `uv add <package>` (runtime) or `uv add --group dev <package>` (dev only). Never use `pip` directly. See `PROJECT_RULES.md` for the full ruleset.

### Implementation notes

The shipped head-pose correction is a lightweight geometric approximation (MediaPipe Face Mesh → OpenCV `solvePnP` → affine warp + alpha blend). It runs on CPU in ~10–20 ms/frame. The original design doc in `plans/head-pose-correction-plan.md` explores a higher-quality LivePortrait (GPU, ~78 FPS on RTX 4090) path as a future upgrade.

---

## License

This project follows the MIT license. The Maxine Eye Contact NIM and NVIDIA AR SDK require acceptance of NVIDIA's respective licenses.
