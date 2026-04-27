# Maxine Eye Contact – Unified TRT-Fused GPU Pipeline

A single Docker container runs FasterLivePortrait (FLP) head-pose correction and
NVIDIA Maxine AR SDK gaze redirection on a shared `cudaStream_t` with zero
device-to-host copies between stages. Every frame crosses the PCIe bus exactly
twice: once inbound from the webcam and once outbound to the v4l2loopback sink.
The result is exposed as `/dev/video10` for use in Zoom, Teams, OBS, or any V4L2
consumer.

## Architecture

```
/dev/video0 (webcam)
    |
    | H2D
    v
FLP TRT engines x 7
  (appearance_feature_extractor, retinaface_det, face_2dpose_106,
   landmark, motion_extractor, warping_spade, stitching)
    |
    | GPU  [paste_back_pytorch monkey-patched -> stash GPU tensor]
    v
torch channel-flip + uint8 cast  (GPU, same stream)
    |
    | GPU
    v
maxine_ar_ext.GazeRedirect.run(in_dptr, pitch, out_dptr, pitch)
  AR SDK TRT engine (NvCVImage_Init, NvAR_SetCudaStream, same stream)
    |
    | D2H
    v
/dev/video10 (v4l2loopback -> Zoom / Teams / OBS)
```

Arrows labelled GPU stay on-device. The only host crossings are the two
labelled H2D and D2H at the top and bottom of the chain.

## Key source files

| File | Role |
|---|---|
| `flp_gpu_adapter.py` | Monkey-patches `paste_back_pytorch`; `FLPFrontalizer` exposes paste-back as a CUDA tensor |
| `gpu_format_convert.py` | GPU RGB-planar-float32 to BGR-uint8 helpers (torch/cupy/nvcv backends) |
| `maxine_fused_pipeline.py` | Single-process entry point; owns the shared `torch.cuda.Stream` |
| `docker/src/maxine_ar_ext/` | pybind11 shim over AR SDK; uses `NvCVImage_Init` + `NvAR_SetCudaStream` |
| `docker/entrypoint.sh` | Checks/builds TRT engines on first start; writes sentinel |
| `setup.sh` | One-command host installer (apt, v4l2loopback, Docker image, weights, systemd) |
| `docker/Dockerfile` | Unified pipeline image |

## Prerequisites

- NVIDIA driver R535 or newer
- Docker with `nvidia-container-toolkit`
- GPU compute capability >= 7.0
- ~40 GB free on `/opt` (image ~10 GB, ONNX weights ~1.5 GB, TRT engines ~8 GB)
- A webcam at `/dev/video0`

## Quick start

```bash
./setup.sh
```

Flags:

| Flag | Effect |
|---|---|
| `--skip-docker-pull` | Skip pulling the GHCR container image |
| `--skip-weights-download` | Skip downloading FasterLivePortrait ONNX weights |
| `--force` | Re-run all steps even if already complete |
| `--help` | Show usage and exit |

After setup, select **MaxineFinal** as your camera in your video app.

## Portrait requirement

Copy a neutral, frontal, well-lit 1080p portrait of yourself to
`/opt/maxine-portrait.jpg` before starting the container. The entrypoint
mounts it read-only at `/app/src-portrait.jpg`. FLP uses this image as the
LivePortrait source face; without it head-pose correction cannot run and the
"no face detected" path activates every frame.

```bash
cp ~/my-portrait.jpg /opt/maxine-portrait.jpg
```

## First-run timing

On the first container start, `docker/entrypoint.sh` compiles all FLP ONNX
models to TRT engines. This takes **5–15 minutes** depending on GPU. Subsequent
starts skip the build because the sentinel file
`/opt/flp-checkpoints/.engines-built` is present.

## Developer workflow

```bash
uv run pytest                          # unit tests (CPU-only safe; GPU tests skip automatically)
uv run ruff check .                    # lint
uv run ruff format --check .           # format check
```

Add runtime dependencies with `uv add <package>`, dev dependencies with
`uv add --group dev <package>`. Never use `pip` directly (see `PROJECT_RULES.md`).

## Troubleshooting

**"No face detected" log messages on every frame.**
Ensure `/opt/maxine-portrait.jpg` is a clear, frontal portrait of you with good
lighting and no obstructions. The file is mounted into the container at startup;
restart the container after replacing it.

**CI fails at "Verify base image exists".**
The proprietary `arsdk-base` image must be built once on a machine with the
NVIDIA AR SDK installed and pushed to GHCR before CI can proceed:

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u n00b001 --password-stdin
./scripts/push-arsdk-base.sh latest
```

See `scripts/push-arsdk-base.sh` for details. Once the base image is in GHCR
the workflow builds the app image automatically on every push to `main`.

**TRT engine build hangs or ONNX files missing.**
Check that all ONNX files are present under
`/opt/flp-checkpoints/liveportrait_onnx/`. Download them on the host:

```bash
huggingface-cli download warmshao/FasterLivePortrait \
    --local-dir /opt/flp-checkpoints
```

Or via Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("warmshao/FasterLivePortrait", local_dir="/opt/flp-checkpoints")
```

## Comparison with the previous architecture

Before commit `4c38c7b` the repo ran two separate containers connected through
a v4l2loopback bridge at `/dev/video11`. Container 1 (C++) ran the AR SDK gaze
stage; container 2 (Python) ran the LivePortrait head-pose stage. Every frame
crossed the PCIe bus four times (H2D and D2H in each container), and a kernel
v4l2loopback device transferred pixel data between them.

The unified pipeline removes `/dev/video11` entirely. Both stages live in one
Python process on one CUDA stream. The inter-stage handoff is a device-pointer
argument to `gaze.run()`; no frame data ever leaves the GPU between FLP and the
AR SDK.

## License

MIT. The NVIDIA AR SDK and FasterLivePortrait require acceptance of their
respective upstream licenses.
