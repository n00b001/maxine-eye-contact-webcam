#!/usr/bin/env python3
"""
Maxine Eye Contact NIM – Real-time Webcam Pipeline
====================================================
Captures from a V4L2 webcam, sends streamable MP4 GOPs to the Maxine Eye
Contact NIM via gRPC, and writes the corrected frames to a v4l2loopback
device for use in video-conferencing apps.

Encoding constraints (validated empirically):
  • Container: standard MP4 (ftyp → moov → mdat) via -movflags +faststart
  • Video:    H.264 8-bit, yuv420p, Baseline or Main profile
  • The NIM uses qtdemux internally; fMP4 / MPEG-TS are NOT accepted.

Latency model (1280×720 @ 30 fps, GOP=30):
  capture 1.0 s  +  encode ~0.05 s  +  NIM ~0.90 s  +  decode ~0.05 s
  ≈ 2.0 s end-to-end per frame.  Output is perfectly cadenced at 30 fps
  because the NIM cycle (≈1.0 s) matches the capture interval (1.0 s).

Usage:
    uv run python maxine_webcam_pipeline.py --output /dev/video10
    uv run python maxine_webcam_pipeline.py --dry-run --show
"""

from __future__ import annotations

import argparse
import contextlib
import os
import queue
import signal
import subprocess
import threading
import time
from collections.abc import Iterator

import cv2
import grpc
import numpy as np

try:
    import eyecontact_pb2
    import eyecontact_pb2_grpc
except ImportError as exc:
    print("ERROR: Protobuf stubs not found. Run ./build_proto.sh first.")
    raise SystemExit(1) from exc

# Optional head-pose correction (requires mediapipe)
try:
    from frontalizer import Frontalizer
    from head_pose_estimator import HeadPoseEstimator
except ImportError as exc:
    HeadPoseEstimator = None  # type: ignore[misc,assignment]
    Frontalizer = None  # type: ignore[misc,assignment]
    print(
        "WARNING: head_pose_estimator/frontalizer not available "
        "(install mediapipe to enable --head-pose):",
        exc,
    )

# Optional LivePortrait engine — generative head-pose correction. Heavy
# deps (torch + CUDA + model weights); install via `uv sync --extra
# liveportrait` and clone vendor/LivePortrait per README.
try:
    from liveportrait_frontalizer import LivePortraitFrontalizer
except ImportError as exc:
    LivePortraitFrontalizer = None  # type: ignore[misc,assignment]
    _LP_IMPORT_ERROR = exc
else:
    _LP_IMPORT_ERROR = None

# ---------------------------------------------------------------------------
# Global shutdown handling
# ---------------------------------------------------------------------------
_shutdown = threading.Event()


def _handle_signal(signum, frame):
    _shutdown.set()


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_ffmpeg(
    args: list[str], stdin: bytes | None = None, timeout: float = 30.0
) -> tuple[bytes, bytes, int]:
    """Run FFmpeg and return (stdout, stderr, returncode)."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if stdin is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = proc.communicate(input=stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
    return stdout, stderr, proc.returncode


def encode_gop_to_mp4(
    frames: list[np.ndarray],
    fps: float,
    *,
    use_nvenc: bool = False,
    bitrate: str = "8M",
    gop_size: int | None = None,
) -> bytes:
    """Encode a list of BGR frames to a streamable MP4 byte blob."""
    import tempfile

    if not frames:
        return b""

    h, w = frames[0].shape[:2]
    gop = gop_size or len(frames)

    # Build encoder args
    if use_nvenc:
        enc = [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p1",  # fastest
            "-tune",
            "ull",  # ultra-low-latency
            "-rc",
            "cbr",
            "-b:v",
            bitrate,
            "-bufsize",
            "4M",
            "-profile:v",
            "main",
            "-g",
            str(gop),
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]
    else:
        enc = [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-b:v",
            bitrate,
            "-bufsize",
            "4M",
            "-profile:v",
            "baseline",
            "-g",
            str(gop),
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]

    flat = b"".join(f.tobytes() for f in frames)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            *enc,
            "-movflags",
            "+faststart",
            tmp_path,
        ]
        stdout, stderr, rc = _run_ffmpeg(cmd, stdin=flat, timeout=60.0)
        if rc != 0:
            err = stderr.decode("utf-8", "replace")[-500:]
            raise RuntimeError(f"FFmpeg encode failed (rc={rc}): {err}")

        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def decode_mp4_to_frames(mp4_data: bytes, shape: tuple[int, int]) -> list[np.ndarray]:
    """Decode an MP4 byte blob to a list of BGR frames using OpenCV."""
    import tempfile

    h, w = shape

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(mp4_data)

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open the MP4 file")

        frames: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        cap.release()
        return frames
    except Exception as exc:
        raise RuntimeError(f"OpenCV decode failed: {exc}") from exc
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


def _set_camera_controls(device: str) -> None:
    """Disable exposure_dynamic_framerate to maintain constant FPS."""
    with contextlib.suppress(Exception):
        subprocess.run(
            ["v4l2-ctl", "-d", device, "--set-ctrl=exposure_dynamic_framerate=0"],
            capture_output=True,
            check=False,
            timeout=5,
        )


def capture_thread(
    device: str,
    width: int,
    height: int,
    fps: float,
    q: queue.Queue[np.ndarray],
    *,
    setup_controls: bool = True,
) -> None:
    """Capture raw BGR frames from V4L2 and push to *q*."""
    if setup_controls:
        _set_camera_controls(device)

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        # Fallback: try index
        try:
            idx = int(device.replace("/dev/video", ""))
            cap = cv2.VideoCapture(idx)
        except ValueError:
            pass

    if not cap.isOpened():
        print(f"[Capture] FATAL: cannot open {device}")
        _shutdown.set()
        return

    # Request MJPEG for highest frame-rate at HD resolutions
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Warm-up
    for _ in range(5):
        cap.read()

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    print(
        f"[Capture] {actual_w}x{actual_h} @ {actual_fps:.1f} fps  fmt={fourcc_str}  device={device}"
    )

    dropped = 0
    while not _shutdown.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.001)
            continue

        # Resize if the camera gave us something else
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        if q.full():
            try:
                q.get_nowait()
                dropped += 1
            except queue.Empty:
                pass
        q.put(frame)

    cap.release()
    if dropped:
        print(f"[Capture] Exited (dropped {dropped} frames)")
    else:
        print("[Capture] Exited")


def nim_pipeline_thread(
    grpc_target: str,
    config: eyecontact_pb2.RedirectGazeConfig,
    gop_size: int,
    fps: float,
    raw_q: queue.Queue[np.ndarray],
    out_q: queue.Queue[np.ndarray],
    *,
    use_nvenc: bool = False,
    bitrate: str = "8M",
    dry_run: bool = False,
    head_pose: bool = False,
    head_pose_strength: float = 1.0,
    head_pose_yaw_limit: float = 45.0,
    head_pose_engine: str = "geometric",
    head_pose_compile: bool = False,
) -> None:
    """
    Collect GOPs from *raw_q*, encode → NIM → decode, push to *out_q*.
    """
    print(
        f"[Pipeline] GOP={gop_size}, NVENC={'yes' if use_nvenc else 'no'}, "
        f"dry_run={'yes' if dry_run else 'no'}, "
        f"head_pose={'yes' if head_pose else 'no'}"
    )

    channel: grpc.Channel | None = None
    stub: eyecontact_pb2_grpc.MaxineEyeContactServiceStub | None = None

    if not dry_run:
        channel = grpc.insecure_channel(
            grpc_target,
            options=[
                ("grpc.max_send_message_length", 128 * 1024 * 1024),
                ("grpc.max_receive_message_length", 128 * 1024 * 1024),
            ],
        )
        stub = eyecontact_pb2_grpc.MaxineEyeContactServiceStub(channel)
        print(f"[Pipeline] gRPC channel -> {grpc_target}")

    # Optional head-pose corrector (initialised once per thread). The
    # engine selector routes between the lightweight MediaPipe+affine path
    # and the LivePortrait generative path.
    hpe: HeadPoseEstimator | None = None
    frontalizer = None  # Frontalizer or LivePortraitFrontalizer
    if head_pose:
        if HeadPoseEstimator is None:
            print(
                "[Pipeline] WARNING: --head-pose requested but mediapipe is not "
                "installed; passing frames through unchanged"
            )
            head_pose = False
        elif head_pose_engine == "liveportrait":
            if LivePortraitFrontalizer is None:
                print(
                    "[Pipeline] WARNING: HEAD_POSE_ENGINE=liveportrait but "
                    f"liveportrait_frontalizer failed to import: {_LP_IMPORT_ERROR}. "
                    "Falling back to geometric engine."
                )
                frontalizer = Frontalizer(strength=head_pose_strength) if Frontalizer else None
            else:
                hpe = HeadPoseEstimator(static_image_mode=False)
                frontalizer = LivePortraitFrontalizer(
                    strength=head_pose_strength, compile_models=head_pose_compile
                )
                print(
                    f"[Pipeline] Head-pose corrector: LivePortrait engine "
                    f"(strength={head_pose_strength}, yaw_limit={head_pose_yaw_limit}, "
                    f"torch_compile={head_pose_compile})"
                )
        elif Frontalizer is not None:
            hpe = HeadPoseEstimator(static_image_mode=False)
            frontalizer = Frontalizer(strength=head_pose_strength)
            print(
                f"[Pipeline] Head-pose corrector: geometric engine "
                f"(strength={head_pose_strength}, yaw_limit={head_pose_yaw_limit})"
            )
        else:
            print("[Pipeline] WARNING: no head-pose engine available; disabled")
            head_pose = False
        if head_pose and (hpe is None or frontalizer is None):
            head_pose = False

    gop_idx = 0
    while not _shutdown.is_set():
        # Collect a GOP
        frames: list[np.ndarray] = []
        t0 = time.monotonic()
        while len(frames) < gop_size and not _shutdown.is_set():
            try:
                f = raw_q.get(timeout=1.0)
                frames.append(f)
            except queue.Empty:
                continue
        if len(frames) < gop_size // 2:
            continue
        t_collect = time.monotonic() - t0

        if dry_run:
            for f in frames:
                if out_q.full():
                    with contextlib.suppress(queue.Empty):
                        out_q.get_nowait()
                out_q.put(f)
            gop_idx += 1
            print(
                f"[Pipeline] GOP {gop_idx:03d}: {len(frames)} frames  "
                f"collect={t_collect:.2f}s  (dry-run)"
            )
            continue

        # Encode
        t_enc0 = time.monotonic()
        try:
            mp4_data = encode_gop_to_mp4(
                frames,
                fps,
                use_nvenc=use_nvenc,
                bitrate=bitrate,
                gop_size=gop_size,
            )
        except RuntimeError as exc:
            print(f"[Pipeline] Encode error: {exc}")
            continue
        t_enc = time.monotonic() - t_enc0

        # gRPC — bind loop variables via default args so the generator
        # captures the values at closure-creation time (ruff B023).
        def _make_requests(
            data: bytes = mp4_data,
            cfg: eyecontact_pb2.RedirectGazeConfig = config,
        ) -> Iterator[eyecontact_pb2.RedirectGazeRequest]:
            yield eyecontact_pb2.RedirectGazeRequest(config=cfg)
            chunk = 64 * 1024
            for i in range(0, len(data), chunk):
                yield eyecontact_pb2.RedirectGazeRequest(video_file_data=data[i : i + chunk])
            print(f"[Pipeline] Sent {len(data)} bytes to NIM")

        t_nim0 = time.monotonic()
        response_data = b""
        try:
            responses = stub.RedirectGaze(_make_requests())
            # First message is the echoed Config
            first = next(responses, None)
            if first is None:
                raise RuntimeError("Empty response stream")
            for resp in responses:
                if resp.HasField("video_file_data"):
                    response_data += resp.video_file_data
                elif resp.HasField("keepalive"):
                    print("[Pipeline] Keepalive received")
                    continue
        except grpc.RpcError as exc:
            print(f"[Pipeline] gRPC error: {exc.code()}: {exc.details()}")
            continue
        except Exception as exc:
            print(f"[Pipeline] Unexpected error: {exc}")
            continue
        t_nim = time.monotonic() - t_nim0

        # Decode
        t_dec0 = time.monotonic()
        try:
            out_frames = decode_mp4_to_frames(response_data, frames[0].shape[:2])
        except RuntimeError as exc:
            print(f"[Pipeline] Decode error: {exc}")
            continue
        t_dec = time.monotonic() - t_dec0

        # Optional head-pose correction (single MediaPipe pass per frame).
        t_hp = 0.0
        if head_pose and hpe is not None and frontalizer is not None:
            t_hp0 = time.monotonic()
            corrected: list[np.ndarray] = []
            for frame in out_frames:
                try:
                    landmarks = hpe.get_landmarks(frame)
                    if landmarks is not None:
                        pose = hpe.estimate_from_landmarks(landmarks, frame.shape[:2])
                        if pose is not None:
                            pitch, yaw, roll = pose
                            if abs(yaw) < head_pose_yaw_limit:
                                warped = frontalizer.frontalize(frame, landmarks, pitch, yaw, roll)
                                if warped is not None:
                                    x_min, y_min = landmarks.min(axis=0).astype(int)
                                    x_max, y_max = landmarks.max(axis=0).astype(int)
                                    face_rect = (x_min, y_min, x_max - x_min, y_max - y_min)
                                    frame = frontalizer.blend_back(frame, warped, face_rect)
                except Exception as exc:
                    print(f"[Pipeline] Head-pose correction warning: {exc}")
                corrected.append(frame)
            out_frames = corrected
            t_hp = time.monotonic() - t_hp0

        gop_idx += 1
        latency = t_collect + t_enc + t_nim + t_dec + t_hp
        print(
            f"[Pipeline] GOP {gop_idx:03d}: "
            f"collect={t_collect:.2f}s  encode={t_enc:.2f}s  "
            f"nim={t_nim:.2f}s  decode={t_dec:.2f}s  "
            f"hp={t_hp:.2f}s  latency≈{latency:.2f}s  "
            f"in={len(frames)} out={len(out_frames)}"
        )

        if len(out_frames) != len(frames):
            print(f"  WARNING: frame count mismatch {len(frames)} → {len(out_frames)}")

        for f in out_frames:
            if out_q.full():
                with contextlib.suppress(queue.Empty):
                    out_q.get_nowait()
            out_q.put(f)

    if hpe is not None:
        hpe.close()
    if channel:
        channel.close()
    print("[Pipeline] Exited")


def output_thread(
    device: str,
    width: int,
    height: int,
    fps: float,
    q: queue.Queue[np.ndarray],
    *,
    show_preview: bool = False,
) -> None:
    """Consume frames from *q* and write to v4l2loopback (via FFmpeg)."""
    if not device:
        print("[Output] No output device specified – frames will be discarded")
        while not _shutdown.is_set():
            with contextlib.suppress(queue.Empty):
                q.get(timeout=1.0)
        print("[Output] Exited")
        return

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-f",
        "v4l2",
        "-pix_fmt",
        "yuv420p",
        device,
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    print(f"[Output] {width}x{height}@{fps:.0f} → {device}")

    frame_interval = 1.0 / fps
    next_time: float | None = None
    frame_idx = 0

    while not _shutdown.is_set():
        try:
            frame = q.get(timeout=2.0)
        except queue.Empty:
            if proc.poll() is not None:
                err = proc.stderr.read().decode("utf-8", "replace")[-500:]
                print(f"[Output] FFmpeg died: {err}")
                break
            continue

        # Pace output to exactly fps
        now = time.monotonic()
        if next_time is None:
            next_time = now
        wait = next_time - now
        if wait > 0:
            time.sleep(wait)
        elif wait < -0.5:
            # We've fallen behind by >0.5 s; reset baseline
            next_time = now

        try:
            proc.stdin.write(frame.tobytes())
            proc.stdin.flush()
        except BrokenPipeError:
            break

        next_time += frame_interval
        frame_idx += 1

        if show_preview:
            cv2.imshow("Maxine Eye Contact", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                _shutdown.set()
                break

    proc.stdin.close()
    proc.wait(timeout=5)
    if show_preview:
        cv2.destroyAllWindows()
    print(f"[Output] Exited ({frame_idx} frames written)")


# ---------------------------------------------------------------------------
# CLI — env-var-driven defaults so the systemd unit can tune via Environment=
# ---------------------------------------------------------------------------


_TRUTHY = {"1", "true", "yes", "on", "y"}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in _TRUTHY


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int, *, base: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw, base)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw not in (None, "") else default


def build_parser() -> argparse.ArgumentParser:
    # Every default here falls through to an env var first so the systemd
    # unit can tune behaviour via Environment= lines (the GAZE_* pattern
    # used by the native AR SDK service). CLI args always win when
    # explicitly provided.
    p = argparse.ArgumentParser(description="Real-time Maxine Eye Contact via NIM + v4l2loopback")
    p.add_argument(
        "--input",
        default=_env_str("INPUT_DEVICE", "/dev/video0"),
        help="V4L2 webcam device (env: INPUT_DEVICE, default: /dev/video0)",
    )
    p.add_argument(
        "--output",
        default=_env_str("OUTPUT_DEVICE", "/dev/video10"),
        help="v4l2loopback output device (env: OUTPUT_DEVICE, default: /dev/video10)",
    )
    p.add_argument(
        "--resolution",
        default=_env_str("RESOLUTION", "720p"),
        choices=["480p", "720p", "1080p"],
        help="Preset resolution: 480p, 720p, 1080p (env: RESOLUTION, default: 720p)",
    )
    p.add_argument("--width", type=int, default=None, help="Override capture width")
    p.add_argument("--height", type=int, default=None, help="Override capture height")
    p.add_argument(
        "--fps",
        type=float,
        default=_env_float("FPS", 30.0),
        help="Target frame-rate (env: FPS, default: 30)",
    )
    p.add_argument(
        "--gop",
        type=int,
        default=_env_int("GOP", 30),
        help="GOP size — frames per NIM request (env: GOP, default: 30)",
    )
    p.add_argument(
        "--nim",
        default=_env_str("NIM_TARGET", "127.0.0.1:8003"),
        help="gRPC target (env: NIM_TARGET, default: 127.0.0.1:8003)",
    )
    p.add_argument(
        "--nvenc",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("NVENC"),
        help="Use NVENC instead of libx264 for encoding (env: NVENC)",
    )
    p.add_argument(
        "--bitrate",
        default=_env_str("BITRATE", "8M"),
        help="Video bitrate (env: BITRATE, default: 8M)",
    )
    p.add_argument(
        "--temporal",
        type=lambda s: int(s, 0),
        default=_env_int("TEMPORAL", 0xFFFFFFFF),
        help="NIM temporal smoothing (env: TEMPORAL, default: 0xFFFFFFFF)",
    )
    p.add_argument(
        "--detect-closure",
        type=lambda s: int(s, 0),
        default=_env_int("DETECT_CLOSURE", 0),
        help="NIM detect_closure (env: DETECT_CLOSURE, default: 0)",
    )
    p.add_argument(
        "--eye-size",
        type=lambda s: int(s, 0),
        default=_env_int("EYE_SIZE", 4),
        help="NIM eye_size_sensitivity, 2-6 (env: EYE_SIZE, default: 4)",
    )
    p.add_argument("--mode", type=lambda s: int(s, 0), default=0, help="DEPRECATED")
    p.add_argument("--quality", type=lambda s: int(s, 0), default=0, help="DEPRECATED")
    p.add_argument("--no-camera-controls", action="store_true", help="Skip v4l2-ctl camera setup")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip NIM; pass frames straight through (test capture/output)",
    )
    p.add_argument(
        "--show", action="store_true", help="Show local preview window (requires GUI OpenCV)"
    )
    p.add_argument(
        "--head-pose",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("HEAD_POSE"),
        help="Enable head-pose correction after NIM gaze redirection (env: HEAD_POSE)",
    )
    p.add_argument(
        "--head-pose-strength",
        type=float,
        default=_env_float("HEAD_POSE_STRENGTH", 1.0),
        help="Head-pose correction strength: 0.0=no correction, 1.0=full "
        "(env: HEAD_POSE_STRENGTH, default: 1.0)",
    )
    p.add_argument(
        "--head-pose-yaw-limit",
        type=float,
        default=_env_float("HEAD_POSE_YAW_LIMIT", 45.0),
        help="Skip correction when |yaw| exceeds this (env: HEAD_POSE_YAW_LIMIT, default: 45.0)",
    )
    p.add_argument(
        "--head-pose-engine",
        default=_env_str("HEAD_POSE_ENGINE", "geometric"),
        choices=["geometric", "liveportrait"],
        help=(
            "Correction backend: 'geometric' (MediaPipe + affine warp, CPU, "
            "roll-only effective) or 'liveportrait' (GPU generative net, "
            "full 3D head rotation, ~37 ms/frame on RTX 4090). "
            "env: HEAD_POSE_ENGINE, default: geometric"
        ),
    )
    p.add_argument(
        "--head-pose-compile",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("HEAD_POSE_COMPILE"),
        help=(
            "Enable torch.compile for the LivePortrait engine "
            "(first-frame latency jumps ~60 s for compilation, then -20..30%% "
            "steady-state). env: HEAD_POSE_COMPILE, default: off"
        ),
    )
    return p


def _resolve_resolution(args: argparse.Namespace) -> tuple[int, int]:
    presets = {
        "480p": (640, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
    }
    w, h = presets[args.resolution]
    if args.width is not None:
        w = args.width
    if args.height is not None:
        h = args.height
    return w, h


def main() -> None:
    args = build_parser().parse_args()
    width, height = _resolve_resolution(args)

    config = eyecontact_pb2.RedirectGazeConfig(
        temporal=args.temporal,
        eye_size_sensitivity=args.eye_size,
    )

    # Queues
    raw_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=120)
    out_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=120)

    threads = [
        threading.Thread(
            target=capture_thread,
            args=(args.input, width, height, args.fps, raw_q),
            kwargs={"setup_controls": not args.no_camera_controls},
            daemon=True,
        ),
        threading.Thread(
            target=nim_pipeline_thread,
            args=(
                args.nim,
                config,
                args.gop,
                args.fps,
                raw_q,
                out_q,
            ),
            kwargs={
                "use_nvenc": args.nvenc,
                "bitrate": args.bitrate,
                "dry_run": args.dry_run,
                "head_pose": args.head_pose,
                "head_pose_strength": args.head_pose_strength,
                "head_pose_yaw_limit": args.head_pose_yaw_limit,
                "head_pose_engine": args.head_pose_engine,
                "head_pose_compile": args.head_pose_compile,
            },
            daemon=True,
        ),
        threading.Thread(
            target=output_thread,
            args=(args.output, width, height, args.fps, out_q),
            kwargs={"show_preview": args.show},
            daemon=True,
        ),
    ]

    print("=" * 60)
    print("Maxine Eye Contact – Real-time Webcam Pipeline")
    print("=" * 60)
    for t in threads:
        t.start()

    # Wait for any thread to die or shutdown signal
    try:
        while not _shutdown.is_set():
            alive = [t for t in threads if t.is_alive()]
            if len(alive) < len(threads):
                dead = [t.name for t in threads if not t.is_alive()]
                print(f"[Main] Thread(s) died: {dead}")
                _shutdown.set()
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        _shutdown.set()

    print("[Main] Shutting down …")
    for t in threads:
        t.join(timeout=5.0)
    print("[Main] Done.")


if __name__ == "__main__":
    main()
