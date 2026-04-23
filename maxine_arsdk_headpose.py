#!/usr/bin/env python3
"""Stage 2 of the combined pipeline: LivePortrait head-pose correction.

Reads the AR-SDK gaze-corrected stream from ``INPUT_DEVICE`` (default
``/dev/video11``), runs MediaPipe face landmarks + LivePortrait
frontalisation, and writes the result to ``OUTPUT_DEVICE`` (default
``/dev/video10``) via v4l2loopback.

End-to-end latency target (1080p @ 30 fps, RTX 4090):

    AR SDK gaze      ~33 ms   (done in Stage 1 Docker container)
    kernel v4l2 hop   ~0 ms
    Python capture    ~1 ms
    MediaPipe face    ~5 ms   (video-mode tracker after first frame)
    LivePortrait      ~25 ms  (fp16 + torch.compile after warmup)
    v4l2 write        ~1 ms
    ------------------------
    total            ~65 ms   < 100 ms budget

All tuning is via env vars (``INPUT_DEVICE``, ``OUTPUT_DEVICE``,
``RESOLUTION``, ``FPS``, ``HEAD_POSE*``) so the systemd unit can drive
everything through ``Environment=`` lines.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Env-var helpers (same style as maxine_webcam_pipeline.py)
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


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw not in (None, "") else default


# ---------------------------------------------------------------------------
# Shutdown handling
# ---------------------------------------------------------------------------
_shutdown = threading.Event()


def _handle_signal(signum, frame):  # pragma: no cover
    _shutdown.set()


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# v4l2 capture / output
# ---------------------------------------------------------------------------


def capture_thread(
    device: str,
    width: int,
    height: int,
    fps: float,
    q: queue.Queue[np.ndarray],
) -> None:
    """Read MJPEG/raw frames from ``device`` into ``q``, dropping when full."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[Capture] FATAL: cannot open {device}", flush=True)
        _shutdown.set()
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    for _ in range(3):
        cap.read()  # warm-up

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    afps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Capture] {aw}x{ah} @ {afps:.1f} fps  device={device}", flush=True)

    dropped = 0
    while not _shutdown.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.001)
            continue
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        if q.full():
            with contextlib.suppress(queue.Empty):
                q.get_nowait()
                dropped += 1
        q.put(frame)

    cap.release()
    print(f"[Capture] exited (dropped {dropped})", flush=True)


def output_thread(
    device: str,
    width: int,
    height: int,
    fps: float,
    q: queue.Queue[np.ndarray],
) -> None:
    """Write BGR frames from ``q`` to ``device`` via ffmpeg (v4l2)."""
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
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"[Output] {width}x{height}@{fps:.0f} -> {device}", flush=True)

    frame_interval = 1.0 / fps
    next_time: float | None = None
    written = 0

    while not _shutdown.is_set():
        try:
            frame = q.get(timeout=2.0)
        except queue.Empty:
            if proc.poll() is not None:
                err = proc.stderr.read().decode("utf-8", "replace")[-500:]
                print(f"[Output] ffmpeg died: {err}", flush=True)
                break
            continue
        now = time.monotonic()
        if next_time is None:
            next_time = now
        wait = next_time - now
        if wait > 0:
            time.sleep(wait)
        elif wait < -0.5:
            next_time = now
        try:
            proc.stdin.write(frame.tobytes())
            proc.stdin.flush()
        except BrokenPipeError:
            break
        next_time += frame_interval
        written += 1

    try:
        proc.stdin.close()
        proc.wait(timeout=3)
    except Exception:
        proc.kill()
    print(f"[Output] exited ({written} frames written)", flush=True)


# ---------------------------------------------------------------------------
# Head-pose processor (per-frame, no batching)
# ---------------------------------------------------------------------------


def _load_hp_modules(engine: str) -> tuple[Callable[..., Any], Callable[..., Any]] | None:
    """Return ``(HeadPoseEstimator, Frontalizer)`` classes, or None on failure."""
    try:
        from head_pose_estimator import HeadPoseEstimator
    except ImportError as exc:
        print(
            f"[Pipeline] FATAL: mediapipe / head_pose_estimator not importable: {exc}",
            flush=True,
        )
        return None
    if engine == "liveportrait":
        try:
            from liveportrait_frontalizer import LivePortraitFrontalizer

            return HeadPoseEstimator, LivePortraitFrontalizer
        except ImportError as exc:
            print(
                f"[Pipeline] LivePortrait not importable ({exc}); falling back to geometric engine",
                flush=True,
            )
    try:
        from frontalizer import Frontalizer

        return HeadPoseEstimator, Frontalizer
    except ImportError as exc:
        print(f"[Pipeline] FATAL: no frontalizer available: {exc}", flush=True)
        return None


def process_thread(
    raw_q: queue.Queue[np.ndarray],
    out_q: queue.Queue[np.ndarray],
    *,
    engine: str,
    strength: float,
    yaw_limit: float,
    compile_models: bool,
) -> None:
    """Pull frames from ``raw_q``, apply head-pose correction, push to ``out_q``."""
    mods = _load_hp_modules(engine)
    if mods is None:
        _shutdown.set()
        return
    hpe_cls, fr_cls = mods

    hpe = hpe_cls(static_image_mode=False)  # video-mode tracker
    fr_kwargs: dict[str, Any] = {"strength": strength}
    if engine == "liveportrait":
        fr_kwargs["compile_models"] = compile_models
    frontalizer = fr_cls(**fr_kwargs)
    print(
        f"[Pipeline] engine={engine}  strength={strength}  yaw_limit={yaw_limit}  "
        f"compile={compile_models}",
        flush=True,
    )

    # Rolling latency stats
    stat_n = 0
    stat_sum = 0.0
    stat_max = 0.0
    last_log = time.monotonic()
    passthrough = 0
    corrected_ct = 0

    while not _shutdown.is_set():
        try:
            frame = raw_q.get(timeout=1.0)
        except queue.Empty:
            continue

        t0 = time.perf_counter()
        landmarks = hpe.get_landmarks(frame)
        out = frame
        if landmarks is not None:
            pose = hpe.estimate_from_landmarks(landmarks, frame.shape[:2])
            if pose is not None:
                pitch, yaw, roll = pose
                if abs(yaw) < yaw_limit:
                    try:
                        warped = frontalizer.frontalize(frame, landmarks, pitch, yaw, roll)
                        if warped is not None:
                            x0, y0 = landmarks.min(axis=0).astype(int)
                            x1, y1 = landmarks.max(axis=0).astype(int)
                            face_rect = (x0, y0, x1 - x0, y1 - y0)
                            out = frontalizer.blend_back(frame, warped, face_rect)
                            corrected_ct += 1
                    except Exception as exc:  # noqa: BLE001
                        print(f"[Pipeline] correction error: {exc}", flush=True)
                        passthrough += 1
                else:
                    passthrough += 1  # yaw out of range -> pass-through
            else:
                passthrough += 1
        else:
            passthrough += 1

        dt_ms = (time.perf_counter() - t0) * 1000
        stat_n += 1
        stat_sum += dt_ms
        stat_max = max(stat_max, dt_ms)

        if out_q.full():
            with contextlib.suppress(queue.Empty):
                out_q.get_nowait()
        out_q.put(out)

        now = time.monotonic()
        if now - last_log >= 5.0 and stat_n:
            print(
                f"[Pipeline] last 5s: avg={stat_sum / stat_n:.1f} ms  "
                f"max={stat_max:.1f} ms  corrected={corrected_ct}  "
                f"passthrough={passthrough}",
                flush=True,
            )
            stat_n = 0
            stat_sum = 0.0
            stat_max = 0.0
            corrected_ct = 0
            passthrough = 0
            last_log = now

    hpe.close()
    print("[Pipeline] exited", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_PRESETS = {"480p": (640, 480), "720p": (1280, 720), "1080p": (1920, 1080)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AR SDK → LivePortrait head-pose correction (combined pipeline stage 2)"
    )
    p.add_argument("--input", default=_env_str("INPUT_DEVICE", "/dev/video11"))
    p.add_argument("--output", default=_env_str("OUTPUT_DEVICE", "/dev/video10"))
    p.add_argument(
        "--resolution",
        default=_env_str("RESOLUTION", "1080p"),
        choices=list(_PRESETS),
    )
    p.add_argument("--fps", type=float, default=_env_float("FPS", 30.0))
    p.add_argument(
        "--head-pose-engine",
        default=_env_str("HEAD_POSE_ENGINE", "liveportrait"),
        choices=["geometric", "liveportrait"],
    )
    p.add_argument(
        "--head-pose-strength", type=float, default=_env_float("HEAD_POSE_STRENGTH", 1.0)
    )
    p.add_argument(
        "--head-pose-yaw-limit", type=float, default=_env_float("HEAD_POSE_YAW_LIMIT", 45.0)
    )
    p.add_argument(
        "--head-pose-compile",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("HEAD_POSE_COMPILE"),
    )
    return p


def main() -> None:  # pragma: no cover
    args = build_parser().parse_args()
    width, height = _PRESETS[args.resolution]

    raw_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)
    out_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)

    threads = [
        threading.Thread(
            target=capture_thread,
            args=(args.input, width, height, args.fps, raw_q),
            daemon=True,
        ),
        threading.Thread(
            target=process_thread,
            args=(raw_q, out_q),
            kwargs={
                "engine": args.head_pose_engine,
                "strength": args.head_pose_strength,
                "yaw_limit": args.head_pose_yaw_limit,
                "compile_models": args.head_pose_compile,
            },
            daemon=True,
        ),
        threading.Thread(
            target=output_thread,
            args=(args.output, width, height, args.fps, out_q),
            daemon=True,
        ),
    ]

    print("=" * 60, flush=True)
    print("Maxine combined pipeline — stage 2 (head-pose correction)", flush=True)
    print("=" * 60, flush=True)
    for t in threads:
        t.start()

    try:
        while not _shutdown.is_set():
            if any(not t.is_alive() for t in threads):
                print("[Main] a thread died; shutting down", flush=True)
                _shutdown.set()
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        _shutdown.set()

    for t in threads:
        t.join(timeout=5.0)
    print("[Main] done", flush=True)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
