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
import atexit
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

# Raised by output_thread once the first frame has been written to the
# intermediate v4l2loopback device. main() waits on this before spawning the
# AR SDK container so AR SDK never tries to open a device with no writer.
_intermediate_ready = threading.Event()


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
    """Read frames from ``device`` via ffmpeg-rawvideo subprocess.

    cv2.VideoCapture + v4l2loopback intermittently wedges in futex waits
    on 1080p MJPG streams; routing through ``ffmpeg -f v4l2 ... -f
    rawvideo -`` works reliably and matches how we do the *output* side.
    """
    # AR SDK binary ALWAYS writes raw BGR24 to the v4l2loopback output device
    # regardless of resolution (the `--mjpeg` flag only applies to camera
    # input). We therefore read BGR24 by default. A previous iteration tried
    # MJPEG / YUYV because of misreading the README's "format" column
    # (720p MJPEG / 1080p MJPEG there refers to CAMERA input format, not
    # v4l2loopback output). YUYV 1080p from the camera caps at 2 fps due
    # to USB 2.0 bandwidth; BGR24 at 1080p via MJPEG input sustains 30 fps.
    input_fmt = _env_str("INPUT_FORMAT", "bgr24")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "v4l2",
        "-input_format",
        input_fmt,
        "-video_size",
        f"{width}x{height}",
        "-i",
        device,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    print(
        f"[Capture] spawning ffmpeg -> {device} ({width}x{height} {input_fmt})",
        flush=True,
    )

    def _spawn() -> subprocess.Popen:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # AR SDK (stage 1) may not yet be writing to the v4l2loopback device when
    # this process starts — `Requires=` in systemd only orders by activation
    # of the docker service, not by the inner binary being ready. Retry the
    # ffmpeg spawn on early failure (VIDIOC_STREAMON: Input/output error
    # happens when no writer has connected yet). Cap retries at ~60 seconds.
    proc = _spawn()
    retry_deadline = time.monotonic() + 60.0

    frame_size = width * height * 3  # BGR24
    dropped = 0
    frames_in = 0
    last_stderr_drain = time.monotonic()
    while not _shutdown.is_set():
        if proc.poll() is not None:
            err = proc.stderr.read().decode("utf-8", "replace")[-500:]
            now = time.monotonic()
            if now < retry_deadline and frames_in == 0:
                last = err.strip().splitlines()[-1] if err else "no stderr"
                print(
                    f"[Capture] ffmpeg exited early ({last}) — retrying in 2 s "
                    "(waiting for stage-1 writer)",
                    flush=True,
                )
                time.sleep(2.0)
                proc = _spawn()
                continue
            print(f"[Capture] FATAL: ffmpeg died: {err}", flush=True)
            _shutdown.set()
            break
        try:
            buf = proc.stdout.read(frame_size)
        except Exception as exc:  # noqa: BLE001
            print(f"[Capture] read error: {exc}", flush=True)
            break
        if not buf or len(buf) < frame_size:
            # ffmpeg is not yet ready or stream paused; brief back-off
            time.sleep(0.005)
            continue
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
        if q.full():
            with contextlib.suppress(queue.Empty):
                q.get_nowait()
                dropped += 1
        q.put(frame)
        frames_in += 1
        if frames_in == 1:
            print(f"[Capture] first frame: {frame.shape} device={device}", flush=True)
        # Periodically drain stderr so the pipe doesn't block ffmpeg
        if time.monotonic() - last_stderr_drain > 2.0:
            last_stderr_drain = time.monotonic()

    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        proc.kill()
    print(f"[Capture] exited (in={frames_in} dropped={dropped})", flush=True)


def output_thread(
    device: str,
    width: int,
    height: int,
    fps: float,
    q: queue.Queue[np.ndarray],
) -> None:
    """Write BGR frames from ``q`` to ``device`` via ffmpeg (v4l2).

    The output pixel format is chosen for the downstream reader: YUYV422 is
    what the AR SDK binary expects when the intermediate v4l2loopback device
    is its camera input. Final-output Zoom/Teams/Chrome readers accept YUYV
    too. Override with ``OUTPUT_FORMAT=yuv420p`` etc. if needed.
    """
    out_fmt = _env_str("OUTPUT_FORMAT", "yuyv422")
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
        out_fmt,
        device,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"[Output] {width}x{height}@{fps:.0f} -> {device} ({out_fmt})", flush=True)

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
        if written == 1:
            # Tell main() it's safe to spawn the AR SDK container now that
            # the intermediate v4l2loopback device has a live writer.
            _intermediate_ready.set()

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
    head_pose: bool,
    engine: str,
    strength: float,
    yaw_limit: float,
    compile_models: bool,
    pitch_strength: float,
    yaw_strength: float,
    roll_strength: float,
    pitch_limit: float,
    roll_limit: float,
    source_image: str | None = None,
) -> None:
    """Pull frames from ``raw_q``, apply head-pose correction, push to ``out_q``."""
    if not head_pose:
        print("[Pipeline] head-pose correction DISABLED; passing through", flush=True)
        while not _shutdown.is_set():
            try:
                frame = raw_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if out_q.full():
                with contextlib.suppress(queue.Empty):
                    out_q.get_nowait()
            out_q.put(frame)
        return

    mods = _load_hp_modules(engine)
    if mods is None:
        _shutdown.set()
        return
    hpe_cls, fr_cls = mods

    hpe = hpe_cls(static_image_mode=False)  # video-mode tracker
    fr_kwargs: dict[str, Any] = {"strength": strength}
    if engine == "liveportrait":
        fr_kwargs["compile_models"] = compile_models
        fr_kwargs["pitch_strength"] = pitch_strength
        fr_kwargs["yaw_strength"] = yaw_strength
        fr_kwargs["roll_strength"] = roll_strength
        fr_kwargs["source_image_path"] = source_image
    frontalizer = fr_cls(**fr_kwargs)
    print(
        f"[Pipeline] head-pose ENABLED  engine={engine}  strength={strength}  "
        f"compile={compile_models}  "
        f"pitch(s={pitch_strength}, lim={pitch_limit})  "
        f"yaw(s={yaw_strength}, lim={yaw_limit})  "
        f"roll(s={roll_strength}, lim={roll_limit})",
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
                # Per-axis skip gates. Exceeding any limit passes the frame
                # through uncorrected — LivePortrait produces uncanny warps
                # at extreme angles, so we prefer the raw feed over a glitch.
                within_gates = (
                    abs(yaw) < yaw_limit and abs(pitch) < pitch_limit and abs(roll) < roll_limit
                )
                if within_gates:
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
                    passthrough += 1  # axis out of range -> pass-through
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
        "--head-pose",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("HEAD_POSE", True),
        help="Enable head-pose correction (env: HEAD_POSE, default: True)",
    )
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
    # Per-axis overrides. Each inherits --head-pose-strength when not set;
    # roll typically wants 0 (preserve natural head tilt) since forcing the
    # head level looks uncanny on a webcam feed.
    _default_strength = _env_float("HEAD_POSE_STRENGTH", 1.0)
    p.add_argument(
        "--head-pose-pitch-strength",
        type=float,
        default=_env_float("HEAD_POSE_PITCH_STRENGTH", _default_strength),
    )
    p.add_argument(
        "--head-pose-yaw-strength",
        type=float,
        default=_env_float("HEAD_POSE_YAW_STRENGTH", _default_strength),
    )
    p.add_argument(
        "--head-pose-roll-strength",
        type=float,
        default=_env_float("HEAD_POSE_ROLL_STRENGTH", _default_strength),
    )
    # Per-axis skip limits (degrees). If |angle| exceeds the axis limit the
    # frame is passed through uncorrected — LivePortrait is unreliable at
    # extreme rotations, so we'd rather show the raw feed than a glitch.
    p.add_argument(
        "--head-pose-yaw-limit", type=float, default=_env_float("HEAD_POSE_YAW_LIMIT", 45.0)
    )
    p.add_argument(
        "--head-pose-pitch-limit",
        type=float,
        default=_env_float("HEAD_POSE_PITCH_LIMIT", 45.0),
    )
    p.add_argument(
        "--head-pose-roll-limit",
        type=float,
        default=_env_float("HEAD_POSE_ROLL_LIMIT", 45.0),
    )
    p.add_argument(
        "--head-pose-compile",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("HEAD_POSE_COMPILE"),
    )
    p.add_argument(
        "--head-pose-source",
        default=_env_str("HEAD_POSE_SOURCE_IMAGE", None),
        help="Path to a static source image for head-pose (env: HEAD_POSE_SOURCE_IMAGE)",
    )
    return p


# ---------------------------------------------------------------------------
# AR SDK Docker container lifecycle (Stage 1 runs as a child of this process
# so there's a single systemd unit owning the whole pipeline).
# ---------------------------------------------------------------------------

_GAZE_ENV = [
    "GAZE_EYE_SIZE",
    "GAZE_LANDMARKS",
    "GAZE_NO_REDIRECT",
    "GAZE_NO_STABILIZE",
    "GAZE_NO_CUDA_GRAPH",
    "GAZE_PITCH_LOW",
    "GAZE_PITCH_HIGH",
    "GAZE_YAW_LOW",
    "GAZE_YAW_HIGH",
    "GAZE_HEAD_PITCH_LOW",
    "GAZE_HEAD_PITCH_HIGH",
    "GAZE_HEAD_YAW_LOW",
    "GAZE_HEAD_YAW_HIGH",
]


def _start_arsdk_container(
    *,
    input_device: str,
    output_device: str,
    width: int,
    height: int,
) -> subprocess.Popen | None:  # pragma: no cover
    """Launch the AR SDK eye-contact Docker container as a child process.

    This runs AFTER the LivePortrait head-pose stage in Python, so:
      * ``input_device`` — host v4l2loopback device that receives Python's
        head-pose-corrected frames. Remapped into the container as
        ``/dev/video0`` because the AR SDK binary opens that path literally.
      * ``output_device`` — host v4l2loopback device that the container
        writes the final gaze-corrected stream into. Remapped as
        ``/dev/video10`` to match the binary's default.

    Returns the Popen, or None if START_ARSDK=0 (user is running Stage 2
    out-of-band). All GAZE_* tuning knobs are forwarded into the container.
    """
    if not _env_bool("START_ARSDK", True):
        print("[ARSDK] START_ARSDK=0, skipping container launch", flush=True)
        return None

    name = _env_str("ARSDK_CONTAINER_NAME", "maxine-arsdk")
    image = _env_str(
        "ARSDK_IMAGE",
        "ghcr.io/n00b001/maxine-eye-contact-webcam/arsdk-gaze:latest",
    )

    # Best-effort clean up any stale container from a previous crash
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)

    # Device remap: host INTERMEDIATE → container /dev/video0 (camera input),
    # host FINAL → container /dev/video10 (binary's default output path).
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--gpus",
        "all",
        "--device",
        f"{input_device}:/dev/video0",
        "--device",
        f"{output_device}:/dev/video10",
        "-e",
        "NVIDIA_VISIBLE_DEVICES=all",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
    ]
    for var in _GAZE_ENV:
        if var in os.environ:
            cmd.extend(["-e", f"{var}={os.environ[var]}"])
    # No `--mjpeg`: container's /dev/video0 is a v4l2loopback device fed by
    # Python via ffmpeg in YUYV422, not a USB webcam that needs MJPEG to hit
    # 30 fps. AR SDK negotiates YUYV natively.
    cmd.extend(
        [
            image,
            "/usr/local/bin/maxine_ar_webcam",
            "/dev/video10",
            str(width),
            str(height),
        ]
    )
    print(
        f"[ARSDK] launching container '{name}': "
        f"host {input_device} -> container /dev/video0 (camera), "
        f"container /dev/video10 -> host {output_device} ({width}x{height})",
        flush=True,
    )
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _stop_arsdk_container(proc: subprocess.Popen | None) -> None:  # pragma: no cover
    if proc is None:
        return
    name = _env_str("ARSDK_CONTAINER_NAME", "maxine-arsdk")
    subprocess.run(["docker", "stop", "-t", "5", name], check=False, capture_output=True)
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)
    with contextlib.suppress(Exception):
        proc.terminate()
        proc.wait(timeout=3)
    with contextlib.suppress(Exception):
        proc.kill()
    print("[ARSDK] container stopped", flush=True)


def _pump_arsdk_logs(proc: subprocess.Popen) -> None:  # pragma: no cover
    """Forward AR SDK container stdout/stderr to this process's stdout so it
    shows up in journalctl alongside the head-pose logs."""
    try:
        for line in iter(proc.stdout.readline, b""):
            if _shutdown.is_set():
                break
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                print(f"[ARSDK] {text}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[ARSDK] log pump error: {exc}", flush=True)


def main() -> None:  # pragma: no cover
    args = build_parser().parse_args()
    width, height = _PRESETS[args.resolution]

    # AR SDK runs AFTER LivePortrait so the eye-gaze redirection corrects
    # the eyes in the head-pose-rotated frame (running gaze first would
    # aim the eyes at the camera, and the subsequent head rotation would
    # then send them off-axis).
    final_output = _env_str("FINAL_OUTPUT", "/dev/video10")
    # arsdk_proc is spawned LATER — only after output_thread has written
    # at least one frame to the intermediate device, so AR SDK doesn't
    # open a v4l2loopback with no writer (which makes the binary exit
    # with "Failed to open camera").
    arsdk_proc: subprocess.Popen | None = None
    atexit.register(lambda: _stop_arsdk_container(arsdk_proc))

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
                "head_pose": args.head_pose,
                "engine": args.head_pose_engine,
                "strength": args.head_pose_strength,
                "yaw_limit": args.head_pose_yaw_limit,
                "compile_models": args.head_pose_compile,
                "pitch_strength": args.head_pose_pitch_strength,
                "yaw_strength": args.head_pose_yaw_strength,
                "roll_strength": args.head_pose_roll_strength,
                "pitch_limit": args.head_pose_pitch_limit,
                "roll_limit": args.head_pose_roll_limit,
                "source_image": args.head_pose_source,
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
    print("Maxine combined pipeline — head pose then eye contact", flush=True)
    print("=" * 60, flush=True)
    for t in threads:
        t.start()

    # Wait until the intermediate v4l2loopback device has a live writer.
    # The Python pipeline must be producing frames BEFORE we spawn the AR
    # SDK container, otherwise the binary opens its "camera" (= our
    # intermediate device), gets no data, and exits with "Failed to open
    # camera". ~180 s covers LivePortrait's torch.compile first-frame cost.
    print("[Main] waiting for intermediate writer…", flush=True)
    if not _intermediate_ready.wait(timeout=180.0):
        print(
            "[Main] FATAL: intermediate output never produced a frame; shutting down",
            flush=True,
        )
        _shutdown.set()
    else:
        arsdk_proc = _start_arsdk_container(
            input_device=args.output,
            output_device=final_output,
            width=width,
            height=height,
        )
        if arsdk_proc is not None:
            threads.append(
                threading.Thread(target=_pump_arsdk_logs, args=(arsdk_proc,), daemon=True)
            )
            threads[-1].start()

    try:
        while not _shutdown.is_set():
            if any(not t.is_alive() for t in threads):
                print("[Main] a thread died; shutting down", flush=True)
                _shutdown.set()
                break
            if arsdk_proc is not None and arsdk_proc.poll() is not None:
                print(
                    f"[Main] AR SDK container exited (rc={arsdk_proc.returncode}); shutting down",
                    flush=True,
                )
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
