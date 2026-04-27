"""
maxine_fused_pipeline.py — Unified GPU pipeline entry point.

Pipeline per frame:
  webcam (ffmpeg) -> FLPFrontalizer (head-pose correction) ->
  RGB->BGR tensor conversion (torch, on-stream) ->
  GazeRedirect (AR SDK) ->
  D2H copy -> ffmpeg v4l2 sink

All GPU work (FLP + format conversion + gaze) runs on a single
torch.cuda.Stream. One D2H transfer per frame.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from collections import deque

import numpy as np

# cv2 is imported lazily inside the passthrough path below. Keeping it out
# of module-level imports means the test suite (which runs on CI without
# opencv in dev deps) can still import this module.
# IMPORT-ORDER INVARIANT: torch MUST be imported BEFORE maxine_ar_ext.
# The AR SDK libs and torch each bundle their own CUDA runtime symbols;
# loading torch first registers its runtime so AR SDK's later dlopen reuses
# those symbols instead of racing its own copy — otherwise a `free():
# invalid pointer` abort fires at the next heap op. maxine_ar_ext is
# imported lazily inside main() further down; keep it that way.
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _env(name: str, default: str) -> str:
    """Return an environment variable or default. Wrapped so defaults are
    easy to audit in one place (`grep _env` shows every knob)."""
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on", "y")


def _build_parser() -> argparse.ArgumentParser:
    """
    Every knob reads from an environment variable first, then the CLI.
    This lets the systemd service file expose every tunable as a single
    `Environment=` line without needing to weave custom argv logic.
    """
    p = argparse.ArgumentParser(
        description=(
            "Maxine fused pipeline: webcam -> FLP frontalization -> AR gaze redirect -> v4l2 sink"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------- Devices + resolution + framerate ----------------
    p.add_argument(
        "--input-device",
        default=_env("INPUT_DEVICE", "/dev/video0"),
        help="V4L2 capture device (env: INPUT_DEVICE)",
    )
    p.add_argument(
        "--output-device",
        default=_env("OUTPUT_DEVICE", "/dev/video10"),
        help="V4L2 sink device (env: OUTPUT_DEVICE)",
    )
    p.add_argument(
        "--width",
        type=int,
        default=_env_int("OUTPUT_WIDTH", 1920),
        help="Output frame width in pixels (env: OUTPUT_WIDTH)",
    )
    p.add_argument(
        "--height",
        type=int,
        default=_env_int("OUTPUT_HEIGHT", 1080),
        help="Output frame height in pixels (env: OUTPUT_HEIGHT)",
    )
    # FLP internally resizes the driving frame to ~192x192 for motion
    # extraction. Capturing at 640x480 saves ffmpeg decode + H2D bandwidth.
    p.add_argument(
        "--capture-width",
        type=int,
        default=_env_int("CAPTURE_WIDTH", 640),
        help="Webcam capture width, smaller = lower latency (env: CAPTURE_WIDTH)",
    )
    p.add_argument(
        "--capture-height",
        type=int,
        default=_env_int("CAPTURE_HEIGHT", 480),
        help="Webcam capture height, smaller = lower latency (env: CAPTURE_HEIGHT)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=_env_int("FPS", 30),
        help="Capture and output frame rate (env: FPS)",
    )
    p.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("MIRROR", True),
        help="Horizontally flip output for selfie orientation (env: MIRROR=0|1)",
    )
    p.add_argument(
        "--src-image",
        default=_env("SRC_IMAGE", None),
        required=_env("SRC_IMAGE", None) is None,
        help="Path to neutral source portrait image (env: SRC_IMAGE)",
    )
    p.add_argument(
        "--model-dir",
        default=_env("MODEL_DIR", "/usr/local/ARSDK/lib/models"),
        help="NVIDIA Maxine AR SDK model directory (env: MODEL_DIR)",
    )
    p.add_argument(
        "--cfg",
        default=_env("FLP_CFG", "vendor/FasterLivePortrait/configs/trt_infer.yaml"),
        help="FasterLivePortrait TRT inference config YAML (env: FLP_CFG)",
    )
    p.add_argument(
        "--log-level",
        default=_env("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (env: LOG_LEVEL)",
    )

    # ---------------- Pipeline-stage toggles ----------------
    p.add_argument(
        "--flp",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("FLP_ENABLED", True),
        help="Enable FasterLivePortrait head-pose correction (env: FLP_ENABLED=0|1)",
    )
    p.add_argument(
        "--gaze",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("GAZE_ENABLED", True),
        help="Enable NVIDIA AR SDK gaze redirection (env: GAZE_ENABLED=0|1)",
    )
    p.add_argument(
        "--overlay",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("OVERLAY_ENABLED", True),
        help=(
            "If true, paste the animated face back into the live webcam "
            "stream (overlay). If false, overwrite the video with the "
            "source portrait (overwrite). (env: OVERLAY_ENABLED=0|1)"
        ),
    )

    # ---------------- FLP stability + per-axis strength ----------------
    p.add_argument(
        "--flp-motion-ema",
        type=float,
        default=_env_float("FLP_MOTION_EMA", 0.3),
        metavar="ALPHA",
        help="EMA alpha for motion smoothing; 1.0 disables (env: FLP_MOTION_EMA)",
    )
    p.add_argument(
        "--flp-pitch-strength",
        type=float,
        default=_env_float("FLP_PITCH_STRENGTH", 1.0),
        metavar="S",
        help="0.0 = no pitch correction, 1.0 = full frontalize (env: FLP_PITCH_STRENGTH)",
    )
    p.add_argument(
        "--flp-yaw-strength",
        type=float,
        default=_env_float("FLP_YAW_STRENGTH", 1.0),
        metavar="S",
        help="0.0 = no yaw correction, 1.0 = full frontalize (env: FLP_YAW_STRENGTH)",
    )
    p.add_argument(
        "--flp-roll-strength",
        type=float,
        default=_env_float("FLP_ROLL_STRENGTH", 1.0),
        metavar="S",
        help="0.0 = no roll correction, 1.0 = full frontalize (env: FLP_ROLL_STRENGTH)",
    )

    # ---------------- AR SDK knobs ----------------
    p.add_argument(
        "--num-landmarks",
        type=int,
        default=_env_int("GAZE_LANDMARKS", 126),
        choices=[68, 126],
        help="Landmark model size (env: GAZE_LANDMARKS)",
    )
    p.add_argument(
        "--no-cuda-graph",
        action="store_true",
        default=not _env_bool("GAZE_CUDA_GRAPH", True),
        help="Disable AR SDK CUDA graph optimisation (env: GAZE_CUDA_GRAPH=0 disables)",
    )
    p.add_argument(
        "--no-stabilize",
        action="store_true",
        default=not _env_bool("GAZE_STABILIZE", True),
        help="Disable AR SDK temporal stabilisation (env: GAZE_STABILIZE=0 disables)",
    )
    p.add_argument(
        "--eye-size-sensitivity",
        type=int,
        default=_env_int("GAZE_EYE_SIZE", 3),
        metavar="N",
        help="Eye-size sensitivity, 0-7 (env: GAZE_EYE_SIZE)",
    )
    p.add_argument(
        "--gaze-pitch-low",
        type=float,
        default=_env_float("GAZE_PITCH_LOW", 20.0),
        metavar="DEG",
        help="Gaze pitch deadzone (env: GAZE_PITCH_LOW)",
    )
    p.add_argument(
        "--gaze-yaw-low",
        type=float,
        default=_env_float("GAZE_YAW_LOW", 20.0),
        metavar="DEG",
        help="Gaze yaw deadzone (env: GAZE_YAW_LOW)",
    )
    p.add_argument(
        "--head-pitch-low",
        type=float,
        default=_env_float("GAZE_HEAD_PITCH_LOW", 15.0),
        metavar="DEG",
        help="Head pitch deadzone (env: GAZE_HEAD_PITCH_LOW)",
    )
    p.add_argument(
        "--head-yaw-low",
        type=float,
        default=_env_float("GAZE_HEAD_YAW_LOW", 25.0),
        metavar="DEG",
        help="Head yaw deadzone (env: GAZE_HEAD_YAW_LOW)",
    )
    p.add_argument(
        "--gaze-pitch-high",
        type=float,
        default=_env_float("GAZE_PITCH_HIGH", 30.0),
        metavar="DEG",
        help="Gaze pitch saturation (env: GAZE_PITCH_HIGH)",
    )
    p.add_argument(
        "--gaze-yaw-high",
        type=float,
        default=_env_float("GAZE_YAW_HIGH", 30.0),
        metavar="DEG",
        help="Gaze yaw saturation (env: GAZE_YAW_HIGH)",
    )
    p.add_argument(
        "--head-pitch-high",
        type=float,
        default=_env_float("GAZE_HEAD_PITCH_HIGH", 25.0),
        metavar="DEG",
        help="Head pitch saturation (env: GAZE_HEAD_PITCH_HIGH)",
    )
    p.add_argument(
        "--head-yaw-high",
        type=float,
        default=_env_float("GAZE_HEAD_YAW_HIGH", 35.0),
        metavar="DEG",
        help="Head yaw saturation (env: GAZE_HEAD_YAW_HIGH)",
    )

    return p


# ---------------------------------------------------------------------------
# ffmpeg subprocess helpers
# ---------------------------------------------------------------------------


def _open_capture(
    device: str, fps: int, width: int, height: int, frame_bytes: int
) -> subprocess.Popen:
    # Low-latency flags: `-fflags nobuffer` disables ffmpeg's input packet
    # buffer (~100-200ms by default), `-flags low_delay` trims decoder
    # reorder buffer, `-thread_queue_size 8` keeps the packet queue tiny
    # so stale frames don't accumulate upstream of us.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-thread_queue_size",
        "8",
        "-f",
        "v4l2",
        # MJPEG at low res: most UVC webcams support 640x480@30 via MJPG
        # with minimal encode delay. YUYV is uncompressed and can pin
        # latency on USB bandwidth at higher resolutions.
        "-input_format",
        "mjpeg",
        "-framerate",
        str(fps),
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
    # bufsize=0 on Popen gives an unbuffered pipe so readinto() returns
    # each frame as soon as ffmpeg writes it (no Python-side buffering
    # between ffmpeg and our main loop).
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        bufsize=0,
    )


def _open_sink(
    device: str, fps: int, width: int, height: int, frame_bytes: int
) -> subprocess.Popen:
    # Output side low-latency flags mirror the capture side. `-vsync
    # passthrough` forwards frames as fast as we write them (no CFR
    # rate regulation), `-max_delay 0` caps the output packet delay,
    # and `-bufsize 0` kills internal rate-control buffering.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-max_delay",
        "0",
        "-thread_queue_size",
        "8",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-vsync",
        "passthrough",
        "-f",
        "v4l2",
        "-pix_fmt",
        "bgr24",
        device,
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        bufsize=0,
    )


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def _install_signal_handlers(capture: subprocess.Popen, sink: subprocess.Popen) -> None:
    """Install SIGTERM/SIGINT handlers that drain the sink and propagate."""

    def _handler(signum: int, _frame) -> None:
        logger.info("Signal %d received — shutting down", signum)
        # Close sink stdin to flush any buffered frames.
        try:
            if sink.stdin:
                sink.stdin.close()
        except OSError:
            pass
        # Give both children up to 5 s to exit.
        deadline = time.monotonic() + 5.0
        for proc in (sink, capture):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()
        # Propagate via default action.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


class _StageTimer:
    """Lightweight wrapper around pairs of CUDA events for one stage."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    def record_start(self) -> None:
        self._start.record()

    def record_end(self) -> None:
        self._end.record()

    def elapsed_ms(self) -> float:
        """Synchronise and return elapsed milliseconds. Only call in the breakdown path."""
        self._end.synchronize()
        return self._start.elapsed_time(self._end)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:  # noqa: C901 (complexity is pipeline logic)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    width: int = args.width  # output (sink) width
    height: int = args.height  # output (sink) height
    capture_width: int = args.capture_width
    capture_height: int = args.capture_height
    fps: int = args.fps
    capture_bytes: int = capture_width * capture_height * 3
    output_bytes: int = width * height * 3

    # ------------------------------------------------------------------
    # Create the shared CUDA stream. All GPU work lives on this stream.
    # ------------------------------------------------------------------
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        # ------------------------------------------------------------------
        # Init FLPFrontalizer inside the stream context so that
        # FasterLivePortrait's internal CUDA kernels are captured on the
        # same stream from the very first call.
        # ------------------------------------------------------------------
        frontalizer = None
        if args.flp:
            logger.info("Initialising FLPFrontalizer …")
            try:
                from flp_gpu_adapter import FLPFrontalizer

                frontalizer = FLPFrontalizer(
                    cfg_path=args.cfg,
                    src_image_path=args.src_image,
                    device="cuda:0",
                    motion_ema_alpha=args.flp_motion_ema,
                    axis_strength=(
                        args.flp_pitch_strength,
                        args.flp_yaw_strength,
                        args.flp_roll_strength,
                    ),
                )
            except Exception as exc:
                logger.error("FLPFrontalizer init failed: %s", exc)
                sys.exit(1)
        else:
            logger.info("FLP disabled (--no-flp / FLP_ENABLED=0) — head-pose correction off")

        # ------------------------------------------------------------------
        # Init GazeRedirect inside the same stream context. The SDK receives
        # stream.cuda_stream so its internal CUDA ops are on the same stream.
        # ------------------------------------------------------------------
        gaze = None
        if args.gaze:
            logger.info("Initialising GazeRedirect …")
            try:
                import maxine_ar_ext

                gaze = maxine_ar_ext.GazeRedirect(
                    width=width,
                    height=height,
                    model_dir=args.model_dir,
                    cuda_stream_ptr=stream.cuda_stream,
                    num_landmarks=args.num_landmarks,
                    use_cuda_graph=not args.no_cuda_graph,
                    stabilize=not args.no_stabilize,
                    gaze_redirect=True,
                    eye_size_sensitivity=args.eye_size_sensitivity,
                    gaze_pitch_threshold_low=args.gaze_pitch_low,
                    gaze_yaw_threshold_low=args.gaze_yaw_low,
                    head_pitch_threshold_low=args.head_pitch_low,
                    head_yaw_threshold_low=args.head_yaw_low,
                    gaze_pitch_threshold_high=args.gaze_pitch_high,
                    gaze_yaw_threshold_high=args.gaze_yaw_high,
                    head_pitch_threshold_high=args.head_pitch_high,
                    head_yaw_threshold_high=args.head_yaw_high,
                )
            except Exception as exc:
                logger.error("GazeRedirect init failed: %s", exc)
                sys.exit(1)
        else:
            logger.info("Gaze disabled (--no-gaze / GAZE_ENABLED=0) — eye redirect off")

        # Pre-allocate output BGR tensor — reused every frame (no per-frame malloc).
        out_bgr = torch.empty((height, width, 3), dtype=torch.uint8, device="cuda")

    # ------------------------------------------------------------------
    # Open ffmpeg subprocesses.
    # ------------------------------------------------------------------
    logger.info(
        "Opening capture on %s (%dx%d @ %d fps)",
        args.input_device,
        capture_width,
        capture_height,
        fps,
    )
    capture = _open_capture(args.input_device, fps, capture_width, capture_height, capture_bytes)

    logger.info("Opening sink on %s (%dx%d)", args.output_device, width, height)
    sink = _open_sink(args.output_device, fps, width, height, output_bytes)

    _install_signal_handlers(capture, sink)

    # ------------------------------------------------------------------
    # Per-frame read buffer (reused, no allocation in the hot loop).
    # Sized for the CAPTURE dimensions, which may be smaller than the
    # output dimensions for latency reasons.
    # ------------------------------------------------------------------
    read_buf = bytearray(capture_bytes)
    # Scratch for passthrough upscaling (cv2.resize into a pre-allocated
    # numpy buffer is faster than allocating each frame).
    passthrough_buf = np.empty((height, width, 3), dtype=np.uint8)

    # Timing state.
    fps_window: deque[float] = deque(maxlen=30)
    t_prev = time.monotonic()

    # Stage timers — allocated once, only used every 300 frames.
    t_frontalize = _StageTimer("frontalize")
    t_flip_cast = _StageTimer("flip+clamp+cast")
    t_gaze = _StageTimer("gaze")

    frame_idx: int = 0

    logger.info("Pipeline running. Press Ctrl-C to stop.")

    while True:
        # ----------------------------------------------------------------
        # Step 1: Read one raw BGR24 frame from capture ffmpeg.
        #
        # bufsize=0 + `-fflags nobuffer` makes readinto() return whatever
        # is immediately available, which is usually less than a full
        # frame on the first read. Loop into a memoryview slice until we
        # have capture_bytes total — or EOF, in which case we stop cleanly.
        # ----------------------------------------------------------------
        mv = memoryview(read_buf)
        total = 0
        eof = False
        while total < capture_bytes:
            n = capture.stdout.readinto(mv[total:])
            if n == 0:
                eof = True
                break
            total += n
        if eof:
            logger.warning(
                "ffmpeg capture EOF after %d of %d bytes — stopping",
                total,
                capture_bytes,
            )
            break

        # Zero-copy numpy view of the read buffer at CAPTURE resolution.
        frame_bgr = np.frombuffer(read_buf, dtype=np.uint8).reshape(
            capture_height, capture_width, 3
        )

        t_now = time.monotonic()
        fps_window.append(t_now - t_prev)
        t_prev = t_now

        do_breakdown = (frame_idx > 0) and (frame_idx % 300 == 0)

        with torch.cuda.stream(stream):
            # ------------------------------------------------------------
            # Step 2: FLP frontalization.
            # ------------------------------------------------------------
            if do_breakdown:
                t_frontalize.record_start()

            flt = None
            if frontalizer is not None:
                flt = frontalizer.frontalize(frame_bgr, overlay=args.overlay)

            if do_breakdown:
                t_frontalize.record_end()

            in_bgr = None
            if flt is None:
                # No face detected OR FLP disabled: we still want Gaze
                # Redirection to work on the raw webcam feed if enabled.
                if gaze is not None:
                    # Upload raw frame to GPU. permute(2,0,1) is HWC->CHW.
                    # interpolate expects (B, C, H, W) float.
                    t_raw = torch.from_numpy(frame_bgr).to(device="cuda", non_blocking=True)
                    in_bgr = (
                        torch.nn.functional.interpolate(
                            t_raw.permute(2, 0, 1).unsqueeze(0).float(),
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .permute(1, 2, 0)
                        .to(torch.uint8)
                        .contiguous()
                    )
            else:
                # --------------------------------------------------------
                # FLP paste-back follows the SOURCE PORTRAIT's native size
                # (after internal resize_to_limit), not the webcam frame
                # size. When --src-image is 740x576 but --width/--height
                # are 1920x1080, flt.shape = (576, 740, 3) and the SDK,
                # which was allocated at 1920x1080, rejects the pointer
                # with "AR SDK Run: invalid argument". Upscale (bilinear,
                # on-stream) so the gaze input matches the allocated size.
                # --------------------------------------------------------
                if flt.shape[0] != height or flt.shape[1] != width:
                    # (H, W, 3) -> (1, 3, H, W) for interpolate, then back.
                    flt_nchw = flt.permute(2, 0, 1).unsqueeze(0)
                    flt_nchw = torch.nn.functional.interpolate(
                        flt_nchw,
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    flt = flt_nchw.squeeze(0).permute(1, 2, 0)

                # --------------------------------------------------------
                # Step 3: RGB float32 HWC -> BGR uint8 HWC, on-stream.
                #
                # flt shape: (H, W, 3), float32, RGB, values in [0, 255].
                # .flip(-1)   — channel axis: RGB -> BGR
                # .clamp(0, 255) — guard paste-back overshoot
                # .to(uint8)  — cast in one op
                # .contiguous() — ensure row-major interleaved for SDK
                #
                # We use inline torch ops rather than gpu_format_convert
                # because gpu_format_convert.rgb_f32_planar_to_bgr_u8_chunky
                # expects (1, 3, H, W) planar float32 scaled to [0, 1],
                # which is NOT what FLP gives us. A reshape + scale just to
                # call that helper would cost more than doing it inline.
                # --------------------------------------------------------
                if do_breakdown:
                    t_flip_cast.record_start()

                in_bgr = flt.flip(-1).clamp(0, 255).to(torch.uint8).contiguous()

                if do_breakdown:
                    t_flip_cast.record_end()

            # ------------------------------------------------------------
            # Step 4-5: Gaze redirect (runs if we have a GPU tensor).
            # ------------------------------------------------------------
            if in_bgr is not None:
                if do_breakdown:
                    t_gaze.record_start()

                if gaze is not None:
                    ok = gaze.run(
                        int(in_bgr.data_ptr()),
                        width * 3,
                        int(out_bgr.data_ptr()),
                        width * 3,
                    )
                else:
                    # Gaze disabled — use FLP/raw input directly.
                    ok = False

                if do_breakdown:
                    t_gaze.record_end()

                # Choose winner: gaze output if SDK accepted the frame,
                # otherwise the head-pose-corrected frame (or raw webcam).
                chosen = out_bgr if ok else in_bgr

                # If FLP was disabled/failed but Gaze succeeded, 'chosen'
                # is out_bgr (redirected raw webcam). If Gaze also failed,
                # we'll still end up here with 'chosen' as in_bgr (raw webcam).

                # --------------------------------------------------------
                # Optional horizontal mirror for selfie orientation.
                # Applied LAST (after gaze) so AR SDK works in webcam-native
                # coordinates — mirroring before gaze would make the SDK's
                # landmarks point at the wrong side of the face.
                # --------------------------------------------------------
                if args.mirror:
                    chosen = chosen.flip(1).contiguous()

                # --------------------------------------------------------
                # Step 6: D2H — single transfer per frame.
                # .cpu() implicitly synchronises the stream for this tensor.
                # --------------------------------------------------------
                cpu_bytes = chosen.cpu().numpy().tobytes()

                try:
                    sink.stdin.write(cpu_bytes)
                except BrokenPipeError:
                    logger.warning("Sink stdin broken — stopping")
                    break

                # FPS log every 30 frames.
                if len(fps_window) == 30:
                    avg_dt = sum(fps_window) / len(fps_window)
                    if frame_idx % 30 == 0 and avg_dt > 0:
                        logger.info("%.1f fps", 1.0 / avg_dt)

                # Stage breakdown every 300 frames (sync cost accepted here).
                if do_breakdown:
                    ms_front = t_frontalize.elapsed_ms()
                    ms_flip = t_flip_cast.elapsed_ms() if flt is not None else 0.0
                    ms_gz = t_gaze.elapsed_ms()
                    # Log gaze diagnostics so we can tell whether the SDK
                    # actually redirected eyes (face_confidence > threshold
                    # AND gaze_vector != (0,0)) or just passed the frame
                    # through unchanged. If you see gaze_vec=(0.0, 0.0)
                    # every time, the SDK rejected the face and gaze is
                    # NOT happening — check face_confidence.
                    try:
                        gv = gaze.gaze_vector if gaze is not None else (0.0, 0.0)
                        fc = gaze.face_confidence if gaze is not None else 0.0
                    except Exception:
                        gv, fc = (0.0, 0.0), 0.0
                    logger.info(
                        "Stage breakdown [frame %d]: "
                        "frontalize=%.2f ms  flip+clamp+cast=%.2f ms  "
                        "gaze=%.2f ms  ok=%s  face_conf=%.2f  gaze_vec=(%.2f, %.2f)",
                        frame_idx,
                        ms_front,
                        ms_flip,
                        ms_gz,
                        ok,
                        fc,
                        gv[0],
                        gv[1],
                    )

                frame_idx += 1
                continue

        # Passthrough path: no face detected AND Gaze failed/disabled,
        # OR both FLP and Gaze explicitly disabled. No GPU round-trip
        # needed for this frame.
        # Resize to output dims on CPU (cv2.resize is very fast for this)
        # and optionally mirror. Sink always writes at (height, width).
        import cv2  # lazy: only imported when we actually run the pipeline

        if capture_width != width or capture_height != height:
            cv2.resize(frame_bgr, (width, height), dst=passthrough_buf)
            pt = passthrough_buf
        else:
            pt = frame_bgr
        if args.mirror:
            pt = cv2.flip(pt, 1)
        try:
            sink.stdin.write(pt.tobytes())
        except BrokenPipeError:
            logger.warning("Sink stdin broken — stopping")
            break

        # FPS log for passthrough frames too.
        if len(fps_window) == 30 and frame_idx % 30 == 0:
            avg_dt = sum(fps_window) / len(fps_window)
            if avg_dt > 0:
                logger.info("%.1f fps (passthrough)", 1.0 / avg_dt)

        frame_idx += 1

    # ------------------------------------------------------------------
    # Shutdown: drain sink, wait for children.
    # ------------------------------------------------------------------
    logger.info("Main loop exited — draining sink")
    try:
        if sink.stdin:
            sink.stdin.close()
    except OSError:
        pass

    deadline = time.monotonic() + 5.0
    for proc in (sink, capture):
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
