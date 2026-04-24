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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Maxine fused pipeline: webcam -> FLP frontalization -> AR gaze redirect -> v4l2 sink"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-device", default="/dev/video0", help="V4L2 capture device")
    p.add_argument("--output-device", default="/dev/video10", help="V4L2 sink device")
    p.add_argument("--width", type=int, default=1920, help="Frame width in pixels")
    p.add_argument("--height", type=int, default=1080, help="Frame height in pixels")
    p.add_argument("--fps", type=int, default=30, help="Capture and output frame rate")
    p.add_argument("--src-image", required=True, help="Path to neutral source portrait image")
    p.add_argument(
        "--model-dir",
        default="/usr/local/ARSDK/lib/models",
        help="NVIDIA Maxine AR SDK model directory",
    )
    p.add_argument(
        "--cfg",
        default="vendor/FasterLivePortrait/configs/trt_infer.yaml",
        help="FasterLivePortrait TRT inference config YAML",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )

    # AR SDK optional knobs — defaults mirror gazeEngine.cpp constants.
    p.add_argument(
        "--num-landmarks",
        type=int,
        default=126,
        choices=[68, 126],
        help="Landmark model size (68 or 126)",
    )
    p.add_argument("--no-cuda-graph", action="store_true", help="Disable CUDA graph optimisation")
    p.add_argument("--no-stabilize", action="store_true", help="Disable temporal stabilisation")
    p.add_argument(
        "--eye-size-sensitivity",
        type=int,
        default=3,
        metavar="N",
        help="Eye-size sensitivity (0-7)",
    )
    p.add_argument("--gaze-pitch-low", type=float, default=20.0, metavar="DEG")
    p.add_argument("--gaze-yaw-low", type=float, default=20.0, metavar="DEG")
    p.add_argument("--head-pitch-low", type=float, default=15.0, metavar="DEG")
    p.add_argument("--head-yaw-low", type=float, default=25.0, metavar="DEG")
    p.add_argument("--gaze-pitch-high", type=float, default=30.0, metavar="DEG")
    p.add_argument("--gaze-yaw-high", type=float, default=30.0, metavar="DEG")
    p.add_argument("--head-pitch-high", type=float, default=25.0, metavar="DEG")
    p.add_argument("--head-yaw-high", type=float, default=35.0, metavar="DEG")

    return p


# ---------------------------------------------------------------------------
# ffmpeg subprocess helpers
# ---------------------------------------------------------------------------


def _open_capture(
    device: str, fps: int, width: int, height: int, frame_bytes: int
) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "v4l2",
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
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        bufsize=frame_bytes * 4,
    )


def _open_sink(
    device: str, fps: int, width: int, height: int, frame_bytes: int
) -> subprocess.Popen:
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
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-f",
        "v4l2",
        "-pix_fmt",
        "bgr24",
        device,
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        bufsize=frame_bytes * 4,
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

    width: int = args.width
    height: int = args.height
    fps: int = args.fps
    frame_bytes: int = width * height * 3  # BGR24 bytes per frame

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
        logger.info("Initialising FLPFrontalizer …")
        try:
            from flp_gpu_adapter import FLPFrontalizer

            frontalizer = FLPFrontalizer(
                cfg_path=args.cfg,
                src_image_path=args.src_image,
                device="cuda:0",
            )
        except Exception as exc:
            logger.error("FLPFrontalizer init failed: %s", exc)
            sys.exit(1)

        # ------------------------------------------------------------------
        # Init GazeRedirect inside the same stream context. The SDK receives
        # stream.cuda_stream so its internal CUDA ops are on the same stream.
        # ------------------------------------------------------------------
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

        # Pre-allocate output BGR tensor — reused every frame (no per-frame malloc).
        out_bgr = torch.empty((height, width, 3), dtype=torch.uint8, device="cuda")

    # ------------------------------------------------------------------
    # Open ffmpeg subprocesses.
    # ------------------------------------------------------------------
    logger.info("Opening capture on %s (%dx%d @ %d fps)", args.input_device, width, height, fps)
    capture = _open_capture(args.input_device, fps, width, height, frame_bytes)

    logger.info("Opening sink on %s", args.output_device)
    sink = _open_sink(args.output_device, fps, width, height, frame_bytes)

    _install_signal_handlers(capture, sink)

    # ------------------------------------------------------------------
    # Per-frame read buffer (reused, no allocation in the hot loop).
    # ------------------------------------------------------------------
    read_buf = bytearray(frame_bytes)

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
        # ----------------------------------------------------------------
        n = capture.stdout.readinto(read_buf)
        if n != frame_bytes:
            logger.warning(
                "ffmpeg capture short-read: got %d of %d bytes — stopping", n, frame_bytes
            )
            break

        # Zero-copy numpy view of the read buffer.
        frame_bgr = np.frombuffer(read_buf, dtype=np.uint8).reshape(height, width, 3)

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

            flt = frontalizer.frontalize(frame_bgr)

            if do_breakdown:
                t_frontalize.record_end()

            if flt is None:
                # No face detected — passthrough original frame bytes.
                # We write the numpy bytes directly (no GPU round-trip needed).
                pass
            else:
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

                # --------------------------------------------------------
                # Steps 4-5: Gaze redirect.
                # --------------------------------------------------------
                if do_breakdown:
                    t_gaze.record_start()

                ok = gaze.run(
                    int(in_bgr.data_ptr()),
                    width * 3,
                    int(out_bgr.data_ptr()),
                    width * 3,
                )

                if do_breakdown:
                    t_gaze.record_end()

                # Choose winner: gaze output if SDK accepted the frame,
                # otherwise the head-pose-corrected frame.
                chosen = out_bgr if ok else in_bgr

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
                    ms_flip = t_flip_cast.elapsed_ms()
                    ms_gz = t_gaze.elapsed_ms()
                    logger.info(
                        "Stage breakdown [frame %d]: "
                        "frontalize=%.2f ms  flip+clamp+cast=%.2f ms  "
                        "gaze=%.2f ms",
                        frame_idx,
                        ms_front,
                        ms_flip,
                        ms_gz,
                    )

                frame_idx += 1
                continue

        # Passthrough path: write the original numpy frame (no GPU upload).
        try:
            sink.stdin.write(bytes(read_buf))
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
