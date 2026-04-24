#!/usr/bin/env python3
"""
measure_latency.py — Run the unified pipeline on a fixed image and report
per-stage latency and total throughput. Bypasses the webcam so we get
deterministic numbers for a known face-containing frame.

Usage (inside the container):
    python3 /app/scripts/measure_latency.py --frames 300

Output: table with p50, p95, p99 latencies for FLP frontalization, format
conversion, AR SDK gaze redirect, D2H copy, and end-to-end.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2
import numpy as np

# Import torch FIRST (see the invariant comment in maxine_fused_pipeline.py)
import torch

sys.path.insert(0, "/app")

import maxine_ar_ext  # noqa: E402

from flp_gpu_adapter import FLPFrontalizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("bench")


def percentile(xs, p):
    return float(np.percentile(xs, p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portrait", default="/app/src-portrait.jpg")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument(
        "--cfg",
        default="/app/vendor/FasterLivePortrait/configs/trt_infer.yaml",
    )
    ap.add_argument("--model-dir", default="/usr/local/ARSDK/lib/models")
    args = ap.parse_args()

    # FLP rescales images internally via resize_to_limit — the output size
    # may not match what we passed in. Run one frame first to learn the
    # actual paste-back shape, then (re)allocate AR SDK + buffers to match.
    raw = cv2.imread(args.portrait)
    if raw is None:
        log.error("cannot read %s", args.portrait)
        return 1
    portrait = raw

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        log.info("Initializing FLPFrontalizer …")
        frontalizer = FLPFrontalizer(cfg_path=args.cfg, src_image_path=args.portrait)

        # Probe: discover FLP's actual paste-back output shape.
        probe = frontalizer.frontalize(portrait)
        if probe is None:
            log.error("FLP found no face in probe frame — use a clearer portrait")
            return 1
        H, W = int(probe.shape[0]), int(probe.shape[1])  # noqa: N806
        log.info("FLP paste-back shape: (%d, %d, 3)", H, W)

        log.info("Initializing GazeRedirect at %dx%d …", W, H)
        gaze = maxine_ar_ext.GazeRedirect(
            width=W,
            height=H,
            model_dir=args.model_dir,
            cuda_stream_ptr=stream.cuda_stream,
        )
        out_bgr = torch.empty((H, W, 3), dtype=torch.uint8, device=device)

    # Warmup
    log.info("Warmup: %d frames", args.warmup)
    for _ in range(args.warmup):
        with torch.cuda.stream(stream):
            flt = frontalizer.frontalize(portrait)
            if flt is not None:
                in_bgr = flt.flip(-1).clamp(0, 255).to(torch.uint8).contiguous()
                gaze.run(
                    int(in_bgr.data_ptr()),
                    W * 3,
                    int(out_bgr.data_ptr()),
                    W * 3,
                )
                _ = out_bgr.cpu().numpy()
        torch.cuda.synchronize()

    # Timed loop
    t_front = []
    t_flip = []
    t_gaze = []
    t_d2h = []
    t_total = []
    no_face = 0

    for i in range(args.frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.cuda.stream(stream):
            flt = frontalizer.frontalize(portrait)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            if flt is None:
                no_face += 1
                continue

            in_bgr = flt.flip(-1).clamp(0, 255).to(torch.uint8).contiguous()
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            gaze.run(
                int(in_bgr.data_ptr()),
                W * 3,
                int(out_bgr.data_ptr()),
                W * 3,
            )
            torch.cuda.synchronize()
            t3 = time.perf_counter()

            _ = out_bgr.cpu().numpy()
            t4 = time.perf_counter()

        t_front.append((t1 - t0) * 1000.0)
        t_flip.append((t2 - t1) * 1000.0)
        t_gaze.append((t3 - t2) * 1000.0)
        t_d2h.append((t4 - t3) * 1000.0)
        t_total.append((t4 - t0) * 1000.0)

        if (i + 1) % 30 == 0:
            log.info(
                "  %3d/%d  front=%.1f  gaze=%.1f  total=%.1f (no_face=%d)",
                i + 1,
                args.frames,
                t_front[-1],
                t_gaze[-1],
                t_total[-1],
                no_face,
            )

    if not t_total:
        log.error("No successful frames — FLP found no face in %s", args.portrait)
        return 2

    n = len(t_total)
    fps = 1000.0 / (sum(t_total) / n)

    log.info("")
    log.info("=" * 62)
    log.info(
        "Measured %d frames (no-face skipped: %d), resolution %dx%d",
        n,
        no_face,
        W,
        H,
    )
    log.info("=" * 62)
    log.info("%-22s  %8s  %8s  %8s", "stage", "p50 ms", "p95 ms", "p99 ms")
    for name, xs in [
        ("FLP frontalize", t_front),
        ("RGB→BGR + cast", t_flip),
        ("AR SDK gaze", t_gaze),
        ("D2H + numpy", t_d2h),
        ("TOTAL per-frame", t_total),
    ]:
        log.info(
            "%-22s  %8.2f  %8.2f  %8.2f",
            name,
            percentile(xs, 50),
            percentile(xs, 95),
            percentile(xs, 99),
        )
    log.info("=" * 62)
    log.info("Throughput: %.1f fps   (target: >=30 fps, p99 total <100ms)", fps)
    ok_fps = fps >= 30.0
    ok_lat = percentile(t_total, 99) < 100.0
    log.info("FPS target:     %s", "PASS" if ok_fps else "FAIL")
    log.info("Latency target: %s", "PASS" if ok_lat else "FAIL")
    return 0 if (ok_fps and ok_lat) else 3


if __name__ == "__main__":
    sys.exit(main())
