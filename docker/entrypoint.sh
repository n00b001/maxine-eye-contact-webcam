#!/usr/bin/env bash
# Entrypoint for the unified Python pipeline container.
#
# Responsibilities:
#   1. Confirm TRT engines exist under /opt/flp-checkpoints/.
#      If any are missing, build them from the ONNX files (which must
#      already be present in /opt/flp-checkpoints/liveportrait_onnx/).
#      TRT engine build is GPU-dependent and can take several minutes;
#      it is intentionally NOT baked into the image.
#   2. exec "$@" so the CMD (or any docker run override) receives signals.
#
# Volume assumptions (from docker-compose.yml):
#   /opt/flp-checkpoints  — host-persistent dir with ONNX + TRT engines
#   /app/src-portrait.jpg — user-supplied neutral portrait (read-only mount)

set -euo pipefail

CHECKPOINTS_DIR="${FLP_CHECKPOINTS_DIR:-/opt/flp-checkpoints}"
ONNX_DIR="$CHECKPOINTS_DIR/liveportrait_onnx"
SCRIPTS_DIR="/app/vendor/FasterLivePortrait/scripts"
SENTINEL="$CHECKPOINTS_DIR/.engines-built"

# ---------------------------------------------------------------------------
# TRT engine list — must match scripts/all_onnx2trt.sh
# ---------------------------------------------------------------------------
declare -A ONNX_TO_PRECISION=(
    ["warping_spade-fix.onnx"]="fp16"
    ["landmark.onnx"]="fp16"
    ["motion_extractor.onnx"]="fp32"
    ["retinaface_det_static.onnx"]="fp16"
    ["face_2dpose_106_static.onnx"]="fp16"
    ["appearance_feature_extractor.onnx"]="fp16"
    ["stitching.onnx"]="fp16"
    ["stitching_eye.onnx"]="fp16"
    ["stitching_lip.onnx"]="fp16"
)

TOTAL_ENGINES="${#ONNX_TO_PRECISION[@]}"

# ---------------------------------------------------------------------------
# Preflight summary (shared between --dry-run and normal start)
# ---------------------------------------------------------------------------
_preflight_summary() {
    echo "[entrypoint] === Preflight Summary ==="
    echo "[entrypoint] Checkpoints dir : $CHECKPOINTS_DIR"
    echo "[entrypoint] ONNX dir        : $ONNX_DIR"
    echo "[entrypoint] Engines expected: $TOTAL_ENGINES"

    local present=0 missing_onnx=() missing_trt=()
    for onnx_name in "${!ONNX_TO_PRECISION[@]}"; do
        trt_name="${onnx_name%.onnx}.trt"
        [ -f "$ONNX_DIR/$onnx_name" ] || missing_onnx+=("$onnx_name")
        if [ -f "$ONNX_DIR/$trt_name" ]; then
            (( present++ )) || true
        else
            missing_trt+=("$trt_name")
        fi
    done

    echo "[entrypoint] TRT engines ready: $present / $TOTAL_ENGINES"
    if [ "${#missing_trt[@]}" -gt 0 ]; then
        echo "[entrypoint] Missing TRT     : ${missing_trt[*]}"
    fi
    if [ "${#missing_onnx[@]}" -gt 0 ]; then
        echo "[entrypoint] Missing ONNX    : ${missing_onnx[*]}"
    fi

    # Device mounts
    if [ -e /dev/video0 ]; then
        echo "[entrypoint] Input device    : /dev/video0 present"
    else
        echo "[entrypoint] WARNING         : /dev/video0 not found — no webcam input" >&2
    fi
    if [ -e /dev/video10 ]; then
        echo "[entrypoint] Output device   : /dev/video10 present"
    else
        echo "[entrypoint] WARNING         : /dev/video10 not found — v4l2loopback not mounted" >&2
    fi

    # Portrait
    if [ -f /app/src-portrait.jpg ]; then
        echo "[entrypoint] Portrait        : /app/src-portrait.jpg present"
    else
        echo "[entrypoint] WARNING         : /app/src-portrait.jpg not found" >&2
        echo "[entrypoint]   Mount with: -v /opt/maxine-portrait.jpg:/app/src-portrait.jpg:ro" >&2
    fi
    echo "[entrypoint] ========================"
}

# ---------------------------------------------------------------------------
# --dry-run: print summary and exit without side effects
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--dry-run" ]; then
    _preflight_summary
    exit 0
fi

# ---------------------------------------------------------------------------
# Device mount warnings (non-fatal — docker-compose.yml controls mounts)
# ---------------------------------------------------------------------------
[ -e /dev/video0 ]  || echo "[entrypoint] WARNING: /dev/video0 not found — webcam input missing" >&2
[ -e /dev/video10 ] || echo "[entrypoint] WARNING: /dev/video10 not found — v4l2loopback output missing" >&2

# ---------------------------------------------------------------------------
# Fast path: sentinel means engines were already verified on a prior run
# ---------------------------------------------------------------------------
if [ -f "$SENTINEL" ] && [ "${FORCE_REBUILD_TRT:-0}" != "1" ]; then
    echo "[entrypoint] TRT engines already built (sentinel present). Skipping build."
else
    # Check whether any TRT engine is missing
    needs_build=0
    for onnx_name in "${!ONNX_TO_PRECISION[@]}"; do
        trt_name="${onnx_name%.onnx}.trt"
        if [ ! -f "$ONNX_DIR/$trt_name" ]; then
            needs_build=1
            break
        fi
    done

    if [ "$needs_build" -eq 1 ]; then
        echo "[entrypoint] One or more TRT engines missing — building now."
        echo "[entrypoint] ONNX source: $ONNX_DIR"
        echo "[entrypoint] This may take several minutes on first run."

        # Verify ONNX source files are present BEFORE starting any build
        missing_onnx=()
        for onnx_name in "${!ONNX_TO_PRECISION[@]}"; do
            [ -f "$ONNX_DIR/$onnx_name" ] || missing_onnx+=("$onnx_name")
        done

        if [ "${#missing_onnx[@]}" -gt 0 ]; then
            echo "[entrypoint] ERROR: The following ONNX source files are missing:" >&2
            for f in "${missing_onnx[@]}"; do
                echo "  $ONNX_DIR/$f" >&2
            done
            echo "" >&2
            echo "[entrypoint] Download them on the HOST (not inside the container):" >&2
            echo "  huggingface-cli download warmshao/FasterLivePortrait --local-dir $CHECKPOINTS_DIR" >&2
            echo "" >&2
            echo "[entrypoint] Or via Python on the host:" >&2
            echo "  python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('warmshao/FasterLivePortrait', local_dir='$CHECKPOINTS_DIR')\"" >&2
            exit 1
        fi

        # Build engines with per-engine progress counter
        built=0
        idx=0
        for onnx_name in "${!ONNX_TO_PRECISION[@]}"; do
            (( idx++ )) || true
            trt_name="${onnx_name%.onnx}.trt"
            trt_path="$ONNX_DIR/$trt_name"
            onnx_path="$ONNX_DIR/$onnx_name"
            precision="${ONNX_TO_PRECISION[$onnx_name]}"

            if [ -f "$trt_path" ]; then
                echo "[entrypoint] Skipping $idx/$TOTAL_ENGINES: $onnx_name (engine already exists)"
                continue
            fi

            echo "[entrypoint] Building $idx/$TOTAL_ENGINES: $onnx_name... (precision=$precision, this can take ~2 min)"
            python3 "$SCRIPTS_DIR/onnx2trt.py" \
                -o "$onnx_path" \
                -e "$trt_path" \
                -p "$precision"
            echo "[entrypoint] Done: $trt_name"
            (( built++ )) || true
        done

        echo "[entrypoint] All TRT engines built ($built new)."
    fi

    # Verify all engines are non-empty (> 1 MB = 1048576 bytes)
    echo "[entrypoint] Verifying TRT engine sizes..."
    for onnx_name in "${!ONNX_TO_PRECISION[@]}"; do
        trt_name="${onnx_name%.onnx}.trt"
        trt_path="$ONNX_DIR/$trt_name"
        if [ ! -f "$trt_path" ]; then
            echo "[entrypoint] ERROR: Expected engine not found after build: $trt_path" >&2
            exit 1
        fi
        size_bytes="$(wc -c < "$trt_path")"
        if [ "$size_bytes" -lt 1048576 ]; then
            echo "[entrypoint] ERROR: Engine suspiciously small (${size_bytes} bytes): $trt_path" >&2
            echo "[entrypoint] The build may have failed silently. Delete the file and restart." >&2
            exit 1
        fi
    done
    echo "[entrypoint] All TRT engines verified."

    # Write sentinel so future starts skip the check loop
    touch "$SENTINEL"
fi

# ---------------------------------------------------------------------------
# Portrait sanity check
# ---------------------------------------------------------------------------
if [ ! -f /app/src-portrait.jpg ]; then
    echo "[entrypoint] WARNING: /app/src-portrait.jpg not found." >&2
    echo "[entrypoint] Mount a neutral frontal portrait:" >&2
    echo "  -v /opt/maxine-portrait.jpg:/app/src-portrait.jpg:ro" >&2
fi

exec "$@"
