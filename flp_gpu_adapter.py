"""
flp_gpu_adapter.py — Zero-copy GPU adapter for FasterLivePortrait.

FasterLivePortraitPipeline composites the frontalized face back into the full
source frame with paste_back_pytorch on the GPU, then does a .cpu().numpy()
D2H at faster_live_portrait_pipeline.py:480 to hand a numpy image back to its
caller. That D2H is exactly what we want to avoid — the AR SDK gaze-redirect
stage can consume the same GPU buffer directly.

This adapter intercepts paste_back_pytorch by replacing the binding in the
pipeline module's namespace at import time. The wrapper calls the real
function, stashes its output tensor in a module-level holder, then returns
the tensor so FLP's internal flow is undisturbed. FLP still produces its
numpy for its own callers; we ignore that and hand the stashed GPU tensor
out of frontalize() instead.

Output contract
---------------
frontalize() returns a torch.cuda.FloatTensor with shape (H, W, 3), values
in the closed interval [0, 255] and RGB colour ordering. Shape matches the
source image (1080p if that's what you fed in). The tensor is valid only
until the next frontalize() call; clone it if you need to keep it.
"""

from __future__ import annotations

import ctypes
import logging
import os
import pathlib
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Resolve FLP imports from the vendored tree without editing vendor files.
# ---------------------------------------------------------------------------
_VENDOR_ROOT = pathlib.Path(__file__).parent / "vendor" / "FasterLivePortrait"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

# Pre-load the grid_sample_3d TRT 10 plugin with RTLD_GLOBAL so FLP's
# later relative-path ctypes.CDLL call (in src/models/predictor.py) is a
# no-op — it can silently fail (the symbols are already in the process
# namespace). The Dockerfile installs the plugin at /usr/local/lib.
_PLUGIN_PATH = os.environ.get(
    "GRID_SAMPLE_3D_PLUGIN",
    "/usr/local/lib/libgrid_sample_3d_plugin.so",
)
if os.path.isfile(_PLUGIN_PATH):
    try:
        ctypes.CDLL(_PLUGIN_PATH, mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "Failed to pre-load grid_sample_3d plugin (%s): %s", _PLUGIN_PATH, exc
        )

from omegaconf import OmegaConf  # noqa: E402
from src.pipelines import faster_live_portrait_pipeline as _flp_pipe_mod  # noqa: E402
from src.pipelines.faster_live_portrait_pipeline import (  # noqa: E402
    FasterLivePortraitPipeline,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level interception: replace paste_back_pytorch in the pipeline
# module's namespace. FLP's _run() method looks up paste_back_pytorch through
# that namespace at call time, so our wrapper is what actually runs.
# ---------------------------------------------------------------------------

_last_paste_back: list[torch.Tensor | None] = [None]
_original_paste_back = _flp_pipe_mod.paste_back_pytorch


def _intercepting_paste_back(img_crop, M_c2o, img_ori, mask_ori):  # noqa: N803
    """Wrap FLP's paste_back_pytorch; stash the full-frame GPU tensor.

    The underlying function returns a (H, W, 3) float32 CUDA tensor in the
    closed interval [0, 255], RGB ordering. We keep a module-level reference
    to that tensor so downstream stages (format conversion, AR SDK gaze
    redirect) can consume its device pointer directly.
    """
    result = _original_paste_back(img_crop, M_c2o, img_ori, mask_ori)
    _last_paste_back[0] = result
    return result


_flp_pipe_mod.paste_back_pytorch = _intercepting_paste_back


# ---------------------------------------------------------------------------
# Motion-smoothing: EMA-blend FLP's motion_extractor output across frames.
#
# FLP's motion_extractor produces slightly different (pitch, yaw, roll, t,
# exp, scale, kp) values every frame even for a completely static input
# face. The jitter amplifies through warping_spade into visible shimmer on
# the frontalized output. A cheap fix is to exponentially-weight the
# current frame's motion against the previous frame's.
#
# MAXINE_FLP_MOTION_EMA env var: 1.0 = smoothing off (raw FLP output),
# 0.3 default = ~65 ms half-response at 30 fps (stable but responsive),
# 0.1 = very smooth but noticeably laggy.
# ---------------------------------------------------------------------------

_motion_prev: list = [None]


def _reset_motion_smoothing() -> None:
    _motion_prev[0] = None


def _install_motion_smoothing(pipe, alpha: float) -> None:
    """Monkey-patch motion_extractor.predict to EMA-smooth its outputs."""
    if alpha >= 1.0:
        return
    ex = pipe.model_dict.get("motion_extractor") if hasattr(pipe, "model_dict") else None
    if ex is None:
        logger.warning("No motion_extractor in pipe.model_dict; skipping smoothing")
        return
    if getattr(ex, "_smoothed", False):
        return
    original = ex.predict

    def smoothed(*data, **kwargs):
        current = original(*data, **kwargs)
        prev = _motion_prev[0]
        if prev is None or len(prev) != len(current):
            _motion_prev[0] = current
            return current
        a = alpha
        try:
            blended = tuple(a * c + (1.0 - a) * p for c, p in zip(current, prev, strict=False))
        except (TypeError, ValueError):
            # If any element isn't array-like, fall back to the raw output.
            _motion_prev[0] = current
            return current
        _motion_prev[0] = blended
        return blended

    ex.predict = smoothed
    ex._smoothed = True


# ---------------------------------------------------------------------------
# Per-axis frontalization strength.
#
# After smoothing, each axis of the driving motion (pitch/yaw/roll) can be
# independently blended toward the source portrait's pose:
#
#     out = (1 - strength) * driving + strength * source_pose
#
# strength=1.0 → axis fully frontalizes (output == source pose for that axis)
# strength=0.0 → axis passes through unchanged (no correction for that axis)
# strength=0.5 → half-way between driving and source
# ---------------------------------------------------------------------------

_source_pose: list = [None]  # tuple(source_pitch, source_yaw, source_roll)


def _capture_source_pose(pipe) -> None:
    """Read the source-image pose out of FLP's src_infos cache."""
    try:
        info = pipe.src_infos[0][0]  # first face of the single source frame
        _source_pose[0] = (info["pitch"].copy(), info["yaw"].copy(), info["roll"].copy())
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        logger.warning("Could not capture source pose for axis-strength blend: %s", exc)


def headpose_predict_to_rotation_matrix(headpose: torch.Tensor) -> torch.Tensor:
    """
    Convert FLP headpose (pitch, yaw, roll) tensor to rotation matrix R.
    Expects headpose in shape (B, 3) or (3,).
    """
    if headpose.ndim == 1:
        headpose = headpose.unsqueeze(0)
    res = torch.zeros((headpose.shape[0], 3, 3), device=headpose.device, dtype=headpose.dtype)

    sin = torch.sin(headpose)
    cos = torch.cos(headpose)

    # ZYX rotation (standard for FLP)
    # R = R_z * R_y * R_x
    res[:, 0, 0] = cos[:, 1] * cos[:, 2]
    res[:, 0, 1] = sin[:, 0] * sin[:, 1] * cos[:, 2] - cos[:, 0] * sin[:, 2]
    res[:, 0, 2] = cos[:, 0] * sin[:, 1] * cos[:, 2] + sin[:, 0] * sin[:, 2]
    res[:, 1, 0] = cos[:, 1] * sin[:, 2]
    res[:, 1, 1] = sin[:, 0] * sin[:, 1] * sin[:, 2] + cos[:, 0] * cos[:, 2]
    res[:, 1, 2] = cos[:, 0] * sin[:, 1] * sin[:, 2] - sin[:, 0] * cos[:, 2]
    res[:, 2, 0] = -sin[:, 1]
    res[:, 2, 1] = sin[:, 0] * cos[:, 1]
    res[:, 2, 2] = cos[:, 0] * cos[:, 1]

    return res


def _install_axis_strength(pipe, strengths: tuple) -> None:
    """Blend each axis of motion_extractor output toward the source pose."""
    sp, sy, sr = strengths
    if sp == 0.0 and sy == 0.0 and sr == 0.0:
        return  # all three zero = pass driving motion through unchanged
    ex = pipe.model_dict.get("motion_extractor") if hasattr(pipe, "model_dict") else None
    if ex is None or getattr(ex, "_axis_blended", False):
        return
    inner = ex.predict  # already wrapped by smoothing if that was installed

    def blend(*data, **kwargs):
        result = list(inner(*data, **kwargs))
        result = inner(*data, **kwargs)
        src = _source_pose[0]
        if src is None or len(result) < 7:
            return result
        src_p, src_y, src_r = src
        p, y, r = result[0], result[1], result[2]
        t, exp_, scale, kp = result[3], result[4], result[5], result[6]
        try:
            p_out = (1.0 - sp) * p + sp * src_p
            y_out = (1.0 - sy) * y + sy * src_y
            r_out = (1.0 - sr) * r + sr * src_r
        except (TypeError, ValueError):
            return tuple(result)

        # Force re-computation of the rotation matrix R from the corrected euler
        # angles if R was in the original result.
        # FLP's result tuple typically has R at index 7 if present.
        if len(result) > 7:
            # We recompute R and update the tuple to keep it consistent.
            # result[0:3] are (pitch, yaw, roll)
            new_headpose = torch.stack([p_out, y_out, r_out], dim=-1)
            new_r = headpose_predict_to_rotation_matrix(new_headpose)
            return (p_out, y_out, r_out, t, exp_, scale, kp, new_r, *result[8:])

            return result
        return (p_out, y_out, r_out, t, exp_, scale, kp)

    ex.predict = blend
    ex._axis_blended = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class FLPFrontalizer:
    """One-time-init wrapper around FasterLivePortraitPipeline.

    Usage
    -----
        frontalizer = FLPFrontalizer(src_image_path="neutral.jpg")
        while capturing:
            t = frontalizer.frontalize(frame_bgr)
            if t is not None:
                # t is (H, W, 3) float32 RGB [0, 255] on CUDA
                consume_on_gpu(t)
    """

    def __init__(
        self,
        cfg_path: str = "vendor/FasterLivePortrait/configs/trt_infer.yaml",
        src_image_path: str | None = None,
        device: str = "cuda:0",
        motion_ema_alpha: float | None = None,
        axis_strength: tuple[float, float, float] | None = None,
    ):
        self._device = torch.device(device)
        torch.cuda.set_device(self._device)

        cfg = OmegaConf.load(cfg_path)
        self._pipe = FasterLivePortraitPipeline(cfg=cfg)

        self._img_src: np.ndarray | None = None
        self._src_info: list | None = None
        self._no_face_streak: int = 0
        self._no_face_log_interval: int = 30

        if src_image_path is not None:
            self._load_source(src_image_path)

        # Install motion smoothing AFTER _load_source so the source-image
        # motion extraction isn't blended with driving-frame extractions.
        # Reset the EMA state so the first driving frame starts clean.
        _reset_motion_smoothing()
        alpha = (
            motion_ema_alpha
            if motion_ema_alpha is not None
            else float(os.environ.get("MAXINE_FLP_MOTION_EMA", "0.3"))
        )
        _install_motion_smoothing(self._pipe, alpha=alpha)

        # Capture source pose + install per-axis strength blend. Order
        # matters: smoothing first (inner), axis-strength on top (outer).
        _capture_source_pose(self._pipe)
        strengths = axis_strength if axis_strength is not None else (1.0, 1.0, 1.0)
        _install_axis_strength(self._pipe, strengths)

    def _load_source(self, src_image_path: str) -> None:
        import cv2

        ok = self._pipe.prepare_source(src_image_path, realtime=True)
        if not ok or not self._pipe.src_infos:
            raise RuntimeError(f"FLP could not detect a face in source image: {src_image_path}")
        self._src_info = self._pipe.src_infos[0]
        img_bgr = cv2.imread(src_image_path)
        self._img_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    def set_source(self, src_image_path: str) -> None:
        """Replace the source portrait at runtime (drain pipeline first)."""
        self._load_source(src_image_path)
        _reset_motion_smoothing()

    def frontalize(self, frame_bgr: np.ndarray, overlay: bool = True) -> torch.Tensor | None:
    def frontalize(self, frame_bgr: np.ndarray) -> torch.Tensor | None:
        """Drive FLP with the current webcam frame.

        Parameters
        ----------
        frame_bgr:
            (H, W, 3) BGR uint8 numpy array — the current webcam frame.
        overlay:
            If true, paste the animated face back into the live webcam
            stream. If false, overwrite with the source portrait background.

        Returns
        -------
        torch.Tensor | None
            Full-frame paste-back as a (H, W, 3) float32 CUDA tensor, RGB,
            values in [0, 255]. Returns None if no face was detected.

        Notes
        -----
        The returned tensor lives on the current CUDA stream. It must be
        consumed (or cloned) before the next frontalize() call, otherwise
        the buffer may be overwritten by the next paste-back.
        """
        if self._src_info is None:
            raise RuntimeError(
                "Source image not set. Call set_source() or pass src_image_path to __init__."
            )

        # Clear the interception slot so a stale tensor from a previous frame
        # can't leak through on a no-face frame.
        _last_paste_back[0] = None

        # When overlay=True, we use the live webcam frame as the background.
        # FLP's run() uses the second argument as the background for paste-back.
        background = frame_bgr if overlay else self._img_src

        # Don't pass realtime= as kwarg — FLP's run() threads it positionally
        # into _run() internally, so a kwarg here causes a duplicate-argument
        # TypeError. FLP defaults work correctly for our continuous capture.
        result = self._pipe.run(
            frame_bgr,
            background,
            self._img_src,
            self._src_info,
        )
        _img_crop, _out_crop, i_p_pstbk_numpy, _motion = result

        if i_p_pstbk_numpy is None or _last_paste_back[0] is None:
            self._no_face_streak += 1
            if self._no_face_streak % self._no_face_log_interval == 1:
                logger.warning(
                    "FLP: no face for %d consecutive frames",
                    self._no_face_streak,
                )
            return None

        self._no_face_streak = 0
        return _last_paste_back[0]

    @property
    def last_output_pixel_format(self) -> str:
        """Colour space of the tensor returned by frontalize(): always 'rgb'."""
        return "rgb"

    @property
    def pipe(self) -> FasterLivePortraitPipeline:
        """Expose the underlying pipeline for advanced access."""
        return self._pipe


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------


def pull_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Synchronously copy a CUDA tensor to host — for debugging only."""
    torch.cuda.current_stream().synchronize()
    return tensor.cpu().numpy()
