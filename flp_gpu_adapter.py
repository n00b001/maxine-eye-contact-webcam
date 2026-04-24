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

import logging
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

    def frontalize(self, frame_bgr: np.ndarray) -> torch.Tensor | None:
        """Drive FLP with the current webcam frame.

        Parameters
        ----------
        frame_bgr:
            (H, W, 3) BGR uint8 numpy array — the current webcam frame.

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

        result = self._pipe.run(
            frame_bgr,
            self._img_src,
            self._src_info,
            realtime=True,
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
