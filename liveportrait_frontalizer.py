"""LivePortrait-based head-pose correction.

Drop-in replacement for :class:`frontalizer.Frontalizer` that uses the
KwaiVGI LivePortrait generative network to rotate the head toward the
camera. Unlike the geometric frontalizer, this path can synthesise the
pixels on the occluded side of the face during large yaw/pitch rotations.

Weights must be present at ``vendor/LivePortrait/pretrained_weights/``.
Dependencies (torch, torchvision, etc.) are installed via
``uv sync --extra liveportrait`` — see ``pyproject.toml``.

API is identical to :class:`frontalizer.Frontalizer`:

    frontalizer = LivePortraitFrontalizer(strength=1.0)
    warped_full = frontalizer.frontalize(frame, landmarks, pitch, yaw, roll)
    corrected  = frontalizer.blend_back(frame, warped_full, face_rect)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_LP_DIR = _HERE / "vendor" / "LivePortrait"
if _LP_DIR.exists() and str(_LP_DIR) not in sys.path:
    sys.path.insert(0, str(_LP_DIR))

_LP_IMPORT_ERROR: Exception | None = None
try:
    import torch  # noqa: F401
    from src.config.inference_config import InferenceConfig
    from src.live_portrait_wrapper import LivePortraitWrapper

    _LP_AVAILABLE = True
except ImportError as exc:
    torch = None  # type: ignore[assignment]
    InferenceConfig = None  # type: ignore[misc,assignment]
    LivePortraitWrapper = None  # type: ignore[misc,assignment]
    _LP_AVAILABLE = False
    _LP_IMPORT_ERROR = exc


_INPUT_SHAPE = (256, 256)
_CROP_PAD_RATIO = 0.6  # extra padding around landmarks bbox before resizing to 256x256
_MIN_CROP_SIZE = 96  # ignore tiny face detections; LivePortrait needs enough pixels
_FEATHER_RATIO = 0.12  # softness of the crop-boundary feather (fraction of crop size)


class LivePortraitFrontalizer:
    """Real head-pose correction via LivePortrait (PyTorch + fp16)."""

    def __init__(
        self,
        strength: float = 1.0,
        compile_models: bool = False,
    ):
        if not _LP_AVAILABLE:
            raise RuntimeError(
                "LivePortrait is not importable — "
                "run `uv sync --extra liveportrait` and ensure "
                f"vendor/LivePortrait/ exists. Original error: {_LP_IMPORT_ERROR}"
            )

        self.strength = float(strength)
        self.compile_models = bool(compile_models)

        cfg = InferenceConfig(
            flag_relative_motion=False,  # we drive with absolute target rotation
            animation_region="pose",  # only modify pose, keep source expression
            flag_stitching=False,  # we handle paste-back ourselves
            flag_eye_retargeting=False,
            flag_lip_retargeting=False,
            flag_normalize_lip=False,
            flag_use_half_precision=True,  # fp16 on RTX 4090
            flag_do_torch_compile=self.compile_models,
            flag_do_crop=False,  # we supply the 256x256 crop directly
            flag_pasteback=False,
        )
        self._wrapper = LivePortraitWrapper(inference_cfg=cfg)
        # Populated by frontalize() so blend_back() can feather at the
        # padded-crop rect the generative net actually drew over — not the
        # tight landmarks bbox the caller passes in, which would bleed
        # original off-axis pixels back into the face.
        self._last_crop_rect: tuple[int, int, int, int] | None = None

    # ------------------------------------------------------------------
    # Cropping helpers
    # ------------------------------------------------------------------
    def _face_crop_rect(
        self, frame: np.ndarray, landmarks: np.ndarray
    ) -> tuple[int, int, int, int] | None:
        """Compute a padded, square-ish face crop rect for LivePortrait.

        Larger than the raw landmarks bbox — LivePortrait was trained on
        portrait crops that include some forehead, cheeks, and chin margin.
        """
        h, w = frame.shape[:2]
        x0, y0 = landmarks.min(axis=0)
        x1, y1 = landmarks.max(axis=0)
        face_w = float(x1 - x0)
        face_h = float(y1 - y0)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        size = max(face_w, face_h) * (1.0 + 2.0 * _CROP_PAD_RATIO)
        half = size / 2.0
        rx = int(max(0, cx - half))
        ry = int(max(0, cy - half))
        rw = int(min(w - rx, size))
        rh = int(min(h - ry, size))
        if rw < _MIN_CROP_SIZE or rh < _MIN_CROP_SIZE:
            return None
        return rx, ry, rw, rh

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def frontalize(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        pitch: float,  # noqa: ARG002 — kept for API compatibility
        yaw: float,  # noqa: ARG002
        roll: float,  # noqa: ARG002
    ) -> np.ndarray | None:
        """Return a full-frame-sized image with the face rotated toward frontal.

        LivePortrait re-estimates the source pose from the crop, so the
        ``pitch`` / ``yaw`` / ``roll`` arguments are accepted for API
        compatibility with the geometric Frontalizer but not used directly.
        """
        if frame is None or frame.size == 0 or landmarks is None or landmarks.size == 0:
            return None

        rect = self._face_crop_rect(frame, landmarks)
        if rect is None:
            return None
        rx, ry, rw, rh = rect

        crop_bgr = frame[ry : ry + rh, rx : rx + rw]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_256 = cv2.resize(crop_rgb, _INPUT_SHAPE, interpolation=cv2.INTER_LINEAR)

        # --- LivePortrait forward pass -------------------------------------
        source = self._wrapper.prepare_source(crop_256)  # 1x3x256x256 cuda
        feature_3d = self._wrapper.extract_feature_3d(source)
        x_s_info = self._wrapper.get_kp_info(source, flag_refine_info=True)
        x_s = self._wrapper.transform_keypoint(x_s_info)

        # Target = blend of source rotation and (0, 0, 0) by strength.
        # strength=1 → full frontal; strength=0 → unchanged (identity).
        target_info = dict(x_s_info)
        alpha = 1.0 - self.strength
        target_info["pitch"] = x_s_info["pitch"] * alpha
        target_info["yaw"] = x_s_info["yaw"] * alpha
        target_info["roll"] = x_s_info["roll"] * alpha
        x_d = self._wrapper.transform_keypoint(target_info)

        out_dct = self._wrapper.warp_decode(feature_3d, x_s, x_d)
        out_rgb = self._wrapper.parse_output(out_dct["out"])[0]  # 256x256x3 RGB
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        # Upscale to the original crop size and paste back into a copy of
        # the full frame. We stash the padded-crop rect so blend_back can
        # feather at that boundary (not at the caller's tight landmarks
        # bbox, which would blend original off-axis pixels back *inside*
        # the face).
        out_crop = cv2.resize(out_bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)
        warped_full = frame.copy()
        warped_full[ry : ry + rh, rx : rx + rw] = out_crop
        self._last_crop_rect = (rx, ry, rw, rh)
        return warped_full

    def blend_back(
        self,
        original: np.ndarray,
        warped_frame: np.ndarray,
        face_rect: tuple[int, int, int, int],  # noqa: ARG002 — caller's rect is ignored
    ) -> np.ndarray:
        """Feather-blend the LP-rendered crop back into the full frame.

        Unlike the geometric Frontalizer, we can't elliptical-mask the
        landmarks bbox: LivePortrait has regenerated the entire head region
        (including hair, forehead, ears), and blending an ellipse inside
        the bbox would bleed the original off-axis face pixels back in at
        the ellipse edges. Instead we feather across the padded crop rect
        that :meth:`frontalize` drew over, so the whole head gets the new
        frontal version and only the last few pixels at the crop border
        fade into the original frame.
        """
        if self._last_crop_rect is None or warped_frame is None:
            return original.copy()

        if original is None or original.size == 0 or warped_frame.shape != original.shape:
            return original.copy() if original is not None else warped_frame

        rx, ry, rw, rh = self._last_crop_rect
        if rw <= 0 or rh <= 0:
            return original.copy()

        # Build a soft feather mask that's ~1 everywhere inside the crop
        # except near the boundary, where it ramps to 0 over a few pixels.
        feather = max(4, int(min(rw, rh) * _FEATHER_RATIO))
        mask = np.zeros((rh, rw), dtype=np.float32)
        inner_l = feather
        inner_r = max(inner_l + 1, rw - feather)
        inner_t = feather
        inner_b = max(inner_t + 1, rh - feather)
        mask[inner_t:inner_b, inner_l:inner_r] = 1.0
        # Gaussian blur the mask so the edge transition is smooth
        k = max(5, (feather * 2 + 1) | 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask = mask[..., None]

        output = original.copy()
        roi_w = warped_frame[ry : ry + rh, rx : rx + rw].astype(np.float32)
        roi_o = original[ry : ry + rh, rx : rx + rw].astype(np.float32)
        blended = roi_w * mask + roi_o * (1.0 - mask)
        output[ry : ry + rh, rx : rx + rw] = np.clip(blended, 0, 255).astype(np.uint8)
        return output

    def close(self) -> None:
        """Release GPU resources (best-effort)."""
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
