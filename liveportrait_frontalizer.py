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
from typing import Any

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
        pitch_strength: float | None = None,
        yaw_strength: float | None = None,
        roll_strength: float | None = None,
        source_image_path: str | None = None,
    ):
        if not _LP_AVAILABLE:
            raise RuntimeError(
                "LivePortrait is not importable — "
                "run `uv sync --extra liveportrait` and ensure "
                f"vendor/LivePortrait/ exists. Original error: {_LP_IMPORT_ERROR}"
            )

        self.strength = float(strength)
        self.compile_models = bool(compile_models)
        # Per-axis strengths. If an axis is None it inherits `strength`, so
        # callers that only pass `strength` keep the original behaviour.
        # Setting an axis to 0 preserves that axis of the source rotation
        # (e.g. roll_strength=0 → do not level the head to horizontal).
        self.pitch_strength = float(strength if pitch_strength is None else pitch_strength)
        self.yaw_strength = float(strength if yaw_strength is None else yaw_strength)
        self.roll_strength = float(strength if roll_strength is None else roll_strength)

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

        # Static source image support
        self.source_image_path = source_image_path
        self._src_data: dict[str, Any] | None = None
        if self.source_image_path:
            self._prepare_static_source(self.source_image_path)

    def _prepare_static_source(self, path: str) -> None:
        """Load and pre-process a static source image."""
        img = cv2.imread(path)
        if img is None:
            print(f"[LivePortrait] WARNING: could not load source image {path}")
            return

        from head_pose_estimator import HeadPoseEstimator

        hpe = HeadPoseEstimator(static_image_mode=True)
        try:
            lms = hpe.get_landmarks(img)
            if lms is None:
                print(f"[LivePortrait] WARNING: no face found in source image {path}")
                return

            rect = self._face_crop_rect(img, lms)
            if rect is None:
                return

            rx, ry, rw, rh = rect
            crop_bgr = img[ry : ry + rh, rx : rx + rw]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_256 = cv2.resize(crop_rgb, _INPUT_SHAPE, interpolation=cv2.INTER_LINEAR)

            source = self._wrapper.prepare_source(crop_256)
            self._src_data = {
                "feature_3d": self._wrapper.extract_feature_3d(source),
                "kp_info": self._wrapper.get_kp_info(source, flag_refine_info=True),
            }
            print(f"[LivePortrait] Loaded static source image: {path}")
        finally:
            hpe.close()

    # ------------------------------------------------------------------
    # Cropping helpers
    # ------------------------------------------------------------------
    def _face_crop_rect(
        self, frame: np.ndarray, landmarks: np.ndarray
    ) -> tuple[int, int, int, int] | None:
        """Compute the ideal square face crop rect (x_min, y_min, x_max, y_max).

        Can return coordinates outside image boundaries; the caller must
        handle padding.
        """
        x0, y0 = landmarks.min(axis=0)
        x1, y1 = landmarks.max(axis=0)
        face_w = float(x1 - x0)
        face_h = float(y1 - y0)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        size = max(face_w, face_h) * (1.0 + 2.0 * _CROP_PAD_RATIO)
        x_min = int(cx - size / 2.0)
        y_min = int(cy - size / 2.0)
        x_max = int(x_min + size)
        y_max = int(y_min + size)

        if (x_max - x_min) < _MIN_CROP_SIZE:
            return None
        return x_min, y_min, x_max, y_max

    def _get_robust_crop(
        self, frame: np.ndarray, rect: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Crop and pad frame to exactly match rect, ensuring it's square."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = rect

        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(w, x2)
        src_y2 = min(h, y2)

        # Partial crop within bounds
        crop_part = frame[src_y1:src_y2, src_x1:src_x2]

        # Padding needed to reach target rect
        pad_t = src_y1 - y1
        pad_b = y2 - src_y2
        pad_l = src_x1 - x1
        pad_r = x2 - src_x2

        crop_full = cv2.copyMakeBorder(
            crop_part, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0
        )
        # Store the actual intersection for paste-back
        return crop_full, (src_x1, src_y1, src_x2, src_y2)

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

        crop_bgr, intersect = self._get_robust_crop(frame, rect)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_256 = cv2.resize(crop_rgb, _INPUT_SHAPE, interpolation=cv2.INTER_LINEAR)

        # --- LivePortrait forward pass -------------------------------------
        # If we have a static source, we use its features but drive it with
        # the motion from the live frame.
        if self._src_data:
            feature_3d = self._src_data["feature_3d"]
            x_s_info = self._src_data["kp_info"]
        else:
            source = self._wrapper.prepare_source(crop_256)
            feature_3d = self._wrapper.extract_feature_3d(source)
            x_s_info = self._wrapper.get_kp_info(source, flag_refine_info=True)

        x_s = self._wrapper.transform_keypoint(x_s_info)

        # Driving info comes from the live crop
        live_source = self._wrapper.prepare_source(crop_256)
        x_d_info = self._wrapper.get_kp_info(live_source, flag_refine_info=True)

        # Target = per-axis blend of LIVE rotation and 0.
        # We want to correct the LIVE pose towards frontal.
        target_info = dict(x_d_info)
        target_info["pitch"] = x_d_info["pitch"] * (1.0 - self.pitch_strength)
        target_info["yaw"] = x_d_info["yaw"] * (1.0 - self.yaw_strength)
        target_info["roll"] = x_d_info["roll"] * (1.0 - self.roll_strength)

        # IMPORTANT: Remove 'R' if it exists, so transform_keypoint recomputes
        # it from the modified euler angles. This is often why correction
        # "doesn't work" in LivePortrait.
        target_info.pop("R", None)

        # If we have a static source, we want to keep its scale/translation
        # to keep the face aligned with the original crop, but take the
        # expression and corrected pose from the live stream.
        if self._src_data:
            target_info["scale"] = x_s_info["scale"]
            target_info["t"] = x_s_info["t"]

        x_d = self._wrapper.transform_keypoint(target_info)

        out_dct = self._wrapper.warp_decode(feature_3d, x_s, x_d)
        out_rgb = self._wrapper.parse_output(out_dct["out"])[0]  # 256x256x3 RGB
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        # Upscale to the original (square) crop size.
        size = rect[2] - rect[0]
        out_crop_full = cv2.resize(out_bgr, (size, size), interpolation=cv2.INTER_LINEAR)

        # Map back to the original frame intersection.
        # rect = (x1, y1, x2, y2)
        # intersect = (src_x1, src_y1, src_x2, src_y2)
        x1, y1, x2, y2 = rect
        ix1, iy1, ix2, iy2 = intersect

        # The part of out_crop_full that corresponds to the intersection
        crop_ix1 = ix1 - x1
        crop_iy1 = iy1 - y1
        crop_ix2 = ix2 - x1
        crop_iy2 = iy2 - y1

        out_crop = out_crop_full[crop_iy1:crop_iy2, crop_ix1:crop_ix2]

        warped_full = frame.copy()
        warped_full[iy1:iy2, ix1:ix2] = out_crop

        # For blend_back, we feather at the intersection boundary.
        self._last_crop_rect = (ix1, iy1, ix2 - ix1, iy2 - iy1)
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
