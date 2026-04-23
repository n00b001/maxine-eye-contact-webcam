"""Lightweight geometric head-pose correction.

Given a frame, the six MediaPipe-derived landmarks
``[nose_tip, chin, eye_L, eye_R, mouth_L, mouth_R]`` (see ``LANDMARK_INDICES``
in ``head_pose_estimator``), and estimated pitch/yaw/roll, rotate the face
toward the camera by an affine warp. The warp is computed from three stable
anchor points (nose tip + the two outer eye corners) and applied to the
whole frame, then alpha-blended into the original so only the face region
is affected.

This is an approximation — full head-pose correction at large angles needs a
generative model (see ``plans/head-pose-correction-plan.md``). At small-to-
moderate pose angles, however, this approach:

  * rotates out head roll (rotation around the forward axis),
  * partially compensates for yaw/pitch foreshortening by stretching the
    inter-eye and eye-to-nose distances by ``1/cos(angle)``,
  * never crops the output smaller than the input (the old implementation
    wrote to a fixed 256×256 window that excluded most of the face).
"""

from __future__ import annotations

import cv2
import numpy as np

# Landmark row indices within the 6-point layout produced by
# ``HeadPoseEstimator``. Kept here (rather than hardcoded integers) so the
# coupling between the two modules is explicit.
_NOSE_TIP = 0
_EYE_L = 2
_EYE_R = 3

# Clamp `1 / cos(angle)` to this range so very large pose angles don't
# produce an unbounded stretch that turns the face into smeared border
# pixels. At yaw ≈ 70° the raw scale factor would be ~2.9; we cap it.
_MAX_SCALE = 1.8
_MIN_SCALE = 0.6


class Frontalizer:
    """Apply a per-frame affine warp that rotates the face toward frontal."""

    def __init__(self, strength: float = 1.0):
        """Initialize the frontalizer.

        Args:
            strength: Blend factor between the identity warp (0.0) and the
                full geometric correction (1.0).
        """
        self.strength = float(strength)

    # ------------------------------------------------------------------
    # Core geometry
    # ------------------------------------------------------------------
    def compute_warp_matrix(
        self,
        src_landmarks: np.ndarray,
        pitch: float,
        yaw: float,
        roll: float,
    ) -> np.ndarray | None:
        """Build a 2×3 affine that undoes pose rotation on the face region.

        Uses ``(nose, eye_L, eye_R)`` as anchor points. Returns ``None`` if
        landmarks are degenerate.
        """
        if src_landmarks is None or src_landmarks.size == 0:
            return None

        src = np.asarray(src_landmarks, dtype=np.float32)
        if len(src) <= max(_NOSE_TIP, _EYE_L, _EYE_R):
            return None

        nose = src[_NOSE_TIP]
        eye_l = src[_EYE_L]
        eye_r = src[_EYE_R]
        pts_src = np.array([nose, eye_l, eye_r], dtype=np.float32)

        # Work in a nose-centred frame so the nose stays fixed and only the
        # eyes move under the correction.
        center = nose.copy()
        pts_centered = pts_src - center

        # Undo roll (rotation around the camera's forward axis).
        roll_rad = np.deg2rad(-roll)
        cr, sr = float(np.cos(roll_rad)), float(np.sin(roll_rad))
        rot = np.array([[cr, -sr], [sr, cr]], dtype=np.float32)
        pts_centered = (rot @ pts_centered.T).T

        # Partially compensate for yaw/pitch foreshortening by stretching
        # the horizontal / vertical components.
        yaw_rad = np.deg2rad(yaw)
        pitch_rad = np.deg2rad(pitch)
        scale_x = float(np.clip(1.0 / max(abs(np.cos(yaw_rad)), 0.1), _MIN_SCALE, _MAX_SCALE))
        scale_y = float(np.clip(1.0 / max(abs(np.cos(pitch_rad)), 0.1), _MIN_SCALE, _MAX_SCALE))
        pts_centered[:, 0] *= scale_x
        pts_centered[:, 1] *= scale_y

        pts_full_correction = pts_centered + center

        # Single linear blend: strength=0 → identity; strength=1 → full warp.
        pts_target = ((1.0 - self.strength) * pts_src + self.strength * pts_full_correction).astype(
            np.float32
        )

        # getAffineTransform requires three non-collinear source points.
        if np.linalg.matrix_rank(pts_src - pts_src.mean(axis=0)) < 2:
            return None

        return cv2.getAffineTransform(pts_src, pts_target)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def frontalize(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        pitch: float,
        yaw: float,
        roll: float,
    ) -> np.ndarray | None:
        """Warp ``frame`` so the face appears more frontal.

        Returns the warped image at the same shape as ``frame`` (previously
        the output was a fixed 256×256 crop of the *top-left* of the warped
        frame, which cut out the face entirely). Returns ``None`` if the
        warp matrix cannot be computed.
        """
        if frame is None or frame.size == 0 or landmarks is None or landmarks.size == 0:
            return None

        warp_matrix = self.compute_warp_matrix(landmarks, pitch, yaw, roll)
        if warp_matrix is None:
            return None

        h, w = frame.shape[:2]
        return cv2.warpAffine(
            frame,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def blend_back(
        self,
        original: np.ndarray,
        warped_frame: np.ndarray,
        face_rect: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Alpha-blend ``warped_frame`` into ``original`` within ``face_rect``.

        ``warped_frame`` must be the *full* warped frame (same shape as
        ``original``) returned by :meth:`frontalize`. An elliptical feathered
        mask restricts blending to the face region so the rest of the image
        is untouched.
        """
        if original is None or original.size == 0:
            return original.copy() if original is not None else None

        if warped_frame is None or warped_frame.size == 0 or warped_frame.shape != original.shape:
            return original.copy()

        x, y, w, h = face_rect
        if w <= 0 or h <= 0:
            return original.copy()

        img_h, img_w = original.shape[:2]
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(img_w, int(x + w))
        y2 = min(img_h, int(y + h))
        if x2 <= x1 or y2 <= y1:
            return original.copy()

        rect_w = x2 - x1
        rect_h = y2 - y1

        # Feathered elliptical mask: ≈1 at centre, tapering to 0 at the
        # rect boundary. Gaussian blur softens the transition so the face
        # region doesn't show a hard ellipse edge.
        mask = np.zeros((rect_h, rect_w), dtype=np.uint8)
        cx, cy = rect_w // 2, rect_h // 2
        ax = max(1, rect_w // 2 - 4)
        ay = max(1, rect_h // 2 - 4)
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
        blur_k = max(5, (min(rect_w, rect_h) // 8) | 1)  # odd kernel size
        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
        mask_f = (mask.astype(np.float32) / 255.0)[..., None]

        output = original.copy()
        roi_original = original[y1:y2, x1:x2].astype(np.float32)
        roi_warped = warped_frame[y1:y2, x1:x2].astype(np.float32)
        blended = roi_warped * mask_f + roi_original * (1.0 - mask_f)
        output[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return output
