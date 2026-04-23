"""Geometric frontalization of a face region using perspective warping."""

from __future__ import annotations

import cv2
import numpy as np


class Frontalizer:
    """Apply perspective warp to rotate a face toward frontal view."""

    def __init__(
        self,
        output_size: tuple[int, int] = (256, 256),
        strength: float = 1.0,
    ):
        """Initialize the frontalizer.

        Args:
            output_size: (width, height) of the output face patch.
            strength: 0.0 = no correction, 1.0 = full correction.
        """
        self.output_size = output_size
        self.strength = strength

    def frontalize(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        pitch: float,
        yaw: float,
        roll: float,
    ) -> np.ndarray | None:
        """Extract and frontalize the face region.

        Args:
            frame: BGR image containing the face.
            landmarks: Array of shape (N, 2) with facial landmark pixel coords.
            pitch: Head pitch in degrees.
            yaw: Head yaw in degrees.
            roll: Head roll in degrees.

        Returns:
            Frontalized face patch as BGR image, or None if warp fails.
        """
        if frame.size == 0 or landmarks.size == 0:
            return None

        warp_matrix = self.compute_warp_matrix(landmarks, pitch, yaw, roll)
        if warp_matrix is None:
            return None

        warped = cv2.warpAffine(
            frame,
            warp_matrix,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return warped

    def compute_warp_matrix(
        self,
        src_landmarks: np.ndarray,
        pitch: float,
        yaw: float,
        roll: float,
    ) -> np.ndarray | None:
        """Compute the perspective transform matrix for frontalization.

        Uses the head pose angles to compute where landmarks would be
        if the head were facing the camera directly.

        Returns:
            2x3 affine transform matrix, or None.
        """
        if src_landmarks.size == 0 or len(src_landmarks) < 3:
            return None

        # Ensure float32 for OpenCV
        src = np.asarray(src_landmarks, dtype=np.float32)

        # Use outer eye corners and nose tip as anchor points.
        # Standard 68-point indices: left eye outer=36, right eye outer=45,
        # nose tip=30.  For 5-point: indices 0,2,4 correspond to those roughly.
        n = len(src)
        if n >= 68:
            left_eye = src[36]
            right_eye = src[45]
            nose_tip = src[30]
        elif n >= 5:
            left_eye = src[0]
            right_eye = src[2]
            nose_tip = src[4]
        else:
            # Fallback: use first three points
            left_eye = src[0]
            right_eye = src[1]
            nose_tip = src[2]

        pts_src = np.array([left_eye, right_eye, nose_tip], dtype=np.float32)

        # Compute target (frontal) positions.
        # Start from the source positions and undo the rotation implied by
        # pitch/yaw/roll.  For a simplified geometric correction we:
        #   1. Center the triangle at the nose tip.
        #   2. Rotate in the opposite direction of roll.
        #   3. Scale the inter-eye distance to undo yaw/pitch foreshortening.
        #   4. Re-apply the center offset.
        #
        # This is a first-order approximation; full 3D projection is overkill
        # for a lightweight real-time warp.
        center = nose_tip.copy()
        pts_centered = pts_src - center

        # Compute the full-strength correction first, then apply a single
        # linear blend against the source positions using `self.strength`.
        # Applying strength to the angles *and* blending the positions would
        # double-apply it (~strength^2 at mid-values).
        roll_rad = np.deg2rad(-roll)
        cos_r = np.cos(roll_rad)
        sin_r = np.sin(roll_rad)
        roll_rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
        pts_centered = (roll_rot @ pts_centered.T).T

        yaw_rad = np.deg2rad(yaw)
        pitch_rad = np.deg2rad(pitch)
        # cos(angle) is the projected length; 1/cos(angle) restores it.
        scale_x = 1.0 / max(np.cos(yaw_rad), 0.1)
        scale_y = 1.0 / max(np.cos(pitch_rad), 0.1)

        pts_centered[:, 0] *= scale_x
        pts_centered[:, 1] *= scale_y

        pts_full_correction = pts_centered + center

        # Single linear blend: strength=0 → identity; strength=1 → full warp.
        pts_target = (1.0 - self.strength) * pts_src + self.strength * pts_full_correction

        # Compute affine transform
        warp_matrix = cv2.getAffineTransform(pts_src, pts_target)
        if warp_matrix is None:
            return None

        return warp_matrix

    def blend_back(
        self,
        original: np.ndarray,
        warped_face: np.ndarray,
        face_rect: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Blend the corrected face patch back into the original frame.

        Args:
            original: Original BGR frame.
            warped_face: Frontalized face patch.
            face_rect: (x, y, w, h) bounding box where to place the face.

        Returns:
            Frame with blended face.
        """
        if original.size == 0 or warped_face.size == 0:
            return original.copy()

        x, y, w, h = face_rect
        if w <= 0 or h <= 0:
            return original.copy()

        # Clamp rect to image bounds
        img_h, img_w = original.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        if x2 <= x1 or y2 <= y1:
            return original.copy()

        # Resize warped face to fit the (possibly clamped) rect
        target_w = x2 - x1
        target_h = y2 - y1
        resized = cv2.resize(warped_face, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        output = original.copy()
        roi = output[y1:y2, x1:x2]

        # Create an elliptical alpha mask for feathered blending
        mask = np.zeros((target_h, target_w), dtype=np.uint8)
        center = (target_w // 2, target_h // 2)
        axes = (max(1, target_w // 2 - 4), max(1, target_h // 2 - 4))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        # Gaussian blur to feather edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        # Normalize mask to [0, 1]
        mask_f = mask.astype(np.float32) / 255.0
        mask_f = np.expand_dims(mask_f, axis=-1)

        # Alpha blend
        blended = resized.astype(np.float32) * mask_f + roi.astype(np.float32) * (1.0 - mask_f)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        output[y1:y2, x1:x2] = blended
        return output
