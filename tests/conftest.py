"""Pytest fixtures for the head-pose correction test suite."""

from __future__ import annotations

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

# Standard 468-point MediaPipe topology subset used by HeadPoseEstimator
# (nose tip, chin, left eye left corner, right eye right corner,
#  left mouth corner, right mouth corner)
DEFAULT_LANDMARKS_2D = np.array(
    [
        [128.0, 128.0],  # nose tip
        [128.0, 200.0],  # chin
        [80.0, 100.0],  # left eye left corner
        [176.0, 100.0],  # right eye right corner
        [90.0, 160.0],  # left mouth corner
        [166.0, 160.0],  # right mouth corner
    ],
    dtype=np.float64,
)


def _draw_face_on_canvas(canvas: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Draw a simple synthetic face on a BGR canvas using the given landmarks."""
    # Draw oval for face outline
    center = tuple(np.mean(landmarks, axis=0).astype(int))
    axes = (60, 80)
    cv2.ellipse(canvas, center, axes, 0, 0, 360, (200, 180, 160), thickness=-1)
    # Draw eyes
    for lm in landmarks[2:4]:
        cv2.circle(canvas, tuple(lm.astype(int)), 8, (30, 30, 30), thickness=-1)
    # Draw nose
    cv2.circle(canvas, tuple(landmarks[0].astype(int)), 6, (160, 140, 120), thickness=-1)
    # Draw mouth
    mouth_center = tuple(np.mean(landmarks[4:6], axis=0).astype(int))
    cv2.ellipse(canvas, mouth_center, (20, 10), 0, 0, 360, (60, 40, 40), thickness=-1)
    return canvas


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Return a generic 640x480 BGR image."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def synthetic_frontal_face() -> np.ndarray:
    """Generate a synthetic frontal face image (256x256) with known landmarks."""
    canvas = np.zeros((256, 256, 3), dtype=np.uint8)
    canvas = _draw_face_on_canvas(canvas, DEFAULT_LANDMARKS_2D)
    return canvas


@pytest.fixture
def synthetic_rotated_face() -> np.ndarray:
    """Return a factory that generates a face rotated by known angles."""

    def _factory(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
        # Start with a frontal face canvas
        canvas = np.zeros((256, 256, 3), dtype=np.uint8)
        canvas = _draw_face_on_canvas(canvas, DEFAULT_LANDMARKS_2D)

        # Build rotation matrix around image center
        center = (128, 128)
        # Apply yaw/roll as 2D rotation for synthetic data generation
        angle = roll + yaw * 0.5  # simplistic coupling for visual effect
        scale = 1.0 - abs(pitch) / 180.0 * 0.2  # slight scale change for pitch
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(canvas, M, (256, 256), borderValue=(0, 0, 0))
        return rotated

    return _factory


@pytest.fixture
def mock_estimator() -> MagicMock:
    """Return a mock HeadPoseEstimator that returns predetermined angles."""
    mock = MagicMock()
    mock.estimate.return_value = (5.0, -10.0, 2.0)
    mock.get_landmarks.return_value = DEFAULT_LANDMARKS_2D.copy()
    return mock


@pytest.fixture
def mock_frontalizer() -> MagicMock:
    """Return a mock Frontalizer that applies an identity-like transform."""
    mock = MagicMock()

    def _identity_frontalize(frame, landmarks, pitch, yaw, roll):
        # Return a copy of the input frame (identity transform)
        return frame.copy()

    def _identity_warp_matrix(src_landmarks, pitch, yaw, roll):
        # Return a 2x3 identity affine matrix
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    def _blend_back(original, warped_face, face_rect):
        # Return a copy of the original (no modification)
        return original.copy()

    mock.frontalize.side_effect = _identity_frontalize
    mock.compute_warp_matrix.side_effect = _identity_warp_matrix
    mock.blend_back.side_effect = _blend_back
    return mock
