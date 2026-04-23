"""Tests for frontalizer.Frontalizer."""

from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np
import pytest

# Frontalizer may not exist yet; import conditionally and skip if missing
try:
    from frontalizer import Frontalizer
except ImportError:
    pytest.skip("frontalizer module not available", allow_module_level=True)


class TestFrontalizerInit:
    """Tests for Frontalizer initialization."""

    def test_init_default(self):
        """Frontalizer initializes with default parameters."""
        frontalizer = Frontalizer()
        assert frontalizer.strength == 1.0

    def test_init_custom_strength(self):
        """Frontalizer accepts custom strength."""
        frontalizer = Frontalizer(strength=0.5)
        assert frontalizer.strength == 0.5


class TestFrontalizerComputeWarpMatrix:
    """Tests for Frontalizer.compute_warp_matrix()."""

    def test_compute_warp_matrix_returns_valid_matrix(self, synthetic_frontal_face):
        """compute_warp_matrix returns a 2x3 affine matrix."""
        frontalizer = Frontalizer()
        landmarks = np.array(
            [
                [128.0, 128.0],
                [128.0, 200.0],
                [80.0, 100.0],
                [176.0, 100.0],
                [90.0, 160.0],
                [166.0, 160.0],
            ],
            dtype=np.float64,
        )

        matrix = frontalizer.compute_warp_matrix(landmarks, pitch=0.0, yaw=0.0, roll=0.0)

        assert matrix is not None
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (2, 3)

    def test_compute_warp_matrix_identity_for_zero_angles(self, synthetic_frontal_face):
        """With zero angles and strength=0, warp matrix should be near-identity."""
        frontalizer = Frontalizer(strength=0.0)
        landmarks = np.array(
            [
                [128.0, 128.0],
                [128.0, 200.0],
                [80.0, 100.0],
                [176.0, 100.0],
                [90.0, 160.0],
                [166.0, 160.0],
            ],
            dtype=np.float64,
        )

        matrix = frontalizer.compute_warp_matrix(landmarks, pitch=0.0, yaw=0.0, roll=0.0)

        # Near-identity affine matrix
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        assert np.allclose(matrix, identity, atol=1e-3)


class TestFrontalizerFrontalize:
    """Tests for Frontalizer.frontalize()."""

    def test_frontalize_returns_full_frame_shape(self, synthetic_frontal_face):
        """frontalize returns an image with the same shape as the input frame.

        Regression guard: the previous implementation wrote to a fixed
        256x256 window anchored at the top-left of the warped frame, which
        excluded the face region entirely and caused the "face removed"
        artefact when blended back.
        """
        frontalizer = Frontalizer()
        landmarks = np.array(
            [
                [128.0, 128.0],
                [128.0, 200.0],
                [80.0, 100.0],
                [176.0, 100.0],
                [90.0, 160.0],
                [166.0, 160.0],
            ],
            dtype=np.float64,
        )

        result = frontalizer.frontalize(
            synthetic_frontal_face, landmarks, pitch=0.0, yaw=0.0, roll=0.0
        )

        assert result is not None
        assert result.shape == synthetic_frontal_face.shape

    def test_frontalize_returns_none_on_failure(self):
        """frontalize returns None if warp matrix computation fails."""
        frontalizer = Frontalizer()
        with patch.object(frontalizer, "compute_warp_matrix", return_value=None):
            result = frontalizer.frontalize(
                np.zeros((256, 256, 3), dtype=np.uint8),
                np.zeros((6, 2), dtype=np.float64),
                pitch=0.0,
                yaw=0.0,
                roll=0.0,
            )
            assert result is None


class TestFrontalizerBlendBack:
    """Tests for Frontalizer.blend_back()."""

    def test_blend_back_same_size_as_input(self, synthetic_frontal_face):
        """blend_back produces output of the same size as the original."""
        frontalizer = Frontalizer()
        original = synthetic_frontal_face
        warped = np.zeros_like(original)  # full-frame warped (same shape)
        face_rect = (64, 64, 128, 128)

        result = frontalizer.blend_back(original, warped, face_rect)

        assert result.shape == original.shape

    def test_blend_back_preserves_original_outside_face_rect(self, synthetic_frontal_face):
        """Pixels outside face_rect should remain unchanged."""
        frontalizer = Frontalizer()
        original = synthetic_frontal_face.copy()
        warped = np.full_like(original, 255)  # bright-white warped frame
        face_rect = (100, 100, 120, 120)

        result = frontalizer.blend_back(original, warped, face_rect)

        # A pixel well outside the face rect must match the original.
        assert np.array_equal(result[0, 0], original[0, 0])

    def test_blend_back_preserves_face_region_not_blackened(self, synthetic_frontal_face):
        """Regression guard for the 'face removed' bug: when the warped
        frame is a real warp of the original (not empty), the blended face
        region must still contain visible content, not be blackened out.
        """
        frontalizer = Frontalizer(strength=1.0)
        original = synthetic_frontal_face.copy()
        warped = cv2.flip(original, 1)  # simulate a real full-frame warp
        face_rect = (64, 64, 128, 128)

        result = frontalizer.blend_back(original, warped, face_rect)

        center_pixel = result[128, 128]
        assert int(center_pixel.sum()) > 0, (
            "face centre pixel is black — frontalizer produced an empty patch"
        )


class TestFrontalizerStrength:
    """Tests for Frontalizer strength parameter behavior."""

    def test_strength_zero_returns_near_identity(self, synthetic_frontal_face):
        """strength=0 returns a near-identity warp (output similar to input)."""
        frontalizer = Frontalizer(strength=0.0)
        landmarks = np.array(
            [
                [128.0, 128.0],
                [128.0, 200.0],
                [80.0, 100.0],
                [176.0, 100.0],
                [90.0, 160.0],
                [166.0, 160.0],
            ],
            dtype=np.float64,
        )

        result = frontalizer.frontalize(
            synthetic_frontal_face, landmarks, pitch=0.0, yaw=0.0, roll=0.0
        )

        assert result is not None
        # With strength=0 the warp should be very close to identity
        diff = cv2.absdiff(synthetic_frontal_face, result)
        mean_diff = np.mean(diff)
        assert mean_diff < 10.0, f"Expected small diff for strength=0, got {mean_diff}"

    def test_strength_one_returns_full_correction(self, synthetic_frontal_face):
        """strength=1 applies the full correction warp."""
        frontalizer = Frontalizer(strength=1.0)
        landmarks = np.array(
            [
                [128.0, 128.0],
                [128.0, 200.0],
                [80.0, 100.0],
                [176.0, 100.0],
                [90.0, 160.0],
                [166.0, 160.0],
            ],
            dtype=np.float64,
        )

        result = frontalizer.frontalize(
            synthetic_frontal_face, landmarks, pitch=10.0, yaw=15.0, roll=5.0
        )

        assert result is not None
        assert result.shape == synthetic_frontal_face.shape
        # Full correction should produce a visibly different image
        diff = cv2.absdiff(synthetic_frontal_face, result)
        mean_diff = np.mean(diff)
        assert mean_diff > 0.0, "Expected some change for strength=1 with non-zero angles"
