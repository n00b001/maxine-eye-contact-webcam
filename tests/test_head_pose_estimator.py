"""Tests for head_pose_estimator.HeadPoseEstimator."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from head_pose_estimator import FACE_MODEL_3D, LANDMARK_INDICES, HeadPoseEstimator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inject_fake_mediapipe():
    """Create a fake mediapipe module tree so patches resolve without install."""
    fake_mp = types.ModuleType("mediapipe")
    fake_solutions = types.ModuleType("mediapipe.solutions")
    fake_face_mesh = types.ModuleType("mediapipe.solutions.face_mesh")
    fake_face_mesh.FaceMesh = MagicMock
    fake_solutions.face_mesh = fake_face_mesh
    fake_mp.solutions = fake_solutions
    sys.modules["mediapipe"] = fake_mp
    sys.modules["mediapipe.solutions"] = fake_solutions
    sys.modules["mediapipe.solutions.face_mesh"] = fake_face_mesh


def _remove_fake_mediapipe():
    for name in list(sys.modules.keys()):
        if name.startswith("mediapipe"):
            del sys.modules[name]


@pytest.fixture(autouse=True)
def _fake_mediapipe_fixture():
    _inject_fake_mediapipe()
    yield
    _remove_fake_mediapipe()


def _make_landmarks(frame_shape):
    """Build a mock landmark object matching MediaPipe structure."""
    h, w = frame_shape

    class Landmark:
        def __init__(self, x, y, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    class FaceLandmarks:
        def __init__(self, landmarks):
            self.landmark = landmarks

    landmarks = []
    for idx in range(max(LANDMARK_INDICES) + 1):
        if idx in LANDMARK_INDICES:
            i = LANDMARK_INDICES.index(idx)
            lm3d = FACE_MODEL_3D[i]
            x = lm3d[0] / 100.0 + 0.5
            y = lm3d[1] / 100.0 + 0.5
            landmarks.append(Landmark(x, y))
        else:
            landmarks.append(Landmark(0.5, 0.5))

    return FaceLandmarks(landmarks)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHeadPoseEstimatorInit:
    """Tests for HeadPoseEstimator initialization."""

    def test_init_default(self):
        """Estimator initializes with default parameters."""
        with patch("mediapipe.solutions.face_mesh.FaceMesh") as MockFaceMesh:
            estimator = HeadPoseEstimator()
            MockFaceMesh.assert_called_once()
            assert estimator.face_mesh is MockFaceMesh.return_value

    def test_init_static_image_mode_true(self):
        """Estimator accepts static_image_mode=True."""
        with patch("mediapipe.solutions.face_mesh.FaceMesh") as MockFaceMesh:
            HeadPoseEstimator(static_image_mode=True)
            kwargs = MockFaceMesh.call_args.kwargs
            assert kwargs.get("static_image_mode") is True

    def test_init_raises_on_missing_mediapipe(self):
        """Estimator raises ImportError when mediapipe is unavailable."""
        _remove_fake_mediapipe()
        with patch("builtins.__import__", side_effect=ImportError("No module named mediapipe")):
            with pytest.raises(ImportError):
                HeadPoseEstimator()
        _inject_fake_mediapipe()


class TestHeadPoseEstimatorEstimate:
    """Tests for HeadPoseEstimator.estimate()."""

    def test_estimate_with_synthetic_frontal_face(self, synthetic_frontal_face):
        """Angles are near zero for a perfectly frontal synthetic face."""
        mock_results = MagicMock()
        mock_results.multi_face_landmarks = [_make_landmarks(synthetic_frontal_face.shape[:2])]

        with patch("mediapipe.solutions.face_mesh.FaceMesh") as MockFaceMesh:
            instance = MockFaceMesh.return_value
            instance.process.return_value = mock_results

            estimator = HeadPoseEstimator(static_image_mode=True)
            angles = estimator.estimate(synthetic_frontal_face)

            assert angles is not None
            pitch, yaw, roll = angles
            # solvePnP on synthetic projected landmarks can deviate a few degrees
            assert abs(pitch) < 15.0, f"Expected pitch near 0, got {pitch}"
            assert abs(yaw) < 15.0, f"Expected yaw near 0, got {yaw}"
            assert abs(roll) < 15.0, f"Expected roll near 0, got {roll}"

    def test_estimate_returns_none_when_no_face(self, sample_frame):
        """estimate() returns None when no face is detected."""
        mock_results = MagicMock()
        mock_results.multi_face_landmarks = None

        with patch("mediapipe.solutions.face_mesh.FaceMesh") as MockFaceMesh:
            instance = MockFaceMesh.return_value
            instance.process.return_value = mock_results

            estimator = HeadPoseEstimator(static_image_mode=True)
            angles = estimator.estimate(sample_frame)

            assert angles is None

    def test_estimate_returns_none_on_solvepnp_failure(self, synthetic_frontal_face):
        """estimate() returns None if solvePnP fails."""
        mock_results = MagicMock()
        mock_results.multi_face_landmarks = [_make_landmarks(synthetic_frontal_face.shape[:2])]

        with (
            patch("mediapipe.solutions.face_mesh.FaceMesh") as MockFaceMesh,
            patch("cv2.solvePnP", return_value=(False, None, None)),
        ):
            instance = MockFaceMesh.return_value
            instance.process.return_value = mock_results

            estimator = HeadPoseEstimator(static_image_mode=True)
            angles = estimator.estimate(synthetic_frontal_face)

            assert angles is None


class TestHeadPoseEstimatorGetLandmarks:
    """Tests for HeadPoseEstimator.get_landmarks()."""

    def test_get_landmarks_returns_array(self, synthetic_frontal_face):
        """get_landmarks returns a numpy array of shape (6, 2)."""
        mock_results = MagicMock()
        mock_results.multi_face_landmarks = [_make_landmarks(synthetic_frontal_face.shape[:2])]

        with patch("mediapipe.solutions.face_mesh.FaceMesh") as MockFaceMesh:
            instance = MockFaceMesh.return_value
            instance.process.return_value = mock_results

            estimator = HeadPoseEstimator(static_image_mode=True)
            landmarks = estimator.get_landmarks(synthetic_frontal_face)

            assert isinstance(landmarks, np.ndarray)
            assert landmarks.shape == (6, 2)

    def test_get_landmarks_returns_none_when_no_face(self, sample_frame):
        """get_landmarks returns None when no face is present."""
        mock_results = MagicMock()
        mock_results.multi_face_landmarks = None

        with patch("mediapipe.solutions.face_mesh.FaceMesh") as MockFaceMesh:
            instance = MockFaceMesh.return_value
            instance.process.return_value = mock_results

            estimator = HeadPoseEstimator(static_image_mode=True)
            landmarks = estimator.get_landmarks(sample_frame)

            assert landmarks is None


class TestHeadPoseEstimatorClose:
    """Tests for HeadPoseEstimator.close()."""

    def test_close_releases_resources(self):
        """close() calls the underlying FaceMesh close method."""
        with patch("mediapipe.solutions.face_mesh.FaceMesh") as MockFaceMesh:
            instance = MockFaceMesh.return_value
            estimator = HeadPoseEstimator()
            estimator.close()
            instance.close.assert_called_once()
