"""Head pose estimation using MediaPipe Face Mesh + OpenCV solvePnP."""

from __future__ import annotations

import cv2
import numpy as np

# 3D facial landmark model (standard 468-point MediaPipe topology)
# Use a simplified subset of key landmarks for solvePnP
# Nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner
FACE_MODEL_3D = np.array(
    [
        [0.0, 0.0, 0.0],  # Nose tip
        [0.0, -63.6, -12.5],  # Chin
        [-43.3, 32.7, -26.0],  # Left eye left corner
        [43.3, 32.7, -26.0],  # Right eye right corner
        [-28.9, -28.9, -24.1],  # Left mouth corner
        [28.9, -28.9, -24.1],  # Right mouth corner
    ],
    dtype=np.float64,
)

# MediaPipe landmark indices corresponding to the above 3D points
LANDMARK_INDICES = [1, 199, 33, 263, 61, 291]


class HeadPoseEstimator:
    """Estimate head pose (pitch, yaw, roll in degrees) from a BGR frame."""

    def __init__(self, static_image_mode: bool = False):
        """Initialize the estimator.

        Args:
            static_image_mode: If True, treats input images as unrelated (slower but more accurate).
                               If False, uses previous frame for tracking (faster, for video).
        """
        # Try to import mediapipe; if not available, raise a clear error
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "mediapipe is required for head pose estimation. Install it with: uv add mediapipe"
            ) from exc

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _build_camera_matrix(self, h: int, w: int) -> np.ndarray:
        """Build a 3x3 camera matrix from image dimensions (focal ≈ width)."""
        focal_length = float(w)
        cx = w / 2.0
        cy = h / 2.0
        return np.array(
            [
                [focal_length, 0.0, cx],
                [0.0, focal_length, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _get_camera_matrix(self, frame: np.ndarray) -> np.ndarray:
        """Build a camera matrix from frame dimensions (BGR frame input)."""
        h, w = frame.shape[:2]
        return self._build_camera_matrix(h, w)

    def get_landmarks(self, frame: np.ndarray) -> np.ndarray | None:
        """Get 2D facial landmarks from frame.

        Args:
            frame: BGR frame.

        Returns:
            Array of shape (6, 2) with landmark pixel coordinates, or None.
        """

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        landmarks = np.zeros((len(LANDMARK_INDICES), 2), dtype=np.float64)

        for i, idx in enumerate(LANDMARK_INDICES):
            lm = face_landmarks.landmark[idx]
            landmarks[i] = [lm.x * w, lm.y * h]

        return landmarks

    def estimate_from_landmarks(
        self,
        landmarks_2d: np.ndarray,
        frame_shape: tuple[int, int],
    ) -> tuple[float, float, float] | None:
        """Estimate head pose from already-extracted 2D landmarks.

        Useful when the caller has already called ``get_landmarks`` and does
        not want to re-run MediaPipe Face Mesh.

        Args:
            landmarks_2d: Array of shape (6, 2) matching ``LANDMARK_INDICES``.
            frame_shape: (height, width) of the source frame for building the
                camera matrix.

        Returns:
            (pitch, yaw, roll) in degrees, or None if solvePnP fails.
        """
        h, w = frame_shape
        camera_matrix = self._build_camera_matrix(h, w)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, _ = cv2.solvePnP(
            FACE_MODEL_3D,
            landmarks_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        # ZYX Euler decomposition (OpenCV camera-space convention).
        pitch = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
        yaw = np.arctan2(
            -rotation_mat[2, 0],
            np.sqrt(rotation_mat[2, 1] ** 2 + rotation_mat[2, 2] ** 2),
        )
        roll = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])

        return (
            float(np.degrees(pitch)),
            float(np.degrees(yaw)),
            float(np.degrees(roll)),
        )

    def estimate(self, frame: np.ndarray) -> tuple[float, float, float] | None:
        """Estimate head pose from a BGR frame.

        Convenience wrapper around ``get_landmarks`` +
        ``estimate_from_landmarks``. Callers that also need the landmarks
        themselves (e.g. for frontalization) should call the two methods
        directly to avoid running MediaPipe twice per frame.

        Returns:
            (pitch, yaw, roll) in degrees, or None if no face detected.
            - pitch: positive = looking up, negative = looking down
            - yaw: positive = looking left, negative = looking right
            - roll: positive = tilting right (clockwise), negative = tilting left
        """
        landmarks_2d = self.get_landmarks(frame)
        if landmarks_2d is None:
            return None
        return self.estimate_from_landmarks(landmarks_2d, frame.shape[:2])

    def close(self) -> None:
        """Release resources."""
        self.face_mesh.close()
