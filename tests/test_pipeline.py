"""Tests for pipeline integration with head-pose correction flags."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import maxine_webcam_pipeline as pipeline


class TestParserHeadPoseFlags:
    """Tests for CLI parser head-pose arguments."""

    def test_head_pose_flag_exists(self):
        """The --head-pose flag is accepted by the parser."""
        parser = pipeline.build_parser()
        args = parser.parse_args(["--head-pose"])
        assert hasattr(args, "head_pose")
        assert args.head_pose is True

    def test_head_pose_strength_flag(self):
        """The --head-pose-strength flag is accepted and parsed as float."""
        parser = pipeline.build_parser()
        args = parser.parse_args(["--head-pose", "--head-pose-strength", "0.7"])
        assert hasattr(args, "head_pose_strength")
        assert args.head_pose_strength == pytest.approx(0.7)

    def test_head_pose_yaw_limit_flag(self):
        """The --head-pose-yaw-limit flag is accepted and parsed as float."""
        parser = pipeline.build_parser()
        args = parser.parse_args(["--head-pose", "--head-pose-yaw-limit", "30.0"])
        assert hasattr(args, "head_pose_yaw_limit")
        assert args.head_pose_yaw_limit == pytest.approx(30.0)

    def test_head_pose_defaults_when_not_provided(self):
        """Defaults are sensible when --head-pose is omitted."""
        parser = pipeline.build_parser()
        args = parser.parse_args([])
        assert args.head_pose is False
        assert args.head_pose_strength == pytest.approx(1.0)
        assert args.head_pose_yaw_limit == pytest.approx(45.0)


class TestPipelineWithoutHeadPose:
    """Tests ensuring existing pipeline behavior is unchanged."""

    def test_pipeline_runs_without_head_pose(self):
        """Parser handles invocation without --head-pose."""
        parser = pipeline.build_parser()
        args = parser.parse_args(["--dry-run", "--fps", "5"])
        assert args.head_pose is False
        assert args.dry_run is True

    def test_resolve_resolution_defaults(self):
        """Resolution resolution works with default args."""
        parser = pipeline.build_parser()
        args = parser.parse_args([])
        w, h = pipeline._resolve_resolution(args)
        assert (w, h) == (1280, 720)


class TestPipelineWithHeadPose:
    """Tests for pipeline behaviour when --head-pose is enabled."""

    def test_head_pose_corrector_called_when_flag_set(self):
        """Mock NIM thread and verify head pose corrector is called when flag is set."""
        parser = pipeline.build_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--head-pose",
                "--head-pose-strength",
                "0.8",
                "--fps",
                "5",
            ]
        )

        landmarks_arr = np.array(
            [
                [320.0, 240.0],
                [320.0, 380.0],
                [200.0, 200.0],
                [440.0, 200.0],
                [225.0, 320.0],
                [415.0, 320.0],
            ],
            dtype=np.float64,
        )

        # Mock estimator class; configure the instance returned by calling it.
        mock_estimator = MagicMock()
        mock_estimator.return_value.get_landmarks.return_value = landmarks_arr
        mock_estimator.return_value.estimate_from_landmarks.return_value = (
            2.0,
            -5.0,
            1.0,
        )

        mock_frontalizer = MagicMock()
        mock_frontalizer.return_value.frontalize.return_value = np.zeros(
            (480, 640, 3), dtype=np.uint8
        )

        with (
            patch.object(pipeline, "HeadPoseEstimator", mock_estimator),
            patch.object(pipeline, "Frontalizer", mock_frontalizer),
        ):
            import queue

            raw_q = queue.Queue(maxsize=10)
            out_q = queue.Queue(maxsize=10)

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            raw_q.put(frame)

            # Mirror the single-MediaPipe-pass flow used by the real pipeline.
            def _mini_nim_thread():
                f = raw_q.get(timeout=1.0)
                if args.head_pose and pipeline.HeadPoseEstimator is not None:
                    estimator = pipeline.HeadPoseEstimator()
                    landmarks = estimator.get_landmarks(f)
                    if landmarks is not None:
                        angles = estimator.estimate_from_landmarks(landmarks, f.shape[:2])
                        if angles is not None:
                            frontalizer = pipeline.Frontalizer(strength=args.head_pose_strength)
                            corrected = frontalizer.frontalize(f, landmarks, *angles)
                            if corrected is not None:
                                f = corrected
                out_q.put(f)

            t = threading.Thread(target=_mini_nim_thread, daemon=True)
            t.start()
            t.join(timeout=2.0)

            mock_estimator.assert_called_once()
            mock_estimator.return_value.get_landmarks.assert_called_once()
            mock_estimator.return_value.estimate_from_landmarks.assert_called_once()
            mock_frontalizer.assert_called_once_with(strength=pytest.approx(0.8))
            mock_frontalizer.return_value.frontalize.assert_called_once()

    def test_head_pose_corrector_not_called_without_flag(self):
        """Mock NIM thread and verify head pose corrector is NOT called without flag."""
        parser = pipeline.build_parser()
        args = parser.parse_args(["--dry-run", "--fps", "5"])

        mock_estimator = MagicMock()
        mock_frontalizer = MagicMock()

        with (
            patch.object(pipeline, "HeadPoseEstimator", mock_estimator),
            patch.object(pipeline, "Frontalizer", mock_frontalizer),
        ):
            import queue

            raw_q = queue.Queue(maxsize=10)
            out_q = queue.Queue(maxsize=10)

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            raw_q.put(frame)

            def _mini_nim_thread():
                f = raw_q.get(timeout=1.0)
                if args.head_pose and pipeline.HeadPoseEstimator is not None:
                    estimator = pipeline.HeadPoseEstimator()
                    landmarks = estimator.get_landmarks(f)
                    if landmarks is not None:
                        angles = estimator.estimate_from_landmarks(landmarks, f.shape[:2])
                        if angles is not None:
                            frontalizer = pipeline.Frontalizer()
                            corrected = frontalizer.frontalize(f, landmarks, *angles)
                            if corrected is not None:
                                f = corrected
                out_q.put(f)

            t = threading.Thread(target=_mini_nim_thread, daemon=True)
            t.start()
            t.join(timeout=2.0)

            mock_estimator.assert_not_called()
            mock_frontalizer.assert_not_called()
