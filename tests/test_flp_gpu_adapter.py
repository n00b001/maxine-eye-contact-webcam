"""
Tests for flp_gpu_adapter.py.

The vendor tree (FasterLivePortrait) requires torchgeometry and TRT engines
that are not available on the host CI machine.  We stub out the missing pieces
before importing the adapter so that import-time side-effects can be tested
without the full Docker environment.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Stub the vendor dependencies that are absent on the host / CI.
# We must insert these stubs into sys.modules BEFORE importing flp_gpu_adapter,
# because it does the vendor sys.path insert and then immediately imports from
# the vendor tree at module level.
# ---------------------------------------------------------------------------


def _install_vendor_stubs() -> None:
    """Install lightweight stubs for every module the adapter imports at top-level."""
    # omegaconf stub
    if "omegaconf" not in sys.modules:
        oc = types.ModuleType("omegaconf")
        oc.OmegaConf = mock.MagicMock()  # type: ignore[attr-defined]
        sys.modules["omegaconf"] = oc

    # Build a stub module hierarchy for src.pipelines.faster_live_portrait_pipeline
    for name in [
        "src",
        "src.pipelines",
        "src.pipelines.faster_live_portrait_pipeline",
    ]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    # The adapter captures `paste_back_pytorch` from the pipeline module at
    # import time, so we must expose it as a callable.
    flp_pipe_mod = sys.modules["src.pipelines.faster_live_portrait_pipeline"]
    if not hasattr(flp_pipe_mod, "paste_back_pytorch"):
        flp_pipe_mod.paste_back_pytorch = mock.MagicMock(  # type: ignore[attr-defined]
            name="paste_back_pytorch"
        )

    # The adapter also imports FasterLivePortraitPipeline from the same package.
    if not hasattr(flp_pipe_mod, "FasterLivePortraitPipeline"):
        flp_pipe_mod.FasterLivePortraitPipeline = mock.MagicMock(  # type: ignore[attr-defined]
            name="FasterLivePortraitPipeline"
        )

    # Make `from src.pipelines.faster_live_portrait_pipeline import X` work.
    sys.modules["src.pipelines"].faster_live_portrait_pipeline = flp_pipe_mod  # type: ignore[attr-defined]


_install_vendor_stubs()

# Now it is safe to import the adapter.
import flp_gpu_adapter  # noqa: E402
from flp_gpu_adapter import FLPFrontalizer, _intercepting_paste_back, pull_to_numpy  # noqa: E402

# ---------------------------------------------------------------------------
# CPU-safe tests (always run)
# ---------------------------------------------------------------------------


def test_module_imports() -> None:
    """flp_gpu_adapter must be importable (stubs already installed above)."""
    # If the import succeeded at module level, the module is in sys.modules.
    assert "flp_gpu_adapter" in sys.modules


def test_paste_back_monkey_patch_installed() -> None:
    """The adapter must have replaced paste_back_pytorch in the pipeline module."""
    import src.pipelines.faster_live_portrait_pipeline as pipe_mod

    # The binding in the pipeline module's namespace must now be our interceptor.
    assert pipe_mod.paste_back_pytorch is _intercepting_paste_back

    # The original (stub) must have been saved and must differ.
    assert flp_gpu_adapter._original_paste_back is not _intercepting_paste_back


def test_frontalizer_init_without_source_raises_on_frontalize() -> None:
    """FLPFrontalizer.frontalize() raises RuntimeError when no source image set."""
    import numpy as np

    with (
        mock.patch("flp_gpu_adapter.FasterLivePortraitPipeline") as mock_pipe,
        mock.patch("flp_gpu_adapter.OmegaConf") as mock_cfg,
    ):
        mock_cfg.load.return_value = mock.MagicMock()
        mock_pipe.return_value = mock.MagicMock()

        # Patch torch.cuda.set_device so __init__ doesn't touch a real GPU.
        with mock.patch("torch.cuda.set_device"):
            frontalizer = FLPFrontalizer.__new__(FLPFrontalizer)
            # Manually set the internal state that __init__ would set,
            # but leave _src_info as None (i.e., no source loaded).
            frontalizer._src_info = None
            frontalizer._no_face_streak = 0
            frontalizer._no_face_log_interval = 30

    dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="Source image not set"):
        frontalizer.frontalize(dummy_frame)


def test_pull_to_numpy_debug_helper_exists() -> None:
    """pull_to_numpy is importable and is callable."""
    assert callable(pull_to_numpy)


def test_install_motion_smoothing_calls():
    mock_pipe = mock.MagicMock()
    mock_ex = mock.MagicMock()
    mock_ex._smoothed = False
    mock_pipe.model_dict = {"motion_extractor": mock_ex}

    # Test with both 1.0 (no smoothing)
    flp_gpu_adapter._install_motion_smoothing(mock_pipe, 1.0, 1.0)
    assert not getattr(mock_ex, "_smoothed", False)

    # Test with smoothing
    flp_gpu_adapter._install_motion_smoothing(mock_pipe, 0.5, 0.8)
    assert mock_ex._smoothed


def test_smoothing_logic():
    mock_pipe = mock.MagicMock()
    mock_ex = mock.MagicMock()
    mock_ex._smoothed = False
    original_predict = mock.MagicMock()

    # (pitch, yaw, roll, t, exp, scale, kp, R)
    p0 = torch.tensor([0.0])
    y0 = torch.tensor([0.0])
    r0 = torch.tensor([0.0])
    t0 = torch.tensor([0.0, 0.0, 0.0])
    e0 = torch.tensor([0.0])
    s0 = torch.tensor([1.0])
    k0 = torch.tensor([0.0])
    r0_mat = torch.eye(3).unsqueeze(0)

    original_predict.return_value = (p0, y0, r0, t0, e0, s0, k0, r0_mat)
    mock_ex.predict = original_predict
    mock_pipe.model_dict = {"motion_extractor": mock_ex}

    flp_gpu_adapter._motion_prev[0] = None
    flp_gpu_adapter._install_motion_smoothing(mock_pipe, 0.5, 1.0)

    # First call sets prev
    res1 = mock_ex.predict()
    for i in range(len(res1)):
        assert torch.equal(flp_gpu_adapter._motion_prev[0][i], res1[i])

    # Second call smooths
    p1 = torch.tensor([1.0])
    e1 = torch.tensor([1.0])
    original_predict.return_value = (p1, y0, r0, t0, e1, s0, k0, r0_mat)

    res2 = mock_ex.predict()
    # Pose alpha = 0.5: 0.5 * 1.0 + 0.5 * 0.0 = 0.5
    assert res2[0].item() == 0.5
    # Exp alpha = 1.0: 1.0 * 1.0 + 0.0 * 0.0 = 1.0
    assert res2[4].item() == 1.0


def test_headpose_predict_to_rotation_matrix():
    hp = torch.tensor([0.0, 0.0, 0.0])
    r_mat = flp_gpu_adapter.headpose_predict_to_rotation_matrix(hp)
    assert torch.allclose(r_mat, torch.eye(3).unsqueeze(0))

    hp2 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    r2_mat = flp_gpu_adapter.headpose_predict_to_rotation_matrix(hp2)
    assert r2_mat.shape == (2, 3, 3)


def test_capture_source_pose_captures_m_c2o():
    mock_pipe = mock.MagicMock()
    mock_pipe.src_infos = [
        [
            {
                "pitch": np.array([0.1]),
                "yaw": np.array([0.2]),
                "roll": np.array([0.3]),
                "M_c2o": np.eye(3)[:2],
            }
        ]
    ]
    flp_gpu_adapter._capture_source_pose(mock_pipe)
    assert flp_gpu_adapter._source_m_c2o[0] is not None
    assert np.array_equal(flp_gpu_adapter._source_m_c2o[0], np.eye(3)[:2])


def test_install_axis_strength():
    mock_pipe = mock.MagicMock()
    mock_ex = mock.MagicMock()
    mock_ex._axis_blended = False
    mock_pipe.model_dict = {"motion_extractor": mock_ex}

    original_pred = mock.MagicMock()
    mock_ex.predict = original_pred

    flp_gpu_adapter._install_axis_strength(mock_pipe, (0.5, 0.5, 0.5))
    assert mock_ex._axis_blended
    assert mock_ex.predict != original_pred


def test_axis_strength_logic():
    mock_pipe = mock.MagicMock()
    mock_ex = mock.MagicMock()
    mock_ex._axis_blended = False
    original_predict = mock.MagicMock()

    res_driving = (
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        torch.eye(3).unsqueeze(0),
    )
    original_predict.return_value = res_driving
    mock_ex.predict = original_predict
    mock_pipe.model_dict = {"motion_extractor": mock_ex}

    flp_gpu_adapter._source_pose[0] = (np.array([0.0]), np.array([0.0]), np.array([0.0]))
    flp_gpu_adapter._install_axis_strength(mock_pipe, (0.5, 0.5, 0.5))

    res = mock_ex.predict()
    assert res[0].item() == 0.5
    assert res[1].item() == 0.5
    assert res[2].item() == 0.5


def test_intercepting_paste_back():
    img_crop = torch.zeros((1, 3, 64, 64))
    m_c2o = torch.eye(3)[:2].unsqueeze(0)
    img_ori = torch.zeros((1, 3, 128, 128))
    mask_ori = torch.zeros((1, 1, 128, 128))

    with mock.patch("flp_gpu_adapter._original_paste_back") as mock_orig:
        mock_orig.return_value = "success"

        # Case 1: Overlay mode, no driving M_c2o. Should use M_c2o from arguments.
        flp_gpu_adapter._is_overlay[0] = True
        flp_gpu_adapter._driving_m_c2o[0] = None
        flp_gpu_adapter._target_background[0] = None
        res = flp_gpu_adapter._intercepting_paste_back(img_crop, m_c2o, img_ori, mask_ori)
        assert res == "success"
        mock_orig.assert_called_with(img_crop, m_c2o, img_ori, mask_ori)

        # Case 2: Overlay mode with driving M_c2o. Should override.
        driving_m = np.eye(3)[:2].reshape(1, 2, 3) * 2.0
        flp_gpu_adapter._driving_m_c2o[0] = driving_m
        flp_gpu_adapter._intercepting_paste_back(img_crop, m_c2o, img_ori, mask_ori)
        args, _kwargs = mock_orig.call_args
        assert torch.allclose(args[1], torch.from_numpy(driving_m))

        # Case 3: Overwrite mode. Should use source M_c2o if available.
        source_m = np.eye(3)[:2].reshape(1, 2, 3) * 3.0
        flp_gpu_adapter._is_overlay[0] = False
        flp_gpu_adapter._source_m_c2o[0] = source_m
        flp_gpu_adapter._intercepting_paste_back(img_crop, m_c2o, img_ori, mask_ori)
        args, _kwargs = mock_orig.call_args
        assert torch.allclose(args[1], torch.from_numpy(source_m))


def test_frontalizer_frontalize_path():
    # Mock cv2
    mock_cv2 = mock.MagicMock()
    with (
        mock.patch.dict(sys.modules, {"cv2": mock_cv2}),
        mock.patch("flp_gpu_adapter.FasterLivePortraitPipeline") as mock_pipe_class,
        mock.patch("flp_gpu_adapter.OmegaConf"),
        mock.patch("torch.cuda.set_device"),
    ):
        mock_pipe = mock_pipe_class.return_value
        mock_pipe.prepare_source.return_value = True
        mock_pipe.src_infos = [
            [{"pitch": np.array([0.0]), "yaw": np.array([0.0]), "roll": np.array([0.0])}]
        ]
        mock_cv2.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.float32)

        # Reset stashed state
        flp_gpu_adapter._last_paste_back[0] = None

        frontalizer = flp_gpu_adapter.FLPFrontalizer(src_image_path="dummy.jpg")

        # Mock pipe.run to set _last_paste_back[0] as side effect
        def mock_run(frame, bg, info):
            flp_gpu_adapter._last_paste_back[0] = torch.tensor([1.0])
            return (None, None, "not_none", None)

        mock_pipe.run.side_effect = mock_run

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Test overlay=True
        res = frontalizer.frontalize(dummy_frame, overlay=True)
        assert res is not None
        assert flp_gpu_adapter._is_overlay[0] is True
        # Background should be based on dummy_frame (converted to RGB)
        assert flp_gpu_adapter._target_background[0].shape == (100, 100, 3)

        # Test overlay=False
        res = frontalizer.frontalize(dummy_frame, overlay=False)
        assert res is not None
        assert flp_gpu_adapter._is_overlay[0] is False
        # Background should be the source image
        assert flp_gpu_adapter._target_background[0] is frontalizer._img_src
