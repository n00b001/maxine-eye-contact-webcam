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

import pytest

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
