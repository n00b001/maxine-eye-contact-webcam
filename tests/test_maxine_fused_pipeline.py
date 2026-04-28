"""
Tests for maxine_fused_pipeline.py.

All tests are CPU-safe.  We verify arg-parser shape and that importing the
module does not trigger subprocess spawns or GPU allocation as a side-effect.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# Import the module under test.
#
# maxine_fused_pipeline.py imports `torch` and `numpy` at top level, both of
# which are safe on a CPU-only machine.  It does NOT call subprocess.Popen or
# allocate GPU tensors at import time — all of that happens inside run().
# ---------------------------------------------------------------------------


def _import_pipeline():
    """Return the maxine_fused_pipeline module, importing it fresh if needed."""
    if "maxine_fused_pipeline" in sys.modules:
        return sys.modules["maxine_fused_pipeline"]
    import maxine_fused_pipeline

    return maxine_fused_pipeline


# ---------------------------------------------------------------------------
# CPU-safe tests
# ---------------------------------------------------------------------------


def test_entry_point_imports_without_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the module must NOT call subprocess.Popen."""
    # Remove from cache so we get a fresh import.
    sys.modules.pop("maxine_fused_pipeline", None)

    popen_calls: list = []

    original_popen = subprocess.Popen

    def _guarded_popen(*args, **kwargs):
        popen_calls.append(args)
        raise AssertionError(f"subprocess.Popen called at import time with: {args!r}")

    monkeypatch.setattr(subprocess, "Popen", _guarded_popen)

    import maxine_fused_pipeline  # noqa: F401  (import for side-effect check)

    monkeypatch.setattr(subprocess, "Popen", original_popen)

    assert popen_calls == [], "Popen was called during import"

    # Clean up so other tests get the real module.
    sys.modules.pop("maxine_fused_pipeline", None)


def test_arg_parser_required_args_present() -> None:
    """_build_parser() must register the expected arguments."""
    mod = _import_pipeline()
    parser = mod._build_parser()

    # Collect all option strings registered with the parser.
    registered: set[str] = set()
    for action in parser._actions:
        registered.update(action.option_strings)

    expected = {
        "--input-device",
        "--output-device",
        "--width",
        "--height",
        "--fps",
        "--src-image",
        "--model-dir",
        "--cfg",
        "--flp-pose-ema",
        "--flp-exp-ema",
        "--flp-motion-ema",
    }
    missing = expected - registered
    assert not missing, f"Parser is missing arguments: {missing}"


def test_arg_parser_defaults() -> None:
    """Verify key argument defaults match documented values."""
    mod = _import_pipeline()
    parser = mod._build_parser()

    # parse_args with only the required --src-image supplied.
    args = parser.parse_args(["--src-image", "dummy.jpg"])

    assert args.input_device == "/dev/video0"
    assert args.output_device == "/dev/video10"
    assert args.width == 1920
    assert args.height == 1080
    assert args.fps == 30
    assert args.model_dir == "/usr/local/ARSDK/lib/models"
    assert args.cfg == "vendor/FasterLivePortrait/configs/trt_infer.yaml"
    assert args.flp_pose_ema == 0.1
    assert args.flp_exp_ema == 1.0
    assert args.flp_motion_ema == 0.3


def test_arg_parser_src_image_is_required() -> None:
    """--src-image must be required (no default)."""
    mod = _import_pipeline()
    parser = mod._build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([])

    assert exc_info.value.code != 0


def test_build_parser_returns_argument_parser() -> None:
    """_build_parser() must return an ArgumentParser instance."""
    import argparse

    mod = _import_pipeline()
    parser = mod._build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
