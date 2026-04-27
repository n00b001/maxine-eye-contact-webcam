"""
Tests for gpu_format_convert.py.

CPU-safe tests always run; CUDA-gated tests are skipped when no GPU is present.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

# gpu_format_convert and its symbols are imported after torch guard above.
import gpu_format_convert  # noqa: E402, I001
from gpu_format_convert import (  # noqa: E402
    IMPL_DEFAULT,
    _IMPL_MAP,
    _impl_nvcv,
    _select_impl,
    rgb_f32_planar_to_bgr_u8_chunky,
)

# ---------------------------------------------------------------------------
# CPU-safe tests (always run)
# ---------------------------------------------------------------------------

HAS_CUDA = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA")


def test_imports_cleanly() -> None:
    """Module must import on a CPU-only machine without raising."""
    import importlib

    importlib.reload(gpu_format_convert)


def test_impl_map_contains_torch_cupy_nvcv() -> None:
    assert set(_IMPL_MAP.keys()) == {"torch", "cupy", "nvcv"}


def test_select_impl_raises_on_unknown_name() -> None:
    with pytest.raises(ValueError, match="bogus"):
        _select_impl("bogus")


def test_select_impl_raises_uses_raise_from_none() -> None:
    """The ValueError must not chain a KeyError (raise … from None)."""
    with pytest.raises(ValueError) as exc_info:
        _select_impl("bogus")
    assert exc_info.value.__cause__ is None


def test_impl_nvcv_raises_not_implemented_error() -> None:
    with pytest.raises(NotImplementedError, match="NvCVImage_Transfer"):
        _impl_nvcv(None, None)


def test_default_impl_is_torch() -> None:
    assert IMPL_DEFAULT == "torch"


def test_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting MAXINE_FORMAT_CONVERT_IMPL to an unknown name raises ValueError.

    The public function reads the env var and forwards it to _select_impl, so
    we verify the dispatch path CPU-safely by calling _select_impl directly
    under the patched environment (mimicking what the public function does).
    """
    monkeypatch.setenv("MAXINE_FORMAT_CONVERT_IMPL", "bogus")
    impl_name = os.environ.get("MAXINE_FORMAT_CONVERT_IMPL", IMPL_DEFAULT)
    with pytest.raises(ValueError, match="bogus"):
        _select_impl(impl_name)


@skip_no_cuda
def test_env_var_override_public_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: env var propagates through the public API on a CUDA machine."""
    monkeypatch.setenv("MAXINE_FORMAT_CONVERT_IMPL", "bogus")
    dummy = torch.zeros(1, 3, 4, 4, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="bogus"):
        rgb_f32_planar_to_bgr_u8_chunky(dummy)


# ---------------------------------------------------------------------------
# CUDA-gated tests
# ---------------------------------------------------------------------------


def _make_probe() -> torch.Tensor:
    """Build the 2×2 probe tensor described in the module's __main__ block."""
    t = torch.zeros(1, 3, 2, 2, device="cuda", dtype=torch.float32)
    t[0, 0, 0, 0] = 1.0  # R=1 at row 0, col 0
    t[0, 1, 0, 1] = 1.0  # G=1 at row 0, col 1
    t[0, 2, 1, 0] = 1.0  # B=1 at row 1, col 0
    return t


@skip_no_cuda
def test_torch_impl_channel_order() -> None:
    """R→(0,0,255), G→(0,255,0), B→(255,0,0) in BGR output."""
    from gpu_format_convert import _impl_torch

    t = _make_probe()
    out = _impl_torch(t, None).cpu().numpy()

    assert tuple(out[0, 0]) == (0, 0, 255), f"pixel (0,0): {out[0, 0]}"
    assert tuple(out[0, 1]) == (0, 255, 0), f"pixel (0,1): {out[0, 1]}"
    assert tuple(out[1, 0]) == (255, 0, 0), f"pixel (1,0): {out[1, 0]}"


@skip_no_cuda
def test_torch_impl_shape_and_dtype() -> None:
    """(1, 3, 7, 5) input → (7, 5, 3) uint8 output."""
    from gpu_format_convert import _impl_torch

    src = torch.rand(1, 3, 7, 5, device="cuda", dtype=torch.float32)
    result = _impl_torch(src, None)

    assert result.shape == (7, 5, 3)
    assert result.dtype == torch.uint8


@skip_no_cuda
def test_torch_impl_clamp_handles_overshoot() -> None:
    """Values above 1.0 must clamp to 255, not wrap or raise."""
    from gpu_format_convert import _impl_torch

    src = torch.full((1, 3, 2, 2), 1.5, device="cuda", dtype=torch.float32)
    result = _impl_torch(src, None)

    assert result.max().item() == 255


@skip_no_cuda
def test_torch_impl_writes_into_out_when_provided() -> None:
    """When an *out* tensor is supplied, its data_ptr is preserved."""
    from gpu_format_convert import _impl_torch

    src = torch.rand(1, 3, 4, 4, device="cuda", dtype=torch.float32)
    out = torch.empty(4, 4, 3, device="cuda", dtype=torch.uint8)
    ptr_before = out.data_ptr()

    returned = _impl_torch(src, out)

    assert returned.data_ptr() == ptr_before
    # The returned tensor should be the same object as out.
    assert returned.data_ptr() == out.data_ptr()
