"""
CHOOSING AN IMPLEMENTATION
===========================

torch: zero extra deps; ~0.5-1.0 ms @ 512²; allocates intermediates.
cupy:  needs cupy-cuda12x; ~0.2-0.4 ms @ 512²; custom kernel, direct to `out`.
nvcv:  needs NvCVImage_Transfer exposed via the maxine_ar_ext pybind shim;
       ~0.1-0.3 ms @ 512².

Choose the implementation by setting IMPL_DEFAULT below, or override at
runtime with the environment variable MAXINE_FORMAT_CONVERT_IMPL.
"""

from __future__ import annotations

import os

import torch

# ---------------------------------------------------------------------------
# Module-level default
# ---------------------------------------------------------------------------

IMPL_DEFAULT: str = "torch"  # "torch" | "cupy" | "nvcv"


# ---------------------------------------------------------------------------
# Private implementations
# ---------------------------------------------------------------------------


def _impl_torch(src: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    """
    Pure-torch implementation.

    Chains mul -> clamp -> cast -> permute -> flip -> contiguous.
    Allocates intermediate tensors; does not require cupy or any extension.
    If *out* is provided the result is copied into it; otherwise returns the
    newly allocated tensor directly.
    """
    # src: (1, 3, H, W) float32 RGB [0, 1]
    result = src.squeeze(0).mul(255.0).clamp_(0, 255).to(torch.uint8)
    # CHW -> HWC, then flip last dim to convert RGB -> BGR
    result = result.permute(1, 2, 0).flip(-1).contiguous()
    if out is not None:
        out.copy_(result)
        return out
    return result


def _impl_cupy(src: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    """
    CuPy RawKernel implementation.

    Fuses permute, channel-flip, scale-and-clamp, and uint8 cast into a
    single CUDA kernel.  One thread per output pixel; three planar reads and
    three interleaved BGR writes.  No shared memory.  Runs on the stream
    passed to the public function via cupy.cuda.ExternalStream so that it
    stays ordered with surrounding PyTorch work on the same CUDA stream.

    Requires: ``uv add cupy-cuda12x``
    """
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError(
            "cupy is required for the 'cupy' implementation. Install it with: uv add cupy-cuda12x"
        ) from exc

    _kernel_src = r"""
extern "C" __global__
void rgb_f32_planar_to_bgr_u8_chunky(
    const float* __restrict__ src,   // (3, H, W) planar, row-major
    unsigned char* __restrict__ dst, // (H, W, 3) interleaved BGR, row-major
    int H,
    int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W;
    if (idx >= total) return;

    int row = idx / W;
    int col = idx % W;
    int plane = H * W;

    // Read R, G, B from the three planes
    float r = src[0 * plane + row * W + col];
    float g = src[1 * plane + row * W + col];
    float b = src[2 * plane + row * W + col];

    // Scale, clamp, cast
    unsigned char ur = (unsigned char)fminf(fmaxf(r * 255.0f, 0.0f), 255.0f);
    unsigned char ug = (unsigned char)fminf(fmaxf(g * 255.0f, 0.0f), 255.0f);
    unsigned char ub = (unsigned char)fminf(fmaxf(b * 255.0f, 0.0f), 255.0f);

    // Write interleaved BGR
    int out_base = (row * W + col) * 3;
    dst[out_base + 0] = ub;
    dst[out_base + 1] = ug;
    dst[out_base + 2] = ur;
}
"""

    _kernel_cache: dict = _impl_cupy.__dict__.setdefault("_kernel_cache", {})
    if "kernel" not in _kernel_cache:
        _kernel_cache["kernel"] = cp.RawKernel(_kernel_src, "rgb_f32_planar_to_bgr_u8_chunky")
    kernel = _kernel_cache["kernel"]

    # src is (1, 3, H, W); squeeze to (3, H, W)
    src_3d = src.squeeze(0).contiguous()
    _, H, W = src_3d.shape  # noqa: N806

    if out is None:
        out = torch.empty((H, W, 3), dtype=torch.uint8, device=src.device)

    src_ptr = cp.cuda.MemoryPointer(
        cp.cuda.UnownedMemory(src_3d.data_ptr(), src_3d.numel() * 4, src_3d),
        0,
    )
    out_ptr = cp.cuda.MemoryPointer(
        cp.cuda.UnownedMemory(out.data_ptr(), out.numel(), out),
        0,
    )

    total_pixels = H * W
    block = 256
    grid = (total_pixels + block - 1) // block

    # Run on the caller's CUDA stream
    current_stream = torch.cuda.current_stream()
    with cp.cuda.ExternalStream(current_stream.cuda_stream):
        kernel(
            (grid,),
            (block,),
            (
                cp.ndarray(shape=(src_3d.numel(),), dtype=cp.float32, memptr=src_ptr),
                cp.ndarray(shape=(out.numel(),), dtype=cp.uint8, memptr=out_ptr),
                H,
                W,
            ),
        )

    return out


def _impl_nvcv(src: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    """
    NvCVImage_Transfer path (not yet implemented).

    What would be needed to implement this:
      1. In ``docker/src/maxine_ar_ext/bindings.cpp``, expose a pybind11
         function such as::

             void nvcv_transfer(
                 uintptr_t src_data_ptr, int src_w, int src_h,
                 NvCVImage_PixelFormat src_fmt, NvCVImage_ComponentType src_ctype,
                 uintptr_t dst_data_ptr, int dst_w, int dst_h,
                 NvCVImage_PixelFormat dst_fmt, NvCVImage_ComponentType dst_ctype,
                 uintptr_t stream_handle
             );

         which internally creates two NvCVImage structs via
         ``NvCVImage_InitView`` (wrapping the external GPU pointers), then
         calls ``NvCVImage_Transfer(&src_img, &dst_img, 1.0f/255.0f,
         (CUstream)stream_handle, nullptr)``.

      2. Rebuild the Docker image / shared library so that
         ``import maxine_ar_ext`` exposes ``maxine_ar_ext.nvcv_transfer``.

      3. Replace this ``NotImplementedError`` with the actual call.

    The SDK handles format conversion, channel reordering, and normalisation
    on its own CUDA stream with no host round-trip.
    """
    raise NotImplementedError(
        "NvCVImage_Transfer path not yet exposed — add it to "
        "docker/src/maxine_ar_ext/bindings.cpp first."
    )


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

_IMPL_MAP = {
    "torch": _impl_torch,
    "cupy": _impl_cupy,
    "nvcv": _impl_nvcv,
}


def _select_impl(name: str):
    """Return the implementation function for *name*, raising ValueError if unknown."""
    try:
        return _IMPL_MAP[name]
    except KeyError:
        raise ValueError(
            f"Unknown format-convert implementation {name!r}. Valid choices: {list(_IMPL_MAP)}"
        ) from None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rgb_f32_planar_to_bgr_u8_chunky(
    src: torch.Tensor,
    out: torch.Tensor | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """
    Convert a (1, 3, H, W) float32 RGB CUDA tensor with values in [0, 1]
    to a (H, W, 3) uint8 BGR CUDA tensor with values in [0, 255].

    All work stays on the given CUDA stream (default: current stream).
    Zero host roundtrip.

    If ``out`` is provided, writes into it; otherwise allocates a new tensor
    on the same device and stream as ``src``.  Output must be row-major
    contiguous (HWC, pitch = W*3).

    Parameters
    ----------
    src:
        Shape (1, 3, H, W), dtype float32, CUDA device, values in [0, 1].
    out:
        Optional pre-allocated (H, W, 3) uint8 CUDA tensor to write into.
        If None, a new tensor is allocated.
    stream:
        CUDA stream on which to schedule work.  Defaults to the current
        PyTorch CUDA stream for ``src``'s device.

    Returns
    -------
    torch.Tensor
        Shape (H, W, 3), dtype uint8, on the same CUDA device as *src*.
    """
    if stream is not None:
        ctx = torch.cuda.stream(stream)
    else:
        ctx = torch.cuda.stream(torch.cuda.current_stream(src.device))

    impl_name = os.environ.get("MAXINE_FORMAT_CONVERT_IMPL", IMPL_DEFAULT)
    impl_fn = _select_impl(impl_name)

    with ctx:
        return impl_fn(src, out)


# TODO(user): pick the implementation you want in IMPL_DEFAULT above.
# If you want to try cupy: `uv add cupy-cuda12x` and set IMPL_DEFAULT = "cupy".
# If you want nvcv, first expose NvCVImage_Transfer from docker/src/maxine_ar_ext/
# bindings.cpp, then set IMPL_DEFAULT = "nvcv".


if __name__ == "__main__":
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("skip: no CUDA")
    # 2x2 probe: pixel (0,0) R=1, G=0, B=0 -> BGR byte should be (0,0,255)
    t = torch.zeros(1, 3, 2, 2, device="cuda", dtype=torch.float32)
    t[0, 0, 0, 0] = 1.0  # R=1 at (0,0)
    t[0, 1, 0, 1] = 1.0  # G=1 at (0,1)
    t[0, 2, 1, 0] = 1.0  # B=1 at (1,0)
    out = rgb_f32_planar_to_bgr_u8_chunky(t)
    assert out.shape == (2, 2, 3) and out.dtype == torch.uint8
    got = out.cpu().numpy()
    assert tuple(got[0, 0]) == (0, 0, 255), f"pixel (0,0) expected BGR=(0,0,255), got {got[0, 0]}"
    assert tuple(got[0, 1]) == (0, 255, 0), f"pixel (0,1) expected BGR=(0,255,0), got {got[0, 1]}"
    assert tuple(got[1, 0]) == (255, 0, 0), f"pixel (1,0) expected BGR=(255,0,0), got {got[1, 0]}"
    print("ok")
