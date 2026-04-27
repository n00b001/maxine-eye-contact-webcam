# maxine_ar_ext

Zero-copy pybind11 shim over NVIDIA Maxine AR SDK's `GazeRedirection`
feature. The shim hands a caller-owned CUDA device pointer and pitch
straight to `NvAR_Run`, with no H2D/D2H copies at the Python<->SDK
boundary and no allocation of image buffers inside the shim.

## What it is

- A single extension module, `maxine_ar_ext`, exposing one class:
  `GazeRedirect`.
- Per-frame input/output frames are bound via `NvCVImage_Init` only
  (never `NvCVImage_Alloc`, never `NvCVImage_Transfer`). The caller owns
  the GPU buffers and the CUDA stream.
- Host-resident SDK outputs (landmarks, gaze vector, head pose,
  bounding boxes, etc.) are kept on the C++ side in `std::vector` /
  POD members, mirroring `docker/src/gazeEngine.cpp`.

## Build (inside the Docker image)

From the container, with the AR SDK already copied to
`/usr/local/ARSDK` by `Dockerfile.base`:

```
cd /app/src/maxine_ar_ext
pip install .
```

Or for an in-place editable build:

```
pip install -e .
```

Override the SDK install prefix by setting `ARSDK_ROOT` before building
(defaults to `/usr/local/ARSDK`):

```
ARSDK_ROOT=/opt/ARSDK pip install .
```

The extension links against `libnvARPose` and `libNVCVImage`. Make sure
`LD_LIBRARY_PATH` includes the AR SDK lib dirs at runtime (the
Dockerfile already sets this).

## Use from Python

```python
import maxine_ar_ext

gaze = maxine_ar_ext.GazeRedirect(
    width=640,
    height=480,
    model_dir="/usr/local/ARSDK/lib/models",
    cuda_stream_ptr=my_stream_ptr,   # int cast of CUstream; 0 => create one
    num_landmarks=126,
    gaze_redirect=True,
)

ok = gaze.run(
    in_dptr=in_gpu_ptr, in_pitch=in_row_bytes,
    out_dptr=out_gpu_ptr, out_pitch=out_row_bytes,
)

if ok:
    pitch, yaw = gaze.gaze_vector
```

`run()` returns `False` when average landmark confidence is below the
no-face threshold; all other SDK errors raise `RuntimeError`.

## Linking

- Headers: `/usr/local/ARSDK/include` and
  `/usr/local/ARSDK/features/nvargazeredirection/include`; also
  `../../arsdk/{include,features/nvargazeredirection/include}` for dev
  builds against the vendored copy.
- Libraries: `/usr/local/ARSDK/lib` with `-lnvARPose -lNVCVImage`.
- Compile flags: `-std=c++17 -O3 -fvisibility=hidden` (mirrors the
  Dockerfile's existing `maxine_ar_webcam` build flags).
