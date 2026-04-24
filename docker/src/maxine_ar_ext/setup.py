"""Build script for the maxine_ar_ext pybind11 extension.

Builds a zero-copy shim over NVIDIA Maxine AR SDK GazeRedirection. The
extension relies on the AR SDK installed by the Dockerfile base image at
/usr/local/ARSDK and falls back to the in-repo vendor tree for local
developer builds outside Docker.
"""

from __future__ import annotations

import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

HERE = os.path.dirname(os.path.abspath(__file__))
# Dev-tree fallback: docker/arsdk sits two levels above this file.
DEV_ARSDK = os.path.normpath(os.path.join(HERE, "..", "..", "arsdk"))

# AR SDK install root inside the container (Dockerfile.base copies the
# proprietary tree here). Headers and libs live under this prefix.
ARSDK_ROOT = os.environ.get("ARSDK_ROOT", "/usr/local/ARSDK")

ext_modules = [
    Pybind11Extension(
        "maxine_ar_ext",
        sources=["bindings.cpp"],
        include_dirs=[
            # Container install paths.
            os.path.join(ARSDK_ROOT, "include"),
            os.path.join(ARSDK_ROOT, "features", "nvargazeredirection", "include"),
            # Dev-tree fallbacks so this also builds outside the container.
            os.path.join(DEV_ARSDK, "include"),
            os.path.join(DEV_ARSDK, "features", "nvargazeredirection", "include"),
        ],
        library_dirs=[os.path.join(ARSDK_ROOT, "lib")],
        libraries=["nvARPose", "NVCVImage"],
        cxx_std=17,
        extra_compile_args=["-O3", "-fvisibility=hidden"],
    ),
]

setup(
    name="maxine_ar_ext",
    version="0.1.0",
    description=(
        "Zero-copy pybind11 shim over NVIDIA Maxine AR SDK GazeRedirection"
    ),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.10",
)
