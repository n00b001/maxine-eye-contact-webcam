#!/usr/bin/env python3
"""
onnx2trt_v10.py — TensorRT 10-compatible ONNX → TRT engine builder.

FLP's vendored scripts/onnx2trt.py targets TensorRT 8.x and uses several
APIs that were removed in TensorRT 10:

  - config.max_workspace_size    (→ config.set_memory_pool_limit)
  - BuilderFlag.STRICT_TYPES     (removed; no direct replacement needed)
  - builder.max_batch_size       (removed; explicit batch is the default)
  - builder.build_engine()       (→ builder.build_serialized_network())
  - The shipped libgrid_sample_3d_plugin.so links against libnvinfer.so.8

This script is a minimal, TRT 10-clean re-implementation of that builder.
Same CLI surface as FLP's script so docker/entrypoint.sh can swap it in
without other changes.

TensorRT 10 supports 3-D grid_sample natively, so no custom plugin is
needed for the warping-spade engine.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import tensorrt as trt

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("onnx2trt_v10")


def build_engine(onnx_path: str, engine_path: str, precision: str, verbose: bool = False) -> None:
    """Parse ONNX and serialize a TensorRT engine to disk."""
    trt_logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    trt.init_libnvinfer_plugins(trt_logger, namespace="")

    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log.error("ONNX parse error %d: %s", i, parser.get_error(i))
            raise RuntimeError(f"ONNX parse failed for {onnx_path}")

    # Log I/O shapes to match FLP's output format.
    for i in range(network.num_inputs):
        t = network.get_input(i)
        log.info("Input  '%s' shape %s dtype %s", t.name, t.shape, t.dtype)
    for i in range(network.num_outputs):
        t = network.get_output(i)
        log.info("Output '%s' shape %s dtype %s", t.name, t.shape, t.dtype)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 * (1 << 30))  # 12 GB

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            log.warning("FP16 not supported natively; building FP32")
        else:
            config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
    elif precision != "fp32":
        raise ValueError(f"unknown precision {precision!r}; expected fp32|fp16|int8")

    log.info("Building %s engine from %s → %s", precision, onnx_path, engine_path)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"build_serialized_network returned None for {onnx_path}")

    os.makedirs(os.path.dirname(os.path.realpath(engine_path)), exist_ok=True)
    with open(engine_path, "wb") as f:
        # IHostMemory supports the buffer protocol; convert to bytes
        # explicitly so file.write receives a bytes object.
        f.write(bytes(serialized))
    log.info("Wrote %d bytes to %s", serialized.nbytes, engine_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--onnx", required=True, help="Path to input .onnx file")
    ap.add_argument("-e", "--engine", required=True, help="Path to output .trt engine")
    ap.add_argument("-p", "--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.onnx):
        log.error("ONNX file not found: %s", args.onnx)
        return 1

    build_engine(args.onnx, args.engine, args.precision, args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
