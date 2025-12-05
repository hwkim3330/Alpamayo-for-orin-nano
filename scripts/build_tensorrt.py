#!/usr/bin/env python3
"""
Alpamayo TensorRT Engine Builder
Converts ONNX models to TensorRT engines optimized for Jetson Orin Nano.

Requirements (on Jetson):
    - JetPack 5.x or 6.x
    - TensorRT (included with JetPack)
    - pycuda (pip install pycuda)

Usage:
    python3 build_tensorrt.py --onnx_dir ./onnx_models --output_dir ./trt_engines
    python3 build_tensorrt.py --onnx ./onnx_models/alpamayo_pipeline.onnx --fp16
"""

import argparse
import os
import sys
from pathlib import Path

# Check if running on Jetson
def check_tensorrt():
    try:
        import tensorrt as trt
        print(f"TensorRT version: {trt.__version__}")
        return True
    except ImportError:
        print("ERROR: TensorRT not found.")
        print("This script must be run on Jetson with JetPack installed.")
        print("\nFor testing on x86, install tensorrt:")
        print("  pip install tensorrt")
        return False


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    int8: bool = False,
    max_batch_size: int = 1,
    workspace_size_gb: float = 1.0,
    verbose: bool = False
):
    """Build TensorRT engine from ONNX model"""
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)

    print(f"\n{'='*50}")
    print(f"Building: {os.path.basename(onnx_path)}")
    print(f"{'='*50}")

    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"  Parsing ONNX model...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("  ERROR: Failed to parse ONNX")
            for i in range(parser.num_errors):
                print(f"    {parser.get_error(i)}")
            return False

    # Print network info
    print(f"  Network inputs:")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"    {inp.name}: {inp.shape}")

    print(f"  Network outputs:")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"    {out.name}: {out.shape}")

    # Create config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace_size_gb * 1024 * 1024 * 1024)
    )

    # Optimization profiles for dynamic shapes
    profile = builder.create_optimization_profile()

    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = list(inp.shape)

        # Handle dynamic batch dimension
        if shape[0] == -1:
            min_shape = [1] + shape[1:]
            opt_shape = [1] + shape[1:]
            max_shape = [max_batch_size] + shape[1:]

            profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
            print(f"  Dynamic shape for {inp.name}: min={min_shape}, opt={opt_shape}, max={max_shape}")

    config.add_optimization_profile(profile)

    # Precision settings
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  Using FP16 precision")
        else:
            print("  Warning: FP16 not supported on this platform")

    if int8:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("  Using INT8 precision (requires calibration)")
            # Note: INT8 requires calibration data
        else:
            print("  Warning: INT8 not supported on this platform")

    # Build engine
    print(f"  Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("  ERROR: Failed to build engine")
        return False

    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    file_size = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"  ✓ Saved: {engine_path} ({file_size:.2f} MB)")

    return True


def build_all_engines(onnx_dir: str, output_dir: str, fp16: bool = True):
    """Build engines for all ONNX files in directory"""
    onnx_files = list(Path(onnx_dir).glob("*.onnx"))

    if not onnx_files:
        print(f"No ONNX files found in {onnx_dir}")
        return

    print(f"Found {len(onnx_files)} ONNX files")

    os.makedirs(output_dir, exist_ok=True)

    results = []
    for onnx_path in onnx_files:
        engine_name = onnx_path.stem + ".engine"
        engine_path = os.path.join(output_dir, engine_name)

        success = build_engine(
            str(onnx_path),
            engine_path,
            fp16=fp16
        )
        results.append((onnx_path.name, success))

    # Summary
    print("\n" + "=" * 50)
    print("Build Summary")
    print("=" * 50)
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")


def main():
    parser = argparse.ArgumentParser(
        description='Build TensorRT engines from ONNX models'
    )
    parser.add_argument('--onnx', type=str,
                        help='Single ONNX file to convert')
    parser.add_argument('--onnx_dir', type=str, default='./onnx_models',
                        help='Directory containing ONNX files')
    parser.add_argument('--output_dir', type=str, default='./trt_engines',
                        help='Output directory for TensorRT engines')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Enable FP16 precision (default: True)')
    parser.add_argument('--int8', action='store_true',
                        help='Enable INT8 precision (requires calibration)')
    parser.add_argument('--workspace', type=float, default=1.0,
                        help='Workspace size in GB (default: 1.0)')
    parser.add_argument('--max_batch', type=int, default=1,
                        help='Maximum batch size (default: 1)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose TensorRT logging')

    args = parser.parse_args()

    # Check TensorRT
    if not check_tensorrt():
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.onnx:
        # Single file
        engine_name = Path(args.onnx).stem + ".engine"
        engine_path = os.path.join(args.output_dir, engine_name)

        build_engine(
            args.onnx,
            engine_path,
            fp16=args.fp16,
            int8=args.int8,
            max_batch_size=args.max_batch,
            workspace_size_gb=args.workspace,
            verbose=args.verbose
        )
    else:
        # All files in directory
        build_all_engines(args.onnx_dir, args.output_dir, fp16=args.fp16)

    print("\nDone! Use inference_trt.py to run inference with TensorRT engines.")


if __name__ == '__main__':
    main()
