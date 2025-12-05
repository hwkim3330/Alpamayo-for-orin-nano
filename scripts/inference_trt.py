#!/usr/bin/env python3
"""
Alpamayo TensorRT Inference Engine
Optimized inference for Jetson Orin Nano.

Usage:
    # As standalone test
    python3 inference_trt.py --engine ./trt_engines/alpamayo_pipeline.engine --benchmark

    # In your code
    from inference_trt import AlpamayoTRT
    engine = AlpamayoTRT('./trt_engines')
    steering, speed = engine.infer(camera_img, bev_grid)
"""

import argparse
import os
import time
from typing import Tuple, Optional

import numpy as np

# Check for TensorRT
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("Warning: TensorRT/PyCUDA not available. Using stub mode.")


class TRTEngine:
    """Single TensorRT engine wrapper"""

    def __init__(self, engine_path: str):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        print(f"Loading engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)

            # Handle dynamic shapes
            if -1 in shape:
                shape = [1 if s == -1 else s for s in shape]

            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'name': name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype
                })
            else:
                self.outputs.append({
                    'name': name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype
                })

        print(f"  Inputs: {[i['name'] for i in self.inputs]}")
        print(f"  Outputs: {[o['name'] for o in self.outputs]}")

    def infer(self, *inputs) -> list:
        """Run inference"""
        # Copy inputs to device
        for i, inp in enumerate(inputs):
            np.copyto(self.inputs[i]['host'], inp.ravel())
            cuda.memcpy_htod_async(
                self.inputs[i]['device'],
                self.inputs[i]['host'],
                self.stream
            )

        # Set tensor addresses
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, self.bindings[i])

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy outputs from device
        outputs = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        self.stream.synchronize()

        for out in self.outputs:
            outputs.append(out['host'].reshape(out['shape']))

        return outputs


class AlpamayoTRT:
    """
    Alpamayo TensorRT inference wrapper.
    Handles both separate and combined model configurations.
    """

    def __init__(self, engine_dir: str, combined: bool = True):
        """
        Initialize TensorRT inference.

        Args:
            engine_dir: Directory containing .engine files
            combined: Use combined pipeline (True) or separate models (False)
        """
        self.combined = combined

        if combined:
            # Single combined engine
            engine_path = os.path.join(engine_dir, 'alpamayo_pipeline.engine')
            if os.path.exists(engine_path):
                self.pipeline = TRTEngine(engine_path)
            else:
                raise FileNotFoundError(f"Combined engine not found: {engine_path}")

            # Initialize hidden state for ConvGRU
            self.lidar_hidden = np.zeros((1, 32, 50, 60), dtype=np.float32)
        else:
            # Separate engines
            self.camera_cnn = TRTEngine(os.path.join(engine_dir, 'camera_cnn.engine'))
            self.lidar_encoder = TRTEngine(os.path.join(engine_dir, 'lidar_encoder.engine'))
            self.planner = TRTEngine(os.path.join(engine_dir, 'planner_head.engine'))

            self.lidar_hidden = np.zeros((1, 32, 50, 60), dtype=np.float32)

        print("AlpamayoTRT initialized")

    def preprocess_camera(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess camera image.

        Args:
            image: BGR image from OpenCV (H, W, 3)

        Returns:
            Normalized tensor (1, 3, 240, 320)
        """
        import cv2

        # Resize to 320x240
        if image.shape[:2] != (240, 320):
            image = cv2.resize(image, (320, 240))

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        return image[np.newaxis, ...]

    def preprocess_lidar(self, scan: dict) -> np.ndarray:
        """
        Convert LaserScan to BEV grid.

        Args:
            scan: Dictionary with 'ranges', 'angle_min', 'angle_increment', 'intensities'

        Returns:
            BEV grid tensor (1, 2, 200, 240)
        """
        # Grid parameters (matching model)
        grid_width = 240
        grid_height = 200
        resolution = 0.05  # 5cm
        x_min, x_max = -2.0, 10.0
        y_min, y_max = -5.0, 5.0

        # Initialize grids
        occupancy = np.zeros((grid_height, grid_width), dtype=np.float32)
        intensity = np.zeros((grid_height, grid_width), dtype=np.float32)

        ranges = np.array(scan['ranges'])
        angles = scan['angle_min'] + np.arange(len(ranges)) * scan['angle_increment']
        intensities = np.array(scan.get('intensities', np.ones_like(ranges) * 0.5))

        # Convert to cartesian
        valid = (ranges > 0.1) & (ranges < 12.0) & np.isfinite(ranges)
        x = ranges[valid] * np.cos(angles[valid])
        y = ranges[valid] * np.sin(angles[valid])
        ints = intensities[valid]

        # Convert to grid indices
        gx = ((x - x_min) / resolution).astype(int)
        gy = ((y - y_min) / resolution).astype(int)

        # Filter valid indices
        valid_idx = (gx >= 0) & (gx < grid_width) & (gy >= 0) & (gy < grid_height)
        gx = gx[valid_idx]
        gy = gy[valid_idx]
        ints = ints[valid_idx]

        # Fill grids
        occupancy[gy, gx] = 1.0
        intensity[gy, gx] = ints

        # Stack and add batch dimension
        bev = np.stack([occupancy, intensity], axis=0)
        return bev[np.newaxis, ...]

    def infer(
        self,
        camera_image: np.ndarray,
        lidar_scan: dict
    ) -> Tuple[float, float]:
        """
        Run inference.

        Args:
            camera_image: BGR image from OpenCV
            lidar_scan: LaserScan data dictionary

        Returns:
            (steering_angle_rad, speed_mps)
        """
        # Preprocess
        camera_tensor = self.preprocess_camera(camera_image)
        bev_tensor = self.preprocess_lidar(lidar_scan)

        if self.combined:
            # Combined pipeline
            outputs = self.pipeline.infer(camera_tensor, bev_tensor, self.lidar_hidden)
            steering = outputs[0][0, 0]
            speed = outputs[1][0, 0]
            self.lidar_hidden = outputs[2]
        else:
            # Separate models
            cam_features = self.camera_cnn.infer(camera_tensor)[0]
            lidar_out = self.lidar_encoder.infer(bev_tensor, self.lidar_hidden)
            lidar_features = lidar_out[0]
            self.lidar_hidden = lidar_out[1]

            planner_out = self.planner.infer(lidar_features, cam_features)
            steering = planner_out[0][0, 0]
            speed = planner_out[1][0, 0]

        return float(steering), float(speed)

    def reset_hidden(self):
        """Reset LSTM/GRU hidden state"""
        self.lidar_hidden = np.zeros((1, 32, 50, 60), dtype=np.float32)


def benchmark(engine_dir: str, num_iterations: int = 100):
    """Benchmark inference speed"""
    print("\n" + "=" * 50)
    print("Benchmarking Alpamayo TensorRT")
    print("=" * 50)

    # Initialize
    engine = AlpamayoTRT(engine_dir)

    # Create dummy data
    camera = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    lidar = {
        'ranges': np.random.uniform(0.5, 10.0, 360).tolist(),
        'angle_min': 0.0,
        'angle_increment': np.pi * 2 / 360,
        'intensities': np.random.uniform(0.0, 1.0, 360).tolist()
    }

    # Warmup
    print("\nWarmup (10 iterations)...")
    for _ in range(10):
        engine.infer(camera, lidar)

    # Benchmark
    print(f"\nBenchmarking ({num_iterations} iterations)...")
    times = []

    for _ in range(num_iterations):
        start = time.perf_counter()
        steering, speed = engine.infer(camera, lidar)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times = np.array(times)

    print("\nResults:")
    print(f"  Mean:   {times.mean():.2f} ms")
    print(f"  Std:    {times.std():.2f} ms")
    print(f"  Min:    {times.min():.2f} ms")
    print(f"  Max:    {times.max():.2f} ms")
    print(f"  FPS:    {1000 / times.mean():.1f}")

    # Check if we meet 20Hz requirement
    target_ms = 50  # 20Hz = 50ms per frame
    if times.mean() < target_ms:
        print(f"\n✓ Meets 20Hz requirement ({times.mean():.1f}ms < {target_ms}ms)")
    else:
        print(f"\n✗ Does not meet 20Hz requirement ({times.mean():.1f}ms > {target_ms}ms)")


def main():
    parser = argparse.ArgumentParser(description='Alpamayo TensorRT Inference')
    parser.add_argument('--engine_dir', type=str, default='./trt_engines',
                        help='Directory containing TensorRT engines')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Benchmark iterations')

    args = parser.parse_args()

    if not TRT_AVAILABLE:
        print("TensorRT not available. Install on Jetson with JetPack.")
        return

    if args.benchmark:
        benchmark(args.engine_dir, args.iterations)
    else:
        # Demo inference
        engine = AlpamayoTRT(args.engine_dir)

        # Dummy data
        camera = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        lidar = {
            'ranges': np.random.uniform(0.5, 10.0, 360).tolist(),
            'angle_min': 0.0,
            'angle_increment': np.pi * 2 / 360,
        }

        steering, speed = engine.infer(camera, lidar)
        print(f"Steering: {np.degrees(steering):.1f}°, Speed: {speed:.2f} m/s")


if __name__ == '__main__':
    main()
