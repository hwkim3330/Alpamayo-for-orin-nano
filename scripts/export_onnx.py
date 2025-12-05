#!/usr/bin/env python3
"""
Alpamayo ONNX Export Script
Exports CameraCNN, LiDARTemporalEncoder, and PlannerHead as ONNX models.
Optimized for Jetson Orin Nano deployment.

Usage:
    python3 export_onnx.py --output_dir ./onnx_models
    python3 export_onnx.py --combined  # Export as single combined model
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.camera_cnn import CameraCNN
from models.lidar_temporal import LiDARTemporalEncoder
from models.planner_head import PlannerHead, FullPipeline


def export_camera_cnn(output_dir: str, opset_version: int = 17):
    """Export CameraCNN to ONNX"""
    print("\n[1/3] Exporting CameraCNN...")

    model = CameraCNN(
        in_channels=3,
        feature_dim=64,
        input_size=(320, 240)  # SD resolution
    )
    model.eval()

    # Dummy input: (batch, channels, height, width)
    dummy_input = torch.randn(1, 3, 240, 320)

    output_path = os.path.join(output_dir, "camera_cnn.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['camera_image'],
        output_names=['camera_features'],
        dynamic_axes={
            'camera_image': {0: 'batch_size'},
            'camera_features': {0: 'batch_size'}
        }
    )

    # Verify
    file_size = os.path.getsize(output_path) / 1024
    print(f"   ✓ Saved: {output_path} ({file_size:.1f} KB)")
    print(f"   Input:  camera_image [B, 3, 240, 320]")
    print(f"   Output: camera_features [B, 64]")

    return output_path


def export_lidar_encoder(output_dir: str, opset_version: int = 17):
    """Export LiDARTemporalEncoder to ONNX"""
    print("\n[2/3] Exporting LiDARTemporalEncoder...")

    model = LiDARTemporalEncoder(
        grid_size=(240, 200),
        in_channels=2,  # occupancy + intensity
        hidden_channels=32,
        feature_dim=32
    )
    model.eval()

    # Dummy input: (batch, channels, height, width)
    # BEV grid: 240x200, 5cm resolution
    dummy_bev = torch.randn(1, 2, 200, 240)

    # For GRU, we need hidden state
    dummy_hidden = torch.zeros(1, 32, 50, 60)  # Downsampled by 4

    output_path = os.path.join(output_dir, "lidar_encoder.onnx")

    # Export with hidden state
    torch.onnx.export(
        model,
        (dummy_bev, dummy_hidden),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['bev_grid', 'hidden_state'],
        output_names=['lidar_features', 'new_hidden_state'],
        dynamic_axes={
            'bev_grid': {0: 'batch_size'},
            'hidden_state': {0: 'batch_size'},
            'lidar_features': {0: 'batch_size'},
            'new_hidden_state': {0: 'batch_size'}
        }
    )

    file_size = os.path.getsize(output_path) / 1024
    print(f"   ✓ Saved: {output_path} ({file_size:.1f} KB)")
    print(f"   Input:  bev_grid [B, 2, 200, 240]")
    print(f"   Input:  hidden_state [B, 32, 50, 60]")
    print(f"   Output: lidar_features [B, 32]")

    return output_path


def export_planner_head(output_dir: str, opset_version: int = 17):
    """Export PlannerHead to ONNX"""
    print("\n[3/3] Exporting PlannerHead...")

    model = PlannerHead(
        lidar_dim=32,
        cam_dim=64,
        hidden_dim=64
    )
    model.eval()

    # Dummy inputs
    dummy_lidar_feat = torch.randn(1, 32)
    dummy_cam_feat = torch.randn(1, 64)

    output_path = os.path.join(output_dir, "planner_head.onnx")

    torch.onnx.export(
        model,
        (dummy_lidar_feat, dummy_cam_feat),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['lidar_features', 'camera_features'],
        output_names=['steering', 'speed', 'uncertainty'],
        dynamic_axes={
            'lidar_features': {0: 'batch_size'},
            'camera_features': {0: 'batch_size'},
            'steering': {0: 'batch_size'},
            'speed': {0: 'batch_size'},
            'uncertainty': {0: 'batch_size'}
        }
    )

    file_size = os.path.getsize(output_path) / 1024
    print(f"   ✓ Saved: {output_path} ({file_size:.1f} KB)")
    print(f"   Input:  lidar_features [B, 32]")
    print(f"   Input:  camera_features [B, 64]")
    print(f"   Output: steering [B, 1], speed [B, 1], uncertainty [B, 1]")

    return output_path


def export_combined_pipeline(output_dir: str, opset_version: int = 17):
    """Export full pipeline as single ONNX model"""
    print("\n[*] Exporting Combined Pipeline...")

    model = FullPipeline(
        camera_size=(320, 240),
        bev_size=(240, 200),
        cam_dim=64,
        lidar_dim=32
    )
    model.eval()

    # Dummy inputs
    dummy_camera = torch.randn(1, 3, 240, 320)
    dummy_bev = torch.randn(1, 2, 200, 240)
    dummy_hidden = torch.zeros(1, 32, 50, 60)

    output_path = os.path.join(output_dir, "alpamayo_pipeline.onnx")

    torch.onnx.export(
        model,
        (dummy_camera, dummy_bev, dummy_hidden),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['camera_image', 'bev_grid', 'lidar_hidden'],
        output_names=['steering', 'speed', 'new_hidden'],
        dynamic_axes={
            'camera_image': {0: 'batch_size'},
            'bev_grid': {0: 'batch_size'},
            'lidar_hidden': {0: 'batch_size'},
            'steering': {0: 'batch_size'},
            'speed': {0: 'batch_size'},
            'new_hidden': {0: 'batch_size'}
        }
    )

    file_size = os.path.getsize(output_path) / 1024
    print(f"   ✓ Saved: {output_path} ({file_size:.1f} KB)")
    print(f"\n   Inputs:")
    print(f"     - camera_image [B, 3, 240, 320]")
    print(f"     - bev_grid [B, 2, 200, 240]")
    print(f"     - lidar_hidden [B, 32, 50, 60]")
    print(f"   Outputs:")
    print(f"     - steering [B, 1]")
    print(f"     - speed [B, 1]")
    print(f"     - new_hidden [B, 32, 50, 60]")

    return output_path


def verify_onnx(onnx_path: str):
    """Verify ONNX model"""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"   ✓ ONNX verification passed: {os.path.basename(onnx_path)}")
        return True
    except ImportError:
        print("   ⚠ onnx package not installed, skipping verification")
        return True
    except Exception as e:
        print(f"   ✗ ONNX verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Export Alpamayo models to ONNX')
    parser.add_argument('--output_dir', type=str, default='./onnx_models',
                        help='Output directory for ONNX files')
    parser.add_argument('--combined', action='store_true',
                        help='Export as single combined model')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version (default: 17)')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify exported models')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 50)
    print("Alpamayo ONNX Export")
    print("=" * 50)
    print(f"Output directory: {args.output_dir}")
    print(f"ONNX opset version: {args.opset}")

    exported = []

    if args.combined:
        # Export combined pipeline
        path = export_combined_pipeline(args.output_dir, args.opset)
        exported.append(path)
    else:
        # Export individual models
        exported.append(export_camera_cnn(args.output_dir, args.opset))
        exported.append(export_lidar_encoder(args.output_dir, args.opset))
        exported.append(export_planner_head(args.output_dir, args.opset))

    # Verify
    if args.verify:
        print("\n" + "=" * 50)
        print("Verifying exported models...")
        for path in exported:
            verify_onnx(path)

    # Summary
    print("\n" + "=" * 50)
    print("Export Summary")
    print("=" * 50)

    total_size = 0
    for path in exported:
        size = os.path.getsize(path) / 1024
        total_size += size
        print(f"  {os.path.basename(path)}: {size:.1f} KB")

    print(f"\n  Total: {total_size:.1f} KB")
    print("\nNext steps:")
    print("  1. Copy ONNX files to Jetson Orin Nano")
    print("  2. Run build_tensorrt.py to convert to TensorRT engines")
    print("  3. Use inference_trt.py for optimized inference")


if __name__ == '__main__':
    main()
