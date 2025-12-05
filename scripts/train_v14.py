#!/usr/bin/env python3
"""
Alpamayo V14-Lite Training Script
=================================

Online imitation learning from teacher signals (joystick/teleop).
Optimized for Jetson Orin Nano.

Usage:
    # Train with ROS2 data collection
    python3 train_v14.py --mode online

    # Train from saved dataset
    python3 train_v14.py --mode offline --data_dir ./data

    # Export trained model
    python3 train_v14.py --export --checkpoint best.pth
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.v14_lite import AlpamayoV14Lite, V14LiteLoss, model_summary


# =============================================================================
# Data Collection (ROS2)
# =============================================================================

class ROSDataCollector:
    """Collect training data from ROS2 topics"""

    def __init__(
        self,
        lidar_topic: str = '/scan',
        camera_topic: str = '/camera/image_raw',
        teacher_topic: str = '/cmd_vel',
        buffer_size: int = 10000
    ):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

        self.lidar_topic = lidar_topic
        self.camera_topic = camera_topic
        self.teacher_topic = teacher_topic

        # ROS2 imports (optional)
        self.ros_available = False
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import LaserScan, Image
            from geometry_msgs.msg import Twist
            self.ros_available = True
            self.rclpy = rclpy
            self.Node = Node
            self.LaserScan = LaserScan
            self.Image = Image
            self.Twist = Twist
        except ImportError:
            print("ROS2 not available. Use offline mode.")

        # Current data
        self.current_lidar = None
        self.current_camera = None
        self.current_teacher = None

        # BEV params
        self.bev_config = {
            'width': 240,
            'height': 200,
            'resolution': 0.05,
            'x_min': -2.0, 'x_max': 10.0,
            'y_min': -5.0, 'y_max': 5.0
        }

    def laserscan_to_bev(self, scan) -> np.ndarray:
        """Convert LaserScan to BEV grid"""
        cfg = self.bev_config

        # Initialize grids
        occupancy = np.zeros((cfg['height'], cfg['width']), dtype=np.float32)
        intensity = np.zeros((cfg['height'], cfg['width']), dtype=np.float32)

        # Convert to cartesian
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        ranges = np.array(scan.ranges)
        intensities = np.array(scan.intensities) if scan.intensities else np.ones_like(ranges) * 0.5

        # Filter valid
        valid = (ranges > 0.1) & (ranges < 12.0) & np.isfinite(ranges)
        x = ranges[valid] * np.cos(angles[valid])
        y = ranges[valid] * np.sin(angles[valid])
        ints = intensities[valid] if len(intensities) == len(ranges) else np.ones(valid.sum()) * 0.5

        # To grid
        gx = ((x - cfg['x_min']) / cfg['resolution']).astype(int)
        gy = ((y - cfg['y_min']) / cfg['resolution']).astype(int)

        mask = (gx >= 0) & (gx < cfg['width']) & (gy >= 0) & (gy < cfg['height'])
        occupancy[gy[mask], gx[mask]] = 1.0
        intensity[gy[mask], gx[mask]] = ints[mask]

        return np.stack([occupancy, intensity], axis=0)

    def image_to_tensor(self, img_msg) -> np.ndarray:
        """Convert ROS Image to tensor"""
        import cv2
        from cv_bridge import CvBridge
        bridge = CvBridge()

        cv_img = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        cv_img = cv2.resize(cv_img, (320, 240))
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return cv_img.astype(np.float32).transpose(2, 0, 1) / 255.0

    def add_sample(self, lidar_bev: np.ndarray, camera: Optional[np.ndarray],
                   steering: float, speed: float):
        """Add training sample to buffer"""
        sample = {
            'lidar_bev': lidar_bev,
            'camera': camera,
            'steering': steering,
            'speed': speed,
            'timestamp': time.time()
        }
        self.buffer.append(sample)

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get random batch from buffer"""
        if len(self.buffer) < batch_size:
            return None

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]

        lidar_batch = np.stack([s['lidar_bev'] for s in samples])
        steering_batch = np.array([[s['steering']] for s in samples])
        speed_batch = np.array([[s['speed']] for s in samples])

        batch = {
            'lidar_bev': torch.from_numpy(lidar_batch).float(),
            'steering_gt': torch.from_numpy(steering_batch).float(),
            'speed_gt': torch.from_numpy(speed_batch).float()
        }

        # Camera (if available)
        if samples[0]['camera'] is not None:
            camera_batch = np.stack([s['camera'] for s in samples])
            batch['camera'] = torch.from_numpy(camera_batch).float()

        return batch

    def save_buffer(self, path: str):
        """Save buffer to disk"""
        data = list(self.buffer)
        np.savez_compressed(path, data=data)
        print(f"Saved {len(data)} samples to {path}")

    def load_buffer(self, path: str):
        """Load buffer from disk"""
        loaded = np.load(path, allow_pickle=True)
        data = loaded['data']
        self.buffer = deque(data, maxlen=self.buffer_size)
        print(f"Loaded {len(self.buffer)} samples from {path}")


# =============================================================================
# Offline Dataset
# =============================================================================

class V14Dataset(Dataset):
    """Dataset for offline training"""

    def __init__(self, data_dir: str, use_camera: bool = True):
        self.data_dir = Path(data_dir)
        self.use_camera = use_camera

        # Load all samples
        self.samples = []
        for npz_file in self.data_dir.glob('*.npz'):
            data = np.load(npz_file, allow_pickle=True)
            if 'data' in data:
                self.samples.extend(data['data'])

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        item = {
            'lidar_bev': torch.from_numpy(sample['lidar_bev']).float(),
            'steering_gt': torch.tensor([sample['steering']]).float(),
            'speed_gt': torch.tensor([sample['speed']]).float()
        }

        if self.use_camera and sample.get('camera') is not None:
            item['camera'] = torch.from_numpy(sample['camera']).float()

        return item


# =============================================================================
# Trainer
# =============================================================================

class V14Trainer:
    """Training manager for V14-Lite"""

    def __init__(
        self,
        model: AlpamayoV14Lite,
        device: str = 'cuda',
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        checkpoint_dir: str = './checkpoints'
    ):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss
        self.criterion = V14LiteLoss(
            occ_weight=0.5,
            flow_weight=0.3,
            traj_weight=1.0,
            ctrl_weight=2.0
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=100,
            T_mult=2
        )

        # Logging
        self.step = 0
        self.best_loss = float('inf')
        self.history = []

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        # Move to device
        lidar = batch['lidar_bev'].to(self.device)
        camera = batch.get('camera')
        if camera is not None:
            camera = camera.to(self.device)

        targets = {
            'steering_gt': batch['steering_gt'].to(self.device),
            'speed_gt': batch['speed_gt'].to(self.device)
        }

        # Forward
        outputs = self.model(lidar_bev=lidar, camera=camera, apply_safety=False)

        # Loss
        losses = self.criterion(outputs, targets)

        # Backward
        self.optimizer.zero_grad()
        losses['total'].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        self.step += 1

        # Return losses as floats
        return {k: v.item() for k, v in losses.items()}

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation pass"""
        self.model.eval()

        total_losses = {}
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                lidar = batch['lidar_bev'].to(self.device)
                camera = batch.get('camera')
                if camera is not None:
                    camera = camera.to(self.device)

                targets = {
                    'steering_gt': batch['steering_gt'].to(self.device),
                    'speed_gt': batch['speed_gt'].to(self.device)
                }

                outputs = self.model(lidar_bev=lidar, camera=camera, apply_safety=False)
                losses = self.criterion(outputs, targets)

                for k, v in losses.items():
                    total_losses[k] = total_losses.get(k, 0) + v.item()

                num_batches += 1

        return {k: v / num_batches for k, v in total_losses.items()}

    def save_checkpoint(self, name: str = 'latest'):
        """Save model checkpoint"""
        path = self.checkpoint_dir / f'{name}.pth'
        torch.save({
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint.get('history', [])
        print(f"Loaded checkpoint from step {self.step}")

    def train_online(
        self,
        collector: ROSDataCollector,
        num_steps: int = 10000,
        batch_size: int = 16,
        log_interval: int = 100,
        save_interval: int = 1000,
        min_samples: int = 500
    ):
        """Online training with ROS2 data collection"""
        print(f"\nOnline training for {num_steps} steps")
        print(f"Waiting for {min_samples} samples...")

        while len(collector.buffer) < min_samples:
            time.sleep(0.1)

        print(f"Starting training with {len(collector.buffer)} samples")

        for step in range(num_steps):
            batch = collector.get_batch(batch_size)
            if batch is None:
                continue

            losses = self.train_step(batch)

            if step % log_interval == 0:
                print(f"Step {step}: loss={losses['total']:.4f}, "
                      f"steer={losses.get('steering', 0):.4f}, "
                      f"speed={losses.get('speed', 0):.4f}")

            if step % save_interval == 0:
                self.save_checkpoint('latest')
                if losses['total'] < self.best_loss:
                    self.best_loss = losses['total']
                    self.save_checkpoint('best')

    def train_offline(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        log_interval: int = 10
    ):
        """Offline training from dataset"""
        print(f"\nOffline training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            epoch_losses = {}
            num_batches = 0

            for batch in train_loader:
                losses = self.train_step(batch)

                for k, v in losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0) + v
                num_batches += 1

            # Average
            epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            self.history.append({'epoch': epoch, 'train': epoch_losses})

            # Validation
            if val_loader is not None:
                val_losses = self.validate(val_loader)
                self.history[-1]['val'] = val_losses
                print(f"Epoch {epoch}: train={epoch_losses['total']:.4f}, "
                      f"val={val_losses['total']:.4f}")

                if val_losses['total'] < self.best_loss:
                    self.best_loss = val_losses['total']
                    self.save_checkpoint('best')
            else:
                print(f"Epoch {epoch}: loss={epoch_losses['total']:.4f}")

            self.save_checkpoint('latest')


# =============================================================================
# Export
# =============================================================================

def export_model(checkpoint_path: str, output_dir: str = './onnx_models'):
    """Export trained model to ONNX"""
    print(f"\nExporting model from {checkpoint_path}")

    # Load model
    model = AlpamayoV14Lite(use_camera=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    # Dummy inputs
    lidar = torch.randn(1, 2, 200, 240)
    camera = torch.randn(1, 3, 240, 320)
    hidden = torch.zeros(1, 64, 50, 60)

    # Export
    output_path = os.path.join(output_dir, 'v14_lite.onnx')

    torch.onnx.export(
        model,
        (lidar, camera, hidden),
        output_path,
        export_params=True,
        opset_version=17,
        input_names=['lidar_bev', 'camera', 'hidden'],
        output_names=['steering', 'speed', 'new_hidden', 'risk'],
        dynamic_axes={
            'lidar_bev': {0: 'batch'},
            'camera': {0: 'batch'},
            'hidden': {0: 'batch'}
        }
    )

    print(f"Exported to {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024:.1f} KB")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Alpamayo V14-Lite')
    parser.add_argument('--mode', choices=['online', 'offline'], default='offline')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint', type=str, help='Resume from checkpoint')
    parser.add_argument('--export', action='store_true', help='Export model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_camera', action='store_true', default=True)

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Export mode
    if args.export:
        if args.checkpoint:
            export_model(args.checkpoint)
        else:
            print("Error: --checkpoint required for export")
        return

    # Create model
    model = AlpamayoV14Lite(use_camera=args.use_camera)
    info = model_summary(model)
    print(f"\nModel: {info['total']/1000:.1f}K parameters")

    # Trainer
    trainer = V14Trainer(
        model=model,
        device=args.device,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )

    # Resume
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Training mode
    if args.mode == 'online':
        collector = ROSDataCollector()
        if not collector.ros_available:
            print("Error: ROS2 not available for online training")
            return

        # TODO: Start ROS2 node for data collection
        trainer.train_online(collector)

    else:  # offline
        if not os.path.exists(args.data_dir):
            print(f"Error: Data directory not found: {args.data_dir}")
            print("\nTo collect data, use online mode with ROS2")
            print("Or create synthetic data with:")
            print("  python3 scripts/generate_synthetic_data.py")
            return

        dataset = V14Dataset(args.data_dir, use_camera=args.use_camera)

        # Split
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )

        trainer.train_offline(
            train_loader,
            val_loader,
            num_epochs=args.epochs
        )

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
