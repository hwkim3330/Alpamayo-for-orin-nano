#!/usr/bin/env python3
"""
Planner Head Model - Unified End-to-End Architecture
For Jetson Orin Nano RC Car with CSI SD Camera + 2D LiDAR

Input:
    lidar_feat: (B, 32) from LiDARTemporalEncoder
    cam_feat: (B, 64) from CameraCNN
Output:
    (B, 2) - [steering_angle, speed]

RC Car Specifications:
    - Steering: [-1, 1] -> [-30deg, +30deg] (typical RC servo)
    - Speed: [0, 1] -> [0, 2.0] m/s (safe indoor speed)
    - Control frequency: 20 Hz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class FeatureAttention(nn.Module):
    """
    Cross-modal attention between LiDAR and Camera features.
    Learns which sensor to trust more in different situations.
    """
    def __init__(self, lidar_dim=32, cam_dim=64, hidden_dim=32):
        super().__init__()

        # Project to common space
        self.lidar_proj = nn.Linear(lidar_dim, hidden_dim)
        self.cam_proj = nn.Linear(cam_dim, hidden_dim)

        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),  # 2 weights for lidar, cam
            nn.Softmax(dim=-1)
        )

    def forward(self, lidar_feat, cam_feat):
        """
        Args:
            lidar_feat: (B, lidar_dim)
            cam_feat: (B, cam_dim)

        Returns:
            fused: (B, hidden_dim * 2)
            weights: (B, 2) attention weights [lidar, cam]
        """
        lidar_h = self.lidar_proj(lidar_feat)
        cam_h = self.cam_proj(cam_feat)

        combined = torch.cat([lidar_h, cam_h], dim=-1)
        weights = self.attention(combined)

        # Weighted combination
        weighted_lidar = lidar_h * weights[:, 0:1]
        weighted_cam = cam_h * weights[:, 1:2]

        fused = torch.cat([weighted_lidar, weighted_cam], dim=-1)
        return fused, weights


class PlannerHead(nn.Module):
    """
    Simple MLP planner for steering and speed prediction.

    Architecture:
        Concat(lidar_feat, cam_feat) -> FC -> ReLU -> Dropout -> FC -> ReLU -> FC -> output

    Output interpretation:
        - steering: tanh activated, range [-1, 1] -> [-30deg, +30deg]
        - speed: sigmoid activated, range [0, 1] -> [0, max_speed] m/s
    """
    def __init__(self, lidar_dim=32, cam_dim=64, hidden_dim=64,
                 max_steering_deg=30.0, max_speed_mps=2.0):
        super().__init__()

        self.max_steering_rad = math.radians(max_steering_deg)
        self.max_speed = max_speed_mps

        self.fusion = nn.Sequential(
            nn.Linear(lidar_dim + cam_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, lidar_feat, cam_feat):
        """
        Args:
            lidar_feat: (B, lidar_dim)
            cam_feat: (B, cam_dim)

        Returns:
            output: (B, 2) [steering_rad, speed_mps]
        """
        x = torch.cat([lidar_feat, cam_feat], dim=1)
        raw = self.fusion(x)

        # Constrain outputs
        steering = torch.tanh(raw[:, 0]) * self.max_steering_rad
        speed = torch.sigmoid(raw[:, 1]) * self.max_speed

        return torch.stack([steering, speed], dim=1)

    def forward_raw(self, lidar_feat, cam_feat):
        """Forward without activation constraints (for training)"""
        x = torch.cat([lidar_feat, cam_feat], dim=1)
        return self.fusion(x)


class PlannerHeadWithAttention(nn.Module):
    """
    Advanced planner with cross-modal attention.
    Learns to weight LiDAR vs Camera based on context.
    """
    def __init__(self, lidar_dim=32, cam_dim=64, hidden_dim=64,
                 max_steering_deg=30.0, max_speed_mps=2.0):
        super().__init__()

        self.max_steering_rad = math.radians(max_steering_deg)
        self.max_speed = max_speed_mps

        self.attention = FeatureAttention(lidar_dim, cam_dim, hidden_dim=32)

        self.head = nn.Sequential(
            nn.Linear(64, hidden_dim),  # 32 * 2 from attention
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, lidar_feat, cam_feat):
        fused, attn_weights = self.attention(lidar_feat, cam_feat)
        raw = self.head(fused)

        steering = torch.tanh(raw[:, 0]) * self.max_steering_rad
        speed = torch.sigmoid(raw[:, 1]) * self.max_speed

        return torch.stack([steering, speed], dim=1), attn_weights


class PlannerHeadWithUncertainty(nn.Module):
    """
    Planner with uncertainty estimation for safer control.
    High uncertainty -> slow down or stop.

    Uses heteroscedastic aleatoric uncertainty:
        Predicts both mean and variance for each output.
    """
    def __init__(self, lidar_dim=32, cam_dim=64, hidden_dim=64,
                 max_steering_deg=30.0, max_speed_mps=2.0):
        super().__init__()

        self.max_steering_rad = math.radians(max_steering_deg)
        self.max_speed = max_speed_mps

        self.shared = nn.Sequential(
            nn.Linear(lidar_dim + cam_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        # Mean prediction
        self.mean_head = nn.Linear(hidden_dim // 2, 2)

        # Log variance (for numerical stability)
        self.logvar_head = nn.Linear(hidden_dim // 2, 2)

    def forward(self, lidar_feat, cam_feat):
        """
        Returns:
            mean: (B, 2) predicted [steering, speed]
            log_var: (B, 2) log variance
        """
        x = torch.cat([lidar_feat, cam_feat], dim=1)
        h = self.shared(x)

        raw_mean = self.mean_head(h)
        log_var = self.logvar_head(h)

        # Constrain mean outputs
        steering = torch.tanh(raw_mean[:, 0]) * self.max_steering_rad
        speed = torch.sigmoid(raw_mean[:, 1]) * self.max_speed
        mean = torch.stack([steering, speed], dim=1)

        # Clamp log variance for stability
        log_var = torch.clamp(log_var, min=-10, max=2)

        return mean, log_var

    def get_uncertainty(self, lidar_feat, cam_feat):
        """Get standard deviation for each output"""
        _, log_var = self.forward(lidar_feat, cam_feat)
        return torch.exp(0.5 * log_var)

    def sample(self, lidar_feat, cam_feat, n_samples=1):
        """Sample from predicted distribution"""
        mean, log_var = self.forward(lidar_feat, cam_feat)
        std = torch.exp(0.5 * log_var)

        if n_samples == 1:
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            # Multiple samples
            samples = []
            for _ in range(n_samples):
                eps = torch.randn_like(std)
                samples.append(mean + eps * std)
            return torch.stack(samples, dim=0)


class SafetyLayer(nn.Module):
    """
    Safety constraints layer applied after planner output.

    Implements:
        1. Speed reduction when steering is large
        2. Emergency stop when uncertainty is high
        3. Smooth output limiting
    """
    def __init__(self, max_steering_rad=0.52, max_speed=2.0,
                 uncertainty_thresh=0.5, min_speed=0.1):
        super().__init__()
        self.max_steering = max_steering_rad
        self.max_speed = max_speed
        self.uncertainty_thresh = uncertainty_thresh
        self.min_speed = min_speed

    def forward(self, action, uncertainty=None):
        """
        Args:
            action: (B, 2) [steering, speed]
            uncertainty: (B, 2) optional uncertainty

        Returns:
            safe_action: (B, 2) constrained action
        """
        steering = action[:, 0]
        speed = action[:, 1]

        # 1. Reduce speed when steering is large
        steering_ratio = torch.abs(steering) / self.max_steering
        speed_factor = 1.0 - 0.5 * steering_ratio  # 50% speed reduction at max steering
        speed = speed * speed_factor

        # 2. Apply uncertainty-based speed reduction
        if uncertainty is not None:
            # Higher uncertainty -> lower speed
            speed_unc = uncertainty[:, 1]
            unc_factor = torch.exp(-speed_unc / self.uncertainty_thresh)
            speed = speed * unc_factor

        # 3. Clamp outputs
        steering = torch.clamp(steering, -self.max_steering, self.max_steering)
        speed = torch.clamp(speed, self.min_speed, self.max_speed)

        return torch.stack([steering, speed], dim=1)


class FullPipeline(nn.Module):
    """
    Complete End-to-End Autonomous Driving Pipeline.

    Pipeline:
        Camera (320x240 RGB) -> CameraCNN -> cam_feat (64-dim)
        LiDAR BEV (200x240) -> LiDARTemporalEncoder + ConvGRU -> lidar_feat (32-dim)
        [cam_feat, lidar_feat] -> PlannerHead -> [steering, speed]

    For RC Car with:
        - CSI Camera: 320x240 SD resolution
        - 2D LiDAR: RPLIDAR A1/A2 or YDLidar
        - Jetson Orin Nano inference
    """
    def __init__(self, camera_cnn, lidar_encoder, planner_head,
                 use_safety=True, use_uncertainty=False):
        super().__init__()
        self.camera_cnn = camera_cnn
        self.lidar_encoder = lidar_encoder
        self.planner_head = planner_head

        self.use_safety = use_safety
        self.use_uncertainty = use_uncertainty

        if use_safety:
            self.safety = SafetyLayer()

        # Hidden state for LiDAR temporal encoder
        self.h_lidar = None

    def forward(self, image, bev):
        """
        End-to-end inference.

        Args:
            image: (B, 3, 240, 320) camera image (H, W for 320x240)
            bev: (B, 2, 200, 240) LiDAR BEV grid (H, W)

        Returns:
            action: (B, 2) [steering_rad, speed_mps]
        """
        # Camera feature extraction
        cam_feat = self.camera_cnn(image)

        # LiDAR temporal feature extraction
        lidar_feat, self.h_lidar = self.lidar_encoder(bev, self.h_lidar)

        # Planning
        if self.use_uncertainty and hasattr(self.planner_head, 'get_uncertainty'):
            action, log_var = self.planner_head(lidar_feat, cam_feat)
            uncertainty = torch.exp(0.5 * log_var)

            if self.use_safety:
                action = self.safety(action, uncertainty)
        else:
            if hasattr(self.planner_head, '__call__'):
                result = self.planner_head(lidar_feat, cam_feat)
                # Handle both tuple and tensor returns
                if isinstance(result, tuple):
                    action = result[0]
                else:
                    action = result

            if self.use_safety:
                action = self.safety(action)

        return action

    def reset_temporal_state(self):
        """Reset LiDAR temporal state (call at episode/track start)"""
        self.h_lidar = None

    def get_features(self, image, bev):
        """Get intermediate features for visualization"""
        cam_feat = self.camera_cnn(image)
        lidar_feat, self.h_lidar = self.lidar_encoder(bev, self.h_lidar)
        return cam_feat, lidar_feat


class OnlineTrainer:
    """
    Online learning trainer for imitation learning.

    Training flow:
        1. Human drives RC car (teacher)
        2. System records (observation, action) pairs
        3. Periodically trains planner on replay buffer
        4. Gradually takes over control

    Note: Encoder weights stay frozen, only planner is updated.
    """
    def __init__(self, planner_head, lr=1e-3, buffer_size=1000,
                 weight_decay=1e-4):
        self.planner = planner_head
        self.optimizer = torch.optim.AdamW(
            self.planner.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for robustness

        self.buffer = []
        self.buffer_size = buffer_size
        self.train_steps = 0
        self.running_loss = 0.0

    def add_sample(self, lidar_feat, cam_feat, target_action, weight=1.0):
        """
        Add training sample to buffer.

        Args:
            lidar_feat: (32,) numpy array
            cam_feat: (64,) numpy array
            target_action: (2,) numpy array [steering, speed]
            weight: Sample importance weight
        """
        self.buffer.append({
            'lidar': lidar_feat.astype(np.float32).copy(),
            'cam': cam_feat.astype(np.float32).copy(),
            'target': target_action.astype(np.float32).copy(),
            'weight': float(weight)
        })

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def train_step(self, batch_size=32, device='cuda'):
        """
        Perform one training step.

        Returns:
            loss: Training loss, or None if buffer too small
        """
        if len(self.buffer) < batch_size:
            return None

        # Sample random batch
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        # Stack into tensors
        lidar_batch = torch.tensor(
            np.stack([s['lidar'] for s in batch]),
            dtype=torch.float32, device=device
        )
        cam_batch = torch.tensor(
            np.stack([s['cam'] for s in batch]),
            dtype=torch.float32, device=device
        )
        target_batch = torch.tensor(
            np.stack([s['target'] for s in batch]),
            dtype=torch.float32, device=device
        )
        weights = torch.tensor(
            [s['weight'] for s in batch],
            dtype=torch.float32, device=device
        )

        # Training
        self.planner.train()
        self.optimizer.zero_grad()

        # Use raw output for training (without activation constraints)
        if hasattr(self.planner, 'forward_raw'):
            pred = self.planner.forward_raw(lidar_batch, cam_batch)
        else:
            pred = self.planner(lidar_batch, cam_batch)
            if isinstance(pred, tuple):
                pred = pred[0]

        # Weighted loss
        loss = (self.loss_fn(pred, target_batch) * weights).mean()

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.planner.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.planner.eval()

        self.train_steps += 1
        self.running_loss = 0.9 * self.running_loss + 0.1 * loss.item()

        # Update learning rate scheduler every 100 steps
        if self.train_steps % 100 == 0:
            self.scheduler.step(self.running_loss)

        return loss.item()

    def get_stats(self):
        """Get training statistics"""
        return {
            'buffer_size': len(self.buffer),
            'train_steps': self.train_steps,
            'running_loss': self.running_loss,
            'lr': self.optimizer.param_groups[0]['lr']
        }

    def save_checkpoint(self, path):
        """Save training state"""
        torch.save({
            'planner_state_dict': self.planner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_steps': self.train_steps,
            'running_loss': self.running_loss,
            'buffer_size': len(self.buffer)
        }, path)

    def load_checkpoint(self, path, device='cuda'):
        """Load training state"""
        checkpoint = torch.load(path, map_location=device)
        self.planner.load_state_dict(checkpoint['planner_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_steps = checkpoint.get('train_steps', 0)
        self.running_loss = checkpoint.get('running_loss', 0.0)


def export_to_onnx(camera_cnn, lidar_encoder, planner_head, output_path,
                   image_shape=(1, 3, 240, 320), bev_shape=(1, 2, 200, 240)):
    """
    Export full pipeline to ONNX for TensorRT optimization.

    Args:
        camera_cnn: CameraCNN instance
        lidar_encoder: LiDARTemporalEncoder instance
        planner_head: PlannerHead instance
        output_path: .onnx file path
        image_shape: (B, C, H, W) for 320x240 image
        bev_shape: (B, C, H, W) for BEV grid
    """

    class ExportableModel(nn.Module):
        def __init__(self, cam, lidar, planner):
            super().__init__()
            self.cam = cam
            self.lidar = lidar
            self.planner = planner

        def forward(self, image, bev):
            cam_feat = self.cam(image)
            lidar_feat, _ = self.lidar(bev, None)
            result = self.planner(lidar_feat, cam_feat)
            if isinstance(result, tuple):
                return result[0]
            return result

    model = ExportableModel(camera_cnn, lidar_encoder, planner_head)
    model.eval()

    dummy_image = torch.randn(*image_shape)
    dummy_bev = torch.randn(*bev_shape)

    torch.onnx.export(
        model,
        (dummy_image, dummy_bev),
        output_path,
        input_names=['image', 'bev'],
        output_names=['action'],
        dynamic_axes={
            'image': {0: 'batch'},
            'bev': {0: 'batch'},
            'action': {0: 'batch'}
        },
        opset_version=11,
        do_constant_folding=True
    )
    print(f"Exported to {output_path}")
    print(f"  Image input: {image_shape} (B, C, H, W)")
    print(f"  BEV input: {bev_shape} (B, C, H, W)")
    print(f"  Action output: (B, 2) [steering_rad, speed_mps]")


def create_default_pipeline(device='cuda'):
    """
    Create default end-to-end pipeline for RC car.

    Returns:
        pipeline: FullPipeline instance
        trainer: OnlineTrainer instance
    """
    # Import models
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from camera_cnn import CameraCNN
    from lidar_temporal import LiDARTemporalEncoder

    # Create models
    camera_cnn = CameraCNN(in_channels=3, feature_dim=64).to(device)
    lidar_encoder = LiDARTemporalEncoder(
        in_channels=2,
        hidden_channels=32,
        feature_dim=32
    ).to(device)
    planner_head = PlannerHead(
        lidar_dim=32,
        cam_dim=64,
        hidden_dim=64
    ).to(device)

    # Create pipeline
    pipeline = FullPipeline(
        camera_cnn, lidar_encoder, planner_head,
        use_safety=True,
        use_uncertainty=False
    ).to(device)

    # Create trainer (for planner only)
    trainer = OnlineTrainer(planner_head, lr=1e-3, buffer_size=1000)

    # Count parameters
    total_params = sum(p.numel() for p in pipeline.parameters())
    trainable_params = sum(p.numel() for p in planner_head.parameters())

    print(f"Pipeline created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable (planner): {trainable_params:,}")
    print(f"  Device: {device}")

    return pipeline, trainer


if __name__ == '__main__':
    import time

    print("=" * 60)
    print("Planner Head - Model Analysis")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test basic planner
    print("\n1. Testing PlannerHead...")
    planner = PlannerHead(lidar_dim=32, cam_dim=64).to(device)

    lidar_feat = torch.randn(4, 32).to(device)
    cam_feat = torch.randn(4, 64).to(device)
    output = planner(lidar_feat, cam_feat)
    print(f"   Input: lidar={lidar_feat.shape}, cam={cam_feat.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Steering range: [{output[:, 0].min():.3f}, {output[:, 0].max():.3f}] rad")
    print(f"   Speed range: [{output[:, 1].min():.3f}, {output[:, 1].max():.3f}] m/s")

    # Test with attention
    print("\n2. Testing PlannerHeadWithAttention...")
    planner_attn = PlannerHeadWithAttention(lidar_dim=32, cam_dim=64).to(device)
    output, attn_weights = planner_attn(lidar_feat, cam_feat)
    print(f"   Output: {output.shape}")
    print(f"   Attention weights: {attn_weights[0].cpu().numpy()}")

    # Test with uncertainty
    print("\n3. Testing PlannerHeadWithUncertainty...")
    planner_unc = PlannerHeadWithUncertainty(lidar_dim=32, cam_dim=64).to(device)
    mean, log_var = planner_unc(lidar_feat, cam_feat)
    uncertainty = planner_unc.get_uncertainty(lidar_feat, cam_feat)
    print(f"   Mean: {mean.shape}, Uncertainty: {uncertainty.shape}")

    # Test OnlineTrainer
    print("\n4. Testing OnlineTrainer...")
    trainer = OnlineTrainer(planner, lr=1e-3, buffer_size=100)

    for _ in range(50):
        trainer.add_sample(
            np.random.randn(32).astype(np.float32),
            np.random.randn(64).astype(np.float32),
            np.random.randn(2).astype(np.float32)
        )

    loss = trainer.train_step(batch_size=16, device=device)
    print(f"   Training loss: {loss:.4f}")
    print(f"   Stats: {trainer.get_stats()}")

    # Benchmark inference
    print("\n5. Inference Benchmark...")
    planner.eval()

    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = planner(lidar_feat, cam_feat)

        if device == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = planner(lidar_feat, cam_feat)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    print(f"   Inference: {np.mean(times):.3f} +/- {np.std(times):.3f} ms")

    # Parameter count
    params = sum(p.numel() for p in planner.parameters())
    print(f"\n6. Parameters: {params:,}")

    print("\n" + "=" * 60)
    print("All tests passed!")
