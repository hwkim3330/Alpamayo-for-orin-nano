#!/usr/bin/env python3
"""
Planner Head Model
Fuses LiDAR and Camera features to predict steering and speed

Input:
    lidar_feat: (B, D_lidar) from LiDARTemporalEncoder
    cam_feat: (B, D_cam) from CameraCNN
Output:
    (B, 2) - [steering_angle, speed]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PlannerHead(nn.Module):
    """
    Simple MLP that fuses LiDAR and Camera features.

    Output interpretation:
        - steering: Normalized steering angle [-1, 1] -> [-30°, +30°]
        - speed: Normalized speed [0, 1] -> [0, max_speed] m/s
    """
    def __init__(self, lidar_dim=32, cam_dim=64, hidden_dim=64, out_dim=2):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(lidar_dim + cam_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, out_dim)
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
            lidar_feat: (B, D_lidar) LiDAR temporal features
            cam_feat: (B, D_cam) Camera features

        Returns:
            output: (B, 2) [steering, speed]
        """
        x = torch.cat([lidar_feat, cam_feat], dim=1)
        return self.fusion(x)


class PlannerHeadWithUncertainty(nn.Module):
    """
    Planner with uncertainty estimation for safer control.
    Outputs mean prediction and variance for each action.
    """
    def __init__(self, lidar_dim=32, cam_dim=64, hidden_dim=64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(lidar_dim + cam_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        # Mean prediction
        self.mean_head = nn.Linear(hidden_dim // 2, 2)

        # Log variance prediction (for numerical stability)
        self.logvar_head = nn.Linear(hidden_dim // 2, 2)

    def forward(self, lidar_feat, cam_feat):
        """
        Returns:
            mean: (B, 2) predicted [steering, speed]
            log_var: (B, 2) log variance for uncertainty
        """
        x = torch.cat([lidar_feat, cam_feat], dim=1)
        h = self.shared(x)
        mean = self.mean_head(h)
        log_var = self.logvar_head(h)
        return mean, log_var

    def sample(self, lidar_feat, cam_feat):
        """Sample from the predicted distribution"""
        mean, log_var = self.forward(lidar_feat, cam_feat)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def get_uncertainty(self, lidar_feat, cam_feat):
        """Get uncertainty (std) for the prediction"""
        _, log_var = self.forward(lidar_feat, cam_feat)
        return torch.exp(0.5 * log_var)


class FullPipeline(nn.Module):
    """
    Complete perception-to-control pipeline.
    Combines CameraCNN, LiDARTemporalEncoder, and PlannerHead.
    """
    def __init__(self, camera_cnn, lidar_encoder, planner_head):
        super().__init__()
        self.camera_cnn = camera_cnn
        self.lidar_encoder = lidar_encoder
        self.planner_head = planner_head

        # Hidden state for LiDAR temporal encoder
        self.h_lidar = None

    def forward(self, image, bev):
        """
        End-to-end inference.

        Args:
            image: (B, 3, H, W) camera image
            bev: (B, 2, H, W) LiDAR BEV grid

        Returns:
            action: (B, 2) [steering, speed]
        """
        # Camera feature
        cam_feat = self.camera_cnn(image)

        # LiDAR temporal feature
        lidar_feat, self.h_lidar = self.lidar_encoder(bev, self.h_lidar)

        # Planner
        action = self.planner_head(lidar_feat, cam_feat)

        return action

    def reset_temporal_state(self):
        """Reset LiDAR temporal state (call at episode start)"""
        self.h_lidar = None


class OnlineTrainer:
    """
    Online learning trainer for the planner head.
    Keeps encoder weights frozen, only updates planner.
    """
    def __init__(self, planner_head, lr=1e-3, buffer_size=1000):
        self.planner = planner_head
        self.optimizer = torch.optim.Adam(self.planner.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.buffer = []
        self.buffer_size = buffer_size

    def add_sample(self, lidar_feat, cam_feat, target_action):
        """
        Add a training sample to the buffer.

        Args:
            lidar_feat: (D_lidar,) numpy array
            cam_feat: (D_cam,) numpy array
            target_action: (2,) numpy array [steering, speed]
        """
        self.buffer.append({
            'lidar': lidar_feat.copy(),
            'cam': cam_feat.copy(),
            'target': target_action.copy()
        })

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def train_step(self, batch_size=32, device='cuda'):
        """
        Perform one training step on random batch.

        Returns:
            loss: Training loss value
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

        # Training step
        self.planner.train()
        self.optimizer.zero_grad()

        pred = self.planner(lidar_batch, cam_batch)
        loss = self.loss_fn(pred, target_batch)

        loss.backward()
        self.optimizer.step()

        self.planner.eval()

        return loss.item()

    def save_checkpoint(self, path):
        """Save planner weights"""
        torch.save({
            'planner_state_dict': self.planner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'buffer_size': len(self.buffer)
        }, path)

    def load_checkpoint(self, path):
        """Load planner weights"""
        checkpoint = torch.load(path)
        self.planner.load_state_dict(checkpoint['planner_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def export_to_onnx(camera_cnn, lidar_encoder, planner_head, output_path,
                   image_shape=(1, 3, 90, 160), bev_shape=(1, 2, 50, 100)):
    """Export the full pipeline to ONNX"""

    class ExportableModel(nn.Module):
        def __init__(self, cam, lidar, planner):
            super().__init__()
            self.cam = cam
            self.lidar = lidar
            self.planner = planner

        def forward(self, image, bev):
            cam_feat = self.cam(image)
            lidar_feat, _ = self.lidar(bev, None)
            return self.planner(lidar_feat, cam_feat)

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
            'image': {0: 'batch_size'},
            'bev': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        },
        opset_version=11
    )
    print(f"Exported full pipeline to {output_path}")


if __name__ == '__main__':
    # Test PlannerHead
    print("Testing PlannerHead...")
    planner = PlannerHead(lidar_dim=32, cam_dim=64)

    lidar_feat = torch.randn(4, 32)
    cam_feat = torch.randn(4, 64)
    output = planner(lidar_feat, cam_feat)
    print(f"Input: lidar={lidar_feat.shape}, cam={cam_feat.shape}")
    print(f"Output: {output.shape}")

    # Test with uncertainty
    print("\nTesting PlannerHeadWithUncertainty...")
    planner_unc = PlannerHeadWithUncertainty(lidar_dim=32, cam_dim=64)
    mean, log_var = planner_unc(lidar_feat, cam_feat)
    uncertainty = planner_unc.get_uncertainty(lidar_feat, cam_feat)
    print(f"Mean: {mean.shape}, Uncertainty: {uncertainty.shape}")

    # Test OnlineTrainer
    print("\nTesting OnlineTrainer...")
    trainer = OnlineTrainer(planner, lr=1e-3, buffer_size=100)

    # Add some samples
    for _ in range(50):
        trainer.add_sample(
            np.random.randn(32).astype(np.float32),
            np.random.randn(64).astype(np.float32),
            np.random.randn(2).astype(np.float32)
        )

    loss = trainer.train_step(batch_size=16, device='cpu')
    print(f"Training loss: {loss:.4f}")

    # Count parameters
    params = sum(p.numel() for p in planner.parameters())
    print(f"\nPlanner parameters: {params:,}")
