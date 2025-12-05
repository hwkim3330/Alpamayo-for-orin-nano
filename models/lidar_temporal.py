#!/usr/bin/env python3
"""
LiDAR Temporal Encoder with ConvGRU
Optimized for RC Car scale LiDAR (2D LaserScan or 3D PointCloud)

Target LiDARs:
    - RPLIDAR A1/A2/A3 (2D, 360°, 8-16m range)
    - YDLidar X2/X4 (2D, 360°, 8-12m range)
    - Livox Mid-40/70 (3D, limited FOV)

BEV Grid Configuration for RC Car:
    - X range: [-2m, 10m] (behind to front)
    - Y range: [-5m, 5m] (left to right)
    - Resolution: 0.05m (5cm per cell)
    - Grid size: 240 x 200 = 48,000 cells

Architecture:
    BEV (2ch) -> Conv Encoder -> ConvGRU -> Global Pool -> FC -> Feature
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvGRUCell(nn.Module):
    """
    Convolutional GRU Cell for spatial-temporal processing.
    Maintains spatial structure while updating temporal state.

    Key differences from standard GRU:
        - Uses 2D convolutions instead of dense layers
        - Preserves spatial dimensions
        - Better for BEV/image sequences
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim

        # Gate convolutions (reset + update gates)
        self.conv_gates = nn.Conv2d(
            input_dim + hidden_dim,
            2 * hidden_dim,
            kernel_size,
            padding=padding,
            bias=True
        )

        # Candidate hidden state convolution
        self.conv_candidate = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim,
            kernel_size,
            padding=padding,
            bias=True
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.conv_gates, self.conv_candidate]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, h_prev=None):
        """
        Args:
            x: (B, C_in, H, W) input
            h_prev: (B, C_hidden, H, W) or None

        Returns:
            h: (B, C_hidden, H, W) updated hidden state
        """
        B, _, H, W = x.shape

        if h_prev is None:
            h_prev = torch.zeros(B, self.hidden_dim, H, W,
                                 device=x.device, dtype=x.dtype)

        # Concatenate input and previous hidden
        combined = torch.cat([x, h_prev], dim=1)

        # Compute gates
        gates = torch.sigmoid(self.conv_gates(combined))
        reset_gate, update_gate = torch.chunk(gates, 2, dim=1)

        # Compute candidate
        combined_reset = torch.cat([x, reset_gate * h_prev], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined_reset))

        # Update hidden state
        h = (1 - update_gate) * h_prev + update_gate * candidate

        return h


class BEVEncoder(nn.Module):
    """
    Encode BEV grid to compact feature map.

    Input: (B, 2, H, W) - occupancy + intensity
    Output: (B, hidden_ch, H/4, W/4)
    """
    def __init__(self, in_channels=2, hidden_channels=32):
        super().__init__()

        self.encoder = nn.Sequential(
            # Stage 1: (B, 2, H, W) -> (B, 16, H/2, W/2)
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Stage 2: (B, 16, H/2, W/2) -> (B, 32, H/4, W/4)
            nn.Conv2d(16, hidden_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            # Refinement
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


class LiDARTemporalEncoder(nn.Module):
    """
    Complete LiDAR temporal encoder for RC car.

    Pipeline:
        BEV (2, H, W) -> BEVEncoder -> ConvGRU -> GlobalPool -> FC -> Feature

    For 240x200 BEV input:
        240x200 -> 60x50 (encoded) -> 60x50 (GRU) -> 32 (pooled) -> feature_dim

    Parameters: ~50K
    Inference: ~3ms on Orin Nano
    """
    def __init__(self, in_channels=2, hidden_channels=32, feature_dim=32):
        super().__init__()

        self.bev_encoder = BEVEncoder(in_channels, hidden_channels)
        self.conv_gru = ConvGRUCell(hidden_channels, hidden_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels, feature_dim)

        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim

    def forward(self, bev, h_prev=None):
        """
        Args:
            bev: (B, 2, H, W) BEV grid or numpy array
            h_prev: Previous hidden state or None

        Returns:
            feature: (B, feature_dim)
            h_new: (B, hidden_channels, H', W') for next step
        """
        # Handle numpy input
        if isinstance(bev, np.ndarray):
            bev = torch.from_numpy(bev)
        if bev.dim() == 3:
            bev = bev.unsqueeze(0)

        bev = bev.float().to(next(self.parameters()).device)

        # Encode BEV
        x = self.bev_encoder(bev)

        # Temporal update
        h_new = self.conv_gru(x, h_prev)

        # Pool and project to feature
        pooled = self.pool(h_new).flatten(1)
        feature = self.fc(pooled)

        return feature, h_new

    def reset_state(self):
        """Reset temporal state (call at episode start)"""
        return None


class BEVGrid:
    """
    Incremental BEV Grid for RC Car scale LiDAR.

    Optimized for:
        - 2D LaserScan (RPLIDAR, YDLidar)
        - Real-time update at 10-20 Hz
        - Temporal decay for moving objects

    Grid Configuration:
        - X: forward direction (vehicle frame)
        - Y: left-right direction
        - Resolution: 5cm recommended for RC car

    Channels:
        - Channel 0: Occupancy (binary)
        - Channel 1: Intensity/Count (normalized)
    """
    def __init__(self,
                 x_range=(-2.0, 10.0),      # meters (behind, front)
                 y_range=(-5.0, 5.0),       # meters (left, right)
                 resolution=0.05,            # meters per cell (5cm)
                 decay_factor=0.85,          # temporal decay
                 max_count=10.0):            # max count for normalization
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.res = resolution
        self.decay_factor = decay_factor
        self.max_count = max_count

        # Grid dimensions
        self.W = int((self.x_max - self.x_min) / self.res)  # X direction (forward)
        self.H = int((self.y_max - self.y_min) / self.res)  # Y direction (left-right)

        # Grid: [occupancy, count]
        self.grid = np.zeros((2, self.H, self.W), dtype=np.float32)

        # Stats
        self.update_count = 0

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        ix = int((x - self.x_min) / self.res)
        iy = int((y - self.y_min) / self.res)

        if 0 <= ix < self.W and 0 <= iy < self.H:
            return iy, ix  # Note: (row, col) = (y, x)
        return None, None

    def update_from_points(self, points_xy, apply_decay=True):
        """
        Update grid from point cloud.

        Args:
            points_xy: (N, 2) array of (x, y) in vehicle frame
            apply_decay: Whether to decay previous observations
        """
        if apply_decay:
            self.grid *= self.decay_factor

        for x, y in points_xy:
            iy, ix = self.world_to_grid(x, y)
            if iy is not None:
                self.grid[0, iy, ix] = 1.0  # Occupancy
                self.grid[1, iy, ix] = min(
                    self.grid[1, iy, ix] + 1.0 / self.max_count,
                    1.0
                )  # Normalized count

        self.update_count += 1

    def update_from_laserscan(self, ranges, angle_min, angle_max,
                               range_min=0.15, range_max=12.0):
        """
        Update grid from LaserScan message.

        Args:
            ranges: Array of range measurements (meters)
            angle_min: Start angle (radians)
            angle_max: End angle (radians)
            range_min: Minimum valid range
            range_max: Maximum valid range
        """
        n_points = len(ranges)
        angles = np.linspace(angle_min, angle_max, n_points)
        ranges = np.array(ranges, dtype=np.float32)

        # Filter valid ranges
        valid = (ranges > range_min) & (ranges < range_max) & np.isfinite(ranges)
        ranges = ranges[valid]
        angles = angles[valid]

        # Polar to Cartesian (vehicle frame)
        # X = forward, Y = left
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        points_xy = np.stack([x, y], axis=1)
        self.update_from_points(points_xy)

    def get_bev(self):
        """Get current BEV as (2, H, W) numpy array"""
        return self.grid.copy()

    def get_tensor(self, device='cpu'):
        """Get BEV as PyTorch tensor (1, 2, H, W)"""
        return torch.from_numpy(self.grid).unsqueeze(0).to(device)

    def get_visualization(self):
        """Get BEV as RGB image for visualization"""
        vis = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        # Green channel: occupancy
        vis[:, :, 1] = (self.grid[0] * 255).astype(np.uint8)

        # Blue channel: count/intensity
        vis[:, :, 0] = (self.grid[1] * 255).astype(np.uint8)

        # Draw ego vehicle (center)
        ego_y = int((0 - self.y_min) / self.res)
        ego_x = int((0 - self.x_min) / self.res)
        if 0 <= ego_x < self.W and 0 <= ego_y < self.H:
            cv_h, cv_w = 6, 3  # Vehicle size in cells
            y1, y2 = max(0, ego_y - cv_w), min(self.H, ego_y + cv_w)
            x1, x2 = max(0, ego_x - cv_h), min(self.W, ego_x + cv_h)
            vis[y1:y2, x1:x2] = [0, 255, 255]  # Yellow

        # Flip for visualization (front = top)
        return np.flipud(vis)

    def reset(self):
        """Clear grid"""
        self.grid.fill(0)
        self.update_count = 0

    def get_info(self):
        """Get grid info string"""
        return (f"BEV Grid: {self.W}x{self.H} cells, "
                f"X:[{self.x_min}, {self.x_max}]m, "
                f"Y:[{self.y_min}, {self.y_max}]m, "
                f"res={self.res}m")


class EgoMotionCompensator:
    """
    Compensate BEV grid for ego motion using odometry.
    Warps previous grid to current vehicle frame.
    """
    def __init__(self, grid_shape, resolution):
        self.H, self.W = grid_shape
        self.res = resolution

        # Pre-compute grid coordinates
        y_coords = np.arange(self.H) * resolution
        x_coords = np.arange(self.W) * resolution
        self.xx, self.yy = np.meshgrid(x_coords, y_coords)

    def warp_grid(self, grid, dx, dy, dtheta):
        """
        Warp grid based on ego motion.

        Args:
            grid: (2, H, W) BEV grid
            dx: Forward displacement (meters)
            dy: Lateral displacement (meters)
            dtheta: Rotation (radians)

        Returns:
            warped: (2, H, W) warped grid
        """
        import cv2

        # Compute transformation matrix
        # Grid coordinates are in meters, need to convert to pixels
        cos_t = np.cos(dtheta)
        sin_t = np.sin(dtheta)

        # Transform: rotate then translate
        # New position = R * (old - center) + center + translation
        center_x = self.W * self.res / 2
        center_y = self.H * self.res / 2

        # Affine matrix for cv2.warpAffine (operates in pixel space)
        dx_px = dx / self.res
        dy_px = dy / self.res
        center_px = (self.W / 2, self.H / 2)

        M = cv2.getRotationMatrix2D(center_px, np.degrees(dtheta), 1.0)
        M[0, 2] += dx_px
        M[1, 2] += dy_px

        # Warp each channel
        warped = np.zeros_like(grid)
        for i in range(grid.shape[0]):
            warped[i] = cv2.warpAffine(
                grid[i], M, (self.W, self.H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        return warped


def export_to_onnx(model, output_path, bev_shape=(1, 2, 200, 240)):
    """Export LiDAR encoder to ONNX"""
    model.eval()
    dummy = torch.randn(*bev_shape)

    # Wrapper without hidden state for simple export
    class SimpleEncoder(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, bev):
            feat, _ = self.encoder(bev, None)
            return feat

    wrapper = SimpleEncoder(model)

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=['bev'],
        output_names=['feature'],
        dynamic_axes={
            'bev': {0: 'batch'},
            'feature': {0: 'batch'}
        },
        opset_version=11
    )
    print(f"Exported to {output_path}")


if __name__ == '__main__':
    import time

    print("=" * 60)
    print("LiDAR Temporal Encoder - Model Analysis")
    print("=" * 60)

    # Test BEVGrid
    print("\n--- BEVGrid Test ---")
    grid = BEVGrid(
        x_range=(-2.0, 10.0),
        y_range=(-5.0, 5.0),
        resolution=0.05
    )
    print(grid.get_info())

    # Simulate LaserScan
    angles = np.linspace(-np.pi, np.pi, 720)
    ranges = np.random.uniform(0.5, 8.0, 720)
    grid.update_from_laserscan(ranges, -np.pi, np.pi)
    bev = grid.get_bev()
    print(f"BEV shape: {bev.shape}")
    print(f"Occupancy cells: {(bev[0] > 0.5).sum()}")

    # Test LiDARTemporalEncoder
    print("\n--- LiDARTemporalEncoder Test ---")
    encoder = LiDARTemporalEncoder(
        in_channels=2,
        hidden_channels=32,
        feature_dim=32
    )

    params = sum(p.numel() for p in encoder.parameters())
    print(f"Parameters: {params:,} ({params/1e3:.1f}K)")

    # Benchmark
    bev_tensor = torch.randn(1, 2, 200, 240)
    encoder.eval()

    with torch.no_grad():
        # Warmup
        h = None
        for _ in range(3):
            _, h = encoder(bev_tensor, h)

        # Benchmark
        times = []
        h = None
        for _ in range(20):
            start = time.perf_counter()
            feat, h = encoder(bev_tensor, h)
            times.append((time.perf_counter() - start) * 1000)

    print(f"Input: {bev_tensor.shape}")
    print(f"Output: {feat.shape}")
    print(f"Inference: {sum(times)/len(times):.2f}ms (CPU)")

    # Test temporal consistency
    print("\n--- Temporal Consistency Test ---")
    h = None
    features = []
    for i in range(5):
        noise = torch.randn(1, 2, 200, 240) * 0.1
        bev_t = bev_tensor + noise
        feat, h = encoder(bev_t, h)
        features.append(feat.clone())

    for i in range(1, len(features)):
        diff = (features[i] - features[i-1]).abs().mean().item()
        print(f"  Step {i}: feature diff = {diff:.4f}")

    # Export
    print("\n--- Export to ONNX ---")
    export_to_onnx(encoder, "lidar_temporal.onnx", (1, 2, 200, 240))
