#!/usr/bin/env python3
"""
LiDAR Temporal Encoder with ConvGRU
Processes BEV sequences and maintains temporal state

Input: BEV grid (B, 2, H, W) - occupancy + intensity channels
Output: Temporal feature (B, hidden_dim)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    """
    Convolutional GRU Cell for spatial-temporal processing.
    Maintains spatial structure while updating temporal state.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim

        # Combined convolution for reset and update gates
        self.conv_gates = nn.Conv2d(
            input_dim + hidden_dim,
            2 * hidden_dim,
            kernel_size,
            padding=padding,
            bias=True
        )

        # Convolution for candidate hidden state
        self.conv_candidate = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim,
            kernel_size,
            padding=padding,
            bias=True
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, h_prev=None):
        """
        Args:
            x: Input tensor (B, C_in, H, W)
            h_prev: Previous hidden state (B, C_hidden, H, W) or None

        Returns:
            h: Updated hidden state (B, C_hidden, H, W)
        """
        B, _, H, W = x.shape

        if h_prev is None:
            h_prev = torch.zeros(
                B, self.hidden_dim, H, W,
                device=x.device, dtype=x.dtype
            )

        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)

        # Compute reset and update gates
        gates = torch.sigmoid(self.conv_gates(combined))
        reset_gate, update_gate = torch.chunk(gates, 2, dim=1)

        # Compute candidate hidden state
        combined_reset = torch.cat([x, reset_gate * h_prev], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined_reset))

        # Update hidden state
        h = (1 - update_gate) * h_prev + update_gate * candidate

        return h


class BEVEncoder(nn.Module):
    """Pre-process BEV grid before ConvGRU"""
    def __init__(self, in_channels=2, out_channels=32):
        super().__init__()

        self.encoder = nn.Sequential(
            # (B, 2, H, W) -> (B, 16, H, W)
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # (B, 16, H, W) -> (B, 32, H/2, W/2)
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # (B, 32, H/2, W/2) -> (B, out_channels, H/2, W/2)
            nn.Conv2d(32, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


class LiDARTemporalEncoder(nn.Module):
    """
    Complete LiDAR temporal encoder pipeline:
    BEV -> BEVEncoder -> ConvGRU -> GlobalAvgPool -> Feature

    Maintains hidden state across frames for temporal consistency.
    """
    def __init__(self, in_channels=2, hidden_channels=32, feature_dim=32):
        super().__init__()

        self.bev_encoder = BEVEncoder(in_channels, hidden_channels)
        self.conv_gru = ConvGRUCell(hidden_channels, hidden_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_channels, feature_dim)

        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim

    def forward(self, bev, h_prev=None, return_hidden=True):
        """
        Args:
            bev: BEV tensor (B, 2, H, W) or numpy array
            h_prev: Previous hidden state or None
            return_hidden: If True, return (feature, h_new). If False, return feature only.

        Returns:
            feature: (B, feature_dim) pooled feature vector
            h_new: (B, hidden_channels, H', W') updated hidden state (if return_hidden)
        """
        # Handle numpy input
        if isinstance(bev, np.ndarray):
            bev = torch.from_numpy(bev)

        # Add batch dimension if needed
        if bev.dim() == 3:
            bev = bev.unsqueeze(0)

        bev = bev.float()

        # Get device from model parameters
        device = next(self.parameters()).device
        bev = bev.to(device)

        # Encode BEV
        x = self.bev_encoder(bev)  # (B, hidden_channels, H', W')

        # Temporal update with ConvGRU
        h_new = self.conv_gru(x, h_prev)  # (B, hidden_channels, H', W')

        # Global average pooling -> feature vector
        pooled = self.pool(h_new)  # (B, hidden_channels, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, hidden_channels)
        feature = self.fc(pooled)  # (B, feature_dim)

        if return_hidden:
            return feature, h_new
        return feature

    def reset_state(self):
        """Reset hidden state (call at episode start)"""
        return None


class BEVGrid:
    """
    Incremental BEV Grid with temporal decay.
    Converts LiDAR points to a 2D occupancy grid.
    """
    def __init__(self,
                 x_range=(0.0, 15.0),
                 y_range=(-7.5, 7.5),
                 resolution=0.15,
                 decay_factor=0.9):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.res = resolution
        self.decay_factor = decay_factor

        self.W = int((self.x_max - self.x_min) / self.res)
        self.H = int((self.y_max - self.y_min) / self.res)

        # Channels: [occupancy, point_count]
        self.grid = np.zeros((2, self.H, self.W), dtype=np.float32)

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        ix = int((x - self.x_min) / self.res)
        iy = int((y - self.y_min) / self.res)

        if 0 <= ix < self.W and 0 <= iy < self.H:
            return iy, ix
        return None, None

    def decay(self):
        """Apply temporal decay to previous observations"""
        self.grid *= self.decay_factor

    def update_from_lidar(self, points_xy, decay=True):
        """
        Update grid with new LiDAR points.

        Args:
            points_xy: (N, 2) array of (x, y) points in vehicle frame
            decay: Whether to apply decay before update
        """
        if decay:
            self.decay()

        occ = self.grid[0]
        cnt = self.grid[1]

        for x, y in points_xy:
            iy, ix = self.world_to_grid(x, y)
            if iy is not None:
                occ[iy, ix] = 1.0
                cnt[iy, ix] += 1.0

        # Normalize count channel
        cnt_max = np.max(cnt)
        if cnt_max > 0:
            self.grid[1] = cnt / cnt_max

    def update_from_laserscan(self, ranges, angle_min, angle_max, range_min=0.1, range_max=12.0):
        """
        Update grid from LaserScan message.

        Args:
            ranges: Array of range measurements
            angle_min: Start angle (rad)
            angle_max: End angle (rad)
            range_min: Minimum valid range
            range_max: Maximum valid range
        """
        angles = np.linspace(angle_min, angle_max, len(ranges))
        ranges = np.array(ranges, dtype=np.float32)

        # Filter valid ranges
        valid = (ranges > range_min) & (ranges < range_max) & np.isfinite(ranges)
        ranges = ranges[valid]
        angles = angles[valid]

        # Convert polar to Cartesian
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        points_xy = np.stack([x, y], axis=1)

        self.update_from_lidar(points_xy)

    def get_bev(self):
        """Get current BEV grid as (2, H, W) numpy array"""
        return self.grid.copy()

    def get_tensor(self, device='cpu'):
        """Get BEV as PyTorch tensor"""
        return torch.from_numpy(self.grid).unsqueeze(0).to(device)

    def reset(self):
        """Clear the grid"""
        self.grid.fill(0)


def export_to_onnx(model, output_path, bev_shape=(1, 2, 100, 100)):
    """Export LiDAR encoder to ONNX (without hidden state for simplicity)"""
    model.eval()
    dummy_bev = torch.randn(*bev_shape)

    # Create a wrapper that doesn't use hidden state
    class SimpleEncoder(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, bev):
            feature, _ = self.encoder(bev, h_prev=None)
            return feature

    simple_model = SimpleEncoder(model)

    torch.onnx.export(
        simple_model,
        dummy_bev,
        output_path,
        input_names=['bev'],
        output_names=['feature'],
        dynamic_axes={
            'bev': {0: 'batch_size'},
            'feature': {0: 'batch_size'}
        },
        opset_version=11
    )
    print(f"Exported to {output_path}")


if __name__ == '__main__':
    # Test BEVGrid
    print("Testing BEVGrid...")
    grid = BEVGrid(x_range=(0, 15), y_range=(-7.5, 7.5), resolution=0.15)
    print(f"Grid shape: {grid.W}x{grid.H}")

    # Simulate LiDAR scan
    angles = np.linspace(-np.pi/2, np.pi/2, 360)
    ranges = np.random.uniform(1, 10, 360)
    grid.update_from_laserscan(ranges, -np.pi/2, np.pi/2)
    bev = grid.get_bev()
    print(f"BEV shape: {bev.shape}")

    # Test LiDARTemporalEncoder
    print("\nTesting LiDARTemporalEncoder...")
    encoder = LiDARTemporalEncoder(in_channels=2, hidden_channels=32, feature_dim=32)

    bev_tensor = torch.randn(2, 2, 50, 100)
    feature, h = encoder(bev_tensor, h_prev=None)
    print(f"Input BEV: {bev_tensor.shape}")
    print(f"Output feature: {feature.shape}")
    print(f"Hidden state: {h.shape}")

    # Test temporal consistency
    feature2, h2 = encoder(bev_tensor, h_prev=h)
    print(f"Second pass feature: {feature2.shape}")

    # Count parameters
    params = sum(p.numel() for p in encoder.parameters())
    print(f"\nParameters: {params:,} ({params/1e6:.2f}M)")

    # Export to ONNX
    export_to_onnx(encoder, "lidar_temporal.onnx", bev_shape=(1, 2, 50, 100))
