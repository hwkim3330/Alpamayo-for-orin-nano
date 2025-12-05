#!/usr/bin/env python3
"""
Camera CNN Model - Lightweight MobileNet-style encoder
Optimized for Jetson Orin Nano (TensorRT compatible)

Input: (B, 3, 90, 160) RGB image
Output: (B, 64) feature vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution (MobileNet style)"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CameraCNN(nn.Module):
    """
    Ultra-lightweight Camera Feature Extractor

    Architecture:
        Input: (B, 3, 90, 160)
        Conv1 + DWConv: 90x160 -> 45x80 (16ch -> 32ch)
        Conv2 + DWConv: 45x80 -> 23x40 (32ch -> 64ch)
        Conv3: 23x40 -> 12x20 (64ch)
        GlobalAvgPool -> FC -> (B, out_dim)

    FLOPs: ~5M (very efficient for Orin Nano)
    """
    def __init__(self, in_channels=3, out_dim=64):
        super().__init__()

        self.features = nn.Sequential(
            # Stage 1: 90x160 -> 45x80
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(16, 32, stride=1),

            # Stage 2: 45x80 -> 23x40
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(32, 64, stride=1),

            # Stage 3: 23x40 -> 12x20
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, out_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) normalized RGB image [0, 1]

        Returns:
            features: (B, out_dim) feature vector
        """
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CameraEncoderWithUncertainty(nn.Module):
    """Camera encoder with uncertainty estimation for safer planning"""
    def __init__(self, in_channels=3, out_dim=64):
        super().__init__()
        self.backbone = CameraCNN(in_channels, out_dim * 2)
        self.out_dim = out_dim

    def forward(self, x):
        """
        Returns:
            mean: (B, out_dim) feature mean
            log_var: (B, out_dim) log variance for uncertainty
        """
        out = self.backbone(x)
        mean, log_var = torch.chunk(out, 2, dim=1)
        return mean, log_var

    def sample(self, x):
        """Sample from the distribution (for training with reparameterization)"""
        mean, log_var = self.forward(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std


def export_to_onnx(model, output_path, input_shape=(1, 3, 90, 160)):
    """Export model to ONNX format for TensorRT"""
    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['image'],
        output_names=['features'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        },
        opset_version=11
    )
    print(f"Exported to {output_path}")


if __name__ == '__main__':
    # Test model
    model = CameraCNN(out_dim=64)
    x = torch.randn(2, 3, 90, 160)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.2f}M)")

    # Export to ONNX
    export_to_onnx(model, "camera_cnn.onnx")
