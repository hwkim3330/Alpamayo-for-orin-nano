#!/usr/bin/env python3
"""
Camera CNN Model for CSI Camera (SD Resolution)
Optimized for Jetson Orin Nano with TensorRT

Supported resolutions:
    - 320x240 (QVGA) - Primary target
    - 640x480 (VGA)
    - 160x120 (QQVGA) - Ultra-fast mode

Architecture: EfficientNet-inspired with Inverted Residuals
Target inference: < 5ms on Orin Nano with TensorRT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, reduced, 1)
        self.fc2 = nn.Conv2d(reduced, channels, 1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class InvertedResidual(nn.Module):
    """
    MobileNetV2-style Inverted Residual Block
    expand -> depthwise -> squeeze
    """
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=4, use_se=True):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_ch == out_ch)

        hidden_ch = in_ch * expand_ratio

        layers = []

        # Expand (pointwise)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU6(inplace=True),
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_ch, hidden_ch, 3, stride=stride, padding=1,
                      groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU6(inplace=True),
        ])

        # Squeeze-and-Excitation
        if use_se:
            layers.append(SEBlock(hidden_ch, reduction=4))

        # Project (pointwise)
        layers.extend([
            nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class CameraCNN(nn.Module):
    """
    Lightweight Camera Feature Extractor for CSI SD Camera

    Input: (B, 3, H, W)
        - 320x240 primary
        - 640x480 also supported
        - 160x120 for ultra-fast

    Output: (B, feature_dim)

    Architecture stages:
        Stage 0: 3 -> 16, stride 2    (H/2, W/2)
        Stage 1: 16 -> 24, stride 2   (H/4, W/4)
        Stage 2: 24 -> 32, stride 2   (H/8, W/8)
        Stage 3: 32 -> 48, stride 2   (H/16, W/16)
        Stage 4: 48 -> 64, stride 1   (H/16, W/16)
        GAP -> FC -> feature_dim

    For 320x240 input:
        320x240 -> 160x120 -> 80x60 -> 40x30 -> 20x15 -> 20x15
        Final feature map: 20x15x64 = 19,200 -> GAP -> 64

    FLOPs: ~15M (very efficient)
    Params: ~100K
    """
    def __init__(self, in_channels=3, feature_dim=64, width_mult=1.0):
        super().__init__()

        def _make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        # Channel configurations [out_ch, stride, expand_ratio, num_blocks]
        config = [
            # out, stride, expand, blocks
            [24, 2, 2, 1],   # Stage 1
            [32, 2, 4, 2],   # Stage 2
            [48, 2, 4, 2],   # Stage 3
            [64, 1, 4, 1],   # Stage 4
        ]

        # Stem: initial conv
        stem_ch = _make_divisible(16 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU6(inplace=True),
        )

        # Build stages
        stages = []
        in_ch = stem_ch

        for out_ch, stride, expand, num_blocks in config:
            out_ch = _make_divisible(out_ch * width_mult)
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                stages.append(InvertedResidual(in_ch, out_ch, stride=s,
                                               expand_ratio=expand, use_se=True))
                in_ch = out_ch

        self.stages = nn.Sequential(*stages)

        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(in_ch, feature_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) normalized RGB [0, 1]

        Returns:
            (B, feature_dim)
        """
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def get_feature_map(self, x):
        """Get intermediate feature map for visualization"""
        x = self.stem(x)
        x = self.stages(x)
        return x


class CameraCNNTiny(nn.Module):
    """
    Ultra-lightweight version for real-time (< 2ms)
    Use when running at high frequency (20+ Hz)

    Input: (B, 3, 160, 120) or (B, 3, 80, 60)
    Output: (B, 32)
    """
    def __init__(self, in_channels=3, feature_dim=32):
        super().__init__()

        self.features = nn.Sequential(
            # 160x120 -> 80x60
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),

            # 80x60 -> 40x30
            nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16, bias=False),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            # 40x30 -> 20x15
            nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU6(inplace=True),

            # 20x15 -> 10x8
            nn.Conv2d(48, 48, 3, stride=2, padding=1, groups=48, bias=False),
            nn.Conv2d(48, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, feature_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


class CameraChangeDetector:
    """
    Efficient change detection for gating CNN inference.
    Uses low-resolution grayscale comparison.

    Strategy:
        1. Downsample to tiny resolution (32x24)
        2. Compute mean absolute difference with previous frame
        3. If diff > threshold AND time > min_interval: run CNN
        4. Otherwise: reuse cached feature
    """
    def __init__(self,
                 small_size=(32, 24),      # Tiny resolution for diff
                 diff_thresh=8.0,          # Mean pixel diff threshold
                 min_interval=0.1,         # Minimum seconds between CNN runs
                 max_interval=0.5):        # Maximum seconds (force run)
        self.small_w, self.small_h = small_size
        self.diff_thresh = diff_thresh
        self.min_interval = min_interval
        self.max_interval = max_interval

        self.prev_small = None
        self.last_cnn_time = 0.0
        self.cached_feature = None

        # Stats
        self.total_frames = 0
        self.cnn_runs = 0

    def should_run_cnn(self, frame_gray):
        """
        Determine if CNN should run based on frame change.

        Args:
            frame_gray: (H, W) grayscale uint8

        Returns:
            need_cnn: bool
            mean_diff: float (for debugging)
        """
        import cv2
        import time

        self.total_frames += 1
        now = time.time()
        elapsed = now - self.last_cnn_time

        # Resize to tiny
        small = cv2.resize(frame_gray, (self.small_w, self.small_h),
                           interpolation=cv2.INTER_AREA)

        # Compute difference
        mean_diff = 0.0
        if self.prev_small is not None:
            diff = cv2.absdiff(small, self.prev_small)
            mean_diff = float(diff.mean())

        self.prev_small = small

        # Decision logic
        if self.cached_feature is None:
            # First frame, must run
            need_cnn = True
        elif elapsed >= self.max_interval:
            # Forced run after max interval
            need_cnn = True
        elif elapsed >= self.min_interval and mean_diff > self.diff_thresh:
            # Change detected after min interval
            need_cnn = True
        else:
            need_cnn = False

        if need_cnn:
            self.cnn_runs += 1
            self.last_cnn_time = now

        return need_cnn, mean_diff

    def update_feature(self, feature):
        """Store computed feature"""
        self.cached_feature = feature

    def get_feature(self):
        """Get cached feature"""
        return self.cached_feature

    def get_stats(self):
        """Get efficiency stats"""
        if self.total_frames == 0:
            return 0.0
        return self.cnn_runs / self.total_frames

    def reset(self):
        """Reset state"""
        self.prev_small = None
        self.cached_feature = None
        self.total_frames = 0
        self.cnn_runs = 0


def preprocess_csi_frame(frame_bgr, target_size=(320, 240), crop_ratio=0.6):
    """
    Preprocess CSI camera frame for CNN.

    Args:
        frame_bgr: (H, W, 3) BGR from camera
        target_size: (W, H) output size
        crop_ratio: Keep bottom portion (road region)

    Returns:
        tensor: (1, 3, H, W) normalized RGB tensor
        gray: (H, W) grayscale for change detection
    """
    import cv2
    import numpy as np

    H, W = frame_bgr.shape[:2]

    # Crop bottom portion (road region)
    y0 = int(H * (1 - crop_ratio))
    roi = frame_bgr[y0:, :, :]

    # Resize
    resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_LINEAR)

    # Grayscale for change detection
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # RGB normalized tensor
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))  # (3, H, W)
    tensor = np.expand_dims(tensor, 0)  # (1, 3, H, W)

    return tensor, gray


def export_to_onnx(model, output_path, input_size=(1, 3, 240, 320)):
    """
    Export model to ONNX for TensorRT optimization.

    Args:
        model: CameraCNN instance
        output_path: .onnx file path
        input_size: (B, C, H, W) - note H, W order
    """
    model.eval()
    dummy = torch.randn(*input_size)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=['image'],
        output_names=['feature'],
        dynamic_axes={
            'image': {0: 'batch'},
            'feature': {0: 'batch'}
        },
        opset_version=11,
        do_constant_folding=True
    )
    print(f"Exported to {output_path}")
    print(f"  Input: {input_size}")
    print(f"  Output: (batch, {model.fc.out_features})")


if __name__ == '__main__':
    import time

    print("=" * 60)
    print("CameraCNN for CSI SD Camera - Model Analysis")
    print("=" * 60)

    # Test different configurations
    configs = [
        ("CameraCNN (320x240)", CameraCNN(feature_dim=64), (1, 3, 240, 320)),
        ("CameraCNN (640x480)", CameraCNN(feature_dim=64), (1, 3, 480, 640)),
        ("CameraCNNTiny (160x120)", CameraCNNTiny(feature_dim=32), (1, 3, 120, 160)),
    ]

    for name, model, input_shape in configs:
        print(f"\n{name}")
        print("-" * 40)

        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,} ({params/1e3:.1f}K)")

        # Test forward pass
        x = torch.randn(*input_shape)
        model.eval()

        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = model(x)

            # Benchmark
            times = []
            for _ in range(20):
                start = time.perf_counter()
                y = model(x)
                times.append((time.perf_counter() - start) * 1000)

        print(f"Input: {input_shape}")
        print(f"Output: {y.shape}")
        print(f"Inference: {sum(times)/len(times):.2f}ms (CPU)")

    # Export main model
    print("\n" + "=" * 60)
    print("Exporting to ONNX...")
    model = CameraCNN(feature_dim=64)
    export_to_onnx(model, "camera_cnn_sd.onnx", (1, 3, 240, 320))
