#!/usr/bin/env python3
"""
Alpamayo V14-Lite: LiDAR-Centric E2E Architecture
=================================================

Tesla V14 스타일이지만 Jetson Orin Nano 제약 조건에 맞춤:
- LiDAR 중심 처리 (주 센서)
- 카메라는 선택적 보조 센서 (나중에 확장 가능)
- ~150K 파라미터
- <10ms 추론 목표

설계 철학:
1. LiDAR가 90% 처리 (RC카 스케일에서 충분)
2. 카메라는 semantic hint만 (있으면 좋음)
3. Occupancy + Flow 통합
4. Deterministic planning

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    2D LiDAR (LaserScan)                      │
│                           ↓                                  │
│              ┌───────────────────────┐                       │
│              │   LiDAR BEV Encoder   │  ~50K params          │
│              │   (Conv + ConvGRU)    │                       │
│              └───────────────────────┘                       │
│                           ↓                                  │
│                    [B, 64, 50, 60]                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    ┌───────────────┐
                    │  (Optional)   │
┌───────────────────│ Camera Hint  │────────────────────────┐
│                   └───────────────┘                        │
│              ┌───────────────────────┐                     │
│              │   Light CNN Encoder   │  ~30K params        │
│              │   (semantic hints)    │                     │
│              └───────────────────────┘                     │
│                           ↓                                │
│                    [B, 32, 50, 60]                          │
└────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    OccupancyNet                              │
│              ┌───────────────────────┐                       │
│              │   2.5D Occupancy      │  ~30K params          │
│              │   + Flow Prediction   │                       │
│              └───────────────────────┘                       │
│                           ↓                                  │
│              - Occupancy [B, Z, H, W]                        │
│              - Flow [B, 2, H, W]                             │
│              - Risk Map [B, 1, H, W]                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    E2E Planner                               │
│              ┌───────────────────────┐                       │
│              │   Cost Volume +       │  ~40K params          │
│              │   Trajectory Head     │                       │
│              └───────────────────────┘                       │
│                           ↓                                  │
│              - Trajectory [B, T, 2]                          │
│              - Control (steering, speed)                     │
└─────────────────────────────────────────────────────────────┘

Total: ~150K parameters
Target: <10ms on Orin Nano
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List


# =============================================================================
# NPU-Optimized Building Blocks
# =============================================================================

class ConvBNAct(nn.Module):
    """Standard Conv-BN-Activation block"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1, act: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise Separable Convolution - NPU friendly"""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = ConvBNAct(in_ch, in_ch, 3, stride, groups=in_ch)
        self.pw = ConvBNAct(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class SEBlock(nn.Module):
    """Squeeze-Excitation block"""
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // reduction),
            nn.SiLU(inplace=True),
            nn.Linear(ch // reduction, ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * w


class ConvGRUCell(nn.Module):
    """Convolutional GRU for temporal encoding"""
    def __init__(self, in_ch: int, hidden_ch: int):
        super().__init__()
        self.hidden_ch = hidden_ch
        combined = in_ch + hidden_ch

        self.gates = nn.Conv2d(combined, hidden_ch * 2, 3, 1, 1, bias=False)
        self.candidate = nn.Conv2d(combined, hidden_ch, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_ch, x.size(2), x.size(3),
                           device=x.device, dtype=x.dtype)

        combined = torch.cat([x, h], dim=1)
        gates = torch.sigmoid(self.gates(combined))
        r, z = gates.chunk(2, dim=1)

        combined_r = torch.cat([x, r * h], dim=1)
        candidate = torch.tanh(self.candidate(combined_r))

        return (1 - z) * h + z * candidate


# =============================================================================
# LiDAR BEV Encoder (Main Sensor)
# =============================================================================

class LiDARBEVEncoder(nn.Module):
    """
    LiDAR to BEV feature encoder
    주 센서로서 모든 spatial reasoning 담당

    Input: [B, 2, 200, 240] (occupancy + intensity)
    Output: [B, 64, 50, 60]

    BEV Grid specs:
    - 240x200 cells → 50x60 features
    - 5cm resolution
    - X: [-2m, 10m], Y: [-5m, 5m]
    """
    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 32,
        out_channels: int = 64
    ):
        super().__init__()

        # Stem: 200x240 → 100x120
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, 16, 3, 2),
            ConvBNAct(16, hidden_channels, 3, 1),
        )

        # Stage 1: 100x120 → 50x60
        self.stage1 = nn.Sequential(
            DWConv(hidden_channels, hidden_channels, stride=2),
            DWConv(hidden_channels, hidden_channels),
            SEBlock(hidden_channels),
        )

        # Stage 2: Feature refinement
        self.stage2 = nn.Sequential(
            DWConv(hidden_channels, out_channels),
            DWConv(out_channels, out_channels),
            SEBlock(out_channels),
        )

        # Temporal GRU
        self.gru = ConvGRUCell(out_channels, out_channels)

    def forward(self, bev: torch.Tensor,
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bev: [B, 2, 200, 240] BEV grid
            hidden: [B, 64, 50, 60] GRU hidden state

        Returns:
            features: [B, 64, 50, 60]
            new_hidden: [B, 64, 50, 60]
        """
        x = self.stem(bev)
        x = self.stage1(x)
        x = self.stage2(x)

        # Temporal encoding
        new_hidden = self.gru(x, hidden)

        return new_hidden, new_hidden


# =============================================================================
# Camera Encoder (Optional - Semantic Hints)
# =============================================================================

class CameraHintEncoder(nn.Module):
    """
    Lightweight camera encoder for semantic hints only
    카메라가 없어도 LiDAR만으로 동작 가능

    Input: [B, 3, 240, 320]
    Output: [B, 32, 50, 60] (BEV aligned)
    """
    def __init__(self, out_channels: int = 32):
        super().__init__()

        # Very light encoder
        self.encoder = nn.Sequential(
            # 240x320 → 60x80
            ConvBNAct(3, 16, 3, stride=4),
            DWConv(16, 24),

            # 60x80 → 30x40
            DWConv(24, 32, stride=2),
            SEBlock(32),

            # 30x40 → 15x20
            DWConv(32, out_channels, stride=2),
        )

        # Project to BEV size
        self.to_bev = nn.Sequential(
            nn.AdaptiveAvgPool2d((50, 60)),
            ConvBNAct(out_channels, out_channels, 1)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [B, 3, 240, 320]

        Returns:
            hints: [B, 32, 50, 60]
        """
        x = self.encoder(image)
        x = self.to_bev(x)
        return x


# =============================================================================
# Sensor Fusion (Optional camera enhancement)
# =============================================================================

class SensorFusion(nn.Module):
    """
    Fuses LiDAR features with optional camera hints
    카메라가 없으면 LiDAR만 사용
    """
    def __init__(self, lidar_ch: int = 64, camera_ch: int = 32, out_ch: int = 64):
        super().__init__()

        self.lidar_ch = lidar_ch
        self.camera_ch = camera_ch

        # LiDAR only path
        self.lidar_proj = ConvBNAct(lidar_ch, out_ch, 1)

        # Camera enhancement (gating)
        self.camera_gate = nn.Sequential(
            ConvBNAct(camera_ch, out_ch, 1),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.Sigmoid()
        )

        # Fusion
        self.fusion = ConvBNAct(out_ch, out_ch, 3)

    def forward(self, lidar_feat: torch.Tensor,
                camera_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            lidar_feat: [B, 64, 50, 60]
            camera_feat: [B, 32, 50, 60] or None

        Returns:
            fused: [B, 64, 50, 60]
        """
        x = self.lidar_proj(lidar_feat)

        if camera_feat is not None:
            # Camera provides attention weights
            gate = self.camera_gate(camera_feat)
            x = x * (1 + gate)  # Residual gating

        return self.fusion(x)


# =============================================================================
# 2.5D Occupancy Network
# =============================================================================

class OccupancyNet(nn.Module):
    """
    2.5D Occupancy prediction (height-aware 2D)
    RC카 스케일에서는 full 3D가 필요 없음

    Outputs:
    - occupancy: 장애물 확률
    - flow: 동적 객체 모션
    - risk: 충돌 위험도 맵
    """
    def __init__(
        self,
        in_channels: int = 64,
        num_height_bins: int = 4,  # RC카: 0-40cm만 관심
    ):
        super().__init__()
        self.num_height_bins = num_height_bins

        # Height-aware processing
        self.height_conv = nn.Sequential(
            DWConv(in_channels, in_channels),
            nn.Conv2d(in_channels, in_channels * num_height_bins, 1)
        )

        # Occupancy head
        self.occ_head = nn.Sequential(
            ConvBNAct(in_channels * num_height_bins, 32, 3),
            nn.Conv2d(32, num_height_bins, 1),
            nn.Sigmoid()
        )

        # Flow head (2D motion vectors)
        self.flow_head = nn.Sequential(
            ConvBNAct(in_channels, 32, 3),
            nn.Conv2d(32, 2, 1),  # dx, dy
            nn.Tanh()  # Normalized motion
        )

        # Risk/cost map
        self.risk_head = nn.Sequential(
            ConvBNAct(in_channels, 16, 3),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, 64, 50, 60]

        Returns:
            occupancy: [B, Z, 50, 60]
            flow: [B, 2, 50, 60]
            risk: [B, 1, 50, 60]
        """
        # Height-aware occupancy
        h_feat = self.height_conv(features)
        occupancy = self.occ_head(h_feat)

        # Motion flow
        flow = self.flow_head(features) * 0.5  # Scale to ±0.5 m/frame

        # Risk map (combines static + dynamic)
        risk = self.risk_head(features)

        return {
            'occupancy': occupancy,
            'flow': flow,
            'risk': risk
        }


# =============================================================================
# E2E Trajectory Planner
# =============================================================================

class TrajectoryPlanner(nn.Module):
    """
    End-to-End trajectory planning
    Cost volume + deterministic trajectory prediction

    No diffusion, no sampling - pure feedforward
    """
    def __init__(
        self,
        in_channels: int = 64,
        num_waypoints: int = 8,  # 0.8s @ 10Hz
        waypoint_dt: float = 0.1,  # 100ms intervals
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_dt = waypoint_dt

        # Cost volume encoder
        self.cost_encoder = nn.Sequential(
            DWConv(in_channels + 1, 32),  # +1 for risk map
            DWConv(32, 32),
            nn.AdaptiveAvgPool2d((10, 12)),
            nn.Flatten()
        )

        cost_feat_dim = 32 * 10 * 12

        # Trajectory head
        self.traj_head = nn.Sequential(
            nn.Linear(cost_feat_dim, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, num_waypoints * 2)
        )

        # Direct control (for immediate actuation)
        self.control_head = nn.Sequential(
            nn.Linear(cost_feat_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 3)  # steering, speed, confidence
        )

    def forward(self, features: torch.Tensor,
                risk_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, 64, 50, 60]
            risk_map: [B, 1, 50, 60]

        Returns:
            waypoints: [B, T, 2]
            steering: [B, 1]
            speed: [B, 1]
            confidence: [B, 1]
        """
        # Combine features with risk
        x = torch.cat([features, risk_map], dim=1)
        x = self.cost_encoder(x)

        # Trajectory
        waypoints = self.traj_head(x)
        waypoints = waypoints.view(-1, self.num_waypoints, 2)

        # Scale waypoints to meters
        waypoints = waypoints * 0.5  # ±0.5m per waypoint

        # Control
        ctrl = self.control_head(x)
        steering = torch.tanh(ctrl[:, 0:1]) * 0.52  # ±30°
        speed = torch.sigmoid(ctrl[:, 1:2]) * 2.0   # 0-2 m/s
        confidence = torch.sigmoid(ctrl[:, 2:3])

        return {
            'waypoints': waypoints,
            'steering': steering,
            'speed': speed,
            'confidence': confidence
        }


# =============================================================================
# Safety Layer
# =============================================================================

class SafetyLayer(nn.Module):
    """
    Rule-based safety constraints
    신경망 출력에 물리적 제약 적용
    """
    def __init__(
        self,
        max_steering: float = 0.52,   # 30°
        max_speed: float = 2.0,       # m/s
        min_obstacle_dist: float = 0.3,  # meters
    ):
        super().__init__()
        self.max_steering = max_steering
        self.max_speed = max_speed
        self.min_obstacle_dist = min_obstacle_dist

    def forward(
        self,
        steering: torch.Tensor,
        speed: torch.Tensor,
        risk: torch.Tensor,
        confidence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply safety constraints

        Args:
            steering: [B, 1] raw steering
            speed: [B, 1] raw speed
            risk: [B, 1, H, W] risk map
            confidence: [B, 1] prediction confidence

        Returns:
            safe_steering: [B, 1]
            safe_speed: [B, 1]
        """
        # Clamp steering
        safe_steering = torch.clamp(steering, -self.max_steering, self.max_steering)

        # Reduce speed based on risk
        front_risk = risk[:, :, :25, 25:35].mean(dim=(1, 2, 3), keepdim=True)
        risk_factor = 1.0 - front_risk.squeeze(-1).squeeze(-1)

        # Reduce speed based on confidence
        conf_factor = 0.5 + 0.5 * confidence

        # Reduce speed on sharp turns
        turn_factor = 1.0 - 0.3 * (steering.abs() / self.max_steering)

        # Apply all factors
        safe_speed = speed * risk_factor * conf_factor * turn_factor
        safe_speed = torch.clamp(safe_speed, 0, self.max_speed)

        return safe_steering, safe_speed


# =============================================================================
# Complete V14-Lite Model
# =============================================================================

class AlpamayoV14Lite(nn.Module):
    """
    LiDAR-centric End-to-End model for RC car

    Features:
    - ~150K parameters
    - LiDAR as primary sensor
    - Optional camera enhancement
    - 2.5D occupancy + flow
    - Deterministic planning
    - Built-in safety layer

    Usage:
        model = AlpamayoV14Lite()

        # LiDAR only
        outputs = model(lidar_bev=bev)

        # With camera
        outputs = model(lidar_bev=bev, camera=img)
    """
    def __init__(
        self,
        bev_size: Tuple[int, int] = (200, 240),  # Input BEV size
        use_camera: bool = True,
        num_waypoints: int = 8,
    ):
        super().__init__()
        self.use_camera = use_camera

        # LiDAR encoder (main)
        self.lidar_encoder = LiDARBEVEncoder(
            in_channels=2,
            hidden_channels=32,
            out_channels=64
        )

        # Camera encoder (optional)
        if use_camera:
            self.camera_encoder = CameraHintEncoder(out_channels=32)
            self.fusion = SensorFusion(lidar_ch=64, camera_ch=32, out_ch=64)
        else:
            self.camera_encoder = None
            self.fusion = SensorFusion(lidar_ch=64, camera_ch=32, out_ch=64)

        # Occupancy
        self.occupancy = OccupancyNet(in_channels=64)

        # Planner
        self.planner = TrajectoryPlanner(
            in_channels=64,
            num_waypoints=num_waypoints
        )

        # Safety
        self.safety = SafetyLayer()

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        lidar_bev: torch.Tensor,
        camera: Optional[torch.Tensor] = None,
        lidar_hidden: Optional[torch.Tensor] = None,
        apply_safety: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            lidar_bev: [B, 2, 200, 240] BEV grid
            camera: [B, 3, 240, 320] optional camera image
            lidar_hidden: [B, 64, 50, 60] GRU hidden state
            apply_safety: whether to apply safety constraints

        Returns:
            Dict with all outputs
        """
        # LiDAR encoding (always)
        lidar_feat, new_hidden = self.lidar_encoder(lidar_bev, lidar_hidden)

        # Camera encoding (optional)
        camera_feat = None
        if camera is not None and self.camera_encoder is not None:
            camera_feat = self.camera_encoder(camera)

        # Fusion
        fused_feat = self.fusion(lidar_feat, camera_feat)

        # Occupancy prediction
        occ_out = self.occupancy(fused_feat)

        # Planning
        plan_out = self.planner(fused_feat, occ_out['risk'])

        # Safety layer
        if apply_safety:
            safe_steering, safe_speed = self.safety(
                plan_out['steering'],
                plan_out['speed'],
                occ_out['risk'],
                plan_out['confidence']
            )
        else:
            safe_steering = plan_out['steering']
            safe_speed = plan_out['speed']

        return {
            # Perception
            'occupancy': occ_out['occupancy'],
            'flow': occ_out['flow'],
            'risk': occ_out['risk'],

            # Planning
            'waypoints': plan_out['waypoints'],
            'confidence': plan_out['confidence'],

            # Control
            'steering': safe_steering,
            'speed': safe_speed,
            'raw_steering': plan_out['steering'],
            'raw_speed': plan_out['speed'],

            # State
            'lidar_hidden': new_hidden,
            'features': fused_feat
        }


# =============================================================================
# Training Utilities
# =============================================================================

class V14LiteLoss(nn.Module):
    """Combined loss for V14-Lite training"""
    def __init__(
        self,
        occ_weight: float = 1.0,
        flow_weight: float = 0.5,
        traj_weight: float = 1.0,
        ctrl_weight: float = 2.0,
    ):
        super().__init__()
        self.occ_weight = occ_weight
        self.flow_weight = flow_weight
        self.traj_weight = traj_weight
        self.ctrl_weight = ctrl_weight

        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: model outputs
            targets: ground truth
                - occ_gt: [B, Z, H, W]
                - flow_gt: [B, 2, H, W]
                - waypoints_gt: [B, T, 2]
                - steering_gt: [B, 1]
                - speed_gt: [B, 1]
        """
        losses = {}

        # Occupancy loss
        if 'occ_gt' in targets:
            losses['occ'] = self.bce(outputs['occupancy'], targets['occ_gt'])

        # Flow loss
        if 'flow_gt' in targets:
            losses['flow'] = self.mse(outputs['flow'], targets['flow_gt'])

        # Trajectory loss
        if 'waypoints_gt' in targets:
            losses['traj'] = self.l1(outputs['waypoints'], targets['waypoints_gt'])

        # Control loss (most important for imitation learning)
        if 'steering_gt' in targets:
            losses['steering'] = self.mse(outputs['steering'], targets['steering_gt'])

        if 'speed_gt' in targets:
            losses['speed'] = self.mse(outputs['speed'], targets['speed_gt'])

        # Total loss
        total = 0
        if 'occ' in losses:
            total += self.occ_weight * losses['occ']
        if 'flow' in losses:
            total += self.flow_weight * losses['flow']
        if 'traj' in losses:
            total += self.traj_weight * losses['traj']
        if 'steering' in losses:
            total += self.ctrl_weight * losses['steering']
        if 'speed' in losses:
            total += self.ctrl_weight * losses['speed']

        losses['total'] = total
        return losses


# =============================================================================
# Model Info & Testing
# =============================================================================

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> Dict:
    """Get detailed model summary"""
    info = {
        'total': count_params(model),
        'modules': {}
    }

    for name, child in model.named_children():
        info['modules'][name] = count_params(child)

    return info


if __name__ == '__main__':
    print("=" * 60)
    print("Alpamayo V14-Lite Test")
    print("=" * 60)

    # Create model
    model = AlpamayoV14Lite(use_camera=True)
    model.eval()

    # Summary
    info = model_summary(model)
    print(f"\nTotal parameters: {info['total']/1000:.1f}K")
    print("\nModule breakdown:")
    for name, params in info['modules'].items():
        print(f"  {name}: {params/1000:.1f}K")

    # Test inference
    print("\n" + "=" * 60)
    print("Inference Test")
    print("=" * 60)

    B = 1
    lidar = torch.randn(B, 2, 200, 240)
    camera = torch.randn(B, 3, 240, 320)

    # LiDAR only
    with torch.no_grad():
        out1 = model(lidar_bev=lidar)
    print("\nLiDAR only outputs:")
    for k, v in out1.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {list(v.shape)}")

    # With camera
    with torch.no_grad():
        out2 = model(lidar_bev=lidar, camera=camera)
    print("\nWith camera outputs:")
    for k, v in out2.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {list(v.shape)}")

    # Benchmark
    print("\n" + "=" * 60)
    print("Benchmark (CPU)")
    print("=" * 60)

    import time

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(lidar_bev=lidar)

    # LiDAR only benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(lidar_bev=lidar)
        times.append((time.perf_counter() - start) * 1000)

    import numpy as np
    times = np.array(times)
    print(f"\nLiDAR only:")
    print(f"  Mean: {times.mean():.2f} ms")
    print(f"  Std:  {times.std():.2f} ms")

    # With camera benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(lidar_bev=lidar, camera=camera)
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    print(f"\nWith camera:")
    print(f"  Mean: {times.mean():.2f} ms")
    print(f"  Std:  {times.std():.2f} ms")
