# Alpamayo - Autonomous RC Car Platform

End-to-end autonomous driving pipeline for Jetson Orin Nano with CSI camera and 2D LiDAR.

**[Live Demo](https://hwkim3330.github.io/Alpamayo-for-orin-nano/)** | **[Dashboard](https://hwkim3330.github.io/Alpamayo-for-orin-nano/fsd/)** | **[BEV Viewer](https://hwkim3330.github.io/Alpamayo-for-orin-nano/fsd/bev.html)**

## Overview

Alpamayo is a Tesla FSD V14-inspired End-to-End autonomous driving system optimized for RC car scale. Key features:

- **V14-Lite Architecture**: LiDAR-centric E2E model (~150K params)
- **Occupancy Network**: 2.5D occupancy + flow prediction
- **Deterministic Planning**: No diffusion, pure feedforward
- **Safety Layer**: Rule-based constraints on neural outputs
- **TensorRT Optimized**: <10ms inference on Orin Nano

## Model Architectures

### V14-Lite (Recommended)

LiDAR-centric design, camera as optional enhancement:

```
┌─────────────────────────────────────────────────────────────┐
│                    2D LiDAR (LaserScan)                     │
│                           ↓                                 │
│              ┌───────────────────────┐                      │
│              │   LiDAR BEV Encoder   │  ~50K params         │
│              │   (Conv + ConvGRU)    │                      │
│              └───────────────────────┘                      │
│                           ↓                                 │
│                    [B, 64, 50, 60]                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
              ┌─────────────────────────┐
              │  (Optional) Camera Hint │  ~30K params
              └─────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    OccupancyNet                             │
│              ┌───────────────────────┐                      │
│              │   2.5D Occupancy      │  ~30K params         │
│              │   + Flow + Risk       │                      │
│              └───────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    E2E Planner                              │
│              ┌───────────────────────┐                      │
│              │   Cost Volume +       │  ~40K params         │
│              │   Trajectory Head     │                      │
│              └───────────────────────┘                      │
│                           ↓                                 │
│              Waypoints + Steering + Speed                   │
└─────────────────────────────────────────────────────────────┘

Total: ~150K parameters | Target: <10ms on Orin Nano
```

### Classic Architecture

Original camera + LiDAR fusion model:

```
┌──────────────┐     ┌──────────────┐
│  CSI Camera  │     │   2D LiDAR   │
│  320x240 RGB │     │ RPLIDAR/YD   │
└──────┬───────┘     └──────┬───────┘
       ↓                    ↓
┌──────────────┐     ┌──────────────┐
│  CameraCNN   │     │  BEV Grid    │
│  ~100K params│     │  + ConvGRU   │
└──────┬───────┘     └──────┬───────┘
       └────────┬───────────┘
                ↓
        ┌───────────────┐
        │ Feature       │
        │ Attention     │
        └───────┬───────┘
                ↓
        ┌───────────────┐
        │ PlannerHead   │
        │ + SafetyLayer │
        └───────┬───────┘
                ↓
        Steering + Speed
```

## Specifications

| Component | V14-Lite | Classic |
|-----------|----------|---------|
| Parameters | ~150K | ~200K |
| Inference | <10ms | <15ms |
| Primary Sensor | LiDAR | Both |
| Occupancy | 2.5D | BEV only |
| Flow Prediction | Yes | No |

| Shared Specs | Value |
|--------------|-------|
| Control Frequency | 20 Hz |
| Camera Resolution | 320x240 |
| BEV Grid | 240x200, 5cm |
| LiDAR Range | X: [-2m, 10m], Y: [-5m, 5m] |

## Project Structure

```
Alpamayo-for-orin-nano/
├── models/
│   ├── v14_lite.py           # V14-Lite E2E model (recommended)
│   ├── v14_architecture.py   # Full V14 reference
│   ├── camera_cnn.py         # Camera encoder
│   ├── lidar_temporal.py     # LiDAR encoder
│   └── planner_head.py       # Classic planner
├── scripts/
│   ├── train_v14.py          # Training script
│   ├── export_onnx.py        # ONNX export
│   ├── build_tensorrt.py     # TensorRT conversion
│   └── inference_trt.py      # TensorRT inference
├── docs/                      # GitHub Pages
│   ├── index.html            # Landing page
│   ├── fsd/index.html        # FSD dashboard
│   ├── fsd/bev.html          # BEV/LiDAR viewer
│   ├── css/                  # Stylesheets
│   └── js/                   # JavaScript + libs
├── web/                       # Local web dashboard
├── launch/                    # ROS2 launch files
└── config/                    # Configuration
```

## Requirements

### Hardware
- Jetson Orin Nano (8GB recommended)
- 2D LiDAR (RPLIDAR A1/A2/A3 or YDLidar X2/X4) - **Required**
- CSI Camera (IMX219, IMX477) - **Optional**
- RC Car chassis with servo steering

### Software
- JetPack 5.x or 6.x
- ROS2 Humble/Jazzy
- Python 3.8+
- PyTorch 2.0+
- TensorRT (included with JetPack)

## Installation

```bash
# Clone repository
cd ~/ros2_ws/src
git clone https://github.com/hwkim3330/Alpamayo-for-orin-nano.git

# Install Python dependencies
pip3 install torch torchvision numpy opencv-python

# Install ROS2 dependencies
sudo apt install ros-humble-rosbridge-server ros-humble-web-video-server

# Build
cd ~/ros2_ws
colcon build --packages-select ros2_web_dashboard
source install/setup.bash
```

## Quick Start

### 1. Training (Imitation Learning)

```bash
# Online training with ROS2
python3 scripts/train_v14.py --mode online

# Offline training from dataset
python3 scripts/train_v14.py --mode offline --data_dir ./data --epochs 100
```

### 2. Export to ONNX

```bash
python3 scripts/export_onnx.py --combined
# or
python3 scripts/train_v14.py --export --checkpoint checkpoints/best.pth
```

### 3. Build TensorRT Engine (on Jetson)

```bash
python3 scripts/build_tensorrt.py --onnx_dir ./onnx_models --fp16
```

### 4. Run Inference

```bash
# Benchmark
python3 scripts/inference_trt.py --benchmark

# With ROS2
ros2 launch ros2_web_dashboard jetson_orin.launch.py
```

### 5. Web Dashboard

Open browser: `http://<jetson-ip>:8000/fsd/`

## Model Details

### V14-Lite (`models/v14_lite.py`)

Tesla FSD V14-inspired architecture:

- **LiDARBEVEncoder**: Main perception backbone
  - Input: 2-channel BEV (occupancy + intensity)
  - ConvGRU for temporal encoding
  - Output: 64-dim spatial features

- **CameraHintEncoder** (Optional): Semantic enhancement
  - Lightweight CNN for hints only
  - Gating mechanism for fusion

- **OccupancyNet**: Unified perception
  - 2.5D occupancy (4 height bins for RC car)
  - Flow field for dynamic objects
  - Risk/cost map for planning

- **TrajectoryPlanner**: E2E planning
  - Cost volume encoding
  - Deterministic trajectory MLP
  - Direct control outputs

- **SafetyLayer**: Rule-based constraints
  - Speed reduction on high risk
  - Turn-based speed limiting
  - Confidence-based modulation

### Usage

```python
from models.v14_lite import AlpamayoV14Lite

model = AlpamayoV14Lite(use_camera=True)

# LiDAR only
outputs = model(lidar_bev=bev)

# With camera
outputs = model(lidar_bev=bev, camera=img)

# Outputs
steering = outputs['steering']      # ±30° in radians
speed = outputs['speed']            # 0-2 m/s
waypoints = outputs['waypoints']    # [B, 8, 2] future positions
risk = outputs['risk']              # [B, 1, 50, 60] risk map
```

## ROS2 Topics

### Subscribed
| Topic | Type | Description |
|-------|------|-------------|
| `/scan` | sensor_msgs/LaserScan | LiDAR input (required) |
| `/camera/image_raw` | sensor_msgs/Image | Camera input (optional) |
| `/cmd_vel` | geometry_msgs/Twist | Teacher signal |

### Published
| Topic | Type | Description |
|-------|------|-------------|
| `/planner/cmd` | geometry_msgs/Vector3 | Control output |
| `/planner/trajectory` | nav_msgs/Path | Planned trajectory |
| `/planner/risk_map` | sensor_msgs/Image | Risk visualization |

## Configuration

Edit `config/v14_params.yaml`:

```yaml
v14_lite:
  use_camera: true
  num_waypoints: 8

lidar:
  bev_width: 240
  bev_height: 200
  resolution: 0.05
  x_range: [-2.0, 10.0]
  y_range: [-5.0, 5.0]

safety:
  max_steering: 0.52  # 30 degrees
  max_speed: 2.0
  min_obstacle_dist: 0.3

training:
  learning_rate: 0.001
  batch_size: 16
  buffer_size: 10000
```

## Performance

### Orin Nano Benchmarks

| Model | CPU | GPU (FP32) | GPU (FP16) | TensorRT |
|-------|-----|------------|------------|----------|
| V14-Lite | ~80ms | ~15ms | ~8ms | ~5ms |
| Classic | ~100ms | ~20ms | ~12ms | ~8ms |

Target: **20Hz (50ms)** - Both models easily meet requirement.

## Comparison with Tesla FSD V14

| Aspect | Tesla V14 | Alpamayo V14-Lite |
|--------|-----------|-------------------|
| Cameras | 8 | 1 (optional) |
| LiDAR | None | Primary sensor |
| Parameters | ~300-500M | ~150K |
| Occupancy | Full 3D | 2.5D |
| Flow | 3D voxel | 2D BEV |
| Hardware | Custom NPU | Orin Nano |
| Latency | 10-20ms | <10ms |

## License

MIT License

## Acknowledgments

- [Tesla AI Day](https://www.tesla.com/AI) - V14 architecture inspiration
- [Robot Web Tools](https://robotwebtools.github.io/)
- [PyTorch](https://pytorch.org/)
- [ROS2](https://docs.ros.org/)
