# Alpamayo - Autonomous RC Car Platform

End-to-end autonomous driving pipeline for Jetson Orin Nano with CSI camera and 2D LiDAR.

**[Live Demo](https://hwkim3330.github.io/Alpamayo-for-orin-nano/)** | **[Dashboard](https://hwkim3330.github.io/Alpamayo-for-orin-nano/fsd/)**

## Overview

Alpamayo is a lightweight perception-to-control system designed for RC car scale autonomous driving. It features:

- **CameraCNN**: EfficientNet-inspired encoder for SD camera (320x240)
- **LiDAR Temporal Encoder**: BEV grid with ConvGRU for 2D LiDAR
- **Sensor Fusion**: Cross-modal attention between camera and LiDAR
- **Online Learning**: Imitation learning from teacher signals

## Model Architecture

```
┌──────────────┐     ┌──────────────┐
│  CSI Camera  │     │   2D LiDAR   │
│  320x240 RGB │     │ RPLIDAR/YD   │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
┌──────────────┐     ┌──────────────┐
│  CameraCNN   │     │  BEV Grid    │
│  ~100K params│     │  240x200     │
│  → 64-dim    │     │  5cm res     │
└──────┬───────┘     └──────┬───────┘
       │                    │
       │             ┌──────▼───────┐
       │             │   ConvGRU    │
       │             │  ~50K params │
       │             │  → 32-dim    │
       │             └──────┬───────┘
       │                    │
       └────────┬───────────┘
                │
        ┌───────▼───────┐
        │ Feature       │
        │ Attention     │
        └───────┬───────┘
                │
        ┌───────▼───────┐
        │ PlannerHead   │
        │ + SafetyLayer │
        └───────┬───────┘
                │
        ┌───────┴───────┐
        ▼               ▼
   ┌─────────┐    ┌─────────┐
   │Steering │    │  Speed  │
   │ ±30°    │    │ 0-2 m/s │
   └─────────┘    └─────────┘
```

## Specifications

| Component | Value |
|-----------|-------|
| Total Parameters | ~150K |
| Inference Time | <15ms |
| Control Frequency | 20 Hz |
| Camera Resolution | 320x240 |
| BEV Grid Size | 240x200 |
| Grid Resolution | 5cm |
| LiDAR Range | X: [-2m, 10m], Y: [-5m, 5m] |

## Project Structure

```
Alpamayo-for-orin-nano/
├── models/                    # PyTorch model definitions
│   ├── camera_cnn.py         # Camera feature extractor
│   ├── lidar_temporal.py     # LiDAR BEV + ConvGRU encoder
│   └── planner_head.py       # Sensor fusion + planner
├── scripts/                   # ROS2 nodes
│   ├── camera_feature_node.py
│   ├── lidar_temporal_node.py
│   ├── planner_node.py
│   └── ...
├── launch/                    # ROS2 launch files
├── web/                       # Web dashboard (local)
│   ├── fsd/                  # FSD-style dashboard
│   └── index.html            # Main ROS2 dashboard
├── docs/                      # GitHub Pages
│   ├── index.html            # Landing page
│   ├── fsd/                  # Demo dashboard
│   └── css/, js/             # Assets
└── config/                    # Configuration files
```

## Requirements

### Hardware
- Jetson Orin Nano (8GB recommended)
- CSI Camera (e.g., IMX219, IMX477)
- 2D LiDAR (RPLIDAR A1/A2/A3 or YDLidar X2/X4)
- RC Car chassis with servo steering

### Software
- JetPack 5.x or 6.x
- ROS2 Humble/Jazzy
- Python 3.8+
- PyTorch 2.0+

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

## Usage

### Launch the system
```bash
# Full pipeline
ros2 launch ros2_web_dashboard dashboard.launch.py

# Jetson-optimized
ros2 launch ros2_web_dashboard jetson_orin.launch.py
```

### Web Dashboard
1. Open browser: `http://<jetson-ip>:8000`
2. Or use the FSD dashboard: `http://<jetson-ip>:8000/fsd/`
3. Enter ROS Bridge IP and port
4. Click Connect

### Training (Imitation Learning)
```bash
# Start training node with teacher input
ros2 run ros2_web_dashboard planner_node.py

# Publish teacher signals
ros2 topic pub /training/teacher geometry_msgs/Vector3 "{x: 0.1, y: 1.0, z: 0}"
```

## Model Details

### CameraCNN (`models/camera_cnn.py`)
- Input: RGB image (320x240)
- Architecture: MobileNetV2-style Inverted Residuals + SE blocks
- Output: 64-dimensional feature vector
- Inference: ~5ms on Orin Nano

### LiDARTemporalEncoder (`models/lidar_temporal.py`)
- Input: 2-channel BEV grid (occupancy, intensity)
- Grid: 240x200 cells, 5cm resolution
- Temporal: ConvGRU for motion encoding
- Output: 32-dimensional feature vector
- Inference: ~3ms on Orin Nano

### PlannerHead (`models/planner_head.py`)
- Input: Concatenated camera + LiDAR features
- Architecture: MLP with cross-modal attention
- Safety: Speed reduction on high steering/uncertainty
- Output: Steering angle (rad) + Speed (m/s)

## ROS2 Topics

### Subscribed
| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | sensor_msgs/Image | Camera input |
| `/scan` | sensor_msgs/LaserScan | LiDAR input |
| `/training/teacher` | geometry_msgs/Vector3 | Teacher signal |

### Published
| Topic | Type | Description |
|-------|------|-------------|
| `/planner/cmd` | geometry_msgs/Vector3 | Control output |
| `/camera/feature` | std_msgs/Float32MultiArray | Camera features |
| `/lidar/feature` | std_msgs/Float32MultiArray | LiDAR features |

## Configuration

Edit `config/dashboard_params.yaml`:

```yaml
camera_feature_node:
  input_topic: /camera/image_raw
  feature_dim: 64
  diff_thresh: 10.0      # Change detection threshold
  min_interval: 0.2      # Minimum CNN interval (sec)

lidar_temporal_node:
  input_topic: /scan
  feature_dim: 32
  hidden_channels: 32

planner_node:
  lidar_dim: 32
  cam_dim: 64
  publish_rate: 20.0
  learning_rate: 0.001
```

## ONNX Export

```python
from models.planner_head import export_to_onnx
from models.camera_cnn import CameraCNN
from models.lidar_temporal import LiDARTemporalEncoder

camera = CameraCNN(feature_dim=64)
lidar = LiDARTemporalEncoder(feature_dim=32)
planner = PlannerHead(lidar_dim=32, cam_dim=64)

export_to_onnx(camera, lidar, planner, "alpamayo_pipeline.onnx")
```

## License

MIT License

## Acknowledgments

- [Robot Web Tools](https://robotwebtools.github.io/)
- [PyTorch](https://pytorch.org/)
- [ROS2](https://docs.ros.org/)
