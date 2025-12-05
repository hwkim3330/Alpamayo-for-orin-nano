# ROS2 Web Dashboard for Jetson Orin Nano

Real-time web-based visualization dashboard for ROS2 robots using [Robot Web Tools](https://robotwebtools.github.io/).

![Dashboard Preview](docs/dashboard-preview.png)

## Features

- **Real-time 3D Visualization** - Using ros3djs and Three.js
- **LiDAR BEV (Bird's Eye View)** - 2D occupancy grid from LiDAR data
- **Camera Streaming** - MJPEG video stream via web_video_server
- **Vehicle State Monitoring** - Speed, steering, battery, system stats
- **Object Detection Display** - Vehicles, pedestrians, cyclists
- **Path Planning Visualization** - Planned trajectory overlay
- **Responsive Design** - Works on desktop, tablet, mobile

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Browser                               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  roslibjs   │  │   ros3djs    │  │  MJPEG Stream │       │
│  │ (WebSocket) │  │   (Three.js) │  │    (HTTP)     │       │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘       │
└─────────┼────────────────┼─────────────────┼───────────────┘
          │                │                 │
          │ ws://9090      │                 │ http://8080
          ▼                ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                 Jetson Orin Nano                             │
│  ┌───────────────────┐  ┌────────────────────┐              │
│  │ rosbridge_server  │  │  web_video_server  │              │
│  │    (WebSocket)    │  │     (MJPEG)        │              │
│  └─────────┬─────────┘  └──────────┬─────────┘              │
│            │                       │                         │
│  ┌─────────▼─────────────────────────────────────────┐      │
│  │                   ROS2 Topics                      │      │
│  │  /scan  /camera/image_raw  /cmd_vel  /odom  ...   │      │
│  └─────────┬─────────────────────────────────────────┘      │
│            │                                                 │
│  ┌─────────▼─────────────────────────────────────────┐      │
│  │              Dashboard Nodes                       │      │
│  │  lidar_bev_node  camera_stream_node  planner_viz  │      │
│  └───────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

### Jetson Orin Nano
- JetPack 5.x or 6.x
- ROS2 Humble or Jazzy
- Python 3.8+

### ROS2 Packages
```bash
sudo apt install ros-humble-rosbridge-server
sudo apt install ros-humble-web-video-server
sudo apt install ros-humble-tf2-ros
sudo apt install ros-humble-cv-bridge
```

### Python Dependencies
```bash
pip3 install opencv-python numpy
```

## Installation

### 1. Clone the repository
```bash
cd ~/ros2_ws/src
git clone https://github.com/hwkim3330/ros2-web-dashboard.git
```

### 2. Build the package
```bash
cd ~/ros2_ws
colcon build --packages-select ros2_web_dashboard
source install/setup.bash
```

### 3. Install dependencies
```bash
rosdep install --from-paths src --ignore-src -r -y
```

## Usage

### Launch the full dashboard
```bash
ros2 launch ros2_web_dashboard dashboard.launch.py
```

### Launch with Jetson-optimized settings
```bash
ros2 launch ros2_web_dashboard jetson_orin.launch.py
```

### Launch with custom parameters
```bash
ros2 launch ros2_web_dashboard dashboard.launch.py \
    lidar_topic:=/lidar/scan \
    camera_topic:=/zed/rgb/image_raw \
    rosbridge_port:=9090
```

### Access the dashboard
Open your web browser and navigate to:
```
http://<jetson-ip>:8000
```

## Configuration

### Edit config file
```bash
nano ~/ros2_ws/src/ros2_web_dashboard/config/dashboard_params.yaml
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rosbridge_port` | 9090 | WebSocket port |
| `web_video_port` | 8080 | Video server port |
| `lidar_topic` | /scan | LiDAR input topic |
| `camera_topic` | /camera/image_raw | Camera input topic |
| `target_fps` | 15.0 | Camera stream FPS |
| `resolution` | 0.1 | BEV grid resolution (m/pixel) |

## Nodes

### lidar_bev_node.py
Converts LiDAR scan to BEV image
- Input: `/scan` (sensor_msgs/LaserScan) or `/lidar/points` (PointCloud2)
- Output: `/lidar/bev_image` (sensor_msgs/Image)

### camera_stream_node.py
Resizes and compresses camera images
- Input: `/camera/image_raw`
- Output: `/camera/image_web`, `/camera/image_compressed`

### vehicle_state_node.py
Aggregates vehicle state into JSON
- Input: `/cmd_vel`, `/odom`, `/battery`
- Output: `/vehicle/state_json`

### planner_viz_node.py
Publishes visualization markers
- Output: `/planner/path`, `/planner/markers`, `/planner/detections_json`

## Web Interface

### Views
1. **3D View** - ros3djs powered 3D visualization with TF, markers, path
2. **BEV** - Bird's Eye View LiDAR visualization
3. **Camera** - Live camera feed
4. **Split** - All views in a grid

### Features
- Real-time connection status
- Vehicle state monitoring (speed, steering, battery)
- Detection list with distance
- System stats (CPU, GPU, RAM)
- Console log viewer
- Settings modal for configuration

## Performance Optimization

### For Jetson Orin Nano (15W mode)
```yaml
camera_stream_node:
  target_width: 480
  target_height: 270
  jpeg_quality: 50
  target_fps: 10.0

lidar_bev_node:
  resolution: 0.15
  decay_factor: 0.8
```

### For Jetson Orin Nano (MAXN mode)
```yaml
camera_stream_node:
  target_width: 1280
  target_height: 720
  jpeg_quality: 80
  target_fps: 30.0
```

## Troubleshooting

### Connection issues
```bash
# Check rosbridge is running
ros2 topic list | grep rosbridge

# Test WebSocket
wscat -c ws://localhost:9090
```

### Video not showing
```bash
# Check web_video_server
curl http://localhost:8080/stream_viewer?topic=/camera/image_web

# List available streams
curl http://localhost:8080/
```

### High latency
- Reduce `target_fps`
- Lower `jpeg_quality`
- Reduce `target_width` and `target_height`

## License

MIT License

## Credits

- [Robot Web Tools](https://robotwebtools.github.io/)
- [roslibjs](https://github.com/RobotWebTools/roslibjs)
- [ros3djs](https://github.com/RobotWebTools/ros3djs)
- [Lucide Icons](https://lucide.dev/)
