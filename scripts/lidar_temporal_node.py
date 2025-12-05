#!/usr/bin/env python3
"""
LiDAR Temporal Encoder Node
Processes LiDAR BEV sequences with ConvGRU for temporal features

Topics:
    Subscribe:
        /lidar/bev_image (Image) - BEV image from lidar_bev_node
        or
        /bev/grid (OccupancyGrid) - Occupancy grid

    Publish:
        /lidar/feature (Float32MultiArray) - Temporal feature vector
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import time
import os
import sys

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

try:
    from lidar_temporal import LiDARTemporalEncoder
except ImportError:
    # Fallback inline definition
    class ConvGRUCell(nn.Module):
        def __init__(self, input_dim, hidden_dim, kernel_size=3):
            super().__init__()
            padding = kernel_size // 2
            self.hidden_dim = hidden_dim
            self.conv_gates = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim, kernel_size, padding=padding)
            self.conv_candidate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)

        def forward(self, x, h_prev=None):
            if h_prev is None:
                h_prev = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3), device=x.device)
            combined = torch.cat([x, h_prev], dim=1)
            gates = torch.sigmoid(self.conv_gates(combined))
            z, r = torch.chunk(gates, 2, dim=1)
            combined_r = torch.cat([x, r * h_prev], dim=1)
            candidate = torch.tanh(self.conv_candidate(combined_r))
            return (1 - z) * h_prev + z * candidate

    class LiDARTemporalEncoder(nn.Module):
        def __init__(self, in_channels=2, hidden_channels=32, feature_dim=32):
            super().__init__()
            self.pre_conv = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, hidden_channels, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            self.conv_gru = ConvGRUCell(hidden_channels, hidden_channels)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(hidden_channels, feature_dim)
            self.hidden_channels = hidden_channels

        def forward(self, bev, h_prev=None, return_hidden=True):
            if isinstance(bev, np.ndarray):
                bev = torch.from_numpy(bev)
            if bev.dim() == 3:
                bev = bev.unsqueeze(0)
            bev = bev.float().to(next(self.parameters()).device)

            x = self.pre_conv(bev)
            h_new = self.conv_gru(x, h_prev)
            pooled = self.pool(h_new).view(h_new.size(0), -1)
            feature = self.fc(pooled)

            if return_hidden:
                return feature, h_new
            return feature


class LidarTemporalNode(Node):
    def __init__(self):
        super().__init__('lidar_temporal_node')

        # Parameters
        self.declare_parameter('input_type', 'image')  # 'image' or 'grid'
        self.declare_parameter('input_topic', '/lidar/bev_image')
        self.declare_parameter('output_topic', '/lidar/feature')
        self.declare_parameter('feature_dim', 32)
        self.declare_parameter('hidden_channels', 32)
        self.declare_parameter('model_path', '')

        self.input_type = self.get_parameter('input_type').value
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.feature_dim = self.get_parameter('feature_dim').value
        self.hidden_channels = self.get_parameter('hidden_channels').value
        self.model_path = self.get_parameter('model_path').value

        # OpenCV bridge
        self.bridge = CvBridge()

        # Device and model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LiDARTemporalEncoder(
            in_channels=2,
            hidden_channels=self.hidden_channels,
            feature_dim=self.feature_dim
        ).to(self.device)
        self.model.eval()

        # Load pretrained weights if available
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.get_logger().info(f'Loaded model from {self.model_path}')
            except Exception as e:
                self.get_logger().warn(f'Could not load model: {e}')

        # Hidden state
        self.h_prev = None

        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber
        if self.input_type == 'image':
            self.sub = self.create_subscription(
                Image, self.input_topic,
                self.image_callback,
                sensor_qos
            )
        else:
            self.sub = self.create_subscription(
                OccupancyGrid, self.input_topic,
                self.grid_callback,
                sensor_qos
            )

        # Publisher
        self.pub = self.create_publisher(Float32MultiArray, self.output_topic, 10)

        # Stats
        self.frame_count = 0
        self.last_log_time = time.time()
        self.processing_times = []

        self.get_logger().info(f'LiDAR Temporal Node started')
        self.get_logger().info(f'  Input: {self.input_topic} ({self.input_type})')
        self.get_logger().info(f'  Output: {self.output_topic} (dim={self.feature_dim})')
        self.get_logger().info(f'  Device: {self.device}')

    def image_callback(self, msg: Image):
        """Process BEV image"""
        start_time = time.time()

        try:
            # Convert to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        # Convert BGR image to 2-channel BEV
        # Channel 0: Green channel (occupancy)
        # Channel 1: Blue channel (intensity)
        bev = np.zeros((2, cv_image.shape[0], cv_image.shape[1]), dtype=np.float32)
        bev[0] = cv_image[:, :, 1].astype(np.float32) / 255.0  # Green
        bev[1] = cv_image[:, :, 0].astype(np.float32) / 255.0  # Blue

        self.process_bev(bev)

        proc_time = time.time() - start_time
        self.processing_times.append(proc_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

    def grid_callback(self, msg: OccupancyGrid):
        """Process OccupancyGrid"""
        start_time = time.time()

        W = msg.info.width
        H = msg.info.height
        data = np.array(msg.data, dtype=np.int8).reshape(H, W)

        # Create 2-channel BEV
        bev = np.zeros((2, H, W), dtype=np.float32)
        bev[0] = (data > 50).astype(np.float32)  # Occupancy
        bev[1] = np.clip(data / 100.0, 0, 1)     # Normalized count

        self.process_bev(bev)

        proc_time = time.time() - start_time
        self.processing_times.append(proc_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

    def process_bev(self, bev):
        """Run temporal encoder on BEV"""
        self.frame_count += 1

        # Run model
        with torch.no_grad():
            feature, self.h_prev = self.model(bev, self.h_prev, return_hidden=True)

        # Publish feature
        feat_np = feature.squeeze(0).cpu().numpy()
        msg_out = Float32MultiArray()
        msg_out.data = feat_np.tolist()
        self.pub.publish(msg_out)

        # Log stats periodically
        now = time.time()
        if now - self.last_log_time > 5.0:
            fps = self.frame_count / (now - self.last_log_time)
            avg_time = np.mean(self.processing_times) * 1000 if self.processing_times else 0
            self.get_logger().info(f'LiDAR Temporal: {fps:.1f} FPS, proc: {avg_time:.1f}ms')
            self.frame_count = 0
            self.last_log_time = now

    def reset_state(self):
        """Reset temporal state (call at episode start)"""
        self.h_prev = None
        self.get_logger().info('Temporal state reset')


def main(args=None):
    rclpy.init(args=args)
    node = LidarTemporalNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
