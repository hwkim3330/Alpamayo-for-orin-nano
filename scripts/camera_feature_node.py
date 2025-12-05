#!/usr/bin/env python3
"""
Camera Feature Node with Change Detection
Processes camera images efficiently using change detection gating

Only runs CNN when:
1. Frame has changed significantly (mean pixel difference > threshold)
2. Minimum time interval has passed since last CNN run

Topics:
    Subscribe:
        /camera/image_raw (Image) - Raw camera input

    Publish:
        /camera/feature (Float32MultiArray) - CNN feature vector
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torch.nn as nn
import time
import os
import sys

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

try:
    from camera_cnn import CameraCNN
except ImportError:
    # Fallback inline definition
    class CameraCNN(nn.Module):
        def __init__(self, out_dim=64):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, out_dim)

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x).view(x.size(0), -1)
            return self.fc(x)


class CameraChangeDetector:
    """
    Efficient change detection for gating CNN inference.
    Uses low-resolution grayscale comparison.
    """
    def __init__(self, small_size=(32, 18), diff_thresh=10.0, min_interval=0.2):
        self.small_w, self.small_h = small_size
        self.diff_thresh = diff_thresh
        self.min_interval = min_interval
        self.prev_small = None
        self.last_cnn_time = 0.0
        self.cached_feature = None

    def preprocess_and_check(self, frame_bgr):
        """
        Check if CNN should run based on frame change and timing.

        Args:
            frame_bgr: (H, W, 3) BGR camera frame

        Returns:
            need_cnn: bool - whether to run CNN
            cnn_input: preprocessed input for CNN (resized RGB)
        """
        H, W = frame_bgr.shape[:2]

        # ROI: lower 60% of frame (road region)
        y0 = int(H * 0.4)
        roi = frame_bgr[y0:, :, :]

        # Create small grayscale for change detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self.small_w, self.small_h))

        # Calculate change
        mean_diff = 0.0
        if self.prev_small is not None:
            diff = cv2.absdiff(small, self.prev_small)
            mean_diff = float(np.mean(diff))
        self.prev_small = small

        # Check conditions
        now = time.time()
        time_ok = (now - self.last_cnn_time) >= self.min_interval
        change_ok = (mean_diff > self.diff_thresh)

        need_cnn = (self.cached_feature is None) or (time_ok and change_ok)

        # Prepare CNN input (160x90 RGB)
        cnn_input = cv2.resize(roi, (160, 90))

        return need_cnn, cnn_input, mean_diff

    def update_cnn_feature(self, feature):
        """Store computed feature and update timestamp"""
        self.cached_feature = feature
        self.last_cnn_time = time.time()

    def get_feature(self):
        """Get cached feature"""
        return self.cached_feature


class CameraFeatureNode(Node):
    def __init__(self):
        super().__init__('camera_feature_node')

        # Parameters
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/camera/feature')
        self.declare_parameter('feature_dim', 64)
        self.declare_parameter('diff_thresh', 10.0)
        self.declare_parameter('min_interval', 0.2)
        self.declare_parameter('model_path', '')

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.feature_dim = self.get_parameter('feature_dim').value
        self.diff_thresh = self.get_parameter('diff_thresh').value
        self.min_interval = self.get_parameter('min_interval').value
        self.model_path = self.get_parameter('model_path').value

        # OpenCV bridge
        self.bridge = CvBridge()

        # Change detector
        self.detector = CameraChangeDetector(
            small_size=(32, 18),
            diff_thresh=self.diff_thresh,
            min_interval=self.min_interval
        )

        # Device and model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CameraCNN(out_dim=self.feature_dim).to(self.device)
        self.model.eval()

        # Load pretrained weights if available
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.get_logger().info(f'Loaded model from {self.model_path}')
            except Exception as e:
                self.get_logger().warn(f'Could not load model: {e}')

        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber
        self.sub = self.create_subscription(
            Image, self.input_topic,
            self.image_callback,
            sensor_qos
        )

        # Publisher
        self.pub = self.create_publisher(Float32MultiArray, self.output_topic, 10)

        # Stats
        self.frame_count = 0
        self.cnn_count = 0
        self.last_log_time = time.time()

        self.get_logger().info(f'Camera Feature Node started')
        self.get_logger().info(f'  Input: {self.input_topic}')
        self.get_logger().info(f'  Output: {self.output_topic} (dim={self.feature_dim})')
        self.get_logger().info(f'  Device: {self.device}')

    def image_callback(self, msg: Image):
        """Process incoming camera image"""
        try:
            # Convert ROS Image to OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        self.frame_count += 1

        # Check if CNN should run
        need_cnn, cnn_input, diff = self.detector.preprocess_and_check(frame)

        feature = None

        if need_cnn:
            self.cnn_count += 1

            # Preprocess for CNN
            rgb = cv2.cvtColor(cnn_input, cv2.COLOR_BGR2RGB)
            img = rgb.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # (3, H, W)
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

            # Run CNN
            with torch.no_grad():
                feature = self.model(img_tensor)

            self.detector.update_cnn_feature(feature)
        else:
            feature = self.detector.get_feature()

        if feature is None:
            return

        # Publish feature
        feat_np = feature.squeeze(0).cpu().numpy()
        msg_out = Float32MultiArray()
        msg_out.data = feat_np.tolist()
        self.pub.publish(msg_out)

        # Log stats periodically
        now = time.time()
        if now - self.last_log_time > 5.0:
            fps = self.frame_count / (now - self.last_log_time)
            cnn_rate = self.cnn_count / (now - self.last_log_time)
            self.get_logger().info(
                f'Camera: {fps:.1f} FPS, CNN: {cnn_rate:.1f}/s ({100*self.cnn_count/max(1,self.frame_count):.0f}%)'
            )
            self.frame_count = 0
            self.cnn_count = 0
            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = CameraFeatureNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
