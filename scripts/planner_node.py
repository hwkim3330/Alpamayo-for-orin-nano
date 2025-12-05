#!/usr/bin/env python3
"""
Planner Node with Online Learning
Fuses LiDAR and Camera features to predict steering and speed
Supports online learning from teacher signals

Topics:
    Subscribe:
        /lidar/feature (Float32MultiArray) - LiDAR temporal features
        /camera/feature (Float32MultiArray) - Camera CNN features
        /training/teacher (Vector3) - Teacher steering/speed signals
        /training/feedback (String) - Human feedback ("good"/"bad")

    Publish:
        /planner/cmd (Vector3) - Predicted [steering, speed, 0]
        /planner/detections_json (String) - Detection list as JSON
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Vector3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import os
from collections import deque
from threading import Lock


class PlannerHead(nn.Module):
    """Simple MLP planner"""
    def __init__(self, lidar_dim, cam_dim, hidden_dim=64, out_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(lidar_dim + cam_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, lidar_feat, cam_feat):
        x = torch.cat([lidar_feat, cam_feat], dim=1)
        return self.fc(x)


class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')

        # Parameters
        self.declare_parameter('lidar_dim', 32)
        self.declare_parameter('cam_dim', 64)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('learning_rate', 1e-3)
        self.declare_parameter('buffer_size', 1000)
        self.declare_parameter('train_interval', 1.0)
        self.declare_parameter('batch_size', 32)
        self.declare_parameter('save_path', '/tmp/planner_checkpoint.pth')

        self.lidar_dim = self.get_parameter('lidar_dim').value
        self.cam_dim = self.get_parameter('cam_dim').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.lr = self.get_parameter('learning_rate').value
        self.buffer_size = self.get_parameter('buffer_size').value
        self.train_interval = self.get_parameter('train_interval').value
        self.batch_size = self.get_parameter('batch_size').value
        self.save_path = self.get_parameter('save_path').value

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        # Model (initialized when we receive first features)
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.MSELoss()

        # Feature cache
        self.lidar_feat = None
        self.cam_feat = None
        self.feat_lock = Lock()

        # Replay buffer for online learning
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.last_train_time = time.time()

        # Feedback tracking
        self.feedback_history = deque(maxlen=100)

        # Subscribers
        self.sub_lidar = self.create_subscription(
            Float32MultiArray, '/lidar/feature',
            self.lidar_callback, 10
        )
        self.sub_cam = self.create_subscription(
            Float32MultiArray, '/camera/feature',
            self.cam_callback, 10
        )
        self.sub_teacher = self.create_subscription(
            Vector3, '/training/teacher',
            self.teacher_callback, 10
        )
        self.sub_feedback = self.create_subscription(
            String, '/training/feedback',
            self.feedback_callback, 10
        )

        # Publishers
        self.pub_cmd = self.create_publisher(Vector3, '/planner/cmd', 10)
        self.pub_detections = self.create_publisher(String, '/planner/detections_json', 10)

        # Timer for main loop
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        # Demo detections
        self.demo_time = 0

        self.get_logger().info('Planner Node started')
        self.get_logger().info(f'  LiDAR dim: {self.lidar_dim}, Cam dim: {self.cam_dim}')
        self.get_logger().info(f'  Publish rate: {self.publish_rate} Hz')

    def lidar_callback(self, msg: Float32MultiArray):
        with self.feat_lock:
            self.lidar_feat = np.array(msg.data, dtype=np.float32)

    def cam_callback(self, msg: Float32MultiArray):
        with self.feat_lock:
            self.cam_feat = np.array(msg.data, dtype=np.float32)

    def teacher_callback(self, msg: Vector3):
        """Store teacher signal for online learning"""
        with self.feat_lock:
            if self.lidar_feat is None or self.cam_feat is None:
                return

            teacher_action = np.array([msg.x, msg.y], dtype=np.float32)
            self.replay_buffer.append({
                'lidar': self.lidar_feat.copy(),
                'cam': self.cam_feat.copy(),
                'target': teacher_action
            })

        self.get_logger().debug(f'Teacher sample added, buffer size: {len(self.replay_buffer)}')

    def feedback_callback(self, msg: String):
        """Process human feedback"""
        feedback = msg.data.lower()
        self.feedback_history.append({
            'type': feedback,
            'timestamp': time.time()
        })

        # Could adjust learning rate or sample weights based on feedback
        if feedback == 'good':
            self.get_logger().info('Received positive feedback')
        elif feedback == 'bad':
            self.get_logger().warn('Received negative feedback')

    def init_model_if_needed(self):
        """Initialize model when we know feature dimensions"""
        if self.model is not None:
            return True

        with self.feat_lock:
            if self.lidar_feat is None or self.cam_feat is None:
                return False

            lidar_dim = len(self.lidar_feat)
            cam_dim = len(self.cam_feat)

        self.model = PlannerHead(lidar_dim, cam_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Try to load checkpoint
        if os.path.exists(self.save_path):
            try:
                checkpoint = torch.load(self.save_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.get_logger().info(f'Loaded checkpoint from {self.save_path}')
            except Exception as e:
                self.get_logger().warn(f'Could not load checkpoint: {e}')

        self.model.eval()
        self.get_logger().info(f'Model initialized: lidar_dim={lidar_dim}, cam_dim={cam_dim}')
        return True

    def timer_callback(self):
        """Main control loop"""
        # Initialize model if needed
        if not self.init_model_if_needed():
            return

        # Get current features
        with self.feat_lock:
            if self.lidar_feat is None or self.cam_feat is None:
                return

            lidar_t = torch.from_numpy(self.lidar_feat).unsqueeze(0).to(self.device)
            cam_t = torch.from_numpy(self.cam_feat).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(lidar_t, cam_t)

        steering = float(output[0, 0].cpu().item())
        speed = float(output[0, 1].cpu().item())

        # Publish command
        cmd_msg = Vector3()
        cmd_msg.x = steering  # radians
        cmd_msg.y = speed     # m/s
        cmd_msg.z = 0.0
        self.pub_cmd.publish(cmd_msg)

        # Publish demo detections
        self.publish_demo_detections()

        # Online training
        now = time.time()
        if now - self.last_train_time > self.train_interval:
            if len(self.replay_buffer) >= self.batch_size:
                self.train_step()
                self.last_train_time = now

    def publish_demo_detections(self):
        """Publish simulated detections for dashboard"""
        self.demo_time += 0.05

        detections = [
            {
                'id': 1,
                'type': 'vehicle',
                'label': 'Car',
                'distance': round(8 + np.sin(self.demo_time) * 2, 1),
                'confidence': 0.95
            },
            {
                'id': 2,
                'type': 'pedestrian',
                'label': 'Person',
                'distance': round(5 + np.cos(self.demo_time) * 1, 1),
                'confidence': 0.88
            },
            {
                'id': 3,
                'type': 'cyclist',
                'label': 'Bike',
                'distance': round(12 + np.sin(self.demo_time * 0.5) * 3, 1),
                'confidence': 0.91
            }
        ]

        msg = String()
        msg.data = json.dumps(detections)
        self.pub_detections.publish(msg)

    def train_step(self):
        """Perform one online training step"""
        if self.model is None:
            return

        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]

        lidar_batch = torch.tensor(
            np.stack([s['lidar'] for s in batch]),
            dtype=torch.float32, device=self.device
        )
        cam_batch = torch.tensor(
            np.stack([s['cam'] for s in batch]),
            dtype=torch.float32, device=self.device
        )
        target_batch = torch.tensor(
            np.stack([s['target'] for s in batch]),
            dtype=torch.float32, device=self.device
        )

        # Training
        self.model.train()
        self.optimizer.zero_grad()

        pred = self.model(lidar_batch, cam_batch)
        loss = self.loss_fn(pred, target_batch)

        loss.backward()
        self.optimizer.step()

        self.model.eval()

        self.get_logger().info(f'Training step: loss={loss.item():.4f}, buffer={len(self.replay_buffer)}')

        # Save checkpoint periodically
        if len(self.replay_buffer) % 100 == 0:
            self.save_checkpoint()

    def save_checkpoint(self):
        """Save model checkpoint"""
        if self.model is None:
            return

        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.save_path)
            self.get_logger().debug(f'Saved checkpoint to {self.save_path}')
        except Exception as e:
            self.get_logger().error(f'Could not save checkpoint: {e}')

    def destroy_node(self):
        """Save before shutdown"""
        self.save_checkpoint()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
