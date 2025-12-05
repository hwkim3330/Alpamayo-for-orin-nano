#!/usr/bin/env python3
"""
LiDAR BEV (Bird's Eye View) Visualization Node
Converts LiDAR point cloud to 2D BEV image for web visualization
Optimized for Jetson Orin Nano
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, Image, LaserScan
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import struct
import time


class LidarBEVNode(Node):
    def __init__(self):
        super().__init__('lidar_bev_node')

        # Parameters
        self.declare_parameter('x_range', [-5.0, 15.0])  # meters (behind, front)
        self.declare_parameter('y_range', [-10.0, 10.0])  # meters (left, right)
        self.declare_parameter('resolution', 0.1)  # meters per pixel
        self.declare_parameter('decay_factor', 0.9)  # temporal decay
        self.declare_parameter('input_topic', '/scan')  # or /lidar/points
        self.declare_parameter('output_topic', '/lidar/bev_image')
        self.declare_parameter('use_pointcloud2', False)  # True for 3D LiDAR

        # Get parameters
        self.x_range = self.get_parameter('x_range').value
        self.y_range = self.get_parameter('y_range').value
        self.resolution = self.get_parameter('resolution').value
        self.decay_factor = self.get_parameter('decay_factor').value
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.use_pointcloud2 = self.get_parameter('use_pointcloud2').value

        # Calculate grid size
        self.grid_width = int((self.x_range[1] - self.x_range[0]) / self.resolution)
        self.grid_height = int((self.y_range[1] - self.y_range[0]) / self.resolution)

        # BEV grid (3 channels: occupancy, intensity, height)
        self.bev_grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.float32)

        # CV Bridge
        self.bridge = CvBridge()

        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        if self.use_pointcloud2:
            self.sub = self.create_subscription(
                PointCloud2,
                self.input_topic,
                self.pointcloud2_callback,
                sensor_qos
            )
        else:
            self.sub = self.create_subscription(
                LaserScan,
                self.input_topic,
                self.laserscan_callback,
                sensor_qos
            )

        # Publisher
        self.pub = self.create_publisher(Image, self.output_topic, 10)

        # Timer for publishing BEV at fixed rate
        self.timer = self.create_timer(0.05, self.publish_bev)  # 20Hz

        # Stats
        self.frame_count = 0
        self.last_log_time = time.time()

        self.get_logger().info(f'LiDAR BEV Node started')
        self.get_logger().info(f'  Grid size: {self.grid_width}x{self.grid_height}')
        self.get_logger().info(f'  X range: {self.x_range}, Y range: {self.y_range}')
        self.get_logger().info(f'  Input: {self.input_topic} -> Output: {self.output_topic}')

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        ix = int((x - self.x_range[0]) / self.resolution)
        iy = int((y - self.y_range[0]) / self.resolution)

        if 0 <= ix < self.grid_width and 0 <= iy < self.grid_height:
            return iy, ix
        return None, None

    def laserscan_callback(self, msg: LaserScan):
        """Process 2D LaserScan data"""
        # Apply decay
        self.bev_grid *= self.decay_factor

        # Convert polar to cartesian
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        # Filter valid ranges
        valid = (ranges > msg.range_min) & (ranges < msg.range_max) & np.isfinite(ranges)
        angles = angles[:len(ranges)][valid]
        ranges = ranges[valid]

        # Convert to XY
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        # Update grid
        for xi, yi in zip(x, y):
            iy, ix = self.world_to_grid(xi, yi)
            if iy is not None:
                self.bev_grid[iy, ix, 0] = 1.0  # Occupancy
                self.bev_grid[iy, ix, 1] = min(1.0, np.sqrt(xi**2 + yi**2) / 15.0)  # Distance

        self.frame_count += 1

    def pointcloud2_callback(self, msg: PointCloud2):
        """Process 3D PointCloud2 data"""
        # Apply decay
        self.bev_grid *= self.decay_factor

        # Parse PointCloud2
        points = self.parse_pointcloud2(msg)

        if points is None or len(points) == 0:
            return

        # Update grid
        for point in points:
            x, y, z = point[:3]
            intensity = point[3] if len(point) > 3 else 0.5

            iy, ix = self.world_to_grid(x, y)
            if iy is not None:
                self.bev_grid[iy, ix, 0] = 1.0  # Occupancy
                self.bev_grid[iy, ix, 1] = min(1.0, intensity)  # Intensity
                self.bev_grid[iy, ix, 2] = min(1.0, (z + 1.0) / 3.0)  # Height

        self.frame_count += 1

    def parse_pointcloud2(self, msg: PointCloud2):
        """Parse PointCloud2 message to numpy array"""
        # Get field offsets
        x_offset = y_offset = z_offset = i_offset = None
        for field in msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset
            elif field.name == 'intensity':
                i_offset = field.offset

        if x_offset is None:
            return None

        # Parse points
        points = []
        data = msg.data
        point_step = msg.point_step

        for i in range(0, len(data), point_step):
            x = struct.unpack('f', data[i+x_offset:i+x_offset+4])[0]
            y = struct.unpack('f', data[i+y_offset:i+y_offset+4])[0]
            z = struct.unpack('f', data[i+z_offset:i+z_offset+4])[0] if z_offset else 0.0
            intensity = struct.unpack('f', data[i+i_offset:i+i_offset+4])[0] if i_offset else 0.5

            if np.isfinite(x) and np.isfinite(y):
                points.append([x, y, z, intensity])

        return np.array(points)

    def publish_bev(self):
        """Publish BEV image"""
        # Convert to uint8 image (BGR for visualization)
        bev_img = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)

        # Channel mapping for visualization
        # Red: Height, Green: Occupancy, Blue: Distance/Intensity
        bev_img[:, :, 2] = (self.bev_grid[:, :, 2] * 255).astype(np.uint8)  # R - height
        bev_img[:, :, 1] = (self.bev_grid[:, :, 0] * 255).astype(np.uint8)  # G - occupancy
        bev_img[:, :, 0] = (self.bev_grid[:, :, 1] * 255).astype(np.uint8)  # B - intensity

        # Draw ego vehicle position (center bottom)
        ego_y = int((0 - self.y_range[0]) / self.resolution)
        ego_x = int((0 - self.x_range[0]) / self.resolution)
        if 0 <= ego_x < self.grid_width and 0 <= ego_y < self.grid_height:
            # Draw small rectangle for ego
            cv_size = 3
            y1, y2 = max(0, ego_y - cv_size), min(self.grid_height, ego_y + cv_size)
            x1, x2 = max(0, ego_x - cv_size*2), min(self.grid_width, ego_x + cv_size*2)
            bev_img[y1:y2, x1:x2] = [0, 255, 255]  # Yellow ego vehicle

        # Flip for correct orientation (front = top)
        bev_img = np.flipud(bev_img)

        # Convert to ROS Image message
        img_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding='bgr8')
        img_msg.header = Header()
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'base_link'

        self.pub.publish(img_msg)

        # Log stats periodically
        now = time.time()
        if now - self.last_log_time > 5.0:
            fps = self.frame_count / (now - self.last_log_time)
            self.get_logger().info(f'LiDAR BEV: {fps:.1f} FPS')
            self.frame_count = 0
            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = LidarBEVNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
