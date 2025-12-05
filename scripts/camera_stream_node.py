#!/usr/bin/env python3
"""
Camera Stream Node for Web Visualization
Processes camera frames and publishes compressed images
Optimized for Jetson Orin Nano with hardware encoding
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import time


class CameraStreamNode(Node):
    def __init__(self):
        super().__init__('camera_stream_node')

        # Parameters
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/camera/image_web')
        self.declare_parameter('compressed_topic', '/camera/image_compressed')
        self.declare_parameter('target_width', 640)
        self.declare_parameter('target_height', 360)
        self.declare_parameter('jpeg_quality', 70)
        self.declare_parameter('target_fps', 15.0)
        self.declare_parameter('draw_info', True)

        # Get parameters
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.compressed_topic = self.get_parameter('compressed_topic').value
        self.target_width = self.get_parameter('target_width').value
        self.target_height = self.get_parameter('target_height').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        self.target_fps = self.get_parameter('target_fps').value
        self.draw_info = self.get_parameter('draw_info').value

        # Frame rate control
        self.min_interval = 1.0 / self.target_fps
        self.last_publish_time = 0.0

        # CV Bridge
        self.bridge = CvBridge()

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber
        self.sub = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            sensor_qos
        )

        # Publishers
        self.pub_image = self.create_publisher(Image, self.output_topic, 10)
        self.pub_compressed = self.create_publisher(CompressedImage, self.compressed_topic, 10)

        # Stats
        self.frame_count = 0
        self.last_log_time = time.time()
        self.processing_times = []

        self.get_logger().info(f'Camera Stream Node started')
        self.get_logger().info(f'  Input: {self.input_topic}')
        self.get_logger().info(f'  Output: {self.output_topic}, {self.compressed_topic}')
        self.get_logger().info(f'  Resolution: {self.target_width}x{self.target_height} @ {self.target_fps}fps')

    def image_callback(self, msg: Image):
        """Process incoming camera image"""
        now = time.time()

        # Frame rate limiting
        if now - self.last_publish_time < self.min_interval:
            return

        start_time = time.time()

        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Resize for web
            h, w = cv_image.shape[:2]
            if w != self.target_width or h != self.target_height:
                cv_image = cv2.resize(cv_image, (self.target_width, self.target_height),
                                      interpolation=cv2.INTER_LINEAR)

            # Draw info overlay
            if self.draw_info:
                cv_image = self.draw_overlay(cv_image)

            # Publish resized image
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            img_msg.header = msg.header
            self.pub_image.publish(img_msg)

            # Publish compressed image for web streaming
            compressed_msg = CompressedImage()
            compressed_msg.header = msg.header
            compressed_msg.format = 'jpeg'
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            compressed_msg.data = cv2.imencode('.jpg', cv_image, encode_params)[1].tobytes()
            self.pub_compressed.publish(compressed_msg)

            self.last_publish_time = now
            self.frame_count += 1

            # Track processing time
            proc_time = time.time() - start_time
            self.processing_times.append(proc_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)

        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')

        # Log stats periodically
        if now - self.last_log_time > 5.0:
            fps = self.frame_count / (now - self.last_log_time)
            avg_proc_time = np.mean(self.processing_times) * 1000 if self.processing_times else 0
            self.get_logger().info(f'Camera: {fps:.1f} FPS, proc: {avg_proc_time:.1f}ms')
            self.frame_count = 0
            self.last_log_time = now

    def draw_overlay(self, image):
        """Draw info overlay on image"""
        h, w = image.shape[:2]

        # Semi-transparent overlay background
        overlay = image.copy()

        # Top bar
        cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)

        # Blend
        alpha = 0.7
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        timestamp = time.strftime('%H:%M:%S')
        cv2.putText(image, f'CAM | {self.target_width}x{self.target_height} | {timestamp}',
                    (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS indicator
        if self.processing_times:
            avg_proc = np.mean(self.processing_times) * 1000
            cv2.putText(image, f'{avg_proc:.0f}ms',
                        (w - 50, 20), font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        return image


def main(args=None):
    rclpy.init(args=args)
    node = CameraStreamNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
