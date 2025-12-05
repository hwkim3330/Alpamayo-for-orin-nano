#!/usr/bin/env python3
"""
Planner Visualization Node
Publishes planned path and detections for web visualization
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, ColorRGBA
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
import json
import math
import time


class PlannerVizNode(Node):
    def __init__(self):
        super().__init__('planner_viz_node')

        # Parameters
        self.declare_parameter('path_topic', '/planner/path')
        self.declare_parameter('markers_topic', '/planner/markers')
        self.declare_parameter('detections_topic', '/planner/detections_json')
        self.declare_parameter('publish_rate', 10.0)

        self.path_topic = self.get_parameter('path_topic').value
        self.markers_topic = self.get_parameter('markers_topic').value
        self.detections_topic = self.get_parameter('detections_topic').value
        self.publish_rate = self.get_parameter('publish_rate').value

        # Detection storage
        self.detections = []

        # Publishers
        self.pub_path = self.create_publisher(Path, self.path_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.markers_topic, 10)
        self.pub_detections = self.create_publisher(String, self.detections_topic, 10)

        # Subscriber for planner output (if available)
        self.sub_planner = self.create_subscription(
            String, '/planner/output', self.planner_callback, 10
        )

        # Timer for demo visualization
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_demo)

        self.time_offset = 0.0

        self.get_logger().info('Planner Visualization Node started')

    def planner_callback(self, msg: String):
        """Process planner output"""
        try:
            data = json.loads(msg.data)
            if 'detections' in data:
                self.detections = data['detections']
            if 'path' in data:
                self.publish_path(data['path'])
        except json.JSONDecodeError:
            pass

    def publish_demo(self):
        """Publish demo visualization data"""
        self.time_offset += 0.1

        # Generate demo path (curved road ahead)
        path = self.generate_demo_path()
        self.pub_path.publish(path)

        # Generate demo detections
        markers = self.generate_demo_markers()
        self.pub_markers.publish(markers)

        # Publish detections as JSON for web
        detections_json = self.generate_demo_detections()
        det_msg = String()
        det_msg.data = json.dumps(detections_json)
        self.pub_detections.publish(det_msg)

    def generate_demo_path(self) -> Path:
        """Generate a demo path"""
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'base_link'

        # Create curved path
        curvature = 0.02 * math.sin(self.time_offset * 0.5)

        for i in range(50):
            t = i * 0.3  # Distance along path
            x = t
            y = curvature * t * t  # Parabolic curve

            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            # Calculate heading
            if i < 49:
                dx = 0.3
                dy = 2 * curvature * t * 0.3
                yaw = math.atan2(dy, dx)
            else:
                yaw = 0.0

            pose.pose.orientation.z = math.sin(yaw / 2)
            pose.pose.orientation.w = math.cos(yaw / 2)

            path.poses.append(pose)

        return path

    def generate_demo_markers(self) -> MarkerArray:
        """Generate demo detection markers"""
        markers = MarkerArray()

        # Define demo detections
        demo_objects = [
            {'id': 1, 'type': 'vehicle', 'x': 8.0, 'y': -1.5, 'w': 2.0, 'l': 4.5, 'h': 1.5},
            {'id': 2, 'type': 'vehicle', 'x': 15.0, 'y': 1.5, 'w': 2.0, 'l': 4.5, 'h': 1.5},
            {'id': 3, 'type': 'pedestrian', 'x': 6.0, 'y': 3.0, 'w': 0.5, 'l': 0.5, 'h': 1.7},
            {'id': 4, 'type': 'cyclist', 'x': 12.0, 'y': -3.5, 'w': 0.6, 'l': 1.8, 'h': 1.5},
        ]

        for obj in demo_objects:
            # Add some movement
            obj['x'] += math.sin(self.time_offset + obj['id']) * 0.5
            obj['y'] += math.cos(self.time_offset * 0.5 + obj['id']) * 0.2

            # Create marker
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = 'base_link'
            marker.ns = obj['type']
            marker.id = obj['id']
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = obj['x']
            marker.pose.position.y = obj['y']
            marker.pose.position.z = obj['h'] / 2

            marker.scale.x = obj['l']
            marker.scale.y = obj['w']
            marker.scale.z = obj['h']

            # Color based on type
            if obj['type'] == 'vehicle':
                marker.color = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.7)
            elif obj['type'] == 'pedestrian':
                marker.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.7)
            elif obj['type'] == 'cyclist':
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.5, a=0.7)
            else:
                marker.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.7)

            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 200000000  # 200ms

            markers.markers.append(marker)

            # Add wireframe (LINE_LIST)
            wire_marker = Marker()
            wire_marker.header = marker.header
            wire_marker.ns = f'{obj["type"]}_wire'
            wire_marker.id = obj['id'] + 100
            wire_marker.type = Marker.LINE_LIST
            wire_marker.action = Marker.ADD

            # Create box wireframe
            hx, hy, hz = obj['l']/2, obj['w']/2, obj['h']
            corners = [
                (-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0),  # Bottom
                (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)  # Top
            ]
            edges = [
                (0,1), (1,2), (2,3), (3,0),  # Bottom
                (4,5), (5,6), (6,7), (7,4),  # Top
                (0,4), (1,5), (2,6), (3,7)   # Vertical
            ]

            for e in edges:
                p1 = Point()
                p1.x = obj['x'] + corners[e[0]][0]
                p1.y = obj['y'] + corners[e[0]][1]
                p1.z = corners[e[0]][2]
                wire_marker.points.append(p1)

                p2 = Point()
                p2.x = obj['x'] + corners[e[1]][0]
                p2.y = obj['y'] + corners[e[1]][1]
                p2.z = corners[e[1]][2]
                wire_marker.points.append(p2)

            wire_marker.scale.x = 0.05  # Line width
            wire_marker.color = marker.color
            wire_marker.color.a = 1.0
            wire_marker.lifetime = marker.lifetime

            markers.markers.append(wire_marker)

        return markers

    def generate_demo_detections(self) -> list:
        """Generate detection list for web UI"""
        detections = [
            {
                'id': 1,
                'type': 'vehicle',
                'label': 'Sedan',
                'distance': round(8.0 + math.sin(self.time_offset) * 0.5, 1),
                'confidence': 0.96,
                'speed': 45.0
            },
            {
                'id': 2,
                'type': 'vehicle',
                'label': 'SUV',
                'distance': round(15.0 + math.sin(self.time_offset) * 0.5, 1),
                'confidence': 0.89,
                'speed': 52.0
            },
            {
                'id': 3,
                'type': 'pedestrian',
                'label': 'Pedestrian',
                'distance': round(6.0 + math.cos(self.time_offset) * 0.3, 1),
                'confidence': 0.94,
                'speed': 1.2
            },
            {
                'id': 4,
                'type': 'cyclist',
                'label': 'Cyclist',
                'distance': round(12.0 + math.sin(self.time_offset) * 0.4, 1),
                'confidence': 0.91,
                'speed': 18.0
            },
        ]
        return detections


def main(args=None):
    rclpy.init(args=args)
    node = PlannerVizNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
