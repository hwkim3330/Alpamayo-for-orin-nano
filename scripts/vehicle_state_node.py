#!/usr/bin/env python3
"""
Vehicle State Publisher Node
Publishes vehicle state for web visualization (speed, steering, battery, etc.)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
import json
import time
import math


class VehicleStateNode(Node):
    def __init__(self):
        super().__init__('vehicle_state_node')

        # Parameters
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('state_topic', '/vehicle/state_json')

        self.publish_rate = self.get_parameter('publish_rate').value
        self.state_topic = self.get_parameter('state_topic').value

        # Vehicle state
        self.state = {
            'speed': 0.0,           # m/s
            'speed_kmh': 0.0,       # km/h
            'steering_angle': 0.0,  # degrees
            'battery_percent': 100.0,
            'battery_voltage': 12.6,
            'position_x': 0.0,
            'position_y': 0.0,
            'heading': 0.0,         # degrees
            'linear_vel': 0.0,
            'angular_vel': 0.0,
            'gear': 'D',            # P, R, N, D
            'mode': 'FSD',          # Manual, FSD, Autopilot
            'timestamp': 0
        }

        # Subscribers
        self.sub_cmd_vel = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.sub_battery = self.create_subscription(
            BatteryState, '/battery', self.battery_callback, 10
        )

        # Publishers
        self.pub_state = self.create_publisher(String, self.state_topic, 10)
        self.pub_speed = self.create_publisher(Float32, '/vehicle/speed', 10)
        self.pub_steering = self.create_publisher(Float32, '/vehicle/steering', 10)

        # Timer
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_state)

        self.get_logger().info(f'Vehicle State Node started')
        self.get_logger().info(f'  Publishing to: {self.state_topic} @ {self.publish_rate}Hz')

    def cmd_vel_callback(self, msg: Twist):
        """Process velocity command"""
        self.state['linear_vel'] = msg.linear.x
        self.state['angular_vel'] = msg.angular.z

        # Estimate speed and steering from cmd_vel
        self.state['speed'] = abs(msg.linear.x)
        self.state['speed_kmh'] = self.state['speed'] * 3.6

        # Convert angular velocity to steering angle (simplified)
        # steering_angle = atan(L * omega / v) where L is wheelbase
        wheelbase = 0.26  # meters (for RC car)
        if abs(msg.linear.x) > 0.01:
            steering_rad = math.atan(wheelbase * msg.angular.z / msg.linear.x)
            self.state['steering_angle'] = math.degrees(steering_rad)
        else:
            self.state['steering_angle'] = 0.0

        # Determine gear
        if msg.linear.x > 0.01:
            self.state['gear'] = 'D'
        elif msg.linear.x < -0.01:
            self.state['gear'] = 'R'
        else:
            self.state['gear'] = 'N'

    def odom_callback(self, msg: Odometry):
        """Process odometry"""
        self.state['position_x'] = msg.pose.pose.position.x
        self.state['position_y'] = msg.pose.pose.position.y

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.state['heading'] = math.degrees(yaw)

        # Update speed from odometry twist
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.state['speed'] = math.sqrt(vx*vx + vy*vy)
        self.state['speed_kmh'] = self.state['speed'] * 3.6

    def battery_callback(self, msg: BatteryState):
        """Process battery state"""
        self.state['battery_percent'] = msg.percentage * 100
        self.state['battery_voltage'] = msg.voltage

    def publish_state(self):
        """Publish complete vehicle state as JSON"""
        self.state['timestamp'] = int(time.time() * 1000)

        # Publish JSON state
        state_msg = String()
        state_msg.data = json.dumps(self.state)
        self.pub_state.publish(state_msg)

        # Publish individual topics for compatibility
        speed_msg = Float32()
        speed_msg.data = float(self.state['speed_kmh'])
        self.pub_speed.publish(speed_msg)

        steering_msg = Float32()
        steering_msg.data = float(self.state['steering_angle'])
        self.pub_steering.publish(steering_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VehicleStateNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
