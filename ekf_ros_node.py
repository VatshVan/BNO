import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
import numpy as np
import math
import pyproj

from .ekf_core import DynamicExtendedKalmanFilter

class EKFFusionNode(Node):
    def __init__(self):
        super().__init__('ekf_fusion_node')

        self.declare_parameter('fallback_lat', 42.293260)
        self.declare_parameter('fallback_lon', -83.709691)
        self.declare_parameter('fallback_alt', 265.80)
        self.declare_parameter('p_matrix_diag', [1.0, 1.0, 1.0, 1.0, 1.0])
        self.declare_parameter('q_matrix_diag', [0.1, 0.1, 0.05, 0.1, 0.05])
        self.declare_parameter('r_matrix_diag', [0.5, 0.5, 0.2, 0.1])
        self.declare_parameter('loop_rate_hz', 100.0)

        p_diag = self.get_parameter('p_matrix_diag').value
        q_diag = self.get_parameter('q_matrix_diag').value
        r_diag = self.get_parameter('r_matrix_diag').value
        
        initial_state = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.ekf = DynamicExtendedKalmanFilter(
            initial_state=initial_state,
            p_matrix=np.diag(p_diag).tolist(),
            q_matrix=np.diag(q_diag).tolist(),
            r_matrix=np.diag(r_diag).tolist()
        )

        self.datum_set = False
        self.datum_lat = 0.0
        self.datum_lon = 0.0
        self.datum_alt = 0.0
        self.transformer = None

        self.latest_v = 0.0
        self.latest_omega = 0.0
        self.latest_x_gnss = 0.0
        self.latest_y_gnss = 0.0
        self.gnss_received = False

        self.gps_sub = self.create_subscription(NavSatFix, '/gps/fix', self.gps_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/wheel/odom', self.odom_callback, 10)

        self.odom_pub = self.create_publisher(Odometry, '/ekf/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.publish_state()  # Publish initial state immediately

        loop_rate = self.get_parameter('loop_rate_hz').value
        self.dt = 1.0 / loop_rate
        self.timer = self.create_timer(self.dt, self.ekf_loop)
        self.last_time = self.get_clock().now()

        self.initialize_datum_timeout()

    def initialize_datum_timeout(self):
        rclpy.spin_once(self, timeout_sec=2.0)
        if not self.datum_set:
            self.get_logger().warn("GNSS timeout. Initializing datum from parameter server.")
            self.datum_lat = self.get_parameter('fallback_lat').value
            self.datum_lon = self.get_parameter('fallback_lon').value
            self.datum_alt = self.get_parameter('fallback_alt').value
            self.setup_coordinate_transformer()

    def setup_coordinate_transformer(self):
        self.transformer = pyproj.Transformer.from_pipeline(
            f"+proj=pipeline +step +proj=cart +ellps=WGS84 "
            f"+step +proj=topocentric +lat_0={self.datum_lat} +lon_0={self.datum_lon} +h_0={self.datum_alt}"
        )
        self.datum_set = True
        self.get_logger().info("Geodetic ENU datum established.")

    def gps_callback(self, msg):
        if not self.datum_set:
            self.datum_lat = msg.latitude
            self.datum_lon = msg.longitude
            self.datum_alt = msg.altitude
            self.setup_coordinate_transformer()

        if self.transformer:
            enu_e, enu_n, enu_u = self.transformer.transform(msg.longitude, msg.latitude, msg.altitude)
            self.latest_x_gnss = enu_e
            self.latest_y_gnss = enu_n
            self.gnss_received = True

    def imu_callback(self, msg):
        self.latest_omega = msg.angular_velocity.z

    def odom_callback(self, msg):
        self.latest_v = msg.twist.twist.linear.x

    def ekf_loop(self):
        if not self.datum_set:
            return

        current_time = self.get_clock().now()
        dt_actual = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        self.ekf.predict(dt_actual)

        if self.gnss_received:
            z = [self.latest_x_gnss, self.latest_y_gnss, self.latest_v, self.latest_omega]
            self.ekf.update(z)

        self.publish_state()

    def get_quaternion_from_yaw(self, yaw):
        q = Quaternion()
        q.w = math.cos(yaw / 2.0)
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        return q

    def publish_state(self):
        current_time = self.get_clock().now().to_msg()
        
        x_est = float(self.ekf.x[0, 0])
        y_est = float(self.ekf.x[1, 0])
        theta_est = float(self.ekf.x[2, 0])
        v_est = float(self.ekf.x[3, 0])
        omega_est = float(self.ekf.x[4, 0])

        odom_msg = Odometry()
        odom_msg.header.stamp = current_time
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        odom_msg.pose.pose.position.x = x_est
        odom_msg.pose.pose.position.y = y_est
        odom_msg.pose.pose.orientation = self.get_quaternion_from_yaw(theta_est)

        odom_msg.twist.twist.linear.x = v_est
        odom_msg.twist.twist.angular.z = omega_est

        self.odom_pub.publish(odom_msg)

        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = x_est
        t.transform.translation.y = y_est
        t.transform.translation.z = 0.0
        t.transform.rotation = self.get_quaternion_from_yaw(theta_est)

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = EKFFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()