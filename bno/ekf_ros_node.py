import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import numpy as np
import math
import os
import pyproj

from .ekf_core import DynamicExtendedKalmanFilter


class EKFFusionNode(Node):
    def __init__(self):
        super().__init__('ekf_fusion_node')

        # ────────────────────────────────────────
        #  Parameter declarations
        # ────────────────────────────────────────
        self.declare_parameter('fallback_lat', 42.293260)
        self.declare_parameter('fallback_lon', -83.709691)
        self.declare_parameter('fallback_alt', 265.80)
        self.declare_parameter('p_matrix_diag', [1.0, 1.0, 1.0, 1.0, 1.0, 0.01])
        self.declare_parameter('q_matrix_diag', [0.1, 0.1, 0.05, 0.1, 0.05, 0.001])
        self.declare_parameter('r_matrix_diag', [0.5, 0.5, 0.2, 0.1])
        self.declare_parameter('loop_rate_hz', 100.0)  # retained for param compat

        # heading injection
        self.declare_parameter('heading_r', 0.1)
        self.declare_parameter('cog_velocity_gate', 1.5)

        # ZARU/ZUPT
        self.declare_parameter('zaru_r_omega', 1e-4)
        self.declare_parameter('zaru_r_v', 1e-4)
        self.declare_parameter('stationary_velocity_threshold', 0.05)
        self.declare_parameter('stationary_duration_sec', 0.05)

        # lever arm compensation
        self.declare_parameter('gnss_lever_arm_x', 0.0)
        self.declare_parameter('gnss_lever_arm_y', 0.0)
        self.declare_parameter('use_tf_lever_arm', False)

        # warm boot persistence
        self.declare_parameter('state_cache_path', '/tmp/ekf_state_cache.json')
        self.declare_parameter('state_cache_rate_hz', 1.0)
        self.declare_parameter('warm_boot_p_inflate', 5.0)
        self.declare_parameter('warm_boot_max_age_sec', 3600.0)

        # divergence diagnostics
        self.declare_parameter('chi2_threshold', 15.0)
        self.declare_parameter('chi2_consec_limit', 10)

        # ────────────────────────────────────────
        #  Read parameters
        # ────────────────────────────────────────
        p_diag       = self.get_parameter('p_matrix_diag').value
        q_diag       = self.get_parameter('q_matrix_diag').value
        r_diag       = self.get_parameter('r_matrix_diag').value
        heading_r    = self.get_parameter('heading_r').value
        cog_gate     = self.get_parameter('cog_velocity_gate').value
        zaru_r_omega = self.get_parameter('zaru_r_omega').value
        zaru_r_v     = self.get_parameter('zaru_r_v').value
        chi2_thresh  = self.get_parameter('chi2_threshold').value
        chi2_limit   = self.get_parameter('chi2_consec_limit').value

        # ────────────────────────────────────────
        #  Instantiate EKF
        # ────────────────────────────────────────
        initial_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.ekf = DynamicExtendedKalmanFilter(
            initial_state=initial_state,
            p_matrix=np.diag(p_diag).tolist(),
            q_matrix=np.diag(q_diag).tolist(),
            r_matrix=np.diag(r_diag).tolist(),
            heading_r=heading_r,
            zaru_r_omega=zaru_r_omega,
            zaru_r_v=zaru_r_v,
            cog_velocity_gate=cog_gate,
            chi2_threshold=chi2_thresh,
            chi2_consec_limit=chi2_limit,
        )

        # ── warm boot: attempt state restoration ──
        self.cache_path   = self.get_parameter('state_cache_path').value
        p_inflate         = self.get_parameter('warm_boot_p_inflate').value
        max_age           = self.get_parameter('warm_boot_max_age_sec').value
        if self.ekf.deserialize_state(self.cache_path, p_inflate=p_inflate,
                                      max_age_sec=max_age):
            self.get_logger().info(
                f"Warm boot: state restored from cache (P inflated x{p_inflate}).")
        else:
            self.get_logger().info("Cold boot: no valid state cache found.")

        # ────────────────────────────────────────
        #  Coordinate transform
        # ────────────────────────────────────────
        self.datum_set = False
        self.datum_lat = 0.0
        self.datum_lon = 0.0
        self.datum_alt = 0.0
        self.transformer = None

        # ────────────────────────────────────────
        #  Latest sensor readings
        # ────────────────────────────────────────
        self.latest_v       = 0.0   # raw wheel encoder speed (3-D hypotenuse)
        self.latest_omega   = 0.0   # geographic yaw rate (projected)
        self.latest_x_gnss  = 0.0
        self.latest_y_gnss  = 0.0
        self.gnss_received  = False
        self.latest_hdop    = 1.0

        # ────────────────────────────────────────
        #  IMU-driven temporal integration state
        # ────────────────────────────────────────
        self.last_imu_stamp = None   # BNO055 hardware timestamp (seconds)
        self.latest_pitch   = 0.0    # pitch for velocity projection
        self.latest_roll    = 0.0    # roll  for gyro projection
        self.latest_v_2d    = 0.0    # pitch-projected 2-D velocity

        # ────────────────────────────────────────
        #  IMU rotational extrinsics cache
        # ────────────────────────────────────────
        self.imu_rotation_cache    = None    # 3x3 rotation matrix or None
        self.imu_rotation_resolved = False   # True after first tf2 lookup

        # ────────────────────────────────────────
        #  GNSS COG heading state
        # ────────────────────────────────────────
        self.prev_x_gnss      = None
        self.prev_y_gnss      = None
        self.cog_velocity_gate = cog_gate
        self.cog_heading_r     = 0.15

        # ────────────────────────────────────────
        #  ZARU/ZUPT stationary detection
        # ────────────────────────────────────────
        self.stationary_v_thresh  = self.get_parameter('stationary_velocity_threshold').value
        self.stationary_duration  = self.get_parameter('stationary_duration_sec').value
        self.stationary_start_time = None

        # ────────────────────────────────────────
        #  Lever Arm Compensation
        # ────────────────────────────────────────
        self.gnss_lever_x = self.get_parameter('gnss_lever_arm_x').value
        self.gnss_lever_y = self.get_parameter('gnss_lever_arm_y').value
        self.use_tf_lever = self.get_parameter('use_tf_lever_arm').value

        # tf2 passive listener for sensor extrinsics
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ────────────────────────────────────────
        #  Subscriptions
        # ────────────────────────────────────────
        self.gps_sub  = self.create_subscription(NavSatFix, '/gps/fix',    self.gps_callback,  100)
        self.imu_sub  = self.create_subscription(Imu,       '/imu/data',   self.imu_callback,  100)
        self.odom_sub = self.create_subscription(Odometry,  '/wheel/odom', self.odom_callback, 100)

        # ────────────────────────────────────────
        #  Publishers
        # ────────────────────────────────────────
        self.odom_pub   = self.create_publisher(Odometry, '/ekf/odom', 10)
        self.diag_pub   = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.publish_state()

        # ────────────────────────────────────────
        #  Architecture note
        # ────────────────────────────────────────
        #  The predict/update cycle is strictly IMU-message-driven
        #  (imu_callback), NOT timer-driven.  dt is derived from
        #  BNO055 hardware timestamps (msg.header.stamp), which
        #  eliminates the systematic temporal jitter introduced by
        #  OS scheduler latency on the host CPU.
        # ────────────────────────────────────────

        # ────────────────────────────────────────
        #  State persistence (serialization) timer
        # ────────────────────────────────────────
        cache_rate = self.get_parameter('state_cache_rate_hz').value
        if cache_rate > 0:
            self.cache_timer = self.create_timer(
                1.0 / cache_rate, self._serialize_state_callback)

    # ==================================================================
    #  Initialization helpers
    # ==================================================================
    def setup_coordinate_transformer(self):
        self.transformer = pyproj.Transformer.from_pipeline(
            f"+proj=pipeline +step +proj=cart +ellps=WGS84 "
            f"+step +proj=topocentric +lat_0={self.datum_lat} "
            f"+lon_0={self.datum_lon} +h_0={self.datum_alt}"
        )
        self.datum_set = True
        self.get_logger().info("Geodetic ENU datum established.")

    # ==================================================================
    #  Lever arm resolution (tf2 or static parameter)
    # ==================================================================
    def _resolve_lever_arm(self):
        """
        Resolve the GNSS antenna lever arm.  If use_tf_lever_arm is True,
        query the tf2 static tree for gps_link -> base_link.  Otherwise
        fall back to the parameter values.

        Returns:
            (lever_x, lever_y) in metres
        """
        if self.use_tf_lever:
            try:
                tf = self.tf_buffer.lookup_transform(
                    'base_link', 'gps_link', rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05))
                return (tf.transform.translation.x,
                        tf.transform.translation.y)
            except Exception:
                pass  # fall through to static params
        return (self.gnss_lever_x, self.gnss_lever_y)

    # ==================================================================
    #  IMU rotational extrinsics (mounting misalignment compensation)
    # ==================================================================
    def _resolve_imu_rotation(self):
        """
        Resolve and cache the IMU mounting rotation from tf2
        (base_link -> imu_link).  The lookup is performed once on the
        first call; subsequent calls return the cached 3x3 matrix.

        Even a 1-2 degree physical mounting skew induces a continuous
        lateral velocity bias under longitudinal acceleration.

        Returns:
            3x3 numpy rotation matrix, or None if no transform is available.
        """
        if self.imu_rotation_resolved:
            return self.imu_rotation_cache

        try:
            tf = self.tf_buffer.lookup_transform(
                'base_link', 'imu_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.0))
            q = tf.transform.rotation
            self.imu_rotation_cache = self._quat_to_rotation_matrix(
                q.x, q.y, q.z, q.w)
            self.get_logger().info(
                'IMU rotational extrinsics resolved from tf2.')
        except Exception:
            self.imu_rotation_cache = None

        self.imu_rotation_resolved = True
        return self.imu_rotation_cache

    @staticmethod
    def _quat_to_rotation_matrix(qx, qy, qz, qw):
        """Convert quaternion (x, y, z, w) to a 3x3 rotation matrix."""
        return np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ], dtype=np.float64)

    # ==================================================================
    #  Sensor callbacks
    # ==================================================================
    def gps_callback(self, msg):
        # ── deterministic GNSS outlier rejection ──
        # NavSatStatus.STATUS_NO_FIX = -1: receiver has lost satellite lock.
        # Do NOT rely on Mahalanobis / chi2 alone — a rapidly inflated P
        # can inadvertently accept stale coordinates from a degraded receiver.
        if msg.status.status < 0:
            self.get_logger().warn(
                'GNSS STATUS_NO_FIX: rejecting position update.',
                throttle_duration_sec=5.0)
            return

        if not self.datum_set:
            self.datum_lat = msg.latitude
            self.datum_lon = msg.longitude
            self.datum_alt = msg.altitude
            self.setup_coordinate_transformer()

        if not self.transformer:
            return

        enu_e, enu_n, _ = self.transformer.transform(
            msg.longitude, msg.latitude, msg.altitude)

        # ── lever arm compensation ──
        lever_x, lever_y = self._resolve_lever_arm()
        if lever_x != 0.0 or lever_y != 0.0:
            theta_est = self.ekf.get_heading()
            enu_e, enu_n = self.ekf.compensate_lever_arm(
                enu_e, enu_n, theta_est, lever_x, lever_y)

        # ── HDOP-based dynamic R scaling ──
        if msg.position_covariance_type != NavSatFix.COVARIANCE_TYPE_UNKNOWN:
            sigma = math.sqrt(max(msg.position_covariance[0], 1.0))
            nominal_sigma = math.sqrt(self.ekf.R_base[0, 0])
            if nominal_sigma > 0:
                hdop_est = sigma / nominal_sigma
            else:
                hdop_est = 1.0
            self.latest_hdop = max(hdop_est, 1.0)
            self.ekf.scale_gnss_covariance(self.latest_hdop)

        # ── GNSS Course Over Ground heading injection ──
        if self.prev_x_gnss is not None:
            dx = enu_e - self.prev_x_gnss
            dy = enu_n - self.prev_y_gnss
            dist = math.hypot(dx, dy)
            if abs(self.latest_v) > self.cog_velocity_gate and dist > 0.01:
                cog_heading = math.atan2(dy, dx)
                self.ekf.update_heading(cog_heading,
                                        r_heading=self.cog_heading_r)

        self.prev_x_gnss   = enu_e
        self.prev_y_gnss   = enu_n
        self.latest_x_gnss = enu_e
        self.latest_y_gnss = enu_n
        self.gnss_received = True

    def imu_callback(self, msg):
        """
        IMU-message-driven predict/update cycle.

        Architectural rationale
        -----------------------
        The state estimation pipeline is synchronised to the BNO055's
        physical sampling clock via hardware timestamps embedded in
        msg.header.stamp.  This eliminates the systematic temporal
        jitter introduced by OS-scheduled software timers (a requested
        10 ms loop may execute at 8-14 ms depending on CPU load).

        3-D attitude projection
        -----------------------
        The BNO055's on-board fusion core provides absolute orientation
        quaternions.  Pitch (theta) and roll (phi) are extracted and used
        to:
          1. Project wheel encoder velocity onto the 2-D horizontal plane:
                 v_2d = v_3d * cos(pitch)
          2. Isolate the geographic yaw rate from the coupled body-frame
             gyroscope vector via the Euler kinematic relation:
                 psi_dot = (q*sin(phi) + r*cos(phi)) / cos(theta)
        """
        # ── compute dt from BNO055 hardware timestamps ──
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_imu_stamp is None:
            self.last_imu_stamp = stamp_sec
            return                          # need two stamps to compute dt

        dt = stamp_sec - self.last_imu_stamp
        self.last_imu_stamp = stamp_sec

        if dt <= 0.0 or dt > 1.0:
            return                          # drop pathological gaps

        # ── extract BNO055 quaternion -> pitch & roll ──
        q = msg.orientation

        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll  = math.atan2(sinr_cosp, cosr_cosp)

        sinp  = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))

        self.latest_pitch = pitch
        self.latest_roll  = roll

        # ── rotational extrinsics: align IMU axes to base_link ──
        # Compensates physical mounting skew of the BNO055 relative to
        # the vehicle's longitudinal axis (base_link).
        gx = msg.angular_velocity.x
        gy = msg.angular_velocity.y
        gz = msg.angular_velocity.z

        R_imu = self._resolve_imu_rotation()
        if R_imu is not None:
            aligned = R_imu @ np.array([gx, gy, gz])
            gx = float(aligned[0])
            gy = float(aligned[1])
            gz = float(aligned[2])

        # ── 3-D attitude projection: isolate geographic yaw rate ──
        #    psi_dot = (q * sin(phi) + r * cos(phi)) / cos(theta)
        #    where (gx, gy, gz) = body-frame (p, q, r)
        #          phi = roll,  theta = pitch
        cos_pitch = math.cos(pitch)
        if abs(cos_pitch) > 1e-6:
            yaw_rate_geo = (gy * math.sin(roll) +
                            gz * math.cos(roll)) / cos_pitch
        else:
            yaw_rate_geo = gz           # near-gimbal-lock fallback

        self.latest_omega = yaw_rate_geo

        # ── project wheel velocity onto 2-D horizontal plane ──
        #    v_2d = v_3d * cos(pitch)
        self.latest_v_2d = self.latest_v * math.cos(pitch)

        # ── predict using hardware-derived dt ──
        if not self.datum_set:
            self.get_logger().info(
                'Awaiting initial GNSS lock to establish local ENU datum...',
                throttle_duration_sec=5.0)
            return

        self.ekf.predict(dt)

        # ── ZARU / ZUPT stationary clamp ──
        current_time = self.get_clock().now()
        if abs(self.latest_v_2d) < self.stationary_v_thresh:
            if self.stationary_start_time is None:
                self.stationary_start_time = current_time
            elapsed = (current_time -
                       self.stationary_start_time).nanoseconds / 1e9
            if elapsed >= self.stationary_duration:
                self.ekf.update_zupt_zaru()
        else:
            self.stationary_start_time = None

        # ── standard sensor fusion update ──
        z = [self.latest_x_gnss, self.latest_y_gnss,
             self.latest_v_2d, self.latest_omega]

        if self.gnss_received:
            mask = [True, True, True, True]
            self.gnss_received = False
        else:
            mask = [False, False, True, True]

        self.ekf.update(z, sensor_mask=mask)

        # ── divergence diagnostics ──
        if self.ekf.diverged:
            self._publish_divergence_diagnostic()
            self.get_logger().fatal(
                'EKF DIVERGENCE DETECTED. '
                'Broadcasting E-Stop and resetting filter.')
            self.ekf.reset(inflate=10.0)

        self.publish_state()

    def odom_callback(self, msg):
        self.latest_v = msg.twist.twist.linear.x

    # ==================================================================
    #  Publishing
    # ==================================================================
    def get_quaternion_from_yaw(self, yaw):
        q = Quaternion()
        q.w = math.cos(yaw / 2.0)
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        return q

    def publish_state(self):
        current_time = self.get_clock().now().to_msg()

        x_est     = float(self.ekf.x[0, 0])
        y_est     = float(self.ekf.x[1, 0])
        theta_est = float(self.ekf.x[2, 0])
        v_est     = float(self.ekf.x[3, 0])
        omega_est = float(self.ekf.x[4, 0])

        odom_msg = Odometry()
        odom_msg.header.stamp    = current_time
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id  = "base_link"

        odom_msg.pose.pose.position.x    = x_est
        odom_msg.pose.pose.position.y    = y_est
        odom_msg.pose.pose.orientation   = self.get_quaternion_from_yaw(theta_est)

        odom_msg.twist.twist.linear.x    = v_est
        odom_msg.twist.twist.angular.z   = omega_est

        self.odom_pub.publish(odom_msg)

        t = TransformStamped()
        t.header.stamp    = current_time
        t.header.frame_id = "odom"
        t.child_frame_id  = "base_link"
        t.transform.translation.x = x_est
        t.transform.translation.y = y_est
        t.transform.translation.z = 0.0
        t.transform.rotation = self.get_quaternion_from_yaw(theta_est)

        self.tf_broadcaster.sendTransform(t)

    # ==================================================================
    #  Divergence diagnostic publisher
    # ==================================================================
    def _publish_divergence_diagnostic(self):
        """
        Broadcast a fatal DiagnosticArray payload to the autonomous
        stack fault manager to trigger an E-Stop.
        """
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()

        status = DiagnosticStatus()
        status.level   = DiagnosticStatus.ERROR
        status.name    = 'ekf_fusion_node: Filter Health'
        status.message = ('FILTER DIVERGENCE - NIS exceeded chi2 threshold '
                          f'for {self.ekf.chi2_consec_limit} consecutive epochs. '
                          'E-Stop recommended.')
        status.hardware_id = 'ekf_fusion'
        status.values = [
            KeyValue(key='chi2_breach_count',
                     value=str(self.ekf.chi2_breach_count)),
            KeyValue(key='chi2_threshold',
                     value=str(self.ekf.chi2_threshold)),
            KeyValue(key='hdop',
                     value=f'{self.latest_hdop:.2f}'),
            KeyValue(key='gyro_bias',
                     value=f'{self.ekf.get_gyro_bias():.6f}'),
        ]
        msg.status.append(status)
        self.diag_pub.publish(msg)

    # ==================================================================
    #  State persistence callback
    # ==================================================================
    def _serialize_state_callback(self):
        """Periodic callback to serialize filter state to disk."""
        try:
            self.ekf.serialize_state(self.cache_path)
        except OSError as e:
            self.get_logger().warn(f"State cache write failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = EKFFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
