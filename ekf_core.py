import numpy as np
import math

class DynamicExtendedKalmanFilter:
    def __init__(self, initial_state, p_matrix, q_matrix, r_matrix, wheelbase=1.0):
        """
        Initializes the EKF for a 4-wheel vehicle using bicycle model kinematics.
        
        State vector: [x, y, theta, v, omega]^T
        - x, y: position in local frame (meters)
        - theta: heading angle (radians)
        - v: linear velocity (m/s)
        - omega: angular velocity (rad/s)
        
        Args:
            initial_state: Initial state vector [x, y, theta, v, omega]
            p_matrix: Initial state covariance matrix (5x5)
            q_matrix: Process noise covariance matrix (5x5)
            r_matrix: Measurement noise covariance matrix (4x4)
            wheelbase: Distance between front and rear axles (meters)
        """
        self.x = np.array(initial_state, dtype=np.float64).reshape(-1, 1)
        self.P = np.array(p_matrix, dtype=np.float64)
        self.Q = np.array(q_matrix, dtype=np.float64)
        self.R = np.array(r_matrix, dtype=np.float64)
        self.wheelbase = wheelbase
        
        # System dimensions
        self.state_dim = 5
        self.meas_dim = 4
        
        # Ensure covariance matrices are symmetric and positive semi-definite
        self.P = 0.5 * (self.P + self.P.T)
        self.Q = 0.5 * (self.Q + self.Q.T)
        self.R = 0.5 * (self.R + self.R.T)

    def _normalize_angle(self, angle):
        """
        Normalizes angle to [-pi, pi] range.
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def predict(self, dt):
        """
        Propagates the state and covariance matrix forward in time using
        the constant turn rate and velocity (CTRV) motion model.
        
        Motion Model (for 4-wheel vehicle):
        - If omega ≈ 0 (straight line motion):
            x_new = x + v * cos(theta) * dt
            y_new = y + v * sin(theta) * dt
            theta_new = theta
        - If omega ≠ 0 (turning motion):
            x_new = x + (v/omega) * (sin(theta + omega*dt) - sin(theta))
            y_new = y + (v/omega) * (cos(theta) - cos(theta + omega*dt))
            theta_new = theta + omega * dt
        """
        if dt <= 0:
            return
            
        x_prev = float(self.x[0, 0])
        y_prev = float(self.x[1, 0])
        theta_prev = float(self.x[2, 0])
        v_prev = float(self.x[3, 0])
        omega_prev = float(self.x[4, 0])
        
        # Threshold for considering omega as zero (avoid division by zero)
        omega_threshold = 1e-6
        
        if abs(omega_prev) < omega_threshold:
            # Straight line motion model
            x_new = x_prev + v_prev * math.cos(theta_prev) * dt
            y_new = y_prev + v_prev * math.sin(theta_prev) * dt
            theta_new = theta_prev
            
            # Jacobian for straight line motion
            F = np.array([
                [1, 0, -v_prev * math.sin(theta_prev) * dt, math.cos(theta_prev) * dt, 0],
                [0, 1,  v_prev * math.cos(theta_prev) * dt, math.sin(theta_prev) * dt, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]
            ], dtype=np.float64)
        else:
            # CTRV motion model (turning)
            v_over_omega = v_prev / omega_prev
            theta_new = theta_prev + omega_prev * dt
            
            sin_theta = math.sin(theta_prev)
            cos_theta = math.cos(theta_prev)
            sin_theta_new = math.sin(theta_new)
            cos_theta_new = math.cos(theta_new)
            
            x_new = x_prev + v_over_omega * (sin_theta_new - sin_theta)
            y_new = y_prev + v_over_omega * (cos_theta - cos_theta_new)
            
            # Jacobian for CTRV motion model
            # Partial derivatives
            dxdtheta = v_over_omega * (cos_theta_new - cos_theta)
            dxdv = (1.0 / omega_prev) * (sin_theta_new - sin_theta)
            dxdomega = (v_prev / (omega_prev * omega_prev)) * (sin_theta - sin_theta_new) + \
                       (v_prev / omega_prev) * cos_theta_new * dt
            
            dydtheta = v_over_omega * (sin_theta_new - sin_theta)
            dydv = (1.0 / omega_prev) * (cos_theta - cos_theta_new)
            dydomega = (v_prev / (omega_prev * omega_prev)) * (cos_theta_new - cos_theta) + \
                       (v_prev / omega_prev) * sin_theta_new * dt
            
            F = np.array([
                [1, 0, dxdtheta, dxdv, dxdomega],
                [0, 1, dydtheta, dydv, dydomega],
                [0, 0, 1, 0, dt],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]
            ], dtype=np.float64)
        
        # Update state
        self.x[0, 0] = x_new
        self.x[1, 0] = y_new
        self.x[2, 0] = self._normalize_angle(theta_new)
        self.x[3, 0] = v_prev  # Velocity assumed constant
        self.x[4, 0] = omega_prev  # Angular velocity assumed constant
        
        # Process noise transformation matrix (maps process noise to state space)
        # Accounts for acceleration and angular acceleration disturbances
        G = np.array([
            [0.5 * dt * dt * math.cos(theta_prev), 0],
            [0.5 * dt * dt * math.sin(theta_prev), 0],
            [0, 0.5 * dt * dt],
            [dt, 0],
            [0, dt]
        ], dtype=np.float64)
        
        # Covariance prediction with Joseph form for numerical stability
        self.P = F @ self.P @ F.T + self.Q
        
        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z, sensor_mask=None):
        """
        Corrects the state estimate using fused sensor measurements.
        
        Measurement vector z = [x_meas, y_meas, v_meas, omega_meas]^T
        - x_meas, y_meas: Position from GNSS (pre-transformed to local frame)
        - v_meas: Linear velocity from odometry/wheel encoders
        - omega_meas: Angular velocity from IMU
        
        Args:
            z: Measurement vector [x, y, v, omega]
            sensor_mask: Optional boolean array to mask unavailable measurements
        """
        z_vector = np.array(z, dtype=np.float64).reshape(-1, 1)
        
        # Observation matrix (linear measurement model)
        # Maps state to measurement: z = H * x
        H = np.array([
            [1, 0, 0, 0, 0],  # x measurement
            [0, 1, 0, 0, 0],  # y measurement
            [0, 0, 0, 1, 0],  # velocity measurement
            [0, 0, 0, 0, 1]   # angular velocity measurement
        ], dtype=np.float64)
        
        # Apply sensor mask if provided (for partial measurements)
        if sensor_mask is not None:
            mask = np.array(sensor_mask, dtype=bool)
            H = H[mask, :]
            z_vector = z_vector[mask, :]
            R = self.R[np.ix_(mask, mask)]
        else:
            R = self.R
        
        # Predicted measurement
        h_x = H @ self.x
        
        # Innovation (measurement residual)
        y = z_vector - h_x
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Ensure S is symmetric
        S = 0.5 * (S + S.T)
        
        # Kalman gain using Cholesky decomposition for numerical stability
        try:
            L = np.linalg.cholesky(S)
            K = self.P @ H.T @ np.linalg.solve(L.T, np.linalg.solve(L, np.eye(S.shape[0])))
        except np.linalg.LinAlgError:
            # Fallback to standard inverse if Cholesky fails
            K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Normalize heading angle
        self.x[2, 0] = self._normalize_angle(float(self.x[2, 0]))
        
        # Covariance update using Joseph form for numerical stability
        # P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        # Ensure symmetry and positive semi-definiteness
        self.P = 0.5 * (self.P + self.P.T)
        
        # Return innovation for outlier detection
        return y, S

    def get_state(self):
        """Returns the current state estimate."""
        return self.x.flatten()

    def get_covariance(self):
        """Returns the current state covariance matrix."""
        return self.P.copy()

    def get_position(self):
        """Returns current position (x, y)."""
        return float(self.x[0, 0]), float(self.x[1, 0])

    def get_heading(self):
        """Returns current heading angle in radians."""
        return float(self.x[2, 0])

    def get_velocity(self):
        """Returns current linear velocity."""
        return float(self.x[3, 0])

    def get_angular_velocity(self):
        """Returns current angular velocity."""
        return float(self.x[4, 0])

    def mahalanobis_distance(self, z):
        """
        Computes Mahalanobis distance for outlier detection.
        
        Args:
            z: Measurement vector
            
        Returns:
            Mahalanobis distance (scalar)
        """
        z_vector = np.array(z, dtype=np.float64).reshape(-1, 1)
        
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ], dtype=np.float64)
        
        h_x = H @ self.x
        y = z_vector - h_x
        S = H @ self.P @ H.T + self.R
        
        try:
            S_inv = np.linalg.inv(S)
            d = float(np.sqrt(y.T @ S_inv @ y))
        except np.linalg.LinAlgError:
            d = float('inf')
        
        return d