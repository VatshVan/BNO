import numpy as np
import math
import json
import os
import time


class DynamicExtendedKalmanFilter:
    def __init__(self, initial_state, p_matrix, q_matrix, r_matrix,
                 heading_r=0.1, zaru_r_omega=1e-4, zaru_r_v=1e-4,
                 cog_velocity_gate=1.5, wheelbase=1.0,
                 chi2_threshold=15.0, chi2_consec_limit=10):
        """
        Augmented EKF for a 4-wheel vehicle using CTRV kinematics with
        online gyroscope bias estimation, absolute heading injection,
        ZARU/ZUPT kinematic constraints, state persistence (warm boot),
        and innovation-based divergence diagnostics.

        State vector: [x, y, theta, v, omega, b_omega]^T
        - x, y       : position in local ENU frame (meters)
        - theta       : heading angle (radians)
        - v           : linear velocity (m/s)
        - omega       : bias-corrected angular velocity (rad/s)
        - b_omega     : gyroscope yaw-rate bias (rad/s)

        Measurement vector (standard update):
            z = [x_gps, y_gps, v_odom, omega_imu]^T
        The IMU measurement model accounts for bias:
            omega_imu = omega + b_omega

        Args:
            initial_state     : Initial state vector [x, y, theta, v, omega, b_omega]
            p_matrix          : Initial state covariance (6x6)
            q_matrix          : Process noise covariance (6x6)
            r_matrix          : Measurement noise covariance (4x4) for [x,y,v,omega_imu]
            heading_r         : Scalar measurement noise variance for heading updates
            zaru_r_omega      : ZARU pseudo-measurement noise variance for omega
            zaru_r_v          : ZUPT pseudo-measurement noise variance for v
            cog_velocity_gate : Minimum speed (m/s) to accept GNSS COG heading
            wheelbase         : Distance between axles (meters), reserved
            chi2_threshold    : NIS threshold for divergence detection (chi2(4) @ 99.5%)
            chi2_consec_limit : Consecutive NIS breaches before declaring divergence
        """
        self.x = np.array(initial_state, dtype=np.float64).reshape(-1, 1)
        self.P = np.array(p_matrix, dtype=np.float64)
        self.Q = np.array(q_matrix, dtype=np.float64)
        self.R = np.array(r_matrix, dtype=np.float64)
        self.R_base = self.R.copy()
        self.wheelbase = wheelbase

        # dimensions
        self.state_dim = 6
        self.meas_dim = 4

        # heading injection parameters
        self.heading_R = float(heading_r)
        self.cog_velocity_gate = float(cog_velocity_gate)

        # ZARU / ZUPT parameters
        self.zaru_r_omega = float(zaru_r_omega)
        self.zaru_r_v = float(zaru_r_v)

        # divergence diagnostics
        self.chi2_threshold = float(chi2_threshold)
        self.chi2_consec_limit = int(chi2_consec_limit)
        self.chi2_breach_count = 0
        self.diverged = False

        # ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)
        self.Q = 0.5 * (self.Q + self.Q.T)
        self.R = 0.5 * (self.R + self.R.T)
        self.R_base = 0.5 * (self.R_base + self.R_base.T)

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_angle(angle):
        """Normalizes angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def _kalman_gain(self, H, R):
        """Compute Kalman gain K = P H^T S^{-1} with Cholesky fallback."""
        S = H @ self.P @ H.T + R
        S = 0.5 * (S + S.T)
        try:
            L = np.linalg.cholesky(S)
            K = self.P @ H.T @ np.linalg.solve(
                L.T, np.linalg.solve(L, np.eye(S.shape[0])))
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.inv(S)
        return K, S

    def _joseph_update(self, K, H, R):
        """Joseph-form covariance update for numerical stability."""
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------
    #  Dynamic GNSS Covariance Scaling (HDOP/VDOP)
    # ------------------------------------------------------------------
    def scale_gnss_covariance(self, hdop):
        """
        Dynamically scale the positional elements of R using the HDOP
        metric broadcast by the GNSS receiver.

        R_pos = R_base_pos * hdop^2

        Args:
            hdop: Horizontal Dilution of Precision (scalar >= 1.0)
        """
        hdop = max(float(hdop), 1.0)
        scale = hdop * hdop
        self.R[0, 0] = self.R_base[0, 0] * scale
        self.R[1, 1] = self.R_base[1, 1] * scale

    # ------------------------------------------------------------------
    #  PREDICT  (CTRV with bias-corrected omega)
    # ------------------------------------------------------------------
    def predict(self, dt):
        """
        Propagate state and covariance using the CTRV motion model.

        The heading integration subtracts the estimated bias:
            theta_{k+1} = theta_k + (omega - b_omega) * dt
        """
        if dt <= 0:
            return

        x_prev     = float(self.x[0, 0])
        y_prev     = float(self.x[1, 0])
        theta_prev = float(self.x[2, 0])
        v_prev     = float(self.x[3, 0])
        omega_prev = float(self.x[4, 0])
        b_omega    = float(self.x[5, 0])

        omega_corr = omega_prev - b_omega
        omega_threshold = 1e-6

        if abs(omega_corr) < omega_threshold:
            cos_t = math.cos(theta_prev)
            sin_t = math.sin(theta_prev)

            x_new     = x_prev + v_prev * cos_t * dt
            y_new     = y_prev + v_prev * sin_t * dt
            theta_new = theta_prev

            F = np.array([
                [1, 0, -v_prev * sin_t * dt,  cos_t * dt,      0,    0],
                [0, 1,  v_prev * cos_t * dt,  sin_t * dt,      0,    0],
                [0, 0,  1,                     0,               dt, -dt],
                [0, 0,  0,                     1,               0,    0],
                [0, 0,  0,                     0,               1,    0],
                [0, 0,  0,                     0,               0,    1],
            ], dtype=np.float64)
        else:
            v_over_w   = v_prev / omega_corr
            theta_new  = theta_prev + omega_corr * dt

            sin_t  = math.sin(theta_prev)
            cos_t  = math.cos(theta_prev)
            sin_tn = math.sin(theta_new)
            cos_tn = math.cos(theta_new)

            x_new = x_prev + v_over_w * (sin_tn - sin_t)
            y_new = y_prev + v_over_w * (cos_t  - cos_tn)

            dxdtheta  = v_over_w * (cos_tn - cos_t)
            dxdv      = (sin_tn - sin_t) / omega_corr
            dxdw_corr = (v_prev / (omega_corr**2)) * (sin_t - sin_tn) \
                      + (v_prev / omega_corr) * cos_tn * dt

            dydtheta  = v_over_w * (sin_tn - sin_t)
            dydv      = (cos_t - cos_tn) / omega_corr
            dydw_corr = (v_prev / (omega_corr**2)) * (cos_tn - cos_t) \
                      + (v_prev / omega_corr) * sin_tn * dt

            F = np.array([
                [1, 0, dxdtheta, dxdv,  dxdw_corr, -dxdw_corr],
                [0, 1, dydtheta, dydv,  dydw_corr, -dydw_corr],
                [0, 0, 1,        0,     dt,         -dt       ],
                [0, 0, 0,        1,     0,           0        ],
                [0, 0, 0,        0,     1,           0        ],
                [0, 0, 0,        0,     0,           1        ],
            ], dtype=np.float64)

        self.x[0, 0] = x_new
        self.x[1, 0] = y_new
        self.x[2, 0] = self._normalize_angle(theta_new)
        self.x[3, 0] = v_prev
        self.x[4, 0] = omega_prev
        self.x[5, 0] = b_omega

        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------
    #  UPDATE  (standard sensor fusion)
    # ------------------------------------------------------------------
    def update(self, z, sensor_mask=None, R_override=None):
        """
        Correct state with fused measurements.

        Measurement vector z = [x_gps, y_gps, v_odom, omega_imu]^T

        Args:
            z           : measurement vector (4,)
            sensor_mask : boolean array to mask unavailable channels
            R_override  : optional (4x4) to replace self.R for this epoch
        Returns:
            (innovation, innovation_covariance)
        """
        z_vec = np.array(z, dtype=np.float64).reshape(-1, 1)

        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
        ], dtype=np.float64)

        R_use = np.array(R_override, dtype=np.float64) if R_override is not None else self.R

        if sensor_mask is not None:
            mask  = np.array(sensor_mask, dtype=bool)
            H     = H[mask, :]
            z_vec = z_vec[mask, :]
            R_use = R_use[np.ix_(mask, mask)]
        else:
            R = self.R
        
        h_x = H @ self.x
        y   = z_vec - h_x

        K, S = self._kalman_gain(H, R_use)

        # NIS-based divergence monitoring
        self._check_nis(y, S)

        self.x = self.x + K @ y
        self.x[2, 0] = self._normalize_angle(float(self.x[2, 0]))

        self._joseph_update(K, H, R_use)
        return y, S

    # ------------------------------------------------------------------
    #  HEADING UPDATE  (absolute heading injection)
    # ------------------------------------------------------------------
    def update_heading(self, theta_meas, r_heading=None):
        """
        Inject an absolute heading measurement (e.g. magnetometer or GNSS COG).

        Args:
            theta_meas : measured heading in radians
            r_heading  : scalar measurement noise variance (overrides default)
        Returns:
            (innovation, innovation_covariance)
        """
        r = r_heading if r_heading is not None else self.heading_R

        H = np.zeros((1, self.state_dim), dtype=np.float64)
        H[0, 2] = 1.0

        R = np.array([[r]], dtype=np.float64)

        h_x = H @ self.x
        y   = np.array([[self._normalize_angle(theta_meas - float(h_x[0, 0]))]],
                        dtype=np.float64)

        K, S = self._kalman_gain(H, R)

        self.x = self.x + K @ y
        self.x[2, 0] = self._normalize_angle(float(self.x[2, 0]))

        self._joseph_update(K, H, R)
        return y, S

    # ------------------------------------------------------------------
    #  ZARU / ZUPT  (stationary kinematic clamp)
    # ------------------------------------------------------------------
    def update_zupt_zaru(self, r_v=None, r_omega=None):
        """
        Zero-velocity / zero-angular-rate pseudo-measurement update.

        Args:
            r_v     : ZUPT noise variance (overrides default)
            r_omega : ZARU noise variance (overrides default)
        Returns:
            (innovation, innovation_covariance)
        """
        rv = r_v if r_v is not None else self.zaru_r_v
        rw = r_omega if r_omega is not None else self.zaru_r_omega

        H = np.zeros((2, self.state_dim), dtype=np.float64)
        H[0, 3] = 1.0
        H[1, 4] = 1.0

        R = np.diag([rv, rw]).astype(np.float64)

        z_vec = np.zeros((2, 1), dtype=np.float64)
        h_x   = H @ self.x
        y     = z_vec - h_x

        K, S = self._kalman_gain(H, R)

        self.x = self.x + K @ y
        self.x[2, 0] = self._normalize_angle(float(self.x[2, 0]))

        self._joseph_update(K, H, R)
        return y, S

    # ------------------------------------------------------------------
    #  Divergence Diagnostics (NIS monitoring)
    # ------------------------------------------------------------------
    def _check_nis(self, y, S):
        """
        Evaluate the Normalised Innovation Squared (NIS) against the
        chi-squared threshold.  If NIS exceeds the threshold for
        chi2_consec_limit consecutive epochs, the filter is diverged.

        NIS = y^T S^{-1} y  ~  chi2(dim(y))
        """
        try:
            S_inv = np.linalg.inv(S)
            nis = float((y.T @ S_inv @ y).item())
        except np.linalg.LinAlgError:
            nis = float('inf')

        if nis > self.chi2_threshold:
            self.chi2_breach_count += 1
        else:
            self.chi2_breach_count = 0

        if self.chi2_breach_count >= self.chi2_consec_limit:
            self.diverged = True
        else:
            self.diverged = False

    def reset(self, state=None, p_matrix=None, inflate=1.0):
        """
        Reset filter state to recover from divergence or warm-boot.

        Args:
            state    : new state vector (6,); if None, resets to zeros
            p_matrix : new covariance (6x6); if None uses identity * inflate
            inflate  : scalar inflation factor for P
        """
        if state is not None:
            self.x = np.array(state, dtype=np.float64).reshape(-1, 1)
        else:
            self.x = np.zeros((self.state_dim, 1), dtype=np.float64)

        if p_matrix is not None:
            self.P = np.array(p_matrix, dtype=np.float64) * inflate
        else:
            self.P = np.eye(self.state_dim, dtype=np.float64) * inflate

        self.P = 0.5 * (self.P + self.P.T)
        self.chi2_breach_count = 0
        self.diverged = False

    # ------------------------------------------------------------------
    #  State Persistence (Warm Boot)
    # ------------------------------------------------------------------
    def serialize_state(self, filepath):
        """
        Serialize the current state vector and covariance to a JSON file
        for non-volatile warm-boot recovery.

        Args:
            filepath: absolute path to the cache file
        """
        cache = {
            'timestamp': time.time(),
            'state': self.x.flatten().tolist(),
            'covariance': self.P.tolist(),
        }
        tmp = filepath + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(cache, f)
        os.replace(tmp, filepath)

    def deserialize_state(self, filepath, p_inflate=5.0, max_age_sec=3600.0):
        """
        Attempt to restore state from a JSON cache file.

        If the cache is valid and not too old, x and P are restored.
        P is inflated by p_inflate to account for unmeasured disturbances
        while powered down.

        Args:
            filepath    : absolute path to the cache file
            p_inflate   : scalar multiplier to inflate the cached P
            max_age_sec : maximum age (seconds) of the cache to accept

        Returns:
            True if state was restored, False otherwise
        """
        if not os.path.isfile(filepath):
            return False

        try:
            with open(filepath, 'r') as f:
                cache = json.load(f)

            age = time.time() - cache['timestamp']
            if age > max_age_sec:
                return False

            state = np.array(cache['state'], dtype=np.float64)
            cov   = np.array(cache['covariance'], dtype=np.float64)

            if state.shape != (self.state_dim,):
                return False
            if cov.shape != (self.state_dim, self.state_dim):
                return False

            self.x = state.reshape(-1, 1)
            self.P = cov * p_inflate
            self.P = 0.5 * (self.P + self.P.T)

            self.chi2_breach_count = 0
            self.diverged = False
            return True

        except (json.JSONDecodeError, KeyError, ValueError):
            return False

    # ------------------------------------------------------------------
    #  Lever Arm Compensation
    # ------------------------------------------------------------------
    @staticmethod
    def compensate_lever_arm(x_gnss, y_gnss, theta, lever_x, lever_y):
        """
        Project GNSS antenna position back to the vehicle kinematic
        center (base_link) by removing the rigid-body rotational offset.

        x_base = x_gnss - (l_x cos(theta) - l_y sin(theta))
        y_base = y_gnss - (l_x sin(theta) + l_y cos(theta))

        Args:
            x_gnss, y_gnss : GNSS position in ENU frame
            theta          : current heading estimate (rad)
            lever_x        : longitudinal offset of antenna from base_link (m)
            lever_y        : lateral offset of antenna from base_link (m)

        Returns:
            (x_base, y_base)
        """
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x_base = x_gnss - (lever_x * cos_t - lever_y * sin_t)
        y_base = y_gnss - (lever_x * sin_t + lever_y * cos_t)
        return x_base, y_base

    # ------------------------------------------------------------------
    #  Accessors
    # ------------------------------------------------------------------
    def get_state(self):
        """Returns the current state estimate as flat array."""
        return self.x.flatten()

    def get_covariance(self):
        """Returns a copy of the state covariance matrix."""
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
        """Returns current bias-corrected angular velocity."""
        return float(self.x[4, 0])

    def get_gyro_bias(self):
        """Returns current estimated gyroscope bias (rad/s)."""
        return float(self.x[5, 0])

    def mahalanobis_distance(self, z):
        """
        Computes Mahalanobis distance of measurement z for outlier gating.

        Args:
            z: Measurement vector [x, y, v, omega_imu]
        Returns:
            Mahalanobis distance (scalar)
        """
        z_vec = np.array(z, dtype=np.float64).reshape(-1, 1)

        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
        ], dtype=np.float64)

        h_x = H @ self.x
        y   = z_vec - h_x
        S   = H @ self.P @ H.T + self.R

        try:
            S_inv = np.linalg.inv(S)
            d = float(np.sqrt((y.T @ S_inv @ y).item()))
        except np.linalg.LinAlgError:
            d = float('inf')

        return d
