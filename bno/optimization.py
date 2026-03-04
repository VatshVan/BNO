import optuna
import numpy as np
import csv
import heapq
import math
import sys
from ekf_core import DynamicExtendedKalmanFilter

def read_csv_generator(file_path, sensor_type):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].isalpha():
                continue
            if any('nan' in str(val).lower() for val in row):
                continue
            yield (int(row[0]), sensor_type, row[1:])

def geodetic_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref):
    """
    Pure Python implementation of WGS84 Geodetic to ENU.
    Inputs must be in radians.
    """
    a = 6378137.0
    e_sq = 0.00669437999014
    
    def geodetic_to_ecef(phi, lam, h):
        N = a / math.sqrt(1 - e_sq * math.sin(phi)**2)
        X = (N + h) * math.cos(phi) * math.cos(lam)
        Y = (N + h) * math.cos(phi) * math.sin(lam)
        Z = (N * (1 - e_sq) + h) * math.sin(phi)
        return X, Y, Z
        
    X, Y, Z = geodetic_to_ecef(lat, lon, alt)
    Xr, Yr, Zr = geodetic_to_ecef(lat_ref, lon_ref, alt_ref)
    
    dX = X - Xr
    dY = Y - Yr
    dZ = Z - Zr
    
    sin_lat, cos_lat = math.sin(lat_ref), math.cos(lat_ref)
    sin_lon, cos_lon = math.sin(lon_ref), math.cos(lon_ref)
    
    e = -sin_lon * dX + cos_lon * dY
    n = -sin_lat * cos_lon * dX - sin_lat * sin_lon * dY + cos_lat * dZ
    u = cos_lat * cos_lon * dX + cos_lat * sin_lon * dY + sin_lat * dZ
    
    return e, n, u

def objective(trial):
    # Log-uniform sampling for covariance magnitudes
    q_pos = trial.suggest_float('q_pos', 1e-4, 1.0, log=True)
    q_theta = trial.suggest_float('q_theta', 1e-5, 1e-1, log=True)
    q_vel = trial.suggest_float('q_vel', 1e-3, 1.0, log=True)
    q_omega = trial.suggest_float('q_omega', 1e-4, 1.0, log=True)
    
    r_pos = trial.suggest_float('r_pos', 0.1, 50.0, log=True)
    r_vel = trial.suggest_float('r_vel', 0.01, 10.0, log=True)
    r_omega = trial.suggest_float('r_omega', 0.01, 5.0, log=True)
    
    p_matrix = np.eye(5).tolist()
    q_matrix = np.diag([q_pos, q_pos, q_theta, q_vel, q_omega]).tolist()
    r_matrix = np.diag([r_pos, r_pos, r_vel, r_omega]).tolist()
    
    ekf = DynamicExtendedKalmanFilter([0.0, 0.0, 0.0, 0.0, 0.0], p_matrix, q_matrix, r_matrix)
    
    data_dir = '/home/vatshvan/ros2_ws/src/bno/test_/data/nclt_data/'
    
    try:
        gen_wheels = read_csv_generator(data_dir + 'wheels.csv', 'wheel')
        gen_imu = read_csv_generator(data_dir + 'ms25.csv', 'imu')
        gen_gps = read_csv_generator(data_dir + 'gps.csv', 'gps')
        gen_gt = read_csv_generator(data_dir + 'groundtruth.csv', 'gt')
    except FileNotFoundError:
        print("Data directory missing or invalid.")
        sys.exit(1)
    
    stream = heapq.merge(gen_wheels, gen_imu, gen_gps, gen_gt, key=lambda x: x[0])
    
    # NCLT Datum Initialization (Radians)
    ref_lat = math.radians(42.293260)
    ref_lon = math.radians(-83.709691)
    ref_alt = 265.80
    
    last_time = None
    latest_v = 0.0
    latest_omega = 0.0
    
    errors = []
    
    try:
        for event in stream:
            ts, s_type, data = event
            
            if last_time is None:
                last_time = ts
                continue
                
            dt = (ts - last_time) / 1e6
            if dt > 0:
                ekf.predict(dt)
                last_time = ts
                
            if s_type == 'wheel':
                latest_v = (float(data[0]) + float(data[1])) / 2.0
            elif s_type == 'imu':
                latest_omega = float(data[8])
            elif s_type == 'gps':
                lat = float(data[2])
                lon = float(data[3])
                alt = float(data[4])
                
                enu_e, enu_n, _ = geodetic_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)
                
                z = [enu_e, enu_n, latest_v, latest_omega]
                
                # Check for Mahalanobis gate output based on your ekf_core setup
                result = ekf.update(z)
                
            elif s_type == 'gt':
                x_true, y_true = float(data[0]), float(data[1])
                x_est, y_est = ekf.get_position()
                errors.append((x_est - x_true)**2 + (y_est - y_true)**2)
                
    except StopIteration:
        pass
        
    if not errors:
        return float('inf')
        
    rmse = math.sqrt(sum(errors) / len(errors))
    return rmse

if __name__ == '__main__':
    # Streamline output for optimization tracking
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("Initiating EKF Covariance Optimization...")
    
    study = optuna.create_study(direction='minimize')
    
    # Execute 100 trials
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    print("\nOptimization Matrix Terminated.")
    print(f"Optimal RMSE: {study.best_value:.3f} meters")
    print("Optimal Covariance Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.6f}")