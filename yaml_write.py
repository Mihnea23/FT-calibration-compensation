#!/usr/bin/env python3

import bagpy
from bagpy import bagreader
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import yaml
import sys
import os

# --- CONFIGURATION ---
BAG_DYNAMIC = 'YOUR_DYNAMIC_BAG'
BAG_STATIC = 'YOUR_STATIC_BAG'
OUTPUT_YAML = 'calib.yaml'

def load_synced_data(bag_path):
    print(f"Loading {bag_path}...")
    if not os.path.exists(bag_path):
        raise FileNotFoundError(f"File not found: {bag_path}")

    b = bagreader(bag_path)
    sys.stdout = open(os.devnull, 'w')
    ft_csv = b.message_by_topic('/bus0/ft_sensor0/ft_sensor_readings/wrench')
    imu_csv = b.message_by_topic('/imu/data_raw')
    sys.stdout = sys.__stdout__
    
    df_ft = pd.read_csv(ft_csv)
    df_imu = pd.read_csv(imu_csv)
    
    df_ft['Time'] = df_ft['Time'].astype(float)
    df_imu['Time'] = df_imu['Time'].astype(float)
    
    # Sync (Tolerance 50ms)
    df = pd.merge_asof(
        df_ft.sort_values('Time'),
        df_imu.sort_values('Time'),
        on='Time',
        direction='nearest',
        tolerance=0.05 
    )
    
    forces = df[['wrench.force.x', 'wrench.force.y', 'wrench.force.z', 
                 'wrench.torque.x', 'wrench.torque.y', 'wrench.torque.z']].to_numpy()
    
    accels = df[['linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z']].to_numpy()
    
    valid_mask = ~np.isnan(forces).any(axis=1) & ~np.isnan(accels).any(axis=1)
    return forces[valid_mask], accels[valid_mask]

def get_static_properties(bag_path):
    """ Calculates Covariance from static data """
    print(f"\n--- Analyzing Static Data ({bag_path}) ---")
    forces, accels = load_synced_data(bag_path)
    
    if len(forces) < 100:
        raise ValueError("Not enough static data")

    # 1. Covariance (Sensor Noise)
    # Center data to get pure noise profile
    forces_centered = forces - np.mean(forces, axis=0)
    covariance = np.cov(forces_centered, rowvar=False)
    noise_std = np.sqrt(np.diag(covariance))
    print(f"Noise Std Dev (Fxyz, Txyz): {np.round(noise_std, 4)}")
    
    bias_init = np.mean(forces, axis=0)

    return covariance, bias_init

def apply_rotation(vecs, r, p, y):
    rot = R.from_euler('xyz', [r, p, y], degrees=False).as_matrix()
    return vecs @ rot.T

# --- PHYSICS MODELS ---

def physics_model(params, accels_raw): # Predicts wrench from mass, CoM, bias, IMU rotation and force scale
    mass = params[0]
    com = params[1:4]
    bias = params[4:10]
    rot = params[10:13]
    scale = params[13:16]
    
    # 1. Align IMU
    accels_aligned = apply_rotation(accels_raw, *rot)
    
    # 2. Gravity Vector (Static assumption: Gravity = -Accel)
    g_sensors = -1.0 * accels_aligned
    norms = np.linalg.norm(g_sensors, axis=1, keepdims=True)
    # Safety check
    norms[norms < 1e-6] = 1.0
    g_vectors = (g_sensors / norms) * 9.81
    
    # 3. Physics Wrench
    F_grav = mass * g_vectors
    T_grav = np.cross(com, F_grav)
    
    # 4. Model Prediction (Measured = Scale * Ideal + Bias)
    F_measured_model = F_grav * scale 
    return np.hstack([F_measured_model, T_grav]) + bias

def residual(params, forces, accels):
    return (forces - physics_model(params, accels)).flatten()

if __name__ == "__main__":
    # --- Step 1: STATIC ANALYSIS ---
    try:
        cov_matrix, bias_init = get_static_properties(BAG_STATIC)
    except Exception as e:
        print(f"Error in static analysis: {e}")
        # Fallback covariance
        cov_matrix = np.eye(6) * 1e-4
        bias_init = np.zeros(6)

    # --- Step 2: DYNAMIC OPTIMIZATION ---
    print(f"\n--- Analyzing Dynamic Data ({BAG_DYNAMIC}) ---")
    forces, accels = load_synced_data(BAG_DYNAMIC)
    
    print(f"Optimizing on {len(forces)} samples...")

    x0 = np.concatenate([
        [0.5],                      # Mass
        [0.0, 0.0, 0.05],           # CoM
        bias_init,                  # Bias
        [0.0, 0.0, 0.0],            # Rotation
        [1.0, 1.0, 1.0]             # Force Scale
    ])

    res = least_squares(residual, x0, args=(forces, accels), verbose=2)
    p = res.x

    # --- Step 3: CALCULATE C MATRIX & SAVE ---
    scale_factors = p[13:16]
    C_diag = 1.0 / scale_factors
    
    # Build 6x6 Identity with corrected Force diagonals
    C_matrix = np.eye(6)
    C_matrix[0,0] = C_diag[0]
    C_matrix[1,1] = C_diag[1]
    C_matrix[2,2] = C_diag[2]

    print("\n=== FINAL RESULTS ===")
    print(f"Mass:      {p[0]:.4f} kg")
    print(f"Force Scale: {scale_factors}")
    print(f"IMU Rot:   {np.degrees(p[10:13])}")

    calib_data = {
        'm': float(p[0]),
        'com': p[1:4].tolist(),
        'bias': p[4:10].tolist(),
        'imu_rot': p[10:13].tolist(),
        'C': C_matrix.tolist(),
        'covariance': cov_matrix.tolist(),
        'force_scale': scale_factors.tolist(),
        'frame_id': 'ft_sensor0'
    }

    with open(OUTPUT_YAML, 'w') as f:
        yaml.dump(calib_data, f)
    print(f"\nSaved complete calibration to {os.path.abspath(OUTPUT_YAML)}")