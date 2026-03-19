#!/usr/bin/env python3

import rospy
import yaml
import numpy as np
from geometry_msgs.msg import WrenchStamped, Vector3, PoseStamped
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray

# ---------------- Helper Classes ----------------

class LowPassFilter:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.val = None

    def update(self, new_val):
        new_val = np.array(new_val, dtype=float)
        if self.val is None:
            self.val = new_val
        else:
            self.val = self.alpha * new_val + (1.0 - self.alpha) * self.val
        return self.val

def euler_to_rotmat(r, p, y):
    cx, sx = np.cos(r), np.sin(r)
    cy, sy = np.cos(p), np.sin(p)
    cz, sz = np.cos(y), np.sin(y)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

# ---------------- Main Node ----------------

class FTCompensatorLPFOnly(object):
    def __init__(self):
        rospy.init_node('ft_compensator_lpf_only', anonymous=False)

        # --- Parameters ---
        default_yaml = '~/calib.yaml'
        self.calib_file = rospy.get_param('~calib_yaml', default_yaml)
        
        self.imu_topic = rospy.get_param('~imu_topic', '/imu/data')
        self.ft_topic = rospy.get_param('~ft_topic', '/bus0/ft_sensor0/ft_sensor_readings/wrench')
        self.pub_topic = rospy.get_param('~publish_topic', '/ft_compensated')
        
        self.ft_raw_is_wrench = rospy.get_param('~ft_raw_is_wrench', True)
        self.debug = rospy.get_param('~debug', True)

        # --- Phase-Matched Filters ---
        self.accel_lpf = LowPassFilter(alpha=0.15) 
        self.ft_lpf = LowPassFilter(alpha=0.15) 

        # --- Load Calibration ---
        self.load_calibration(self.calib_file)

        # --- State ---
        self.accel_filtered = None
        self.w_filtered = None
        
        # Stats Storage - Z-axis only
        self.fz_errors = []
        self.tz_errors = []
        self.max_z_samples = 20000 
        
        #ArUco watchdog (for a more accurate result comparison)
        self.last_aruco_time = None
        self.aruco_lost = False
        self.q_aruco = None
        rospy.Subscriber('/robot_pose', PoseStamped, self.aruco_watchdog_cb)
        
        # Debugging
        self.peak_err = np.zeros(3)
        self.last_print_time = rospy.get_time()

        # --- Subscribers ---
        rospy.Subscriber(self.imu_topic, Imu, self.imu_cb)
        
        if self.ft_raw_is_wrench:
            rospy.Subscriber(self.ft_topic, WrenchStamped, self.ft_cb_wrench)
        else:
            rospy.Subscriber(self.ft_topic, Float32MultiArray, self.ft_cb_array)

        # --- Publisher ---
        self.pub = rospy.Publisher(self.pub_topic, WrenchStamped, queue_size=10)
        
        rospy.loginfo(f"FT Compensator Ready.")
        rospy.on_shutdown(self.print_stats)
        
        # Main Loop
        r = rospy.Rate(100)
        while not rospy.is_shutdown():
            self.update()
            r.sleep()

    def load_calibration(self, path):
        """ Loads parameters and builds C matrix dynamically """
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            
            self.mass = float(data.get('m', 0.0))
            self.com = np.array(data.get('r', [0,0,0]), dtype=float)
            if 'com' in data: self.com = np.array(data['com'], dtype=float)
            
            self.bias = np.array(data.get('bias', np.zeros(6)), dtype=float)
            if 'o' in data: self.bias = np.array(data['o'], dtype=float)

            self.imu_rot = np.array(data.get('imu_rot', [0,0,0]), dtype=float)
            self.R_align = euler_to_rotmat(*self.imu_rot)
            
            if 'C' in data and data['C'] is not None:
                self.C = np.array(data['C'], dtype=float).reshape(6,6)
                rospy.loginfo("Loaded Matrix C from YAML.")
            else:
                scales = np.array(data.get('force_scale', [1.0, 1.0, 1.0]), dtype=float)
                scales[np.abs(scales) < 1e-6] = 1.0 
                c_diags = 1.0 / scales
                self.C = np.eye(6)
                self.C[0,0] = c_diags[0]
                self.C[1,1] = c_diags[1]
                self.C[2,2] = c_diags[2]
                rospy.loginfo(f"Calculated Matrix C from scales: {scales}")

            self.frame_id = data.get('frame_id', 'ft_sensor0')

        except Exception as e:
            rospy.logerr(f"Failed to load calibration: {e}")
            self.C = np.eye(6)
            self.bias = np.zeros(6)
            self.mass = 0.0
            self.com = np.zeros(3)
            self.R_align = np.eye(3)

        cov = np.array(data.get('covariance', np.eye(6)))
        self.noise_std = np.sqrt(np.diag(cov))
        self.noise_gate = 3.0 * self.noise_std
        
    def imu_cb(self, msg):
        raw_accel = np.array([
            msg.linear_acceleration.x, 
            msg.linear_acceleration.y, 
            msg.linear_acceleration.z
        ])
        self.accel_filtered = self.accel_lpf.update(raw_accel)

    def ft_cb_wrench(self, msg):
        w = msg.wrench
        raw_w = np.array([w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z])
        self.w_filtered = self.ft_lpf.update(raw_w)
        self.frame_id = msg.header.frame_id

    def ft_cb_array(self, msg):
        raw_w = np.array(msg.data[0:6])
        self.w_filtered = self.ft_lpf.update(raw_w)
        
    def aruco_watchdog_cb(self, msg):
        self.last_aruco_time = rospy.get_time()
        q = msg.pose.orientation
        self.q_aruco = np.array([q.w, q.x, q.y, q.z])

    def update(self):
        if self.accel_filtered is None or self.w_filtered is None:
            return

        if self.last_aruco_time is not None:
            aruco_age = rospy.get_time() - self.last_aruco_time
            if aruco_age >= 1.0:
                if not self.aruco_lost:
                    self.aruco_lost = True
                    print("\n[WATCHDOG] ArUco lost — stopping data collection.")
                    self.print_stats()
                return

        if self.aruco_lost:
            return

        if self.q_aruco is None:
            return

        # Align IMU Acceleration to FT Frame
        accel_aligned = self.R_align @ self.accel_filtered

        # Dynamic Inertial Compensation (LPF only — no gravity separation)
        F_physics = -1.0 * self.mass * accel_aligned
        T_physics = np.cross(self.com, F_physics)
        W_physics = np.concatenate([F_physics, T_physics])

        # Apply Calibration
        W_centered  = self.w_filtered - self.bias
        W_corrected = self.C @ W_centered
        W_comp      = W_corrected - W_physics

        # Noise gating
        for i in range(6):
            if abs(W_comp[i]) < self.noise_gate[i]:
                W_comp[i] = 0.0

        # Project compensated wrench into world frame using ArUco orientation
        r_fused = R.from_quat([self.q_aruco[1], self.q_aruco[2],
                               self.q_aruco[3], self.q_aruco[0]])
        W_comp_world = np.concatenate([
            r_fused.apply(W_comp[0:3]),
            r_fused.apply(W_comp[3:6])
        ])

        # Store Z-axis statistics
        self.fz_errors.append(W_comp_world[2])
        self.tz_errors.append(W_comp_world[5])

        # Debug output
        if self.debug:
            now = rospy.get_time()
            if now - self.last_print_time > 0.2:
                print(f"ArUco ✓ | Fz: {W_comp[2]:6.3f} N | Tz: {W_comp[5]:6.3f} N·m")
                self.last_print_time = now

        # Publish compensated wrench in world frame
        msg = WrenchStamped()
        msg.header.stamp    = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.wrench.force    = Vector3(*W_comp_world[0:3])
        msg.wrench.torque   = Vector3(*W_comp_world[3:6])
        self.pub.publish(msg)

    def print_stats(self):
        if not hasattr(self, "fz_errors") or len(self.fz_errors) == 0:
            print("\n[Stats] No data collected.")
            return

        fz_data = np.array(self.fz_errors, dtype=float)
        tz_data = np.array(self.tz_errors, dtype=float)

        print("\n" + "="*70)
        print("FINAL STATISTICS - Z-AXIS COMPENSATION (LPF only)")
        print("="*70)
        
        print(f"\nForce Z-axis:")
        print(f"  Samples:          {len(fz_data)}")
        print(f"  Mean Bias:        {np.mean(fz_data):8.5f} N")
        print(f"  Mean Abs Error:   {np.mean(np.abs(fz_data)):8.5f} N")
        print(f"  Std Dev:          {np.std(fz_data, ddof=1):8.5f} N")

        print(f"\nTorque Z-axis:")
        print(f"  Samples:          {len(tz_data)}")
        print(f"  Mean Bias:        {np.mean(tz_data):8.5f} N·m")
        print(f"  Mean Abs Error:   {np.mean(np.abs(tz_data)):8.5f} N·m")
        print(f"  Std Dev:          {np.std(tz_data, ddof=1):8.5f} N·m")

        print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    try:
        FTCompensatorLPFOnly()
    except rospy.ROSInterruptException:
        pass