#!/usr/bin/env python3

import rospy
import yaml
import numpy as np
from geometry_msgs.msg import WrenchStamped, Vector3, PoseStamped
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation as R

# ---------------- Madgwick Filter ----------------
class MadgwickFilter:
    def __init__(self, beta=0.033, freq=100.0):
        self.beta = beta
        self.freq = freq
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, gyro, accel):
        q = self.q
        norm = np.linalg.norm(accel)
        if norm == 0: return self.q
        ax, ay, az = accel / norm
        gx, gy, gz = gyro
        
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - ax,
            2*(q[0]*q[1] + q[2]*q[3]) - ay,
            2*(0.5 - q[1]**2 - q[2]**2) - az
        ])
        J = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
            [0, -4*q[1], -4*q[2], 0]
        ])
        step = J.T @ f
        step /= np.linalg.norm(step)

        q_dot = 0.5 * np.array([
            -q[1]*gx - q[2]*gy - q[3]*gz,
             q[0]*gx + q[2]*gz - q[3]*gy,
             q[0]*gy - q[1]*gz + q[3]*gx,
             q[0]*gz + q[1]*gy - q[2]*gx
        ])
        q_dot -= self.beta * step
        self.q += q_dot * (1.0 / self.freq)
        self.q /= np.linalg.norm(self.q)
        return self.q

# ---------------- Main Node ----------------

class FTCompensatorMadgwickOnly(object):
    def __init__(self):
        rospy.init_node('ft_compensator_madgwick_only', anonymous=False)

        # --- Parameters ---
        default_yaml = '~/calib.yaml'
        self.calib_file = rospy.get_param('~calib_yaml', default_yaml)
        
        self.imu_topic = rospy.get_param('~imu_topic', '/imu/data')
        self.ft_topic = rospy.get_param('~ft_topic', '/bus0/ft_sensor0/ft_sensor_readings/wrench')
        self.pub_topic = rospy.get_param('~publish_topic', '/ft_compensated')
        self.debug = rospy.get_param('~debug', True)

        # --- Filter & State ---
        self.madgwick = MadgwickFilter(beta=0.033, freq=100.0)
        
        self.load_calibration(self.calib_file)
        
        self.q_fused = np.array([1.0, 0.0, 0.0, 0.0])
        self.accel_raw = None
        self.gyro_raw = None 
        self.w_raw = None
        
        # Stats Storage - Z-axis only
        self.fz_errors = []
        self.tz_errors = []
        self.last_print_time = rospy.get_time()
        
        #ArUco watchdog (for a more accurate result comparison)
        self.last_aruco_time = None
        self.aruco_lost = False
        rospy.Subscriber('/robot_pose', PoseStamped, self.aruco_watchdog_cb)

        # --- Subscribers ---
        rospy.Subscriber(self.imu_topic, Imu, self.imu_cb)
        rospy.Subscriber(self.ft_topic, WrenchStamped, self.ft_cb_wrench)

        self.pub = rospy.Publisher(self.pub_topic, WrenchStamped, queue_size=10)
        
        rospy.loginfo("FT Compensator Ready.")

    def run(self):
        r = rospy.Rate(100)
        try:
            while not rospy.is_shutdown():
                self.update()
                r.sleep()
        except rospy.ROSInterruptException:
            pass
        finally:
            self.print_final_stats()

    def load_calibration(self, path):
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            self.mass = float(data.get('m', 0.0))
            self.com = np.array(data.get('r', [0,0,0]), dtype=float)
            if 'com' in data: self.com = np.array(data['com'], dtype=float)
            
            self.bias = np.array(data.get('bias', [0]*6), dtype=float)
            
            # --- Load Frame Alignment Rotation ---
            r_imu, p_imu, y_imu = data.get('imu_rot', [0,0,0])
            self.R_align = R.from_euler('xyz', [r_imu, p_imu, y_imu], degrees=False).as_matrix()

            if 'C' in data and data['C'] is not None:
                self.C = np.array(data['C'], dtype=float).reshape(6,6)
            else:
                scales = np.array(data.get('force_scale', [1,1,1]), dtype=float)
                scales[np.abs(scales)<1e-6]=1.0
                c_diags = 1.0/scales
                self.C = np.eye(6)
                self.C[0,0], self.C[1,1], self.C[2,2] = c_diags
                
            self.frame_id = data.get('frame_id', 'ft_sensor0')
        except Exception as e:
            rospy.logerr(f"Calib Error: {e}")
            self.C = np.eye(6)
            self.bias = np.zeros(6)
            self.mass = 0.0
            self.com = np.zeros(3)
            self.R_align = np.eye(3)
            
        # Covariance for Noise Gating
        cov = np.array(data.get('covariance', np.eye(6)))
        self.noise_std = np.sqrt(np.diag(cov))
        self.noise_gate = 3.0 * self.noise_std

    def imu_cb(self, msg):
        self.accel_raw = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        self.gyro_raw = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.q_fused = self.madgwick.update(self.gyro_raw, self.accel_raw)

    def ft_cb_wrench(self, msg):
        w = msg.wrench
        self.w_raw = np.array([w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z])
        self.frame_id = msg.header.frame_id
        
    def aruco_watchdog_cb(self, msg):
        self.last_aruco_time = rospy.get_time()

    def update(self):
        if self.w_raw is None or self.accel_raw is None or self.gyro_raw is None:
            return

        if self.last_aruco_time is not None:
            aruco_age = rospy.get_time() - self.last_aruco_time
            if aruco_age >= 1.0:
                if not self.aruco_lost:
                    self.aruco_lost = True
                    print("\n[WATCHDOG] ArUco lost — stopping data collection.")
                    self.print_final_stats()
                return

        if self.aruco_lost:
            return

        # Align raw IMU measurements to sensor frame
        accel_aligned = self.R_align @ self.accel_raw
        gyro_aligned  = self.R_align @ self.gyro_raw

        # Extract gravity vector from Madgwick orientation
        r_fused  = R.from_quat([self.q_fused[1], self.q_fused[2],
                                self.q_fused[3], self.q_fused[0]])
        g_world  = np.array([0.0, 0.0, -9.81])
        g_sensor = self.R_align @ r_fused.inv().apply(g_world)

        # Separate linear acceleration from gravitational component
        a_linear = accel_aligned - g_sensor

        # Inertial force from linear acceleration only
        F_base = -self.mass * a_linear

        # Gravitational force from Madgwick orientation
        F_gravity = -self.mass * g_sensor

        # Centrifugal force
        wx_r          = np.cross(gyro_aligned, self.com)
        F_centrifugal = self.mass * np.cross(gyro_aligned, wx_r)

        # Total physics force and torque in sensor frame
        F_physics = F_base + F_gravity - F_centrifugal
        T_physics = np.cross(self.com, F_physics)
        W_physics = np.concatenate([F_physics, T_physics])

        # Apply calibration and subtract physics wrench
        W_centered  = self.w_raw - self.bias
        W_corrected = self.C @ W_centered
        W_comp      = W_corrected - W_physics

        # Noise gating
        for i in range(6):
            if abs(W_comp[i]) < self.noise_gate[i]:
                W_comp[i] = 0.0
      
        # Project compensated wrench into world frame
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

        # Publish compensated wrench
        msg = WrenchStamped()
        msg.header.stamp    = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.wrench.force    = Vector3(*W_comp_world[0:3])
        msg.wrench.torque   = Vector3(*W_comp_world[3:6])
        self.pub.publish(msg)
        
    def print_final_stats(self):
        if not self.fz_errors:
            print("\n[Stats] No data collected.")
            return

        fz_data = np.array(self.fz_errors)
        tz_data = np.array(self.tz_errors)

        print("\n" + "="*70)
        print("FINAL STATISTICS - Z-AXIS COMPENSATION (Madgwick only)")
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
    node = FTCompensatorMadgwickOnly()
    node.run()