#!/usr/bin/env python3

import rospy
import yaml
import numpy as np
from geometry_msgs.msg import WrenchStamped, Vector3, PoseStamped
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

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

def perform_slerp(q1, q2, fraction):
    """ Spherical Linear Interpolation """
    
    r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
    
    key_rots = R.from_quat(np.vstack([r1.as_quat(), r2.as_quat()]))
    slerp = Slerp([0, 1], key_rots)
    
    interp_rot = slerp([fraction])
    x, y, z, w = interp_rot.as_quat()[0]
    return np.array([w, x, y, z])

# ---------------- Main Node ----------------

class FTCompensatorMadgwickArUco(object):
    def __init__(self):
        rospy.init_node('ft_compensator_madgwick_aruco')

        # Config
        self.calib_file = rospy.get_param('~calib_yaml', '~/calib.yaml')
        self.load_calibration(self.calib_file)
        self.debug = True
        
        # State
        self.madgwick = MadgwickFilter(beta=0.033)
        self.q_aruco = None
        self.aruco_weight = 0.02
        self.last_aruco_time = None
        self.aruco_lost = False
        
        self.accel_raw = None
        self.gyro_raw = None
        self.w_raw = None
        
        # Error tracking (Z-axis only)
        self.fz_errors = [] 
        self.tz_errors = []  
        self.last_print_time = rospy.get_time()

        # Subs/Pubs
        rospy.Subscriber('/imu/data', Imu, self.imu_cb)
        rospy.Subscriber('/robot_pose', PoseStamped, self.aruco_cb)
        rospy.Subscriber('/bus0/ft_sensor0/ft_sensor_readings/wrench', WrenchStamped, self.ft_cb)
        self.pub = rospy.Publisher('/ft_compensated', WrenchStamped, queue_size=10)
        
        rospy.loginfo("FT Compensator Ready.")
        
        # Loop for Safe Exit
        r = rospy.Rate(100)
        try:
            while not rospy.is_shutdown():
                self.update()
                r.sleep()
        except rospy.ROSInterruptException:
            pass
        finally:
            self.print_stats()

    def load_calibration(self, path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        self.m = float(data['m'])
        self.r = np.array(data['com'])
        self.bias = np.array(data['bias'])
        self.C = np.array(data['C'])
        
        # Covariance for Noise Gating
        cov = np.array(data.get('covariance', np.eye(6)))
        self.noise_std = np.sqrt(np.diag(cov))
        self.noise_gate = 3.0 * self.noise_std # 3*Sigma
        
        r, p, y = data.get('imu_rot', [0,0,0])
        self.R_align = R.from_euler('xyz', [r,p,y], degrees=False).as_matrix()

    def imu_cb(self, msg):
        self.accel_raw = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        self.gyro_raw = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        
        q_mad = self.madgwick.update(self.gyro_raw, self.accel_raw)
        
        # SLERP FUSION
        if self.q_aruco is not None:
            # Check dot product for shortest path
            if np.dot(q_mad, self.q_aruco) < 0:
                self.q_aruco = -self.q_aruco
            
            # Weighted Slerp
            q_fused_wxyz = perform_slerp(q_mad, self.q_aruco, self.aruco_weight)
            self.madgwick.q = q_fused_wxyz
            self.q_final = q_fused_wxyz
        else:
            self.q_final = q_mad

    def aruco_cb(self, msg):
        q = msg.pose.orientation
        self.q_aruco = np.array([q.w, q.x, q.y, q.z])
        self.last_aruco_time = rospy.get_time()

    def ft_cb(self, msg):
        self.w_raw = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                               msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

    def update(self):
        if self.w_raw is None or self.accel_raw is None or self.gyro_raw is None:
            return

        # Check ArUco freshness
        aruco_active = False
        if self.q_aruco is not None and self.last_aruco_time is not None:
            aruco_age    = rospy.get_time() - self.last_aruco_time
            aruco_active = (aruco_age < 1.0)

        if not aruco_active:
            if not self.aruco_lost:
                self.aruco_lost = True
                print("\n[WATCHDOG] ArUco lost — stopping data collection.")
                self.print_stats()
            return

        self.aruco_lost = False

        # Align raw IMU measurements to sensor frame
        accel_aligned = self.R_align @ self.accel_raw
        gyro_aligned  = self.R_align @ self.gyro_raw

        # Extract gravity vector from fused orientation (Madgwick + ArUco SLERP)
        r_fused  = R.from_quat([self.q_final[1], self.q_final[2],
                                self.q_final[3], self.q_final[0]])
        g_world  = np.array([0.0, 0.0, -9.81])
        g_sensor = self.R_align @ r_fused.inv().apply(g_world)

        # Separate linear acceleration from gravitational component
        a_linear = accel_aligned - g_sensor

        # Inertial force from linear acceleration only
        F_base = -self.m * a_linear

        # Gravitational force from fused orientation
        F_gravity = -self.m * g_sensor

        # Centrifugal force
        wx_r          = np.cross(gyro_aligned, self.r)
        F_centrifugal = self.m * np.cross(gyro_aligned, wx_r)

        # Total physics force and torque in sensor frame
        F_physics = F_base + F_gravity - F_centrifugal
        T_physics = np.cross(self.r, F_physics)
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
        
    def print_stats(self):
        if not self.fz_errors:
            print("\n[Stats] No data collected.")
            return

        fz_data = np.array(self.fz_errors)
        tz_data = np.array(self.tz_errors)

        print("\n" + "="*70)
        print("FINAL STATISTICS - Z-AXIS COMPENSATION (Madgwick + Aruco)")
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
    FTCompensatorMadgwickArUco()