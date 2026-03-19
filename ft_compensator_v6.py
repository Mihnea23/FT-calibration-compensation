#!/usr/bin/env python3

import rospy
import yaml
import numpy as np
from geometry_msgs.msg import WrenchStamped, Vector3, PoseStamped
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation as R

# ---------------- Main Node ----------------

class FTCompensatorArUcoOnly(object):
    def __init__(self):
        rospy.init_node('ft_compensator_aruco_only', anonymous=False)

        # Config
        self.calib_file = rospy.get_param('~calib_yaml', '~/calib.yaml')
        self.load_calibration(self.calib_file)
        self.debug = True
        
        # State
        self.q_aruco = None
        self.accel_raw = None
        self.gyro_raw = None
        self.w_raw = None
        self.last_aruco_time = None
        self.aruco_lost = False
        
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
        self.noise_gate = 3.0 * self.noise_std
        
        r_imu, p_imu, y_imu = data.get('imu_rot', [0,0,0])
        self.R_align = R.from_euler('xyz', [r_imu, p_imu, y_imu], degrees=False).as_matrix()

    def imu_cb(self, msg):
        """Store raw IMU data without any filtering"""
        self.accel_raw = np.array([msg.linear_acceleration.x, 
                                   msg.linear_acceleration.y, 
                                   msg.linear_acceleration.z])
        self.gyro_raw = np.array([msg.angular_velocity.x, 
                                  msg.angular_velocity.y, 
                                  msg.angular_velocity.z])

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

        # Extract gravity from ArUco orientation
        r_aruco  = R.from_quat([self.q_aruco[1], self.q_aruco[2],
                                self.q_aruco[3], self.q_aruco[0]])
        g_world  = np.array([0.0, 0.0, -9.81])
        g_sensor = self.R_align @ r_aruco.inv().apply(g_world)

        # Separate linear acceleration from gravitational component
        a_linear = accel_aligned - g_sensor

        # Inertial and gravitational forces
        F_base    = -self.m * a_linear
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

        # Extract rotation from ArUco orientation
        r_fused = R.from_quat([self.q_aruco[1], self.q_aruco[2],
                               self.q_aruco[3], self.q_aruco[0]])

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

        # Publish compensated wrench in world frame
        msg = WrenchStamped()
        msg.header.stamp    = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.wrench.force    = Vector3(*W_comp_world[0:3])
        msg.wrench.torque   = Vector3(*W_comp_world[3:6])
        self.pub.publish(msg)

    def print_stats(self):
        """Print final Z-axis statistics"""
        if not self.fz_errors:
            print("\n[Stats] No data collected.")
            return

        fz_data = np.array(self.fz_errors)
        tz_data = np.array(self.tz_errors)

        print("\n" + "="*70)
        print("FINAL STATISTICS - Z-AXIS COMPENSATION (ArUco only)")
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
    FTCompensatorArUcoOnly()