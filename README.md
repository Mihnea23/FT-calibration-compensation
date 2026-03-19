# FT-calibration-compensation

A ROS-based framework for offline calibration and real-time dynamic compensation 
of wrist-mounted force/torque sensors on robotic end-effectors.

## Overview

This repository implements a two-phase pipeline:

- **Phase 1 — Offline Calibration:** Estimates sensor bias, noise covariance, 
  effective mass, center of mass, IMU alignment, and a 6×6 calibration matrix 
  from static and dynamic ROS bag recordings.

- **Phase 2 — Real-Time Compensation:** A ROS node that fuses IMU measurements 
  with ArUco-based visual pose estimation to subtract gravity and inertial 
  wrenches from raw F/T sensor readings.
  
## Reference

Developed as part of a Bachelor's thesis at TU Wien, Automation and Control 
Institute (ACIN), February 2026.

## Sources
- [Madgwick Filter AHRS_Documentation](https://ahrs.readthedocs.io/en/latest/filters/madgwick.html)
- [Force Torque Tools (KTH)](https://github.com/kth-ros-pkg/force_torque_tools)
- [Inertial Wrench Compensation Demo (Bota Systems)](https://gitlab.com/botasys/legacy/inertial_wrench_compensation_demo)
- [MSVC Inertial Wrench Compensation Library (Bota Systems)](https://gitlab.com/botasys/legacy/msvc_inertial_wrench_copmesation_library)
- [Payload Utils Python (Bota Systems)](https://code.botasys.com/en/gen_a/layer1/payload_utils/payload_utils_py.html)
