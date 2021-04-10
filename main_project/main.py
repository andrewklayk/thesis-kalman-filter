import numpy as np
import pandas as pd
import pymap3d as pm
import kf_python.unscented_kf as UKF
import ctrv
from constants import wheel_r, gyro_static

from gps_transform import wgs_to_enu, wgs_to_ecef


# CTRV State: [x, y, sin(gyro_angle), cos(gyro_angle)]
# CTRV Control: velocity (sensors); turn_rate (imu)
# CTRV Measurement: x, y (gps)

# CTRA State: [x, y, v, sin(gyro_angle), cos(gyro_angle)]
# CTRA Control: acceleration (imu); turn_rate (imu)
# CTRA Measurement: x, y (gps); velocity (sensors)

# Speed measurement function (only for CTRA, in CTRV speed is one of the controls)
def measmt_func_spd(state: np.ndarray):
    return state[3]


def process_speed_ctra(state):
    raise NotImplementedError


# 0: Timestamp, 1: Lat, 2: Lon, 3: Pdop, 4: velocity, 5: altitude, 6: orientation,
# 7: accelX, 8: accelY, 9: accelZ, 10:gyroX, 11: wsrr, 12: wsrl
def run_ctrv(inputs: np.ndarray):
    current_state = np.array([0, 0, 0, 0])
    current_cov = np.identity(4)
    last_imu_timestamp = 0.
    # Control vector: [velocity, yaw speed]
    current_control = np.zeros(2)
    ukf = UKF.UnscentedKF(process_speed_ctra, measmt_func_spd, np.identity(4), np.identity(2), 4, alpha=np.sqrt(3),
                          beta=2, kappa=1)
    for input in inputs:
        # if sensor (speed) data is available
        if not (np.isnan(input[11]) or np.isnan(input[12])):
            current_control[0] = (((input[11] + input[12]) / 2) - gyro_static) * wheel_r
        # if IMU (yaw rate) data is available
        if not np.isnan(input[10]):
            current_control[1] = input[10]
            last_imu_timestamp = input[0]
        # if GPS data is available
        if not np.isnan(input[2]):
            # Predict with last velocity and yaw rate, update with last gps
            current_state, current_control = ukf.propagate(current_state, current_cov,
                                                           current_control, ctrv.transit_ctrv,
                                                           input[1:3], ctrv.measmt_func_gps,
                                                           input[0] - last_imu_timestamp)


# 0: Timestamp, 1: Lat, 2: Lon, 3: Pdop, 4: velocity, 5: altitude, 6: orientation,
# 7: accelX, 8: accelY, 9: accelZ, 10:gyroX, 11: wsrr, 12: wsrl
# def run(inputs: np.array):
#     for input in inputs:
#         if not np.isnan(input[10]):
#             last_hdg = input[10]
#             last_hdg_time = input[0]
#         if not np.isnan(input['LAT']):
#             measmt_func_gps()
#         if not np.isnan(input['rearWheelSpeed']):
#             raise NotImplementedError
#             # process_speed()


if __name__ == '__main__':
    data = pd.read_csv('res/total.csv')
    imu = data[['systemTimestamp', 'gyroX']]
    run(data)
