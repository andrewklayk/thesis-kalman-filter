import numpy as np
import pandas as pd
import kf_python.unscented_kf as UKF
from main_project.movement_models import ctrv
from constants import wheel_r, gyro_static, get_enu_reference

from gps_transform import wgs_to_enu


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
    states = []
    current_state = np.array([0, 0, 0, 0])
    current_cov = np.identity(4) * 1e-4
    last_imu_timestamp = 0.
    reference_ecef = np.zeros(3)
    reference_matrix = np.zeros((3, 3))
    reference_matrix_T = np.zeros((3,3))
    initialized = False
    # Control vector: [velocity, yaw speed]
    current_control = np.zeros(2)
    current_measurement = np.zeros(4)
    ukf = UKF.UnscentedKF(process_speed_ctra, measmt_func_spd, np.identity(4) * 100, np.identity(4) * 1e-6, 4,
                          alpha=np.sqrt(3), beta=2, kappa=1)
    for input in inputs:
        # if sensor (speed) data is available
        if not (np.isnan(input[11]) or np.isnan(input[12])):
            current_control[0] = input[13] #(((input[11] + input[12]) / 2) - gyro_static) * wheel_r
        # if IMU (yaw rate) data is available
        if not np.isnan(input[10]):
            current_control[1] = input[10]
            last_imu_timestamp = input[0]
        # if GPS data is available
        if not np.isnan(input[2]):
            # Transform GPS to ENU coordinates
            gps_radians = np.radians(input[1:3])
            gps_enu = wgs_to_enu(lat=gps_radians[0], lon=gps_radians[1], alt=0,
                                 ecef0=reference_ecef, ref_matrix=reference_matrix)
            current_measurement[:2] = gps_enu[:2]
            gps_orient = np.radians(90-input[6])
            current_measurement[2] = np.sin(gps_orient)
            current_measurement[3] = np.cos(gps_orient)
            # Setup initial state and ENU reference
            if not initialized:
                reference_ecef, reference_matrix = get_enu_reference(gps_radians[0], gps_radians[1])
                reference_matrix_T = reference_matrix.T
                current_state = current_measurement
                initialized = True
            # Predict with last velocity and yaw rate, update with last gps
            current_state, current_cov = ukf.propagate(current_state, current_cov,
                                                       current_control, ctrv.transit_ctrv,
                                                       current_measurement, ctrv.measmt_func_gps,
                                                       (input[0] - last_imu_timestamp) * 1e-6)
            states.append(np.degrees(ctrv.state_to_latlon(current_state, reference_ecef, reference_matrix_T)))
    return states


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
    data = pd.read_csv('inputs/total.csv')
    data = data.to_numpy()
    # imu = data[['systemTimestamp', 'gyroX']]
    reslist = run_ctrv(data)
    res = pd.DataFrame(reslist, columns=['latitude', 'longitude', 'alt'])
    res.to_csv('results_ctrv.txt')
