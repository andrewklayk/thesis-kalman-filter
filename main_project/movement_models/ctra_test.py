import numpy as np
import pandas as pd

from main_project.constants import gyro_static
from main_project.gps_transform import wgs_to_enu, get_enu_reference
from main_project.movement_models import ctra


def measmt_func_spd(state: np.ndarray):
    return state[3]
def process_speed_ctra(state):
    raise NotImplementedError

# 0: Timestamp, 1: Lat, 2: Lon, 3: Pdop, 4: velocity, 5: altitude, 6: orientation,
# 7: accelX, 8: accelY, 9: accelZ, 10:gyroX, 11: wsrr, 12: wsrl, 13: car speed
def run_CTRA(inputs: np.ndarray):
    states = []
    current_state = np.zeros(5)
    current_cov = np.identity(5) * 1e-9
    last_timestamp = 0.
    reference_ecef = np.zeros(3)
    reference_matrix = np.zeros((3, 3))
    reference_matrix_T = np.zeros((3, 3))
    initialized = False
    # Control vector: [velocity, yaw speed]
    current_control = np.zeros(2)
    current_measurement = np.zeros(5)
    for input in inputs:
        # if sensor (speed) data is available
        if initialized and not (np.isnan(input[11]) or np.isnan(input[12])):
            current_measurement[2] = input[13]
            # Predict with last velocity and yaw rate, update with last gps
            current_state = ctra.transit_ctra(current_state, current_control, (input[0] - last_timestamp) * 1e-6)

            last_timestamp = input[0]
            states.append(np.degrees(ctra.state_to_latlon(current_state, reference_ecef, reference_matrix_T)))
        # if IMU (yaw rate) data is available
        if not np.isnan(input[10]):
            current_control[0] = -input[8]#/current_state[3] + input[9]/current_state[4]
            current_control[1] = input[10] - gyro_static
        # if GPS data is available
        if not np.isnan(input[2]):
            # Transform GPS to ENU coordinates
            gps_radians = np.radians(input[1:3])
            gps_enu = wgs_to_enu(lat=gps_radians[0], lon=gps_radians[1], alt=0,
                                 ecef0=reference_ecef, ref_matrix=reference_matrix)
            current_measurement[:2] = gps_enu[:2]
            current_measurement[2] = input[4]
            gps_orient = np.radians(90 - input[6])
            current_measurement[3] = np.sin(gps_orient)
            current_measurement[4] = np.cos(gps_orient)
            # Setup initial state and ENU reference
            if not initialized:
                reference_ecef, reference_matrix = get_enu_reference(gps_radians[0], gps_radians[1])
                reference_matrix_T = reference_matrix.T
                current_state = current_measurement
                last_timestamp = input[0]
                initialized = True
            # Predict with last velocity and yaw rate, update with last gps
            current_state = ctra.transit_ctra(current_state, current_control, (input[0] - last_timestamp)*1e-6)
            last_timestamp = input[0]
            states.append(np.degrees(ctra.state_to_latlon(current_state, reference_ecef, reference_matrix_T)))
    return states


if __name__ == '__main__':
    data = pd.read_csv('main_project/inputs/total.csv')
    data = data.to_numpy()[2588:]
    # imu = data[['systemTimestamp', 'gyroX']]
    reslist = run_CTRA(data)
    res = pd.DataFrame(reslist, columns=['latitude', 'longitude', 'alt'])
    res.to_csv('results_ctra.txt')