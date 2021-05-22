import numpy as np
import pandas as pd

from main_project.constants import gyro_static
from main_project.gps_transform import wgs_to_enu, get_enu_reference
from main_project.movement_models import ctrv


def gen_test_control_straight(size: int):
    data = []
    for i in range(size):
        if i < size / 2:
            data.append(np.array([5, 0]))
        if i == size / 2:
            data.append(np.array([5, np.radians(4)]))
        else:
            data.append(np.array([5, 0]))
    return data


def test_ctrv_fake_data(start, inp):
    states_gps = []
    states_enu = []
    current_state = np.zeros(4)
    last_predict = 0
    m = np.zeros(4)
    # Transform GPS to ENU coordinates
    gps_radians = np.radians(start)
    reference_ecef, reference_matrix = get_enu_reference(gps_radians[0], gps_radians[1])
    gps_enu = wgs_to_enu(lat=gps_radians[0], lon=gps_radians[1], alt=0,
                         ecef0=reference_ecef, ref_matrix=reference_matrix)
    m[:2] = gps_enu[:2]
    current_state[:] = m[:]
    current_state[2] = np.sin(0)
    current_state[3] = np.cos(0)
    for i in inp:
        current_state = ctrv.transit_ctrv(current_state, i, 1)
        states_enu.append(current_state)
        states_gps.append(np.degrees(ctrv.state_to_latlon(current_state, reference_ecef, reference_matrix.T)))
    return states_gps


def test_ctrv_real_data(inputs: np.ndarray):
    states = []
    states_enu = []
    current_state = np.zeros(4)
    last_predict = 0.
    reference_ecef = np.zeros(3)
    reference_matrix_T = np.zeros((3, 3))
    current_measurement = np.zeros(4)
    initialized = False
    # Control vector: [velocity, yaw speed]
    current_control = np.zeros(2)
    for input in inputs:
        # if sensor (speed) data is available
        if not (np.isnan(input[11]) or np.isnan(input[12])):
            current_control[0] = input[13]
        # if IMU (yaw rate) data is available
        if not np.isnan(input[10]):
            current_control[1] = input[10] - gyro_static
        # if GPS data is available
        if not np.isnan(input[2]):
            if not initialized:
                last_predict = input[0]
                # Transform GPS to ENU coordinates
                gps_radians = np.radians(input[1:3])
                # Setup initial state and ENU reference
                reference_ecef, reference_matrix = get_enu_reference(gps_radians[0], gps_radians[1])
                gps_enu = wgs_to_enu(lat=gps_radians[0], lon=gps_radians[1], alt=0,
                                     ecef0=reference_ecef, ref_matrix=reference_matrix)
                current_measurement[:2] = gps_enu[:2]
                gps_orient = np.radians(90-input[6])
                current_measurement[2] = np.sin(gps_orient)
                current_measurement[3] = np.cos(gps_orient)
                reference_matrix_T = reference_matrix.T
                current_state = current_measurement
                initialized = True
            current_state = ctrv.transit_ctrv(current_state, current_control, (input[0] - last_predict) * 1e-6)
            last_predict = input[0]
            states_enu.append(current_state)
            states_gps = np.degrees(ctrv.state_to_latlon(current_state, reference_ecef, reference_matrix_T))
            states.append(states_gps)
    return states

def test_ctrv_real_data_new(inputs: np.ndarray):
    states = []
    states_enu = []
    current_state = np.zeros(4)
    last_predict = 0.
    reference_ecef = np.zeros(3)
    reference_matrix_T = np.zeros((3, 3))
    current_measurement = np.zeros(4)
    initialized = False
    # Control vector: [velocity, yaw speed]
    current_control = np.zeros(2)
    for input in inputs:
        # if sensor (speed) data is available
        if not (np.isnan(input[11]) or np.isnan(input[12])):
            current_control[0] = input[13]
        # if IMU (yaw rate) data is available
        if not np.isnan(input[10]):
            current_control[1] = input[10] - gyro_static
        # if GPS data is available
        if not np.isnan(input[2]):
            if not initialized:
                last_predict = input[0]
                # Transform GPS to ENU coordinates
                gps_radians = np.radians(input[1:3])
                # Setup initial state and ENU reference
                reference_ecef, reference_matrix = get_enu_reference(gps_radians[0], gps_radians[1])
                gps_enu = wgs_to_enu(lat=gps_radians[0], lon=gps_radians[1], alt=0,
                                     ecef0=reference_ecef, ref_matrix=reference_matrix)
                current_measurement[:2] = gps_enu[:2]
                gps_orient = np.radians(90-input[6])
                current_measurement[2] = np.sin(gps_orient)
                current_measurement[3] = np.cos(gps_orient)
                reference_matrix_T = reference_matrix.T
                current_state = current_measurement
                initialized = True
            current_state = ctrv.transit_ctrv(current_state, current_control, (input[0] - last_predict) * 1e-6)
            last_predict = input[0]
            states_enu.append(current_state)
            states_gps = np.degrees(ctrv.state_to_latlon(current_state, reference_ecef, reference_matrix_T))
            states.append(states_gps)
    return states


if __name__ == '__main__':
    data = pd.read_csv('main_project/inputs/total.csv')
    data = data.to_numpy()[2588:]
    reslist = test_ctrv_real_data(data)
    res = pd.DataFrame(reslist, columns=['latitude', 'longitude', 'alt'])
    res.to_csv('results_ctrv_nofilter.txt')
