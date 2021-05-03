import numpy as np
from main_project.gps_transform import enu_to_wgs


# CTRV State: [x, y, sin(gyro_angle), cos(gyro_angle)]
# CTRV Control: velocity (sensors); turn_rate (imu)
# CTRV Measurement: x, y (gps)


# GPS measurement function
def measmt_func_gps(state: np.ndarray):
    return state


# Speed measurement function
def measmt_func_spd(state: np.ndarray):
    return state[2]


def transit_ctrv(state: np.ndarray, u: np.ndarray, delta_t: float) -> np.ndarray:
    """State transition function for CTRV model.


    :param state: current state [x, y, sin(angle), cos(angle)];
    :param u: control vector [velocity, yaw rate];
    :param delta_t: delta time
    :returns: new state [x, y, sin(angle), cos(angle)]
    """
    v = u[0]
    w = -u[1]
    new_state = np.empty_like(state)
    new_state[:] = state
    if abs(w) < 1e-6:
        new_state += [
            v * state[3] * delta_t,
            v * state[2] * delta_t,
            0,
            0
        ]
    else:
        vdivw = v / w
        new_angle = np.arctan2(state[2], state[3]) + w * delta_t
        sin_new_angle = np.sin(new_angle)
        cos_new_angle = np.cos(new_angle)
        new_state += [
            vdivw * (sin_new_angle - state[2]),
            -vdivw * (cos_new_angle - state[3]),
            0,
            0
        ]
        new_state[2] = sin_new_angle
        new_state[3] = cos_new_angle
    return new_state


def state_to_latlon(state: np.ndarray, reference: np.ndarray, reference_matrix):
    enu = np.zeros(3)
    enu[0] = state[0]
    enu[1] = state[1]
    return enu_to_wgs(reference, enu, reference_matrix)


def pretty_output(state: np.ndarray):
    return "{0:10.4f}, {1:10.4f}, {2}".format(state[0], state[1], np.arctan2(state[2], state[3]))
