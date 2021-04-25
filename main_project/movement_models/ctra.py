import numpy as np
from main_project.gps_transform import enu_to_wgs

# CTRA State: [x, y, v, sin(gyro_angle), cos(gyro_angle)]
# CTRA Control: imu
# CTRA Measurement: gps, sensors


# GPS measurement function
def measmt_func_gps(state: np.ndarray):
    return state

# Speed measurement function
def measmt_func_sensor(state: np.ndarray):
    return state[2]

def transit_ctra(state: np.ndarray, u: np.ndarray, delta_t: float) -> np.ndarray:
    """State transition function for CTRV model.


    :param state: current state [x, y, v, sin(angle), cos(angle)];
    :param u: control vector [acceleration, yaw rate];
    :param delta_t: delta time
    :returns: new state [x, y, v, sin(angle), cos(angle)]
    """
    a = u[0]
    w = -u[1]
    new_state = np.empty_like(state)
    new_state[:] = state
    new_angle = np.arctan2(state[2], state[3]) + w * delta_t
    sin_new_angle = np.sin(new_angle)
    cos_new_angle = np.cos(new_angle)
    if abs(w) < 1e-6:
        new_state += [
            ((state[2]+a*w*delta_t)*sin_new_angle + a*cos_new_angle - state[2]*w*state[3] - a*state[4]),
            (-(state[2]+a*w*delta_t)*cos_new_angle + a*sin_new_angle + state[2]*w*state[4] - a*state[3]),
            a*delta_t,
            0,
            0
        ]
    else:
        new_state += [
            (1/(w**2)) * (w*(state[2]+a*delta_t)*sin_new_angle + a*cos_new_angle - state[2]*w*state[3] - a*state[4]),
            (1/(w**2)) * (-w*(state[2]+a*delta_t)*cos_new_angle + a*sin_new_angle + state[2]*w*state[4] - a*state[3]),
            a*delta_t,
            0,
            0
        ]
    new_state[3] = sin_new_angle
    new_state[4] = cos_new_angle
    return new_state

def state_to_latlon(state: np.ndarray, reference: np.ndarray, reference_matrix):
    enu = np.zeros(3)
    enu[0] = state[0]
    enu[1] = state[1]
    return enu_to_wgs(reference, enu, reference_matrix)

def pretty_output(state: np.ndarray):
    return "{0:10.4f}, {1:10.4f}, {2}, {3}".format(state[0], state[1], state[3], np.arctan2(state[3], state[4]))
