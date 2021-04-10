import numpy as np

# CTRV State: [x, y, sin(gyro_angle), cos(gyro_angle)]
# CTRV Control: velocity (sensors); turn_rate (imu)
# CTRV Measurement: x, y (gps)


# GPS measurement function
def measmt_func_gps(state: np.ndarray):
    return state[:2]


# CTRV state transition function
# Parameters:
#   state: current state [x, y, sin(angle), cos(angle)];
#   u: control vector [velocity, yaw rate]
#   delta_t: delta time
# Returns: new state
def transit_ctrv(state: np.ndarray, u: np.ndarray, delta_t: float) -> np.ndarray:
    v = u[0]
    w = u[1]
    new_state = state
    if w == 0:
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
            vdivw * sin_new_angle - vdivw * state[2],
            -vdivw * cos_new_angle + vdivw * state[3],
            sin_new_angle,
            cos_new_angle
        ]
    return new_state
