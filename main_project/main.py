import numpy as np
import pandas as pd
import pymap3d as pm
from gps_transform import wgs_to_enu


def process_gps():
    # convert GPS readings in WGS84 to ENU
    gps_enu = wgs_to_enu()
    # do kalman filter stuff
    raise NotImplementedError


def process_speed():
    raise NotImplementedError


# 0: Timestamp, 1: Lat, 2: Lon, 3: Pdop, 4: velocity, 5: altitude, 6: orientation,
# 7: accelX, 8: accelY, 9: accelZ, 10:gyroX, 11: wsrr, 12: wsrl
def run(inputs: np.array):
    last_hdg: float = 0
    last_hdg_time: float = 0
    for input in inputs:
        if not np.isnan(input[10]):
            last_hdg = input[10]
            last_hdg_time = input[0]
        if not np.isnan(input['LAT']):
            process_gps()
        if not np.isnan(input['rearWheelSpeed']):
            process_speed()


if __name__ == '__main__':
    data = pd.read_csv('res/total.csv')
    imu = data[['systemTimestamp', 'gyroX']]
    run(data)
