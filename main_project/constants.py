import numpy as np
from gps_transform import wgs_to_enu, wgs_to_ecef

# Wheel radius:
wheel_r = 0.376
gyro_static = 0.032925

# Point of reference for the ENU system
lat0 = 39
lon0 = -132
h0 = 0
sp_0 = np.sin(lat0)
cp_0 = np.cos(lat0)
sl_0 = np.sin(lon0)
cl_0 = np.cos(lon0)

ecef_0 = wgs_to_ecef(h0, sp_0, cp_0, sl_0, cl_0)

m = np.array(
    [[-sl_0, cl_0, 0],
     [-cl_0 * sp_0, -sl_0 * sp_0, cp_0],
     [cl_0 * cp_0, sl_0 * cp_0, sp_0]]
)