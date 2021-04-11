import numpy as np
from gps_transform import wgs_to_enu, wgs_to_ecef

# Wheel radius:
wheel_r = 0.376
gyro_static = 0.032925

# Point of reference for the ENU system
lat0 = 39
lon0 = -132
h0 = 0


# sp_0 = np.sin(lat0)
# cp_0 = np.cos(lat0)
# sl_0 = np.sin(lon0)
# cl_0 = np.cos(lon0)

def get_enu_reference(lat_0, lon_0, h_0=0):
    sp_0 = np.sin(lat_0)
    cp_0 = np.cos(lat_0)
    sl_0 = np.sin(lon_0)
    cl_0 = np.cos(lon_0)
    return wgs_to_ecef(h_0, sp_0, cp_0, sl_0, cl_0), create_ref_matrix(sl_0, cl_0, sp_0, cp_0)


def create_ref_matrix(sl_0, cl_0, sp_0, cp_0):
    return np.array(
        [[-sl_0, cl_0, 0],
         [-cl_0 * sp_0, -sl_0 * sp_0, cp_0],
         [cl_0 * cp_0, sl_0 * cp_0, sp_0]]
    )
