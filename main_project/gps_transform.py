import numpy as np

# earth semi-major axis (equatorial radius)
a: float = 6378137
# earth semi-minor axis (polar radius)
b: float = 6356752.3142
# earth first numerical eccentricity
e2: float = 1 - (b / a) ** 2


# TODO: more precise (ref: muzhik tak skazal) or fast (ref: straya) convertions may exist;
# they are to be implemented later

def wgs_to_enu(lat: float, lon: float, alt: float, ecef0: np.ndarray,  ref_matrix: np.ndarray,) -> np.ndarray((3,)):
    sp = np.sin(lat)
    cp = np.cos(lat)
    sl = np.sin(lon)
    cl = np.cos(lon)
    ecef1 = wgs_to_ecef(alt, sp, cp, sl, cl)
    return ecef_to_enu(ecef0, ecef1, ref_matrix)


def wgs_to_ecef(h: float, sp: float, cp: float, sl: float, cl: float) -> np.ndarray((3,)):
    N = a / (np.sqrt(1 - e2 * (sp ** 2)))
    temp = (N + h) * cp
    x = temp * cl
    y = temp * sl
    z = (((b / a) ** 2) * N + h) * sp
    return np.array([x, y, z])


def ecef_to_enu(ecef0: np.ndarray, ecef1: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray((3,)):
    enu = transform_matrix @ (ecef1 - ecef0)
    return enu


# def ecef_to_enu(ecef0: np.ndarray, ecef1: np.ndarray,
#                 sp: float, cp: float, sl: float, cl: float) -> np.ndarray((3,)):
#     m = np.array(
#         [[-sl, cl, 0],
#          [-cl * sp, -sl * sp, cp],
#          [cl * cp, sl * cp, sp]]
#     )
#     enu = m @ (ecef1 - ecef0)
#     return enu


def enu_to_ecef():
    raise NotImplementedError


def ecef_to_wgs():
    raise NotImplementedError
