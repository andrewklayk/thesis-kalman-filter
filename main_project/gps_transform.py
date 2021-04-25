import numpy as np
import pandas as pd

# earth semi-major axis (equatorial radius)
a: float = 6378137
# earth semi-minor axis (polar radius)
b: float = 6356752.3142
# earth first numerical eccentricity
e2: float = 1 - (b / a) ** 2


def create_ref_matrix(sl_0, cl_0, sp_0, cp_0):
    return np.array(
        [[-sl_0, cl_0, 0],
         [-cl_0 * sp_0, -sl_0 * sp_0, cp_0],
         [cl_0 * cp_0, sl_0 * cp_0, sp_0]]
    )


def get_enu_reference(lat_0, lon_0, h_0=0):
    sp_0 = np.sin(lat_0)
    cp_0 = np.cos(lat_0)
    sl_0 = np.sin(lon_0)
    cl_0 = np.cos(lon_0)
    return wgs_to_ecef(h_0, sp_0, cp_0, sl_0, cl_0), create_ref_matrix(sl_0, cl_0, sp_0, cp_0)


# TODO: more precise (ref: muzhik tak skazal) or fast (ref: straya) conversions may exist

def wgs_to_enu(lat: float, lon: float, alt: float, ecef0: np.ndarray, ref_matrix: np.ndarray, ) -> np.ndarray((3,)):
    """Convert WGS84 coordinates of a point to ENU coordinates with a specified point of reference (0, 0, 0).

    :param lat: Latitude of point IN RADIANS
    :param lon: Longitude of point IN RADIANS
    :param alt: Altitude of point (NOT IN RADIANS XDD)))00))0)
    :param ecef0: Reference point in ECEF coordinates (easy to precalculate)
    :param ref_matrix: Matrix for ECEF->ENU transformation (easy to precalculate)
    :return: ENU coordinates of point in an ndarray [x,y,z]
    :rtype: np.ndarray
    """
    sp = np.sin(lat)
    cp = np.cos(lat)
    sl = np.sin(lon)
    cl = np.cos(lon)
    ecef1 = wgs_to_ecef(alt, sp, cp, sl, cl)
    return ecef_to_enu(ecef0, ecef1, ref_matrix)


def wgs_to_ecef_raw(phi: float, lam: float, h: float) -> np.ndarray((3,)):
    """Convert WGS84 coordinates to ECEF coordinates.

    :param phi: Latitude (IN RADIANS)
    :param lam: Longitude (IN RADIANS)
    :param h: height
    :return: ECEF coordinates in a ndarray [X, Y, Z]
    :rtype: np.ndarray
    """
    sp = np.sin(phi)
    cp = np.cos(phi)
    sl = np.sin(lam)
    cl = np.cos(lam)
    N = a / (np.sqrt(1 - e2 * (sp ** 2)))
    temp = (N + h) * cp
    x = temp * cl
    y = temp * sl
    z = (((b / a) ** 2) * N + h) * sp
    return np.array([x, y, z])


def wgs_to_ecef(h: float, sp: float, cp: float, sl: float, cl: float) -> np.ndarray((3,)):
    """Convert WGS84 coordinates to ECEF coordinates. DON'T FORGET TO CONVERT TO RADIANS!

    :param h: height
    :param sp: sin latitude
    :param cp: cos latitude
    :param sl: sin longitude
    :param cl: cos longitude
    :return: ECEF coordinates in a ndarray: [X, Y, Z]
    :rtype: np.ndarray
    """
    N = a / (np.sqrt(1 - e2 * (sp ** 2)))
    temp = (N + h) * cp
    x = temp * cl
    y = temp * sl
    z = (((b / a) ** 2) * N + h) * sp
    return np.array([x, y, z])


def ecef_to_enu(ecef0: np.ndarray, ecef1: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray((3,)):
    enu = transform_matrix @ (ecef1 - ecef0)
    return enu


# Source: ESA
def enu_to_ecef(ecef0: np.ndarray, enu: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray((3,)):
    ecef = transform_matrix @ enu + ecef0
    return ecef


# def ecef_to_wgs_wiki(x: float, y: float, z: float = 0) -> np.ndarray((3,)):


# Source: eceftowgs.pdf
def ecef_to_wgs(x: float, y: float, z: float = 0) -> np.ndarray((3,)):
    w2 = x ** 2 + y ** 2
    l = e2 / 2
    l2 = l ** 2
    m = w2 / (a ** 2)
    n = ((1 - e2) * z / b) ** 2
    p = (m + n - 4 * l2) / 6
    G = m * n * l2
    H = 2 * p ** 3 + G
    # if H < Hmin, abort
    C = np.cbrt(((H + G + 2 * np.sqrt(H * G)) / 2))
    i = -(2 * l2 + m + n) / 2
    P = p ** 2
    beta = i / 3 - C - P / C
    k = l2 * (l2 - m - n)
    t = np.sqrt(np.sqrt(beta ** 2 - k) - (beta + i) / 2) - np.sign(m - n) * np.sqrt(abs((beta - i) / 2))
    F = t ** 4 + 2 * i * t ** 2 + 2 * l * (m - n) * t + k
    dFdt = 4 * t ** 3 + 4 * i * t + 2 * l * (m - n)
    deltat = -F / dFdt
    u = t + deltat + l
    v = t + deltat - l
    w = np.sqrt(w2)
    phi = np.arctan2(z * u, w * v)
    deltaw = w * (1 - 1 / u)
    deltaz = z * (1 - (1 - e2) / v)
    h = np.sign(u - 1) * np.sqrt(deltaw ** 2 + deltaz ** 2)
    lam = np.arctan2(y, x)
    return np.array([phi, lam, h])


def enu_to_wgs(ecef0: np.ndarray, enu: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray((3,)):
    ecef = enu_to_ecef(ecef0, enu, transform_matrix)
    return ecef_to_wgs(ecef[0], ecef[1], ecef[2])


def gen_test_enu(size: int):
    data = []
    for i in range(size):
        data.append(np.array([i, 2*i, 0]))
    return data


if __name__ == '__main__':
    test_wgs = pd.read_csv('inputs/gps.txt', sep=" ")
    test_wgs = test_wgs[['LAT', 'LON']].to_numpy()

    test_enu = gen_test_enu(size=1000)

    test_wgstoecef = []
    i = 0
    for wgs in test_wgs:
        gpsr = np.radians(wgs)
        test_wgstoecef.append(wgs_to_ecef(0, np.sin(gpsr[0]), np.cos(gpsr[0]), np.sin(gpsr[1]), np.cos(gpsr[1])))
        i += 1

    test_wgstoenu = []
    gps_radians = np.radians(test_wgs[0])
    reference_ecef, reference_matrix = get_enu_reference(gps_radians[0], gps_radians[1])

    test_enu_to_wgs_fake = []
    for t in test_enu:
        test_enu_to_wgs_fake.append(np.degrees(enu_to_wgs(reference_ecef, t, reference_matrix.T)))
    pd.DataFrame(test_enu_to_wgs_fake, columns=['latitude', 'longitude', 'alt']).to_csv('test_enu.txt')

    for wgs in test_wgs:
        gps_radians = np.radians(wgs)
        gps_enu = wgs_to_enu(gps_radians[0], gps_radians[1], alt=0,
                             ecef0=reference_ecef, ref_matrix=reference_matrix)
        test_wgstoenu.append(gps_enu)

    test_enutowgs = []
    for enu in test_wgstoenu:
        test_enutowgs.append(np.degrees(enu_to_wgs(reference_ecef, enu, reference_matrix.T)))
    pd.DataFrame(test_enutowgs, columns=['latitude', 'longitude', 'alt']).to_csv('test_enu_to_wgs.txt')
    print(wgs_to_ecef_raw(np.radians(40).item(), np.radians(319).item(), 149.2))
    # print(ecef_to_wgs(10000, 15000, 150000))
