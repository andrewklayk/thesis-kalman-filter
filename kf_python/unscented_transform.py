import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, cholesky


def calculate_lambda(L: int, alpha: float = 1, k: float = 0):
    return (alpha ** 2) * (L + k) - L


def calc_sigma_points(x_mean: np.array, x_cov: np.array, _lambda: float, a: float, b: float):
    dim: int = x_mean.shape[0]
    matrix = (dim + _lambda) * x_cov
    # u, s, vh = np.linalg.svd(matrix)
    # s[s < 0] = 1e-5
    # matrix = (u * s) @ vh
    eigval, eigvec = np.linalg.eig(matrix)
    if len(eigval[eigval < 0]) > 0:
        eigval[eigval < 0] = 1e-4
        matrix = eigvec.dot(eigval * np.identity(dim)).dot(np.linalg.inv(eigvec))
    sq_rt_matrix = cholesky(matrix)
    sigma_vectors = np.full(shape=(2 * dim + 1, dim), fill_value=x_mean.astype(float))
    sigma_vectors[1:(dim + 1)] += sq_rt_matrix[0:dim]
    sigma_vectors[(dim + 1):] -= sq_rt_matrix[0:]
    w_m = np.full(shape=(2 * dim + 1), fill_value=1 / (2 * (dim + _lambda)))
    w_m[0] = _lambda / (dim + _lambda)
    w_c = np.copy(w_m)
    w_c[0] += 1 - a ** 2 + b
    return sigma_vectors, w_m, w_c


def unscented_transform(x_mean, x_cov, alpha, beta, k, transformation):
    dim = x_mean.shape[0]
    l = calculate_lambda(dim, alpha, k)
    sigma_pts, Wm, Wc = calc_sigma_points(x_mean, x_cov, l, alpha, beta)
    y = transformation(sigma_pts)
    y_hat = np.dot(Wm, y)
    Py = np.multiply(Wc, (y - y_hat).T).dot(y - y_hat)
    return y_hat, Py, y, Wm, Wc


if __name__ == '__main__':
    rng = np.random.default_rng()
    m = np.array([2, 2])
    Px = np.array([[0.6, 0.3],
                   [0.3, 1]])
    distr = rng.multivariate_normal(mean=m, cov=Px, check_valid='raise', size=10000)
    sample_mean = np.mean(distr ** 3, axis=0)
    sample_cov = np.cov(distr ** 3, rowvar=False)
    UT_mean, UT_cov, _, w1, w2 = unscented_transform(m, Px, 1e-4, 2, 0, lambda x: (x**3))
    print("Sample mean: {0}, UT mean: {1}".format(sample_mean, UT_mean))
    print("Sample cov:")
    print(sample_cov)
    print("UT cov:")
    print(UT_cov)
    plt.scatter((distr ** 3).T[0], (distr ** 3).T[1], s=1)
    plt.show()
