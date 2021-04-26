import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky


def calculate_lambda(L: int, alpha: float = 1, k: float = 0) -> float:
    """
    Calculate the Lambda parameter for unscented transform.

    :param L: dimension of the distribution
    :param alpha: primary scaling parameter
    :param k: secondary scaling parameter
    :return: calculated value of parameter lambda
    :rtype: float
    """
    return (alpha ** 2) * (L + k) - L


def calc_weights(alpha: float, beta: float, L: int, _lambda: float) -> tuple:
    """
    Calculate weights for unscented transform.

    :param alpha: primary scaling parameter
    :param beta: prior distribution information parameter
    :param L: dimension of the distribution
    :param _lambda: parameter lambda
    :return: tuple of weights w_m and w_c
    :rtype: tuple
    """
    w_m = np.full(shape=(2 * L + 1), fill_value=1 / (2 * (L + _lambda)))
    w_m[0] = _lambda / (L + _lambda)
    w_c = np.empty_like(w_m)
    w_c[:] = w_m
    w_c[0] += 1 - alpha ** 2 + beta
    return w_m, w_c


def calc_sigma_points(x_mean: np.array, x_cov: np.array, _lambda: float) -> np.ndarray:
    dim: int = x_mean.shape[0]
    matrix = (dim + _lambda) * x_cov
    eigval, eigvec = np.linalg.eig(matrix)
    if len(eigval[eigval < 0]) > 0:
        eigval[eigval <= 0] = 1e-4
        matrix = eigvec.dot(eigval * np.identity(dim)).dot(np.linalg.inv(eigvec))
    sq_rt_matrix = cholesky(matrix)
    sigma_vectors = np.full(shape=(2 * dim + 1, dim), fill_value=x_mean.astype(float))
    sigma_vectors[1:(dim + 1)] += sq_rt_matrix[0:dim]
    sigma_vectors[(dim + 1):] -= sq_rt_matrix[0:]
    return sigma_vectors


def unscented_transform(x_mean, x_cov, alpha, beta, k, transformation):
    dim = x_mean.shape[0]
    l = calculate_lambda(dim, alpha, k)
    Wm, Wc = calc_weights(alpha=alpha, beta=beta, L=dim, _lambda=l)
    sigma_pts = calc_sigma_points(x_mean, x_cov, l)
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
    UT_mean, UT_cov, _, w1, w2 = unscented_transform(m, Px, 1e-4, 2, 0, lambda x: (x ** 3))
    print("Sample mean: {0}, UT mean: {1}".format(sample_mean, UT_mean))
    print("Sample cov:")
    print(sample_cov)
    print("UT cov:")
    print(UT_cov)
    plt.scatter((distr ** 3).T[0], (distr ** 3).T[1], s=1)
    plt.show()
