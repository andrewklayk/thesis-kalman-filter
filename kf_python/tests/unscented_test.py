import numpy as np
import matplotlib.pyplot as plt

from unscented_kf import UnscentedKF

# State transition error covariance matrix
R = np.array([[0.25, 0],
              [0, 0.25]])

# Observation error covariance matrix
Q = np.array([[1.2, 0.25],
              [0.25, 1.2]])


def tr_f(x, u):
    rng = np.random.default_rng()
    noise = rng.multivariate_normal([0, 0], R)

    return x + u ** 2 + noise


def obs_f(x):
    return x + rng.multivariate_normal(mean=[0, 0], cov=Q)


if __name__ == '__main__':
    N = 100  # Number of observations / timeframes
    n = 2  # Dimension of the state vector X:
    x = np.zeros((N, n))
    z = np.zeros((N, n))
    x[0] = [5, 5]
    z[0] = [5, 5]
    u = np.ones(N) * 1 + np.random.normal(0, 2, N)  # Control vector

    rng = np.random.default_rng()

    NUM_EXPERIMENTS = 100
    absdif_filt = absdif_obs = 0

    kf = UnscentedKF(f=tr_f, h=obs_f, R=R, Q=Q, L=n, alpha=np.sqrt(3), beta=2, kappa=1)

    for exp in range(NUM_EXPERIMENTS):
        # Generate true state based on initial state and state transition error
        for i in range(N - 1):
            x[i + 1] = tr_f(x[i], u[i + 1])
            z[i + 1] = obs_f(x[i + 1])

        start_cov = R
        m, c = kf.run(x[0], start_cov, u, z, N)

        absdif_filt += np.mean(abs(x - m), axis=0)
        absdif_obs += np.mean(abs(x - z), axis=0)

    print("Average absolute error of observations: {0} over {1} experiments".format(
        absdif_obs / NUM_EXPERIMENTS, NUM_EXPERIMENTS))
    print("Average absolute error of filter-inferred vals: {0} over {1} experiments".format(
        absdif_filt / NUM_EXPERIMENTS, NUM_EXPERIMENTS))

    plt.figure(1)
    plt.subplot(211)
    plt.plot(np.arange(N), x.T[0], marker='o', label='real x')
    plt.plot(np.arange(N), m.T[0], marker='x', linestyle='dashed', label='filtered x')
    plt.plot(np.arange(N), z.T[0], marker='+', linestyle='dashed', label='observed x')
    plt.subplot(212)
    plt.plot(np.arange(N), x.T[1], marker='o', label='real x')
    plt.plot(np.arange(N), m.T[1], marker='x', linestyle='dashed', label='filtered x')
    plt.plot(np.arange(N), z.T[1], marker='+', linestyle='dashed', label='observed x')
    plt.show()
