import numpy as np
from extended_kf import ExtendedKF


def transition_function(x_prev, u):
    return x_prev / u

def tf_der(x_prev, u):
    return -u / (x_prev ** 2)

def measurement_function(x):
    return x ** 2

def mf_der(x):
    return 2 * x


if __name__ == '__main__':
    N = 100
    controls = np.arange(1, N + 1)
    true_states = np.arange(1, N + 1)
    observations = np.zeros(N)

    rng = np.random.default_rng()
    R = np.array([25])
    Q = np.array([169])

    sqdif_filtered = 0
    sqdif_observed = 0
    NUM_EXPERIMENTS = 10

    for i in range(NUM_EXPERIMENTS):
        for j in range(0, N - 1):
            true_states[j + 1] = transition_function(true_states[j], controls[j]) + np.random.normal(0, R, 1)
        for j in range(0, N):
            observations[j] = measurement_function(true_states[j]) + np.random.normal(0, Q, 1)
        kf = ExtendedKF(transition_function, tf_der, measurement_function, mf_der, R, Q)

        m, c = kf.run(observations, np.array([[25, 0],[0,16]] * N), controls, observations, N)

        sqdif_filtered += np.mean((true_states - m) ** 2, axis=0)
        sqdif_observed += np.mean((true_states - observations) ** 2, axis=0)

    print("Average squared error of observations: {0} over {1} experiments".format(
        sqdif_observed / NUM_EXPERIMENTS, NUM_EXPERIMENTS))
    print("Average squared error of filter-inferred vals: {0} over {1} experiments".format(
        sqdif_filtered / NUM_EXPERIMENTS, NUM_EXPERIMENTS))
