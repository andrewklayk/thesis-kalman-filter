import numpy as np


class LinearKF:
    def __init__(self, A, B, C, R, Q):
        self.A = A  # state matrix
        self.B = B  # control matrix
        self.C = C  # measurement matrix
        self.R = R  # state transition uncertainty covariance matrix
        self.Q = Q  # measurement error covariance matrix

    def predict(self, mean_prev, cov_prev, u):
        mean_prior = self.A.dot(mean_prev) + self.B.dot(u)
        cov_prior = np.dot(np.dot(self.A, cov_prev), self.A.T) + self.R
        return mean_prior, cov_prior

    def update(self, mean_prior, cov_prior, z):
        # measurement covariance matrix
        z_cov = self.C.dot(cov_prior).dot(self.C.T) + self.Q
        kalman_gain = cov_prior.dot(self.C.T).dot(np.linalg.inv(z_cov))
        mean_post = mean_prior + kalman_gain.dot((z - self.C.dot(mean_prior).T).T)
        cov_post = (np.identity(self.A.shape[0]) - kalman_gain.dot(self.C)).dot(cov_prior)
        return mean_post, cov_post

    def propagate_one_step(self, m_prev, S_prev, u, z):
        mean_prior, cov_prior = self.predict(m_prev, S_prev, u)
        mean_post, cov_post = self.update(mean_prior, cov_prior, z)
        return mean_post, cov_post

    def run(self, bel_means, bel_covs, controls, observations, num_iterations):
        posterior_means = np.ndarray(bel_means.shape)
        posterior_means[0] = bel_means[0]
        posterior_covs = np.ndarray(bel_covs.shape)
        posterior_covs[0] = bel_covs[0]
        for i in range(1, num_iterations):
            res = self.propagate_one_step(m_prev=np.transpose([posterior_means[i - 1]]), S_prev=posterior_covs[i - 1],
                                          u=controls[i], z=observations[i])
            posterior_means[i] = res[0].reshape(2)
            posterior_covs[i] = res[1].reshape((2, 2))
        return posterior_means, posterior_covs
