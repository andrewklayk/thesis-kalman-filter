import numpy as np


class ExtendedKF:
    def __init__(self, g, g_derivative, h, h_derivative, R, Q):
        self.g = g  # state transition function g(u_t, mu_(t-1))
        self.g_der = g_derivative  # state transition function derivative w.r.t. x
        self.h = h  # measurement function h(prior_mu_t)
        self.h_der = h_derivative  # measurement function derivative w.r.t. x
        self.R = R  # state transition uncertainty covariance matrix
        self.Q = Q  # measurement error covariance matrix

    def predict(self, m_prev, S_prev, u):
        G = self.g_der(u, m_prev)
        mu_prior = self.g(u, m_prev)
        S_prior = G.dot(S_prev).dot(G.T) + self.R
        return mu_prior, S_prior

    def update(self, m_prior, S_prior, z):
        H = self.h_der(m_prior)
        kalman_gain = (S_prior.dot(H.T)).dot(np.linalg.inv((H.dot(S_prior)).dot(H.T) + self.Q))
        m_post = m_prior + kalman_gain.dot(z - self.h(m_prior))
        S_post = (np.identity(m_prior.shape[0]) - kalman_gain.dot(H)).dot(S_prior)
        return m_post, S_post

    def single_iter(self, m_prev, S_prev, u, z):
        m_prior, S_prior = self.predict(m_prev, S_prev, u)
        m_post, S_post = self.update(m_prior, S_prior, z)
        return m_post, S_post

    def run(self, bel_means, bel_covs, controls, observations, num_iterations):
        posterior_means = np.ndarray(bel_means.shape)
        posterior_means[0] = bel_means[0]
        posterior_covs = np.ndarray(bel_covs.shape)
        posterior_covs[0] = bel_covs[0]
        for i in range(1, num_iterations):
            res = self.single_iter(m_prev=np.transpose([posterior_means[i - 1]]), S_prev=posterior_covs[i - 1],
                                   u=controls[i],
                                   z=observations[i])
            posterior_means[i] = res[0].reshape(2)
            posterior_covs[i] = res[1].reshape((2, 2))
        return posterior_means, posterior_covs
