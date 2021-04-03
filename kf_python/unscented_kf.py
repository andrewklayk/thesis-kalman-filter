import numpy as np
from unscented_transform import calc_sigma_points, calculate_lambda


class UnscentedKF:
    def __init__(self, f, h, R, Q, L, alpha, beta, kappa):
        self.state_trans_func = f  # state transition function g(u_t, mu_(t-1))
        self.obs_func = h  # observation function h(prior_mu_t)
        self.R = R  # state transition uncertainty covariance matrix (process noise cov)
        self.Q = Q  # measurement error covariance matrix (msmt noise cov)
        self.L = L # dimension
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def propagate(self, prev_mean, prev_cov, u, z):
        lam = calculate_lambda(L=self.L, alpha=self.alpha, k=self.kappa)
        # Calculate sigma points for state
        sigma_points_state, wm_state, wc_state = calc_sigma_points(x_mean=prev_mean, x_cov=prev_cov,
                                                                   _lambda=lam, a=self.alpha, b=self.beta)
        # Propagate through state transition function
        sigma_points_state_propagated = []
        for sp in sigma_points_state:
            sigma_points_state_propagated.append(self.state_trans_func(sp, u))
        # sigma_points_state_propagated = self.state_trans_func(sigma_points_state.T, u)
        sigma_points_state_propagated = np.array(sigma_points_state_propagated)
        # Predict state mean and covariance
        predict_state_mean = np.dot(wm_state, sigma_points_state_propagated)
        dif_state = sigma_points_state_propagated - predict_state_mean
        predict_state_cov = (wc_state * dif_state.T).dot(dif_state) + self.R

        # Calculate sigma points for observation
        sigma_points_obs, wm, wc = calc_sigma_points(x_mean=predict_state_mean, x_cov=predict_state_cov,
                                                     _lambda=lam, a=self.alpha, b=self.beta)
        sigma_points_obs_propagated = self.obs_func(sigma_points_obs)
        # Calculate mean and covariance of observation
        obs_mean = np.dot(wm, sigma_points_obs_propagated)
        dif_obs = sigma_points_obs_propagated - obs_mean
        obs_cov = (wc * dif_obs.T).dot(dif_obs) + self.Q
        # Cross-covariance between predicted state and predicted observation
        cross_cov = (wc * dif_state.T).dot(dif_obs)
        KalmanGain = cross_cov.dot(np.linalg.inv(obs_cov))
        state_mean_post = predict_state_mean + KalmanGain.dot((z - obs_mean).T)
        state_cov_post = predict_state_cov - KalmanGain.dot(obs_cov).dot(KalmanGain.T)
        return state_mean_post, state_cov_post

    def run(self, x_0, cov_0, controls, observations, num_iterations):
        posterior_means = np.zeros((num_iterations,) + x_0.shape)
        posterior_means[0] = x_0
        posterior_covs = np.ndarray((num_iterations,) + cov_0.shape)
        posterior_covs[0] = cov_0
        for i in range(1, num_iterations):
            res = self.propagate(prev_mean=posterior_means[i - 1], prev_cov=posterior_covs[i - 1],
                                 u=controls[i],
                                 z=observations[i])
            posterior_means[i] = res[0]
            posterior_covs[i] = res[1]
        return posterior_means, posterior_covs
