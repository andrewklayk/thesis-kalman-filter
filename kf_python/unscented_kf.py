import numpy as np
from unscented_transform import calc_sigma_points, calculate_lambda


class UnscentedKF:
    def __init__(self, f, h, R, Q, L, alpha, beta, kappa):
        self.state_trans_func = f  # state transition function g(u_t, mu_(t-1))
        self.obs_func = h  # observation function h(prior_mu_t)
        self.R = R  # state transition uncertainty covariance matrix (process noise cov)
        self.Q = Q  # measurement error covariance matrix (msmt noise cov)
        self.L = L  # dimension
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self._lambda = calculate_lambda(L=self.L, alpha=self.alpha, k=self.kappa)

    def predict(self, prev_mean, prev_cov, u, trans_f, delta_t=1):
        # Calculate sigma points for state
        sigma_points_state, wm_state, wc_state = calc_sigma_points(x_mean=prev_mean, x_cov=prev_cov,
                                                                   _lambda=self._lambda, a=self.alpha, b=self.beta)
        # Propagate through state transition function
        sigma_points_state_propagated = []
        for sp in sigma_points_state:
            sigma_points_state_propagated.append(trans_f(sp, u, delta_t))
        sigma_points_state_propagated = np.array(sigma_points_state_propagated)
        # Predict state mean and covariance
        predict_state_mean = np.dot(wm_state, sigma_points_state_propagated)
        dif_state = sigma_points_state_propagated - predict_state_mean
        predict_state_cov = (wc_state * dif_state.T).dot(dif_state) + self.R
        return predict_state_mean, predict_state_cov, dif_state

    def update(self, predict_mean, predict_cov, dif_state, z, measurement_f):
        # Calculate sigma points for observation
        sigma_points_obs, wm, wc = calc_sigma_points(x_mean=predict_mean, x_cov=predict_cov,
                                                     _lambda=self._lambda, a=self.alpha, b=self.beta)
        sigma_points_obs_propagated = measurement_f(sigma_points_obs)
        # Calculate mean and covariance of observation
        obs_mean = np.dot(wm, sigma_points_obs_propagated)
        dif_obs = sigma_points_obs_propagated - obs_mean
        obs_cov = (wc * dif_obs.T).dot(dif_obs) + self.Q
        # Cross-covariance between predicted state and predicted observation
        cross_cov = (wc * dif_state.T).dot(dif_obs)
        KalmanGain = cross_cov.dot(np.linalg.inv(obs_cov))
        state_mean_post = predict_mean + KalmanGain.dot((z - obs_mean).T)
        state_cov_post = predict_cov - KalmanGain.dot(obs_cov).dot(KalmanGain.T)
        return state_mean_post, state_cov_post

    def propagate(self, mean: np.ndarray, cov: np.ndarray, u: np.ndarray, trans_f, z: np.ndarray,
                  measurement_f, delta_t_u=1):
        predicted_next_mean, predicted_next_cov, dif = self.predict(mean, cov, u, trans_f, delta_t_u)
        return self.update(predicted_next_mean, predicted_next_cov, dif, z, measurement_f)

    def defalut_run(self, x_0, cov_0, controls, observations, num_iterations):
        posterior_means = np.zeros((num_iterations,) + x_0.shape)
        posterior_means[0] = x_0
        posterior_covs = np.ndarray((num_iterations,) + cov_0.shape)
        posterior_covs[0] = cov_0
        for i in range(1, num_iterations):
            res = self.propagate(mean=posterior_means[i - 1], cov=posterior_covs[i - 1], u=controls[i],
                                 trans_f=self.state_trans_func, z=observations[i], measurement_f=self.obs_func)
            posterior_means[i] = res[0]
            posterior_covs[i] = res[1]
        return posterior_means, posterior_covs
