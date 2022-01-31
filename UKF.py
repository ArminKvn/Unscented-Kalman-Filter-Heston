from scipy.stats import norm
import numpy as np
from statistics import variance as sample_var
import math


class Heston_Unscented_Kalman_Filter(object):
    """ This object contains all sigma points, evolution of sigma points, and predicted returns

    Attributes:
    -----------
    L : int : Dimension of the model

    alpha : float : Spread of the sigma points around the mean X bar. Set to 10 ^ -3

    k : float : Secondary scaling parameter set to 0 in this case

    Beta : float : incorporates our information form the prior. Set to 2 for Gaussian

    lamb : float : composite scaling parameter

    P_x : Array : Covariance Matrix of the random variable vector X

    x_bar : float : Mean of random variable vector x

    y_bar : float : Mean of random variable vector y

    X : array : 2L + 1 vector of sigma points x_i

    Param = 1 x 5 Vector containing the values of:
            [Mu, Kappa, Theta, Xi, Rho] in this specific order
            where Mu = Mean drift of the asset
            Kappa = Reversion Speed
            Theta = Reversion Level
            Xi = Volatility of the Volatility
            Rho = Correlation between the Brownian Motions

    h : float : delta_t, set to 1/252 here for 252 trading days in the year

    weight_m : 1 x 2L+1 array:  for generating weighted x_bar and y_bar

    weight_c : 1 x 2L+1 array: for generating covariance matrices of sigma points


    Class Methods:
    -------------
    update_x : updates unobserved sigma points -> the volatility

    update_y : updates the observed sigma points -> the returns

    generate_ss : calculates the weighted sum_squares deviation from the mean of updates

    update : time update and measurement update to calculate the error used for getting the MLE

    Raised errors:
    -------------
    typeErrors will be raised when dimensions of the weights matrix or particles matrix changes after broadcasting
    """

    def __init__(self, observations, h=1 / 252, Param=[0.1, 1, 2, 3, 0.1], INIT_values=[0.05, 2]):
        # Parameters of the particle filter
        self.mu = np.array(Param[0])
        self.kappa = np.array(Param[1])
        self.theta = np.array(Param[2])
        self.xi = np.array(Param[3])
        self.rho = np.array(Param[4])
        self.delta_t = np.array(h)
        self.INIT = INIT_values  # initial values given for volatility at time 0, and its initial covariance matrix(here it would be just a scalar)
        self.log_like = 0

        self.obs_len = len(observations)
        self.observations = observations
        self.state = 0

        self.L = 1  # we only have one latent/ unobserved variable -> volatility
        self.alpha = 10 ** -3  # as per suggestion of the author of the paper
        self.beta = 2  # Assuming Gaussian
        self.k = 0  # secondary Scaling parameter, set to 0
        self.lamb = (self.alpha ** 2) * (self.L + self.k) - self.L

        self.weight_m = np.array(
            [(self.lamb / (self.L + self.lamb)), 1 / (2 * (self.L + self.lamb)), 1 / (2 * (self.L + self.lamb))])
        self.weight_c = np.array(
            [(self.lamb / (self.lamb + self.L)) + (1 - self.alpha ** 2 + self.beta), 1 / (2 * (self.L + self.lamb)),
             1 / (2 * (self.L + self.lamb))])

        # to track the volatility covariance at each time step
        self.P_x = np.array([self.INIT[1]]) + np.zeros(self.obs_len + 1)

        # to track the volatility  at each time step
        self.x = np.array(self.INIT[0]) + np.zeros(self.obs_len + 1)

        # to apply the unscented transform to observed values at each time step for all sigma points (here we have 2L+1 = 3 sigma points)
        self.y = np.zeros(2 * self.L + 1)

        # to keep track of our estimated returns
        self.y_estimate = np.zeros(self.obs_len)

        self.measurement_noise = np.array([0, math.sqrt(self.L + self.lamb), -1 * math.sqrt(self.L + self.lamb)])
        self.transition_noise = np.array([0, math.sqrt(self.L + self.lamb), -1 * math.sqrt(self.L + self.lamb)])

    """
    Methods:
    --------
    """

    def RELU(self, x):
        return np.maximum(0, x)

    def shape_checker(self, np_array, expected_shape: tuple, array_name):
        if np_array.shape != expected_shape:
            raise ValueError(f"{array_name} has mutated, expected{expected_shape}, received {np_array.shape}")

    def update_x(self, transformed_volatility, epsilon):
        temp = self.RELU(transformed_volatility + self.kappa * (self.theta - transformed_volatility) * self.delta_t
                         + self.xi * np.dot(np.sqrt(transformed_volatility * self.delta_t), epsilon)
                         ).reshape(1, 3)

        self.shape_checker(temp, (1, 3), "update_x")
        return temp[0]

    def update_y(self, updated_transformed_volatility, transformed_return, epsilon, eta):
        temp = (
                transformed_return + (self.mu - updated_transformed_volatility * 0.5) * self.delta_t +
                np.sqrt(self.delta_t * updated_transformed_volatility) *
                (self.rho * epsilon + math.sqrt(1 - self.rho ** 2) * eta)
        ).reshape(1, 3)
        self.shape_checker(temp, (1, 3), "update_y")
        return temp[0]

    def generate_ss(self, updated_x, updated_x_bar, updated_y, updated_y_bar, weights):
        temp = 0
        for i in range(2 * self.L + 1):
            temp += weights[i] * (updated_x[i] - updated_x_bar) * (updated_y[i] - updated_y_bar)
        return temp

    def update(self):
        while self.state < self.obs_len:
            # applying our unscented transform to the point x at time_step: state. we will have the point itself, and 2 symmetric points around it.
            current_latent_sigma_points = np.array([self.x[self.state], self.x[self.state] + math.sqrt(
                self.P_x[self.state] * (self.L + self.lamb)),
                                                    self.RELU(self.x[self.state] - math.sqrt(
                                                        self.P_x[self.state] * (self.L + self.lamb)))])

            # updating the unobserved sigma points (x)  and calculating new mean and variances for them
            current_latent_sigma_points = self.update_x(current_latent_sigma_points, self.weight_m)
            x_bar = np.average(current_latent_sigma_points, weights=self.weight_m)
            p_bar = self.generate_ss(current_latent_sigma_points, x_bar, current_latent_sigma_points, x_bar,
                                     self.weight_m)

            # updating the corresponding y values and calculating the mean and variances for them. In this step we add the correlated noise as well
            self.y = self.update_y(current_latent_sigma_points, self.observations[self.state], self.measurement_noise,
                                   self.transition_noise)
            y_bar = np.average(self.y, weights=self.weight_m)
            p_yy = self.generate_ss(self.y, y_bar, self.y, y_bar, self.weight_c)

            p_xy = self.generate_ss(current_latent_sigma_points, x_bar, self.y, y_bar, self.weight_c)
            u = p_yy

            K = np.dot(p_xy, 1 / p_yy)
            self.x[self.state + 1] = self.RELU(x_bar + K * (self.observations[self.state] - y_bar))
            self.P_x[self.state + 1] = self.RELU(p_bar - K * p_yy * K)
            self.y_estimate[self.state] = y_bar
            error = self.observations[self.state] - y_bar
            MLE = (1 / math.sqrt(math.pi * 2 * u)) * math.exp(- (error ** 2) / 2)
            self.log_like += math.log(MLE)
            self.state += 1

