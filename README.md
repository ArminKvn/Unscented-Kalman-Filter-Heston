# Unscented Kalman Filter Heston
  This Heston UKF class calculates the log liklihood values of the initial values given to the class for the heston model (heston, 1993) using Unscented Kalman Filters.

Heston model is an extremly important Stochastic volatility model. Many of the current advanced models are variations of this model. For example, we arrive at the bates model simply allows stochastic jumps for the heston model.

A good introduction to the heston model can be found at: https://en.wikipedia.org/wiki/Heston_model

Initial Paper for the Unscented Kalman filter:
E. A. Wan and R. Van Der Merwe, "The unscented Kalman filter for nonlinear estimation," Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium (Cat. No.00EX373), Lake Louise, Alberta, Canada, 2000, pp. 153-158, doi: 10.1109/ASSPCC.2000.882463.

A Short explenation of the UKF:
Due to the non-linear nature of our dynamics in the Heston Model, we are unable to use Kalman Filters and instead we use the unscented Kalman filter (UKF). 
Typically, it is much easier to estimate a probability distribution than it is to approximate an arbitrary nonlinear function (Julier, Uhlmann, DurrantWhyte, 2000). Therefore, in the UKF we use unscented transform to pass a probability distribution through a non-linear function allowing us to use similar steps as the prediction and
correction steps of the Kalman Filter.

A known issue regarding the UKF and the stochastic volatility models comes from the parameters Kappa. Large values of Kappa could break down the algorithm since they make the covariance matrix non reversible (determinant approaches 0), which further leads to the breakdown of the choleski decomposition. 

