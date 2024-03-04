# `gmm`: Generalized Method of Moments Estimation

Solves problems defined as solutions to moment conditions / estimating equations $E[g(\theta; Z)] = 0$ for some function $g(\theta; Z)$ of the data $Z$. Common examples include $g(\theta) = X(y - X\theta)$ for OLS and $g(\theta) = z(y - X\theta)$ for IV.

Supports both  `scipy.optimize.minimize` and [`pytorch.minimize`](https://pytorch-minimize.readthedocs.io/en/latest/api/index.html#functional-api) to solve the GMM for just- and over-identified problems (with Identity of Optimal weight matrix) and computes HAC-robust standard errors. See OLS and IV examples in `example.ipynb`.

![image](https://github.com/apoorvalal/gmm/assets/12086926/d1292160-d671-4573-b350-25ddaef8c3e3)

The scipy optimizer uses an analytic expression for the jacobian of linear moment conditions, while the `pytorch.minimize` version uses forward-mode autodiff and therefore supports both linear and non-linear moment conditions.

## Todo

+ [X] Support numerical optimization via [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize)
+ [ ] Support Empirical Likelihood and Generalized Empirical Likelihood
+ [ ] Support Bayesian Bootstrap with exponential(1) weights for inference
