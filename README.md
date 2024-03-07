# `gmm`: Generalized Method of Moments Estimation

![image](https://github.com/apoorvalal/gmm/assets/12086926/081d4454-cbc8-42ba-91ea-5adff7be03ca)


Solves problems defined as solutions to moment conditions / estimating equations $E[g(Z; \theta)] = 0$ for some function $g(Z; \theta)$ of the data $Z$. Common examples include $g(\theta) = X(y - X\theta)$ for OLS and $g(\theta) = z(y - X\theta)$ for IV. Solutions for a k-dimensional parameter $\theta$ are defined by solving the following optimization problem

$$
\hat{\theta} = \text{argmin}_{\theta}  \left(n^{-1} \sum_i g(Z, \theta)' \right) \mathbf{W} \left(n^{-1}  \sum_i  g(Z, \theta)' \right) 
$$

for a moment condition $g(\cdot)$ and a $m \times m$ weight matrix $\mathbf{W}$. 
For a just-identified problem (M = K), the choice of the weight matrix $\mathbf{W}$ does not matter. For over-identified problems (M > K), it does. Hansen(1982) covers the details and won him the Nobel prize.

Supports both  `scipy.optimize.minimize` and [`pytorch.minimize`](https://pytorch-minimize.readthedocs.io/en/latest/api/index.html#functional-api) to solve the GMM for just- and over-identified problems (with Identity or Optimal weight matrix) and computes HAC-robust standard errors. See OLS and IV examples in `example.ipynb`.


The scipy optimizer uses an analytic expression for the jacobian of linear moment conditions, while the `pytorch.minimize` version uses forward-mode autodiff and therefore supports both linear and non-linear moment conditions.

## Todo

+ [X] Support numerical optimization via [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize)
+ [ ] Support Empirical Likelihood and Generalized Empirical Likelihood
+ [ ] Support Bayesian Bootstrap with exponential(1) weights for inference


## References
+ [Newey and McFadden (1994)](https://users.ssc.wisc.edu/~xshi/econ715/chap36neweymacfadden.pdf)
+ Microeconometrics [Chapter 6], Cameron and Trivedi
