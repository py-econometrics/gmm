# `gmm`: A library for Z-Estimation

![image](https://github.com/apoorvalal/gmm/assets/12086926/081d4454-cbc8-42ba-91ea-5adff7be03ca)


Functions to solve for parameters $\theta$ defined as solutions to moment conditions / estimating equations
$$E[g(Z; \theta)] = 0$$

for some function $g(Z; \theta)$ of the data $Z$, which is typically a matrix of residuals (hence has expectation 0 in each column). Common examples include $g(\theta) = X(y - X\theta)$ for OLS and $g(\theta) = z(y - X\theta)$ for IV.

Look in the `notebooks/` directory for examples of how to use the library.

## `gmm.GMMEstimator`

solves for a k-dimensional parameter $\theta$ are defined by solving the following optimization problem

$$
\hat{\theta} = \text{argmin}_{\theta}  \left(n^{-1} \sum_i g(Z, \theta)' \right) \mathbf{W} \left(n^{-1}  \sum_i  g(Z, \theta)' \right)
$$

for a moment condition $g(\cdot)$ and a $m \times m$ weight matrix $\mathbf{W}$.
For a just-identified problem (M = K), the choice of the weight matrix $\mathbf{W}$ does not matter. For over-identified problems (M > K), it does. Hansen(1982) covers the details and won him the Nobel prize.

Supports both  `scipy.optimize.minimize` and [`pytorch.minimize`](https://pytorch-minimize.readthedocs.io/en/latest/api/index.html#functional-api) to solve the GMM for just- and over-identified problems (with Identity or Optimal weight matrix) and computes HAC-robust standard errors. See OLS and IV examples in `example.ipynb`.

The scipy optimizer uses an analytic expression for the jacobian of linear moment conditions, while the `pytorch.minimize` version uses forward-mode autodiff and therefore supports both linear and non-linear moment conditions.

## `gmm.GELEstimator`

Solves the same category of problem using generalized empirical likelihood (Exponential tilting by default, but also supports CUE) by solving the following optimization problem

$$
\min_{\pi, \theta} I_{\lambda}(\iota / N, \pi) \; \text{subject to} \; \sum_{i} \pi_i g(Z; \theta) = 0 \; \text{ and } \; \sum_i \pi_i = 1
$$

where $I_\lambda(\iota/N, \pi)$ is a Cressie-Read power-divergence family discrepancy statistic. Intuitively, this approach solves for a weight vector $\pi$ that is minimally different from uniform weights $\iota/N$ while satisfying the moment condition $g(\cdot) = 0$. Different choices of $\lambda$ correspond to different divergence measures, with $\lambda = 0$ corresponding to the empirical likelihood and $\lambda = 1$ corresponding to the chi-squared distance, and $\lambda = -1$ coressponding to the Kullback-Liebler distance. This problem appears daunting since we have (seemingly unnecessarily) added a n-vector of weights $pi$ to the problem, but it turns out that this is a very powerful approach to solving for the parameter $\theta$ in a variety of settings. We use the saddle-point representation (Newey and Smith 2004) that concentrates out the probability weights $\pi$

$$
\min_{\theta \in \Theta} \sup_{\lambda \in \Lambda_n} \sum_i \rho (\lambda ' g(Z_i, \theta))
$$

where $\rho$ is a smooth scalar function that satisfies $\rho(0) = 0, \partial\rho(0)/\partial v = \partial^2 \rho(0) / \partial v^2 = -1$ (which have one-to-one mappings with the Cressie-Read family). We support the following $\rho$s:

+ Empirical Likelihood: $\rho(v) = log(1 - v)$
+ Exponential tilting: $\rho(v) = 1 - \exp(v)$
+ Continuously Updated (CU) GMM: $\rho(v) = -(1/2)v^2 - v$


## Todo / Planned feature updates

+ [X] Support numerical optimization via [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize)
+ [X] Support Empirical Likelihood and Generalized Empirical Likelihood
+ [ ] Support Bayesian Bootstrap with exponential(1) weights for inference
+ [ ] Refactor GMM estimators to accept a single data argument instead of separate X, y, z, etc.

## References
+ [Newey and McFadden (1994)](https://users.ssc.wisc.edu/~xshi/econ715/chap36neweymacfadden.pdf)
+ Microeconometrics [Chapter 6], Cameron and Trivedi
+ Guido Imbens' lectures - [public version](https://www.nber.org/sites/default/files/2022-09/slides_15_el.pdf)
+ Anatolyev and Gospodinov, Methods for Estimation and Inference in Modern Econometrics
+ Owen, Empirical Likelihood
