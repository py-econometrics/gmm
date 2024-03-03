import numpy as np
import pandas as pd
from scipy.optimize import minimize


# %%

class GMMEstimator:
    def __init__(self, moment_cond, weighting_matrix='optimal'):
        """Minimal implementation of the GMM estimator

        Args:
            moment_cond (function): Moment condition. Returns L X n matrix of moments
            weighting_matrix (str, optional): What kind of weight matrix to use. Defaults to 'optimal'.

        ref: Hansen (1982), Cameron and Trivedi (2005)
        """
        self.moment_cond = moment_cond
        self.weighting_matrix = weighting_matrix

    def gmm_objective(self, beta):
        """
        Quadratic form to be minimized.
        """
        moments = self.moment_cond(self.z, self.y, self.x, beta)
        if self.weighting_matrix == 'optimal':
            self.W = self.optimal_weighting_matrix(moments)
        else:
            self.W = np.eye(moments.shape[1])
        return (1/self.n) * moments.sum(axis=0).T @ self.W @ moments.sum(axis=0)

    def optimal_weighting_matrix(self, moments):
        """
        Optimal Weight matrix
        """
        return np.linalg.inv((1 / self.n) * (moments.T @ moments))

    def jacobian_moment_cond(self, z, y, x):
        """
        Analytic Jacobian for linear IV.
        """
        # !TODO implement this with jax for arbitrary moment conditions
        return -z.T @ x

    def fit(self, z, y, x):
        self.z, self.y, self.x = z, y, x
        self.n, self.k = x.shape
        # minimize the objective function
        result = minimize(self.gmm_objective,
                x0 = np.random.rand(self.k),
            method = "L-BFGS-B")
        # solution
        self.theta = result.x

        # Standard error calculation
        try:
            # self.Γ = jax.jacrev(self.moment_cond)(self.z, self.y, self.x, self.theta)
            self.Γ = self.jacobian_moment_cond(self.z, self.y, self.x)
            self.vθ = np.linalg.inv(self.Γ.T @ self.W @ self.Γ)
            self.std_errors = np.sqrt(self.n * np.diag(self.vθ))
        except:
            self.std_errors = None

    def summary(self):
        return pd.DataFrame({'coef': self.theta, 'std err': self.std_errors})
