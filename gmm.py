import numpy as np
import pandas as pd

import scipy
import torch
import torchmin


# %%
class GMMEstimatorScipy:
    """Class to create GMM estimator using scipy"""

    def __init__(self, moment_cond, weighting_matrix="optimal"):
        """Generalized Method of Moments Estimator with Scipy"""
        self.moment_cond = moment_cond
        self.weighting_matrix = weighting_matrix

    def gmm_objective(self, beta):
        """
        Quadratic form to be minimized.
        """
        moments = self.moment_cond(self.z, self.y, self.x, beta)
        if self.weighting_matrix == "optimal":
            self.W = self.optimal_weighting_matrix(moments)
        else:
            self.W = np.eye(moments.shape[1])
        mavg = moments.mean(axis=0)
        return mavg.T @ self.W @ mavg

    def optimal_weighting_matrix(self, moments):
        """
        Optimal Weight matrix
        """
        return np.linalg.inv((1 / self.n) * (moments.T @ moments))

    def fit(self, z, y, x, verbose=False, fit_method=None):
        if fit_method is None:
            fit_method = "L-BFGS-B"
        self.z, self.y, self.x = z, y, x
        self.n, self.k = x.shape
        # minimize the objective function
        result = scipy.optimize.minimize(
            self.gmm_objective,
            x0=np.random.rand(self.k),
            method=fit_method,
            options={"disp": verbose},
        )
        self.theta = result.x
        # Standard error calculation
        try:
            self.Gamma = self.jacobian_moment_cond()
            self.vθ = np.linalg.inv(self.Gamma.T @ self.W @ self.Gamma)
            self.std_errors = np.sqrt(self.n * np.diag(self.vθ))
        except:
            self.std_errors = None

    def jacobian_moment_cond(self):
        """Jacobian of the moment condition"""
        self.jac_est = -self.z.T @ self.x
        return self.jac_est


# %%
class GMMEstimatorTorch:
    """Class to create GMM estimator using torch"""

    def __init__(self, moment_cond, weighting_matrix="optimal"):
        """Generalized Method of Moments Estimator in PyTorch"""
        self.moment_cond = moment_cond
        self.weighting_matrix = weighting_matrix

    def gmm_objective(self, beta):
        """
        Quadratic form to be minimized.
        """
        moments = self.moment_cond(self.z, self.y, self.x, beta)
        if self.weighting_matrix == "optimal":
            self.W = self.optimal_weighting_matrix(moments)
        else:
            self.W = torch.eye(moments.shape[1])
        mavg = moments.mean(axis=0)
        return torch.matmul(
            mavg.unsqueeze(-1).T,
            torch.matmul(self.W, mavg),
        )

    def optimal_weighting_matrix(self, moments):
        """
        Optimal Weight matrix
        """
        return torch.inverse((1 / self.n) * torch.matmul(moments.T, moments))

    def fit(self, z, y, x, verbose=False, fit_method=None):
        if fit_method is None:
            fit_method = "l-bfgs"
        # minimize blackbox using pytorch
        self.z, self.y, self.x = (
            torch.tensor(z, dtype=torch.float64),
            torch.tensor(y, dtype=torch.float64),
            torch.tensor(x, dtype=torch.float64),
        )
        self.n, self.k = x.shape
        beta_init = torch.tensor(
            np.random.rand(self.k), dtype=torch.float64, requires_grad=True
        )
        result = torchmin.minimize(
            self.gmm_objective, beta_init, method=fit_method, tol=1e-5, disp=verbose
        )
        self.W = self.W.detach().numpy()
        # solution
        self.theta = result.x

        # Standard error calculation
        try:
            self.Gamma = self.jacobian_moment_cond()
            self.vθ = np.linalg.inv(self.Gamma.T @ self.W @ self.Gamma)
            self.std_errors = np.sqrt(self.n * np.diag(self.vθ))
        except:
            self.std_errors = None

    def jacobian_moment_cond(self):
        """
        Jacobian of the moment condition
        """
        # forward mode automatic differentiation wrt 3rd arg (parameter vector)
        self.jac = torch.func.jacfwd(self.moment_cond, argnums=3)
        self.jac_est = (
            self.jac(self.z, self.y, self.x, self.theta).sum(axis=0).detach().numpy()
        )
        return self.jac_est


# %%
class GMMEstimator:
    """Class to create GMM estimator using scipy or torch backend."""

    def __init__(self, moment_cond, weighting_matrix="optimal", opt="scipy"):
        self.estimator = self._create_GMMEstimator(moment_cond, weighting_matrix, opt)

    def gmm_objective(self, beta):
        return self.estimator.gmm_objective(beta)

    def optimal_weighting_matrix(self, moments):
        return self.estimator.optimal_weighting_matrix(moments)

    def fit(self, z, y, x, verbose=False, fit_method=None):
        self.estimator.fit(z, y, x, verbose, fit_method)

    def jacobian_moment_cond(self):
        return self.estimator.jacobian_moment_cond()

    def summary(self):
        return pd.DataFrame(
            {"coef": self.estimator.theta, "std err": self.estimator.std_errors}
        )

    def _create_GMMEstimator(self, moment_cond, weighting_matrix, opt):
        if opt == "scipy":
            return GMMEstimatorScipy(moment_cond, weighting_matrix)
        elif opt == "torch":
            return GMMEstimatorTorch(moment_cond, weighting_matrix)


# %% moment conditions to pass to GMM class
def iv_moment_pytorch(z, y, x, beta):
    """Linear IV moment condition in torch"""
    return z * (y - x @ beta).unsqueeze(-1)


def iv_moment_numpy(z, y, x, beta):
    """Linear IV moment condition in numpy"""
    return z * (y - x @ beta)[:, np.newaxis]


# %%
