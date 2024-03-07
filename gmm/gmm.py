from abc import abstractmethod

import numpy as np
import pandas as pd

import scipy
import torch
import torchmin


class GMMEstimator:
    """Class to create GMM estimator using scipy or torch backend."""

    def __new__(cls, moment_cond, weighting_matrix="optimal", backend="scipy"):
        backend = backend.lower()
        estimator = _BACKENDS.get(backend)
        if estimator is None:
            raise ValueError(
                f"Backend {backend} is not supported. "
                f"Supported backend are: {list(_BACKENDS.keys())}"
            )
        return super(GMMEstimator, cls).__new__(estimator)

    def __init__(self, moment_cond, weighting_matrix="optimal", backend="scipy"):
        self.moment_cond = moment_cond
        self.weighting_matrix = weighting_matrix

    @abstractmethod
    def gmm_objective(self, beta):
        raise NotImplementedError

    @abstractmethod
    def optimal_weighting_matrix(self, moments):
        raise NotImplementedError

    @abstractmethod
    def fit(self, z, y, x, verbose=False, fit_method=None, iid=True):
        raise NotImplementedError

    @abstractmethod
    def jacobian_moment_cond(self):
        raise NotImplementedError

    def summary(self, prec=4):
        if not hasattr(self, "theta_") and not hasattr(self, "std_errors_"):
            raise ValueError(
                "Estimator not fitted yet. Make sure you call `fit()` before `summary()`."
            )
        return pd.DataFrame(
            {
                "coef": np.round(self.theta_, prec),
                "std err": np.round(self.std_errors_, prec),
            }
        )


class GMMEstimatorScipy(GMMEstimator):
    """Class to create GMM estimator using scipy"""

    def __init__(self, moment_cond, weighting_matrix="optimal", backend="scipy"):
        """Generalized Method of Moments Estimator with Scipy"""
        super().__init__(moment_cond, weighting_matrix, backend)

    def gmm_objective(self, beta):
        """
        Quadratic form to be minimized.
        """
        moments = self.moment_cond(self.z_, self.y_, self.x_, beta)
        if self.weighting_matrix == "optimal":
            self.W_ = self.optimal_weighting_matrix(moments)
        else:
            self.W_ = np.eye(moments.shape[1])
        mavg = moments.mean(axis=0)
        return mavg.T @ self.W_ @ mavg

    def optimal_weighting_matrix(self, moments):
        """
        Optimal Weight matrix
        """
        return np.linalg.inv((1 / self.n_) * (moments.T @ moments))

    def fit(self, z, y, x, verbose=False, fit_method=None, iid=True):
        if fit_method is None:
            fit_method = "L-BFGS-B"
        self.z_, self.y_, self.x_ = z, y, x
        self.n_, self.k_ = x.shape
        # minimize the objective function
        result = scipy.optimize.minimize(
            self.gmm_objective,
            x0=np.random.rand(self.k_),
            method=fit_method,
            options={"disp": verbose},
        )
        self.theta_ = result.x
        # Standard error calculation
        try:
            self.Gamma_ = self.jacobian_moment_cond()
            if iid:
                self.vtheta_ = np.linalg.inv(self.Gamma_.T @ self.W_ @ self.Gamma_)
            else:
                self.vtheta_ = (
                    np.linalg.inv(self.Gamma_.T @ self.W_ @ self.Gamma_)
                    @ self.Gamma_.T
                    @ self.W_
                    @ self.Omega_
                    @ self.W_
                    @ self.Gamma_
                    @ np.linalg.inv(self.Gamma_.T @ self.W_ @ self.Gamma_)
                )
            self.std_errors_ = np.sqrt(self.n_ * np.diag(self.vtheta_))
        except:
            self.std_errors_ = None

    def jacobian_moment_cond(self):
        """Jacobian of the moment condition"""
        self.jac_est_ = -self.z_.T @ self.x_
        return self.jac_est_

    @staticmethod
    def iv_moment(z, y, x, beta):
        """Linear IV moment condition in numpy"""
        return z * (y - x @ beta)[:, np.newaxis]


# %%
class GMMEstimatorTorch(GMMEstimator):
    """Class to create GMM estimator using torch"""

    def __init__(self, moment_cond, weighting_matrix="optimal", backend="torch"):
        """Generalized Method of Moments Estimator in PyTorch"""
        super().__init__(moment_cond, weighting_matrix, backend)

    def gmm_objective(self, beta):
        """
        Quadratic form to be minimized.
        """
        moments = self.moment_cond(self.z_, self.y_, self.x_, beta)
        if self.weighting_matrix == "optimal":
            self.W_ = self.optimal_weighting_matrix(moments)
        else:
            self.W_ = torch.eye(moments.shape[1])
        mavg = moments.mean(axis=0)
        return torch.matmul(
            mavg.unsqueeze(-1).T,
            torch.matmul(self.W_, mavg),
        )

    def optimal_weighting_matrix(self, moments):
        """
        Optimal Weight matrix
        """
        return torch.inverse((1 / self.n_) * torch.matmul(moments.T, moments))

    def fit(self, z, y, x, verbose=False, fit_method=None, iid=True):
        if fit_method is None:
            fit_method = "l-bfgs"
        # minimize blackbox using pytorch
        self.z_, self.y_, self.x_ = (
            torch.tensor(z, dtype=torch.float64),
            torch.tensor(y, dtype=torch.float64),
            torch.tensor(x, dtype=torch.float64),
        )
        self.n_, self.k_ = x.shape
        beta_init = torch.tensor(
            np.random.rand(self.k_), dtype=torch.float64, requires_grad=True
        )
        result = torchmin.minimize(
            self.gmm_objective, beta_init, method=fit_method, tol=1e-5, disp=verbose
        )
        self.W_ = self.W_.detach().numpy()
        # solution
        self.theta_ = result.x

        # Standard error calculation
        try:
            self.Omega_ = np.linalg.inv(self.W_)
            self.Gamma_ = self.jacobian_moment_cond()
            if iid:
                self.vtheta_ = np.linalg.inv(self.Gamma_.T @ self.W_ @ self.Gamma_)
            else:
                self.vtheta_ = (
                    np.linalg.inv(self.Gamma_.T @ self.W_ @ self.Gamma_)
                    @ self.Gamma_.T
                    @ self.W_
                    @ self.Omega_
                    @ self.W_
                    @ self.Gamma_
                    @ np.linalg.inv(self.Gamma_.T @ self.W_ @ self.Gamma_)
                )
            self.std_errors_ = np.sqrt(self.n_ * np.diag(self.vtheta_))
        except:
            self.std_errors_ = None

    def jacobian_moment_cond(self):
        """
        Jacobian of the moment condition
        """
        # forward mode automatic differentiation wrt 3rd arg (parameter vector)
        self.jac_ = torch.func.jacfwd(self.moment_cond, argnums=3)
        self.jac_est_ = (
            self.jac_(self.z_, self.y_, self.x_, self.theta_)
            .sum(axis=0)
            .detach()
            .numpy()
        )
        return self.jac_est_

    @staticmethod
    def iv_moment(z, y, x, beta):
        """Linear IV moment condition in torch"""
        return z * (y - x @ beta).unsqueeze(-1)


_BACKENDS = {
    "scipy": GMMEstimatorScipy,
    "torch": GMMEstimatorTorch,
}
