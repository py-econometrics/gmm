from abc import abstractmethod

import numpy as np
from scipy.optimize import minimize
from typing import Callable

import torch
import torchmin


# tilt functions
def rho_exponential(v):
    return 1 - np.exp(v)


def rho_cue(v):
    return -0.5 * v**2 - v


def rho_el(v):
    return np.log(1 - v)

def rho_exponential_torch(v):
    return 1 - torch.exp(v)




class GELEstimator:
    """
    Class for Generalized empirical likelihood estimation for vector-valued problems.
    :param m: Moment condition function.
    :param rho: Tilt function. Defaults to rho_exponential.
    :param min_method: Optimization method for the inner minimization problem_.
    """

    def __new__(cls, m, backend="scipy"):
        backend = backend.lower()
        estimator = _BACKENDS.get(backend)
        if estimator is None:
            raise ValueError(
                f"Backend {backend} is not supported. "
                f"Supported backend are: {list(_BACKENDS.keys())}"
            )
        return super(GELEstimator, cls).__new__(estimator)

    def __init__(
        self,
        m,
        rho=rho_exponential,
        min_method="L-BFGS-B",
        verbose=False,
        log=False,
        backend="scipy",
    ):
        self.m = m
        self.rho = rho
        self._min_method = min_method
        self._verbose = verbose
        self._backend = backend

    @abstractmethod
    def fit(self, D, startval, startval2=None):
        raise NotImplementedError

    @abstractmethod
    def _outer_maximization(self, theta, D, startval2):
        raise NotImplementedError

    @abstractmethod
    def _inner_minimisation(self, lam, theta, D):
        raise NotImplementedError

    def summary(self):
        """Summary table
        Returns:
            np.array: Summary table with estimated coefficients and standard errors
        """
        return np.c_[self.est[:, None], self.se[:, None]]


class GELEstimatorScipy(GELEstimator):
    """
    Class to create GEL object using scipy
    """

    def __init__(
        self,
        m,
        rho=rho_exponential,
        min_method="L-BFGS-B",
        verbose=False,
        backend="scipy",
    ):
        super().__init__(m, rho, min_method, verbose, backend)

    def fit(self, D, startval, startval2=None):
        """Fit GEL estimator

        Args:
            D (ndarray): data matrix
            startval (ndarray): starting values (conformable with moment function m)
        """
        if startval2 is None:
            startval2 = startval
        # Outer maximization
        result = minimize(  # objective function has -1 in front
            self._outer_maximization,
            startval,
            args = (D, startval2),
            method=self._min_method,
            options={"disp": True if self._verbose else False},
        )
        # standard error
        self.est = result.x
        self.Sigma = np.cov(self.m(D, self.est).T)
        self.se = np.sqrt(np.diag(self.Sigma) / D.shape[0])

    # Function for the outer maximization over lambda given a fixed theta
    def _outer_maximization(self, theta, D, startval2):
        result = minimize(
            self._inner_minimisation,
            startval2,
            args=(theta, D),
            method=self._min_method,
            options={"disp": True if self._verbose else False},
        )
        return -result.fun

    # Objective function for the inner minimisation
    def _inner_minimisation(self, lam, theta, D):
        moments = self.m(D, theta)  # Moment conditions
        obj_value = -np.sum(self.rho(np.dot(moments, lam)))
        return obj_value


class GELEstimatorTorch(GELEstimator):
    """
    Class to create GEL object using torch
    """

    def __init__(
        self,
        m,
        rho=rho_exponential_torch,
        min_method="l-bfgs",
        verbose=False,
        backend="torch",
    ):
        super().__init__(m, rho, min_method, verbose, backend)

    def fit(self, D, startval, startval2=None):
        """Fit GEL estimator

        Args:
            D (ndarray): data matrix
            startval (ndarray): starting values (conformable with moment function m)
        """
        if startval2 is None:
            startval2 = startval
        # Outer maximization
        result = torchmin.minimize(  # objective function has -1 in front
            lambda theta: -self._outer_maximization(theta, D, startval2),
            startval,
            method=self._min_method,
            options={"disp": True if self._verbose else False},
        )
        # standard error
        self.est = result.x
        self.Sigma = np.cov(self.m(D, self.est).T)
        self.se = np.sqrt(np.diag(self.Sigma) / D.shape[0])

    def _outer_maximization(self, theta, D, startval2):
        result = torchmin.minimize(
            lambda x: self._inner_minimisation(x, theta, D),
            startval2,
            method=self._min_method,
            options={"disp": True if self._verbose else False},
        )
        return -result.fun

    def _inner_minimisation(self, lam, theta, D):
        moments = self.m(D, theta)
        obj_value = -torch.sum(self.rho(torch.matmul(moments, lam)))
        return obj_value


_BACKENDS = {
    "scipy": GELEstimatorScipy,
    "torch": GELEstimatorTorch,
}
