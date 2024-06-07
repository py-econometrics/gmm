import numpy as np
from scipy.optimize import minimize
from typing import Callable
import logging


# tilt functions
def rho_exponential(v):
    return 1 - np.exp(v)


def rho_cue(v):
    return -0.5 * v**2 - v


def rho_el(v):
    return np.log(1 - v)


class GELEstimator:
    """
    Class for Generalized empirical likelihood estimation for vector-valued problems.
    :param m: Moment condition function.
    :param rho: Tilt function. Defaults to rho_exponential.
    :param min_method: Optimization method for the inner minimization problem_.
    """

    def __init__(
        self,
        m: Callable,
        rho: Callable = rho_exponential,
        min_method: str = "L-BFGS-B",
        verbose=False,
        log=False,
    ):
        self.m = m
        self.rho = rho
        self._min_method = min_method
        self._verbose = verbose
        if log:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

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
            lambda theta: self._outer_maximization(theta, D, startval2),
            startval,
            method=self._min_method,
            options={"disp": True if self._verbose else False},
        )
        # standard error
        self.est = result.x
        self.Sigma = np.cov(self.m(D, self.est).T)
        self.se = np.sqrt(np.diag(self.Sigma) / D.shape[0])

    def summary(self):
        """Summary table
        Returns:
            np.array: Summary table with estimated coefficients and standard errors
        """
        return np.c_[self.est[:, None], self.se[:, None]]

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
        logging.info(f"Inner minimisation: lam={lam}, Objective value: {obj_value}")
        return obj_value
