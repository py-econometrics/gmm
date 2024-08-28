import numpy as np
from scipy.optimize import minimize
from typing import Callable, Optional
import logging


# tilt functions
def rho_exponential(v: np.ndarray) -> np.ndarray:
    return 1 - np.exp(v)


def rho_cue(v: np.ndarray) -> np.ndarray:
    return -0.5 * v**2 - v


def rho_el(v: np.ndarray) -> np.ndarray:
    return np.log(1 - v)


class GELEstimator:
    """
    Class for Generalized empirical likelihood estimation for vector-valued problems.
    """

    def __init__(
        self,
        m: Callable[[np.ndarray, np.ndarray], np.ndarray],
        rho: Callable[[np.ndarray], np.ndarray] = rho_exponential,
        min_method: str = "L-BFGS-B",
        verbose: bool = False,
        log: bool = False,
    ):
        self.m = m
        self.rho = rho
        self._min_method = min_method
        self._verbose = verbose
        self.est: Optional[np.ndarray] = None
        self.Sigma: Optional[np.ndarray] = None
        self.se: Optional[np.ndarray] = None

        if log:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

    def fit(
        self,
        D: np.ndarray,
        startval: np.ndarray,
        startval2: Optional[np.ndarray] = None,
    ) -> None:
        """Fit GEL estimator"""
        if startval2 is None:
            startval2 = startval
        # Outer max
        result = minimize(
            lambda theta: self._outer_maximisation(theta, D, startval2),
            startval,
            method=self._min_method,
            options={"disp": self._verbose},
        )
        # standard error
        self.est = result.x
        self.Sigma = np.cov(self.m(D, self.est).T)
        self.se = np.sqrt(np.diag(self.Sigma) / D.shape[0])

    def summary(self) -> np.ndarray:
        """Summary table"""
        if self.est is None or self.se is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return np.c_[self.est[:, None], self.se[:, None]]

    def _outer_maximisation(
        self, theta: np.ndarray, D: np.ndarray, startval2: np.ndarray
    ) -> float:
        result = minimize(
            self._inner_minimisation,
            startval2,
            args=(theta, D),
            method=self._min_method,
            options={"disp": self._verbose},
        )
        return -result.fun

    def _inner_minimisation(
        self, lam: np.ndarray, theta: np.ndarray, D: np.ndarray
    ) -> float:
        moments = self.m(D, theta)  # Moment conditions
        obj_value = -np.sum(self.rho(np.dot(moments, lam)))
        logging.info(f"Inner minimisation: lam={lam}, Objective value: {obj_value}")
        return obj_value
