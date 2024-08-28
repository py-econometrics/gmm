from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, Literal

import numpy as np
import pandas as pd
import scipy
import torch
import torchmin


class GMMEstimator(ABC):
    """Abstract base class for GMM estimators."""

    def __new__(
        cls,
        moment_cond: Callable,
        weighting_matrix: Union[str, np.ndarray] = "optimal",
        backend: str = "scipy",
    ):
        backend = backend.lower()
        estimator = _BACKENDS.get(backend)
        if estimator is None:
            raise ValueError(
                f"Backend {backend} is not supported. "
                f"Supported backends are: {list(_BACKENDS.keys())}"
            )
        return super(GMMEstimator, cls).__new__(estimator)

    def __init__(
        self,
        moment_cond: Callable,
        weighting_matrix: Union[str, np.ndarray] = "optimal",
        backend: str = "scipy",
    ):
        self.moment_cond = moment_cond
        self.weighting_matrix = weighting_matrix

    @abstractmethod
    def gmm_objective(self, beta: np.ndarray) -> float:
        pass

    @abstractmethod
    def optimal_weighting_matrix(self, moments: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def fit(
        self,
        z: np.ndarray,
        y: np.ndarray,
        x: np.ndarray,
        verbose: bool = False,
        fit_method: Optional[str] = None,
        iid: bool = True,
    ) -> None:
        pass

    @abstractmethod
    def jacobian_moment_cond(self) -> np.ndarray:
        pass

    def summary(self, prec: int = 4, alpha: float = 0.05) -> pd.DataFrame:
        if not hasattr(self, "theta_") and not hasattr(self, "std_errors_"):
            raise ValueError(
                "Estimator not fitted yet. Make sure you call `fit()` before `summary()`."
            )
        return pd.DataFrame(
            {
                "coef": np.round(self.theta_, prec),
                "std err": np.round(self.std_errors_, prec),
                "t": np.round(self.theta_ / self.std_errors_, prec),
                "p-value": np.round(
                    2
                    * (
                        1 - scipy.stats.norm.cdf(np.abs(self.theta_ / self.std_errors_))
                    ),
                    prec,
                ),
                f"[{alpha/2}": np.round(
                    self.theta_
                    - scipy.stats.norm.ppf(1 - alpha / 2) * self.std_errors_,
                    prec,
                ),
                f"{1 - alpha/2}]": np.round(
                    self.theta_
                    + scipy.stats.norm.ppf(1 - alpha / 2) * self.std_errors_,
                    prec,
                ),
            }
        )


class GMMEstimatorScipy(GMMEstimator):
    """Class to create GMM estimator using scipy"""

    def __init__(
        self,
        moment_cond: Callable,
        weighting_matrix: Union[str, np.ndarray] = "optimal",
        backend: str = "scipy",
    ):
        super().__init__(moment_cond, weighting_matrix, backend)
        self.z_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None
        self.x_: Optional[np.ndarray] = None
        self.n_: Optional[int] = None
        self.k_: Optional[int] = None
        self.W_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None
        self.Gamma_: Optional[np.ndarray] = None
        self.vtheta_: Optional[np.ndarray] = None
        self.std_errors_: Optional[np.ndarray] = None
        self.Omega_: Optional[np.ndarray] = None

    def gmm_objective(self, beta: np.ndarray) -> float:
        moments = self.moment_cond(self.z_, self.y_, self.x_, beta)
        if self.weighting_matrix == "optimal":
            self.W_ = self.optimal_weighting_matrix(moments)
        else:
            self.W_ = np.eye(moments.shape[1])
        mavg = moments.mean(axis=0)
        return float(mavg.T @ self.W_ @ mavg)

    def optimal_weighting_matrix(self, moments: np.ndarray) -> np.ndarray:
        return np.linalg.inv((1 / self.n_) * (moments.T @ moments))

    def fit(
        self,
        z: np.ndarray,
        y: np.ndarray,
        x: np.ndarray,
        verbose: bool = False,
        fit_method: Optional[str] = None,
        iid: bool = True,
    ) -> None:
        if fit_method is None:
            fit_method = "L-BFGS-B"
        self.z_, self.y_, self.x_ = z, y, x
        self.n_, self.k_ = x.shape
        result = scipy.optimize.minimize(
            self.gmm_objective,
            x0=np.random.rand(self.k_),
            method=fit_method,
            options={"disp": verbose},
        )
        self.theta_ = result.x
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

    def jacobian_moment_cond(self) -> np.ndarray:
        self.jac_est_ = -self.z_.T @ self.x_
        return self.jac_est_

    @staticmethod
    def iv_moment(
        z: np.ndarray, y: np.ndarray, x: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        return z * (y - x @ beta)[:, np.newaxis]


class GMMEstimatorTorch(GMMEstimator):
    """Class to create GMM estimator using torch"""

    def __init__(
        self,
        moment_cond: Callable,
        weighting_matrix: Union[str, torch.Tensor] = "optimal",
        backend: str = "torch",
    ):
        super().__init__(moment_cond, weighting_matrix, backend)
        self.z_: Optional[torch.Tensor] = None
        self.y_: Optional[torch.Tensor] = None
        self.x_: Optional[torch.Tensor] = None
        self.n_: Optional[int] = None
        self.k_: Optional[int] = None
        self.W_: Optional[torch.Tensor] = None
        self.theta_: Optional[torch.Tensor] = None
        self.Gamma_: Optional[np.ndarray] = None
        self.vtheta_: Optional[np.ndarray] = None
        self.std_errors_: Optional[np.ndarray] = None
        self.Omega_: Optional[np.ndarray] = None

    def gmm_objective(self, beta: torch.Tensor) -> torch.Tensor:
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

    def optimal_weighting_matrix(self, moments: torch.Tensor) -> torch.Tensor:
        return torch.inverse((1 / self.n_) * torch.matmul(moments.T, moments))

    def fit(
        self,
        z: np.ndarray,
        y: np.ndarray,
        x: np.ndarray,
        verbose: bool = False,
        fit_method: Optional[str] = None,
        iid: bool = True,
    ) -> None:
        if fit_method is None:
            fit_method = "l-bfgs"
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
        self.theta_ = result.x.detach().numpy()
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

    def jacobian_moment_cond(self) -> np.ndarray:
        self.jac_ = torch.func.jacfwd(self.moment_cond, argnums=3)
        self.jac_est_ = (
            self.jac_(self.z_, self.y_, self.x_, torch.tensor(self.theta_))
            .sum(axis=0)
            .detach()
            .numpy()
        )
        return self.jac_est_

    @staticmethod
    def iv_moment(
        z: torch.Tensor, y: torch.Tensor, x: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        return z * (y - x @ beta).unsqueeze(-1)


_BACKENDS = {
    "scipy": GMMEstimatorScipy,
    "torch": GMMEstimatorTorch,
}
