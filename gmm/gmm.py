from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, Literal, Dict, Tuple

import numpy as np
import pandas as pd
import scipy
import torch
import torchmin

import pyensmallen as pe
from joblib import Parallel, delayed, parallel_backend
import time


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
        bootstrap: bool = False,
        n_bootstrap: int = 1000,
    ) -> None:
        pass

    @abstractmethod
    def bootstrap_se(
        self, n_bootstrap: int = 1000, seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute bootstrap standard errors using the linear approximation method.

        Args:
            n_bootstrap: Number of bootstrap iterations
            seed: Random seed for reproducibility

        Returns:
            Bootstrap standard errors for each parameter
        """
        pass

    @abstractmethod
    def jacobian_moment_cond(self) -> np.ndarray:
        pass

    def summary(
        self, prec: int = 4, alpha: float = 0.05, method: str = "asymptotic"
    ) -> pd.DataFrame:
        """
        Generate summary statistics for the fitted model.

        Args:
            prec: Precision for rounding results
            alpha: Significance level for confidence intervals
            method: Method for calculating standard errors, either "asymptotic" or "bootstrap"

        Returns:
            DataFrame with model summary statistics
        """
        if not hasattr(self, "theta_"):
            raise ValueError(
                "Estimator not fitted yet. Make sure you call `fit()` before `summary()`."
            )

        # Determine which standard errors to use
        if method == "bootstrap":
            if not hasattr(self, "bootstrap_std_errors_"):
                raise ValueError(
                    "Bootstrap standard errors not available. Call fit() with bootstrap=True or "
                    "manually call bootstrap_se() before calling summary with method='bootstrap'."
                )
            std_errors = self.bootstrap_std_errors_
        else:  # Default to asymptotic
            if not hasattr(self, "std_errors_"):
                raise ValueError(
                    "Asymptotic standard errors not available. There might have been an error during estimation."
                )
            std_errors = self.std_errors_

        return pd.DataFrame(
            {
                "coef": np.round(self.theta_, prec),
                "std err": np.round(std_errors, prec),
                "t": np.round(self.theta_ / std_errors, prec),
                "p-value": np.round(
                    2 * (1 - scipy.stats.norm.cdf(np.abs(self.theta_ / std_errors))),
                    prec,
                ),
                f"[{alpha / 2}": np.round(
                    self.theta_ - scipy.stats.norm.ppf(1 - alpha / 2) * std_errors,
                    prec,
                ),
                f"{1 - alpha / 2}]": np.round(
                    self.theta_ + scipy.stats.norm.ppf(1 - alpha / 2) * std_errors,
                    prec,
                ),
            }
        )

    def compare_se_methods(self, prec: int = 4) -> pd.DataFrame:
        """
        Compare asymptotic and bootstrap standard errors side by side.

        Args:
            prec: Precision for rounding results

        Returns:
            DataFrame comparing asymptotic and bootstrap standard errors
        """
        if not hasattr(self, "theta_") or not hasattr(self, "std_errors_"):
            raise ValueError("Estimator not fitted with asymptotic standard errors.")

        if not hasattr(self, "bootstrap_std_errors_"):
            raise ValueError(
                "Bootstrap standard errors not available. Call bootstrap_se() first."
            )

        return pd.DataFrame(
            {
                "parameter": [f"beta_{i}" for i in range(len(self.theta_))],
                "estimate": np.round(self.theta_, prec),
                "asymptotic_se": np.round(self.std_errors_, prec),
                "bootstrap_se": np.round(self.bootstrap_std_errors_, prec),
                "se_ratio": np.round(
                    self.bootstrap_std_errors_ / self.std_errors_, prec
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
        bootstrap: bool = False,
        n_bootstrap: int = 1000,
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

        # Calculate asymptotic standard errors
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
            self.std_errors_ = np.sqrt(np.diag(self.vtheta_) / self.n_)
        except Exception as e:
            if verbose:
                print(f"Error computing asymptotic standard errors: {e}")
            self.std_errors_ = None

        # Calculate bootstrap standard errors if requested
        if bootstrap:
            try:
                self.bootstrap_std_errors_ = self.bootstrap_se(n_bootstrap=n_bootstrap)
            except Exception as e:
                if verbose:
                    print(f"Error computing bootstrap standard errors: {e}")
                self.bootstrap_std_errors_ = None

    def jacobian_moment_cond(self) -> np.ndarray:
        self.jac_est_ = -self.z_.T @ self.x_
        return self.jac_est_

    def bootstrap_se(
        self, n_bootstrap: int = 1000, seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute bootstrap standard errors using the linear approximation method.

        Args:
            n_bootstrap: Number of bootstrap iterations
            seed: Random seed for reproducibility

        Returns:
            Bootstrap standard errors for each parameter
        """
        if not hasattr(self, "theta_"):
            raise ValueError("Model must be fitted first")

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        n = self.n_  # Sample size

        # influence function: \psi = − (G'WG)^{-1} G W g (z_t , theta_0)
        # φ (z, τ ) = D (τ)^{-1} m (z, τ) ; D(τ) = D(W) = G'W G; m(z, τ) = G'W g(z, theta_0)

        # Step 1: Calculate the matrix M = -(G'WG)^(-1)G'W
        G = self.jacobian_moment_cond()  # Jacobian of moment conditions
        M = -np.linalg.inv(G.T @ self.W_ @ G) @ G.T @ self.W_

        # Step 2: Calculate the individual moment conditions for each observation
        # For the IV case: z_i * (y_i - x_i @ theta)
        residuals = self.y_ - self.x_ @ self.theta_
        individual_moments = self.z_ * residuals[:, np.newaxis]

        # Mean of the moments
        mean_moments = np.mean(individual_moments, axis=0)

        # Step 3: Generate bootstrap samples and calculate theta*
        bootstrap_thetas = np.zeros((n_bootstrap, len(self.theta_)))

        for b in range(n_bootstrap):
            # Draw a bootstrap sample with replacement
            indices = np.random.choice(n, n, replace=True)

            # Calculate the mean of the bootstrap sample moments
            bootstrap_moments = np.mean(individual_moments[indices], axis=0)

            # Center the bootstrap moments
            centered_moments = bootstrap_moments - mean_moments

            # Calculate theta* using the linear approximation
            # sqrt(n)(θ̂* - θ̂) = M * sqrt(n) * (Z* - Z̄)
            # Therefore: θ̂* = θ̂ + M @ (Z* - Z̄)
            bootstrap_thetas[b] = self.theta_ + M @ centered_moments

        # Step 4: Calculate bootstrap standard errors
        bootstrap_se = np.std(bootstrap_thetas, axis=0)

        return bootstrap_se

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
        bootstrap: bool = False,
        n_bootstrap: int = 1000,
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

        # Calculate asymptotic standard errors
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
            self.std_errors_ = np.sqrt(np.diag(self.vtheta_) / self.n_)
        except Exception as e:
            if verbose:
                print(f"Error computing asymptotic standard errors: {e}")
            self.std_errors_ = None

        # Calculate bootstrap standard errors if requested
        if bootstrap:
            try:
                self.bootstrap_std_errors_ = self.bootstrap_se(n_bootstrap=n_bootstrap)
            except Exception as e:
                if verbose:
                    print(f"Error computing bootstrap standard errors: {e}")
                self.bootstrap_std_errors_ = None

    def jacobian_moment_cond(self) -> np.ndarray:
        self.jac_ = torch.func.jacfwd(self.moment_cond, argnums=3)
        self.jac_est_ = (
            self.jac_(self.z_, self.y_, self.x_, torch.tensor(self.theta_))
            .sum(axis=0)
            .detach()
            .numpy()
        )
        return self.jac_est_

    def bootstrap_se(
        self, n_bootstrap: int = 1000, seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute bootstrap standard errors using the linear approximation method.

        Args:
            n_bootstrap: Number of bootstrap iterations
            seed: Random seed for reproducibility

        Returns:
            Bootstrap standard errors for each parameter
        """
        if not hasattr(self, "theta_"):
            raise ValueError("Model must be fitted first")

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        n = self.n_  # Sample size

        # Step 1: Calculate the matrix M = -(G'WG)^(-1)G'W
        G = self.jacobian_moment_cond()  # Jacobian of moment conditions
        M = -np.linalg.inv(G.T @ self.W_ @ G) @ G.T @ self.W_

        # Step 2: Calculate the individual moment conditions for each observation
        # Convert theta to tensor for moment calculation
        theta_tensor = torch.tensor(self.theta_, dtype=torch.float64)

        # For the IV case, calculate residuals
        residuals = (self.y_ - self.x_ @ theta_tensor).detach().numpy()

        # Calculate individual moments (converting torch tensors to numpy)
        z_np = self.z_.detach().numpy()
        individual_moments = z_np * residuals[:, np.newaxis]

        # Mean of the moments
        mean_moments = np.mean(individual_moments, axis=0)

        # Step 3: Generate bootstrap samples and calculate theta*
        bootstrap_thetas = np.zeros((n_bootstrap, len(self.theta_)))

        for b in range(n_bootstrap):
            # Draw a bootstrap sample with replacement
            indices = np.random.choice(n, n, replace=True)

            # Calculate the mean of the bootstrap sample moments
            bootstrap_moments = np.mean(individual_moments[indices], axis=0)

            # Center the bootstrap moments
            centered_moments = bootstrap_moments - mean_moments

            # Calculate theta* using the linear approximation
            bootstrap_thetas[b] = self.theta_ + M @ centered_moments

        # Step 4: Calculate bootstrap standard errors
        bootstrap_se = np.std(bootstrap_thetas, axis=0)

        return bootstrap_se

    @staticmethod
    def iv_moment(
        z: torch.Tensor, y: torch.Tensor, x: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        return z * (y - x @ beta).unsqueeze(-1)


class GMMEstimatorPyEnsmallen(GMMEstimator):
    """Class to create GMM estimator using PyEnsmallen"""

    def __init__(
        self,
        moment_cond: Callable,
        weighting_matrix: Union[str, np.ndarray] = "optimal",
        backend: str = "pyensmallen",
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
        self.bootstrap_thetas_: Optional[np.ndarray] = None
        self.bootstrap_std_errors_: Optional[np.ndarray] = None

    def gmm_objective(self, beta: np.ndarray, gradient: np.ndarray) -> float:
        """
        Compute GMM objective function value and gradient.
        This follows PyEnsmallen's expected interface where the gradient is filled in-place.

        Args:
            beta: Parameter vector
            gradient: Will be filled with gradient values

        Returns:
            Objective function value
        """
        moments = self.moment_cond(self.z_, self.y_, self.x_, beta)

        if self.weighting_matrix == "optimal":
            self.W_ = self.optimal_weighting_matrix(moments)
        elif isinstance(self.weighting_matrix, str):
            self.W_ = np.eye(moments.shape[1])
        elif self.W_ is None:
            self.W_ = np.asarray(self.weighting_matrix)

        mavg = moments.mean(axis=0)
        obj_value = float(mavg.T @ self.W_ @ mavg)

        # Calculate the gradient
        # For the generic case, we use numerical differentiation
        # This could be replaced with analytical gradients for specific moment conditions
        h = 1e-8  # Small step size
        for i in range(len(beta)):
            beta_plus = beta.copy()
            beta_plus[i] += h
            moments_plus = self.moment_cond(self.z_, self.y_, self.x_, beta_plus)
            mavg_plus = moments_plus.mean(axis=0)
            obj_plus = float(mavg_plus.T @ self.W_ @ mavg_plus)
            gradient[i] = (obj_plus - obj_value) / h

        return obj_value

    def optimal_weighting_matrix(self, moments: np.ndarray) -> np.ndarray:
        """Calculate optimal weighting matrix: (E[g_i g_i'])^(-1)"""
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
        """
        Fit the GMM model using PyEnsmallen optimizer.

        Args:
            z: Instrument matrix
            y: Outcome vector
            x: Covariate matrix (including intercept)
            verbose: Whether to print optimization details
            fit_method: Specific optimization method (ignored, uses L-BFGS)
            iid: Whether to assume IID errors for standard error calculation
        """
        # Store data
        self.z_, self.y_, self.x_ = z, y, x
        self.n_, self.k_ = x.shape

        # Initialize optimizer (PyEnsmallen uses L-BFGS by default)
        optimizer = pe.L_BFGS()

        # Initial point (zeros or small random values)
        initial_point = np.zeros(self.k_)

        # Run optimization
        self.theta_ = optimizer.optimize(
            lambda params, gradient: self.gmm_objective(params, gradient), initial_point
        )

        # Compute standard errors
        try:
            # Calculate Jacobian of moment conditions
            self.Gamma_ = self.jacobian_moment_cond()

            # Calculate variance-covariance matrix
            if iid:
                # Simpler formula for IID case
                self.vtheta_ = np.linalg.inv(self.Gamma_.T @ self.W_ @ self.Gamma_)
            else:
                # Robust formula for non-IID case (HAC)
                self.Omega_ = np.linalg.inv(self.W_)
                self.vtheta_ = (
                    np.linalg.inv(self.Gamma_.T @ self.W_ @ self.Gamma_)
                    @ self.Gamma_.T
                    @ self.W_
                    @ self.Omega_
                    @ self.W_
                    @ self.Gamma_
                    @ np.linalg.inv(self.Gamma_.T @ self.W_ @ self.Gamma_)
                )

            # Standard errors (divided by sqrt(n))
            self.std_errors_ = np.sqrt(np.diag(self.vtheta_) / self.n_)
        except Exception as e:
            if verbose:
                print(f"Error computing standard errors: {e}")
            self.std_errors_ = None

    def jacobian_moment_cond(self) -> np.ndarray:
        """
        Compute Jacobian of moment conditions.
        For IV moment conditions, this is -E[z'x].
        For other moment conditions, would need specific implementations.
        """
        # For standard IV moments we know the analytical form
        if (
            isinstance(self.moment_cond, type(self.iv_moment))
            or self.moment_cond.__name__ == "iv_moment"
        ):
            jac = -self.z_.T @ self.x_ / self.n_
        else:
            # Numerical approximation for other moment conditions
            jac = np.zeros((self.z_.shape[1], self.k_))
            h = 1e-8
            beta = self.theta_

            for i in range(self.k_):
                beta_plus = beta.copy()
                beta_plus[i] += h
                moments = self.moment_cond(self.z_, self.y_, self.x_, beta)
                moments_plus = self.moment_cond(self.z_, self.y_, self.x_, beta_plus)
                jac[:, i] = (moments_plus.mean(axis=0) - moments.mean(axis=0)) / h

        return jac

    @staticmethod
    def iv_moment(
        z: np.ndarray, y: np.ndarray, x: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        """Standard IV moment condition: z_i * (y_i - x_i'β)"""
        return z * (y - x @ beta)[:, np.newaxis]

    def bootstrap_se(
        self,
        n_bootstrap: int = 1000,
        seed: Optional[int] = None,
        verbose: bool = False,
        n_jobs: int = 1,
    ) -> np.ndarray:
        """
        Compute bootstrap standard errors using full optimization for each sample.

        Args:
            n_bootstrap: Number of bootstrap iterations
            seed: Random seed for reproducibility
            verbose: Whether to print progress
            n_jobs: Number of parallel jobs (-1 for all cores)

        Returns:
            Bootstrap standard errors
        """
        if not hasattr(self, "theta_"):
            raise ValueError("Model must be fitted first")

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Create bootstrap indices
        n = self.n_
        bootstrap_indices = np.random.choice(n, size=(n_bootstrap, n), replace=True)

        # Define the bootstrap function to be run in parallel
        def run_bootstrap(bootstrap_index, indices):
            # Set a unique seed for each bootstrap iteration to avoid RNG conflicts
            np.random.seed(seed + bootstrap_index if seed is not None else None)

            # Sample data using the indices
            z_boot = self.z_[indices]
            y_boot = self.y_[indices]
            x_boot = self.x_[indices]

            # Create a new estimator and fit
            estimator = GMMEstimatorPyEnsmallen(self.moment_cond, self.weighting_matrix)
            estimator.fit(z_boot, y_boot, x_boot, verbose=False)

            # Return the bootstrap estimates
            return estimator.theta_

        # Run bootstrap in parallel
        if verbose:
            print(f"Running {n_bootstrap} bootstrap iterations with {n_jobs} jobs...")
            start_time = time.time()

        # Use loky backend with controlled threading to avoid parallelism issues
        with parallel_backend("loky", inner_max_num_threads=1):
            bootstrap_results = Parallel(n_jobs=n_jobs)(
                delayed(run_bootstrap)(i, indices)
                for i, indices in enumerate(bootstrap_indices)
            )

        if verbose:
            end_time = time.time()
            print(f"Bootstrap completed in {end_time - start_time:.2f} seconds")

        # Convert results to array
        bootstrap_thetas = np.array(bootstrap_results)

        # Calculate bootstrap standard errors
        bootstrap_se = np.std(bootstrap_thetas, axis=0)

        # Store the bootstrap results
        self.bootstrap_thetas_ = bootstrap_thetas
        self.bootstrap_std_errors_ = bootstrap_se

        return bootstrap_se


_BACKENDS = {
    "scipy": GMMEstimatorScipy,
    "torch": GMMEstimatorTorch,
    "pyensmallen": GMMEstimatorPyEnsmallen,
}
