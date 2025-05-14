"""Module containing a Gaussian Process model."""

import logging
import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch.constraints import Interval

__all__ = ["GaussianProcess"]

logger = logging.getLogger(__name__)


def _convert_to_tensor(
    x: list | tuple | pd.Series | pd.DataFrame | np.ndarray | torch.Tensor,
    requires_grad: bool = None,
) -> torch.Tensor:
    """
    Ensures that the input is returned as a torch tensor.
    Inputs can be lists, pandas series, pandas dataframes, numpy arrays and torch tensors.
    :param x: Input to convert
    :return: Converted input
    """
    if isinstance(x, list | tuple):
        x = np.array(x)
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = torch.from_numpy(x.to_numpy()).float()
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if not isinstance(x, torch.Tensor):
        raise ValueError(
            "Input must be a list, tuple, pandas series, "
            + "pandas dataframe, numpy array or torch tensor."
        )
    if requires_grad is not None:
        x.requires_grad_(requires_grad)
    return x


class GaussianProcess(gpytorch.models.ExactGP):
    """
    A Gaussian Process model.
    """

    def __init__(
        self,
        lengthscale_constraint: Interval | None = Interval(1e-5, 1e5),
        noise_constraint: Interval | None = Interval(1e-4, 5e-2),
        fixed_lengthscale: float = None,
        fixed_noise: float = None,
        initial_lengthscale: float = None,
        initial_noise: float = None,
    ) -> None:
        """
        Initializes the Gaussian Process model.
        :param lengthscale_constraint: The kernel lengthscale constraint.
        :param noise_constraint: The noise constraint.
        :param fixed_lengthscale: The fixed lengthscale value. If not None, the lengthscale
            will be optimized using maximum likelihood estimation.
        :param fixed_noise: The fixed noise value. If not None, the noise will be optimized
            using maximum likelihood estimation.
        :param initial_lengthscale: The initial lengthscale for the optimization (if not fixed).
        :param initial_noise: The initial noise for the optimization (if not fixed).
        """
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint
        )
        if fixed_noise is not None:
            likelihood.noise = fixed_noise
            likelihood.raw_noise.requires_grad_(False)
        elif initial_noise is not None:
            likelihood.noise = initial_noise
        super().__init__(None, None, likelihood)
        # Initialize mean and covariance modules
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel(
            lengthscale_constraint=lengthscale_constraint
        )
        if fixed_lengthscale is not None:
            self.covar_module.lengthscale = fixed_lengthscale
            self.covar_module.raw_lengthscale.requires_grad_(False)
        elif initial_lengthscale is not None:
            self.covar_module.lengthscale = initial_lengthscale
        # Set attributes
        self.lengthscale_constraint = lengthscale_constraint
        self.noise_constraint = noise_constraint
        self.fixed_lengthscale = fixed_lengthscale
        self.fixed_noise = fixed_noise
        # Initialize training data
        self.train_inputs = None
        self.train_targets = None

    def _calculate_loss(
        self, mll: gpytorch.mlls.ExactMarginalLogLikelihood
    ) -> torch.Tensor:
        """
        Calculates the negative marginal log likelihood of the model.
        :param mll: The marginal log likelihood object.
        :return: The negative marginal log likelihood.
        """
        output = self(self.train_inputs[0])
        if len(output.mean.shape) == 1:
            # Make sure that the optimization works for 1D input spaces
            self.train_targets = self.train_targets.squeeze(-1)
        return -mll(output, self.train_targets)

    def fit(
        self,
        train_inputs: torch.Tensor | pd.Series | np.ndarray | list,
        train_targets: torch.Tensor | pd.Series | np.ndarray | list,
        number_of_test_points: int = 100,
        training_iterations: int = 5000,
        early_stop_const_iterations: int | None = 10,
    ) -> None:
        """
        Fits the model to the provided training data.
        :param train_inputs: The training inputs.
        :param train_targets: The training targets.
        :param number_of_test_points: The number of test points used to find a good
            starting point for the gradient-based optimization.
        :param training_iterations: The number of training iterations.
        :param early_stopping_constant_iterations: The number of iterations after which
            to stop if the loss does not change. If None, the training will not stop early.
        """
        # Convert inputs to torch tensors
        train_inputs = _convert_to_tensor(train_inputs)
        train_targets = _convert_to_tensor(train_targets)
        self.set_train_data(train_inputs, train_targets, strict=False)
        # Set model to training mode
        self.train()
        self.likelihood.train()
        if self.fixed_lengthscale is not None and self.fixed_noise is not None:
            return
        # Initialize MLL
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        if self.fixed_lengthscale is None:
            # Evaluate lengthscales over the entire lengthscale interval to find a good
            # starting point for the gradient-based optimization
            test_lengthscales = np.logspace(
                np.log10(self.lengthscale_constraint.lower_bound.item()),
                np.log10(self.lengthscale_constraint.upper_bound.item()),
                num=number_of_test_points,
            )
            min_loss = torch.tensor(float("inf"))
            best_lengthscale = None
            for lengthscale in test_lengthscales:
                self.covar_module.lengthscale = lengthscale
                loss = self._calculate_loss(mll)
                logger.debug(
                    "Tested lengthscale %.5f and got loss %.5f",
                    lengthscale,
                    loss.item(),
                )
                if loss < min_loss:
                    min_loss = loss
                    best_lengthscale = lengthscale
            logger.info(
                "Found best starting lengthscale: %.5f with loss %.5f",
                best_lengthscale,
                min_loss,
            )
            self.covar_module.lengthscale = best_lengthscale
        # Initialize gradient optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        # Run iterative optimization
        loss_history = []
        for i in range(training_iterations):
            optimizer.zero_grad()
            loss = self._calculate_loss(mll)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            # Log loss and hyperparameters
            logger.debug(
                "Iter %d/%d - Loss: %.3f   Lengthscale: %.3f  Noise: %.3f",
                i + 1,
                training_iterations,
                loss.item(),
                self.covar_module.lengthscale.item(),
                self.likelihood.noise.item(),
            )
            # Check for convergence
            if early_stop_const_iterations is not None:
                if i > early_stop_const_iterations:
                    history = loss_history[-early_stop_const_iterations:]
                    if np.isclose(
                        history, loss_history[-1], rtol=1e-3, atol=1e-5
                    ).all():
                        break
        # Log final lengthscale and noise
        logger.info(
            "Fitted within %d steps: Lengthscale: %.3f Noise: %.3f",
            i + 1,
            self.covar_module.lengthscale.item(),
            self.likelihood.noise.item(),
        )

    def predict(
        self,
        x: torch.Tensor | pd.Series | np.ndarray | list,
    ) -> gpytorch.distributions.MultivariateNormal:
        """
        Predicts the target values for the provided inputs
        based on the previously fitted model.
        :param x: The inputs to predict the target values for.
        :return: The predicted target values.
        """
        # Convert input to torch tensor
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x = _convert_to_tensor(x, requires_grad=False)
            if len(x) == 0:
                # Prevent errors for inputs with length zero
                x = torch.empty(0, self.train_inputs[0].shape[1]).float()
            # Set model to evaluation mode
            self.eval()
            self.likelihood.eval()
            # Make predictions
            observed_pred = self.likelihood(self(x))
            return observed_pred

    @property
    def lengthscale(self) -> float:
        """
        Returns the lengthscale of the kernel.
        :return: The lengthscale of the kernel.
        """
        return self.covar_module.lengthscale.item()

    @property
    def noise(self) -> float:
        """
        Returns the noise of the likelihood.
        :return: The noise of the likelihood.
        """
        return self.likelihood.noise.item()

    def evaluate_metrics(
        self,
        x_test: torch.Tensor | pd.Series | np.ndarray,
        y_test: torch.Tensor | pd.Series | np.ndarray,
    ) -> dict[str, float]:
        """
        Evaluates the model's performance on the provided test data
        using various metrics.
        These metrics include the negative log predictive density (NLPD),
        the mean standardized log loss (MSLL), the 95% quantile coverage error (Q95CE),
        the mean squared error (MSE), and the mean absolute error (MAE).
        :param x_test: The test inputs.
        :param y_test: The test targets.
        :return: A dictionary containing the evaluated metrics.
        """
        # Convert y_test to torch tensors
        y_test = _convert_to_tensor(y_test)
        # Evaluate metrics
        metrics = {}
        prediction = self.predict(x_test)
        nlpd = gpytorch.metrics.negative_log_predictive_density(prediction, y_test)
        metrics["Negative Log Predictive Density (NLPD)"] = nlpd.item()
        msll = gpytorch.metrics.mean_standardized_log_loss(prediction, y_test)
        metrics["Mean Standardized Log Loss (MSLL)"] = msll.item()
        qce = gpytorch.metrics.quantile_coverage_error(prediction, y_test, 0.95)
        metrics["Quantile 95% Coverage Error (Q95CE)"] = qce.item()
        mse = gpytorch.metrics.mean_squared_error(prediction, y_test, squared=True)
        metrics["Mean Squared Error (MSE)"] = mse.item()
        mae = gpytorch.metrics.mean_absolute_error(prediction, y_test)
        metrics["Mean Absolute Error (MAE)"] = mae.item()
        return metrics

    def forward(  # pylint: disable=arguments-differ
        self,
        x: torch.Tensor,
    ) -> gpytorch.distributions.MultivariateNormal:
        """
        Overrides the abstract forward method of the parent class.
        :param x: The input data.
        :param kwargs: (Ignored)
        :return: The distribution of the output data.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
