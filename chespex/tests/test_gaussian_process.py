"""Tests the GaussianProcess class."""

import numpy as np
from chespex.optimization.gaussian_process import GaussianProcess


def test_gaussian_process_1():
    """Fits the Gaussian Process model to data from a 1D sine function
    and tests the model's predictions."""
    # Generate data
    x = np.array([0.1, 0.2, 1.0, 4.0, 5.0, 5.5, 8.0, 10.0])
    y = np.sin(x)
    # Fit model
    model = GaussianProcess()
    model.fit(x, y)
    # Test model predictions
    expected_prediction_mean = np.array(
        [0.005, 0.897, 0.799, -0.194, -0.962, -0.664, 0.393, 1.007, 0.466, -0.543]
    )
    expected_prediction_stddev = np.array(
        [0.0218, 0.0185, 0.1226, 0.0637, 0.0177, 0.0152, 0.0961, 0.0430, 0.1486, 0.0141]
    )
    x_test = np.linspace(0, 10, 10)
    prediction = model.predict(x_test)
    assert np.isclose(
        prediction.mean.numpy(), expected_prediction_mean, atol=5e-3
    ).all(), "Mean prediction mismatch"
    assert np.isclose(
        prediction.stddev.numpy(), expected_prediction_stddev, atol=5e-3
    ).all(), "Stddev prediction mismatch"
    # Check model lengthscale and noise
    assert np.isclose(model.lengthscale, 1.77, atol=5e-3), "Lengthscale mismatch"
    assert np.isclose(model.noise, 0.0001, atol=1e-4), "Noise mismatch"
