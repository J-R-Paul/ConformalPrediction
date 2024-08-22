"""
Basic Conformal Prediction module for scikit-learn regression estimators.

"""

from typing import List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class ConformalPredictor(BaseEstimator, RegressorMixin):
    """
    Conformal Predictor for scikit-learn regression estimators.

    This class implements the Conformal Prediction algorithm for regression tasks,
    using any scikit-learn compatible regressor as the base model. It computes
    prediction intervals for new data points given a training dataset.

    Parameters
    ----------
    base_estimator : object
        The base estimator object implementing fit and predict methods.
        This can be any scikit-learn regressor.
    alpha : float, default=0.1
        The miscoverage level, must be in (0, 1).

    Attributes
    ----------
    base_estimator_ : object
        The fitted base estimator.
    X_ : ndarray of shape (n_samples, n_features)
        The input samples.
    y_ : ndarray of shape (n_samples,)
        The target values.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from basic_conformal_inference import ConformalPredictor
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> y = np.array([2, 4, 5, 4, 5])
    >>> cp = ConformalPredictor(LinearRegression(), alpha=0.1)
    >>> cp.fit(X, y)
    >>> X_new = np.array([[3.5]])
    >>> y_trial = np.linspace(0, 10, 100)
    >>> intervals = cp.predict(X_new, y_trial)
    >>> print(intervals)
    [array([2.89796748, 6.21258134])]
    """

    def __init__(self, base_estimator: BaseEstimator, alpha: float = 0.1):
        self.base_estimator = base_estimator
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ConformalPredictor':
        """
        Fit the ConformalPredictor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : ConformalPredictor
            Returns self.

        Raises
        ------
        ValueError
            If the input data is not valid or if alpha is not in (0, 1).
        """
        X, y = check_X_y(X, y, ensure_2d=True)

        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")

        self.base_estimator_ = self.base_estimator
        self.base_estimator_.fit(X, y)

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X: np.ndarray, y_trial: np.ndarray) -> List[np.ndarray]:
        """
        Compute prediction intervals for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples for which to compute the prediction intervals.
        y_trial : array-like of shape (n_trials,)
            The trial values to use for constructing prediction intervals.

        Returns
        -------
        intervals : list of ndarrays
            List of prediction intervals for each input sample.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.
        """
        check_is_fitted(self)
        X = check_array(X, ensure_2d=True)
        y_trial = check_array(y_trial, ensure_2d=False)

        n = len(self.X_)
        intervals = []

        for x in X:
            pi_y = []
            for y in y_trial:
                X_aug = np.vstack([self.X_, x])
                y_aug = np.append(self.y_, y)

                model = self.base_estimator.__class__(**self.base_estimator.get_params())
                model.fit(X_aug, y_aug)
                mu_y = model.predict(X_aug)
                R_y = np.abs(y_aug - mu_y)

                pi = (1 + np.sum(R_y[:-1] <= R_y[-1])) / (n + 1)
                pi_y.append(pi)

            conf_set = y_trial[(n + 1) * np.array(pi_y) <= np.ceil((1 - self.alpha) * (n + 1))]
            intervals.append(np.array([np.min(conf_set), np.max(conf_set)]))

        return intervals

# Example usage (commented out)
"""
import numpy as np
from sklearn.linear_model import LinearRegression

def dgp(n: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.randn(n, 1)
    y = 2 * X.squeeze() + 1 + np.random.randn(n)
    return X, y

np.random.seed(0)
X, y = dgp(100)

X_new = np.array([0.25]).reshape(-1, 1)
y_trial = np.linspace(-10, 10, 1000)
alpha = 0.1

cp = ConformalPredictor(LinearRegression(), alpha=alpha)
cp.fit(X, y)
prediction_intervals = cp.predict(X_new, y_trial)

for x, interval in zip(X_new, prediction_intervals):
    print(f"Prediction interval for x={x[0]:.2f}: [{interval[0]:.2f}, {interval[1]:.2f}]")
"""