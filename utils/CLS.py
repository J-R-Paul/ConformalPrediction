from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y, check_array
from scipy.optimize import nnls
from scipy.optimize import minimize
from sklearn.utils.validation import check_is_fitted
import numpy as np


# ConstrLS = "Constrained Least Squares" i.e. NNLS.
class ConstrLS(BaseEstimator):
    """
    Constrained least squares, weights sum to 1 and optionally >= 0
    """
    _estimator_type = "regressor"

    def __init__(self, unit_interval: bool = True):
        self.unit_interval = unit_interval

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        xdim = X.shape[1]

        # Use nnls to get initial guess
        coef0, rnorm = nnls(X, y)

        # Define minimisation function
        def fn(coef, X, y):
            return np.linalg.norm(X.dot(coef) - y)

        # Constraints and bounds
        cons = {'type': 'eq', 'fun': lambda coef: np.sum(coef) - 1}
        if self.unit_interval == True:
            bounds = [[0.0, 1.0] for i in range(xdim)]
        else:
            bounds = None

        # Do minimisation
        fit = minimize(fn, coef0, args=(X, y), method='SLSQP',
                       bounds=bounds, constraints=cons)
        self.coef_ = fit.x
        self.is_fitted_ = True
        self.cvalid = X
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return np.matmul(X, self.coef_)

