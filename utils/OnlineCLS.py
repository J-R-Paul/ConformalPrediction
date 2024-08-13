import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

def simplex_projection(v):
    """Project onto the probability simplex."""
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

class OnlineConstrLS(BaseEstimator, RegressorMixin):
    """
    Online Constrained Least Squares
    """
    def __init__(self, unit_interval=True, learning_rate=0.01):
        self.unit_interval = unit_interval
        self.learning_rate = learning_rate
        
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        n_features = X.shape[1]
        
        # Initialize weights
        self.coef_ = np.ones(n_features) / n_features
        
        for x_t, y_t in zip(X, y):
            self._update(x_t, y_t)
        
        self.is_fitted_ = True
        return self
    
    def _update(self, x_t, y_t):
        # Compute prediction and error
        y_pred = np.dot(self.coef_, x_t)
        error = y_pred - y_t
        
        # Compute gradient
        gradient = error * x_t
        
        # Update weights
        self.coef_ -= self.learning_rate * gradient
        
        # Project onto constraints
        self.coef_ = simplex_projection(self.coef_)
        
        if self.unit_interval:
            self.coef_ = np.clip(self.coef_, 0, 1)
    
    def partial_fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        
        if not hasattr(self, 'is_fitted_'):
            n_features = X.shape[1]
            self.coef_ = np.ones(n_features) / n_features
        
        for x_t, y_t in zip(X, y):
            self._update(x_t, y_t)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return np.dot(X, self.coef_)
