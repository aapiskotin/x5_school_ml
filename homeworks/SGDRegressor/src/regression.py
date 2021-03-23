from typing import Optional

import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.base import BaseEstimator, RegressorMixin


class GDLinearRegression(BaseEstimator, RegressorMixin):
    """
    Similar to sklearn.linear_model.SGDRegressor with default parameters (and learning_rate=='constant')
    that not presented in GDLinearRegression.__init__ and uses gradient descent optimization technique
    (not stochastic instead of SGDRegressor)
    """

    def __init__(self,
                 penalty: Optional[str] = 'l2',
                 l1_ratio: float = 0.15,
                 alpha: float = 0.00001,
                 fit_intercept: bool = True,
                 max_iter: int = 10000,
                 tol: float = 1e-3,
                 eta0: float = 0.01,
                 random_state: Optional[int] = None):
        if random_state is None:
            self.random_state = 648
        self.rnd_gen = np.random.default_rng(self.random_state)
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.learning_rate = eta0
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        """
        Fit model using gradient descent method
        :param X: training data
        :param y: target values for training data
        :return: None
        """
        if self.fit_intercept:
            X = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))
        self.w_ = self.rnd_gen.normal(size=X.shape[1], )
        for _ in range(self.max_iter):
            grad = self._grad_loss_func(X, y)
            self.w_ -= grad * self.learning_rate
            if np.sum(np.abs(grad)) < self.tol:
                break
        if self.fit_intercept:
            self.coef_ = self.w_[1:]
            self.intercept_ = self.w_[0:1]
        else:
            self.coef_ = self.w_
        return self

    def predict(self, X):
        """
        Predict using model.
        :param X: test data for predict in
        :return: y_test: predicted values
        """
        try:
            if self.fit_intercept:
                X = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))
            return np.dot(X, self.w_)
        except:
            raise Exception("Model is not fitted")

    def _grad_loss_func(self, X, y):
        grad = (-2*np.dot(y.T, X) + 2*np.dot(self.w_.T,
                                             np.matmul(X.T, X))) / X.shape[0]
        if self.penalty == 'L1':
            grad += self.l1_penalty()
        elif self.penalty == 'L2':
            grad += self.l2_penalty()
        elif self.penalty == 'elasticnet':
            grad += self.l1_penalty() + self.l2_penalty()

        return grad

    def l1_penalty(self):
        return self.alpha * np.sign(self._weights) / X.shape[0]

    def l2_penalty(self):
        return self.alpha * 2 * self._weights / X.shape[0]


if __name__ == "__main__":
    from sklearn.linear_model import SGDRegressor
    from sklearn.datasets import make_regression

    X, y = make_regression()

    gd_reg = GDLinearRegression().fit(X, y)
    sgd_reg = SGDRegressor(learning_rate='constant').fit(X, y)

    assert gd_reg.coef_.shape == sgd_reg.coef_.shape
    assert gd_reg.intercept_.shape == sgd_reg.intercept_.shape

    from sklearn.metrics import mean_squared_error as mse

    assert mse(y, gd_reg.predict(X)) <= mse(y, sgd_reg.predict(X))
