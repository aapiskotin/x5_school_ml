import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class GDLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, lambda_coef=0.1, regularization=None, alpha=0.5):
        """
        :param lambda_coef: constant coef for gradient descent step
        :param regularization: regularizarion type ("L1" or "L2") or None
        :param alpha: regularizarion coefficent
        """
        self.regularization = regularization
        self.lambda_coef = lambda_coef
        self.alpha = alpha
        self.fitted = False

    def fit(self, X_train, y_train, epoches=1000):
        """
        Fit model using gradient descent method
        :param X_train: training data
        :param y_train: target values for training data
        :return: None
        """
        X_train = np.hstack((np.ones(X_train.shape[0]).reshape(-1, 1), X_train))
        self._weights = np.random.normal(size=X_train.shape[1])
        for _ in range(epoches):
            grad = self._grad_loss_func(X_train, y_train)
            self._weights -= grad * self.lambda_coef
            if np.sum(np.abs(grad)) < 1e-10:
                break
        return self

    def predict(self, X_test):
        """
        Predict using model.
        :param X_test: test data for predict in
        :return: y_test: predicted values
        """
        try:
            return np.dot(np.hstack((np.ones(X_test.shape[0]).reshape(-1, 1), X_test)),  self._weights)
        except:
            raise Exception("Model is not fitted")

    def _grad_loss_func(self, X, y):
        result = (-2 * np.dot(y.T, X) + 2 * np.dot(self._weights.T, np.matmul(X.T, X))) / X.shape[0]
        if self.regularization is None:
            return result
        elif self.regularization == 'L1':
            return result + self.alpha * np.sign(self._weights) / X.shape[0]
        elif self.regularization == 'L2':
            return result + self.alpha * 2 * self._weights / X.shape[0]

    def __getattr__(self, item):
        if item == 'coef_':
            return self._weights[1:]
        elif item == 'intercept_':
            return self._weights[0]
        raise AttributeError


if __name__ == "__main__":
    from sklearn.linear_model import SGDRegressor
    from sklearn.datasets import make_regression
    from numpy.testing import assert_array_almost_equal

    X, y = make_regression()

    gd_reg = GDLinearRegression().fit(X, y)
    sgd_reg = SGDRegressor(learning_rate='constant').fit(X, y)
    assert_array_almost_equal(gd_reg.coef_, sgd_reg.coef_, decimal=3)
    assert_array_almost_equal(gd_reg.intercept_, sgd_reg.intercept_, decimal=3)
