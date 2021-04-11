import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 # Default arguments are taken from
                 # both SGDRegressor and LogisticRegression
                 theta=0.01,
                 alpha=0.0001,
                 tol=1e-3,
                 max_iter=1000,
                 random_state=None,
                 fit_intercept=True):
        self.theta = theta
        self.alpha = alpha
        self.tol = tol  # epsilon
        self.max_iter = max_iter  # максимальное количество шагов градиентного спуска
        self.random_state = random_state
        self.fit_intercept = fit_intercept

        if random_state is None:
            self.rand_gen = np.random
        else:
            self.rand_gen = np.random.default_rng(self.random_state)

    def fit(self, X, y):
        '''
            X - np.ndarray - [N, n_features] - матрица объектов-признаков
            y - np.ndarray - [N, ] - вектор целевых переменных
        NB:
            В качестве начального приближения w0 рекомендуется выбрать случайный вектор, элементы которого порождены N(0, 1).
        '''
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        if self.fit_intercept:
            X = np.hstack((X, np.ones(X.shape[0]).reshape(-1, 1)))
        # self.weights_ = self.rand_gen.standard_normal(size=X.shape[1], )
        self.weights_ = np.zeros(shape=X.shape[1], dtype=np.float64)
        for _ in range(self.max_iter):
            grad = self._logloss_grad(X, y)
            self.weights_ -= grad * self.theta
            # Same as norm(w_i - w_{i-1})**2
            if np.linalg.norm(self.theta * grad) ** 2 <= self.tol:
                break
        if self.fit_intercept:
            self.coef_ = self.weights_[0:-1]
            self.intercept_ = self.weights_[-1:]
        else:
            self.coef_ = self.weights_
        return self

    def _logloss_grad(self, X, y):
        x_multipicator = -y / (1 + np.exp(y * np.dot(X, self.weights_)))
        # Use np.mean instead of np.sum
        # because overflow encounters on first steps of gradient descent
        return np.sum((X.T * x_multipicator).T,
                       axis=0) + self.alpha * self.weights_

    def _logloss(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.log(1 + np.exp(-y * np.dot(X, self.weights_))) + \
               self.alpha / 2 * np.linalg.norm(self.weights_) ** 2

    def score(self, X, y, sample_weight=None):
        '''
            X - np.ndarray - [N, n_features] - матрица объектов-признаков
        Return:
            y - np.ndarray - [N, ] - вектор скоров, X.dot(w)
        '''
        check_is_fitted(self)
        X = check_array(X)
        return accuracy_score(self.predict(X), y)

    def _logit(self, X):
        ones_proba = 1 / (1 + np.exp(-np.dot(X, self.weights_)))
        zeros_proba = 1 - ones_proba
        return np.vstack([zeros_proba, ones_proba]).T

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        '''
            X - np.ndarray - [N, n_features] - матрица объектов-признаков
        Return:
            y - np.ndarray - [N, 2] - матрица вероятностей:
                                    - в первом столбце вероятность класса "-1",
                                    - во втором столбце - вероятность класса "1"
        '''
        check_is_fitted(self)
        X: np.ndarray
        X = check_array(X)
        if self.fit_intercept:
            X = np.hstack((X, np.ones(X.shape[0]).reshape(-1, 1)))
        return self._logit(X)

    def predict(self, X):
        '''
            X - np.ndarray - [N, n_features] - матрица объектов-признаков
        Return:
            y - np.ndarray - [N, ] - вектор предсказаний {"-1", "1"}
        '''
        check_is_fitted(self)
        X: np.ndarray
        X = check_array(X)

        return ((self.predict_proba(X)[:, 1] > 0.5) - 0.5) * 2
