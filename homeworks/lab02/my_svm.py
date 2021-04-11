import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class CustomSVC(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 theta=0.1,
                 C=1.0,
                 tol=1e-3,
                 max_iter=1000,
                 random_state=None):
        self.theta = theta
        self.C = C
        self.tol = tol  # epsilon
        self.max_iter = max_iter  # максимальное количество шагов градиентного спуска
        self.random_state = random_state
        self._b = 0.0

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

        self._weights = self.rand_gen.standard_normal(size=X.shape[1], )
        # self._weights = np.zeros(shape=X.shape[1])
        for _ in range(self.max_iter):
            w_grad, b_grad = self._loss_grad(X, y)
            self._weights -= w_grad * self.theta
            self._b -= self.theta * b_grad
            # Same as norm(w_i - w_{i-1})**2
            if np.linalg.norm(self.theta*w_grad)**2 <= self.tol:
                break
        self.coef_ = self._weights
        self.intercept_ = self._b

        return self

    def _loss_grad(self, X, y):
        # Gradient for max(0, 1-yWX)
        max_grad = (y*(np.dot(X, self._weights) + self._b) <= 1) * 1
        w_grad = self._weights / self.C - np.dot(max_grad * y, X)
        b_grad = -np.sum(max_grad*y)
        return w_grad, b_grad

    def _loss_func(self, X: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError

    def score(self, X, y, sample_weight=None):
        '''
            X - np.ndarray - [N, n_features] - матрица объектов-признаков
        Return:
            y - np.ndarray - [N, ] - вектор скоров
        '''
        check_is_fitted(self)
        X = check_array(X)
        return accuracy_score(self.predict(X), y)

    def predict(self, X):
        '''
            X - np.ndarray - [N, n_features] - матрица объектов-признаков
        Return:
            y - np.ndarray - [N, ] - вектор предсказаний {"-1", "1"}
        '''
        check_is_fitted(self)
        X: np.ndarray
        X = check_array(X)

        return np.sign(np.dot(X, self._weights) + self._b)
