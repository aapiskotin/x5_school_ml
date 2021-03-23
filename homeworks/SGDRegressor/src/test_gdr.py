import pytest
from regression import GDLinearRegression

from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error as mse
from numpy.testing import assert_array_almost_equal


def test_one():
    X, y = make_regression(random_state=42)

    gd_reg = GDLinearRegression().fit(X, y)
    sgd_reg = SGDRegressor(learning_rate='constant').fit(X, y)

    assert gd_reg.coef_.shape == sgd_reg.coef_.shape
    assert gd_reg.intercept_.shape == sgd_reg.intercept_.shape
    assert mse(y, gd_reg.predict(X)) <= mse(y, sgd_reg.predict(X))


def test_two():
    X, y = make_regression(
        n_samples=1000,
        n_features=30,
        n_informative=20,
        random_state=42
    )

    gd_reg = GDLinearRegression().fit(X, y)
    sgd_reg = SGDRegressor(learning_rate='constant').fit(X, y)

    assert_array_almost_equal(gd_reg.coef_, sgd_reg.coef_, decimal=0)
    assert_array_almost_equal(gd_reg.intercept_, sgd_reg.intercept_, decimal=0)


def test_three():
    X, y = make_regression(
        n_samples=1000,
        n_features=30,
        n_informative=20,
        random_state=42
    )
    params = dict(
        l1_ratio=0.3,
        alpha=0.0002,
        fit_intercept=True,
        max_iter=3000,
        tol=1e-2,
        eta0=0.02
    )
    gd_reg = GDLinearRegression(**params).fit(X, y)
    sgd_reg = SGDRegressor(learning_rate='constant', **params).fit(X, y)

    assert_array_almost_equal(gd_reg.coef_, sgd_reg.coef_, decimal=0)
    assert_array_almost_equal(gd_reg.intercept_, sgd_reg.intercept_, decimal=0)


def test_four():
    X, y = make_regression(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    params = dict(
        l1_ratio=0.1,
        alpha=0.0003,
        fit_intercept=False,
        max_iter=3000,
        tol=1e-2,
        eta0=0.01
    )
    gd_reg = GDLinearRegression(**params).fit(X, y)
    sgd_reg = SGDRegressor(learning_rate='constant', **params).fit(X, y)

    assert_array_almost_equal(gd_reg.coef_, sgd_reg.coef_, decimal=0)
