"""Part 2, Task B: an sklearn-style estimator and model factories."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from .transform import build_preprocessor


class GDLinearRegression(BaseEstimator, RegressorMixin):
    """Linear regression trained by batch gradient descent.

    Model  y_hat = X @ w + b
    Loss   J = (1 / 2m) * sum((y_hat - y)^2)
    Grads  dJ/dw = X.T @ err / m,   dJ/db = mean(err)
    Step   w -= lr * dJ/dw,          b -= lr * dJ/db

    fit(X, y) sets: coef_ (array), intercept_ (float), loss_history_ (list
    of the loss at every iteration) and returns self.
    """

    def __init__(self, lr: float = 0.1, n_iter: int = 1000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        m = X.shape[0]
        w, b = np.zeros(X.shape[1]), 0.0
        self.loss_history_ = []
        for _ in range(self.n_iter):
            err = X @ w + b - y                       # residuals, shape (m,)
            self.loss_history_.append(float((err ** 2).sum() / (2 * m)))
            w = w - self.lr * (X.T @ err / m)         # vectorized gradient step
            b = b - self.lr * err.mean()
        self.coef_, self.intercept_ = w, float(b)
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


def make_regressor(kind: str = "linear", degree: int = 2, alpha: float = 1.0):
    """Return Pipeline([('prep', build_preprocessor(...)), ('model', estimator)]).

    kinds: 'dummy' (mean baseline), 'linear', 'ridge', 'poly' (Ridge on
    polynomial features of the numeric columns), 'gd' (our own estimator).
    Unknown kind: raise ValueError.
    """
    estimators = {
        "dummy": DummyRegressor(strategy="mean"),
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=alpha),
        "poly": Ridge(alpha=alpha),
        "gd": GDLinearRegression(lr=0.05, n_iter=2000),
    }
    if kind not in estimators:
        raise ValueError(f"unknown regressor {kind!r}; choose from {list(estimators)}")
    prep = build_preprocessor(degree=degree if kind == "poly" else 1)
    return Pipeline([("prep", prep), ("model", estimators[kind])])


def make_classifier(kind: str = "logistic", C: float = 1.0, class_weight=None):
    """Return Pipeline([('prep', ...), ('model', estimator)]).

    kinds: 'dummy' (majority class, still needs predict_proba) and 'logistic'.
    """
    estimators = {
        # DummyClassifier has predict_proba: it returns the class frequencies
        # from training, the same for every row.
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(C=C, class_weight=class_weight, max_iter=2000),
    }
    if kind not in estimators:
        raise ValueError(f"unknown classifier {kind!r}; choose from {list(estimators)}")
    return Pipeline([("prep", build_preprocessor()), ("model", estimators[kind])])
