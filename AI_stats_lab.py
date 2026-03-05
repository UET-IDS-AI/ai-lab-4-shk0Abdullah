"""
AI_stats_lab.py

Autograded lab: Gradient Descent + Linear Regression (Diabetes)

You must implement the TODO functions below.
Do not change function names or return signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import datasets

# =========================
# Helpers (you may use these)
# =========================

#  theta0 bias 
#  weight slope theta1

def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Add a bias (intercept) column of ones to X."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return np.hstack([np.ones((X.shape[0], 1)), X])


def standardize_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize using train statistics only.
    Returns: X_train_std, X_test_std, mean, std
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


@dataclass
class GDResult:
    theta: np.ndarray              # (d, )
    losses: np.ndarray             # (T, )
    thetas: np.ndarray             # (T, d) trajectory


# =========================
# Q1: Gradient descent + visualization data
# =========================

def gradient_descent_linreg(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 200,
    theta0: Optional[np.ndarray] = None,
) -> GDResult:

    m, d = X.shape

    if theta0 is None:
        theta = np.zeros(d)
    else:
        theta = theta0.copy()

    losses = []
    thetas = []

    for _ in range(epochs):

        # prediction
        y_pred = X @ theta

        # error
        error = y_pred - y

        # loss (MSE)
        loss = np.mean(error ** 2)
        losses.append(loss)

        # gradient
        gradient = (X.T @ error) / m

        # update
        theta = theta - lr * gradient

        # store theta path
        thetas.append(theta.copy())

    return GDResult(
        theta=theta,
        losses=np.array(losses),
        thetas=np.array(thetas)
    )


def visualize_gradient_descent(
    lr: float = 0.1,
    epochs: int = 60,
    seed: int = 0,
) -> Dict[str, np.ndarray]:

    np.random.seed(seed)

    n = 50
    X_feature = np.random.randn(n, 1)

    true_theta0 = 2
    true_theta1 = 3

    noise = 0.5 * np.random.randn(n)

    y = true_theta0 + true_theta1 * X_feature[:, 0] + noise

    X = add_bias_column(X_feature)

    result = gradient_descent_linreg(X, y, lr=lr, epochs=epochs)

    return {
        "theta_path": result.thetas[:, :2],
        "losses": result.losses,
        "X": X,
        "y": y,
    }

# =========================
# Q2: Diabetes regression using gradient descent
# =========================

from sklearn.datasets import load_diabetes


def diabetes_linear_gd(
    lr: float = 0.05,
    epochs: int = 2000,
    test_size: float = 0.2,
    seed: int = 0,
):

    np.random.seed(seed)

    data = load_diabetes()
    X = data.data
    y = data.target

    m = X.shape[0]

    idx = np.random.permutation(m)
    split = int(m * (1 - test_size))

    train_idx = idx[:split]
    test_idx = idx[split:]

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    X_train, X_test, _, _ = standardize_train_test(X_train, X_test)

    X_train = add_bias_column(X_train)
    X_test = add_bias_column(X_test)

    result = gradient_descent_linreg(X_train, y_train, lr=lr, epochs=epochs)

    theta = result.theta

    train_pred = X_train @ theta
    test_pred = X_test @ theta

    train_mse = mse(y_train, train_pred)
    test_mse = mse(y_test, test_pred)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q3: Diabetes regression using analytical solution
# =========================

def diabetes_linear_analytical(
    ridge_lambda: float = 1e-8,
    test_size: float = 0.2,
    seed: int = 0,
):

    np.random.seed(seed)

    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    X = data.data
    y = data.target

    m = X.shape[0]

    idx = np.random.permutation(m)
    split = int(m * (1 - test_size))

    train_idx = idx[:split]
    test_idx = idx[split:]

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    X_train, X_test, _, _ = standardize_train_test(X_train, X_test)

    X_train = add_bias_column(X_train)
    X_test = add_bias_column(X_test)

    d = X_train.shape[1]

    I = np.eye(d)

    theta = np.linalg.inv(X_train.T @ X_train + ridge_lambda * I) @ (X_train.T @ y_train)

    train_pred = X_train @ theta
    test_pred = X_test @ theta

    train_mse = mse(y_train, train_pred)
    test_mse = mse(y_test, test_pred)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta

# =========================
# Q4: Compare GD vs analytical
# =========================

def diabetes_compare_gd_vs_analytical(
    lr: float = 0.05,
    epochs: int = 4000,
    test_size: float = 0.2,
    seed: int = 0,
):

    gd_train_mse, gd_test_mse, gd_train_r2, gd_test_r2, theta_gd = diabetes_linear_gd(
        lr, epochs, test_size, seed
    )

    an_train_mse, an_test_mse, an_train_r2, an_test_r2, theta_an = diabetes_linear_analytical(
        1e-8, test_size, seed
    )

    theta_l2_diff = np.linalg.norm(theta_gd - theta_an)

    theta_cosine_sim = np.dot(theta_gd, theta_an) / (
        np.linalg.norm(theta_gd) * np.linalg.norm(theta_an)
    )
    # Changing to rerun the yml
    return {
        "theta_l2_diff": float(theta_l2_diff),
        "train_mse_diff": float(abs(gd_train_mse - an_train_mse)),
        "test_mse_diff": float(abs(gd_test_mse - an_test_mse)),
        "train_r2_diff": float(abs(gd_train_r2 - an_train_r2)),
        "test_r2_diff": float(abs(gd_test_r2 - an_test_r2)),
        "theta_cosine_sim": float(theta_cosine_sim),
    }
