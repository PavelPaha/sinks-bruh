from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.optimize import minimize

from .metrics import roc_auc


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def zscore_fit(x: np.ndarray) -> Dict[str, float]:
    mu = float(np.mean(x))
    sd = float(np.std(x) + 1e-12)
    return {"mean": mu, "std": sd}


def zscore_apply(x: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    return (x - stats["mean"]) / (stats["std"] + 1e-12)


@dataclass(frozen=True)
class Fit:
    w: np.ndarray
    b: float


def fit_logistic(X: np.ndarray, y: np.ndarray) -> Fit:
    # X: [n, d], y: {0,1}
    n, d = X.shape

    def obj(theta: np.ndarray) -> float:
        w = theta[:d]
        b = theta[d]
        p = sigmoid(X @ w + b)
        return logloss(y, p)

    theta0 = np.zeros(d + 1, dtype=float)
    res = minimize(obj, theta0, method="BFGS")
    theta = res.x
    return Fit(w=theta[:d], b=float(theta[d]))


def predict_proba(X: np.ndarray, fit: Fit) -> np.ndarray:
    return sigmoid(X @ fit.w + fit.b)


def eval_probs(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {"logloss": logloss(y, p), "auroc": roc_auc(y, p)}

