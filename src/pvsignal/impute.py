"""Missing-age mechanisms and imputation methods, and their effect on adjusted signal detection.

Age is the confounder in the simulation, and it is often missing in real reports.
``make_missing`` removes ages under three mechanisms:

  MCAR : missing completely at random.
  MAR  : missingness depends on observed variables (sex, number of drugs).
  MNAR : missingness depends on the age itself (older patients more often missing).

Imputers return a filled age vector:
  mean        : replace by the observed mean.
  hotdeck     : replace by a random observed age.
  regression  : stochastic regression on sex, drug count and the drug and event indicators
                (the event and the drugs must be in the imputation model, otherwise the
                imputed age is uncorrelated with them and the confounding comes back).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .signals import incidence
from .simulate import SimData

AGE_BANDS = [30, 45, 60, 75]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _calibrate(score: np.ndarray, frac: float) -> np.ndarray:
    """Missing probability sigmoid(b + score) with b chosen (bisection) so the mean equals frac."""
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _sigmoid(mid + score).mean() < frac:
            lo = mid
        else:
            hi = mid
    return _sigmoid((lo + hi) / 2 + score)


def design_matrix(sim: SimData) -> np.ndarray:
    """Predictors available even when age is missing: sex, drug count, drug and event indicators."""
    n = len(sim.reports)
    D, E, _, _ = incidence(sim.drugs, sim.events, n)
    sex = (sim.reports["sex"].to_numpy() == "F").astype(float)[:, None]
    n_drugs = np.asarray(D.sum(axis=1))
    return np.hstack([sex, n_drugs, D.toarray(), E.toarray()])


def make_missing(sim: SimData, mechanism: str, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask: True where the age is set to missing."""
    n = len(sim.reports)
    age = sim.reports["age"].to_numpy()
    if mechanism == "MCAR":
        p = np.full(n, frac)
    elif mechanism == "MAR":
        X = design_matrix(sim)
        score = 1.2 * (X[:, 0] - X[:, 0].mean()) + 0.8 * (X[:, 1] - X[:, 1].mean())
        p = _calibrate(score, frac)
    elif mechanism == "MNAR":
        p = _calibrate(1.5 * (age - age.mean()) / age.std(), frac)
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    return rng.random(n) < p


def impute_age(method: str, age_obs: np.ndarray, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fill NaNs in ``age_obs``. ``X`` is the design matrix (needed for 'regression')."""
    miss = np.isnan(age_obs)
    out = age_obs.copy()
    obs = age_obs[~miss]
    if method == "mean":
        out[miss] = obs.mean()
    elif method == "hotdeck":
        out[miss] = rng.choice(obs, miss.sum())
    elif method == "regression":
        model = Ridge(alpha=1.0).fit(X[~miss], obs)
        resid_sd = np.std(obs - model.predict(X[~miss]))
        out[miss] = model.predict(X[miss]) + rng.normal(0, resid_sd, miss.sum())
    else:
        raise ValueError(f"unknown imputation method {method!r}")
    return np.clip(out, 0, 100)


def age_strata(age: np.ndarray) -> np.ndarray:
    return np.digitize(age, AGE_BANDS)
