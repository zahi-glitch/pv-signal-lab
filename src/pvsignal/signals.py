"""Disproportionality signal detection with a statistical null model.

Every (drug, event) pair gets a 2x2 table of report counts:

              event    no event
    drug        a          b
    no drug     c          d

PRR = [a/(a+b)] / [c/(c+d)],   ROR = ad/(bc),   SE(ln ROR) = sqrt(1/a+1/b+1/c+1/d).

Three detection rules are provided:
  * ``flag_evans``  : the common screening rule (PRR >= 2, a >= 3, chi2 >= 4).
  * ``flag_fdr``    : one-sided Wald test on ln(ROR) with Benjamini-Hochberg FDR control.
  * ``flag_fdr`` on the Mantel-Haenszel output of ``mh_adjusted``: the same test after
    stratifying (e.g. by age band), which removes confounding by the stratifier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse, stats


def incidence(drugs: pd.DataFrame, events: pd.DataFrame, n_reports: int):
    """Sparse report x drug and report x event indicator matrices plus the name arrays.

    Assumes report_id values are 0..n_reports-1 and (report_id, drug) / (report_id, event) are unique.
    """
    d_codes, d_names = pd.factorize(drugs["drug"], sort=True)
    e_codes, e_names = pd.factorize(events["event"], sort=True)
    ones_d = np.ones(len(drugs))
    ones_e = np.ones(len(events))
    D = sparse.coo_matrix(
        (ones_d, (drugs["report_id"].to_numpy(), d_codes)), shape=(n_reports, len(d_names))
    ).tocsr()
    E = sparse.coo_matrix(
        (ones_e, (events["report_id"].to_numpy(), e_codes)), shape=(n_reports, len(e_names))
    ).tocsr()
    return D, E, np.asarray(d_names), np.asarray(e_names)


def _cells(D, E):
    n = D.shape[0]
    a = (D.T @ E).toarray()
    n_d = np.asarray(D.sum(axis=0)).ravel()
    n_e = np.asarray(E.sum(axis=0)).ravel()
    b = n_d[:, None] - a
    c = n_e[None, :] - a
    d = n - a - b - c
    return a, b, c, d


def contingency(drugs: pd.DataFrame, events: pd.DataFrame, n_reports: int) -> pd.DataFrame:
    """One row per (drug, event) pair with the 2x2 counts a, b, c, d."""
    D, E, dn, en = incidence(drugs, events, n_reports)
    a, b, c, d = _cells(D, E)
    nd, ne = a.shape
    return pd.DataFrame(
        {
            "drug": np.repeat(dn, ne),
            "event": np.tile(en, nd),
            "a": a.ravel(),
            "b": b.ravel(),
            "c": c.ravel(),
            "d": d.ravel(),
        }
    )


def disproportionality(tab: pd.DataFrame) -> pd.DataFrame:
    """Add PRR, ROR, 95% lower bound, Wald z, one-sided p and chi-square to a contingency table."""
    t = tab.copy()
    cells = t[["a", "b", "c", "d"]].to_numpy(dtype=float)
    zero = (cells == 0).any(axis=1)
    cc = cells + 0.5 * zero[:, None]  # Haldane-Anscombe correction only where needed
    a, b, c, d = cc.T
    n = a + b + c + d
    t["prr"] = (a / (a + b)) / (c / (c + d))
    t["ror"] = a * d / (b * c)
    t["ln_ror"] = np.log(t["ror"])
    t["se"] = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    t["ror_lcl"] = np.exp(t["ln_ror"] - 1.96 * t["se"])
    t["z"] = t["ln_ror"] / t["se"]
    t["p"] = stats.norm.sf(t["z"])
    t["chi2"] = n * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    return t


def flag_evans(t: pd.DataFrame) -> pd.Series:
    return (t["prr"] >= 2) & (t["a"] >= 3) & (t["chi2"] >= 4)


def benjamini_hochberg(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Boolean mask of hypotheses rejected by the BH step-up procedure at level q."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    passed = p[order] <= q * (np.arange(1, m + 1) / m)
    if not passed.any():
        return np.zeros(m, dtype=bool)
    cutoff = p[order][np.nonzero(passed)[0].max()]
    return p <= cutoff


def flag_fdr(t: pd.DataFrame, q: float = 0.05, min_a: int = 3, p_col: str = "p") -> pd.Series:
    """BH on the one-sided p-values of pairs with at least ``min_a`` co-reports; effect must be > 1."""
    tested = (t["a"] >= min_a).to_numpy()
    out = np.zeros(len(t), dtype=bool)
    rej = benjamini_hochberg(t.loc[tested, p_col].to_numpy(), q)
    out[np.nonzero(tested)[0][rej]] = True
    return pd.Series(out, index=t.index) & (t["ror"] > 1)


def mh_adjusted(
    drugs: pd.DataFrame, events: pd.DataFrame, strata: np.ndarray, n_reports: int
) -> pd.DataFrame:
    """Mantel-Haenszel odds ratio per (drug, event) pair, stratified by ``strata``.

    ``strata`` is an integer array per report; a negative value excludes the report
    (used for complete-case analysis). Variance: Robins-Breslow-Greenland.
    Returned columns include ``a`` (total co-reports among included reports), ``ror``
    (the MH odds ratio, so the same flag functions apply) and ``p`` (one-sided).
    """
    D, E, dn, en = incidence(drugs, events, n_reports)
    R = S = PR = PS_QR = QS = 0.0
    a_tot = 0.0
    for s in np.unique(strata[strata >= 0]):
        idx = np.nonzero(strata == s)[0]
        a, b, c, d = _cells(D[idx], E[idx])
        n = float(len(idx))
        if n < 2:
            continue
        r, sc = a * d / n, b * c / n
        P, Q = (a + d) / n, (b + c) / n
        R = R + r
        S = S + sc
        PR = PR + P * r
        PS_QR = PS_QR + P * sc + Q * r
        QS = QS + Q * sc
        a_tot = a_tot + a
    with np.errstate(divide="ignore", invalid="ignore"):
        mh = R / S
        var = PR / (2 * R**2) + PS_QR / (2 * R * S) + QS / (2 * S**2)
        ln = np.log(mh)
        se = np.sqrt(var)
        z = ln / se
    nd, ne = mh.shape
    out = pd.DataFrame(
        {
            "drug": np.repeat(dn, ne),
            "event": np.tile(en, nd),
            "a": a_tot.ravel(),
            "ror": mh.ravel(),
            "ln_ror": ln.ravel(),
            "se": se.ravel(),
            "z": z.ravel(),
        }
    )
    out["p"] = stats.norm.sf(out["z"].to_numpy())
    out["p"] = out["p"].fillna(1.0)  # undefined (e.g. empty strata) -> not significant
    return out


def evaluate(t: pd.DataFrame, flagged: pd.Series, truth: pd.DataFrame) -> dict:
    """Compare flagged pairs with the simulation truth.

    sensitivity       : share of injected true signals flagged
    fa_confounded     : share of confounded (non-causal, age-driven) pairs flagged
    fa_null           : share of all other pairs flagged
    """
    kind = t[["drug", "event"]].merge(truth[["drug", "event", "kind"]], how="left", on=["drug", "event"])
    kind = kind["kind"].fillna("null").to_numpy()
    f = flagged.to_numpy()
    res = {}
    for name, label in [("sensitivity", "true"), ("fa_confounded", "confounded"), ("fa_null", "null")]:
        m = kind == label
        res[name] = float(f[m].mean()) if m.any() else float("nan")
    res["n_flagged"] = int(f.sum())
    return res
