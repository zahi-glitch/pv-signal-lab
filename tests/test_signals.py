import numpy as np
import pandas as pd
import pytest

from pvsignal.signals import (
    benjamini_hochberg,
    contingency,
    disproportionality,
    evaluate,
    flag_evans,
    flag_fdr,
    mh_adjusted,
)
from pvsignal.simulate import SimConfig, simulate


def test_hand_computed_2x2():
    tab = pd.DataFrame({"drug": ["X"], "event": ["Y"], "a": [20], "b": [80], "c": [100], "d": [1800]})
    t = disproportionality(tab)
    assert t["prr"].iloc[0] == pytest.approx((20 / 100) / (100 / 1900))
    assert t["ror"].iloc[0] == pytest.approx(20 * 1800 / (80 * 100))
    se = np.sqrt(1 / 20 + 1 / 80 + 1 / 100 + 1 / 1800)
    assert t["se"].iloc[0] == pytest.approx(se)
    assert t["ror_lcl"].iloc[0] == pytest.approx(np.exp(np.log(4.5) - 1.96 * se))
    assert bool(flag_evans(t).iloc[0])


def test_zero_cell_gets_continuity_correction():
    tab = pd.DataFrame({"drug": ["X"], "event": ["Y"], "a": [0], "b": [50], "c": [10], "d": [940]})
    t = disproportionality(tab)
    assert np.isfinite(t["ror"].iloc[0]) and np.isfinite(t["p"].iloc[0])


def test_contingency_matches_crosstab():
    sim = simulate(SimConfig(n_reports=3000, n_drugs=8, n_events=10, n_true_signals=2, n_age_events=2, n_elderly_drugs=2, seed=3))
    n = len(sim.reports)
    tab = contingency(sim.drugs, sim.events, n)
    merged = sim.drugs.merge(sim.events, on="report_id")
    ct = pd.crosstab(merged["drug"], merged["event"])
    row = tab[(tab.drug == ct.index[0]) & (tab.event == ct.columns[0])].iloc[0]
    assert row["a"] == ct.iloc[0, 0]
    assert row[["a", "b", "c", "d"]].sum() == n


def test_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    rej = benjamini_hochberg(p, q=0.05)
    assert rej.sum() == 2  # textbook example: thresholds 0.005, 0.010, 0.015, ...
    assert not benjamini_hochberg(np.array([0.5, 0.6])).any()


def test_stratification_removes_confounding():
    sim = simulate(SimConfig(n_reports=40_000, seed=5))
    n = len(sim.reports)
    crude = disproportionality(contingency(sim.drugs, sim.events, n))
    strata = np.digitize(sim.reports["age"].to_numpy(), [30, 45, 60, 75])
    adj = mh_adjusted(sim.drugs, sim.events, strata, n)
    e_crude = evaluate(crude, flag_fdr(crude), sim.truth)
    e_adj = evaluate(adj, flag_fdr(adj), sim.truth)
    assert e_crude["fa_confounded"] > 0.3
    assert e_adj["fa_confounded"] < 0.1
    assert e_adj["sensitivity"] >= 0.75


def test_excluded_reports_do_not_count():
    sim = simulate(SimConfig(n_reports=5000, n_drugs=8, n_events=10, n_true_signals=2, n_age_events=2, n_elderly_drugs=2, seed=1))
    n = len(sim.reports)
    all_in = mh_adjusted(sim.drugs, sim.events, np.zeros(n, dtype=int), n)
    half = np.where(np.arange(n) % 2 == 0, 0, -1)
    some = mh_adjusted(sim.drugs, sim.events, half, n)
    assert some["a"].sum() < all_in["a"].sum()
