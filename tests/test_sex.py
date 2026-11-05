import numpy as np
import pandas as pd
import pytest

from pvsignal.signals import contingency, flag_by_group, heterogeneity, stratum_tables, disproportionality
from pvsignal.simulate import SimConfig, simulate

NO_AGE = dict(n_elderly_drugs=0, n_age_events=0)


def test_default_simulation_unchanged_by_the_new_option():
    a = simulate(SimConfig(n_reports=5000, n_drugs=10, n_events=12, n_true_signals=3, n_age_events=2, n_elderly_drugs=2, seed=7))
    assert (a.truth.loc[a.truth.kind == "true", "affects"] == "both").all()
    assert (a.truth.loc[a.truth.kind == "confounded", "affects"] == "-").all()


def test_sex_specific_signal_acts_in_one_sex_only():
    sim = simulate(SimConfig(n_reports=80_000, n_true_signals=4, n_sex_specific=2, rr_range=(4.0, 5.0), seed=2, **NO_AGE))
    spec = sim.truth[sim.truth.affects.isin(["F", "M"])]
    assert len(spec) == 2
    sex = sim.reports.set_index("report_id")["sex"]
    for r in spec.itertuples():
        has_drug = set(sim.drugs.loc[sim.drugs.drug == r.drug, "report_id"])
        has_event = set(sim.events.loc[sim.events.event == r.event, "report_id"])

        def rate(with_drug, s):
            ids = [i for i in sex.index[sex == s] if (i in has_drug) == with_drug]
            return np.mean([i in has_event for i in ids])

        other = "M" if r.affects == "F" else "F"
        assert rate(True, r.affects) > 2.5 * rate(False, r.affects)  # strong effect in the affected sex
        assert rate(True, other) < 1.6 * rate(False, other)  # about no effect in the other sex


def test_stratum_tables_add_up_to_the_pooled_table():
    sim = simulate(SimConfig(n_reports=6000, n_drugs=8, n_events=10, n_true_signals=2, n_sex_specific=1, seed=4, **NO_AGE))
    n = len(sim.reports)
    codes = np.where(sim.reports["sex"].to_numpy() == "F", 0, 1)
    tabs = stratum_tables(sim.drugs, sim.events, codes, {0: "F", 1: "M"}, n)
    pooled = contingency(sim.drugs, sim.events, n)
    assert (tabs["F"][["drug", "event"]].values == pooled[["drug", "event"]].values).all()
    for c in "abcd":
        assert (tabs["F"][c].to_numpy() + tabs["M"][c].to_numpy() == pooled[c].to_numpy()).all()


def test_heterogeneity_by_hand():
    f = disproportionality(pd.DataFrame({"drug": ["X"], "event": ["Y"], "a": [40], "b": [60], "c": [100], "d": [1800]}))
    m = disproportionality(pd.DataFrame({"drug": ["X"], "event": ["Y"], "a": [20], "b": [80], "c": [100], "d": [1800]}))
    h = heterogeneity(f, m).iloc[0]
    se1 = np.sqrt(1 / 40 + 1 / 60 + 1 / 100 + 1 / 1800)
    se2 = np.sqrt(1 / 20 + 1 / 80 + 1 / 100 + 1 / 1800)
    z = (np.log(12) - np.log(4.5)) / np.sqrt(se1**2 + se2**2)
    assert h["z_het"] == pytest.approx(z)
    assert h["p_het"] == pytest.approx(0.0, abs=1) and 0 < h["p_het"] < 0.1
    same = heterogeneity(f, f).iloc[0]
    assert same["z_het"] == pytest.approx(0.0) and same["p_het"] == pytest.approx(1.0)


def test_flag_by_group_uses_one_correction_and_returns_a_column_per_group():
    sim = simulate(SimConfig(n_reports=40_000, n_true_signals=4, n_sex_specific=2, rr_range=(4.0, 5.0), seed=6, **NO_AGE))
    n = len(sim.reports)
    codes = np.where(sim.reports["sex"].to_numpy() == "F", 0, 1)
    wide = flag_by_group(stratum_tables(sim.drugs, sim.events, codes, {0: "F", 1: "M"}, n))
    assert {"drug", "event", "F", "M"} <= set(wide.columns)
    m = wide.merge(sim.truth, on=["drug", "event"])
    for r in m[m.affects.isin(["F", "M"])].itertuples():
        assert getattr(r, r.affects)  # flagged in the affected sex
