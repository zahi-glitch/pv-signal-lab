import numpy as np
import pandas as pd

from pvsignal.db import counts_from_sql, load_sqlite
from pvsignal.impute import design_matrix, impute_age, make_missing
from pvsignal.signals import contingency
from pvsignal.simulate import SimConfig, simulate

SMALL = SimConfig(n_reports=4000, n_drugs=10, n_events=12, n_true_signals=3, n_age_events=2, n_elderly_drugs=2, seed=7)


def test_simulation_is_reproducible_and_well_formed():
    a, b = simulate(SMALL), simulate(SMALL)
    pd.testing.assert_frame_equal(a.reports, b.reports)
    pd.testing.assert_frame_equal(a.events, b.events)
    n = len(a.reports)
    assert (a.reports["report_id"].to_numpy() == np.arange(n)).all()
    assert a.events["report_id"].between(0, n - 1).all()
    assert not a.drugs.duplicated().any() and not a.events.duplicated().any()
    assert set(a.events["report_id"]) == set(range(n))  # every kept report has an event
    assert (a.truth["kind"] == "true").sum() == 3


def test_injected_signal_raises_event_rate():
    sim = simulate(SimConfig(n_reports=40_000, seed=11))
    row = sim.truth[sim.truth.kind == "true"].iloc[0]
    has_drug = sim.drugs.loc[sim.drugs.drug == row.drug, "report_id"]
    has_event = set(sim.events.loc[sim.events.event == row.event, "report_id"])
    with_drug = np.mean([r in has_event for r in has_drug])
    all_ids = np.arange(len(sim.reports))
    no_drug = np.setdiff1d(all_ids, has_drug.to_numpy())
    without = np.mean([r in has_event for r in no_drug])
    assert with_drug > 1.5 * without


def test_sql_counts_equal_numpy_counts():
    sim = simulate(SMALL)
    con = load_sqlite(sim)
    sql = counts_from_sql(con)
    num = contingency(sim.drugs, sim.events, len(sim.reports))
    m = sql.merge(num, on=["drug", "event"], suffixes=("_sql", "_np"))
    assert len(m) == len(sql) > 0
    for c in "abcd":
        assert (m[f"{c}_sql"] == m[f"{c}_np"]).all()


def test_missing_mechanisms_hit_the_target_fraction():
    sim = simulate(SimConfig(n_reports=20_000, seed=2))
    rng = np.random.default_rng(0)
    for mech in ["MCAR", "MAR", "MNAR"]:
        miss = make_missing(sim, mech, 0.3, rng)
        assert abs(miss.mean() - 0.3) < 0.02
    age = sim.reports["age"].to_numpy()
    miss = make_missing(sim, "MNAR", 0.3, rng)
    assert age[miss].mean() > age[~miss].mean() + 5  # older patients go missing more often


def test_imputers_fill_everything_and_regression_beats_mean_on_rmse():
    sim = simulate(SimConfig(n_reports=20_000, seed=4))
    rng = np.random.default_rng(1)
    age = sim.reports["age"].to_numpy()
    miss = make_missing(sim, "MCAR", 0.3, rng)
    obs = np.where(miss, np.nan, age)
    X = design_matrix(sim)
    rmse = {}
    for m in ["mean", "hotdeck", "regression"]:
        filled = impute_age(m, obs, X, rng)
        assert not np.isnan(filled).any()
        assert np.allclose(filled[~miss], age[~miss])
        rmse[m] = np.sqrt(np.mean((filled[miss] - age[miss]) ** 2))
    # The mean is the best point predictor (lowest RMSE); both random-draw imputers add noise on purpose,
    # hot-deck the most. RMSE is not the criterion that matters for confounder adjustment (see README).
    assert rmse["hotdeck"] > rmse["regression"] > rmse["mean"] > 0
