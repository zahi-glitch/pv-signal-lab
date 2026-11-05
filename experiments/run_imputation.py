"""How missing ages and imputation choices change age-adjusted signal detection.

Age is the confounder. Remove 30% of ages under three mechanisms (MCAR, MAR, MNAR), then
detect signals with the age-stratified test after each strategy:

  oracle          : true ages (upper bound)
  crude           : ignore age, no adjustment (the naive baseline)
  complete_case   : drop reports with missing age
  mean, hotdeck, regression : impute, then stratify

Reports sensitivity, false alarms on confounded and null pairs, and age RMSE on the imputed entries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pvsignal.impute import age_strata, design_matrix, impute_age, make_missing
from pvsignal.signals import contingency, disproportionality, evaluate, flag_fdr, mh_adjusted
from pvsignal.simulate import SimConfig, simulate

OUT = Path(__file__).resolve().parent.parent / "results"
MECHS = ["MCAR", "MAR", "MNAR"]
IMPUTERS = ["mean", "hotdeck", "regression"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n", type=int, default=40_000)
    ap.add_argument("--frac", type=float, default=0.3)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rows = []
    for seed in range(args.seeds):
        sim = simulate(SimConfig(n_reports=args.n, seed=seed))
        n = len(sim.reports)
        age = sim.reports["age"].to_numpy()
        X = design_matrix(sim)
        rng = np.random.default_rng(1000 + seed)

        crude = disproportionality(contingency(sim.drugs, sim.events, n))
        rows.append({"mechanism": "-", "method": "crude", "seed": seed, "age_rmse": np.nan, **evaluate(crude, flag_fdr(crude), sim.truth)})
        mh = mh_adjusted(sim.drugs, sim.events, age_strata(age), n)
        rows.append({"mechanism": "-", "method": "oracle", "seed": seed, "age_rmse": 0.0, **evaluate(mh, flag_fdr(mh), sim.truth)})

        for mech in MECHS:
            miss = make_missing(sim, mech, args.frac, rng)
            obs = np.where(miss, np.nan, age)
            # complete case: exclude reports with missing age
            strata = np.where(miss, -1, age_strata(age))
            t = mh_adjusted(sim.drugs, sim.events, strata, n)
            rows.append({"mechanism": mech, "method": "complete_case", "seed": seed, "age_rmse": np.nan, **evaluate(t, flag_fdr(t), sim.truth)})
            for m in IMPUTERS:
                filled = impute_age(m, obs, X, rng)
                t = mh_adjusted(sim.drugs, sim.events, age_strata(filled), n)
                rmse = float(np.sqrt(np.mean((filled[miss] - age[miss]) ** 2)))
                rows.append({"mechanism": mech, "method": m, "seed": seed, "age_rmse": rmse, **evaluate(t, flag_fdr(t), sim.truth)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "imputation_raw.csv", index=False)
    agg = df.groupby(["mechanism", "method"]).agg(
        sensitivity=("sensitivity", "mean"),
        fa_confounded=("fa_confounded", "mean"),
        fa_null=("fa_null", "mean"),
        age_rmse=("age_rmse", "mean"),
    )
    order = ["crude", "oracle", "complete_case", "mean", "hotdeck", "regression"]
    agg = agg.reset_index()
    agg["_o"] = agg["method"].map({m: i for i, m in enumerate(order)})
    agg["_m"] = agg["mechanism"].map({"-": 0, "MCAR": 1, "MAR": 2, "MNAR": 3})
    agg = agg.sort_values(["_m", "_o"]).drop(columns=["_o", "_m"])
    agg.to_csv(OUT / "imputation.csv", index=False)
    with open(OUT / "imputation.md", "w") as f:
        f.write(f"Mean over {args.seeds} simulated databases, {int(args.frac * 100)}% of ages removed.\n\n")
        f.write(agg.round(3).to_string(index=False))
        f.write("\n")
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
