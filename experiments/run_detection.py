"""Detection power and false-alarm rates on synthetic reports with known truth.

For each database size, simulate many databases and score three rules:
  evans   : PRR >= 2, a >= 3, chi2 >= 4 on the crude 2x2 tables
  ror_fdr : one-sided Wald test on the crude ln(ROR), Benjamini-Hochberg q = 0.05
  mh_fdr  : the same test on the Mantel-Haenszel OR stratified by age band

Writes results/detection.csv, results/detection.md (and results/detection.png if matplotlib is installed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pvsignal.impute import age_strata
from pvsignal.signals import contingency, disproportionality, evaluate, flag_evans, flag_fdr, mh_adjusted
from pvsignal.simulate import SimConfig, simulate

OUT = Path(__file__).resolve().parent.parent / "results"


def score_all(sim) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    n = len(sim.reports)
    crude = disproportionality(contingency(sim.drugs, sim.events, n))
    mh = mh_adjusted(sim.drugs, sim.events, age_strata(sim.reports["age"].to_numpy()), n)
    return {
        "evans": (crude, flag_evans(crude)),
        "ror_fdr": (crude, flag_fdr(crude)),
        "mh_fdr": (mh, flag_fdr(mh)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--sizes", type=int, nargs="+", default=[5_000, 10_000, 20_000, 40_000])
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    rows, rr_rows = [], []
    for size in args.sizes:
        for seed in range(args.seeds):
            sim = simulate(SimConfig(n_reports=size, seed=seed))
            for method, (t, flag) in score_all(sim).items():
                rows.append({"n_simulated": size, "n_reports": len(sim.reports), "seed": seed, "method": method, **evaluate(t, flag, sim.truth)})
                if size == 20_000:  # sensitivity by relative risk of the injected signal
                    f = t[["drug", "event"]].assign(flag=flag.to_numpy()).merge(sim.truth[sim.truth.kind == "true"], on=["drug", "event"])
                    f["rr_bin"] = pd.cut(f["rr"], [2, 3, 4, 5], include_lowest=True).astype(str)
                    rr_rows += [{"method": method, "rr_bin": r.rr_bin, "flagged": bool(r.flag)} for r in f.itertuples()]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "detection_raw.csv", index=False)
    agg = (
        df.groupby(["n_simulated", "method"])
        .agg(reports=("n_reports", "mean"), sensitivity=("sensitivity", "mean"), fa_confounded=("fa_confounded", "mean"), fa_null=("fa_null", "mean"))
        .reset_index()
    )
    agg["reports"] = agg["reports"].round(0).astype(int)
    agg.to_csv(OUT / "detection.csv", index=False)
    rr = pd.DataFrame(rr_rows).groupby(["method", "rr_bin"])["flagged"].mean().unstack("method").round(2)

    with open(OUT / "detection.md", "w") as f:
        f.write(f"Mean over {args.seeds} simulated databases per size. Reports = reports kept after dropping those with no event.\n\n")
        f.write(agg.round(3).to_markdown(index=False) if hasattr(agg, "to_markdown") and _has_tabulate() else agg.round(3).to_string(index=False))
        f.write("\n\nSensitivity by relative risk of the injected signal (20,000 simulated, about 12,000 reports kept):\n\n")
        f.write(rr.to_string())
        f.write("\n")
    print(agg.round(3).to_string(index=False))
    print(rr.to_string())

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
        for m, g in agg.groupby("method"):
            ax[0].plot(g["reports"], g["sensitivity"], marker="o", label=m)
            ax[1].plot(g["reports"], g["fa_confounded"], marker="o", label=m)
        ax[0].set(title="Injected signals found", xlabel="reports", ylabel="sensitivity", ylim=(0, 1.02))
        ax[1].set(title="Confounded (no-effect) pairs flagged", xlabel="reports", ylabel="false-alarm rate", ylim=(0, 1.02))
        ax[0].legend(frameon=False)
        for a in ax:
            a.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / "detection.png", dpi=150)
    except ImportError:
        pass


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401

        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
