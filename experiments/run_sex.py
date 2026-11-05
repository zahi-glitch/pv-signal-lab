"""Experiment 4: sex as an effect modifier.

Half of the true signals act in one sex only. Compare the pooled analysis (ignores sex) with
sex-specific analysis (one table per sex, one joint FDR correction) plus a heterogeneity test.
Age confounding is switched off so only the sex effect is studied.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pvsignal.signals import (
    contingency,
    disproportionality,
    flag_by_group,
    flag_fdr,
    heterogeneity,
    stratum_tables,
)
from pvsignal.simulate import SimConfig, simulate

OUT = Path(__file__).resolve().parents[1] / "results"
SIZES = [20_000, 40_000, 80_000]
SEEDS = range(20)
N_TRUE, N_SS = 8, 4


def one_run(n_reports: int, seed: int) -> pd.DataFrame:
    sim = simulate(
        SimConfig(
            n_reports=n_reports,
            n_true_signals=N_TRUE,
            n_sex_specific=N_SS,
            rr_range=(2.0, 5.0),
            n_elderly_drugs=0,
            n_age_events=0,
            seed=seed,
        )
    )
    n = len(sim.reports)
    pooled = disproportionality(contingency(sim.drugs, sim.events, n))
    pooled["pooled"] = flag_fdr(pooled).to_numpy()
    codes = np.where(sim.reports["sex"].to_numpy() == "F", 0, 1)
    tabs = stratum_tables(sim.drugs, sim.events, codes, {0: "F", 1: "M"}, n)
    wide = flag_by_group(tabs)
    het = heterogeneity(tabs["F"], tabs["M"])
    df = pooled[["drug", "event", "pooled"]].merge(wide, on=["drug", "event"]).merge(het, on=["drug", "event"])
    df = df.merge(sim.truth[["drug", "event", "kind", "affects"]], how="left", on=["drug", "event"])
    df["kind"] = df["kind"].fillna("null")
    df["affects"] = df["affects"].fillna("-")
    df["any_sex"] = df["F"] | df["M"]
    df["n_reports"], df["seed"] = n_reports, seed
    return df


def summarise(df: pd.DataFrame) -> dict:
    both = df[(df.kind == "true") & (df.affects == "both")]
    spec = df[(df.kind == "true") & df.affects.isin(["F", "M"])]
    null = df[df.kind == "null"]
    hit_spec = spec[spec.any_sex]
    hit_both = both[both.any_sex]
    right = [(r.F and not r.M) if r.affects == "F" else (r.M and not r.F) for r in spec.itertuples()]
    return {
        "sens_both_pooled": both["pooled"].mean(),
        "sens_both_by_sex": both["any_sex"].mean(),
        "sens_spec_pooled": spec["pooled"].mean(),
        "sens_spec_by_sex": spec["any_sex"].mean(),
        "correct_sex": float(np.mean(right)),
        "het_power": float((hit_spec["p_het"] < 0.05).mean()) if len(hit_spec) else float("nan"),
        "het_false": float((hit_both["p_het"] < 0.05).mean()) if len(hit_both) else float("nan"),
        "fa_null_pooled": null["pooled"].mean(),
        "fa_null_by_sex": null["any_sex"].mean(),
        "n_null_flag_by_sex_per_run": null["any_sex"].sum() / df["seed"].nunique(),
    }


def main() -> None:
    rows = []
    for n in SIZES:
        df = pd.concat([one_run(n, s) for s in SEEDS], ignore_index=True)
        rows.append({"n_reports": n, **summarise(df)})
        print("done", n, flush=True)
    res = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "sex.csv", index=False)

    pct = lambda x: f"{100 * x:.0f}%"
    lines = [
        "# Experiment 4: sex as an effect modifier",
        "",
        f"{len(SEEDS)} simulated databases per size; {N_TRUE} true signals per database, {N_SS} of them in one sex only "
        "(relative risk 2 to 5); no age confounding. Pooled = ignore sex; by sex = one table per sex, one joint "
        "Benjamini-Hochberg correction (q = 0.05) over both sexes' tests. Heterogeneity = Wald z-test that ln ROR differs "
        "between women and men (p < 0.05).",
        "",
        "| Reports | Found, both-sex signals: pooled | by sex | Found, one-sex signals: pooled | by sex | Correct sex named | Heterogeneity power | False heterogeneity | False alarms (null): pooled | by sex |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res.itertuples():
        lines.append(
            f"| {r.n_reports:,} | {pct(r.sens_both_pooled)} | {pct(r.sens_both_by_sex)} | {pct(r.sens_spec_pooled)} | "
            f"{pct(r.sens_spec_by_sex)} | {pct(r.correct_sex)} | {pct(r.het_power)} | {pct(r.het_false)} | "
            f"{100 * r.fa_null_pooled:.2f}% | {100 * r.fa_null_by_sex:.2f}% |"
        )
    lines += [
        "",
        "Reading: *Correct sex named* = flagged in the affected sex and not in the other. *Heterogeneity power* = share of "
        "flagged one-sex signals whose sex difference the test detects. *False heterogeneity* = same test on flagged "
        "both-sex signals (should be near 5%). Counts of true signals are small (4 + 4 per database), so percentages are "
        "pooled over databases and still noisy.",
    ]
    (OUT / "sex.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
