"""Synthetic spontaneous-report data with known ground truth.

Why simulate: on real reporting data nobody knows which drug-event pairs are true
signals, so detection power and false-alarm rates cannot be measured. Here they can.

Model (one report = one patient):
  * age ~ Normal(55, 18) clipped to [0, 100]; sex uniform.
  * Each report lists 1-4 drugs. Drug popularity is Zipf-like. A few "elderly" drugs
    are used more by older patients (exp(age_beta * z), z = standardised age).
  * Each event has a baseline probability. A few "age" events are more frequent in
    older patients (same age effect). This creates CONFOUNDED pairs: an elderly drug
    and an age event co-occur more often, with no causal link.
  * "True" signals are drug-event pairs whose event probability is multiplied by a
    relative risk (rr) when the drug is on the report.
  * Reports with no event are dropped (a report needs at least one reaction), which
    slightly distorts the marginals, as in real databases.

Ground truth: ``truth`` has kind "true" (injected effect) or "confounded" (no effect,
shared dependence on age). Every other pair is a plain null. For true signals the column
``affects`` says who is affected: "both" sexes or only "F" or "M" (effect modification,
switched on with ``n_sex_specific``; off by default so earlier results are unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimConfig:
    n_reports: int = 40_000
    n_drugs: int = 30
    n_events: int = 40
    n_true_signals: int = 8
    rr_range: tuple[float, float] = (2.0, 5.0)
    n_age_events: int = 6
    n_elderly_drugs: int = 6
    age_beta: float = 1.0
    n_sex_specific: int = 0  # how many true signals act in one sex only (effect modification); 0 = off
    seed: int = 0


@dataclass
class SimData:
    reports: pd.DataFrame  # report_id, age, sex
    drugs: pd.DataFrame  # report_id, drug (unique pairs)
    events: pd.DataFrame  # report_id, event (unique pairs)
    truth: pd.DataFrame  # drug, event, kind, rr
    config: SimConfig


def simulate(config: SimConfig | None = None) -> SimData:
    cfg = config or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    n, nd, ne = cfg.n_reports, cfg.n_drugs, cfg.n_events

    age = np.clip(rng.normal(55, 18, n), 0, 100)
    z = (age - 55) / 18
    sex = rng.choice(np.array(["F", "M"]), n)

    # --- drugs: weighted sampling without replacement (Gumbel top-k) ---
    pop = 1.0 / np.arange(1, nd + 1) ** 0.8
    rng.shuffle(pop)
    elderly = rng.choice(nd, cfg.n_elderly_drugs, replace=False)
    logw = np.tile(np.log(pop), (n, 1))
    logw[:, elderly] += cfg.age_beta * z[:, None]
    keys = logw + rng.gumbel(size=(n, nd))
    k = np.minimum(1 + rng.poisson(1.0, n), 4)
    ranks = np.argsort(np.argsort(-keys, axis=1), axis=1)
    present = ranks < k[:, None]

    # --- events ---
    p = np.exp(rng.uniform(np.log(0.004), np.log(0.06), ne))
    age_events = rng.choice(ne, cfg.n_age_events, replace=False)
    P = np.tile(p, (n, 1))
    P[:, age_events] *= np.exp(cfg.age_beta * z)[:, None]

    clean_drugs = np.setdiff1d(np.arange(nd), elderly)
    clean_events = np.setdiff1d(np.arange(ne), age_events)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < cfg.n_true_signals:
        pairs.add((int(rng.choice(clean_drugs)), int(rng.choice(clean_events))))
    true_rows = []
    n_ss = min(cfg.n_sex_specific, cfg.n_true_signals)
    for k, (d, e) in enumerate(sorted(pairs)):
        rr = float(rng.uniform(*cfg.rr_range))
        affects, mask = "both", present[:, d]
        if k < n_ss:  # effect modification: the relative risk applies to one sex only
            affects = "F" if rng.random() < 0.5 else "M"
            mask = mask & (sex == affects)
        P[mask, e] *= rr
        true_rows.append((d, e, "true", rr, affects))
    for d in elderly:
        for e in age_events:
            true_rows.append((int(d), int(e), "confounded", 1.0, "-"))

    P = np.clip(P, 0, 0.9)
    Y = rng.random((n, ne)) < P
    keep = Y.any(axis=1)

    dn =np.array([f"D{i + 1:02d}" for i in range(nd)])
    en = np.array([f"E{i + 1:02d}" for i in range(ne)])
    ri, di = np.nonzero(present[keep])
    rj, ej = np.nonzero(Y[keep])
    reports = pd.DataFrame(
        {"report_id": np.arange(keep.sum()), "age": age[keep], "sex": sex[keep]}
    )
    drugs = pd.DataFrame({"report_id": ri, "drug": dn[di]})
    events = pd.DataFrame({"report_id": rj, "event": en[ej]})
    truth = pd.DataFrame(true_rows, columns=["drug_i", "event_i", "kind", "rr", "affects"])
    truth["drug"] = dn[truth["drug_i"].to_numpy()]
    truth["event"] = en[truth["event_i"].to_numpy()]
    truth = truth[["drug", "event", "kind", "rr", "affects"]]
    return SimData(reports, drugs, events, truth, cfg)
