"""Loaders for the public openFDA API (adverse-event reports and drug labels).

NOT exercised by the test suite or the experiments in this repository, which use
synthetic data so that the truth is known. Use these to run the same pipeline on real
reports: ``load_sqlite(fetch_event_reports(...))`` and ``signals.contingency(...)``.

Caveats that matter for the analysis:
  * Spontaneous reports are voluntary and unverified. A flagged pair is a hypothesis
    for review, not evidence that the drug causes the event.
  * Counts are not incidence: there is no denominator of exposed patients.
  * Reports can be duplicated and drug names are free text; normalise before serious use.
  * Check the openFDA terms of service and rate limits (no key: roughly 240 requests/minute;
    ``skip`` is limited to 25,000, so use date windows in ``search`` for larger pulls).
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

import pandas as pd

from .rag import Passage

BASE = "https://api.fda.gov/drug"
AGE_UNIT_TO_YEARS = {"800": 10.0, "801": 1.0, "802": 1 / 12, "803": 1 / 52, "804": 1 / 365, "805": 1 / 8760}
LABEL_SECTIONS = [
    "indications_and_usage",
    "dosage_and_administration",
    "contraindications",
    "warnings_and_cautions",
    "warnings",
    "adverse_reactions",
    "use_in_specific_populations",
]


@dataclass
class ReportTables:
    reports: pd.DataFrame
    drugs: pd.DataFrame
    events: pd.DataFrame


def _get(endpoint: str, params: dict) -> dict:
    url = f"{BASE}/{endpoint}.json?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def parse_event_results(results: list[dict], suspect_only: bool = True) -> ReportTables:
    """Turn openFDA drug/event results into the three-table schema used everywhere else."""
    rep, dr, ev = [], [], []
    for i, r in enumerate(results):
        p = r.get("patient", {})
        age = None
        if p.get("patientonsetage"):
            try:
                age = float(p["patientonsetage"]) * AGE_UNIT_TO_YEARS.get(str(p.get("patientonsetageunit", "801")), 1.0)
            except ValueError:
                age = None
        sex = {"1": "M", "2": "F"}.get(str(p.get("patientsex", "")), None)
        rep.append({"report_id": i, "age": age, "sex": sex})
        for d in p.get("drug", []):
            if suspect_only and str(d.get("drugcharacterization")) != "1":
                continue
            name = d.get("medicinalproduct")
            if name:
                dr.append({"report_id": i, "drug": name.strip().upper()})
        for e in p.get("reaction", []):
            if e.get("reactionmeddrapt"):
                ev.append({"report_id": i, "event": e["reactionmeddrapt"].strip().upper()})
    reports = pd.DataFrame(rep, columns=["report_id", "age", "sex"])
    drugs = pd.DataFrame(dr, columns=["report_id", "drug"]).drop_duplicates()
    events = pd.DataFrame(ev, columns=["report_id", "event"]).drop_duplicates()
    return ReportTables(reports, drugs, events)


def fetch_event_reports(search: str | None = None, n: int = 5000, page: int = 1000, pause: float = 0.3) -> ReportTables:
    results: list[dict] = []
    skip = 0
    while len(results) < n:
        params = {"limit": min(page, n - len(results)), "skip": skip}
        if search:
            params["search"] = search
        batch = _get("event", params).get("results", [])
        if not batch:
            break
        results += batch
        skip += len(batch)
        time.sleep(pause)
    return parse_event_results(results)


def fetch_labels(generic_names: list[str], max_chars: int = 900) -> list[Passage]:
    """Drug-label sections split into passages of at most ``max_chars`` characters."""
    out: list[Passage] = []
    for name in generic_names:
        res = _get("label", {"search": f'openfda.generic_name:"{name}"', "limit": 1}).get("results", [])
        if not res:
            continue
        label = res[0]
        for sec in LABEL_SECTIONS:
            text = " ".join(label.get(sec, []))
            for j in range(0, len(text), max_chars):
                chunk = text[j : j + max_chars].strip()
                if chunk:
                    out.append(Passage(f"{name}:{sec}:{j // max_chars}", name, sec.replace("_", " "), chunk))
    return out
