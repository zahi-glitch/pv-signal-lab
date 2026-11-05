"""SQLite layer: load reports into tables and compute the 2x2 counts in plain SQL.

The same counts come out of the vectorised NumPy path in ``signals.contingency``
(tested to be identical); the SQL version is what you would run against a warehouse.
Pairs with no co-report (a = 0) do not appear in the SQL result.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from .simulate import SimData

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (report_id INTEGER PRIMARY KEY, age REAL, sex TEXT);
CREATE TABLE IF NOT EXISTS drugs   (report_id INTEGER NOT NULL REFERENCES reports(report_id),
                                    drug TEXT NOT NULL, PRIMARY KEY (report_id, drug));
CREATE TABLE IF NOT EXISTS events  (report_id INTEGER NOT NULL REFERENCES reports(report_id),
                                    event TEXT NOT NULL, PRIMARY KEY (report_id, event));
CREATE INDEX IF NOT EXISTS idx_drugs_drug ON drugs(drug);
CREATE INDEX IF NOT EXISTS idx_events_event ON events(event);
"""

COUNTS_SQL = """
WITH n  AS (SELECT COUNT(*) AS N FROM reports),
     nd AS (SELECT drug,  COUNT(*) AS n_d FROM drugs  GROUP BY drug),
     ne AS (SELECT event, COUNT(*) AS n_e FROM events GROUP BY event),
     ab AS (SELECT d.drug, e.event, COUNT(*) AS a
            FROM drugs d JOIN events e ON e.report_id = d.report_id
            GROUP BY d.drug, e.event)
SELECT ab.drug, ab.event, ab.a,
       nd.n_d - ab.a                      AS b,
       ne.n_e - ab.a                      AS c,
       n.N - nd.n_d - ne.n_e + ab.a       AS d
FROM ab
JOIN nd ON nd.drug = ab.drug
JOIN ne ON ne.event = ab.event
CROSS JOIN n
ORDER BY ab.drug, ab.event;
"""


def load_sqlite(sim: SimData, path: str = ":memory:") -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    sim.reports.to_sql("reports", con, if_exists="append", index=False)
    sim.drugs.to_sql("drugs", con, if_exists="append", index=False)
    sim.events.to_sql("events", con, if_exists="append", index=False)
    con.commit()
    return con


def counts_from_sql(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(COUNTS_SQL, con)
