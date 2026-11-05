from pvsignal.openfda import parse_event_results

RESULTS = [
    {
        "patient": {
            "patientonsetage": "60",
            "patientonsetageunit": "801",
            "patientsex": "2",
            "drug": [
                {"medicinalproduct": "drug a ", "drugcharacterization": "1"},
                {"medicinalproduct": "Drug B", "drugcharacterization": "2"},
            ],
            "reaction": [{"reactionmeddrapt": "Nausea"}, {"reactionmeddrapt": "nausea"}, {"reactionmeddrapt": "Rash"}],
        }
    },
    {"patient": {"patientonsetage": "6", "patientonsetageunit": "802", "patientsex": "1", "drug": [], "reaction": []}},
    {"patient": {"patientsex": "0", "drug": [{"medicinalproduct": "DRUG A", "drugcharacterization": "1"}], "reaction": [{"reactionmeddrapt": "RASH"}]}},
]


def test_parse_event_results():
    t = parse_event_results(RESULTS)
    assert t.reports.loc[0, "age"] == 60 and t.reports.loc[0, "sex"] == "F"
    assert t.reports.loc[1, "age"] == 0.5 and t.reports.loc[1, "sex"] == "M"  # 6 months
    assert t.reports.loc[2, "age"] is None or t.reports["age"].isna().iloc[2]
    assert set(t.drugs["drug"]) == {"DRUG A"}  # suspect drugs only, names normalised
    assert len(t.drugs) == 2
    assert len(t.events) == 3  # NAUSEA de-duplicated within a report
    assert len(parse_event_results(RESULTS, suspect_only=False).drugs) == 3
