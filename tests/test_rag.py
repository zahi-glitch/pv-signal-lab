from pathlib import Path

import pytest

from pvsignal.rag import Retriever, answer, build_prompt, evaluate_rag, load_jsonl, load_passages

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_passages(DATA / "example_labels.jsonl"))


def test_answers_a_covered_question_with_citation(retriever):
    a = answer("What is the starting dose of Norimex?", retriever)
    assert not a.abstained and a.citations == ["nor_dose"]
    assert "500 mg" in a.text


def test_abstains_on_unknown_drug_and_off_topic(retriever):
    for q in ["What are the side effects of Lumavex?", "Who won the football match yesterday?"]:
        a = answer(q, retriever)
        assert a.abstained and a.citations == [] and a.text == "Not in the sources."


def test_llm_is_not_called_when_abstaining(retriever):
    calls = []
    answer("What is the shelf life of Cardiolan tablets?", retriever, llm=lambda p: calls.append(p) or "x")
    assert calls == []
    answer("What is the starting dose of Norimex?", retriever, llm=lambda p: calls.append(p) or "x")
    assert len(calls) == 1 and "ONLY the sources" in calls[0] and "[nor_dose]" in calls[0]


def test_no_wrong_answers_on_example_set(retriever):
    qa = load_jsonl(DATA / "example_qa.jsonl")
    row = evaluate_rag(retriever, qa, [0.4])[0]
    assert row["answered_unanswerable"] == 0.0  # never answers a question the sources cannot answer
    assert row["answered_correctly"] + row["wrongly_abstained"] == pytest.approx(1.0)  # no confident wrong answers
    assert row["recall@3"] == 1.0


def test_prompt_lists_sources(retriever):
    hits = retriever.search("Norimex dose", 2)
    p = build_prompt("Norimex dose", hits)
    assert all(f"[{h.pid}]" in p for h in hits)
