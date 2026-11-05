"""Retrieval quality and the abstention trade-off of the label assistant on the example set."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pvsignal.rag import Retriever, evaluate_rag, load_jsonl, load_passages

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    retriever = Retriever(load_passages(ROOT / "data" / "example_labels.jsonl"))
    qa = load_jsonl(ROOT / "data" / "example_qa.jsonl")
    df = pd.DataFrame(evaluate_rag(retriever, qa, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "rag_sweep.csv", index=False)
    print(df.round(2).to_string(index=False))

    print("\nPer question (confidence of the top passage; answered at tau = 0.4?):")
    for x in qa:
        top = retriever.search(x["q"], 1)[0]
        c = retriever.confidence(x["q"], top)
        print(f"  {c:4.2f}  {'answer ' if c >= 0.4 else 'abstain'}  gold={str(x['gold']):10s} top={top.pid:9s} {x['q']}")


if __name__ == "__main__":
    main()
