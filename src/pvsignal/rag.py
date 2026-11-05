"""A small label assistant: hybrid retrieval, cited answers, and an abstention gate.

Design choice: the gate sits BEFORE the language model. If the retrieved passage does not
cover the question, the system says "not in the sources" and never calls the model, so a
retrieval miss cannot turn into a confident hallucination.

Retrieval = BM25 and TF-IDF cosine, merged by reciprocal rank fusion.
Confidence = idf-weighted share of the question's content terms that appear in the top
passage. A question naming an unknown drug, or asking about a topic the label never
mentions, leaves heavy-idf terms uncovered and scores low.

The language model is optional and pluggable: ``answer(..., llm=callable)`` builds a prompt
from the retrieved passages and calls ``llm(prompt) -> str``. Without it, the answer is the
best-matching sentence of the top passage (extractive), with its citation.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

STOP = set(
    "a an and are as at be by can do does for from has have how i if in is it of on or that the "
    "this to was what when which who why will with you your me my not no yes there their they used "
    "use take taken".split()
)
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    out = []
    for w in TOKEN.findall(text.lower()):
        if w in STOP:
            continue
        for suf in ("ing", "ed", "es", "s"):  # crude stemmer, enough for a lexical baseline
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


@dataclass
class Passage:
    pid: str
    drug: str
    section: str
    text: str


@dataclass
class Answer:
    text: str
    citations: list[str]
    confidence: float
    abstained: bool


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_passages(path: str | Path) -> list[Passage]:
    return [Passage(**row) for row in load_jsonl(path)]


class Retriever:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        docs = [tokenize(f"{p.drug} {p.section} {p.text}") for p in passages]
        self.docs = docs
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / self.N
        df: dict[str, int] = {}
        for d in docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self.df = df
        self.tf = [{t: d.count(t) for t in set(d)} for d in docs]
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, sublinear_tf=True, token_pattern=None)
        self.matrix = self.tfidf.fit_transform([f"{p.drug} {p.section} {p.text}" for p in passages])

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def _bm25(self, q: list[str]) -> np.ndarray:
        scores = np.zeros(self.N)
        for i, (d, tf) in enumerate(zip(self.docs, self.tf)):
            for t in set(q):
                f = tf.get(t, 0)
                if f:
                    scores[i] += self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
        return scores

    def search(self, query: str, k: int = 3) -> list[Passage]:
        q = tokenize(query)
        r_bm25 = np.argsort(np.argsort(-self._bm25(q)))
        cos = (self.matrix @ self.tfidf.transform([query]).T).toarray().ravel()
        r_cos = np.argsort(np.argsort(-cos))
        fused = 1 / (60 + r_bm25 + 1) + 1 / (60 + r_cos + 1)
        return [self.passages[i] for i in np.argsort(-fused)[:k]]

    def confidence(self, query: str, passage: Passage) -> float:
        """idf-weighted coverage of the question's content terms by the passage (0..1)."""
        q = set(tokenize(query))
        if not q:
            return 0.0
        covered = set(tokenize(f"{passage.drug} {passage.section} {passage.text}"))
        # a term absent from the whole corpus gets the largest idf: it is maximally informative
        w = {t: (self.idf(t) if t in self.df else self.idf_max()) for t in q}
        return sum(w[t] for t in q if t in covered) / sum(w.values())

    def idf_max(self) -> float:
        return math.log(1 + (self.N + 0.5) / 0.5)


def _best_sentences(query: str, text: str, n: int = 1) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    q = set(tokenize(query))
    scored = sorted(sents, key=lambda s: -len(q & set(tokenize(s))))
    return " ".join(scored[:n])


def build_prompt(query: str, passages: list[Passage]) -> str:
    src = "\n".join(f"[{p.pid}] ({p.drug}, {p.section}) {p.text}" for p in passages)
    return (
        "Answer the question using ONLY the sources below. Cite source ids in square brackets. "
        "If the sources do not contain the answer, reply exactly: Not in the sources.\n\n"
        f"Sources:\n{src}\n\nQuestion: {query}\nAnswer:"
    )


def answer(query: str, retriever: Retriever, tau: float = 0.4, k: int = 3, llm=None) -> Answer:
    hits = retriever.search(query, k)
    conf = retriever.confidence(query, hits[0])
    if conf < tau:
        return Answer("Not in the sources.", [], conf, True)
    if llm is not None:
        return Answer(llm(build_prompt(query, hits)), [p.pid for p in hits], conf, False)
    return Answer(_best_sentences(query, hits[0].text), [hits[0].pid], conf, False)


def evaluate_rag(retriever: Retriever, qa: list[dict], taus: list[float], k: int = 3) -> list[dict]:
    """Retrieval quality on answerable questions, and the abstention trade-off across thresholds."""
    ans = [x for x in qa if x["gold"]]
    una = [x for x in qa if not x["gold"]]
    ranks = []
    for x in ans:
        pids = [p.pid for p in retriever.search(x["q"], len(retriever.passages))]
        ranks.append(pids.index(x["gold"]) + 1)
    ranks = np.array(ranks)
    base = {
        "recall@1": float((ranks <= 1).mean()),
        "recall@3": float((ranks <= 3).mean()),
        "mrr": float((1 / ranks).mean()),
    }
    rows = []
    for tau in taus:
        a_conf = [answer(x["q"], retriever, tau, k) for x in ans]
        u_conf = [answer(x["q"], retriever, tau, k) for x in una]
        right = [
            (not a.abstained) and a.citations[0] == x["gold"] for a, x in zip(a_conf, ans)
        ]
        rows.append(
            {
                "tau": tau,
                "answered_correctly": float(np.mean(right)),
                "wrongly_abstained": float(np.mean([a.abstained for a in a_conf])),
                "abstained_on_unanswerable": float(np.mean([a.abstained for a in u_conf])),
                "answered_unanswerable": float(np.mean([not a.abstained for a in u_conf])),
                **base,
            }
        )
    return rows
