# pv-signal-lab

Signal detection, missing-data handling and a source-grounded label assistant for spontaneous adverse-event reports (pharmacovigilance), built and tested on **synthetic data with known ground truth**.

## The question

Reporting databases such as FDA FAERS tell you that a drug and an event were reported together more often than expected. They cannot tell you which flagged pairs are real, because nobody knows the truth. This repo makes the truth known by simulation, and uses it to answer three practical questions:

1. **Detection.** How many true signals do standard screening rules find, and how many no-effect pairs do they flag, at different database sizes?
2. **Confounding and missing data.** Age drives both drug use and event frequency. How badly does that create false signals, does stratifying fix it, and what happens when age is missing and has to be imputed?
3. **Trusting a language-model assistant.** Can a label assistant refuse to answer when the sources do not cover the question, instead of guessing?

All numbers below are from synthetic data. They show how the methods behave under stated assumptions, not what any real drug does.

## Results

### 1. Detection, and the cost of ignoring age

Mean over 20 simulated databases per size. 8 injected true signals (relative risk 2 to 5), 36 "confounded" pairs (elderly drug x age-related event, no causal effect), the rest null pairs.

| Reports | Rule | Sensitivity | Confounded pairs flagged | Other null pairs flagged |
|---:|---|---:|---:|---:|
| 12,307 | Evans (PRR >= 2, a >= 3, chi2 >= 4) | 0.94 | 0.16 | 0.006 |
| 12,307 | ROR test, BH q = 0.05 (crude) | 0.91 | **0.65** | 0.003 |
| 12,307 | ROR test, BH q = 0.05, **age-stratified** (Mantel-Haenszel) | 0.88 | **0.004** | 0.001 |
| 24,130 | ROR test, crude | 0.98 | **0.83** | 0.003 |
| 24,130 | ROR test, age-stratified | 0.98 | **0.01** | 0.001 |

- A statistically valid test on crude counts gets *worse* with more data: more reports make the age-driven association significant. Evans' rule happens to flag fewer of these because it also demands PRR >= 2, which is a rule of thumb, not error control.
- Stratifying by age band removes almost all confounded false alarms and costs some sensitivity in small databases (0.61 vs 0.68 at about 3,000 reports).
- Weak signals (relative risk 2 to 3) are the ones missed: sensitivity about 0.65 to 0.8 at about 12,000 reports, against 0.97 to 1.0 for relative risk 3 and above.

Full table: [`results/detection.md`](results/detection.md), figure: [`results/detection.png`](results/detection.png).

### 2. Missing age and imputation

30% of ages removed (10 simulated databases, about 24,000 reports each). Age-stratified test after each strategy.

| Strategy | MCAR | MAR | MNAR |
|---|---:|---:|---:|
| Ignore age (crude) | 0.84 | 0.84 | 0.84 |
| Complete cases | 0.006 | 0.006 | 0.000 |
| Mean imputation | 0.28 | 0.38 | 0.49 |
| Hot-deck | 0.44 | 0.49 | 0.69 |
| Regression imputation (sex, drug count, drugs, events) | 0.014 | 0.017 | 0.028 |
| True ages (upper bound) | 0.011 | 0.011 | 0.011 |

Share of the confounded no-effect pairs flagged (lower is better). Sensitivity stays at 0.98 to 0.99 for every strategy except complete cases under MAR (0.975).

- **Mean and hot-deck imputation bring the confounding back**, most under MNAR. Mean imputation also gives the *lowest* age error (RMSE 18 years vs 20.5 for regression under MCAR), so a low imputation error is not what protects the signal test.
- **Regression imputation works because the drugs and events are in the imputation model.** The imputed age then carries the age information that drives the association. Imputation that ignores them cannot.
- Complete-case analysis does well here because the confounder is only used for stratification and enough reports remain. It would lose power in a smaller database or at higher missingness; this experiment does not cover that.

Full table: [`results/imputation.md`](results/imputation.md).

### 3. Label assistant with abstention

14 fictional label passages, 24 questions (14 answerable, 10 not). Hybrid retrieval (BM25 + TF-IDF, reciprocal rank fusion); the answer is a cited sentence, or "Not in the sources." when the top passage does not cover the question. At the default threshold (tau = 0.4):

| Measure | Value |
|---|---:|
| Gold passage in top 3 (answerable) | 1.00 |
| Gold passage ranked first | 0.79 |
| Answerable questions answered correctly | 0.64 |
| Answerable questions wrongly refused | 0.36 |
| Unanswerable questions answered | 0.00 |
| Confident wrong answers | 0.00 |

The gate never answers wrongly, but it is conservative: paraphrases that share no words with the label ("injected" vs "administered subcutaneously") are refused. A dense-embedding retriever is the next step. This is a toy set of 24 questions, so read it as a demonstration of the evaluation method, not as a benchmark. The sweep over thresholds is in [`results/rag_sweep.csv`](results/rag_sweep.csv).

### 4. Sex as an effect modifier

Age is a confounder: it distorts the signal and should be adjusted away. Sex can be different: a drug may raise an event's risk in women and not in men. Then there is no single "true" odds ratio to adjust towards, and the useful output is the signal *per sex* plus a test of whether the sexes differ. Here 4 of 8 true signals act in one sex only (relative risk 2 to 5), age confounding is off, 20 simulated databases per size. "By sex" = one table per sex with one joint Benjamini-Hochberg correction; heterogeneity = Wald z-test on the difference of the two ln ROR (p < 0.05). This is a simple Wald test, not a Breslow-Day test.

| Reports | One-sex signals found: pooled | by sex | Correct sex named | Heterogeneity power | False heterogeneity | False alarms (null): pooled | by sex |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,000 | 62% | 70% | 70% | 91% | 6% | 0.10% | 0.20% |
| 40,000 | 80% | 86% | 86% | 88% | 5% | 0.08% | 0.13% |
| 80,000 | 91% | 98% | 98% | 99% | 4% | 0.04% | 0.10% |

- A one-sex signal is diluted in the pooled table (about half of the exposed reports are unaffected), so pooled analysis misses more of them, most clearly in smaller databases.
- When the by-sex analysis flags a one-sex signal, it names the right sex in every case here; the test then detects the sex difference in about 9 of 10 cases. On both-sex signals it fires about as often as the 5% level promises.
- Cost: twice as many tests, so false alarms among null pairs roughly double (still about 0.1%).
- Only 4 + 4 true signals per database, pooled over 20 databases: percentages are noisy. Details in [`results/sex.md`](results/sex.md).

## How it works

| Module | What it does |
|---|---|
| `simulate.py` | Synthetic reports: age, sex, drugs, events; injected signals (optionally one-sex only); age confounding; ground-truth table |
| `signals.py` | 2x2 counts, PRR / ROR with confidence limits, Evans rule, one-sided Wald test with Benjamini-Hochberg FDR, Mantel-Haenszel age adjustment, per-stratum tables (sex), heterogeneity test, scoring against truth |
| `impute.py` | MCAR / MAR / MNAR missing-age mechanisms; mean, hot-deck and stochastic regression imputation |
| `db.py` | SQLite schema and the same 2x2 counts in plain SQL (tested equal to the NumPy path) |
| `rag.py` | Hybrid retrieval, abstention gate, cited extractive answer, optional LLM hook, evaluation |
| `openfda.py` | Loaders for real openFDA reports and labels (see limits below) |

The language model is optional: `answer(question, retriever, llm=callable)` builds a prompt from the retrieved passages and calls your function, but only after the gate has accepted the question.

## Run it

```bash
git clone <this repo> && cd pv-signal-lab
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[plots,test]"
pytest                                      # 22 tests
python experiments/run_detection.py         # about 10 s
python experiments/run_imputation.py        # about 10 s
python experiments/run_rag_eval.py
python experiments/run_sex.py               # about 1 min
```

## Limits (read before using on real data)

- **Synthetic data.** Real reporting has duplicates, free-text drug names, reporting stimulated by publicity, co-medication and many more confounders than age. The simulation has age, plus sex as an effect modifier in experiment 4.
- **Reports are not incidence.** Without an exposure denominator, a flagged pair is a hypothesis for clinical review, not evidence of causation.
- **Reports with no event are dropped** in the simulation, as in real databases, which slightly distorts the marginals.
- **`openfda.py` is untested against the live API.** The parser is unit-tested on a hand-written example of the response format; the network was not reachable when this was built. Check the openFDA terms and rate limits before pulling data.
- **Detection thresholds are not tuned.** Evans' rule and BH q = 0.05 are used as published defaults.

## Roadmap

- [ ] Run the pipeline on real openFDA reports for one drug class and review the top flags against the labels
- [ ] Add a Bayesian shrinkage rule (empirical-Bayes gamma-Poisson shrinker) to the comparison
- [x] Sex as an effect modifier (experiment 4)
- [ ] Add co-medication as a confounder and a propensity-style adjustment
- [ ] Projection: forecast report counts per drug-event pair and flag departures
- [ ] Dense-embedding retriever and a larger, hand-labelled question set from real labels
- [ ] Deploy the assistant as a small service on a cloud platform (Azure) and document the setup
