# Experiment 4: sex as an effect modifier

20 simulated databases per size; 8 true signals per database, 4 of them in one sex only (relative risk 2 to 5); no age confounding. Pooled = ignore sex; by sex = one table per sex, one joint Benjamini-Hochberg correction (q = 0.05) over both sexes' tests. Heterogeneity = Wald z-test that ln ROR differs between women and men (p < 0.05).

| Reports | Found, both-sex signals: pooled | by sex | Found, one-sex signals: pooled | by sex | Correct sex named | Heterogeneity power | False heterogeneity | False alarms (null): pooled | by sex |
|---|---|---|---|---|---|---|---|---|---|
| 20,000 | 81% | 82% | 62% | 70% | 70% | 91% | 6% | 0.10% | 0.20% |
| 40,000 | 99% | 95% | 80% | 86% | 86% | 88% | 5% | 0.08% | 0.13% |
| 80,000 | 98% | 96% | 91% | 98% | 98% | 99% | 4% | 0.04% | 0.10% |

Reading: *Correct sex named* = flagged in the affected sex and not in the other. *Heterogeneity power* = share of flagged one-sex signals whose sex difference the test detects. *False heterogeneity* = same test on flagged both-sex signals (should be near 5%). Counts of true signals are small (4 + 4 per database), so percentages are pooled over databases and still noisy.
