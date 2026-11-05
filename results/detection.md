Mean over 20 simulated databases per size. Reports = reports kept after dropping those with no event.

|   n_simulated | method   |   reports |   sensitivity |   fa_confounded |   fa_null |
|--------------:|:---------|----------:|--------------:|----------------:|----------:|
|          5000 | evans    |      2959 |         0.825 |           0.24  |     0.022 |
|          5000 | mh_fdr   |      2959 |         0.612 |           0.001 |     0.002 |
|          5000 | ror_fdr  |      2959 |         0.681 |           0.215 |     0.003 |
|         10000 | evans    |      5917 |         0.888 |           0.147 |     0.016 |
|         10000 | mh_fdr   |      5917 |         0.75  |           0     |     0.001 |
|         10000 | ror_fdr  |      5917 |         0.794 |           0.406 |     0.003 |
|         20000 | evans    |     12307 |         0.938 |           0.164 |     0.006 |
|         20000 | mh_fdr   |     12307 |         0.875 |           0.004 |     0.001 |
|         20000 | ror_fdr  |     12307 |         0.906 |           0.654 |     0.003 |
|         40000 | evans    |     24130 |         0.962 |           0.079 |     0.002 |
|         40000 | mh_fdr   |     24130 |         0.981 |           0.01  |     0.001 |
|         40000 | ror_fdr  |     24130 |         0.981 |           0.826 |     0.003 |

Sensitivity by relative risk of the injected signal (20,000 simulated, about 12,000 reports kept):

method        evans  mh_fdr  ror_fdr
rr_bin                              
(1.999, 3.0]    0.8    0.65     0.71
(3.0, 4.0]      1.0    0.97     1.00
(4.0, 5.0]      1.0    1.00     1.00
