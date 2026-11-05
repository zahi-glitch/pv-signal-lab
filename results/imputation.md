Mean over 10 simulated databases, 30% of ages removed.

mechanism        method  sensitivity  fa_confounded  fa_null  age_rmse
        -         crude        0.988          0.839    0.003       NaN
        -        oracle        0.988          0.011    0.001     0.000
     MCAR complete_case        0.988          0.006    0.000       NaN
     MCAR          mean        0.988          0.275    0.001    18.299
     MCAR       hotdeck        0.988          0.436    0.001    25.955
     MCAR    regression        0.988          0.014    0.001    20.547
      MAR complete_case        0.975          0.006    0.001       NaN
      MAR          mean        0.988          0.383    0.001    18.322
      MAR       hotdeck        0.988          0.492    0.001    25.966
      MAR    regression        0.988          0.017    0.001    20.426
     MNAR complete_case        0.988          0.000    0.001       NaN
     MNAR          mean        0.988          0.489    0.001    25.095
     MNAR       hotdeck        0.988          0.686    0.002    29.964
     MNAR    regression        0.988          0.028    0.001    22.557
