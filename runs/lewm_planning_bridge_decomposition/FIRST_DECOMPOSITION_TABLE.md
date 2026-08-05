# First decomposition table

Produced 2026-07-18 11:27:00 MST from the fixed consumed 20x64 pilot.

| link | fixed metric | result | descriptive decision |
|:--|:--|:--|:--|
| candidate rollout transfer | adaptive-minus-matched raw terminal MSE | -2.7352994e-06 (95% cluster interval -6.3920558e-05, +5.5192696e-05) | supported |
| candidate rollout transfer | adaptive-minus-matched whitened terminal MSE | -0.00014018128 (95% cluster interval -0.00049388372, +0.00015512488) | supported |
| goal-cost alignment ceiling | actual-latent raw cost vs physical error where rank is defined | 9 starts; mean Spearman 0.0237205251946946; all-start mean top-5 overlap 0.04 vs chance 0.078125 | unsupported |
| candidate informativeness | informative starts under fixed rule | 0/20 | unsupported |
| downstream planning evidence | all required links plus selection change | goal-cost alignment and candidate informativeness remain unsupported, and adaptive changes no informative-start selection; the nominal prior ranking label does not justify planning or control | unsupported |
