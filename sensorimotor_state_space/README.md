# Sensorimotor State-Space and Trial-by-Trial Modeling

A compact computational human-behavior modeling demo for sensorimotor adaptation.

## Questions

This folder focuses on two complementary questions:

1. Latent-state modeling: how retention and error sensitivity evolve across repeated trials.

    x[t+1] = A * x[t] + B * e[t] + w[t]

2. Trial-by-trial correction: how strongly error on trial t predicts correction on trial t+1.

    correction[t+1] = beta * error[t] + intercept

These simple models can later be extended to hierarchical Bayesian models, Kalman filters, multi-rate adaptation models, or neural-behavior joint models.

## Files

- model.py: simulation and fitting of a one-state adaptation model.
- trial_by_trial.py: error-to-next-trial correction gain.
- demo.py: state-space parameter recovery from synthetic behavior.
- human_behavior_demo.py: synthetic Land vs Water comparison.
- test_model.py: parameter-recovery test.

## Why this matters

The modeling layer turns repeated human behavioral observations into interpretable computational parameters such as retention, error sensitivity, and trial-by-trial correction gain.

The public demo uses synthetic data only. Real participant data are not included.

## Run

    python sensorimotor_state_space/demo.py
    python sensorimotor_state_space/human_behavior_demo.py

## Next extensions

- fit participant-level models
- compare Land vs Water parameters
- compare experienced vs non-experienced groups
- hierarchical Bayesian estimation
- connect latent behavioral states to EEG/SEP measures
