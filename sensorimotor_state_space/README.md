# Sensorimotor State-Space Modeling

A compact computational sensorimotor-learning demo.

## Goal

Estimate how a latent motor state is retained and updated from trial-by-trial error:

    x[t+1] = A * x[t] + B * e[t] + w[t]

where:

- A = retention
- B = error sensitivity / learning rate
- e[t] = target - observed output

This folder uses synthetic data so the model can be shared publicly without study data.

## What it demonstrates

- simulation of trial-by-trial adaptation
- parameter fitting for A and B
- parameter recovery on synthetic data
- a starting point for applying the same model to real behavioral data

## Run

    python sensorimotor_state_space/demo.py

The long-term use case is to compare fitted parameters across environments or participant groups while keeping the model assumptions explicit.
