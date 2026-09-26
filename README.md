# neuro-scaling

A reproducible EEG decoding demo that explores how classification performance changes with available data size.

## Why this repository exists

My research background is in human sensorimotor neuroscience and EEG/SEP analysis. This repository is a computational extension: it uses a public motor-imagery dataset to demonstrate neural decoding, leakage-safe model evaluation, and scaling-curve analysis.

## Workflow

1. Download public PhysioNet EEGBCI motor-imagery recordings with MNE-Python.
2. Retain T1/T2 imagined-movement epochs.
3. Compute channel-wise band-power features for a lightweight baseline decoder.
4. Evaluate performance across increasing data fractions with preprocessing learned only inside each training fold.
5. Train a compact PyTorch CNN and evaluate it on held-out subjects.

## Methods

- MNE-Python for EEG loading and preprocessing
- NumPy/SciPy for numerical analysis
- scikit-learn for baseline decoding and cross-validation
- PyTorch for a compact CNN decoder
- Exploratory power-law fitting of performance vs. data scale

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Usage

    python main.py download
    python main.py scaling
    python main.py decode
    python main.py all

## Output

- scaling_curve.png: cross-validated decoding accuracy across data fractions
- Console output for held-out-subject CNN decoding

## Scope and limitations

This is a methodological portfolio project, not a claim of a neural scaling law in the formal large-model sense. The dataset and model are intentionally modest. The emphasis is on reproducible evaluation, avoiding preprocessing leakage, and connecting EEG analysis with computational modeling.
