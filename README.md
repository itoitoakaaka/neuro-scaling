# neuro-scaling

Neural scaling laws analysis and deep learning-based neural decoding using large-scale brain data (EEG / MEG / ECoG).

## Overview

This repository investigates how model performance scales with data size in neural decoding tasks — a phenomenon known as "neural scaling laws." It provides tools to load EEG data, extract features, compute scaling curves, and train CNN-based decoders using PyTorch.

## Features

- **Data Loading** (`data_loader.py`): Load and preprocess EEG data from the PhysioNet Motor Imagery dataset via MNE-Python.
- **Scaling Analysis** (`scaling_analysis.py`): Compute accuracy vs. data size curves and fit power-law models (y = a·x^b + c).
- **Neural Decoder** (`decoder.py`): Convolutional neural network for EEG classification (motor imagery left/right hand).
- **CLI Interface** (`main.py`): Simple command-line interface for download, scaling analysis, and decoding.

## Requirements

- Python 3.8+
- MNE-Python
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- PyTorch

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Download sample EEG data
python main.py download

# Run scaling law analysis (uses subjects 1-5)
python main.py scaling

# Train a neural decoder
python main.py decode

# Run the full pipeline
python main.py all
```

## Output

- `scaling_curve.png`: Plot of classification accuracy vs. data fraction with power-law fit.
