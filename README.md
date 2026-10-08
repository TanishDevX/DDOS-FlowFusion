# FlowFusion: A Leakage-Free Benchmark and Architecture for DDoS Detection

## Overview

This repository contains an academic benchmarking suite and a novel neural network architecture (`FlowFusion`) designed for robust DDoS traffic detection. The core problem this project addresses is **temporal data leakage** in network intrusion detection systems (NIDS). By strictly separating the training dataset (January 12th) from the testing dataset (March 11th) on the CIC-DDoS2019 dataset, and aggressively dropping leaky features (e.g., `Inbound`, `Timestamp`), this project provides a rigorous real-world evaluation.

Furthermore, we propose **FlowFusion**, a multi-branch deep learning model that leverages a Feature Interaction Network (FIN) and Structural Group Attention to analyze non-linear interactions across tabular networking features, achieving near state-of-the-art accuracy while maintaining high model interpretability.

## Features

- **Strict Leakage-Free Protocol**: Standardized cross-session evaluation (Train: Jan 12, Test: Mar 11) for true generalization.
- **Novel FlowFusion Architecture**: Custom Keras 3 neural network utilizing cross-group Hadamard interactions and Contextual Scalar Gating.
- **Multiple Baseline Architectures**: Clean implementations of XGBoost, LightGBM, Residual MLPs, and FT-Transformers.
- **Bayesian Hyperparameter Optimization**: Automated tuning pipeline via `Optuna`.
- **Built-in Explainability**: Advanced model introspection using SHAP (for tree models) and Integrated Gradients (for deep learning).
- **Interactive REST API**: A FastAPI wrapper with `/predict` and `/explain` endpoints for inference and attack simulation.

## Architecture

The project consists of three major pipelines:
1. **Data Pipeline (`src/data`)**: Handles missing values (median imputation), outlier removal, robust scaling, class balancing, and data augmentation (MixUp, Gaussian Noise).
2. **Modeling Pipeline (`src/models` & `scripts`)**: Defines the architectures and handles Optuna hyperparameter searches, early stopping, and focal loss techniques to tackle class imbalance.
3. **Inference & Explainability (`src/deployment` & `src/evaluation`)**: Evaluates structural feature interactions and generates SHAP/IG attribution scores, deployable via an interactive web API.

## Project Structure

```text
DDOS/
├── config/
│   └── config.yaml          # Centralized hyperparameter and configuration hub
├── data/
│   ├── raw/                 # Ignored: Original CIC-DDoS2019 CSV files
│   └── processed/           # Ignored: Scaled and encoded Numpy arrays
├── experiments/
│   └── results/             # Ignored: Model weights, CSV metrics, and charts
├── notebooks/
│   └── 01_eda.ipynb         # Exploratory Data Analysis
├── scripts/                 # Entry points for preprocessing, training, optimization, and simulation
├── src/
│   ├── data/                # Scalers, feature audit, and augmentations
│   ├── deployment/          # FastAPI application
│   ├── evaluation/          # Metrics and explainability modules
│   ├── models/              # Neural Network layer definitions (FlowFusion, MLP, FT-Transformer)
│   └── utils/               # Logging and reproducibility utilities
├── requirements.txt         # Project dependencies
└── README.md
```

## Requirements

- Python 3.10+
- Hardware: Multi-core CPU for tree models; NVIDIA GPU (e.g., T4/L4) heavily recommended for training FlowFusion and FT-Transformers.
- Data: The CIC-DDoS2019 dataset (must be downloaded manually).

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/DDOS-FlowFusion.git
cd DDOS-FlowFusion

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

Download the CIC-DDoS2019 dataset and place the CSVs in the respective directories:
```text
data/raw/01-12/    # Place January 12 training CSVs here
data/raw/03-11/    # Place March 11 testing CSVs here
```

## Usage

### 1. Data Preprocessing
Cleans the raw CSVs, extracts features, and drops leaky artifacts.
```bash
python scripts/preprocess.py
```

### 2. Hyperparameter Optimization
Use Optuna to find the optimal Bayesian search space for the models.
```bash
# Tune a specific model
python scripts/optuna_search.py --model xgboost --n-trials 30

# Or tune all models consecutively
python scripts/optuna_search.py --model all --n-trials 30
```

### 3. Training & Evaluation
Train the final models on multiple random seeds to establish statistical variance.
```bash
python scripts/run_all_experiments.py
```

### 4. Simulating Live Inference (API)
Start the FastAPI server:
```bash
python -m uvicorn src.deployment.app:app --host 0.0.0.0 --port 8000
```
Then, in a new terminal, run the attack simulator which streams actual test vectors into the API:
```bash
python scripts/simulate_attacks.py --num-per-class 10 --delay 0.5
```

## Future Improvements
- **Live Packet Capture Integration**: Porting the pipeline to ingest live network interfaces via PCAP/eBPF instead of dataset vectors.
- **Open-Set Anomaly Detection**: Implementing an autoencoder pipeline to flag Zero-Day DDoS attacks not present in the 6 primary classes.

## License
MIT License
