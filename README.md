# GNN and Continuous Convolutions based N-body simulations

This project aims to accelerate simulations of complex systems by integrating Graph Neural Networks (GNNs) and Continuous Convolutions with traditional numerical methods. It provides tools for generating initial conditions and running simulation experiments.

## Project Overview

- **Simulator & Initial Condition Generation:**  
  The core simulation environment, including initial condition generation, is implemented in the `src/galaxify` directory.

- **Experiment Modules:**  
  Two main experiments are supported:
  - **Continuous Convolution Experiment:** Focuses on training and evaluating continuous convolutions models for simulation tasks.
  - **GNN Experiment:** Focuses on training and evaluating GNN models for simulation tasks.

  Results are stored in the `results/` directory, separated into `contconv` and `gnn` subfolders, while pre-trained weights reside in `contconv_weights` and `gnn_weights`.

## Requirements

- **Python 3.10**  
- Key libraries include:
  - `pandas`
  - `PyTorch` and `torch-geometric` (with all related dependencies)
  - `numpy`
  - `tqdm`

Install all required packages with:
```bash
pip install -r requirements.txt
