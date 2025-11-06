# PPO-GNN: Constraint-Aware Deep Reinforcement Learning for Stochastic Fuel Delivery Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


**Official Implementation of "Constraint-Aware PPO-GNN with Closed-Loop Validation for Stochastic Fuel Delivery Optimization"**

---

## 📖 Overview

This repository contains the complete implementation of our hybrid framework that integrates **Proximal Policy Optimization (PPO)**, **Graph Neural Networks (GNNs)**, and a novel **three-tier constraint validation mechanism** for solving stochastic vehicle routing problems in fuel delivery operations.

### 🎯 Key Innovation

Unlike traditional one-way validation approaches, our framework creates a **closed-loop feedback system** where the deterministic model serves as a dynamic feasibility oracle, continuously refining the learned policy through adaptive tiered penalties:

- **Tier 1** (V ≤ 5%): Tolerance zone for exploration
- **Tier 2** (5% < V ≤ 25%): Fine-tuning with 1.5× penalty adjustments (1,000 episodes)
- **Tier 3** (V > 25%): Full re-training with 10× penalty restructuring + policy reset (10,000 episodes)

---

## ✨ Key Features

- ✅ **Near-Optimal Performance:** 1.6-2.0% verified optimality gap on tractable instances
- ✅ **Computational Efficiency:** 24-51× faster than commercial solvers (Gurobi)
- ✅ **Scalability:** Handles networks from 10 to 200+ nodes (exact solvers fail at 200 nodes)
- ✅ **GNN Integration:** 12.3% cost reduction and 79.2% fewer violations vs. flat features
- ✅ **Constraint-Aware Learning:** Bidirectional feedback for feasibility-driven policy refinement
- ✅ **Comprehensive Baselines:** Classical PPO, PPO-MLP, Clarke-Wright, Gurobi integration

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/YourUsername/PPO-GNN-Fuel-Delivery.git
cd PPO-GNN-Fuel-Delivery

# Create virtual environment
conda create -n ppo_gnn python=3.8
conda activate ppo_gnn

# Install dependencies
pip install -r requirements.txt

# Install Gurobi (optional, for exact solver comparison)
# Requires Gurobi license: https://www.gurobi.com/downloads/
pip install gurobipy
```

### Dependencies
```
torch==1.12.0
torch-geometric==2.1.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
gurobipy>=10.0.0  # Optional
tqdm>=4.62.0
pyyaml>=5.4.0
tensorboard>=2.8.0
```

---

## 🎮 Training

### Train PPO-GNN (Full Framework)
```bash
python src/training/train_ppo_gnn.py \
    --config configs/ppo_gnn_config.yaml \
    --dataset data/synthetic_networks/large_100_nodes \
    --epochs 50000 \
    --early_stop 5000 \
    --gpu 0
```

### Train Baselines
```bash
# Classical PPO (no GNN)
python src/training/train_classical_ppo.py --config configs/classical_ppo_config.yaml

# PPO-MLP (MLP encoder without message-passing)
python src/training/train_ppo_mlp.py --config configs/ppo_mlp_config.yaml
```

---

## 📊 Evaluation

### Run Full Benchmark
```bash
python experiments/benchmark_comparison.py \
    --methods ppo_gnn classical_ppo clarke_wright gurobi \
    --dataset data/synthetic_networks/large_100_nodes \
    --output results/benchmark_results.csv
```

### Reproduce Paper Results
```bash
# Ablation Study (Section 5.2)
python experiments/ablation_study.py --output results/ablation/

# Optimality Gap Analysis (Section 5.4)
python experiments/optimality_gap_analysis.py --output results/gaps/

# LP Lower Bounds (Section 5.4)
python experiments/lp_relaxation_bounds.py --output results/lp_bounds/
```

---

## 📁 Repository Structure
```
PPO-GNN-Fuel-Delivery/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── configs/                           # Configuration files
│   ├── ppo_gnn_config.yaml           # PPO-GNN hyperparameters (Table 5.1)
│   ├── classical_ppo_config.yaml     # Classical PPO baseline
│   ├── ppo_mlp_config.yaml           # PPO-MLP baseline
│   └── reward_weights.yaml           # Reward function weights (λ₁-λ₅)
│
├── src/                               # Source code
│   ├── models/                        # Neural network architectures
│   │   ├── ppo_gnn.py                # Complete PPO-GNN framework
│   │   ├── gnn_encoder.py            # 3-layer GNN with message-passing (Eq. 1)
│   │   ├── policy_network.py         # Policy π_θ (actor)
│   │   ├── value_network.py          # Value V_φ (critic)
│   │   └── gae.py                    # Generalized Advantage Estimation (Eq. 9)
│   │
│   ├── baselines/                     # Baseline methods
│   │   ├── classical_ppo.py          # PPO with flat features (512-dim)
│   │   ├── ppo_mlp.py                # PPO with MLP encoder (no message-passing)
│   │   ├── clarke_wright.py          # Clarke-Wright Savings Algorithm
│   │   └── gurobi_solver.py          # Exact MILP solver integration
│   │
│   ├── validation/                    # Three-tier validation mechanism
│   │   ├── deterministic_model.py    # Feasibility oracle (Section 3.2)
│   │   ├── violation_analysis.py     # V_total computation (Eq. 11)
│   │   ├── adaptive_penalties.py     # Tiered penalty system (Algorithm 2)
│   │   └── constraint_checker.py     # Constraint evaluation utilities
│   │
│   ├── training/                      # Training procedures
│   │   ├── ppo_trainer.py            # PPO optimization loop
│   │   ├── algorithm1_training.py    # Complete training (Algorithm 1)
│   │   ├── algorithm2_validation.py  # Constraint validation (Algorithm 2)
│   │   └── early_stopping.py         # Convergence monitoring
│   │
│   ├── evaluation/                    # Evaluation metrics
│   │   ├── optimality_gap.py         # Verified/bounded gap analysis
│   │   ├── lp_relaxation.py          # LP lower bounds (Section 5.4)
│   │   ├── constraint_metrics.py     # Violation rate computation
│   │   └── performance_metrics.py    # Cost, unmet demand, solve time
│   │
│   └── utils/                         # Utility functions
│       ├── graph_utils.py            # Graph construction from networks
│       ├── data_loader.py            # Dataset loading utilities
│       ├── visualization.py          # Route plotting, training curves
│       └── logger.py                 # TensorBoard logging
│
├── data/                              # Synthetic datasets
│   ├── synthetic_networks/
│   │   ├── small_10_nodes/           # 10 nodes, 30 edges
│   │   ├── medium_50_nodes/          # 50 nodes, 150 edges
│   │   ├── large_100_nodes/          # 100 nodes, 300 edges
│   │   └── xlarge_200_nodes/         # 200 nodes, 600 edges
│   │
│   ├── generation_scripts/           # Data generation utilities
│   │   ├── generate_networks.py      # Network topology generator
│   │   ├── generate_demands.py       # Stochastic demand generator
│   │   └── generate_constraints.py   # Operational constraint generator
│   │
│   └── README_DATA.md                # Dataset documentation
│
├── experiments/                       # Experiment scripts
│   ├── ablation_study.py             # GNN integration ablation (Section 5.2)
│   ├── benchmark_comparison.py       # All methods comparison (Section 5.3)
│   ├── optimality_gap_analysis.py    # Gap analysis (Section 5.4)
│   ├── lp_relaxation_bounds.py       # Lower bound computation (Section 5.4)
│   ├── hyperparameter_tuning.py      # Grid search experiments
│   └── generalization_test.py        # Cross-size generalization (Table 5.3)
│
├── checkpoints/                       # Pre-trained models
│   ├── ppo_gnn_best.pth              # Best PPO-GNN checkpoint
│   ├── classical_ppo_best.pth        # Classical PPO checkpoint
│   └── ppo_mlp_best.pth              # PPO-MLP checkpoint
│
├── notebooks/                         # Jupyter tutorials
│   ├── 01_demo_training.ipynb        # Training walkthrough
│   ├── 02_evaluation.ipynb           # Evaluation and metrics
│   ├── 03_visualization.ipynb        # Route visualization
│   └── 04_ablation_analysis.ipynb    # Ablation study analysis
│
├── results/                           # Experiment outputs (created at runtime)
│   ├── ablation/
│   ├── benchmark/
│   ├── gaps/
│   └── figures/
│
└── docs/                              # Documentation
    ├── INSTALLATION.md               # Detailed setup guide
    ├── REPRODUCTION.md               # Reproduce paper results
    ├── API.md                        # Code API documentation
    └── FAQ.md                        # Frequently asked questions
```

---

## 📊 Datasets

We provide synthetic fuel delivery networks across four scales:

| Scale | Nodes | Edges | Vehicles | Use Case |
|-------|-------|-------|----------|----------|
| **Small** | 10 | 30 | 3 | Urban district distribution |
| **Medium** | 50 | 150 | 10 | Regional delivery network |
| **Large** | 100 | 300 | 15 | National distribution |
| **Very Large** | 200 | 600 | 25 | Industrial-scale operations |

Each instance includes:
- **Network Topology:** Spatial coordinates, distances, travel times
- **Stochastic Demand:** Mean (μᵢ) and standard deviation (σᵢ) per station
- **Operational Constraints:** Vehicle capacities, time windows, fuel type compatibility
- **Validation Sets:** 20 test instances per scale for ablation study

### Generate New Instances
```bash
python data/generation_scripts/generate_networks.py \
    --num_nodes 100 \
    --num_edges 300 \
    --num_instances 50 \
    --output data/custom_networks/
```

---

## 🧪 Reproducing Paper Results

### Table 5.2: Ablation Study
```bash
python experiments/ablation_study.py \
    --methods ppo_flat ppo_mlp ppo_gnn \
    --dataset data/synthetic_networks/medium_50_nodes \
    --num_episodes 50000 \
    --output results/ablation/table_5_2.csv
```

**Expected Results:**
| Method | Total Cost ($) | Violations (%) | Unmet Demand (%) | Training Time (h) |
|--------|----------------|----------------|------------------|-------------------|
| PPO-Flat | 14,200 ± 380 | 18.3 | 8.2 | 2.8 |
| PPO-MLP | 13,500 ± 290 | 12.1 | 6.5 | 3.1 |
| **PPO-GNN** | **12,450 ± 210** | **3.8** | **4.1** | 3.5 |

### Table 5.4: Overall Performance Comparison
```bash
python experiments/benchmark_comparison.py \
    --methods gurobi ppo_gnn classical_ppo clarke_wright \
    --dataset data/synthetic_networks/large_100_nodes \
    --gurobi_time_limit 7200 \
    --output results/benchmark/table_5_4.csv
```

**Expected Results:**
| Method | Total Cost ($) | Gap (%) | Unmet Demand (%) | Solve Time (s) | Violations (%) |
|--------|----------------|---------|------------------|----------------|----------------|
| Gurobi | 12,560* | 0.0 | 0 | 7,200+ | 0 |
| **PPO-GNN** | **12,800** | **1.9** | **2** | **180** | **1** |
| Classical DRL | 13,800 | 9.9 | 8 | 120 | 6 |
| Heuristic | 14,200 | 13.1 | 5 | 60 | 3 |

*Best solution found after 2-hour time limit (MIP gap: 1.2%)

### Table 5.5: Detailed Gurobi Comparison
```bash
python experiments/optimality_gap_analysis.py \
    --sizes 10 50 100 200 \
    --methods gurobi ppo_gnn \
    --output results/gaps/table_5_5.csv
```

**Expected Results:**
| Instance Size | Gurobi Cost ($) | PPO-GNN Cost ($) | Gap (%) | Time Gurobi (s) | Time PPO-GNN (s) |
|---------------|-----------------|------------------|---------|-----------------|------------------|
| Small (10) | 4,320** | 4,390 | 1.6 | 285 | 12 |
| Medium (50) | 8,650** | 8,820 | 2.0 | 4,580 | 90 |
| Large (100) | 12,560* | 12,800 | 1.9 | 7,200+ | 180 |
| Very Large (200) | N/A*** | 24,500 | N/A | 14,400+ | 420 |

**Proven optimal (MIP gap = 0%)  
*Best known (MIP gap = 1.2%)  
***No solution within 4 hours

### Table 5.6: LP Lower Bounds
```bash
python experiments/lp_relaxation_bounds.py \
    --sizes 10 50 100 200 \
    --output results/lp_bounds/table_5_6.csv
```

**Expected Results:**
| Instance Size | LP Lower Bound ($) | Gurobi Upper ($) | PPO-GNN ($) | Gap Range (%) |
|---------------|--------------------|------------------|-------------|---------------|
| Small (10) | 4,280 | 4,320 | 4,390 | [0.9, 2.6] |
| Medium (50) | 8,520 | 8,650 | 8,820 | [1.5, 3.5] |
| Large (100) | 12,180 | 12,560 | 12,800 | [2.0, 5.1] |
| Very Large (200) | 23,400 | N/A | 24,500 | [4.7, N/A] |

---

## 🎓 Key Results Summary

### Performance Highlights

- **✅ 1.6-2.0% verified optimality gap** on small/medium instances (proven optimal)
- **✅ 1.9-3.1% bounded gap** on large instances (via MIP gap analysis)
- **✅ 2.6-5.1% gap** via LP relaxation lower bounds
- **✅ 24-51× faster** than Gurobi commercial solver
- **✅ 12.3% cost reduction** from GNN integration (ablation study)
- **✅ 79.2% fewer violations** compared to flat feature encoding
- **✅ 7.2% cost reduction** vs. Classical DRL
- **✅ 9.9% cost reduction** vs. Clarke-Wright heuristic
- **✅ Only viable method** for 200+ node instances (Gurobi fails)

### Computational Efficiency

| Method | 10 Nodes | 50 Nodes | 100 Nodes | 200 Nodes |
|--------|----------|----------|-----------|-----------|
| Gurobi | 285s | 4,580s | 7,200s+ | Fail (>4h) |
| PPO-GNN | **12s** | **90s** | **180s** | **420s** |
| **Speedup** | **24×** | **51×** | **40×** | **∞** |

---

## 🔧 Configuration

### Hyperparameters (Table 5.1)

All hyperparameters are configured in `configs/ppo_gnn_config.yaml`:
```yaml
# PPO Policy Network
policy:
  learning_rate: 3.0e-4
  epsilon_clip: 0.2
  entropy_coef: 0.01
  architecture: [256, 128, 64]

# Value Function
value:
  learning_rate: 1.0e-3
  gae_lambda: 0.95
  architecture: [256, 128]

# GNN Encoder
gnn:
  learning_rate: 1.0e-4
  num_layers: 3
  hidden_dim: 128

# Training
training:
  discount_factor: 0.99
  batch_size: 256
  epochs_per_update: 10
  max_episodes: 50000
  early_stop_patience: 5000

# Reward Weights
rewards:
  lambda_cost: 1.0
  lambda_dispersion: 0.5
  lambda_delay: 0.8
  lambda_unmet: 1.2
  lambda_constraint: 2.0

# Three-Tier Validation
validation:
  tier1_threshold: 0.05  # 5%
  tier2_threshold: 0.25  # 25%
  tier2_penalty_multiplier: 1.5
  tier2_episodes: 1000
  tier3_penalty_multiplier: 10.0
  tier3_episodes: 10000
  validation_frequency: 1000  # Every 1000 episodes
```

---

## 📈 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir runs/
```

**Logged Metrics:**
- Policy loss (L^CLIP)
- Value loss (L^VF)
- Total reward per episode
- Constraint violation rate (V_total)
- Optimality gap (vs. Gurobi when available)
- Tier activation frequency (Tier 1/2/3)

### Checkpointing

Models are saved automatically:
- **Best checkpoint:** Lowest validation cost
- **Latest checkpoint:** Most recent training state
- **Periodic checkpoints:** Every 5,000 episodes
```python
# Load pre-trained model
from src.models.ppo_gnn import PPOGNN

model = PPOGNN.load_from_checkpoint('checkpoints/ppo_gnn_best.pth')
```

---

## 📝 Citation

```bibtex
@article{Argoubi2025ppo_gnn,
  title={Constraint-Aware PPO-GNN with Closed-Loop Validation for Stochastic Fuel Delivery Optimization},
  author={Majdi Argoubi},
  journal={Optimization and Engineering},
  year={2025},
  volume={XX},
  pages={XXX-XXX},
  doi={10.XXXX/XXXXX}
}
```

---

## 📧 Contact & Support

- **Primary Author:** Majdi Argoubi (mejdiargoubi@yahoo.fr)
- **Issues:** Please use [GitHub Issues](https://github.com/ARGOUBI25/PPO-GNN-Fuel-Delivery/issues)
- **Discussions:** [GitHub Discussions](https://github.com/ARGOUBI25/PPO-GNN-Fuel-Delivery/discussions)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- PyTorch and PyTorch Geometric teams for excellent frameworks
- Gurobi Optimization for academic licenses
- Reviewers for valuable feedback that improved this work

---

## 🔗 Related Projects

- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [RL4CO: Reinforcement Learning for Combinatorial Optimization](https://github.com/kaist-silab/rl4co)
- [OR-Tools: Google's Operations Research Tools](https://github.com/google/or-tools)

---

**⭐ If you find this work useful, please consider starring the repository!**
