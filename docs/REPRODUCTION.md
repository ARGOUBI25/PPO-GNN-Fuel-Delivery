# Reproducing Paper Results

This guide provides step-by-step instructions to reproduce all experimental results, tables, and figures from the paper **"Constraint-Aware PPO-GNN with Closed-Loop Validation for Stochastic Fuel Delivery Optimization"**.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Reproduction (All Results)](#quick-reproduction-all-results)
4. [Section 5.1: Experimental Configuration](#section-51-experimental-configuration)
5. [Section 5.2: Ablation Study](#section-52-ablation-study)
6. [Section 5.3: Overall Performance Comparison](#section-53-overall-performance-comparison)
7. [Section 5.4: Detailed Gurobi Comparison](#section-54-detailed-gurobi-comparison)
8. [Section 5.5: Visual Analysis](#section-55-visual-analysis)
9. [Hardware and Time Requirements](#hardware-and-time-requirements)
10. [Pre-trained Models](#pre-trained-models)
11. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This guide reproduces:

- **Table 5.1:** Hyperparameter configuration (already in `configs/`)
- **Table 5.2:** Ablation study results (PPO-Flat, PPO-MLP, PPO-GNN)
- **Table 5.3:** Generalization across network sizes
- **Table 5.4:** Overall performance comparison (all methods, 100 nodes)
- **Table 5.5:** Detailed Gurobi comparison across problem scales
- **Table 5.6:** LP lower bounds for optimality gap analysis
- **Figure 5.1:** Cost and unmet demand comparison
- **Figure 5.2:** Optimized delivery routes visualization

---

## ✅ Prerequisites

### 1. Installation

Complete the installation as described in [INSTALLATION.md](INSTALLATION.md).

### 2. Verify Installation
```bash
python scripts/verify_installation.py
```

### 3. GPU Availability (Recommended)
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

> **Note:** Training can be done on CPU but will take significantly longer (~10× slower).

### 4. Gurobi License (Optional)

Required only for exact solver comparison (Tables 5.4, 5.5, 5.6).
```bash
python -c "import gurobipy; print('Gurobi Available')"
```

If not available, you can still reproduce PPO-GNN, Classical DRL, and heuristic results.

---

## 🚀 Quick Reproduction (All Results)

Run the complete reproduction pipeline:
```bash
# Full reproduction (requires GPU + Gurobi)
bash scripts/reproduce_all.sh

# Without Gurobi (skips exact solver comparison)
bash scripts/reproduce_all.sh --no-gurobi

# CPU-only mode (slower)
bash scripts/reproduce_all.sh --device cpu
```

**Estimated Time:**
- With GPU + Gurobi: ~48 hours
- With GPU, no Gurobi: ~36 hours
- CPU-only: ~120 hours

**Output:**
All results will be saved to `results/reproduction/` with subdirectories for each table/figure.

---

## 📊 Section 5.1: Experimental Configuration

### Table 5.1: Hyperparameter Configuration

Hyperparameters are already configured in `configs/ppo_gnn_config.yaml`.

**Verify configuration:**
```bash
python scripts/verify_hyperparameters.py
```

**Expected output:**
```
✓ Policy learning rate (α_θ): 3e-4
✓ Value learning rate (α_φ): 1e-3
✓ GNN learning rate (α_ψ): 1e-4
✓ Clipping parameter (ε): 0.2
✓ Discount factor (γ): 0.99
✓ GAE parameter (λ): 0.95
✓ Batch size (B): 256
✓ Optimization epochs (K): 10
✓ Maximum episodes: 50,000
✓ GNN layers (L): 3
✓ GNN hidden dimension: 128
✓ Reward weights: λ₁=1.0, λ₂=0.5, λ₃=0.8, λ₄=1.2, λ₅=2.0
✓ Tier-1 threshold: 0.05 (5%)
✓ Tier-2 threshold: 0.25 (25%)
✓ Validation frequency: 1,000 episodes

All hyperparameters match Table 5.1 ✓
```

---

## 🔬 Section 5.2: Ablation Study

### Table 5.2: Impact of GNN Integration

**Reproduce complete ablation study:**
```bash
python experiments/ablation_study.py \
    --methods ppo_flat ppo_mlp ppo_gnn \
    --dataset data/synthetic_networks/medium_50_nodes \
    --num_episodes 50000 \
    --num_test_instances 20 \
    --seeds 42 123 456 789 1011 \
    --output results/ablation/table_5_2.csv
```

**Arguments:**
- `--methods`: Architectures to compare (PPO-Flat, PPO-MLP, PPO-GNN)
- `--dataset`: Medium-scale instances (50 nodes, 15 vehicles)
- `--num_episodes`: Training duration (50,000 episodes)
- `--num_test_instances`: Validation instances (20)
- `--seeds`: Random seeds for statistical significance (5 runs)
- `--output`: Results CSV file

**Estimated Time:** ~15 hours (GPU), ~50 hours (CPU)

**Expected Results:**

| Method | Total Cost ($) | Violations (%) | Unmet Demand (%) | Training Time (h) |
|--------|----------------|----------------|------------------|-------------------|
| PPO-Flat | 14,200 ± 380 | 18.3 | 8.2 | 2.8 |
| PPO-MLP | 13,500 ± 290 | 12.1 | 6.5 | 3.1 |
| **PPO-GNN** | **12,450 ± 210** | **3.8** | **4.1** | **3.5** |

**Verify results:**
```bash
python scripts/compare_results.py \
    --result_file results/ablation/table_5_2.csv \
    --reference_file results/reference/table_5_2_reference.csv \
    --tolerance 5%
```

---

### Table 5.3: Generalization Across Network Sizes

**Test generalization to unseen network sizes:**
```bash
python experiments/generalization_test.py \
    --trained_size 50 \
    --test_sizes 30 50 70 \
    --methods ppo_flat ppo_mlp ppo_gnn \
    --num_test_instances 10 \
    --checkpoint_dir checkpoints/ \
    --output results/ablation/table_5_3.csv
```

**Estimated Time:** ~2 hours (using pre-trained models)

**Expected Results:**

| Method | 30 nodes (smaller) | 50 nodes (trained) | 70 nodes (larger) |
|--------|-------------------|-------------------|-------------------|
| PPO-Flat | +23% gap | baseline | +45% gap |
| PPO-MLP | +15% gap | baseline | +32% gap |
| **PPO-GNN** | **+8% gap** | baseline | **+12% gap** |

---

### Individual Ablation Experiments

**Train individual models:**
```bash
# PPO-Flat (baseline)
python src/training/train_classical_ppo.py \
    --config configs/classical_ppo_config.yaml \
    --dataset data/synthetic_networks/medium_50_nodes \
    --output checkpoints/ppo_flat_best.pth

# PPO-MLP (MLP encoder, no message-passing)
python src/training/train_ppo_mlp.py \
    --config configs/ppo_mlp_config.yaml \
    --dataset data/synthetic_networks/medium_50_nodes \
    --output checkpoints/ppo_mlp_best.pth

# PPO-GNN (full framework)
python src/training/train_ppo_gnn.py \
    --config configs/ppo_gnn_config.yaml \
    --dataset data/synthetic_networks/medium_50_nodes \
    --output checkpoints/ppo_gnn_best.pth
```

**Monitor training:**
```bash
tensorboard --logdir runs/
```

Navigate to `http://localhost:6006` to view training curves.

---

## 📈 Section 5.3: Overall Performance Comparison

### Table 5.4: Performance Comparison (Large-Scale, 100 Nodes)

**Reproduce complete benchmark:**
```bash
python experiments/benchmark_comparison.py \
    --methods gurobi ppo_gnn classical_ppo clarke_wright \
    --dataset data/synthetic_networks/large_100_nodes \
    --num_test_instances 20 \
    --gurobi_time_limit 7200 \
    --gurobi_threads 8 \
    --output results/benchmark/table_5_4.csv
```

**Arguments:**
- `--methods`: All comparison methods
- `--dataset`: Large-scale instances (100 nodes, 15 vehicles)
- `--gurobi_time_limit`: 2-hour time limit (7200 seconds)
- `--gurobi_threads`: Number of CPU threads for Gurobi

**Estimated Time:** ~24 hours (includes Gurobi solving)

**Expected Results:**

| Method | Total Cost ($) | Gap (%) | Unmet Demand (%) | Solve Time (s) | Violations (%) |
|--------|----------------|---------|------------------|----------------|----------------|
| Gurobi | 12,560* | 0.0 | 0 | 7,200+ | 0 |
| **PPO-GNN** | **12,800** | **1.9** | **2** | **180** | **1** |
| Classical DRL | 13,800 | 9.9 | 8 | 120 | 6 |
| Heuristic | 14,200 | 13.1 | 5 | 60 | 3 |

*Best solution found (MIP gap: 1.2%)

---

### Individual Method Evaluation

**Evaluate specific methods:**
```bash
# PPO-GNN
python src/evaluation/evaluate.py \
    --method ppo_gnn \
    --checkpoint checkpoints/ppo_gnn_best.pth \
    --dataset data/synthetic_networks/large_100_nodes \
    --output results/benchmark/ppo_gnn_results.json

# Classical PPO
python src/evaluation/evaluate.py \
    --method classical_ppo \
    --checkpoint checkpoints/classical_ppo_best.pth \
    --dataset data/synthetic_networks/large_100_nodes \
    --output results/benchmark/classical_ppo_results.json

# Clarke-Wright Heuristic
python src/baselines/clarke_wright.py \
    --dataset data/synthetic_networks/large_100_nodes \
    --output results/benchmark/clarke_wright_results.json

# Gurobi (requires license)
python src/baselines/gurobi_solver.py \
    --dataset data/synthetic_networks/large_100_nodes \
    --time_limit 7200 \
    --mip_gap 0.01 \
    --threads 8 \
    --output results/benchmark/gurobi_results.json
```

---

## 🎯 Section 5.4: Detailed Gurobi Comparison

### Table 5.5: Multi-Scale Comparison with Gurobi

**Reproduce across all problem scales:**
```bash
python experiments/optimality_gap_analysis.py \
    --sizes 10 50 100 200 \
    --methods gurobi ppo_gnn \
    --num_instances_per_size 10 \
    --gurobi_time_limit 14400 \
    --output results/gaps/table_5_5.csv
```

**Estimated Time:** ~30 hours (Gurobi dominates time)

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

---

### Table 5.6: LP Lower Bounds

**Compute LP relaxation lower bounds:**
```bash
python experiments/lp_relaxation_bounds.py \
    --sizes 10 50 100 200 \
    --num_instances_per_size 10 \
    --methods lp_relaxation gurobi ppo_gnn \
    --output results/lp_bounds/table_5_6.csv
```

**Estimated Time:** ~8 hours

**Expected Results:**

| Instance Size | LP Lower Bound ($) | Gurobi Upper ($) | PPO-GNN ($) | Gap Range (%) |
|---------------|--------------------|------------------|-------------|---------------|
| Small (10) | 4,280 | 4,320 | 4,390 | [0.9, 2.6] |
| Medium (50) | 8,520 | 8,650 | 8,820 | [1.5, 3.5] |
| Large (100) | 12,180 | 12,560 | 12,800 | [2.0, 5.1] |
| Very Large (200) | 23,400 | N/A | 24,500 | [4.7, N/A] |

**Interpretation:**
- **Gap Range:** [gap vs. upper bound, gap vs. lower bound]
- **LP bounds** provide guaranteed lower bounds on optimal integer solution
- **LP-IP gap** of 0.9-1.5% validates model tightness

---

### Detailed Gap Analysis

**Compute verified, bounded, and LP-based gaps:**
```bash
python src/evaluation/optimality_gap.py \
    --instance data/synthetic_networks/large_100_nodes/instance_01.json \
    --ppo_gnn_checkpoint checkpoints/ppo_gnn_best.pth \
    --gurobi_time_limit 7200 \
    --compute_lp_bound \
    --output results/gaps/detailed_gap_analysis.json
```

**Output includes:**
- Verified gap (if Gurobi proves optimality)
- Bounded gap range (via MIP gap)
- LP lower bound gap
- Computational time comparison

---

## 📊 Section 5.5: Visual Analysis

### Figure 5.1: Cost and Unmet Demand Comparison

**Generate comparison chart:**
```bash
python src/utils/visualization.py \
    --plot_type cost_unmet_comparison \
    --results_files results/benchmark/table_5_4.csv \
    --methods gurobi ppo_gnn classical_ppo clarke_wright \
    --output results/figures/cost_unmet.png \
    --format png \
    --dpi 300
```

**Expected output:**
- Bar chart comparing total cost across methods
- Bar chart comparing unmet demand percentage
- Saved as `results/figures/cost_unmet.png`

---

### Figure 5.2: Optimized Delivery Routes

**Visualize routes for specific instance:**
```bash
python src/utils/visualization.py \
    --plot_type route_comparison \
    --instance data/synthetic_networks/large_100_nodes/instance_01.json \
    --methods ppo_gnn classical_ppo clarke_wright \
    --checkpoints checkpoints/ppo_gnn_best.pth checkpoints/classical_ppo_best.pth \
    --output results/figures/routes.png \
    --format png \
    --dpi 300
```

**Expected output:**
- Side-by-side route visualizations
- Network graph with colored routes
- Metrics overlay (distance, time, violations)
- Saved as `results/figures/routes.png`

---

## ⏱️ Hardware and Time Requirements

### Recommended Hardware

| Component | Recommended | Minimum |
|-----------|-------------|---------|
| **GPU** | NVIDIA RTX 3090 (24GB) | NVIDIA GTX 1080 (8GB) |
| **CPU** | Intel i9-12900K (16 cores) | Intel i7 (8 cores) |
| **RAM** | 64 GB | 16 GB |
| **Storage** | 100 GB SSD | 50 GB HDD |

### Time Estimates

| Experiment | GPU (RTX 3090) | GPU (GTX 1080) | CPU-only |
|------------|----------------|----------------|----------|
| **Table 5.2** (Ablation) | 15 hours | 30 hours | 50 hours |
| **Table 5.3** (Generalization) | 2 hours | 4 hours | 8 hours |
| **Table 5.4** (Benchmark) | 24 hours | 40 hours | 80 hours |
| **Table 5.5** (Gurobi Multi-Scale) | 30 hours | N/A | N/A |
| **Table 5.6** (LP Bounds) | 8 hours | 12 hours | 20 hours |
| **Figures** | 1 hour | 2 hours | 4 hours |
| **Total (with Gurobi)** | **~48 hours** | **~72 hours** | **~120 hours** |
| **Total (no Gurobi)** | **~36 hours** | **~50 hours** | **~90 hours** |

---

## 💾 Pre-trained Models

Skip training by using pre-trained checkpoints:

### Download Pre-trained Models
```bash
# Download from release
wget https://github.com/YourUsername/PPO-GNN-Fuel-Delivery/releases/download/v1.0/checkpoints.zip

# Extract
unzip checkpoints.zip -d checkpoints/

# Verify checksums
md5sum -c checkpoints/checksums.md5
```

**Available checkpoints:**
- `ppo_gnn_best.pth` - Best PPO-GNN model (trained on 100-node instances)
- `classical_ppo_best.pth` - Classical PPO baseline
- `ppo_mlp_best.pth` - PPO-MLP baseline
- `ppo_gnn_50nodes.pth` - PPO-GNN trained on 50-node instances (for generalization test)

### Use Pre-trained Models
```bash
# Evaluate PPO-GNN (skip training)
python src/evaluation/evaluate.py \
    --method ppo_gnn \
    --checkpoint checkpoints/ppo_gnn_best.pth \
    --dataset data/synthetic_networks/large_100_nodes \
    --output results/ppo_gnn_eval.json
```

---

## 🔄 Parallel Execution

Speed up reproduction using parallel execution:

### Run Multiple Seeds in Parallel
```bash
# Train 5 models with different seeds in parallel
parallel -j 5 python src/training/train_ppo_gnn.py \
    --config configs/ppo_gnn_config.yaml \
    --seed {} \
    --output checkpoints/ppo_gnn_seed_{}.pth \
    ::: 42 123 456 789 1011
```

### Distributed Training (Multiple GPUs)
```bash
# Use PyTorch distributed training
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    src/training/train_ppo_gnn_distributed.py \
    --config configs/ppo_gnn_config.yaml
```

---

## 🔍 Troubleshooting

### Issue 1: Out of Memory During Training

**Solution:**
Reduce batch size in config:
```yaml
# configs/ppo_gnn_config.yaml
training:
  batch_size: 128  # Reduced from 256
```

---

### Issue 2: Gurobi Timeout

**Solution:**
Increase time limit or reduce problem size:
```bash
python src/baselines/gurobi_solver.py \
    --time_limit 14400 \  # 4 hours instead of 2
    --mip_gap 0.05        # Allow 5% gap for faster termination
```

---

### Issue 3: Training Divergence

**Solution:**
Reduce learning rates:
```yaml
# configs/ppo_gnn_config.yaml
policy:
  learning_rate: 1.0e-4  # Reduced from 3e-4
```

---

### Issue 4: Results Don't Match Paper

**Possible causes:**
1. **Random seed:** Ensure consistent seeds across runs
2. **Hardware differences:** GPU/CPU may produce slightly different results
3. **Package versions:** Verify PyTorch/PyG versions match `requirements.txt`

**Solution:**
```bash
# Check package versions
pip list | grep torch
pip list | grep numpy

# Use reference seeds
--seeds 42 123 456 789 1011
```

---

## 📋 Verification Checklist

After reproduction, verify results:
```bash
# Run verification script
python scripts/verify_reproduction.py \
    --results_dir results/ \
    --reference_dir results/reference/ \
    --tolerance 5%
```

**Expected output:**
```
✓ Table 5.2: Ablation study results within 5% tolerance
✓ Table 5.3: Generalization results within 5% tolerance
✓ Table 5.4: Benchmark comparison within 5% tolerance
✓ Table 5.5: Gurobi comparison within 5% tolerance
✓ Table 5.6: LP bounds within 5% tolerance
✓ Figure 5.1: Generated successfully
✓ Figure 5.2: Generated successfully

All results verified! ✓
```

---

## 📧 Support

If you encounter issues during reproduction:

1. **Check logs:**
```bash
   cat logs/reproduction.log
```

2. **Open GitHub issue:** Include error message, OS, hardware specs

3. **Contact authors:** See README.md for contact information

---

## 📚 Additional Resources

- **Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)
- **Supplementary Materials:** `supplementary/`
- **API Documentation:** [API.md](API.md)
- **FAQ:** [FAQ.md](FAQ.md)

---

**Happy reproducing! 🚀**

For questions about specific experiments, refer to the experiment-specific README files in `experiments/`.
