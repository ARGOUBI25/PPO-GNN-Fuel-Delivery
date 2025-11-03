# Frequently Asked Questions (FAQ)

Common questions about the PPO-GNN framework for fuel delivery optimization.

---

## 📋 Table of Contents

1. [General Questions](#general-questions)
2. [Installation & Setup](#installation--setup)
3. [Training & Usage](#training--usage)
4. [Experiments & Results](#experiments--results)
5. [Technical Details](#technical-details)
6. [Troubleshooting](#troubleshooting)
7. [Performance & Optimization](#performance--optimization)
8. [Comparison & Baselines](#comparison--baselines)
9. [Extending the Framework](#extending-the-framework)
10. [Licensing & Citation](#licensing--citation)

---

## 🌟 General Questions

### Q1: What is PPO-GNN?

**A:** PPO-GNN is a hybrid framework that combines Proximal Policy Optimization (PPO), Graph Neural Networks (GNNs), and a novel three-tier constraint validation mechanism for solving stochastic vehicle routing problems in fuel delivery operations. It achieves near-optimal solutions (1.6-2.0% gap) while being 24-51× faster than commercial solvers.

---

### Q2: What makes this framework novel?

**A:** Three key innovations:

1. **Closed-loop constraint validation:** Uses the deterministic model as a dynamic feasibility oracle that continuously refines the policy, unlike standard one-way validation (accept/reject).

2. **Three-tier adaptive penalty system:** Graduated responses (tolerance ≤5%, fine-tuning 5-25%, re-training >25%) enable constraint-aware learning without suppressing exploration.

3. **GNN-based constraint encoding:** Embeds constraint structures into state representation, enabling inherently feasible policies rather than relying solely on post-hoc corrections.

---

### Q3: What problem does this solve?

**A:** Stochastic fuel delivery optimization with:
- Uncertain demand (modeled as random variables with μ, σ)
- Heterogeneous vehicle fleet (different capacities, fuel types)
- Time window constraints
- Vehicle capacity limits
- Multi-objective optimization (cost, demand satisfaction, constraint adherence)

---

### Q4: Who should use this framework?

**A:** 

- **Researchers:** Working on vehicle routing, reinforcement learning, or constraint-aware optimization
- **Logistics Companies:** Optimizing fuel distribution, parcel delivery, or waste collection
- **Students:** Learning about hybrid AI approaches for combinatorial optimization
- **Practitioners:** Deploying real-world routing solutions

---

### Q5: What are the main results?

**A:**

| Metric | Value |
|--------|-------|
| **Verified optimality gap** | 1.6-2.0% (proven optimal instances) |
| **Bounded gap** | 1.9-3.1% (large instances) |
| **Speedup vs. Gurobi** | 24-51× faster |
| **Cost reduction vs. DRL** | 7.2% |
| **Cost reduction vs. heuristics** | 9.9% |
| **GNN benefit** | 12.3% cost, 79.2% fewer violations |

---

## 🔧 Installation & Setup

### Q6: What are the system requirements?

**A:**

**Minimum:**
- Python 3.8+
- 8 GB RAM
- 5 GB disk space
- CPU (Intel i5 or equivalent)

**Recommended:**
- Python 3.8-3.10
- 32 GB RAM
- NVIDIA GPU (RTX 3090 or equivalent, 8+ GB VRAM)
- CUDA 11.6+
- 100 GB SSD

---

### Q7: Can I run this on CPU only?

**A:** Yes, but training will be ~10× slower. For inference (using pre-trained models), CPU is acceptable. Modify config:
```yaml
experiment:
  device: "cpu"
```

---

### Q8: Do I need Gurobi?

**A:** **No**, Gurobi is optional and only needed for:
- Exact solver comparison (Tables 5.4, 5.5, 5.6)
- LP relaxation lower bounds (Table 5.6)

The PPO-GNN framework works fully without Gurobi. You can still compare against heuristics and classical DRL.

---

### Q9: How do I get a Gurobi license?

**A:** 

**Academic (Free):**
1. Register at [Gurobi Academia](https://www.gurobi.com/academia/)
2. Request free academic license (requires .edu email)
3. Install: `pip install gurobipy`
4. Activate: `grbgetkey YOUR-LICENSE-KEY`

**Commercial:** Visit [Gurobi Downloads](https://www.gurobi.com/downloads/)

---

### Q10: Installation failed with "torch-scatter not found"

**A:** PyTorch Geometric requires specific versions. Install in order:
```bash
# 1. Install PyTorch first
pip install torch==1.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# 2. Install PyG dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-1.12.0+cu116.html

# 3. Install PyG
pip install torch-geometric==2.1.0
```

Replace `cu116` with your CUDA version (`cu113`, `cu102`, or `cpu`).

---

## 🏋️ Training & Usage

### Q11: How long does training take?

**A:**

| Setup | Time |
|-------|------|
| **PPO-GNN (GPU)** | ~15 hours (50,000 episodes, 100 nodes) |
| **PPO-GNN (CPU)** | ~50 hours |
| **Ablation study** | ~15 hours (GPU, 3 methods) |
| **Full benchmark** | ~48 hours (GPU + Gurobi) |

---

### Q12: Can I use pre-trained models?

**A:** Yes! Download from releases:
```bash
wget https://github.com/ARGOUBI25/PPO-GNN-Fuel-Delivery/releases/download/v1.0/checkpoints.zip
unzip checkpoints.zip -d checkpoints/
```

Then evaluate directly:
```bash
python src/evaluation/evaluate.py \
    --checkpoint checkpoints/ppo_gnn_best.pth \
    --dataset data/synthetic_networks/large_100_nodes
```

---

### Q13: How do I train on my own dataset?

**A:**

1. **Format your data** following `data/README_DATA.md`
2. **Generate graph representation:**
```python
   from src.utils.graph_utils import build_graph
   graph = build_graph(your_network_data)
```
3. **Update config:**
```yaml
   environment:
     network_size: <your_num_nodes>
     num_vehicles: <your_fleet_size>
```
4. **Train:**
```bash
   python src/training/train_ppo_gnn.py --config configs/your_config.yaml
```

---

### Q14: What if training doesn't converge?

**A:** Try these fixes:

1. **Reduce learning rates:**
```yaml
   policy:
     learning_rate: 1.0e-4  # Instead of 3e-4
```

2. **Increase batch size:**
```yaml
   training:
     batch_size: 512  # Instead of 256
```

3. **Add gradient clipping:**
```yaml
   training:
     gradient_clip: 0.5
```

4. **Check logs:**
```bash
   tensorboard --logdir runs/
```
   Look for NaN losses, exploding gradients, or policy entropy collapse.

---

### Q15: How do I monitor training?

**A:**

**TensorBoard:**
```bash
tensorboard --logdir runs/
# Open http://localhost:6006
```

**Weights & Biases (optional):**
```yaml
logging:
  wandb: true
  wandb_project: "my-project"
```

**Logs include:**
- Policy/value losses
- Rewards per episode
- Constraint violation rates
- Optimality gap (if Gurobi available)
- Tier activation frequency

---

## 📊 Experiments & Results

### Q16: Why is my optimality gap different from the paper?

**A:** Expected variations (±2-3%) due to:

1. **Random seeds:** Use same seeds for reproducibility:
```bash
   --seeds 42 123 456 789 1011
```

2. **Hardware differences:** GPU/CPU produce slightly different floating-point results

3. **Package versions:** Ensure exact versions from `requirements.txt`

4. **Hyperparameters:** Verify config matches Table 5.1

If gap > 5% different, open a GitHub issue.

---

### Q17: How do I reproduce Table 5.2 (Ablation Study)?

**A:**
```bash
python experiments/ablation_study.py \
    --methods ppo_flat ppo_mlp ppo_gnn \
    --dataset data/synthetic_networks/medium_50_nodes \
    --num_episodes 50000 \
    --seeds 42 123 456 789 1011 \
    --output results/ablation/table_5_2.csv
```

Expected time: ~15 hours (GPU)

---

### Q18: Why does Gurobi fail on 200-node instances?

**A:** This is expected! The deterministic MILP becomes computationally intractable for large-scale instances:

- **10 nodes:** Gurobi proves optimality in ~285s
- **50 nodes:** Gurobi proves optimality in ~4,580s
- **100 nodes:** Gurobi finds best solution in 7,200s (MIP gap 1.2%)
- **200 nodes:** Gurobi fails within 4 hours (no feasible solution)

This demonstrates **PPO-GNN's scalability advantage** — it solves 200-node instances in 7 minutes.

---

### Q19: What are "verified" vs "bounded" vs "LP-based" gaps?

**A:**

| Type | When | Calculation |
|------|------|-------------|
| **Verified gap** | Gurobi proves optimality (MIP gap = 0%) | `(C_PPO - C_optimal) / C_optimal` |
| **Bounded gap** | Gurobi terminates with MIP gap > 0% | Range: `[gap_vs_upper, gap_vs_lower]` |
| **LP-based gap** | LP relaxation computed | `(C_PPO - C_LP) / C_LP` |

**Example (100 nodes):**
- Gurobi best: $12,560 (MIP gap 1.2%)
- PPO-GNN: $12,800
- Bounded range: [1.9%, 3.1%]

---

### Q20: Why is Clarke-Wright's gap so large (13.1%)?

**A:** This is **consistent with literature** for stochastic VRP with time windows. Clarke-Wright is a greedy heuristic that:

1. Makes locally-optimal decisions (immediate distance savings)
2. Cannot anticipate future demand variations
3. Doesn't account for global network effects

PPO-GNN learns anticipatory policies through 50,000 episodes, enabling globally-informed decisions.

**The 9.9% improvement** (from 13.1% to 1.9%) demonstrates substantial value of learning-based optimization.

---

## 🔬 Technical Details

### Q21: What is the three-tier validation system?

**A:**

| Tier | Threshold | Action | Penalty | Episodes |
|------|-----------|--------|---------|----------|
| **1** | V ≤ 5% | Tolerance (continue) | No change | - |
| **2** | 5% < V ≤ 25% | Fine-tuning | λ_c ← 1.5λ_c | 1,000 |
| **3** | V > 25% | Re-training | λ_c ← 10λ_c, reset θ | 10,000 |

**Key innovation:** Graduated responses enable constraint learning without suppressing exploration.

---

### Q22: How does GNN improve performance?

**A:** GNNs capture spatial dependencies through message-passing:
```
h_i^(l+1) = σ(W^(l) h_i^(l) + Σ_{j∈N(i)} 1/|N(i)| W^(l) h_j^(l))
```

**Benefits:**
- **12.3% cost reduction** vs flat features
- **79.2% fewer violations** vs flat features
- **Better generalization:** 8-12% degradation vs 23-45% for flat features

**Why?** GNNs aggregate neighborhood information, enabling globally-informed decisions rather than locally-optimal choices.

---

### Q23: What are the hyperparameters from Table 5.1?

**A:**
```yaml
Policy LR (α_θ): 3×10⁻⁴
Value LR (α_φ): 1×10⁻³
GNN LR (α_ψ): 1×10⁻⁴
Clipping (ε): 0.2
Discount (γ): 0.99
GAE (λ): 0.95
Batch size: 256
Epochs (K): 10
Max episodes: 50,000
GNN layers: 3
Hidden dim: 128
Reward weights: λ₁=1.0, λ₂=0.5, λ₃=0.8, λ₄=1.2, λ₅=2.0
```

All in `configs/ppo_gnn_config.yaml`.

---

### Q24: What is the reward function?

**A:** Multi-objective reward (Equation 2):
```
R(s,a) = -λ₁C_total - λ₂C_dispersion - λ₃C_delay - λ₄C_unmet - λ₅C_constraint
```

Where:
- **C_total:** Operational costs (fuel, time, vehicle usage)
- **C_dispersion:** Vehicle distribution complexity
- **C_delay:** Late delivery penalties
- **C_unmet:** Unsatisfied demand penalties
- **C_constraint:** Constraint violation penalties

---

### Q25: How does PPO work?

**A:** PPO updates policy via clipped objective (Equation 8):
```
L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
```

**Trust region principle:** Limits policy changes to prevent catastrophic updates. The clipping parameter ε=0.2 restricts probability ratio to [0.8, 1.2].

---

## 🛠️ Troubleshooting

### Q26: CUDA out of memory error

**A:**

1. **Reduce batch size:**
```yaml
   training:
     batch_size: 128  # Instead of 256
```

2. **Enable gradient accumulation:**
```yaml
   training:
     gradient_accumulation_steps: 2
```

3. **Use mixed precision:**
```yaml
   training:
     mixed_precision: true
```

4. **Reduce network size** or **use smaller GPU-friendly models**

---

### Q27: Training is very slow on CPU

**A:**

**Speed up CPU training:**

1. **Use fewer episodes:**
```yaml
   training:
     max_episodes: 10000  # Instead of 50000
```

2. **Parallelize data loading:**
```yaml
   experiment:
     num_workers: 4
```

3. **Reduce network size** for experimentation

4. **Use pre-trained models** for evaluation

**Or:** Rent cloud GPU (AWS, Google Colab, Lambda Labs)

---

### Q28: ValueError: "graph has no edges"

**A:** Check dataset format:
```python
# Verify network structure
import json
with open('data/synthetic_networks/large_100_nodes/instance_01.json') as f:
    network = json.load(f)
    print(f"Nodes: {len(network['nodes'])}")
    print(f"Edges: {len(network['edges'])}")
```

Ensure `edges` list is non-empty. See `data/README_DATA.md` for format specification.

---

### Q29: ModuleNotFoundError for custom module

**A:**

1. **Install package in development mode:**
```bash
   pip install -e .
```

2. **Add to PYTHONPATH:**
```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/PPO-GNN-Fuel-Delivery"
```

3. **Use absolute imports:**
```python
   from src.models.ppo_gnn import PPOGNN
```

---

### Q30: Gurobi license error

**A:**

1. **Check license file:**
```bash
   ls ~/.gurobi/
   cat ~/.gurobi/gurobi.lic
```

2. **Verify environment variable:**
```bash
   echo $GRB_LICENSE_FILE
```

3. **Re-activate license:**
```bash
   grbgetkey YOUR-LICENSE-KEY
```

4. **Academic licenses expire** — renew annually at [Gurobi Academia](https://www.gurobi.com/academia/)

---

## ⚡ Performance & Optimization

### Q31: How can I speed up training?

**A:**

1. **Use multiple GPUs:**
```bash
   python -m torch.distributed.launch --nproc_per_node=4 src/training/train_ppo_gnn.py
```

2. **Enable mixed precision:**
```yaml
   training:
     mixed_precision: true
```

3. **Increase batch size** (if GPU memory allows)

4. **Reduce validation frequency:**
```yaml
   validation:
     validation_frequency: 5000  # Instead of 1000
```

5. **Use compiled models** (PyTorch 2.0):
```python
   model = torch.compile(model)
```

---

### Q32: Can I train on multiple datasets simultaneously?

**A:** Yes, using parallel training:
```bash
parallel -j 4 python src/training/train_ppo_gnn.py \
    --dataset {} \
    --output checkpoints/{/}.pth \
    ::: data/synthetic_networks/*/
```

Or use distributed training with different datasets per GPU.

---

### Q33: How much VRAM is needed?

**A:**

| Network Size | Batch Size | VRAM Required |
|--------------|------------|---------------|
| 10 nodes | 256 | ~2 GB |
| 50 nodes | 256 | ~6 GB |
| 100 nodes | 256 | ~12 GB |
| 200 nodes | 256 | ~24 GB |

**Reduce VRAM usage:**
- Reduce batch size
- Use gradient checkpointing
- Enable mixed precision (FP16)

---

## 🔄 Comparison & Baselines

### Q34: Why compare against Clarke-Wright and not other heuristics?

**A:** Clarke-Wright was selected because it:

1. **Widely deployed** in commercial logistics systems
2. **Computationally efficient** (~60s for 100-node instances)
3. **Representative benchmark** with 13.1% gap consistent with literature

**Other heuristics** (Sweep, Nearest Neighbor) perform similarly or worse. Clarke-Wright is the strongest heuristic baseline.

---

### Q35: Why not compare against other DRL algorithms (A3C, DDPG, SAC)?

**A:** Our focus is evaluating **GNN integration benefit within PPO**, not comparing DRL algorithms. PPO is state-of-the-art for combinatorial optimization (established by prior work).

**Classical PPO baseline** (same PPO, no GNN) isolates GNN's impact through controlled comparison.

---

### Q36: How does PPO-GNN compare to Attention-based models?

**A:** Attention mechanisms and GNNs serve different purposes:

- **GNNs:** Capture spatial structure via message-passing on explicit graph topology
- **Attention:** Learn implicit relationships via query-key-value mechanisms

**For fuel delivery:** Explicit graph structure (roads, distances) makes GNNs more natural. Future work could explore GNN + Attention hybrids.

---

## 🔧 Extending the Framework

### Q37: Can I apply this to other routing problems?

**A:** **Yes!** The framework is designed for general VRP variants:

**Tested on:**
- Fuel delivery (paper focus)
- Parcel delivery
- Waste collection

**Adaptable to:**
- Multi-depot VRP
- Pickup and delivery
- Time-dependent travel times
- Electric vehicle routing

**Steps:**
1. Define your problem constraints
2. Update reward function
3. Modify environment accordingly

---

### Q38: How do I add a new constraint?

**A:**

1. **Define constraint in deterministic model:**
```python
   # src/validation/deterministic_model.py
   def check_new_constraint(self, solution):
       # Your constraint logic
       return is_satisfied, violation_amount
```

2. **Add to violation analysis:**
```python
   # src/validation/violation_analysis.py
   violations['new_constraint'] = compute_new_constraint_violation(solution)
```

3. **Update reward weights:**
```yaml
   # configs/ppo_gnn_config.yaml
   constraints:
     new_constraint: 1.5  # Importance weight
```

---

### Q39: Can I use this with real-world data?

**A:** **Yes**, but requires:

1. **Data preprocessing:** Convert to required format (see `data/README_DATA.md`)
2. **Demand modeling:** Estimate μ, σ from historical data
3. **Calibration:** Tune cost parameters, time windows to match reality
4. **Validation:** Test on held-out historical instances

**See `notebooks/04_real_world_adaptation.ipynb`** (if available) for guidance.

---

### Q40: How do I add a new baseline method?

**A:**

1. **Implement in `src/baselines/`:**
```python
   # src/baselines/my_method.py
   class MyMethod:
       def solve(self, network):
           # Your algorithm
           return routes, cost
```

2. **Register in benchmark:**
```python
   # experiments/benchmark_comparison.py
   METHODS = {
       'my_method': MyMethod,
       ...
   }
```

3. **Run comparison:**
```bash
   python experiments/benchmark_comparison.py --methods ppo_gnn my_method
```

---

## 📜 Licensing & Citation

### Q41: What is the license?

**A:** **MIT License** — free for academic and commercial use, modification, and distribution.

See [LICENSE](../LICENSE) file for full terms.

---

### Q42: How do I cite this work?

**A:**
```bibtex
@article{Argoubi2025ppo_gnn,
  title={Constraint-Aware PPO-GNN with Closed-Loop Validation for Stochastic Fuel Delivery Optimization},
  author={Majdi Argoubi},
  journal={Coming soon},
  year={2025},
  volume={XX},
  pages={XXX-XXX},
  doi={10.XXXX/XXXXX}
}
```

---

### Q43: Can I use this commercially?

**A:** **Yes**, under MIT License terms. You can:
- Use in commercial products
- Modify and redistribute
- Sublicense

**Requirements:**
- Include original copyright notice
- Include MIT License text

**No warranty provided** — see LICENSE for details.

---

### Q44: Can I contribute to the project?

**A:** **Absolutely!** Contributions welcome:

1. **Fork repository**
2. **Create feature branch:** `git checkout -b feature/amazing-feature`
3. **Commit changes:** `git commit -m 'Add amazing feature'`
4. **Push to branch:** `git push origin feature/amazing-feature`
5. **Open Pull Request**

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

### Q45: Where can I get help?

**A:**

1. **Check documentation:**
   - [README.md](../README.md)
   - [INSTALLATION.md](INSTALLATION.md)
   - [REPRODUCTION.md](REPRODUCTION.md)
   - [API.md](API.md)

2. **Search existing issues:** [GitHub Issues](https://github.com/ARGOUBI25/PPO-GNN-Fuel-Delivery/issues)

3. **Ask in discussions:** [GitHub Discussions](https://github.com/ARGOUBI25/PPO-GNN-Fuel-Delivery/discussions)

4. **Contact authors:** See README for email addresses

5. **Report bugs:** Open GitHub issue with:
   - Error message
   - System info (OS, Python version, GPU)
   - Minimal reproducible example

---

## 📚 Additional Resources

- **Paper:** Coming soon
- **Supplementary Materials:** Available in repository
- **Tutorial Videos:** Coming soon
- **Blog Post:** ** Coming soon

---

**Still have questions? Open a discussion on GitHub!** 💬
