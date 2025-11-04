# Synthetic Datasets Documentation

This directory contains synthetic hydrogen delivery networks for benchmarking.

## 📊 Dataset Structure
```
data/
├── synthetic_networks/
│   ├── small_10_nodes/           # 10 nodes, ~30 edges
│   ├── medium_50_nodes/          # 50 nodes, ~150 edges
│   ├── large_100_nodes/          # 100 nodes, ~300 edges
│   └── xlarge_200_nodes/         # 200 nodes, ~600 edges
└── generation_scripts/           # Generation utilities
```

## 🔧 Generating Datasets

### 1. Generate Networks
```bash
cd data/generation_scripts/
python generate_networks.py --output-dir ../synthetic_networks/ --seed 42
```

This creates network topologies with:
- Node coordinates (random or clustered)
- Distance/time matrices
- Network metadata

### 2. Generate Demands
```bash
python generate_demands.py --networks-dir ../synthetic_networks/ --seed 42
```

Creates stochastic demands for each network:
- **Low demand**: 50-100 kg H₂ per node
- **Medium demand**: 100-200 kg H₂ per node
- **High demand**: 200-400 kg H₂ per node

Each demand file includes:
- Mean demand μᵢ
- Standard deviation σᵢ (CV = 0.2)
- 100 demand scenarios for evaluation

### 3. Generate Constraints
```bash
python generate_constraints.py --networks-dir ../synthetic_networks/ --seed 42
```

Creates operational constraints:
- Vehicle capacities (1.5× average demand)
- Time windows (0-480 minutes)
- Cost parameters (fuel, driver, fixed)
- Service times

## 📁 File Formats

### network.json
```json
{
  "name": "medium_50_nodes",
  "num_nodes": 50,
  "num_edges": 1225,
  "nodes": [
    {
      "id": 0,
      "coordinates": [50.0, 50.0],
      "type": "depot"
    },
    ...
  ],
  "distance_matrix": [[...]],
  "time_matrix": [[...]]
}
```

### demands_medium.json
```json
{
  "demand_level": "medium",
  "cv": 0.2,
  "nodes": [
    {
      "node_id": 1,
      "demand_mean": 150.5,
      "demand_std": 30.1
    },
    ...
  ],
  "scenarios": [...]
}
```

### constraints_medium.json
```json
{
  "vehicles": [
    {
      "id": 0,
      "capacity": 225.0,
      "fuel_capacity": 100.0,
      "fixed_cost": 50.0
    },
    ...
  ],
  "time_windows": [...],
  "costs": {...}
}
```

## 📈 Dataset Statistics

| Network | Nodes | Edges | Avg Distance | Total Demand (medium) |
|---------|-------|-------|--------------|----------------------|
| Small   | 10    | 45    | ~35 km       | ~1,350 kg            |
| Medium  | 50    | 1,225 | ~40 km       | ~7,500 kg            |
| Large   | 100   | 4,950 | ~42 km       | ~15,000 kg           |
| XLarge  | 200   | 19,900| ~45 km       | ~30,000 kg           |

## 🎯 Usage in Experiments
```python
from src.utils.data_utils import DataLoader

# Load instance
loader = DataLoader(data_dir='data/synthetic_networks/')
network = loader.load_instances('medium_50_nodes/network.json')[0]
demands = loader.load_instances('medium_50_nodes/demands_medium.json')[0]
constraints = loader.load_instances('medium_50_nodes/constraints_medium.json')[0]

# Combine into instance
instance = {
    'network': network,
    'demands': demands,
    'constraints': constraints
}
```

## 🔄 Regenerating Data

To regenerate all datasets with new seed:
```bash
cd data/generation_scripts/
./generate_all.sh --seed 123
```

Or individually:
```bash
python generate_networks.py --seed 123
python generate_demands.py --seed 123
python generate_constraints.py --seed 123
```

## 📝 Notes

- All distances in kilometers
- All times in minutes
- All demands in kg H₂
- Depot always at node 0
- Complete graphs (all edges present)
- Stochastic demands: dᵢ ~ N(μᵢ, σᵢ²)
