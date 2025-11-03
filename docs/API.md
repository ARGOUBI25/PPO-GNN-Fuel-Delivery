# API Documentation

Complete API reference for the PPO-GNN Fuel Delivery Optimization framework.

---

## 📋 Table of Contents

1. [Models](#models)
   - [PPO-GNN](#ppo-gnn)
   - [GNN Encoder](#gnn-encoder)
   - [Policy Network](#policy-network)
   - [Value Network](#value-network)
   - [GAE](#generalized-advantage-estimation)
2. [Baselines](#baselines)
   - [Classical PPO](#classical-ppo)
   - [PPO-MLP](#ppo-mlp)
   - [Clarke-Wright](#clarke-wright-heuristic)
   - [Gurobi Solver](#gurobi-solver)
3. [Validation](#validation)
   - [Deterministic Model](#deterministic-model)
   - [Violation Analysis](#violation-analysis)
   - [Adaptive Penalties](#adaptive-penalties)
4. [Training](#training)
   - [PPO Trainer](#ppo-trainer)
   - [Algorithm 1 Training](#algorithm-1-training)
   - [Algorithm 2 Validation](#algorithm-2-validation)
5. [Evaluation](#evaluation)
   - [Optimality Gap](#optimality-gap)
   - [LP Relaxation](#lp-relaxation)
   - [Constraint Metrics](#constraint-metrics)
6. [Utilities](#utilities)
   - [Graph Utils](#graph-utils)
   - [Data Loader](#data-loader)
   - [Visualization](#visualization)
   - [Logger](#logger)

---

## 🧠 Models

### PPO-GNN

**Location:** `src/models/ppo_gnn.py`

Complete PPO-GNN framework integrating policy network, value network, and GNN encoder.

#### Class: `PPOGNN`
```python
class PPOGNN(nn.Module):
    """
    Complete PPO-GNN framework for fuel delivery optimization.
    
    Integrates:
    - GNN encoder for spatial feature extraction (Section 4.1)
    - Policy network π_θ for action selection (Section 4.2)
    - Value network V_φ for state evaluation (Section 4.2)
    - Three-tier constraint validation (Section 4.3)
    
    Args:
        config (dict): Configuration dictionary from ppo_gnn_config.yaml
        num_nodes (int): Number of stations in the network
        num_vehicles (int): Fleet size
        device (str): 'cuda' or 'cpu'
    
    Attributes:
        gnn_encoder (GNNEncoder): Graph neural network encoder
        policy_net (PolicyNetwork): Actor network
        value_net (ValueNetwork): Critic network
        optimizer_policy (torch.optim.Adam): Policy optimizer
        optimizer_value (torch.optim.Adam): Value optimizer
        optimizer_gnn (torch.optim.Adam): GNN optimizer
    
    Example:
        >>> config = load_config('configs/ppo_gnn_config.yaml')
        >>> model = PPOGNN(config, num_nodes=100, num_vehicles=15, device='cuda')
        >>> state = env.reset()
        >>> action, log_prob, value = model.act(state)
    """
    
    def __init__(self, config, num_nodes, num_vehicles, device='cuda'):
        """Initialize PPO-GNN framework."""
        
    def act(self, state, deterministic=False):
        """
        Select action given current state.
        
        Args:
            state (dict): State dictionary containing:
                - 'graph': torch_geometric.data.Data object
                - 'vehicle_states': Tensor [num_vehicles, state_dim]
                - 'demands': Tensor [num_nodes]
                - 'time_windows': Tensor [num_nodes, 2]
            deterministic (bool): If True, select argmax action (no sampling)
        
        Returns:
            action (torch.Tensor): Selected action [num_vehicles]
            log_prob (torch.Tensor): Log probability of action
            value (torch.Tensor): State value estimate V_φ(s)
        
        Example:
            >>> action, log_prob, value = model.act(state, deterministic=False)
        """
    
    def evaluate(self, states, actions):
        """
        Evaluate actions for PPO update (Algorithm 1, lines 33-39).
        
        Args:
            states (dict): Batch of states
            actions (torch.Tensor): Batch of actions [batch_size, num_vehicles]
        
        Returns:
            log_probs (torch.Tensor): Log probabilities [batch_size]
            values (torch.Tensor): State values [batch_size]
            entropy (torch.Tensor): Policy entropy [batch_size]
        """
    
    def update(self, rollouts, epochs=10, clip_param=0.2):
        """
        Update policy and value networks using PPO (Algorithm 1, lines 33-49).
        
        Args:
            rollouts (dict): Experience buffer containing:
                - 'states': List of states
                - 'actions': Tensor [num_steps, num_vehicles]
                - 'rewards': Tensor [num_steps]
                - 'log_probs': Tensor [num_steps]
                - 'values': Tensor [num_steps]
                - 'dones': Tensor [num_steps]
            epochs (int): Number of optimization epochs K (default: 10)
            clip_param (float): PPO clipping parameter ε (default: 0.2)
        
        Returns:
            logs (dict): Training statistics
                - 'policy_loss': Policy loss L^CLIP
                - 'value_loss': Value loss L^VF
                - 'entropy': Entropy bonus
                - 'approx_kl': Approximate KL divergence
        
        Example:
            >>> logs = model.update(rollouts, epochs=10, clip_param=0.2)
            >>> print(f"Policy Loss: {logs['policy_loss']:.4f}")
        """
    
    def save(self, path):
        """Save model checkpoint."""
        
    @classmethod
    def load(cls, path, device='cuda'):
        """Load model from checkpoint."""
```

---

### GNN Encoder

**Location:** `src/models/gnn_encoder.py`

3-layer graph neural network with message-passing (Equation 1, Section 4.1).

#### Class: `GNNEncoder`
```python
class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for spatial feature extraction.
    
    Implements message-passing mechanism (Eq. 1):
        h_i^(l+1) = σ(W^(l) h_i^(l) + Σ_{j∈N(i)} 1/|N(i)| W^(l) h_j^(l))
    
    Args:
        input_dim (int): Input node feature dimension
        hidden_dim (int): Hidden dimension (default: 128)
        num_layers (int): Number of GNN layers L (default: 3)
        activation (str): Activation function (default: 'relu')
        dropout (float): Dropout rate (default: 0.0)
        normalize (bool): Apply layer normalization (default: True)
    
    Example:
        >>> gnn = GNNEncoder(input_dim=32, hidden_dim=128, num_layers=3)
        >>> node_features = torch.randn(100, 32)
        >>> edge_index = torch.tensor([[0,1,2], [1,2,0]], dtype=torch.long)
        >>> embeddings = gnn(node_features, edge_index)
        >>> print(embeddings.shape)  # [100, 128]
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, 
                 activation='relu', dropout=0.0, normalize=True):
        """Initialize GNN encoder."""
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass through GNN layers.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, input_dim]
            edge_index (torch.LongTensor): Edge indices [2, num_edges]
            edge_attr (torch.Tensor, optional): Edge features [num_edges, edge_dim]
        
        Returns:
            h (torch.Tensor): Node embeddings [num_nodes, hidden_dim]
        """
```

---

### Policy Network

**Location:** `src/models/policy_network.py`

Actor network π_θ for action selection (Section 4.2).

#### Class: `PolicyNetwork`
```python
class PolicyNetwork(nn.Module):
    """
    Policy network (actor) for action selection.
    
    Maps state representations to action probabilities.
    
    Args:
        input_dim (int): State embedding dimension
        hidden_dims (list): Hidden layer dimensions (default: [256, 128, 64])
        num_actions (int): Action space size
        activation (str): Activation function (default: 'relu')
        dropout (float): Dropout rate (default: 0.0)
    
    Example:
        >>> policy = PolicyNetwork(input_dim=128, hidden_dims=[256,128,64], num_actions=10)
        >>> state_embedding = torch.randn(1, 128)
        >>> action_probs = policy(state_embedding)
        >>> action = torch.multinomial(action_probs, 1)
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], 
                 num_actions=None, activation='relu', dropout=0.0):
        """Initialize policy network."""
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): State embedding [batch_size, input_dim]
        
        Returns:
            action_probs (torch.Tensor): Action probabilities [batch_size, num_actions]
        """
    
    def sample_action(self, state, deterministic=False):
        """
        Sample action from policy distribution.
        
        Args:
            state (torch.Tensor): State embedding [batch_size, input_dim]
            deterministic (bool): If True, return argmax action
        
        Returns:
            action (torch.Tensor): Sampled action
            log_prob (torch.Tensor): Log probability of action
        """
```

---

### Value Network

**Location:** `src/models/value_network.py`

Critic network V_φ for state evaluation (Section 4.2).

#### Class: `ValueNetwork`
```python
class ValueNetwork(nn.Module):
    """
    Value network (critic) for state evaluation.
    
    Estimates state value V_φ(s) for advantage computation.
    
    Args:
        input_dim (int): State embedding dimension
        hidden_dims (list): Hidden layer dimensions (default: [256, 128])
        activation (str): Activation function (default: 'relu')
        dropout (float): Dropout rate (default: 0.0)
    
    Example:
        >>> value_net = ValueNetwork(input_dim=128, hidden_dims=[256, 128])
        >>> state_embedding = torch.randn(1, 128)
        >>> value = value_net(state_embedding)
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128], 
                 activation='relu', dropout=0.0):
        """Initialize value network."""
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): State embedding [batch_size, input_dim]
        
        Returns:
            value (torch.Tensor): State value estimate [batch_size, 1]
        """
```

---

### Generalized Advantage Estimation

**Location:** `src/models/gae.py`

GAE for advantage computation (Equation 9, Section 4.2.3).

#### Function: `compute_gae`
```python
def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    """
    Compute Generalized Advantage Estimation (Eq. 9).
    
    A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    
    Args:
        rewards (torch.Tensor): Rewards [num_steps]
        values (torch.Tensor): State values V_φ(s) [num_steps]
        dones (torch.Tensor): Done flags [num_steps]
        gamma (float): Discount factor γ (default: 0.99)
        lambda_ (float): GAE parameter λ (default: 0.95)
    
    Returns:
        advantages (torch.Tensor): Advantages [num_steps]
        returns (torch.Tensor): Discounted returns [num_steps]
    
    Example:
        >>> advantages, returns = compute_gae(rewards, values, dones)
        >>> print(f"Mean Advantage: {advantages.mean():.4f}")
    """
```

---

## 🎯 Baselines

### Classical PPO

**Location:** `src/baselines/classical_ppo.py`

PPO with flat features (512-dim), no GNN (Section 5.1.3).

#### Class: `ClassicalPPO`
```python
class ClassicalPPO(nn.Module):
    """
    Classical PPO baseline without GNN augmentation.
    
    Uses flat feature encoding (512-dimensional vector) instead of 
    graph-based representation. Same PPO algorithm and hyperparameters 
    as PPO-GNN for fair comparison.
    
    Args:
        config (dict): Configuration from classical_ppo_config.yaml
        input_dim (int): Flattened feature dimension (default: 512)
        num_actions (int): Action space size
        device (str): 'cuda' or 'cpu'
    
    Example:
        >>> config = load_config('configs/classical_ppo_config.yaml')
        >>> model = ClassicalPPO(config, input_dim=512, num_actions=100)
    """
```

---

### PPO-MLP

**Location:** `src/baselines/ppo_mlp.py`

PPO with MLP encoder, no message-passing (Section 5.2).

#### Class: `PPOMLP`
```python
class PPOMLP(nn.Module):
    """
    PPO with MLP encoder (ablation baseline).
    
    Processes node features independently through MLP [256, 128, 64],
    then aggregates via mean pooling. No message-passing aggregation.
    
    Args:
        config (dict): Configuration from ppo_mlp_config.yaml
        node_feature_dim (int): Input features per node
        num_nodes (int): Number of nodes in network
        device (str): 'cuda' or 'cpu'
    
    Example:
        >>> model = PPOMLP(config, node_feature_dim=32, num_nodes=100)
    """
```

---

### Clarke-Wright Heuristic

**Location:** `src/baselines/clarke_wright.py`

Clarke-Wright Savings Algorithm (Section 5.1.3).

#### Class: `ClarkeWrightSolver`
```python
class ClarkeWrightSolver:
    """
    Clarke-Wright Savings Algorithm for VRP.
    
    Constructs routes by iteratively merging pairs with maximum savings:
        s_ij = d_0i + d_0j - d_ij
    
    Args:
        network (dict): Network data
        vehicle_capacity (float): Vehicle capacity Q_k
        time_windows (bool): Enable time window constraints (default: True)
    
    Example:
        >>> solver = ClarkeWrightSolver(network, vehicle_capacity=1000)
        >>> routes, cost = solver.solve()
        >>> print(f"Total Cost: ${cost:.2f}")
    """
    
    def solve(self):
        """
        Solve VRP using Clarke-Wright algorithm.
        
        Returns:
            routes (list): List of routes, each route is list of node indices
            cost (float): Total routing cost
            time (float): Computation time (seconds)
        """
```

---

### Gurobi Solver

**Location:** `src/baselines/gurobi_solver.py`

Exact MILP solver using Gurobi (Section 3.2).

#### Class: `GurobiSolver`
```python
class GurobiSolver:
    """
    Exact solver using Gurobi commercial optimizer.
    
    Implements deterministic equivalent model (Section 3.2).
    
    Args:
        network (dict): Network data
        config (dict): Solver configuration
            - time_limit (int): Time limit in seconds (default: 7200)
            - mip_gap (float): MIP gap tolerance (default: 0.01)
            - threads (int): Number of threads (default: 8)
    
    Example:
        >>> solver = GurobiSolver(network, config={'time_limit': 7200})
        >>> solution = solver.solve()
        >>> print(f"Optimal Cost: ${solution['cost']:.2f}")
        >>> print(f"MIP Gap: {solution['mip_gap']:.2%}")
    """
    
    def solve(self):
        """
        Solve MILP using Gurobi.
        
        Returns:
            solution (dict):
                - 'cost': Total cost
                - 'routes': Optimized routes
                - 'time': Solve time (seconds)
                - 'mip_gap': Final MIP gap
                - 'status': 'optimal', 'feasible', or 'infeasible'
        """
```

---

## ✅ Validation

### Deterministic Model

**Location:** `src/validation/deterministic_model.py`

Feasibility oracle (Section 4.3).

#### Class: `DeterministicModel`
```python
class DeterministicModel:
    """
    Deterministic equivalent model for feasibility checking.
    
    Serves as feasibility oracle in three-tier validation (Section 4.3).
    
    Args:
        network (dict): Network topology and constraints
        config (dict): Model configuration
    
    Example:
        >>> model = DeterministicModel(network, config)
        >>> is_feasible = model.check_feasibility(solution)
    """
    
    def check_feasibility(self, solution):
        """
        Check if solution satisfies all constraints.
        
        Evaluates constraints from Section 3.1:
        - Vehicle capacity (Constraint 1)
        - Demand satisfaction (Constraint 2)
        - Time windows (Constraint 19)
        - Sequencing (Constraints 11-17)
        
        Args:
            solution (dict): Solution to validate
        
        Returns:
            is_feasible (bool): True if all constraints satisfied
            violations (dict): Violation details per constraint type
        """
```

---

### Violation Analysis

**Location:** `src/validation/violation_analysis.py`

Compute V_total (Equation 11, Section 4.3).

#### Function: `compute_violation_score`
```python
def compute_violation_score(solution, constraints, weights):
    """
    Compute total violation score V_total (Eq. 11).
    
    V_total = Σ_{c∈C} w_c · max(0, (actual_c - limit_c) / limit_c)
    
    Args:
        solution (dict): Solution to evaluate
        constraints (dict): Constraint limits
        weights (dict): Importance weights w_c per constraint type
    
    Returns:
        V_total (float): Total violation score
        violations_by_type (dict): Breakdown by constraint type
    
    Example:
        >>> V_total, breakdown = compute_violation_score(solution, constraints, weights)
        >>> print(f"Total Violations: {V_total:.2%}")
        >>> print(f"Capacity Violations: {breakdown['capacity']:.2%}")
    """
```

---

### Adaptive Penalties

**Location:** `src/validation/adaptive_penalties.py`

Tiered penalty system (Algorithm 2, Section 4.3).

#### Class: `AdaptivePenaltyManager`
```python
class AdaptivePenaltyManager:
    """
    Three-tier adaptive penalty system.
    
    Implements graduated response based on violation severity:
    - Tier 1 (≤5%): Tolerance
    - Tier 2 (5-25%): Fine-tuning (1.5× penalties, 1,000 episodes)
    - Tier 3 (>25%): Re-training (10× penalties, reset θ, 10,000 episodes)
    
    Args:
        tier1_threshold (float): Tier 1 threshold (default: 0.05)
        tier2_threshold (float): Tier 2 threshold (default: 0.25)
        tier2_multiplier (float): Tier 2 penalty multiplier (default: 1.5)
        tier3_multiplier (float): Tier 3 penalty multiplier (default: 10.0)
    
    Example:
        >>> manager = AdaptivePenaltyManager()
        >>> tier, action = manager.assess_violations(V_total=0.15)
        >>> print(f"Tier {tier}: {action}")
    """
    
    def assess_violations(self, V_total):
        """
        Determine tier and recommended action.
        
        Args:
            V_total (float): Total violation score
        
        Returns:
            tier (int): 1, 2, or 3
            action (str): 'continue', 'fine_tune', or 're_train'
            penalty_adjustments (dict): New penalty values λ_c
        """
    
    def apply_penalties(self, reward_weights, violated_constraints, tier):
        """
        Apply tier-specific penalty adjustments.
        
        Args:
            reward_weights (dict): Current reward weights
            violated_constraints (list): List of violated constraint types
            tier (int): Current tier (1, 2, or 3)
        
        Returns:
            new_weights (dict): Adjusted reward weights
        """
```

---

## 🏋️ Training

### PPO Trainer

**Location:** `src/training/ppo_trainer.py`

Main PPO optimization loop.

#### Class: `PPOTrainer`
```python
class PPOTrainer:
    """
    PPO training coordinator.
    
    Orchestrates training loop including:
    - Rollout collection
    - PPO updates (Algorithm 1)
    - Constraint validation (Algorithm 2)
    - Checkpointing and logging
    
    Args:
        model (PPOGNN): PPO-GNN model
        env (gym.Env): Fuel delivery environment
        config (dict): Training configuration
        logger (Logger): TensorBoard logger
    
    Example:
        >>> trainer = PPOTrainer(model, env, config, logger)
        >>> trainer.train(num_episodes=50000)
    """
    
    def train(self, num_episodes=50000):
        """
        Execute complete training procedure.
        
        Implements Algorithm 1 from Section 4.2.3.
        
        Args:
            num_episodes (int): Maximum training episodes
        
        Returns:
            training_logs (dict): Training statistics
        """
    
    def collect_rollouts(self, num_steps=256):
        """
        Collect experience rollouts.
        
        Args:
            num_steps (int): Number of steps to collect
        
        Returns:
            rollouts (dict): Experience buffer
        """
```

---

## 📊 Evaluation

### Optimality Gap

**Location:** `src/evaluation/optimality_gap.py`

Compute verified, bounded, and LP-based gaps (Section 5.4).

#### Function: `compute_optimality_gap`
```python
def compute_optimality_gap(ppo_solution, gurobi_solution=None, lp_bound=None):
    """
    Compute optimality gap metrics.
    
    Three types of gaps (Section 5.4):
    1. Verified gap: vs proven optimal (if Gurobi MIP gap = 0%)
    2. Bounded gap: range via Gurobi MIP gap
    3. LP-based gap: vs LP relaxation lower bound
    
    Args:
        ppo_solution (dict): PPO-GNN solution
            - 'cost': Total cost
            - 'routes': Routes
        gurobi_solution (dict, optional): Gurobi solution
            - 'cost': Best-known cost
            - 'mip_gap': MIP gap (0.0 if proven optimal)
        lp_bound (float, optional): LP lower bound
    
    Returns:
        gaps (dict):
            - 'verified_gap': If Gurobi MIP gap = 0%
            - 'bounded_gap_range': [lower, upper] if Gurobi available
            - 'lp_gap': vs LP bound if available
            - 'gap_type': 'verified', 'bounded', or 'lp_only'
    
    Example:
        >>> gaps = compute_optimality_gap(ppo_sol, gurobi_sol, lp_bound)
        >>> print(f"Verified Gap: {gaps['verified_gap']:.2%}")
    """
```

---

### LP Relaxation

**Location:** `src/evaluation/lp_relaxation.py`

Compute LP lower bounds (Section 5.4).

#### Function: `compute_lp_lower_bound`
```python
def compute_lp_lower_bound(network, config):
    """
    Compute LP relaxation lower bound.
    
    Relaxes integer variables to continuous:
    - x_ik, v_{c,k}, w_k, s_{k,ij} ∈ [0,1] (instead of {0,1})
    - n_ik, t_{k,i} ∈ R^+ (instead of N)
    
    Args:
        network (dict): Network topology and constraints
        config (dict): Solver configuration
    
    Returns:
        lp_bound (float): LP lower bound
        solve_time (float): Solve time (seconds)
        status (str): 'optimal', 'infeasible'
    
    Example:
        >>> lp_bound, time, status = compute_lp_lower_bound(network, config)
        >>> print(f"LP Lower Bound: ${lp_bound:.2f} (solved in {time:.2f}s)")
    """
```

---

## 🛠️ Utilities

### Graph Utils

**Location:** `src/utils/graph_utils.py`

Graph construction from network data.

#### Function: `build_graph`
```python
def build_graph(network_data):
    """
    Convert network data to PyTorch Geometric graph.
    
    Args:
        network_data (dict):
            - 'nodes': List of station dictionaries
            - 'edges': List of road connections
            - 'depot': Depot information
    
    Returns:
        data (torch_geometric.data.Data): Graph object
            - x: Node features [num_nodes, node_dim]
            - edge_index: Edge indices [2, num_edges]
            - edge_attr: Edge features [num_edges, edge_dim]
    
    Example:
        >>> network = load_network('data/synthetic_networks/large_100_nodes/')
        >>> graph = build_graph(network)
        >>> print(graph)
    """
```

---

### Data Loader

**Location:** `src/utils/data_loader.py`

Dataset loading utilities.

#### Class: `FuelDeliveryDataset`
```python
class FuelDeliveryDataset(torch.utils.data.Dataset):
    """
    Dataset for fuel delivery instances.
    
    Args:
        data_dir (str): Path to dataset directory
        split (str): 'train', 'val', or 'test'
        transform (callable, optional): Data transformation
    
    Example:
        >>> dataset = FuelDeliveryDataset('data/synthetic_networks/large_100_nodes/', split='train')
        >>> loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in loader:
        >>>     graphs, targets = batch
    """
```

---

### Visualization

**Location:** `src/utils/visualization.py`

Route plotting and training curves.

#### Function: `plot_routes`
```python
def plot_routes(network, routes, save_path=None):
    """
    Visualize delivery routes on network graph.
    
    Args:
        network (dict): Network topology
        routes (list): List of routes
        save_path (str, optional): Path to save figure
    
    Example:
        >>> plot_routes(network, ppo_routes, 'results/figures/routes_ppo.png')
    """
```

#### Function: `plot_training_curves`
```python
def plot_training_curves(logs, metrics=['reward', 'loss', 'violations'], save_path=None):
    """
    Plot training curves from logs.
    
    Args:
        logs (dict): Training logs from TensorBoard
        metrics (list): Metrics to plot
        save_path (str, optional): Path to save figure
    
    Example:
        >>> logs = load_tensorboard_logs('runs/ppo_gnn/')
        >>> plot_training_curves(logs, save_path='results/figures/training.png')
    """
```

---

### Logger

**Location:** `src/utils/logger.py`

TensorBoard logging wrapper.

#### Class: `Logger`
```python
class Logger:
    """
    TensorBoard logging wrapper.
    
    Args:
        log_dir (str): Directory for logs
        experiment_name (str): Experiment name
    
    Example:
        >>> logger = Logger('runs/', 'ppo_gnn_exp1')
        >>> logger.log_scalar('reward', reward, step)
        >>> logger.log_histogram('policy_weights', model.policy_net.parameters(), step)
    """
    
    def log_scalar(self, tag, value, step):
        """Log scalar value."""
        
    def log_histogram(self, tag, values, step):
        """Log histogram of values."""
```

---

## 📚 Additional Resources

- **Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)
- **Installation Guide:** [INSTALLATION.md](INSTALLATION.md)
- **Reproduction Guide:** [REPRODUCTION.md](REPRODUCTION.md)
- **FAQ:** [FAQ.md](FAQ.md)

---

**For implementation details, see source code with inline documentation.**
