"""
Classical PPO (Baseline without GNN)
PPO with flat feature encoding (512-dimensional vector).

Section 5.1.3: Classical DRL baseline for comparison.
Same PPO algorithm and hyperparameters as PPO-GNN for fair comparison.

Author: Majdi Argoubi
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import numpy as np

import sys
sys.path.append('..')
from models.policy_network import PolicyNetwork
from models.value_network import ValueNetwork
from models.gae import compute_gae


class FlatFeatureEncoder(nn.Module):
    """
    Flat feature encoder (no GNN).
    
    Concatenates all node features, edge features, and vehicle states into
    a single 512-dimensional vector (Section 5.1.3).
    
    Args:
        input_dim: Flattened input dimension (default: 512)
        hidden_dims: MLP architecture (default: [512, 256, 128])
        activation: Activation function
        dropout: Dropout rate
        normalization: 'batch' or 'layer' normalization
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: List[int] = [512, 256, 128],
        activation: str = 'relu',
        dropout: float = 0.0,
        normalization: str = 'batch'
    ):
        super(FlatFeatureEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Activation
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'elu':
            self.activation = F.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.layers = nn.ModuleList(layers)
        
        # Normalization
        if normalization == 'batch':
            self.norms = nn.ModuleList([
                nn.BatchNorm1d(dim) for dim in hidden_dims
            ])
        elif normalization == 'layer':
            self.norms = nn.ModuleList([
                nn.LayerNorm(dim) for dim in hidden_dims
            ])
        else:
            self.norms = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode flat features.
        
        Args:
            x: Flattened features [batch_size, input_dim]
        
        Returns:
            encoding: Encoded features [batch_size, hidden_dims[-1]]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Normalization
            if self.norms is not None:
                if isinstance(self.norms[i], nn.BatchNorm1d) and x.size(0) == 1:
                    # Skip batch norm for single sample
                    pass
                else:
                    x = self.norms[i](x)
            
            x = self.activation(x)
            x = self.dropout(x)
        
        return x


class ClassicalPPO(nn.Module):
    """
    Classical PPO baseline without GNN augmentation.
    
    Uses flat feature encoding (512-dimensional vector) instead of 
    graph-based representation. Same PPO algorithm and hyperparameters 
    as PPO-GNN for fair comparison (Table 5.1).
    
    Architecture:
    - Flat encoder: [512, 256, 128]
    - Policy network: [256, 128, 64]
    - Value network: [256, 128]
    
    Args:
        config: Configuration from classical_ppo_config.yaml
        input_dim: Flattened feature dimension (default: 512)
        num_actions: Action space size
        device: 'cuda' or 'cpu'
    
    Example:
        >>> config = load_config('configs/classical_ppo_config.yaml')
        >>> model = ClassicalPPO(config, input_dim=512, num_actions=1500)
        >>> state = flatten_state(env.reset())  # Flatten to 512-dim
        >>> action, log_prob, value = model.act(state)
    """
    
    def __init__(
        self,
        config: Dict,
        input_dim: int = 512,
        num_actions: int = None,
        device: str = 'cuda'
    ):
        super(ClassicalPPO, self).__init__()
        
        self.config = config
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.device = device
        
        # Extract configs
        encoder_config = config.get('encoder', {})
        policy_config = config['policy']
        value_config = config['value']
        
        # Flat encoder
        encoder_arch = encoder_config.get('architecture', [512, 256, 128])
        self.encoder = FlatFeatureEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_arch,
            activation=encoder_config.get('activation', 'relu'),
            dropout=encoder_config.get('dropout', 0.0),
            normalization=encoder_config.get('normalization', 'batch')
        )
        
        # State dimension after encoding
        state_dim = encoder_arch[-1]
        
        # Policy network
        self.policy_net = PolicyNetwork(
            input_dim=state_dim,
            hidden_dims=policy_config['architecture'],
            num_actions=num_actions,
            activation=policy_config['activation'],
            dropout=policy_config.get('dropout', 0.0)
        )
        
        # Value network
        self.value_net = ValueNetwork(
            input_dim=state_dim,
            hidden_dims=value_config['architecture'],
            activation=value_config['activation'],
            dropout=value_config.get('dropout', 0.0)
        )
        
        # Optimizers (same learning rates as PPO-GNN for fair comparison)
        self.optimizer_policy = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=policy_config['learning_rate']
        )
        
        self.optimizer_value = torch.optim.Adam(
            self.value_net.parameters(),
            lr=value_config['learning_rate']
        )
        
        self.optimizer_encoder = torch.optim.Adam(
            self.encoder.parameters(),
            lr=encoder_config.get('learning_rate', 1e-4)
        )
        
        # PPO hyperparameters
        self.clip_param = policy_config['epsilon_clip']
        self.entropy_coef = policy_config['entropy_coef']
        self.gamma = config['training']['discount_factor']
        self.gae_lambda = value_config['gae_lambda']
        
        self.to(device)
    
    def flatten_state(self, state: Dict) -> torch.Tensor:
        """
        Flatten graph state to fixed-size vector.
        
        Concatenates:
        - Node features (demand, time windows, coordinates, etc.)
        - Pairwise distances (flattened distance matrix)
        - Vehicle states (location, capacity, fuel type, etc.)
        
        Args:
            state: State dictionary with graph structure
        
        Returns:
            flat_state: Flattened state [input_dim]
        """
        # Extract components
        node_features = state['node_features']  # [num_nodes, node_dim]
        vehicle_states = state['vehicle_states']  # [num_vehicles, vehicle_dim]
        
        # Flatten node features
        node_flat = node_features.flatten()
        
        # Flatten vehicle states
        vehicle_flat = vehicle_states.flatten()
        
        # Concatenate
        flat_state = torch.cat([node_flat, vehicle_flat], dim=0)
        
        # Pad or truncate to input_dim
        if flat_state.size(0) < self.input_dim:
            padding = torch.zeros(self.input_dim - flat_state.size(0))
            flat_state = torch.cat([flat_state, padding], dim=0)
        elif flat_state.size(0) > self.input_dim:
            flat_state = flat_state[:self.input_dim]
        
        return flat_state.to(self.device)
    
    def forward(self, flat_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Classical PPO.
        
        Args:
            flat_state: Flattened state [batch_size, input_dim]
        
        Returns:
            action_probs: Action probabilities [batch_size, num_actions]
            state_value: State value [batch_size, 1]
        """
        # Encode state
        state_encoding = self.encoder(flat_state)
        
        # Policy and value
        action_probs = self.policy_net(state_encoding)
        state_value = self.value_net(state_encoding)
        
        return action_probs, state_value
    
    def act(
        self,
        state: Dict,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action given current state.
        
        Args:
            state: State dictionary (will be flattened)
            deterministic: If True, select argmax action
        
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        with torch.no_grad():
            # Flatten state
            flat_state = self.flatten_state(state).unsqueeze(0)
            
            # Forward pass
            action_probs, state_value = self.forward(flat_state)
            action_probs = action_probs.squeeze(0)
            state_value = state_value.squeeze(0)
            
            # Sample action
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
            
            # Log probability
            log_prob = torch.log(action_probs[action] + 1e-8)
            
            return action, log_prob, state_value
    
    def evaluate(
        self,
        states: List[Dict],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Args:
            states: List of state dictionaries
            actions: Batch of actions [batch_size]
        
        Returns:
            log_probs: Log probabilities [batch_size]
            values: State values [batch_size]
            entropy: Policy entropy [batch_size]
        """
        # Flatten states
        flat_states = torch.stack([self.flatten_state(s) for s in states])
        
        # Forward pass
        action_probs, state_values = self.forward(flat_states)
        
        # Distribution
        dist = torch.distributions.Categorical(action_probs)
        
        # Log probabilities and entropy
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, state_values.squeeze(-1), entropy
    
    def update(
        self,
        rollouts: Dict,
        epochs: int = 10,
        batch_size: int = 256,
        clip_param: Optional[float] = None
    ) -> Dict:
        """
        Update policy and value networks using PPO.
        
        Same update procedure as PPO-GNN (Algorithm 1).
        
        Args:
            rollouts: Experience buffer
            epochs: Number of optimization epochs
            batch_size: Mini-batch size
            clip_param: PPO clipping parameter
        
        Returns:
            logs: Training statistics
        """
        if clip_param is None:
            clip_param = self.clip_param
        
        # Extract rollouts
        states = rollouts['states']
        actions = rollouts['actions'].to(self.device)
        rewards = rollouts['rewards'].to(self.device)
        old_log_probs = rollouts['old_log_probs'].to(self.device)
        old_values = rollouts['values'].to(self.device)
        dones = rollouts['dones'].to(self.device)
        
        # Compute GAE advantages
        advantages, returns = compute_gae(
            rewards, old_values, dones,
            gamma=self.gamma,
            lambda_=self.gae_lambda
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training statistics
        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        clip_fractions = []
        
        # PPO epochs
        for epoch in range(epochs):
            num_samples = len(states)
            indices = np.random.permutation(num_samples)
            
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Batch data
                batch_states = [states[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Evaluate actions
                log_probs, values, entropy = self.evaluate(batch_states, batch_actions)
                
                # Probability ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages
                
                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss
                
                # Backward pass
                self.optimizer_policy.zero_grad()
                self.optimizer_value.zero_grad()
                self.optimizer_encoder.zero_grad()
                
                total_loss.backward()
                
                # Gradient clipping
                if 'gradient_clip' in self.config['training']:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer_policy.step()
                self.optimizer_value.step()
                self.optimizer_encoder.step()
                
                # Statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean().item()
                    approx_kls.append(approx_kl)
                    
                    clip_fraction = ((ratio - 1.0).abs() > clip_param).float().mean().item()
                    clip_fractions.append(clip_fraction)
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'approx_kl': np.mean(approx_kls),
            'clip_fraction': np.mean(clip_fractions)
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'config': self.config,
            'input_dim': self.input_dim,
            'num_actions': self.num_actions,
            'encoder': self.encoder.state_dict(),
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'optimizer_policy': self.optimizer_policy.state_dict(),
            'optimizer_value': self.optimizer_value.state_dict(),
            'optimizer_encoder': self.optimizer_encoder.state_dict(),
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cuda'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            config=checkpoint['config'],
            input_dim=checkpoint['input_dim'],
            num_actions=checkpoint['num_actions'],
            device=device
        )
        
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.policy_net.load_state_dict(checkpoint['policy_net'])
        model.value_net.load_state_dict(checkpoint['value_net'])
        model.optimizer_policy.load_state_dict(checkpoint['optimizer_policy'])
        model.optimizer_value.load_state_dict(checkpoint['optimizer_value'])
        model.optimizer_encoder.load_state_dict(checkpoint['optimizer_encoder'])
        
        return model


if __name__ == '__main__':
    # Test Classical PPO
    import yaml
    
    with open('configs/classical_ppo_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = ClassicalPPO(
        config=config,
        input_dim=512,
        num_actions=1500,
        device='cpu'
    )
    
    print("✓ Classical PPO initialized successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with dummy state
    dummy_state = {
        'node_features': torch.randn(100, 4),
        'vehicle_states': torch.randn(15, 8)
    }
    
    action, log_prob, value = model.act(dummy_state)
    print(f"✓ Action: {action.item()}, Value: {value.item():.2f}")
