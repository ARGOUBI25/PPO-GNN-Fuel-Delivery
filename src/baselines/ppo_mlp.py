"""
PPO-MLP (Ablation Baseline)
PPO with MLP encoder (no message-passing).

Section 5.2: Ablation study - processes nodes independently through MLP,
then aggregates via mean pooling. No message-passing aggregation.

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


class MLPEncoder(nn.Module):
    """
    MLP encoder for node-wise processing (no message-passing).
    
    Processes each node independently through MLP [256, 128, 64],
    then aggregates node embeddings using mean/sum/max pooling.
    
    Section 5.2: "3 layers, [256, 128, 64]"
    
    Args:
        node_feature_dim: Input features per node
        hidden_dims: MLP architecture (default: [256, 128, 64])
        num_layers: Number of MLP layers (default: 3)
        aggregation: Aggregation method ('mean', 'sum', 'max')
        activation: Activation function
        dropout: Dropout rate
        normalization: 'layer' or 'batch' normalization
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        num_layers: int = 3,
        aggregation: str = 'mean',
        activation: str = 'relu',
        dropout: float = 0.0,
        normalization: str = 'layer'
    ):
        super(MLPEncoder, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dims = hidden_dims
        self.aggregation = aggregation
        
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
        
        # MLP layers (shared across all nodes)
        layers = []
        prev_dim = node_feature_dim
        
        for hidden_dim in hidden_dims[:num_layers]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.layers = nn.ModuleList(layers)
        
        # Layer normalization
        if normalization == 'layer':
            self.norms = nn.ModuleList([
                nn.LayerNorm(dim) for dim in hidden_dims[:num_layers]
            ])
        elif normalization == 'batch':
            self.norms = nn.ModuleList([
                nn.BatchNorm1d(dim) for dim in hidden_dims[:num_layers]
            ])
        else:
            self.norms = None
        
        self.output_dim = hidden_dims[num_layers - 1]
    
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Encode node features independently (no message-passing).
        
        Args:
            node_features: Node feature matrix [num_nodes, node_feature_dim]
        
        Returns:
            graph_embedding: Aggregated graph embedding [output_dim]
        """
        x = node_features
        
        # Process each node independently through MLP
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Normalization
            if self.norms is not None:
                if isinstance(self.norms[i], nn.BatchNorm1d) and x.size(0) == 1:
                    pass  # Skip batch norm for single node
                else:
                    x = self.norms[i](x)
            
            x = self.activation(x)
            x = self.dropout(x)
        
        # Aggregate node embeddings
        if self.aggregation == 'mean':
            graph_embedding = x.mean(dim=0)
        elif self.aggregation == 'sum':
            graph_embedding = x.sum(dim=0)
        elif self.aggregation == 'max':
            graph_embedding = x.max(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        return graph_embedding


class PPOMLP(nn.Module):
    """
    PPO with MLP encoder (ablation baseline).
    
    Processes node features independently through MLP [256, 128, 64],
    then aggregates via mean pooling. No message-passing aggregation.
    
    Section 5.2: Ablation study to isolate impact of message-passing.
    Same hyperparameters as PPO-GNN except no graph convolutions.
    
    Args:
        config: Configuration from ppo_mlp_config.yaml
        node_feature_dim: Input features per node
        num_nodes: Number of nodes in network
        num_vehicles: Number of vehicles
        num_actions: Action space size
        device: 'cuda' or 'cpu'
    
    Example:
        >>> config = load_config('configs/ppo_mlp_config.yaml')
        >>> model = PPOMLP(config, node_feature_dim=32, num_nodes=100, num_vehicles=15)
        >>> state = env.reset()
        >>> action, log_prob, value = model.act(state)
    """
    
    def __init__(
        self,
        config: Dict,
        node_feature_dim: int = 32,
        num_nodes: int = 100,
        num_vehicles: int = 15,
        num_actions: int = None,
        device: str = 'cuda'
    ):
        super(PPOMLP, self).__init__()
        
        self.config = config
        self.node_feature_dim = node_feature_dim
        self.num_nodes = num_nodes
        self.num_vehicles = num_vehicles
        self.device = device
        
        # Extract configs
        encoder_config = config.get('encoder', {})
        policy_config = config['policy']
        value_config = config['value']
        
        # MLP encoder
        self.encoder = MLPEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dims=encoder_config.get('hidden_dims', [256, 128, 64]),
            num_layers=encoder_config.get('num_layers', 3),
            aggregation=encoder_config.get('aggregation', 'mean'),
            activation=encoder_config.get('activation', 'relu'),
            dropout=encoder_config.get('dropout', 0.0),
            normalization=encoder_config.get('normalization', 'layer')
        )
        
        # State dimension = graph embedding + vehicle states
        state_dim = self.encoder.output_dim + num_vehicles * 8  # Vehicle state dim
        
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
        
        # Optimizers
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
    
    def forward(
        self,
        node_features: torch.Tensor,
        vehicle_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through PPO-MLP.
        
        Args:
            node_features: Node features [num_nodes, node_feature_dim]
            vehicle_states: Vehicle states [num_vehicles, vehicle_state_dim]
        
        Returns:
            action_probs: Action probabilities [num_actions]
            state_value: State value estimate
        """
        # Encode graph (no message-passing)
        graph_embedding = self.encoder(node_features)
        
        # Concatenate with vehicle states
        vehicle_embedding = vehicle_states.flatten()
        state_embedding = torch.cat([graph_embedding, vehicle_embedding], dim=0)
        
        # Policy and value
        action_probs = self.policy_net(state_embedding.unsqueeze(0))
        state_value = self.value_net(state_embedding.unsqueeze(0))
        
        return action_probs.squeeze(0), state_value.squeeze(0)
    
    def act(
        self,
        state: Dict,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action given current state.
        
        Args:
            state: State dictionary containing:
                - 'node_features': [num_nodes, node_feature_dim]
                - 'vehicle_states': [num_vehicles, vehicle_state_dim]
            deterministic: If True, select argmax action
        
        Returns:
            action: Selected action
            log_prob: Log probability
            value: State value
        """
        with torch.no_grad():
            node_features = state['node_features'].to(self.device)
            vehicle_states = state['vehicle_states'].to(self.device)
            
            # Forward pass
            action_probs, state_value = self.forward(node_features, vehicle_states)
            
            # Sample action
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
            
            log_prob = torch.log(action_probs[action] + 1e-8)
            
            return action, log_prob, state_value
    
    def evaluate(
        self,
        states: List[Dict],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        log_probs = []
        values = []
        entropies = []
        
        for i, state in enumerate(states):
            node_features = state['node_features'].to(self.device)
            vehicle_states = state['vehicle_states'].to(self.device)
            
            action_probs, state_value = self.forward(node_features, vehicle_states)
            
            dist = torch.distributions.Categorical(action_probs)
            log_prob = dist.log_prob(actions[i])
            entropy = dist.entropy()
            
            log_probs.append(log_prob)
            values.append(state_value)
            entropies.append(entropy)
        
        return (
            torch.stack(log_probs),
            torch.stack(values),
            torch.stack(entropies)
        )
    
    def update(
        self,
        rollouts: Dict,
        epochs: int = 10,
        batch_size: int = 256,
        clip_param: Optional[float] = None
    ) -> Dict:
        """Update using PPO (same as other baselines)."""
        if clip_param is None:
            clip_param = self.clip_param
        
        states = rollouts['states']
        actions = rollouts['actions'].to(self.device)
        rewards = rollouts['rewards'].to(self.device)
        old_log_probs = rollouts['old_log_probs'].to(self.device)
        old_values = rollouts['values'].to(self.device)
        dones = rollouts['dones'].to(self.device)
        
        advantages, returns = compute_gae(
            rewards, old_values, dones,
            gamma=self.gamma,
            lambda_=self.gae_lambda
        )
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        clip_fractions = []
        
        for epoch in range(epochs):
            num_samples = len(states)
            indices = np.random.permutation(num_samples)
            
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = [states[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                log_probs, values, entropy = self.evaluate(batch_states, batch_actions)
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_returns)
                entropy_loss = -entropy.mean()
                
                total_loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss
                
                self.optimizer_policy.zero_grad()
                self.optimizer_value.zero_grad()
                self.optimizer_encoder.zero_grad()
                
                total_loss.backward()
                
                if 'gradient_clip' in self.config['training']:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer_policy.step()
                self.optimizer_value.step()
                self.optimizer_encoder.step()
                
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
        """Save checkpoint."""
        torch.save({
            'config': self.config,
            'node_feature_dim': self.node_feature_dim,
            'num_nodes': self.num_nodes,
            'num_vehicles': self.num_vehicles,
            'encoder': self.encoder.state_dict(),
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'optimizer_policy': self.optimizer_policy.state_dict(),
            'optimizer_value': self.optimizer_value.state_dict(),
            'optimizer_encoder': self.optimizer_encoder.state_dict(),
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cuda'):
        """Load from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            config=checkpoint['config'],
            node_feature_dim=checkpoint['node_feature_dim'],
            num_nodes=checkpoint['num_nodes'],
            num_vehicles=checkpoint['num_vehicles'],
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
    # Test PPO-MLP
    import yaml
    
    with open('configs/ppo_mlp_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = PPOMLP(
        config=config,
        node_feature_dim=32,
        num_nodes=100,
        num_vehicles=15,
        num_actions=1500,
        device='cpu'
    )
    
    print("✓ PPO-MLP initialized successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with dummy state
    dummy_state = {
        'node_features': torch.randn(100, 32),
        'vehicle_states': torch.randn(15, 8)
    }
    
    action, log_prob, value = model.act(dummy_state)
    print(f"✓ Action: {action.item()}, Value: {value.item():.2f}")
