"""
PPO-GNN Framework
Complete implementation integrating GNN encoder, policy network, and value network.

Author: Your Name
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np

from .gnn_encoder import GNNEncoder
from .policy_network import PolicyNetwork
from .value_network import ValueNetwork
from .gae import compute_gae


class PPOGNN(nn.Module):
    """
    Complete PPO-GNN framework for fuel delivery optimization.
    
    Integrates:
    - GNN encoder for spatial feature extraction (Section 4.1, Eq. 1)
    - Policy network π_θ for action selection (Section 4.2)
    - Value network V_φ for state evaluation (Section 4.2)
    - Three-tier constraint validation (Section 4.3)
    
    Architecture follows Figure 3 from the paper.
    
    Args:
        config (dict): Configuration dictionary from ppo_gnn_config.yaml
        num_nodes (int): Number of stations in the network
        num_vehicles (int): Fleet size
        node_feature_dim (int): Dimension of node features
        edge_feature_dim (int): Dimension of edge features
        device (str): Device to use ('cuda' or 'cpu')
    
    Attributes:
        gnn_encoder (GNNEncoder): Graph neural network encoder
        policy_net (PolicyNetwork): Actor network π_θ
        value_net (ValueNetwork): Critic network V_φ
        optimizer_policy (torch.optim.Adam): Policy optimizer (α_θ = 3e-4)
        optimizer_value (torch.optim.Adam): Value optimizer (α_φ = 1e-3)
        optimizer_gnn (torch.optim.Adam): GNN optimizer (α_ψ = 1e-4)
    
    Example:
        >>> config = load_config('configs/ppo_gnn_config.yaml')
        >>> model = PPOGNN(config, num_nodes=100, num_vehicles=15)
        >>> state = env.reset()
        >>> action, log_prob, value = model.act(state)
    """
    
    def __init__(
        self,
        config: Dict,
        num_nodes: int,
        num_vehicles: int,
        node_feature_dim: int = 32,
        edge_feature_dim: int = 8,
        device: str = 'cuda'
    ):
        super(PPOGNN, self).__init__()
        
        self.config = config
        self.num_nodes = num_nodes
        self.num_vehicles = num_vehicles
        self.device = device
        
        # Extract hyperparameters from config
        gnn_config = config['gnn']
        policy_config = config['policy']
        value_config = config['value']
        
        # GNN Encoder (Section 4.1)
        self.gnn_encoder = GNNEncoder(
            input_dim=node_feature_dim,
            hidden_dim=gnn_config['hidden_dim'],
            num_layers=gnn_config['num_layers'],
            activation=gnn_config['activation'],
            dropout=gnn_config.get('dropout', 0.0),
            normalize=gnn_config.get('normalize', True)
        )
        
        # State dimension after GNN encoding
        state_dim = gnn_config['hidden_dim'] * num_nodes + num_vehicles * 8  # Vehicle states
        
        # Policy Network (Actor)
        self.policy_net = PolicyNetwork(
            input_dim=state_dim,
            hidden_dims=policy_config['architecture'],
            num_actions=num_nodes * num_vehicles,  # Each vehicle can go to any node
            activation=policy_config['activation'],
            dropout=policy_config.get('dropout', 0.0)
        )
        
        # Value Network (Critic)
        self.value_net = ValueNetwork(
            input_dim=state_dim,
            hidden_dims=value_config['architecture'],
            activation=value_config['activation'],
            dropout=value_config.get('dropout', 0.0)
        )
        
        # Optimizers (Table 5.1)
        self.optimizer_policy = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=policy_config['learning_rate']
        )
        
        self.optimizer_value = torch.optim.Adam(
            self.value_net.parameters(),
            lr=value_config['learning_rate']
        )
        
        self.optimizer_gnn = torch.optim.Adam(
            self.gnn_encoder.parameters(),
            lr=gnn_config['learning_rate']
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
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        vehicle_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through PPO-GNN.
        
        Args:
            node_features: Node feature matrix [num_nodes, node_feature_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_feature_dim] (optional)
            vehicle_states: Vehicle state matrix [num_vehicles, vehicle_state_dim]
        
        Returns:
            action_probs: Action probability distribution [num_actions]
            state_value: State value estimate V_φ(s)
        """
        # GNN encoding (Eq. 1)
        node_embeddings = self.gnn_encoder(node_features, edge_index, edge_attr)
        
        # Flatten and concatenate with vehicle states
        graph_embedding = node_embeddings.flatten()
        vehicle_embedding = vehicle_states.flatten()
        state_embedding = torch.cat([graph_embedding, vehicle_embedding], dim=0)
        
        # Policy and value networks
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
                - 'node_features': Tensor [num_nodes, node_feature_dim]
                - 'edge_index': Tensor [2, num_edges]
                - 'edge_attr': Tensor [num_edges, edge_feature_dim]
                - 'vehicle_states': Tensor [num_vehicles, vehicle_state_dim]
            deterministic: If True, select argmax action (no sampling)
        
        Returns:
            action: Selected action [num_vehicles]
            log_prob: Log probability of action
            value: State value estimate V_φ(s)
        
        Example:
            >>> action, log_prob, value = model.act(state, deterministic=False)
            >>> print(f"Action: {action}, Value: {value:.2f}")
        """
        with torch.no_grad():
            # Extract state components
            node_features = state['node_features'].to(self.device)
            edge_index = state['edge_index'].to(self.device)
            edge_attr = state.get('edge_attr')
            if edge_attr is not None:
                edge_attr = edge_attr.to(self.device)
            vehicle_states = state['vehicle_states'].to(self.device)
            
            # Forward pass
            action_probs, state_value = self.forward(
                node_features, edge_index, edge_attr, vehicle_states
            )
            
            # Sample or select argmax
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
            
            # Compute log probability
            log_prob = torch.log(action_probs[action] + 1e-8)
            
            return action, log_prob, state_value
    
    def evaluate(
        self,
        states: List[Dict],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update (Algorithm 1, lines 33-39).
        
        Used during training to compute log probabilities, values, and entropy
        for the PPO objective function.
        
        Args:
            states: List of state dictionaries
            actions: Batch of actions [batch_size, num_vehicles]
        
        Returns:
            log_probs: Log probabilities [batch_size]
            values: State values [batch_size]
            entropy: Policy entropy [batch_size]
        """
        batch_size = len(states)
        log_probs = []
        values = []
        entropies = []
        
        for i, state in enumerate(states):
            # Extract state components
            node_features = state['node_features'].to(self.device)
            edge_index = state['edge_index'].to(self.device)
            edge_attr = state.get('edge_attr')
            if edge_attr is not None:
                edge_attr = edge_attr.to(self.device)
            vehicle_states = state['vehicle_states'].to(self.device)
            
            # Forward pass
            action_probs, state_value = self.forward(
                node_features, edge_index, edge_attr, vehicle_states
            )
            
            # Distribution
            dist = torch.distributions.Categorical(action_probs)
            
            # Log probability of taken action
            action = actions[i]
            log_prob = dist.log_prob(action)
            
            # Entropy
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
        """
        Update policy and value networks using PPO (Algorithm 1, lines 33-49).
        
        Implements the PPO-Clip objective:
            L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
        
        Args:
            rollouts: Experience buffer containing:
                - 'states': List of states
                - 'actions': Tensor [num_steps]
                - 'rewards': Tensor [num_steps]
                - 'old_log_probs': Tensor [num_steps]
                - 'values': Tensor [num_steps]
                - 'dones': Tensor [num_steps]
            epochs: Number of optimization epochs K (default: 10)
            batch_size: Mini-batch size B (default: 256)
            clip_param: PPO clipping parameter ε (default: from config)
        
        Returns:
            logs: Training statistics
                - 'policy_loss': Mean policy loss
                - 'value_loss': Mean value loss
                - 'entropy': Mean entropy
                - 'approx_kl': Approximate KL divergence
                - 'clip_fraction': Fraction of clipped ratios
        
        Example:
            >>> logs = model.update(rollouts, epochs=10)
            >>> print(f"Policy Loss: {logs['policy_loss']:.4f}")
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
        
        # Compute advantages using GAE (Eq. 9)
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
            # Mini-batch training
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
                
                # Probability ratio r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages
                
                # Policy loss (Eq. 8)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss
                
                # Backward pass
                self.optimizer_policy.zero_grad()
                self.optimizer_value.zero_grad()
                self.optimizer_gnn.zero_grad()
                
                total_loss.backward()
                
                # Gradient clipping
                if 'gradient_clip' in self.config['training']:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer_policy.step()
                self.optimizer_value.step()
                
                # Update GNN less frequently (Algorithm 1, line 50)
                if epoch % self.config['gnn'].get('update_frequency', 1000) == 0:
                    self.optimizer_gnn.step()
                
                # Statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                
                # Approximate KL divergence
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean().item()
                    approx_kls.append(approx_kl)
                    
                    # Clip fraction
                    clip_fraction = ((ratio - 1.0).abs() > clip_param).float().mean().item()
                    clip_fractions.append(clip_fraction)
        
        # Return averaged statistics
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'approx_kl': np.mean(approx_kls),
            'clip_fraction': np.mean(clip_fractions)
        }
    
    def save(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        
        Example:
            >>> model.save('checkpoints/ppo_gnn_best.pth')
        """
        torch.save({
            'config': self.config,
            'num_nodes': self.num_nodes,
            'num_vehicles': self.num_vehicles,
            'gnn_encoder': self.gnn_encoder.state_dict(),
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'optimizer_policy': self.optimizer_policy.state_dict(),
            'optimizer_value': self.optimizer_value.state_dict(),
            'optimizer_gnn': self.optimizer_gnn.state_dict(),
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cuda'):
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file
            device: Device to load model on
        
        Returns:
            model: Loaded PPOGNN model
        
        Example:
            >>> model = PPOGNN.load('checkpoints/ppo_gnn_best.pth', device='cuda')
        """
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            config=checkpoint['config'],
            num_nodes=checkpoint['num_nodes'],
            num_vehicles=checkpoint['num_vehicles'],
            device=device
        )
        
        model.gnn_encoder.load_state_dict(checkpoint['gnn_encoder'])
        model.policy_net.load_state_dict(checkpoint['policy_net'])
        model.value_net.load_state_dict(checkpoint['value_net'])
        model.optimizer_policy.load_state_dict(checkpoint['optimizer_policy'])
        model.optimizer_value.load_state_dict(checkpoint['optimizer_value'])
        model.optimizer_gnn.load_state_dict(checkpoint['optimizer_gnn'])
        
        return model


if __name__ == '__main__':
    # Test instantiation
    import yaml
    
    with open('configs/ppo_gnn_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = PPOGNN(
        config=config,
        num_nodes=100,
        num_vehicles=15,
        device='cpu'
    )
    
    print("✓ PPO-GNN model initialized successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
