"""
Policy Network (Actor)
Neural network for action selection π_θ(a|s).

Author: Majdi Argoubi
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class PolicyNetwork(nn.Module):
    """
    Policy network (actor) for action selection.
    
    Maps state representations to action probabilities using a multi-layer
    perceptron with softmax output.
    
    Architecture (Table 5.1): [256, 128, 64]
    Learning rate α_θ: 3×10⁻⁴
    
    Args:
        input_dim (int): State embedding dimension
        hidden_dims (list): Hidden layer dimensions (default: [256, 128, 64])
        num_actions (int): Action space size
        activation (str): Activation function ('relu', 'tanh', 'elu')
        dropout (float): Dropout rate (default: 0.0)
        output_activation (str): Output activation ('softmax', 'log_softmax')
    
    Example:
        >>> policy = PolicyNetwork(input_dim=128, hidden_dims=[256,128,64], num_actions=100)
        >>> state_embedding = torch.randn(1, 128)
        >>> action_probs = policy(state_embedding)
        >>> action = torch.multinomial(action_probs, 1)
        >>> print(f"Selected action: {action.item()}")
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        num_actions: int = None,
        activation: str = 'relu',
        dropout: float = 0.0,
        output_activation: str = 'softmax'
    ):
        super(PolicyNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_actions = num_actions
        self.output_activation = output_activation
        
        # Activation function
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
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.ModuleList(layers)
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, num_actions)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform initialization."""
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        # Output layer with smaller initialization for stability
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.01)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through policy network.
        
        Args:
            x: State embedding [batch_size, input_dim]
        
        Returns:
            action_probs: Action probabilities [batch_size, num_actions]
        """
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output layer
        logits = self.output_layer(x)
        
        # Apply output activation
        if self.output_activation == 'softmax':
            action_probs = F.softmax(logits, dim=-1)
        elif self.output_activation == 'log_softmax':
            action_probs = F.log_softmax(logits, dim=-1)
        else:
            action_probs = logits
        
        return action_probs
    
    def sample_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy distribution.
        
        Args:
            state: State embedding [batch_size, input_dim]
            deterministic: If True, return argmax action (no sampling)
            temperature: Temperature for softmax (lower = more deterministic)
        
        Returns:
            action: Sampled action [batch_size]
            log_prob: Log probability of action [batch_size]
        
        Example:
            >>> action, log_prob = policy.sample_action(state, deterministic=False)
            >>> print(f"Action: {action.item()}, Log Prob: {log_prob.item():.4f}")
        """
        # Get action probabilities
        action_probs = self.forward(state)
        
        # Apply temperature
        if temperature != 1.0:
            logits = torch.log(action_probs + 1e-8) / temperature
            action_probs = F.softmax(logits, dim=-1)
        
        # Sample or select argmax
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
        
        # Compute log probability
        log_prob = torch.log(action_probs.gather(-1, action.unsqueeze(-1)) + 1e-8)
        log_prob = log_prob.squeeze(-1)
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions under current policy.
        
        Used during PPO update to compute log probabilities and entropy.
        
        Args:
            state: State embedding [batch_size, input_dim]
            action: Actions to evaluate [batch_size]
        
        Returns:
            log_prob: Log probabilities [batch_size]
            entropy: Entropy of action distribution [batch_size]
        """
        # Get action probabilities
        action_probs = self.forward(state)
        
        # Create distribution
        dist = torch.distributions.Categorical(action_probs)
        
        # Log probability of actions
        log_prob = dist.log_prob(action)
        
        # Entropy
        entropy = dist.entropy()
        
        return log_prob, entropy


class MultiHeadPolicyNetwork(PolicyNetwork):
    """
    Multi-head policy network for multi-vehicle routing.
    
    Separate policy heads for each vehicle to enable independent decisions
    while sharing feature extraction.
    
    Args:
        input_dim: State embedding dimension
        hidden_dims: Shared hidden layer dimensions
        num_actions_per_vehicle: Action space size per vehicle
        num_vehicles: Number of vehicles
        activation: Activation function
        dropout: Dropout rate
    
    Example:
        >>> policy = MultiHeadPolicyNetwork(
        ...     input_dim=128, 
        ...     hidden_dims=[256, 128], 
        ...     num_actions_per_vehicle=100,
        ...     num_vehicles=15
        ... )
        >>> state = torch.randn(1, 128)
        >>> actions = policy.sample_multi_vehicle_actions(state)
        >>> print(f"Actions for {len(actions)} vehicles")
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_actions_per_vehicle: int,
        num_vehicles: int,
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        # Initialize base class with dummy num_actions
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_actions=num_actions_per_vehicle,
            activation=activation,
            dropout=dropout
        )
        
        self.num_vehicles = num_vehicles
        self.num_actions_per_vehicle = num_actions_per_vehicle
        
        # Remove single output layer
        delattr(self, 'output_layer')
        
        # Create separate output heads for each vehicle
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], num_actions_per_vehicle)
            for _ in range(num_vehicles)
        ])
        
        # Initialize heads
        for head in self.output_heads:
            nn.init.xavier_uniform_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through multi-head policy.
        
        Args:
            x: State embedding [batch_size, input_dim]
        
        Returns:
            action_probs_list: List of action probabilities for each vehicle
                Each element: [batch_size, num_actions_per_vehicle]
        """
        # Shared hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Separate heads for each vehicle
        action_probs_list = []
        for head in self.output_heads:
            logits = head(x)
            action_probs = F.softmax(logits, dim=-1)
            action_probs_list.append(action_probs)
        
        return action_probs_list
    
    def sample_multi_vehicle_actions(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Sample actions for all vehicles.
        
        Args:
            state: State embedding [batch_size, input_dim]
            deterministic: If True, select argmax actions
        
        Returns:
            actions: List of actions for each vehicle [batch_size]
            log_probs: List of log probabilities [batch_size]
        """
        action_probs_list = self.forward(state)
        
        actions = []
        log_probs = []
        
        for action_probs in action_probs_list:
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
            
            log_prob = torch.log(action_probs.gather(-1, action.unsqueeze(-1)) + 1e-8)
            log_prob = log_prob.squeeze(-1)
            
            actions.append(action)
            log_probs.append(log_prob)
        
        return actions, log_probs


if __name__ == '__main__':
    # Test PolicyNetwork
    print("Testing PolicyNetwork...")
    policy = PolicyNetwork(input_dim=128, hidden_dims=[256, 128, 64], num_actions=100)
    state = torch.randn(4, 128)  # Batch of 4 states
    
    # Forward pass
    action_probs = policy(state)
    print(f"✓ Action probs shape: {action_probs.shape}")
    print(f"✓ Action probs sum: {action_probs.sum(dim=-1)}")
    
    # Sample actions
    actions, log_probs = policy.sample_action(state)
    print(f"✓ Sampled actions: {actions}")
    print(f"✓ Log probs: {log_probs}")
    
    # Test MultiHeadPolicyNetwork
    print("\nTesting MultiHeadPolicyNetwork...")
    multi_policy = MultiHeadPolicyNetwork(
        input_dim=128,
        hidden_dims=[256, 128],
        num_actions_per_vehicle=100,
        num_vehicles=15
    )
    
    actions_list, log_probs_list = multi_policy.sample_multi_vehicle_actions(state)
    print(f"✓ Actions for {len(actions_list)} vehicles")
    print(f"✓ Total parameters: {sum(p.numel() for p in multi_policy.parameters()):,}")
