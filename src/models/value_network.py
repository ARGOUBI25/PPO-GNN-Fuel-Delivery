"""
Value Network (Critic)
Neural network for state value estimation V_φ(s).

Author: Majdi Argoubi
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ValueNetwork(nn.Module):
    """
    Value network (critic) for state evaluation.
    
    Estimates state value V_φ(s) for advantage computation in PPO.
    Used in GAE (Equation 9) to compute advantages.
    
    Architecture (Table 5.1): [256, 128]
    Learning rate α_φ: 1×10⁻³
    
    Args:
        input_dim (int): State embedding dimension
        hidden_dims (list): Hidden layer dimensions (default: [256, 128])
        activation (str): Activation function ('relu', 'tanh', 'elu')
        dropout (float): Dropout rate (default: 0.0)
        output_activation (str): Output activation (None, 'tanh')
    
    Example:
        >>> value_net = ValueNetwork(input_dim=128, hidden_dims=[256, 128])
        >>> state_embedding = torch.randn(1, 128)
        >>> value = value_net(state_embedding)
        >>> print(f"State value: {value.item():.2f}")
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        activation: str = 'relu',
        dropout: float = 0.0,
        output_activation: Optional[str] = None
    ):
        super(ValueNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
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
        
        # Output layer (scalar value)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for layer in self.hidden_layers:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
        
        # Output layer with small initialization
        nn.init.orthogonal_(self.output_layer.weight, gain=0.01)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value network.
        
        Args:
            x: State embedding [batch_size, input_dim]
        
        Returns:
            value: State value estimate [batch_size, 1]
        """
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output layer
        value = self.output_layer(x)
        
        # Optional output activation
        if self.output_activation == 'tanh':
            value = torch.tanh(value)
        
        return value
    
    def predict_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict state value (wrapper for forward).
        
        Args:
            state: State embedding [batch_size, input_dim]
        
        Returns:
            value: State value [batch_size]
        """
        value = self.forward(state)
        return value.squeeze(-1)


class DualValueNetwork(nn.Module):
    """
    Dual value network for estimating both V(s) and Q(s,a).
    
    Useful for more sophisticated advantage estimation or for implementing
    algorithms like TD3 or SAC in future extensions.
    
    Args:
        input_dim: State embedding dimension
        action_dim: Action dimension
        hidden_dims: Hidden layer dimensions
        activation: Activation function
        dropout: Dropout rate
    
    Example:
        >>> dual_value = DualValueNetwork(input_dim=128, action_dim=10, hidden_dims=[256, 128])
        >>> state = torch.randn(4, 128)
        >>> action = torch.randn(4, 10)
        >>> v_value = dual_value.forward_v(state)
        >>> q_value = dual_value.forward_q(state, action)
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 128],
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        super(DualValueNetwork, self).__init__()
        
        # V(s) network
        self.v_net = ValueNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout
        )
        
        # Q(s,a) network
        self.q_net = ValueNetwork(
            input_dim=input_dim + action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout
        )
    
    def forward_v(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute state value V(s).
        
        Args:
            state: State embedding [batch_size, input_dim]
        
        Returns:
            v_value: State value [batch_size]
        """
        return self.v_net.predict_value(state)
    
    def forward_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute state-action value Q(s,a).
        
        Args:
            state: State embedding [batch_size, input_dim]
            action: Action [batch_size, action_dim]
        
        Returns:
            q_value: State-action value [batch_size]
        """
        # Concatenate state and action
        state_action = torch.cat([state, action], dim=-1)
        return self.q_net.predict_value(state_action)
    
    def compute_advantage(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute advantage A(s,a) = Q(s,a) - V(s).
        
        Args:
            state: State embedding [batch_size, input_dim]
            action: Action [batch_size, action_dim]
        
        Returns:
            advantage: Advantage [batch_size]
        """
        v_value = self.forward_v(state)
        q_value = self.forward_q(state, action)
        advantage = q_value - v_value
        return advantage


class EnsembleValueNetwork(nn.Module):
    """
    Ensemble of value networks for uncertainty estimation.
    
    Uses multiple value networks and averages their predictions to reduce
    variance in value estimation.
    
    Args:
        input_dim: State embedding dimension
        hidden_dims: Hidden layer dimensions
        num_networks: Number of networks in ensemble (default: 5)
        activation: Activation function
        dropout: Dropout rate
    
    Example:
        >>> ensemble = EnsembleValueNetwork(input_dim=128, num_networks=5)
        >>> state = torch.randn(4, 128)
        >>> mean_value, std_value = ensemble(state)
        >>> print(f"Value: {mean_value[0]:.2f} ± {std_value[0]:.2f}")
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        num_networks: int = 5,
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        super(EnsembleValueNetwork, self).__init__()
        
        self.num_networks = num_networks
        
        # Create ensemble of value networks
        self.networks = nn.ModuleList([
            ValueNetwork(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout
            )
            for _ in range(num_networks)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        return_std: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: State embedding [batch_size, input_dim]
            return_std: If True, return both mean and std
        
        Returns:
            If return_std:
                mean_value: Mean value estimate [batch_size]
                std_value: Standard deviation [batch_size]
            Else:
                mean_value: Mean value estimate [batch_size]
        """
        # Get predictions from all networks
        predictions = []
        for net in self.networks:
            value = net.predict_value(x)
            predictions.append(value)
        
        predictions = torch.stack(predictions, dim=0)  # [num_networks, batch_size]
        
        # Compute mean and std
        mean_value = predictions.mean(dim=0)
        
        if return_std:
            std_value = predictions.std(dim=0)
            return mean_value, std_value
        else:
            return mean_value


if __name__ == '__main__':
    # Test ValueNetwork
    print("Testing ValueNetwork...")
    value_net = ValueNetwork(input_dim=128, hidden_dims=[256, 128])
    state = torch.randn(4, 128)  # Batch of 4 states
    
    # Forward pass
    values = value_net(state)
    print(f"✓ Values shape: {values.shape}")
    print(f"✓ Values: {values.squeeze().tolist()}")
    
    # Predict value
    predicted_values = value_net.predict_value(state)
    print(f"✓ Predicted values shape: {predicted_values.shape}")
    
    # Test DualValueNetwork
    print("\nTesting DualValueNetwork...")
    dual_value = DualValueNetwork(input_dim=128, action_dim=10, hidden_dims=[256, 128])
    action = torch.randn(4, 10)
    
    v_values = dual_value.forward_v(state)
    q_values = dual_value.forward_q(state, action)
    advantages = dual_value.compute_advantage(state, action)
    
    print(f"✓ V(s): {v_values.tolist()}")
    print(f"✓ Q(s,a): {q_values.tolist()}")
    print(f"✓ A(s,a): {advantages.tolist()}")
    
    # Test EnsembleValueNetwork
    print("\nTesting EnsembleValueNetwork...")
    ensemble = EnsembleValueNetwork(input_dim=128, num_networks=5)
    mean_values, std_values = ensemble(state)
    
    print(f"✓ Mean values: {mean_values.tolist()}")
    print(f"✓ Std values: {std_values.tolist()}")
    print(f"✓ Total parameters: {sum(p.numel() for p in ensemble.parameters()):,}")
