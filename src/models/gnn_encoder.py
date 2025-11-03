"""
GNN Encoder
3-layer Graph Neural Network with message-passing (Equation 1, Section 4.1).

Author: Your Name
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Optional


class GNNLayer(MessagePassing):
    """
    Single GNN layer implementing message-passing (Eq. 1).
    
    h_i^(l+1) = σ(W^(l) h_i^(l) + Σ_{j∈N(i)} 1/|N(i)| W^(l) h_j^(l))
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        aggr: Aggregation scheme ('mean', 'sum', 'max')
        bias: Use bias term
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'mean',
        bias: bool = True
    ):
        super(GNNLayer, self).__init__(aggr=aggr)
        
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        self.lin.reset_parameters()
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_attr_dim] (optional)
        
        Returns:
            out: Updated node features [num_nodes, out_channels]
        """
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Transform features
        x = self.lin(x)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        return out
    
    def message(
        self,
        x_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Construct messages from neighbors.
        
        Args:
            x_j: Neighbor node features [num_edges, out_channels]
            edge_attr: Edge attributes [num_edges, edge_attr_dim]
        
        Returns:
            messages: Messages to aggregate [num_edges, out_channels]
        """
        # Simple message: just neighbor features
        # Can be extended to include edge attributes
        return x_j
    
    def aggregate(
        self,
        inputs: torch.Tensor,
        index: torch.Tensor,
        dim_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Aggregate messages from neighbors.
        
        Uses mean aggregation: 1/|N(i)| Σ_{j∈N(i)} m_j
        """
        return super().aggregate(inputs, index, dim_size)


class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for spatial feature extraction.
    
    Implements L-layer message-passing GNN (Section 4.1).
    Default: L=3 layers with hidden_dim=128 (Table 5.1).
    
    Args:
        input_dim: Input node feature dimension
        hidden_dim: Hidden dimension (default: 128)
        num_layers: Number of GNN layers L (default: 3)
        activation: Activation function ('relu', 'tanh', 'elu')
        dropout: Dropout rate (default: 0.0)
        normalize: Apply layer normalization (default: True)
        residual: Use residual connections (default: False)
    
    Example:
        >>> gnn = GNNEncoder(input_dim=32, hidden_dim=128, num_layers=3)
        >>> node_features = torch.randn(100, 32)
        >>> edge_index = torch.tensor([[0,1,2], [1,2,0]], dtype=torch.long)
        >>> embeddings = gnn(node_features, edge_index)
        >>> print(embeddings.shape)  # [100, 128]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        activation: str = 'relu',
        dropout: float = 0.0,
        normalize: bool = True,
        residual: bool = False
    ):
        super(GNNEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.normalize = normalize
        self.residual = residual
        
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
        
        # GNN layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GNNLayer(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim))
        
        # Layer normalization
        if normalize:
            self.norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
        else:
            self.norms = None
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GNN layers.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
        
        Returns:
            h: Node embeddings [num_nodes, hidden_dim]
        """
        h = x
        
        for i, layer in enumerate(self.layers):
            # Save for residual connection
            h_prev = h if self.residual and i > 0 else None
            
            # GNN layer
            h = layer(h, edge_index, edge_attr)
            
            # Activation
            h = self.activation(h)
            
            # Layer normalization
            if self.normalize:
                h = self.norms[i](h)
            
            # Dropout
            h = self.dropout(h)
            
            # Residual connection
            if h_prev is not None:
                h = h + h_prev
        
        return h


if __name__ == '__main__':
    # Test GNN encoder
    gnn = GNNEncoder(input_dim=32, hidden_dim=128, num_layers=3)
    
    # Dummy data
    num_nodes = 100
    node_features = torch.randn(num_nodes, 32)
    edge_index = torch.randint(0, num_nodes, (2, 300))
    
    # Forward pass
    embeddings = gnn(node_features, edge_index)
    
    print("✓ GNN Encoder test passed")
    print(f"  Input shape: {node_features.shape}")
    print(f"  Output shape: {embeddings.shape}")
    print(f"  Parameters: {sum(p.numel() for p in gnn.parameters()):,}")
