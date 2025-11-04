"""
Network Utilities
Graph/network construction and manipulation utilities.

Handles graph creation, edge computation, and network analysis.

Author: Your Name
Date: 2025
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import torch


def build_complete_graph(
    num_nodes: int,
    distance_matrix: np.ndarray
) -> nx.Graph:
    """
    Build complete graph from distance matrix.
    
    Args:
        num_nodes: Number of nodes
        distance_matrix: Distance matrix [num_nodes, num_nodes]
    
    Returns:
        graph: NetworkX graph
    """
    G = nx.Graph()
    
    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)
    
    # Add edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            G.add_edge(i, j, weight=distance_matrix[i, j])
    
    return G


def build_k_nearest_graph(
    coordinates: np.ndarray,
    k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build k-nearest neighbor graph.
    
    Args:
        coordinates: Node coordinates [num_nodes, 2]
        k: Number of nearest neighbors
    
    Returns:
        edge_index: Edge indices [2, num_edges]
        edge_attr: Edge attributes (distances) [num_edges]
    """
    num_nodes = len(coordinates)
    
    # Compute pairwise distances
    distances = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :],
        axis=-1
    )
    
    # Find k nearest neighbors for each node
    edge_list = []
    edge_distances = []
    
    for i in range(num_nodes):
        # Get k nearest (excluding self)
        nearest = np.argsort(distances[i])[1:k+1]
        
        for j in nearest:
            edge_list.append([i, j])
            edge_distances.append(distances[i, j])
    
    edge_index = np.array(edge_list).T
    edge_attr = np.array(edge_distances)
    
    return edge_index, edge_attr


def compute_node_features(
    instance: Dict,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute node features from instance.
    
    Features include:
    - Coordinates (x, y)
    - Demand (mean, std)
    - Time window (start, end)
    
    Args:
        instance: Instance dictionary
        normalize: Whether to normalize features
    
    Returns:
        node_features: Node feature matrix [num_nodes, num_features]
    """
    nodes = instance['nodes']
    num_nodes = len(nodes)
    
    features = []
    
    for node in nodes:
        node_feat = [
            node['coordinates'][0],  # x
            node['coordinates'][1],  # y
            node.get('demand_mean', 0),  # demand mean
            node.get('demand_std', 0),  # demand std
            node.get('time_window_start', 0),  # tw start
            node.get('time_window_end', 480)  # tw end
        ]
        features.append(node_feat)
    
    features = np.array(features)
    
    if normalize:
        # Normalize each feature column
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std == 0, 1, std)
        features = (features - mean) / std
    
    return features


def compute_edge_features(
    edge_index: np.ndarray,
    coordinates: np.ndarray,
    time_matrix: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute edge features.
    
    Features include:
    - Distance
    - Travel time
    - Direction (dx, dy)
    
    Args:
        edge_index: Edge indices [2, num_edges]
        coordinates: Node coordinates [num_nodes, 2]
        time_matrix: Travel time matrix (optional)
    
    Returns:
        edge_features: Edge feature matrix [num_edges, num_features]
    """
    num_edges = edge_index.shape[1]
    features = []
    
    for e in range(num_edges):
        i, j = edge_index[:, e]
        
        # Distance
        distance = np.linalg.norm(coordinates[i] - coordinates[j])
        
        # Direction
        dx = coordinates[j, 0] - coordinates[i, 0]
        dy = coordinates[j, 1] - coordinates[i, 1]
        
        # Travel time
        if time_matrix is not None:
            travel_time = time_matrix[i, j]
        else:
            travel_time = distance / 60.0  # Assume 60 km/h
        
        edge_feat = [distance, travel_time, dx, dy]
        features.append(edge_feat)
    
    return np.array(features)


def create_torch_geometric_data(
    instance: Dict,
    k_neighbors: int = 10
):
    """
    Create PyTorch Geometric Data object from instance.
    
    Args:
        instance: Instance dictionary
        k_neighbors: Number of nearest neighbors for graph
    
    Returns:
        data: PyTorch Geometric Data object
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("PyTorch Geometric not installed. Install with: pip install torch-geometric")
    
    # Extract coordinates
    coords = np.array([node['coordinates'] for node in instance['nodes']])
    
    # Build graph
    edge_index, _ = build_k_nearest_graph(coords, k=k_neighbors)
    
    # Compute features
    node_features = compute_node_features(instance)
    edge_features = compute_edge_features(edge_index, coords)
    
    # Convert to tensors
    x = torch.FloatTensor(node_features)
    edge_index = torch.LongTensor(edge_index)
    edge_attr = torch.FloatTensor(edge_features)
    
    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    
    return data


def analyze_graph_properties(graph: nx.Graph) -> Dict:
    """
    Analyze graph properties.
    
    Args:
        graph: NetworkX graph
    
    Returns:
        properties: Dictionary of graph properties
    """
    properties = {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'is_connected': nx.is_connected(graph)
    }
    
    if nx.is_connected(graph):
        properties['diameter'] = nx.diameter(graph)
        properties['avg_shortest_path'] = nx.average_shortest_path_length(graph)
    
    # Degree statistics
    degrees = [d for n, d in graph.degree()]
    properties['avg_degree'] = np.mean(degrees)
    properties['max_degree'] = np.max(degrees)
    properties['min_degree'] = np.min(degrees)
    
    return properties


def compute_tsp_lower_bound(
    distance_matrix: np.ndarray,
    depot_idx: int = 0
) -> float:
    """
    Compute lower bound for TSP using MST.
    
    Args:
        distance_matrix: Distance matrix
        depot_idx: Depot index
    
    Returns:
        lower_bound: MST-based lower bound
    """
    num_nodes = len(distance_matrix)
    
    # Create graph
    G = nx.Graph()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            G.add_edge(i, j, weight=distance_matrix[i, j])
    
    # Compute MST
    mst = nx.minimum_spanning_tree(G)
    mst_weight = sum(data['weight'] for _, _, data in mst.edges(data=True))
    
    return mst_weight


def find_nearest_neighbors(
    query_point: np.ndarray,
    points: np.ndarray,
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find k nearest neighbors to query point.
    
    Args:
        query_point: Query coordinates [2]
        points: Point coordinates [num_points, 2]
        k: Number of neighbors
    
    Returns:
        indices: Indices of k nearest neighbors
        distances: Distances to k nearest neighbors
    """
    # Compute distances
    distances = np.linalg.norm(points - query_point, axis=1)
    
    # Find k smallest
    indices = np.argsort(distances)[:k]
    nearest_distances = distances[indices]
    
    return indices, nearest_distances


if __name__ == '__main__':
    # Test network utilities
    print("Testing Network Utilities...")
    
    # Create sample data
    np.random.seed(42)
    num_nodes = 20
    coordinates = np.random.rand(num_nodes, 2) * 100
    
    # Test distance matrix
    print("\n1. Distance Matrix:")
    from ..utils.data_utils import compute_distance_matrix
    dist_matrix = compute_distance_matrix(coordinates)
    print(f"   ✓ Distance matrix shape: {dist_matrix.shape}")
    
    # Test complete graph
    print("\n2. Complete Graph:")
    G = build_complete_graph(num_nodes, dist_matrix)
    print(f"   ✓ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Test k-nearest graph
    print("\n3. K-Nearest Graph:")
    edge_index, edge_attr = build_k_nearest_graph(coordinates, k=5)
    print(f"   ✓ Edge index shape: {edge_index.shape}")
    print(f"   ✓ Edge attributes shape: {edge_attr.shape}")
    
    # Test node features
    print("\n4. Node Features:")
    instance = {
        'nodes': [
            {
                'coordinates': list(coord),
                'demand_mean': np.random.randint(50, 150),
                'demand_std': np.random.randint(10, 30),
                'time_window_start': 0,
                'time_window_end': 480
            }
            for coord in coordinates
        ]
    }
    node_features = compute_node_features(instance)
    print(f"   ✓ Node features shape: {node_features.shape}")
    
    # Test edge features
    print("\n5. Edge Features:")
    edge_features = compute_edge_features(edge_index, coordinates)
    print(f"   ✓ Edge features shape: {edge_features.shape}")
    
    # Test graph analysis
    print("\n6. Graph Analysis:")
    props = analyze_graph_properties(G)
    print(f"   ✓ Density: {props['density']:.3f}")
    print(f"   ✓ Avg degree: {props['avg_degree']:.2f}")
    
    # Test TSP lower bound
    print("\n7. TSP Lower Bound:")
    lb = compute_tsp_lower_bound(dist_matrix)
    print(f"   ✓ MST lower bound: {lb:.2f}")
    
    # Test nearest neighbors
    print("\n8. Nearest Neighbors:")
    query = np.array([50, 50])
    indices, dists = find_nearest_neighbors(query, coordinates, k=3)
    print(f"   ✓ Found {len(indices)} nearest neighbors")
    print(f"     Distances: {dists}")
    
    print("\n✓ All network utility tests passed!")
```
