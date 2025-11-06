"""
Data Utilities
Data processing and loading utilities.

Handles dataset loading, preprocessing, and augmentation.

Author: Majdi Argoubi
Date: 2025
"""

import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, List, Tuple, Optional
import os


class DataLoader:
    """
    Data loader for VRP instances.
    
    Loads instances from various formats (JSON, pickle, CSV).
    
    Args:
        data_dir: Directory containing data files
        format: Data format ('json', 'pickle', 'csv')
    
    Example:
        >>> loader = DataLoader(data_dir='data/vrp_instances/')
        >>> instances = loader.load_instances('train_set.json')
    """
    
    def __init__(
        self,
        data_dir: str = 'data/',
        format: str = 'json'
    ):
        self.data_dir = data_dir
        self.format = format
        
        os.makedirs(data_dir, exist_ok=True)
    
    def load_instances(
        self,
        filename: str
    ) -> List[Dict]:
        """
        Load VRP instances from file.
        
        Args:
            filename: Filename (with or without path)
        
        Returns:
            instances: List of instance dictionaries
        """
        # Construct full path
        if not os.path.isabs(filename):
            filepath = os.path.join(self.data_dir, filename)
        else:
            filepath = filename
        
        # Load based on format
        if filepath.endswith('.json'):
            return self._load_json(filepath)
        elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            return self._load_pickle(filepath)
        elif filepath.endswith('.csv'):
            return self._load_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
    
    def _load_json(self, filepath: str) -> List[Dict]:
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle both single instance and list
        if isinstance(data, dict):
            return [data]
        else:
            return data
    
    def _load_pickle(self, filepath: str) -> List[Dict]:
        """Load from pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            return [data]
        else:
            return data
    
    def _load_csv(self, filepath: str) -> List[Dict]:
        """Load from CSV file (node coordinates and demands)."""
        df = pd.read_csv(filepath)
        
        # Convert to instance format
        instance = {
            'nodes': [],
            'num_nodes': len(df)
        }
        
        for _, row in df.iterrows():
            node = {
                'id': int(row.get('id', 0)),
                'coordinates': [float(row.get('x', 0)), float(row.get('y', 0))],
                'demand_mean': float(row.get('demand', 0)),
                'demand_std': float(row.get('demand_std', 0)),
                'time_window_start': float(row.get('tw_start', 0)),
                'time_window_end': float(row.get('tw_end', 480))
            }
            instance['nodes'].append(node)
        
        return [instance]
    
    def save_instances(
        self,
        instances: List[Dict],
        filename: str
    ):
        """
        Save instances to file.
        
        Args:
            instances: List of instance dictionaries
            filename: Output filename
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if filename.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(instances, f, indent=2)
        elif filename.endswith('.pkl') or filename.endswith('.pickle'):
            with open(filepath, 'wb') as f:
                pickle.dump(instances, f)
        else:
            raise ValueError(f"Unsupported format for saving: {filename}")
        
        print(f"Saved {len(instances)} instances to {filepath}")


class InstanceGenerator:
    """
    Generate synthetic VRP instances.
    
    Creates instances with various characteristics for testing.
    
    Args:
        seed: Random seed
    
    Example:
        >>> generator = InstanceGenerator(seed=42)
        >>> instance = generator.generate(num_nodes=50)
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate(
        self,
        num_nodes: int = 100,
        depot_location: Tuple[float, float] = (50, 50),
        area_size: float = 100.0,
        demand_range: Tuple[int, int] = (50, 150),
        demand_std_range: Tuple[int, int] = (10, 30),
        vehicle_capacity: float = 1000.0,
        num_vehicles: int = 15,
        time_window_width: float = 480.0
    ) -> Dict:
        """
        Generate a single VRP instance.
        
        Args:
            num_nodes: Number of nodes including depot
            depot_location: Depot coordinates
            area_size: Size of area (square)
            demand_range: Range for demand mean
            demand_std_range: Range for demand std
            vehicle_capacity: Vehicle capacity
            num_vehicles: Number of vehicles
            time_window_width: Time window width
        
        Returns:
            instance: Instance dictionary
        """
        nodes = []
        
        # Depot
        depot = {
            'id': 0,
            'coordinates': list(depot_location),
            'demand_mean': 0,
            'demand_std': 0,
            'time_window_start': 0,
            'time_window_end': time_window_width
        }
        nodes.append(depot)
        
        # Customer nodes
        for i in range(1, num_nodes):
            node = {
                'id': i,
                'coordinates': [
                    np.random.uniform(0, area_size),
                    np.random.uniform(0, area_size)
                ],
                'demand_mean': np.random.randint(*demand_range),
                'demand_std': np.random.randint(*demand_std_range),
                'time_window_start': 0,
                'time_window_end': time_window_width
            }
            nodes.append(node)
        
        instance = {
            'nodes': nodes,
            'num_nodes': num_nodes,
            'vehicle_capacity': vehicle_capacity,
            'num_vehicles': num_vehicles,
            'max_route_time': time_window_width,
            'vehicle_speed': 60.0,
            'fuel_cost_per_km': 0.5,
            'driver_cost_per_hour': 25.0,
            'vehicle_fixed_cost': 50.0
        }
        
        return instance
    
    def generate_batch(
        self,
        num_instances: int,
        **kwargs
    ) -> List[Dict]:
        """Generate multiple instances."""
        instances = []
        for i in range(num_instances):
            instance = self.generate(**kwargs)
            instance['name'] = f'instance_{i}'
            instances.append(instance)
        
        return instances


def normalize_coordinates(
    coordinates: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize coordinates to zero mean and unit variance.
    
    Args:
        coordinates: Node coordinates [num_nodes, 2]
        mean: Pre-computed mean (optional)
        std: Pre-computed std (optional)
    
    Returns:
        normalized: Normalized coordinates
        mean: Mean used for normalization
        std: Std used for normalization
    """
    if mean is None:
        mean = np.mean(coordinates, axis=0)
    if std is None:
        std = np.std(coordinates, axis=0)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
    
    normalized = (coordinates - mean) / std
    
    return normalized, mean, std


def compute_distance_matrix(
    coordinates: np.ndarray,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Compute distance matrix from coordinates.
    
    Args:
        coordinates: Node coordinates [num_nodes, 2]
        metric: Distance metric ('euclidean', 'manhattan')
    
    Returns:
        distance_matrix: Distance matrix [num_nodes, num_nodes]
    """
    num_nodes = len(coordinates)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                if metric == 'euclidean':
                    distance_matrix[i, j] = np.linalg.norm(
                        coordinates[i] - coordinates[j]
                    )
                elif metric == 'manhattan':
                    distance_matrix[i, j] = np.abs(
                        coordinates[i] - coordinates[j]
                    ).sum()
                else:
                    raise ValueError(f"Unknown metric: {metric}")
    
    return distance_matrix


def split_dataset(
    instances: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        instances: List of instances
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        shuffle: Shuffle before splitting
        seed: Random seed
    
    Returns:
        train_set: Training instances
        val_set: Validation instances
        test_set: Test instances
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    n = len(instances)
    indices = np.arange(n)
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_set = [instances[i] for i in train_indices]
    val_set = [instances[i] for i in val_indices]
    test_set = [instances[i] for i in test_indices]
    
    return train_set, val_set, test_set


if __name__ == '__main__':
    # Test data utilities
    print("Testing Data Utilities...")
    
    # Test instance generator
    print("\n1. Instance Generator:")
    generator = InstanceGenerator(seed=42)
    instance = generator.generate(num_nodes=20)
    print(f"   ✓ Generated instance with {instance['num_nodes']} nodes")
    
    # Test batch generation
    instances = generator.generate_batch(num_instances=5, num_nodes=10)
    print(f"   ✓ Generated {len(instances)} instances")
    
    # Test data loader (save and load)
    print("\n2. Data Loader:")
    loader = DataLoader(data_dir='test_data/')
    loader.save_instances(instances, 'test_instances.json')
    print("   ✓ Saved instances")
    
    loaded = loader.load_instances('test_instances.json')
    print(f"   ✓ Loaded {len(loaded)} instances")
    
    # Test normalization
    print("\n3. Coordinate Normalization:")
    coords = np.array([[0, 0], [10, 10], [20, 20]])
    normalized, mean, std = normalize_coordinates(coords)
    print(f"   ✓ Normalized coordinates")
    print(f"     Mean: {mean}")
    print(f"     Std: {std}")
    
    # Test distance matrix
    print("\n4. Distance Matrix:")
    dist_matrix = compute_distance_matrix(coords)
    print(f"   ✓ Distance matrix shape: {dist_matrix.shape}")
    print(f"     Distance (0,1): {dist_matrix[0, 1]:.2f}")
    
    # Test dataset split
    print("\n5. Dataset Split:")
    train, val, test = split_dataset(instances, shuffle=True)
    print(f"   ✓ Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    print("\n✓ All data utility tests passed!")
