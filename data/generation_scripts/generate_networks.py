"""
Network Topology Generator
Generate synthetic hydrogen delivery networks with varying complexity.

Creates networks of sizes: 10, 50, 100, 200 nodes.

Author: Your Name
Date: 2025
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple
import argparse


class NetworkGenerator:
    """
    Generate synthetic network topologies.
    
    Creates networks with:
    - Random spatial distribution of nodes
    - Depot location
    - Distance/time matrices
    - Network metadata
    
    Args:
        seed: Random seed for reproducibility
    
    Example:
        >>> generator = NetworkGenerator(seed=42)
        >>> network = generator.generate(num_nodes=50, name='medium_network')
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate(
        self,
        num_nodes: int,
        name: str,
        area_size: float = 100.0,
        depot_location: str = 'center',
        clustering: bool = False,
        num_clusters: int = 5
    ) -> Dict:
        """
        Generate a network topology.
        
        Args:
            num_nodes: Total number of nodes (including depot)
            name: Network name
            area_size: Size of square area (km)
            depot_location: 'center', 'corner', or 'random'
            clustering: Whether to cluster customer nodes
            num_clusters: Number of clusters (if clustering=True)
        
        Returns:
            network: Network dictionary
        """
        print(f"Generating network: {name} ({num_nodes} nodes)")
        
        # Generate depot
        depot = self._generate_depot(area_size, depot_location)
        
        # Generate customer nodes
        if clustering:
            customer_coords = self._generate_clustered_nodes(
                num_nodes - 1, area_size, num_clusters
            )
        else:
            customer_coords = self._generate_random_nodes(
                num_nodes - 1, area_size
            )
        
        # Combine depot and customers
        all_coords = np.vstack([depot['coordinates'], customer_coords])
        
        # Build nodes list
        nodes = [depot]
        for i in range(1, num_nodes):
            node = {
                'id': i,
                'coordinates': customer_coords[i-1].tolist(),
                'type': 'customer'
            }
            nodes.append(node)
        
        # Compute distance matrix
        distance_matrix = self._compute_distances(all_coords)
        
        # Compute time matrix (assuming 60 km/h)
        time_matrix = distance_matrix / 60.0 * 60  # minutes
        
        # Build edge list (complete graph)
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge = {
                    'from': i,
                    'to': j,
                    'distance': float(distance_matrix[i, j]),
                    'time': float(time_matrix[i, j])
                }
                edges.append(edge)
        
        # Network metadata
        network = {
            'name': name,
            'num_nodes': num_nodes,
            'num_edges': len(edges),
            'area_size': area_size,
            'depot_location': depot_location,
            'clustering': clustering,
            'nodes': nodes,
            'edges': edges,
            'distance_matrix': distance_matrix.tolist(),
            'time_matrix': time_matrix.tolist(),
            'metadata': {
                'generated_with': 'NetworkGenerator',
                'seed': self.seed,
                'avg_distance': float(np.mean(distance_matrix[distance_matrix > 0])),
                'max_distance': float(np.max(distance_matrix)),
                'min_distance': float(np.min(distance_matrix[distance_matrix > 0]))
            }
        }
        
        return network
    
    def _generate_depot(
        self,
        area_size: float,
        location: str
    ) -> Dict:
        """Generate depot node."""
        if location == 'center':
            coords = np.array([area_size / 2, area_size / 2])
        elif location == 'corner':
            coords = np.array([0, 0])
        elif location == 'random':
            coords = np.random.rand(2) * area_size
        else:
            coords = np.array([area_size / 2, area_size / 2])
        
        return {
            'id': 0,
            'coordinates': coords.tolist(),
            'type': 'depot'
        }
    
    def _generate_random_nodes(
        self,
        num_nodes: int,
        area_size: float
    ) -> np.ndarray:
        """Generate randomly distributed nodes."""
        return np.random.rand(num_nodes, 2) * area_size
    
    def _generate_clustered_nodes(
        self,
        num_nodes: int,
        area_size: float,
        num_clusters: int
    ) -> np.ndarray:
        """Generate clustered nodes."""
        # Generate cluster centers
        cluster_centers = np.random.rand(num_clusters, 2) * area_size
        
        # Assign nodes to clusters
        nodes_per_cluster = num_nodes // num_clusters
        coords = []
        
        for i in range(num_clusters):
            center = cluster_centers[i]
            cluster_size = nodes_per_cluster
            
            # Last cluster gets remaining nodes
            if i == num_clusters - 1:
                cluster_size = num_nodes - len(coords)
            
            # Generate nodes around center
            cluster_nodes = center + np.random.randn(cluster_size, 2) * (area_size / 10)
            
            # Clip to area
            cluster_nodes = np.clip(cluster_nodes, 0, area_size)
            
            coords.extend(cluster_nodes)
        
        return np.array(coords)
    
    def _compute_distances(self, coordinates: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance matrix."""
        num_nodes = len(coordinates)
        distances = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    distances[i, j] = np.linalg.norm(
                        coordinates[i] - coordinates[j]
                    )
        
        return distances
    
    def generate_suite(
        self,
        output_dir: str = '../synthetic_networks/'
    ):
        """
        Generate complete suite of networks.
        
        Creates: small (10), medium (50), large (100), xlarge (200).
        """
        os.makedirs(output_dir, exist_ok=True)
        
        configs = [
            {'num_nodes': 10, 'name': 'small_10_nodes', 'clustering': False},
            {'num_nodes': 50, 'name': 'medium_50_nodes', 'clustering': True, 'num_clusters': 5},
            {'num_nodes': 100, 'name': 'large_100_nodes', 'clustering': True, 'num_clusters': 10},
            {'num_nodes': 200, 'name': 'xlarge_200_nodes', 'clustering': True, 'num_clusters': 15}
        ]
        
        networks = []
        
        for config in configs:
            network = self.generate(**config)
            networks.append(network)
            
            # Save individual network
            network_dir = os.path.join(output_dir, config['name'])
            os.makedirs(network_dir, exist_ok=True)
            
            filepath = os.path.join(network_dir, 'network.json')
            with open(filepath, 'w') as f:
                json.dump(network, f, indent=2)
            
            print(f"  ✓ Saved to {filepath}")
        
        # Save summary
        summary = {
            'num_networks': len(networks),
            'networks': [
                {
                    'name': n['name'],
                    'num_nodes': n['num_nodes'],
                    'num_edges': n['num_edges'],
                    'avg_distance': n['metadata']['avg_distance']
                }
                for n in networks
            ]
        }
        
        summary_path = os.path.join(output_dir, 'networks_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Generated {len(networks)} networks")
        print(f"  Summary saved to {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate network topologies')
    parser.add_argument('--output-dir', type=str, default='../synthetic_networks/',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    generator = NetworkGenerator(seed=args.seed)
    generator.generate_suite(output_dir=args.output_dir)
