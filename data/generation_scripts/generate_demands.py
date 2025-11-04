"""
Stochastic Demand Generator
Generate stochastic hydrogen demands for network nodes.

Creates demands with:
- Mean demand per node
- Standard deviation (uncertainty)
- Temporal patterns (optional)
- Demand scenarios

Author: Your Name
Date: 2025
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional
import argparse


class DemandGenerator:
    """
    Generate stochastic demands for hydrogen delivery.
    
    Creates demands following normal distribution:
        d_i ~ N(μ_i, σ_i²)
    
    Args:
        seed: Random seed
    
    Example:
        >>> generator = DemandGenerator(seed=42)
        >>> demands = generator.generate(network, demand_level='medium')
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate(
        self,
        network: Dict,
        demand_level: str = 'medium',
        cv: float = 0.2,
        temporal_variation: bool = False
    ) -> Dict:
        """
        Generate demands for network.
        
        Args:
            network: Network dictionary
            demand_level: 'low', 'medium', 'high' (affects mean demand)
            cv: Coefficient of variation (σ/μ) for stochasticity
            temporal_variation: Include time-of-day variations
        
        Returns:
            demands: Demand dictionary
        """
        num_nodes = network['num_nodes']
        
        # Base demand levels (kg H2 per day)
        demand_levels = {
            'low': (50, 100),
            'medium': (100, 200),
            'high': (200, 400)
        }
        
        min_demand, max_demand = demand_levels[demand_level]
        
        demands = {
            'network_name': network['name'],
            'demand_level': demand_level,
            'cv': cv,
            'temporal_variation': temporal_variation,
            'nodes': []
        }
        
        # Depot has no demand
        depot_demand = {
            'node_id': 0,
            'demand_mean': 0,
            'demand_std': 0,
            'demand_min': 0,
            'demand_max': 0
        }
        demands['nodes'].append(depot_demand)
        
        # Generate customer demands
        for i in range(1, num_nodes):
            # Random mean demand
            mean_demand = np.random.uniform(min_demand, max_demand)
            
            # Standard deviation based on CV
            std_demand = cv * mean_demand
            
            # Min/max bounds (3σ rule)
            min_bound = max(0, mean_demand - 3 * std_demand)
            max_bound = mean_demand + 3 * std_demand
            
            node_demand = {
                'node_id': i,
                'demand_mean': float(mean_demand),
                'demand_std': float(std_demand),
                'demand_min': float(min_bound),
                'demand_max': float(max_bound),
                'cv': cv
            }
            
            # Add temporal variation if requested
            if temporal_variation:
                node_demand['hourly_factors'] = self._generate_hourly_factors()
            
            demands['nodes'].append(node_demand)
        
        # Generate scenarios
        demands['scenarios'] = self._generate_scenarios(demands['nodes'], num_scenarios=100)
        
        # Statistics
        customer_demands = [d['demand_mean'] for d in demands['nodes'][1:]]
        demands['statistics'] = {
            'total_mean_demand': float(np.sum(customer_demands)),
            'avg_demand_per_node': float(np.mean(customer_demands)),
            'std_demand_per_node': float(np.std(customer_demands)),
            'max_demand': float(np.max(customer_demands)),
            'min_demand': float(np.min(customer_demands))
        }
        
        return demands
    
    def _generate_hourly_factors(self) -> List[float]:
        """Generate hourly demand factors (24 hours)."""
        # Peak hours: 8-12, 14-18
        base_pattern = np.ones(24) * 0.7
        
        # Morning peak
        base_pattern[8:12] = 1.2
        
        # Afternoon peak
        base_pattern[14:18] = 1.1
        
        # Add noise
        noise = np.random.randn(24) * 0.1
        factors = base_pattern + noise
        
        # Normalize to mean = 1.0
        factors = factors / factors.mean()
        
        return factors.tolist()
    
    def _generate_scenarios(
        self,
        nodes: List[Dict],
        num_scenarios: int = 100
    ) -> List[Dict]:
        """Generate demand scenarios for stochastic evaluation."""
        scenarios = []
        
        for scenario_id in range(num_scenarios):
            scenario = {
                'scenario_id': scenario_id,
                'demands': []
            }
            
            for node in nodes:
                if node['demand_mean'] == 0:  # Depot
                    scenario['demands'].append(0)
                else:
                    # Sample from normal distribution
                    demand_sample = np.random.normal(
                        node['demand_mean'],
                        node['demand_std']
                    )
                    # Clip to bounds
                    demand_sample = np.clip(
                        demand_sample,
                        node['demand_min'],
                        node['demand_max']
                    )
                    scenario['demands'].append(float(demand_sample))
            
            scenarios.append(scenario)
        
        return scenarios
    
    def generate_suite(
        self,
        networks_dir: str = '../synthetic_networks/',
        output_dir: Optional[str] = None
    ):
        """
        Generate demands for all networks.
        
        Args:
            networks_dir: Directory with network files
            output_dir: Output directory (defaults to networks_dir)
        """
        if output_dir is None:
            output_dir = networks_dir
        
        network_folders = ['small_10_nodes', 'medium_50_nodes', 
                          'large_100_nodes', 'xlarge_200_nodes']
        
        for folder in network_folders:
            network_path = os.path.join(networks_dir, folder, 'network.json')
            
            if not os.path.exists(network_path):
                print(f"⚠️  Network not found: {network_path}")
                continue
            
            # Load network
            with open(network_path, 'r') as f:
                network = json.load(f)
            
            print(f"\nGenerating demands for {network['name']}...")
            
            # Generate for different demand levels
            for demand_level in ['low', 'medium', 'high']:
                demands = self.generate(
                    network,
                    demand_level=demand_level,
                    cv=0.2,
                    temporal_variation=False
                )
                
                # Save
                output_path = os.path.join(
                    output_dir, folder, f'demands_{demand_level}.json'
                )
                
                with open(output_path, 'w') as f:
                    json.dump(demands, f, indent=2)
                
                print(f"  ✓ {demand_level}: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate stochastic demands')
    parser.add_argument('--networks-dir', type=str, default='../synthetic_networks/',
                       help='Directory with network files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    generator = DemandGenerator(seed=args.seed)
    generator.generate_suite(networks_dir=args.networks_dir)
