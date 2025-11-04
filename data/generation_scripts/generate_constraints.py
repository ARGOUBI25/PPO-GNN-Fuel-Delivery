"""
Operational Constraint Generator
Generate operational constraints for VRP instances.

Creates:
- Vehicle capacities
- Time windows
- Vehicle availability
- Fuel capacities
- Cost parameters

Author: Your Name
Date: 2025
"""

import numpy as np
import json
import os
from typing import Dict, List
import argparse


class ConstraintGenerator:
    """
    Generate operational constraints.
    
    Creates constraints based on network size and realistic parameters.
    
    Args:
        seed: Random seed
    
    Example:
        >>> generator = ConstraintGenerator(seed=42)
        >>> constraints = generator.generate(network, demands)
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate(
        self,
        network: Dict,
        demands: Dict,
        vehicle_capacity_factor: float = 1.5,
        time_window_width: float = 480.0,
        heterogeneous_fleet: bool = False
    ) -> Dict:
        """
        Generate constraints for instance.
        
        Args:
            network: Network dictionary
            demands: Demand dictionary
            vehicle_capacity_factor: Ratio of capacity to avg demand
            time_window_width: Time window width (minutes)
            heterogeneous_fleet: Use different vehicle types
        
        Returns:
            constraints: Constraint dictionary
        """
        num_nodes = network['num_nodes']
        
        # Compute vehicle parameters
        avg_demand = demands['statistics']['avg_demand_per_node']
        total_demand = demands['statistics']['total_mean_demand']
        
        # Vehicle capacity (Section 3.1)
        vehicle_capacity = avg_demand * vehicle_capacity_factor
        
        # Number of vehicles (enough to serve all demand)
        num_vehicles = int(np.ceil(total_demand / vehicle_capacity)) + 2
        num_vehicles = max(num_vehicles, max(3, num_nodes // 10))
        
        constraints = {
            'network_name': network['name'],
            'demand_level': demands['demand_level'],
            'vehicles': self._generate_vehicles(
                num_vehicles, 
                vehicle_capacity,
                heterogeneous_fleet
            ),
            'time_windows': self._generate_time_windows(
                num_nodes,
                time_window_width
            ),
            'costs': self._generate_costs(),
            'operational': {
                'max_route_time': float(time_window_width),
                'max_route_distance': float(network['metadata']['max_distance'] * 3),
                'service_time_per_unit': 0.1,  # minutes per kg
                'vehicle_speed': 60.0  # km/h
            }
        }
        
        return constraints
    
    def _generate_vehicles(
        self,
        num_vehicles: int,
        base_capacity: float,
        heterogeneous: bool
    ) -> List[Dict]:
        """Generate vehicle fleet."""
        vehicles = []
        
        if heterogeneous:
            # 3 vehicle types: small (70%), medium (20%), large (10%)
            types = [
                {'type': 'small', 'capacity': base_capacity * 0.7, 'cost_multiplier': 0.8},
                {'type': 'medium', 'capacity': base_capacity * 1.0, 'cost_multiplier': 1.0},
                {'type': 'large', 'capacity': base_capacity * 1.5, 'cost_multiplier': 1.3}
            ]
            
            for i in range(num_vehicles):
                if i < int(num_vehicles * 0.7):
                    vehicle_type = types[0]
                elif i < int(num_vehicles * 0.9):
                    vehicle_type = types[1]
                else:
                    vehicle_type = types[2]
                
                vehicle = {
                    'id': i,
                    'type': vehicle_type['type'],
                    'capacity': float(vehicle_type['capacity']),
                    'fuel_capacity': 100.0,  # kg H2
                    'fixed_cost': 50.0 * vehicle_type['cost_multiplier'],
                    'available': True
                }
                vehicles.append(vehicle)
        
        else:
            # Homogeneous fleet
            for i in range(num_vehicles):
                vehicle = {
                    'id': i,
                    'type': 'standard',
                    'capacity': float(base_capacity),
                    'fuel_capacity': 100.0,
                    'fixed_cost': 50.0,
                    'available': True
                }
                vehicles.append(vehicle)
        
        return vehicles
    
    def _generate_time_windows(
        self,
        num_nodes: int,
        window_width: float
    ) -> List[Dict]:
        """Generate time windows for nodes."""
        time_windows = []
        
        # Depot: full day
        depot_tw = {
            'node_id': 0,
            'earliest': 0.0,
            'latest': window_width,
            'service_time': 0.0
        }
        time_windows.append(depot_tw)
        
        # Customers: varying windows
        for i in range(1, num_nodes):
            # Random time window within day
            start = np.random.uniform(0, window_width * 0.3)
            end = start + np.random.uniform(window_width * 0.5, window_width)
            end = min(end, window_width)
            
            # Service time proportional to expected demand
            service_time = np.random.uniform(10, 30)
            
            tw = {
                'node_id': i,
                'earliest': float(start),
                'latest': float(end),
                'service_time': float(service_time)
            }
            time_windows.append(tw)
        
        return time_windows
    
    def _generate_costs(self) -> Dict:
        """Generate cost parameters (Section 3.1)."""
        return {
            'fuel_cost_per_km': 0.5,  # $/km
            'driver_cost_per_hour': 25.0,  # $/hour
            'vehicle_fixed_cost': 50.0,  # $ per vehicle used
            'penalty_unmet_demand': 10.0,  # $ per kg unmet
            'penalty_time_window': 5.0,  # $ per minute late
            'penalty_capacity': 100.0  # $ per kg over capacity
        }
    
    def generate_suite(
        self,
        networks_dir: str = '../synthetic_networks/',
        output_dir: Optional[str] = None
    ):
        """Generate constraints for all instances."""
        if output_dir is None:
            output_dir = networks_dir
        
        network_folders = ['small_10_nodes', 'medium_50_nodes',
                          'large_100_nodes', 'xlarge_200_nodes']
        
        for folder in network_folders:
            network_path = os.path.join(networks_dir, folder, 'network.json')
            
            if not os.path.exists(network_path):
                continue
            
            # Load network
            with open(network_path, 'r') as f:
                network = json.load(f)
            
            print(f"\nGenerating constraints for {network['name']}...")
            
            # For each demand level
            for demand_level in ['low', 'medium', 'high']:
                demands_path = os.path.join(
                    networks_dir, folder, f'demands_{demand_level}.json'
                )
                
                if not os.path.exists(demands_path):
                    continue
                
                with open(demands_path, 'r') as f:
                    demands = json.load(f)
                
                # Generate constraints
                constraints = self.generate(
                    network,
                    demands,
                    vehicle_capacity_factor=1.5,
                    heterogeneous_fleet=False
                )
                
                # Save
                output_path = os.path.join(
                    output_dir, folder, f'constraints_{demand_level}.json'
                )
                
                with open(output_path, 'w') as f:
                    json.dump(constraints, f, indent=2)
                
                print(f"  ✓ {demand_level}: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate operational constraints')
    parser.add_argument('--networks-dir', type=str, default='../synthetic_networks/',
                       help='Directory with network files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    generator = ConstraintGenerator(seed=args.seed)
    generator.generate_suite(networks_dir=args.networks_dir)
