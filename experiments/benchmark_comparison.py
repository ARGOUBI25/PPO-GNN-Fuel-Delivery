"""
Benchmark Comparison
Compare all methods on standard benchmarks (Section 5.3).

Methods:
- PPO-GNN (proposed)
- Classical PPO
- PPO-MLP
- Clarke-Wright
- Gurobi (exact solver)

Author: Majdi Argoubi
Date: 2025
"""

import sys
sys.path.append('..')

import numpy as np
import json
import argparse
from pathlib import Path
import time

from src.evaluation.benchmarking import Benchmark, InstanceGenerator
from src.baselines.clarke_wright import ClarkeWrightSolver
from src.utils.logger import Logger


class BenchmarkComparison:
    """
    Comprehensive benchmark comparison (Section 5.3).
    
    Evaluates methods on multiple instance sizes with statistical analysis.
    """
    
    def __init__(
        self,
        data_dir: str = '../data/synthetic_networks/',
        output_dir: str = '../results/benchmark/',
        device: str = 'cuda'
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger = Logger(
            log_dir=output_dir,
            experiment_name='benchmark_comparison'
        )
        
        print("=" * 80)
        print("BENCHMARK COMPARISON")
        print("=" * 80)
    
    def run_benchmark(
        self,
        instance_sizes: list = [10, 50, 100],
        num_instances_per_size: int = 5,
        num_runs: int = 10
    ):
        """
        Run comprehensive benchmark.
        
        Args:
            instance_sizes: List of instance sizes
            num_instances_per_size: Instances per size
            num_runs: Runs per instance
        """
        print(f"\nBenchmark Configuration:")
        print(f"  Instance sizes: {instance_sizes}")
        print(f"  Instances per size: {num_instances_per_size}")
        print(f"  Runs per instance: {num_runs}")
        
        # Generate instances
        print("\nGenerating test instances...")
        generator = InstanceGenerator(base_network={'vehicle_capacity': 1000})
        instances = generator.generate_suite(
            sizes=instance_sizes,
            instances_per_size=num_instances_per_size
        )
        print(f"  ✓ Generated {len(instances)} instances")
        
        # Define methods
        methods = self._define_methods()
        
        # Optimal costs (if known)
        optimal_costs = {}  # Would be populated with known optima
        
        # Run benchmark
        print("\nRunning benchmark...")
        benchmark = Benchmark(
            instances=instances,
            methods=methods,
            optimal_costs=optimal_costs
        )
        
        results = benchmark.run(
            num_runs=num_runs,
            timeout=7200,  # 2 hour timeout
            verbose=True
        )
        
        # Generate report
        print("\nGenerating report...")
        summary = benchmark.generate_report()
        
        # Print results
        benchmark.print_summary()
        
        # Export results
        benchmark.export_results(
            output_dir=self.output_dir,
            format='csv'
        )
        
        benchmark.export_results(
            output_dir=self.output_dir,
            format='json'
        )
        
        benchmark.export_results(
            output_dir=self.output_dir,
            format='latex'
        )
        
        print(f"\n✓ Benchmark complete. Results saved to {self.output_dir}")
        
        return results, summary
    
    def _define_methods(self):
        """Define benchmark methods."""
        methods = {}
        
        # Clarke-Wright heuristic
        def clarke_wright_method(instance):
            network = {
                'nodes': instance['nodes'],
                'vehicle_capacity': instance.get('vehicle_capacity', 1000),
                'num_vehicles': instance.get('num_vehicles', 20),
                'max_route_time': 480,
                'vehicle_speed': 60.0
            }
            
            solver = ClarkeWrightSolver(
                network=network,
                vehicle_capacity=instance.get('vehicle_capacity', 1000),
                time_windows=False
            )
            
            routes, cost, solve_time = solver.solve()
            
            return {
                'cost': cost,
                'num_routes': len(routes),
                'feasible': True,
                'computation_time': solve_time
            }
        
        methods['Clarke-Wright'] = clarke_wright_method
        
        # Add other methods (PPO-GNN, etc.) here
        # In practice, these would load trained models
        
        print("\n  Methods defined:")
        for method_name in methods.keys():
            print(f"    - {method_name}")
        
        return methods


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run benchmark comparison')
    parser.add_argument('--data-dir', type=str, default='../data/synthetic_networks/')
    parser.add_argument('--output-dir', type=str, default='../results/benchmark/')
    parser.add_argument('--sizes', nargs='+', type=int, default=[10, 50, 100])
    parser.add_argument('--num-instances', type=int, default=5)
    parser.add_argument('--num-runs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    benchmark = BenchmarkComparison(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    benchmark.run_benchmark(
        instance_sizes=args.sizes,
        num_instances_per_size=args.num_instances,
        num_runs=args.num_runs
    )
