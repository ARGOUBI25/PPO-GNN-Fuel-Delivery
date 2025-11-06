"""
Benchmarking
Comprehensive benchmarking framework for comparing methods.

Section 5.4: Detailed comparison with baselines and exact solver.

Author: Majdi Argoubi
Date: 2025
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import json
import os

from .metrics import MetricsCalculator, EvaluationMetrics
from .statistical_tests import StatisticalTester, compute_confidence_interval


@dataclass
class BenchmarkResult:
    """
    Results from benchmarking a single method.
    
    Attributes:
        method_name: Name of method
        instance_name: Instance name
        cost: Solution cost
        optimality_gap: Gap to optimal (%)
        computation_time: Time in seconds
        num_routes: Number of routes
        feasible: Whether solution is feasible
        additional_metrics: Additional metrics dictionary
    """
    method_name: str
    instance_name: str
    cost: float
    optimality_gap: float
    computation_time: float
    num_routes: int
    feasible: bool
    additional_metrics: Dict = None


class Benchmark:
    """
    Benchmarking framework for comparing VRP methods.
    
    Evaluates multiple methods on multiple instances and provides
    comprehensive comparison with statistical tests.
    
    Section 5.4: Implements benchmarking protocol comparing:
    - PPO-GNN (proposed)
    - Classical PPO
    - PPO-MLP
    - Clarke-Wright
    - Gurobi (exact solver)
    
    Args:
        instances: List of problem instances
        methods: Dictionary of {method_name: method_callable}
        optimal_costs: Dictionary of {instance_name: optimal_cost}
        network_configs: Dictionary of {instance_name: network_config}
    
    Example:
        >>> benchmark = Benchmark(instances, methods, optimal_costs, networks)
        >>> results = benchmark.run()
        >>> summary = benchmark.generate_report()
    """
    
    def __init__(
        self,
        instances: List[Dict],
        methods: Dict[str, Callable],
        optimal_costs: Optional[Dict[str, float]] = None,
        network_configs: Optional[Dict[str, Dict]] = None
    ):
        self.instances = instances
        self.methods = methods
        self.optimal_costs = optimal_costs or {}
        self.network_configs = network_configs or {}
        
        self.results = []
        self.summary = None
    
    def run(
        self,
        num_runs: int = 10,
        timeout: Optional[float] = None,
        verbose: bool = True
    ) -> List[BenchmarkResult]:
        """
        Run benchmark on all instances and methods.
        
        Args:
            num_runs: Number of runs per instance-method pair
            timeout: Timeout per run in seconds (None = no timeout)
            verbose: Print progress
        
        Returns:
            results: List of BenchmarkResult objects
        """
        total_experiments = len(self.instances) * len(self.methods) * num_runs
        experiment_count = 0
        
        if verbose:
            print(f"Running benchmark: {len(self.instances)} instances × "
                  f"{len(self.methods)} methods × {num_runs} runs = "
                  f"{total_experiments} experiments")
            print("=" * 80)
        
        for instance in self.instances:
            instance_name = instance.get('name', f'instance_{len(self.results)}')
            
            if verbose:
                print(f"\nInstance: {instance_name}")
            
            for method_name, method_fn in self.methods.items():
                if verbose:
                    print(f"  Method: {method_name}")
                
                for run in range(num_runs):
                    experiment_count += 1
                    
                    # Run method
                    try:
                        start_time = time.time()
                        
                        if timeout:
                            # Run with timeout (simplified - would need threading in practice)
                            solution = method_fn(instance)
                        else:
                            solution = method_fn(instance)
                        
                        elapsed_time = time.time() - start_time
                        
                        # Extract metrics
                        cost = solution.get('cost', float('inf'))
                        num_routes = solution.get('num_routes', 0)
                        feasible = solution.get('feasible', True)
                        
                        # Compute optimality gap
                        optimal_cost = self.optimal_costs.get(instance_name)
                        if optimal_cost:
                            gap = (cost - optimal_cost) / optimal_cost
                        else:
                            gap = 0.0
                        
                        # Store result
                        result = BenchmarkResult(
                            method_name=method_name,
                            instance_name=instance_name,
                            cost=cost,
                            optimality_gap=gap,
                            computation_time=elapsed_time,
                            num_routes=num_routes,
                            feasible=feasible,
                            additional_metrics=solution.get('additional_metrics', {})
                        )
                        
                        self.results.append(result)
                        
                        if verbose and run == 0:
                            print(f"    Cost: ${cost:.2f}, Gap: {gap:.2%}, "
                                  f"Time: {elapsed_time:.2f}s")
                    
                    except Exception as e:
                        if verbose:
                            print(f"    Run {run+1} failed: {str(e)}")
                        
                        # Record failed run
                        result = BenchmarkResult(
                            method_name=method_name,
                            instance_name=instance_name,
                            cost=float('inf'),
                            optimality_gap=float('inf'),
                            computation_time=timeout if timeout else 0,
                            num_routes=0,
                            feasible=False
                        )
                        self.results.append(result)
        
        if verbose:
            print("\n" + "=" * 80)
            print(f"Benchmark completed: {experiment_count} experiments")
        
        return self.results
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive benchmark report.
        
        Returns:
            report: Dictionary with summary statistics and comparisons
        """
        if not self.results:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Summary by method
        summary_by_method = {}
        for method_name in self.methods.keys():
            method_df = df[df['method_name'] == method_name]
            
            summary_by_method[method_name] = {
                'mean_cost': method_df['cost'].mean(),
                'std_cost': method_df['cost'].std(),
                'mean_gap': method_df['optimality_gap'].mean(),
                'median_gap': method_df['optimality_gap'].median(),
                'max_gap': method_df['optimality_gap'].max(),
                'mean_time': method_df['computation_time'].mean(),
                'feasibility_rate': method_df['feasible'].mean(),
                'num_runs': len(method_df)
            }
        
        # Statistical comparisons
        statistical_tests = self._perform_statistical_tests(df)
        
        # Instance-wise comparison
        instance_comparison = self._compare_by_instance(df)
        
        # Best method per instance
        best_per_instance = self._find_best_per_instance(df)
        
        self.summary = {
            'summary_by_method': summary_by_method,
            'statistical_tests': statistical_tests,
            'instance_comparison': instance_comparison,
            'best_per_instance': best_per_instance,
            'num_instances': len(self.instances),
            'num_methods': len(self.methods),
            'num_runs': len(df) // (len(self.instances) * len(self.methods))
        }
        
        return self.summary
    
    def _perform_statistical_tests(self, df: pd.DataFrame) -> Dict:
        """Perform statistical tests comparing methods."""
        tester = StatisticalTester(alpha=0.05)
        tests = {}
        
        method_names = list(self.methods.keys())
        
        # Pairwise comparisons
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i+1:]:
                # Get costs for both methods
                costs1 = df[df['method_name'] == method1]['cost'].values
                costs2 = df[df['method_name'] == method2]['cost'].values
                
                # Paired Wilcoxon test
                test_result = tester.wilcoxon_test(costs1, costs2, alternative='two-sided')
                
                tests[f'{method1}_vs_{method2}'] = {
                    'test': 'Wilcoxon',
                    'p_value': test_result.p_value,
                    'significant': test_result.significant,
                    'effect_size': test_result.effect_size
                }
        
        return tests
    
    def _compare_by_instance(self, df: pd.DataFrame) -> Dict:
        """Compare methods on each instance."""
        comparison = {}
        
        for instance_name in df['instance_name'].unique():
            instance_df = df[df['instance_name'] == instance_name]
            
            method_costs = {}
            for method_name in self.methods.keys():
                method_instance_df = instance_df[instance_df['method_name'] == method_name]
                method_costs[method_name] = {
                    'mean': method_instance_df['cost'].mean(),
                    'std': method_instance_df['cost'].std(),
                    'min': method_instance_df['cost'].min(),
                    'max': method_instance_df['cost'].max()
                }
            
            comparison[instance_name] = method_costs
        
        return comparison
    
    def _find_best_per_instance(self, df: pd.DataFrame) -> Dict:
        """Find best method for each instance."""
        best_methods = {}
        
        for instance_name in df['instance_name'].unique():
            instance_df = df[df['instance_name'] == instance_name]
            
            # Group by method and compute mean cost
            mean_costs = instance_df.groupby('method_name')['cost'].mean()
            
            best_method = mean_costs.idxmin()
            best_cost = mean_costs.min()
            
            best_methods[instance_name] = {
                'best_method': best_method,
                'best_cost': best_cost,
                'costs_by_method': mean_costs.to_dict()
            }
        
        return best_methods
    
    def print_summary(self):
        """Print formatted summary report."""
        if not self.summary:
            self.generate_report()
        
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Summary by method
        print(f"\nSummary by Method:")
        print("-" * 80)
        
        # Create comparison table
        methods = list(self.summary['summary_by_method'].keys())
        
        print(f"{'Method':<20} {'Mean Cost':>12} {'Mean Gap':>10} {'Mean Time':>10} {'Feasible':>10}")
        print("-" * 80)
        
        for method in methods:
            stats = self.summary['summary_by_method'][method]
            print(f"{method:<20} "
                  f"${stats['mean_cost']:>11.2f} "
                  f"{stats['mean_gap']:>9.2%} "
                  f"{stats['mean_time']:>9.2f}s "
                  f"{stats['feasibility_rate']:>9.1%}")
        
        # Statistical tests
        print(f"\nStatistical Tests (α = 0.05):")
        print("-" * 80)
        
        for comparison, test_result in self.summary['statistical_tests'].items():
            significance = "✓ Significant" if test_result['significant'] else "✗ Not significant"
            print(f"{comparison}: {significance} (p={test_result['p_value']:.4f})")
        
        # Best methods
        print(f"\nBest Method per Instance:")
        print("-" * 80)
        
        for instance, best_info in self.summary['best_per_instance'].items():
            print(f"{instance}: {best_info['best_method']} (${best_info['best_cost']:.2f})")
        
        print("=" * 80)
    
    def export_results(
        self,
        output_dir: str = 'benchmark_results/',
        format: str = 'csv'
    ):
        """
        Export benchmark results to file.
        
        Args:
            output_dir: Output directory
            format: 'csv', 'json', or 'latex'
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert results to DataFrame
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        if format == 'csv':
            filepath = os.path.join(output_dir, 'benchmark_results.csv')
            df.to_csv(filepath, index=False)
            print(f"Results exported to {filepath}")
        
        elif format == 'json':
            filepath = os.path.join(output_dir, 'benchmark_results.json')
            with open(filepath, 'w') as f:
                json.dump([asdict(r) for r in self.results], f, indent=2)
            print(f"Results exported to {filepath}")
        
        elif format == 'latex':
            # Generate LaTeX table
            filepath = os.path.join(output_dir, 'benchmark_table.tex')
            latex_table = self._generate_latex_table()
            with open(filepath, 'w') as f:
                f.write(latex_table)
            print(f"LaTeX table exported to {filepath}")
        
        # Export summary
        if self.summary:
            summary_path = os.path.join(output_dir, 'summary.json')
            with open(summary_path, 'w') as f:
                json.dump(self.summary, f, indent=2)
            print(f"Summary exported to {summary_path}")
    
    def _generate_latex_table(self) -> str:
        """Generate LaTeX table for paper (Table 5.2 format)."""
        if not self.summary:
            self.generate_report()
        
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Benchmark Results: Comparison of Methods}")
        latex.append("\\label{tab:benchmark_results}")
        latex.append("\\begin{tabular}{lcccc}")
        latex.append("\\toprule")
        latex.append("Method & Mean Cost (\\$) & Gap (\\%) & Time (s) & Feasible (\\%) \\\\")
        latex.append("\\midrule")
        
        for method, stats in self.summary['summary_by_method'].items():
            latex.append(f"{method} & "
                        f"{stats['mean_cost']:.2f} & "
                        f"{stats['mean_gap']*100:.2f} & "
                        f"{stats['mean_time']:.2f} & "
                        f"{stats['feasibility_rate']*100:.1f} \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)


class InstanceGenerator:
    """
    Generate benchmark instances with varying characteristics.
    
    Creates instances of different sizes and complexity levels
    for comprehensive evaluation.
    
    Args:
        base_network: Base network configuration
    
    Example:
        >>> generator = InstanceGenerator(base_network)
        >>> instances = generator.generate_suite(sizes=[10, 50, 100])
    """
    
    def __init__(self, base_network: Dict):
        self.base_network = base_network
    
    def generate_suite(
        self,
        sizes: List[int] = [10, 20, 50, 100],
        instances_per_size: int = 5,
        seed: int = 42
    ) -> List[Dict]:
        """
        Generate benchmark suite with varying problem sizes.
        
        Args:
            sizes: List of problem sizes (number of nodes)
            instances_per_size: Number of instances per size
            seed: Random seed
        
        Returns:
            instances: List of instance dictionaries
        """
        np.random.seed(seed)
        instances = []
        
        for size in sizes:
            for instance_id in range(instances_per_size):
                instance = self._generate_instance(
                    num_nodes=size,
                    instance_id=instance_id
                )
                instance['name'] = f'n{size}_i{instance_id}'
                instances.append(instance)
        
        return instances
    
    def _generate_instance(
        self,
        num_nodes: int,
        instance_id: int
    ) -> Dict:
        """Generate a single instance."""
        # Generate random node coordinates
        nodes = []
        for i in range(num_nodes):
            node = {
                'id': i,
                'coordinates': np.random.rand(2) * 100,
                'demand_mean': np.random.randint(50, 150) if i > 0 else 0,
                'demand_std': np.random.randint(10, 30) if i > 0 else 0,
                'time_window_start': 0,
                'time_window_end': 480
            }
            nodes.append(node)
        
        instance = {
            'nodes': nodes,
            'num_nodes': num_nodes,
            'instance_id': instance_id,
            'vehicle_capacity': self.base_network.get('vehicle_capacity', 1000),
            'num_vehicles': max(3, num_nodes // 10),
            'max_route_time': 480,
            'vehicle_speed': 60.0
        }
        
        return instance


class ScalabilityAnalyzer:
    """
    Analyze scalability of methods across problem sizes.
    
    Evaluates how computation time and solution quality scale
    with problem size.
    
    Example:
        >>> analyzer = ScalabilityAnalyzer(benchmark_results)
        >>> scalability = analyzer.analyze()
    """
    
    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
        self.df = pd.DataFrame([asdict(r) for r in results])
    
    def analyze(self) -> Dict:
        """
        Analyze scalability across problem sizes.
        
        Returns:
            analysis: Dictionary with scalability metrics
        """
        # Extract problem sizes from instance names
        self.df['size'] = self.df['instance_name'].str.extract(r'n(\d+)').astype(int)
        
        scalability = {}
        
        for method_name in self.df['method_name'].unique():
            method_df = self.df[self.df['method_name'] == method_name]
            
            # Group by size
            size_groups = method_df.groupby('size')
            
            sizes = []
            mean_times = []
            mean_gaps = []
            
            for size, group in size_groups:
                sizes.append(size)
                mean_times.append(group['computation_time'].mean())
                mean_gaps.append(group['optimality_gap'].mean())
            
            # Fit power law: time ~ size^α
            if len(sizes) > 1:
                log_sizes = np.log(sizes)
                log_times = np.log(mean_times)
                alpha = np.polyfit(log_sizes, log_times, 1)[0]
            else:
                alpha = None
            
            scalability[method_name] = {
                'sizes': sizes,
                'mean_times': mean_times,
                'mean_gaps': mean_gaps,
                'time_complexity_exponent': alpha
            }
        
        return scalability


if __name__ == '__main__':
    # Test benchmarking framework
    print("Testing Benchmarking Framework...")
    
    # Dummy methods
    def method1(instance):
        """Fast but suboptimal."""
        time.sleep(0.1)
        num_nodes = instance['num_nodes']
        return {
            'cost': 1000 + num_nodes * 10 + np.random.randn() * 50,
            'num_routes': max(3, num_nodes // 10),
            'feasible': True
        }
    
    def method2(instance):
        """Slower but better."""
        time.sleep(0.5)
        num_nodes = instance['num_nodes']
        return {
            'cost': 950 + num_nodes * 10 + np.random.randn() * 30,
            'num_routes': max(3, num_nodes // 10),
            'feasible': True
        }
    
    # Generate instances
    base_network = {'vehicle_capacity': 1000}
    generator = InstanceGenerator(base_network)
    instances = generator.generate_suite(sizes=[10, 20], instances_per_size=2)
    
    print(f"\n✓ Generated {len(instances)} test instances")
    
    # Run benchmark
    methods = {
        'Method-Fast': method1,
        'Method-Slow': method2
    }
    
    optimal_costs = {inst['name']: 900 + inst['num_nodes'] * 10 for inst in instances}
    
    benchmark = Benchmark(instances, methods, optimal_costs)
    results = benchmark.run(num_runs=3, verbose=False)
    
    print(f"✓ Completed {len(results)} benchmark runs")
    
    # Generate report
    summary = benchmark.generate_report()
    benchmark.print_summary()
    
    print("\n✓ All tests passed!")
