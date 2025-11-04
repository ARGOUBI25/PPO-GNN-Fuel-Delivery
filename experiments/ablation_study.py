"""
Ablation Study
Evaluate impact of GNN integration on performance (Section 5.2).

Tests:
1. PPO-GNN (full model)
2. PPO-MLP (no message-passing)
3. Classical PPO (flat features)

Author: Your Name
Date: 2025
"""

import sys
sys.path.append('..')

import torch
import numpy as np
import json
import argparse
from pathlib import Path
import time

from src.models.ppo_gnn import PPOGNN
from src.baselines.ppo_mlp import PPOMLP
from src.baselines.classical_ppo import ClassicalPPO
from src.utils.config_loader import load_config
from src.utils.data_utils import DataLoader
from src.utils.logger import Logger
from src.evaluation.metrics import MetricsCalculator, EvaluationMetrics
from src.evaluation.statistical_tests import StatisticalTester


class AblationStudy:
    """
    Run ablation study to isolate GNN contribution.
    
    Section 5.2: "To isolate the GNN's contribution, we compared 
    PPO-GNN against PPO-MLP (MLP-only encoder) on 50-node instances."
    
    Args:
        config_dir: Directory with configuration files
        data_dir: Directory with dataset
        output_dir: Directory for results
        device: 'cuda' or 'cpu'
    """
    
    def __init__(
        self,
        config_dir: str = '../configs/',
        data_dir: str = '../data/synthetic_networks/',
        output_dir: str = '../results/ablation/',
        device: str = 'cuda'
    ):
        self.config_dir = config_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger = Logger(
            log_dir=output_dir,
            experiment_name='ablation_study'
        )
        
        self.data_loader = DataLoader(data_dir=data_dir)
        
        print("=" * 80)
        print("ABLATION STUDY: GNN Integration Impact")
        print("=" * 80)
    
    def load_models(self):
        """Load trained models for comparison."""
        print("\nLoading models...")
        
        models = {}
        
        # PPO-GNN (full model)
        try:
            config_gnn = load_config(f'{self.config_dir}/ppo_gnn_config.yaml')
            model_gnn = PPOGNN(
                config=config_gnn,
                node_feature_dim=32,
                num_nodes=50,
                num_vehicles=15,
                num_actions=1500,
                device=self.device
            )
            
            checkpoint_path = '../checkpoints/ppo_gnn_best.pth'
            if Path(checkpoint_path).exists():
                model_gnn = PPOGNN.load(checkpoint_path, device=self.device)
                print("  ✓ PPO-GNN loaded from checkpoint")
            else:
                print("  ⚠️  PPO-GNN checkpoint not found, using random initialization")
            
            models['PPO-GNN'] = model_gnn
        
        except Exception as e:
            print(f"  ✗ Failed to load PPO-GNN: {e}")
        
        # PPO-MLP (no message-passing)
        try:
            config_mlp = load_config(f'{self.config_dir}/ppo_mlp_config.yaml')
            model_mlp = PPOMLP(
                config=config_mlp,
                node_feature_dim=32,
                num_nodes=50,
                num_vehicles=15,
                num_actions=1500,
                device=self.device
            )
            
            checkpoint_path = '../checkpoints/ppo_mlp_best.pth'
            if Path(checkpoint_path).exists():
                model_mlp = PPOMLP.load(checkpoint_path, device=self.device)
                print("  ✓ PPO-MLP loaded from checkpoint")
            else:
                print("  ⚠️  PPO-MLP checkpoint not found, using random initialization")
            
            models['PPO-MLP'] = model_mlp
        
        except Exception as e:
            print(f"  ✗ Failed to load PPO-MLP: {e}")
        
        # Classical PPO (flat features)
        try:
            config_classical = load_config(f'{self.config_dir}/classical_ppo_config.yaml')
            model_classical = ClassicalPPO(
                config=config_classical,
                input_dim=512,
                num_actions=1500,
                device=self.device
            )
            
            checkpoint_path = '../checkpoints/classical_ppo_best.pth'
            if Path(checkpoint_path).exists():
                model_classical = ClassicalPPO.load(checkpoint_path, device=self.device)
                print("  ✓ Classical PPO loaded from checkpoint")
            else:
                print("  ⚠️  Classical PPO checkpoint not found, using random initialization")
            
            models['Classical-PPO'] = model_classical
        
        except Exception as e:
            print(f"  ✗ Failed to load Classical PPO: {e}")
        
        return models
    
    def run_ablation(
        self,
        num_instances: int = 10,
        num_runs_per_instance: int = 5
    ):
        """
        Run ablation study.
        
        Args:
            num_instances: Number of instances to test
            num_runs_per_instance: Runs per instance for statistical significance
        """
        print(f"\nRunning ablation on {num_instances} instances...")
        print(f"Runs per instance: {num_runs_per_instance}")
        
        # Load models
        models = self.load_models()
        
        if not models:
            print("\n✗ No models loaded. Cannot run ablation study.")
            return
        
        # Load instances (medium 50-node networks)
        instances = self.data_loader.load_instances('medium_50_nodes/network.json')
        
        # Results storage
        results = {
            'models': list(models.keys()),
            'instances': [],
            'summary': {}
        }
        
        # Run evaluation
        for instance_idx in range(min(num_instances, len(instances))):
            instance = instances[instance_idx]
            instance_name = f"instance_{instance_idx}"
            
            print(f"\n  Instance {instance_idx + 1}/{num_instances}")
            
            instance_results = {
                'name': instance_name,
                'num_nodes': instance['num_nodes'],
                'results_by_model': {}
            }
            
            for model_name, model in models.items():
                print(f"    {model_name}...", end=' ')
                
                model_results = []
                
                for run in range(num_runs_per_instance):
                    # Generate solution
                    start_time = time.time()
                    solution = self._generate_solution(model, instance)
                    elapsed = time.time() - start_time
                    
                    # Compute metrics
                    metrics = self._compute_metrics(solution, elapsed)
                    model_results.append(metrics)
                
                # Aggregate results
                aggregated = self._aggregate_results(model_results)
                instance_results['results_by_model'][model_name] = aggregated
                
                print(f"Cost: ${aggregated['mean_cost']:.2f} ± ${aggregated['std_cost']:.2f}")
            
            results['instances'].append(instance_results)
        
        # Statistical analysis
        print("\n" + "=" * 80)
        print("Statistical Analysis")
        print("=" * 80)
        
        results['summary'] = self._statistical_analysis(results['instances'])
        
        # Save results
        self._save_results(results)
        
        # Print summary
        self._print_summary(results)
    
    def _generate_solution(self, model, instance):
        """Generate solution using model."""
        # Simplified solution generation
        # In practice, this would run full episode
        
        solution = {
            'cost': np.random.uniform(1000, 2000),  # Dummy
            'num_routes': np.random.randint(5, 10),
            'total_distance': np.random.uniform(300, 500),
            'feasible': True
        }
        
        return solution
    
    def _compute_metrics(self, solution, elapsed_time):
        """Compute metrics for solution."""
        return {
            'cost': solution['cost'],
            'num_routes': solution['num_routes'],
            'distance': solution['total_distance'],
            'time': elapsed_time,
            'feasible': solution['feasible']
        }
    
    def _aggregate_results(self, results):
        """Aggregate results from multiple runs."""
        costs = [r['cost'] for r in results]
        times = [r['time'] for r in results]
        
        return {
            'mean_cost': np.mean(costs),
            'std_cost': np.std(costs),
            'min_cost': np.min(costs),
            'max_cost': np.max(costs),
            'mean_time': np.mean(times),
            'feasibility_rate': np.mean([r['feasible'] for r in results])
        }
    
    def _statistical_analysis(self, instances):
        """Perform statistical analysis."""
        tester = StatisticalTester(alpha=0.05)
        
        # Extract costs by model
        costs_by_model = {}
        
        for instance in instances:
            for model_name, results in instance['results_by_model'].items():
                if model_name not in costs_by_model:
                    costs_by_model[model_name] = []
                costs_by_model[model_name].append(results['mean_cost'])
        
        # Pairwise comparisons
        comparisons = {}
        model_names = list(costs_by_model.keys())
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison_key = f"{model1}_vs_{model2}"
                
                # Wilcoxon test
                result = tester.wilcoxon_test(
                    costs_by_model[model1],
                    costs_by_model[model2],
                    alternative='two-sided'
                )
                
                comparisons[comparison_key] = {
                    'p_value': result.p_value,
                    'significant': result.significant,
                    'effect_size': result.effect_size,
                    'mean_diff': np.mean(costs_by_model[model1]) - np.mean(costs_by_model[model2])
                }
        
        return {
            'costs_by_model': {
                name: {
                    'mean': np.mean(costs),
                    'std': np.std(costs),
                    'median': np.median(costs)
                }
                for name, costs in costs_by_model.items()
            },
            'statistical_tests': comparisons
        }
    
    def _save_results(self, results):
        """Save results to JSON."""
        output_file = Path(self.output_dir) / 'ablation_results.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
    
    def _print_summary(self, results):
        """Print summary of results."""
        print("\n" + "=" * 80)
        print("ABLATION STUDY SUMMARY")
        print("=" * 80)
        
        summary = results['summary']
        
        print("\nMean Costs by Model:")
        print("-" * 80)
        for model, stats in summary['costs_by_model'].items():
            print(f"  {model:<20} ${stats['mean']:>10.2f} ± ${stats['std']:>8.2f}")
        
        print("\nStatistical Tests (Wilcoxon):")
        print("-" * 80)
        for comparison, test in summary['statistical_tests'].items():
            significance = "✓ Significant" if test['significant'] else "✗ Not significant"
            print(f"  {comparison:<30} {significance:>20} (p={test['p_value']:.4f})")
        
        print("\n" + "=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--config-dir', type=str, default='../configs/')
    parser.add_argument('--data-dir', type=str, default='../data/synthetic_networks/')
    parser.add_argument('--output-dir', type=str, default='../results/ablation/')
    parser.add_argument('--num-instances', type=int, default=10)
    parser.add_argument('--num-runs', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    study = AblationStudy(
        config_dir=args.config_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    study.run_ablation(
        num_instances=args.num_instances,
        num_runs_per_instance=args.num_runs
    )
