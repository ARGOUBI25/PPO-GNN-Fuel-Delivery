"""
Hyperparameter Tuning
Grid search over hyperparameter space.

Tunes:
- Learning rates (α_θ, α_φ, α_ψ)
- PPO clip parameter (ε)
- Entropy coefficient (β)
- GAE lambda (λ)

Author: Majdi Argoubi
Date: 2025
"""

import sys
sys.path.append('..')

import numpy as np
import json
import itertools
import argparse
from pathlib import Path
from copy import deepcopy

from src.utils.logger import Logger
from src.utils.config_loader import load_config


class HyperparameterTuning:
    """
    Grid search hyperparameter tuning.
    
    Explores hyperparameter space to find optimal configuration.
    """
    
    def __init__(
        self,
        base_config_path: str = '../configs/ppo_gnn_config.yaml',
        output_dir: str = '../results/tuning/',
        device: str = 'cuda'
    ):
        self.base_config = load_config(base_config_path)
        self.output_dir = output_dir
        self.device = device
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger = Logger(
            log_dir=output_dir,
            experiment_name='hyperparameter_tuning'
        )
        
        print("=" * 80)
        print("HYPERPARAMETER TUNING")
        print("=" * 80)
    
    def define_search_space(self):
        """Define hyperparameter search space."""
        search_space = {
            'policy_lr': [1e-4, 3e-4, 1e-3],
            'value_lr': [5e-4, 1e-3, 3e-3],
            'gnn_lr': [5e-5, 1e-4, 5e-4],
            'epsilon_clip': [0.1, 0.2, 0.3],
            'entropy_coef': [0.001, 0.01, 0.05],
            'gae_lambda': [0.90, 0.95, 0.98],
            'discount_factor': [0.95, 0.99],
            'batch_size': [1024, 2048, 4096]
        }
        
        return search_space
    
    def run_grid_search(
        self,
        search_space: dict = None,
        max_trials: int = 50,
        episodes_per_trial: int = 5000
    ):
        """
        Run grid search.
        
        Args:
            search_space: Hyperparameter search space
            max_trials: Maximum number of trials
            episodes_per_trial: Training episodes per trial
        """
        if search_space is None:
            search_space = self.define_search_space()
        
        print(f"\nSearch space:")
        for param, values in search_space.items():
            print(f"  {param}: {values}")
        
        # Generate all combinations
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        
        all_combinations = list(itertools.product(*param_values))
        
        print(f"\nTotal combinations: {len(all_combinations)}")
        print(f"Running {min(max_trials, len(all_combinations))} trials")
        
        # Sample trials if too many
        if len(all_combinations) > max_trials:
            trial_indices = np.random.choice(
                len(all_combinations),
                size=max_trials,
                replace=False
            )
            trials = [all_combinations[i] for i in trial_indices]
        else:
            trials = all_combinations
        
        results = []
        
        for trial_idx, param_combo in enumerate(trials):
            print(f"\n{'='*80}")
            print(f"Trial {trial_idx + 1}/{len(trials)}")
            print(f"{'='*80}")
            
            # Create config for this trial
            trial_config = self._create_trial_config(param_names, param_combo)
            
            # Print config
            print("\nHyperparameters:")
            for param, value in zip(param_names, param_combo):
                print(f"  {param}: {value}")
            
            # Run training (simplified)
            trial_result = self._run_trial(
                trial_config,
                episodes=episodes_per_trial,
                trial_id=trial_idx
            )
            
            trial_result['hyperparameters'] = dict(zip(param_names, param_combo))
            results.append(trial_result)
            
            # Print result
            print(f"\n  Result: Cost = {trial_result['final_cost']:.2f}")
        
        # Find best configuration
        best_trial = min(results, key=lambda x: x['final_cost'])
        
        print("\n" + "=" * 80)
        print("BEST CONFIGURATION")
        print("=" * 80)
        print(f"\nBest cost: {best_trial['final_cost']:.2f}")
        print("\nBest hyperparameters:")
        for param, value in best_trial['hyperparameters'].items():
            print(f"  {param}: {value}")
        
        # Save results
        self._save_results(results, best_trial)
        
        return results, best_trial
    
    def _create_trial_config(self, param_names, param_values):
        """Create config for trial."""
        config = deepcopy(self.base_config)
        
        # Update config with trial parameters
        param_dict = dict(zip(param_names, param_values))
        
        if 'policy_lr' in param_dict:
            config['policy']['learning_rate'] = param_dict['policy_lr']
        
        if 'value_lr' in param_dict:
            config['value']['learning_rate'] = param_dict['value_lr']
        
        if 'gnn_lr' in param_dict:
            config['gnn']['learning_rate'] = param_dict['gnn_lr']
        
        if 'epsilon_clip' in param_dict:
            config['policy']['epsilon_clip'] = param_dict['epsilon_clip']
        
        if 'entropy_coef' in param_dict:
            config['policy']['entropy_coef'] = param_dict['entropy_coef']
        
        if 'gae_lambda' in param_dict:
            config['value']['gae_lambda'] = param_dict['gae_lambda']
        
        if 'discount_factor' in param_dict:
            config['training']['discount_factor'] = param_dict['discount_factor']
        
        if 'batch_size' in param_dict:
            config['training']['batch_size'] = param_dict['batch_size']
        
        return config
    
    def _run_trial(self, config, episodes, trial_id):
        """Run single trial (simplified)."""
        # In practice, this would train the model
        # For now, simulate with random performance
        
        final_cost = 1000 + np.random.randn() * 100
        
        return {
            'trial_id': trial_id,
            'final_cost': float(final_cost),
            'episodes': episodes
        }
    
    def _save_results(self, results, best_trial):
        """Save tuning results."""
        output_file = Path(self.output_dir) / 'tuning_results.json'
        
        data = {
            'all_trials': results,
            'best_trial': best_trial,
            'num_trials': len(results)
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter tuning')
    parser.add_argument('--config', type=str, default='../configs/ppo_gnn_config.yaml')
    parser.add_argument('--output-dir', type=str, default='../results/tuning/')
    parser.add_argument('--max-trials', type=int, default=50)
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    tuning = HyperparameterTuning(
        base_config_path=args.config,
        output_dir=args.output_dir,
        device=args.device
    )
    
    tuning.run_grid_search(
        max_trials=args.max_trials,
        episodes_per_trial=args.episodes
    )
