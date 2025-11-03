#!/usr/bin/env python3
"""
Verify Hyperparameters
Checks that config files match Table 5.1 from the paper.
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}\n")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓{Colors.END} {text}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗{Colors.END} {text}")

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print_error(f"Failed to load {config_path}: {str(e)}")
        return None

def check_value(actual, expected, name: str, tolerance: float = 1e-6) -> bool:
    """Check if actual value matches expected."""
    if isinstance(expected, (int, float)):
        if abs(actual - expected) <= tolerance:
            print_success(f"{name}: {actual}")
            return True
        else:
            print_error(f"{name}: {actual} (expected {expected})")
            return False
    else:
        if actual == expected:
            print_success(f"{name}: {actual}")
            return True
        else:
            print_error(f"{name}: {actual} (expected {expected})")
            return False

def verify_config(config_path: Path) -> bool:
    """Verify configuration against Table 5.1."""
    
    print(f"\n{Colors.BOLD}Verifying {config_path.name}...{Colors.END}\n")
    
    config = load_config(config_path)
    if config is None:
        return False
    
    all_passed = True
    
    # Expected values from Table 5.1
    expected = {
        'policy.learning_rate': 3e-4,
        'value.learning_rate': 1e-3,
        'gnn.learning_rate': 1e-4,
        'policy.epsilon_clip': 0.2,
        'training.discount_factor': 0.99,
        'value.gae_lambda': 0.95,
        'training.batch_size': 256,
        'training.epochs_per_update': 10,
        'training.max_episodes': 50000,
        'gnn.num_layers': 3,
        'gnn.hidden_dim': 128,
        'rewards.lambda_cost': 1.0,
        'rewards.lambda_dispersion': 0.5,
        'rewards.lambda_delay': 0.8,
        'rewards.lambda_unmet': 1.2,
        'rewards.lambda_constraint': 2.0,
        'validation.tier1_threshold': 0.05,
        'validation.tier2_threshold': 0.25,
        'validation.validation_frequency': 1000,
    }
    
    # Check each parameter
    for key, expected_val in expected.items():
        parts = key.split('.')
        
        # Navigate nested dict
        try:
            val = config
            for part in parts:
                val = val[part]
            
            all_passed &= check_value(val, expected_val, key)
            
        except KeyError:
            print_error(f"{key}: not found in config")
            all_passed = False
    
    return all_passed

def main():
    """Main verification routine."""
    print_header("Hyperparameter Verification (Table 5.1)")
    
    # Config files to verify
    config_files = [
        Path('configs/ppo_gnn_config.yaml'),
        Path('configs/classical_ppo_config.yaml'),
        Path('configs/ppo_mlp_config.yaml'),
    ]
    
    results = {}
    
    for config_file in config_files:
        if not config_file.exists():
            print_error(f"{config_file} not found")
            results[config_file.name] = False
            continue
        
        passed = verify_config(config_file)
        results[config_file.name] = passed
    
    # Summary
    print_header("Verification Summary")
    
    for config_name, passed in results.items():
        if passed:
            print_success(f"{config_name}: All hyperparameters match Table 5.1")
        else:
            print_error(f"{config_name}: Some hyperparameters don't match")
    
    if all(results.values()):
        print(f"\n{Colors.GREEN}✓ All configurations verified!{Colors.END}\n")
        return 0
    else:
        print(f"\n{Colors.RED}✗ Some configurations have incorrect hyperparameters{Colors.END}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
