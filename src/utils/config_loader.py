"""
Configuration Loader
Load and validate YAML configuration files.

Ensures configuration consistency and provides default values.

Author: Your Name
Date: 2025
"""

import yaml
import os
from typing import Dict, Any, Optional
from copy import deepcopy


class ConfigLoader:
    """
    Configuration loader with validation.
    
    Loads YAML configuration files and validates against schema.
    Provides default values for missing parameters.
    
    Args:
        config_path: Path to YAML configuration file
        validate: Whether to validate configuration (default: True)
    
    Example:
        >>> loader = ConfigLoader('configs/ppo_gnn_config.yaml')
        >>> config = loader.load()
        >>> print(config['training']['learning_rate'])
    """
    
    def __init__(
        self,
        config_path: str,
        validate: bool = True
    ):
        self.config_path = config_path
        self.validate = validate
        
        # Default configuration schema
        self.default_config = self._get_default_config()
    
    def load(self) -> Dict:
        """
        Load configuration from file.
        
        Returns:
            config: Configuration dictionary
        """
        # Check file exists
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        # Load YAML
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Merge with defaults
        config = self._merge_with_defaults(config)
        
        # Validate
        if self.validate:
            self._validate_config(config)
        
        return config
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'experiment': {
                'name': 'ppo_gnn_experiment',
                'seed': 42,
                'device': 'cuda',
                'num_workers': 4
            },
            'network': {
                'num_nodes': 100,
                'num_vehicles': 15,
                'vehicle_capacity': 1000.0,
                'max_route_time': 480.0,
                'vehicle_speed': 60.0,
                'fuel_cost_per_km': 0.5,
                'driver_cost_per_hour': 25.0,
                'vehicle_fixed_cost': 50.0
            },
            'gnn': {
                'num_layers': 3,
                'hidden_dim': 64,
                'activation': 'relu',
                'dropout': 0.0,
                'aggregation': 'mean',
                'learning_rate': 1e-4,
                'update_frequency': 1000
            },
            'policy': {
                'architecture': [256, 128, 64],
                'activation': 'relu',
                'dropout': 0.0,
                'learning_rate': 3e-4,
                'epsilon_clip': 0.2,
                'entropy_coef': 0.01
            },
            'value': {
                'architecture': [256, 128],
                'activation': 'relu',
                'dropout': 0.0,
                'learning_rate': 1e-3,
                'gae_lambda': 0.95
            },
            'training': {
                'max_episodes': 50000,
                'batch_size': 2048,
                'epochs_per_update': 10,
                'episode_length': 500,
                'discount_factor': 0.99,
                'gradient_clip': 0.5,
                'early_stop_patience': 5000,
                'min_delta': 0.01
            },
            'validation': {
                'enabled': True,
                'validation_frequency': 1000,
                'constraint_weights': {
                    'capacity': 2.0,
                    'time_window': 1.5,
                    'demand': 1.8,
                    'sequencing': 1.0,
                    'station_limit': 1.2,
                    'max_route_time': 1.0
                },
                'adaptive_penalties': {
                    'tier1_threshold': 0.05,
                    'tier2_threshold': 0.25,
                    'tier2_multiplier': 1.5,
                    'tier2_episodes': 1000,
                    'tier3_multiplier': 10.0,
                    'tier3_episodes': 10000,
                    'selective_adjustment': True
                },
                'deterministic_model': {
                    'capacity_tolerance': 0.05,
                    'time_window_tolerance': 0.1,
                    'confidence_level': 0.95
                }
            },
            'logging': {
                'log_dir': 'logs/',
                'log_frequency': 100,
                'save_frequency': 5000,
                'tensorboard': True
            }
        }
    
    def _merge_with_defaults(self, config: Dict) -> Dict:
        """Merge configuration with defaults."""
        merged = deepcopy(self.default_config)
        
        def merge_dict(base: Dict, override: Dict):
            """Recursively merge dictionaries."""
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(merged, config)
        return merged
    
    def _validate_config(self, config: Dict):
        """Validate configuration."""
        # Check required keys
        required_sections = ['network', 'training', 'policy', 'value']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate value ranges
        self._validate_positive('network.vehicle_capacity', config['network']['vehicle_capacity'])
        self._validate_positive('training.max_episodes', config['training']['max_episodes'])
        self._validate_range('training.discount_factor', config['training']['discount_factor'], 0, 1)
        self._validate_range('policy.epsilon_clip', config['policy']['epsilon_clip'], 0, 1)
        self._validate_range('value.gae_lambda', config['value']['gae_lambda'], 0, 1)
        
        # Validate device
        valid_devices = ['cuda', 'cpu']
        device = config['experiment'].get('device', 'cuda')
        if device not in valid_devices:
            raise ValueError(f"Invalid device: {device}. Must be one of {valid_devices}")
    
    def _validate_positive(self, name: str, value: Any):
        """Validate value is positive."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    
    def _validate_range(self, name: str, value: Any, min_val: float, max_val: float):
        """Validate value is in range."""
        if not (min_val <= value <= max_val):
            raise ValueError(f"{name} must be in [{min_val}, {max_val}], got {value}")
    
    def save(self, config: Dict, output_path: str):
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Configuration saved to {output_path}")


def load_config(config_path: str) -> Dict:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        config: Configuration dictionary
    """
    loader = ConfigLoader(config_path)
    return loader.load()


def create_config_template(output_path: str = 'config_template.yaml'):
    """
    Create configuration template file.
    
    Args:
        output_path: Output file path
    """
    loader = ConfigLoader('dummy.yaml', validate=False)
    loader.save(loader.default_config, output_path)
    print(f"Configuration template created: {output_path}")


if __name__ == '__main__':
    # Test config loader
    print("Testing Config Loader...")
    
    # Create template
    create_config_template('test_config_template.yaml')
    print("✓ Template created")
    
    # Load template
    loader = ConfigLoader('test_config_template.yaml')
    config = loader.load()
    print("✓ Config loaded successfully")
    
    # Check some values
    print(f"  Learning rate: {config['policy']['learning_rate']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  GNN layers: {config['gnn']['num_layers']}")
    
    # Test validation
    try:
        bad_config = {'training': {'discount_factor': 1.5}}  # Invalid
        loader.default_config = bad_config
        loader._validate_config(bad_config)
    except ValueError as e:
        print(f"✓ Validation caught error: {e}")
    
    print("\n✓ All config loader tests passed!")
