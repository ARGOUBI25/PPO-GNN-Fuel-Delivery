"""
Logger
TensorBoard and file logging utilities.

Provides comprehensive logging for training, validation, and evaluation.

Author: Majdi Argoubi
Date: 2025
"""

import os
import json
import time
from typing import Dict, Any, Optional, Union
from datetime import datetime
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


class Logger:
    """
    Comprehensive logger for experiments.
    
    Supports TensorBoard logging, file logging, and console output.
    
    Args:
        log_dir: Directory for logs
        experiment_name: Name of experiment
        use_tensorboard: Enable TensorBoard (default: True)
        console_log: Enable console logging (default: True)
    
    Example:
        >>> logger = Logger(log_dir='logs/', experiment_name='ppo_gnn_run1')
        >>> logger.log_scalar('train/reward', 150.5, step=1000)
        >>> logger.log_text('info', 'Training started')
        >>> logger.close()
    """
    
    def __init__(
        self,
        log_dir: str = 'logs/',
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
        console_log: bool = True
    ):
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.console_log = console_log
        
        # Create experiment name with timestamp
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f'experiment_{timestamp}'
        
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        
        # Create directories
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # TensorBoard writer
        if self.use_tensorboard:
            tensorboard_dir = os.path.join(self.experiment_dir, 'tensorboard')
            self.tb_writer = SummaryWriter(tensorboard_dir)
            print(f"TensorBoard logging to: {tensorboard_dir}")
        else:
            self.tb_writer = None
        
        # File logger
        self.log_file = os.path.join(self.experiment_dir, 'log.txt')
        self.metrics_file = os.path.join(self.experiment_dir, 'metrics.json')
        
        # Metrics storage
        self.metrics = {}
        
        # Start time
        self.start_time = time.time()
        
        # Log initialization
        self.log_text('init', f'Experiment: {experiment_name}')
        self.log_text('init', f'Log directory: {self.experiment_dir}')
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int
    ):
        """
        Log scalar value.
        
        Args:
            tag: Metric name (e.g., 'train/loss')
            value: Scalar value
            step: Step/episode number
        """
        # TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)
        
        # Store in metrics
        if tag not in self.metrics:
            self.metrics[tag] = {'steps': [], 'values': []}
        
        self.metrics[tag]['steps'].append(step)
        self.metrics[tag]['values'].append(float(value))
    
    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: int
    ):
        """
        Log multiple scalars.
        
        Args:
            main_tag: Main tag (e.g., 'train')
            tag_scalar_dict: Dictionary of {subtag: value}
            step: Step number
        """
        if self.tb_writer:
            self.tb_writer.add_scalars(main_tag, tag_scalar_dict, step)
        
        # Also log individually
        for tag, value in tag_scalar_dict.items():
            full_tag = f"{main_tag}/{tag}"
            self.log_scalar(full_tag, value, step)
    
    def log_histogram(
        self,
        tag: str,
        values: np.ndarray,
        step: int
    ):
        """
        Log histogram.
        
        Args:
            tag: Histogram name
            values: Array of values
            step: Step number
        """
        if self.tb_writer:
            self.tb_writer.add_histogram(tag, values, step)
    
    def log_text(
        self,
        tag: str,
        text: str,
        step: Optional[int] = None
    ):
        """
        Log text message.
        
        Args:
            tag: Message tag
            text: Text message
            step: Step number (optional)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Format message
        if step is not None:
            message = f"[{timestamp}] [{tag}] [Step {step}] {text}"
        else:
            message = f"[{timestamp}] [{tag}] {text}"
        
        # Console
        if self.console_log:
            print(message)
        
        # File
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
        
        # TensorBoard
        if self.tb_writer and step is not None:
            self.tb_writer.add_text(tag, text, step)
    
    def log_config(self, config: Dict):
        """
        Log configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_file = os.path.join(self.experiment_dir, 'config.json')
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.log_text('config', f'Configuration saved to {config_file}')
        
        # Also log to TensorBoard
        if self.tb_writer:
            config_text = json.dumps(config, indent=2)
            self.tb_writer.add_text('config', f'```json\n{config_text}\n```')
    
    def log_hyperparameters(
        self,
        hparams: Dict,
        metrics: Dict
    ):
        """
        Log hyperparameters with metrics.
        
        Args:
            hparams: Hyperparameter dictionary
            metrics: Metric dictionary
        """
        if self.tb_writer:
            self.tb_writer.add_hparams(hparams, metrics)
    
    def log_figure(
        self,
        tag: str,
        figure,
        step: int
    ):
        """
        Log matplotlib figure.
        
        Args:
            tag: Figure name
            figure: Matplotlib figure
            step: Step number
        """
        if self.tb_writer:
            self.tb_writer.add_figure(tag, figure, step)
    
    def save_checkpoint_info(
        self,
        checkpoint_path: str,
        metrics: Dict,
        step: int
    ):
        """
        Save checkpoint information.
        
        Args:
            checkpoint_path: Path to checkpoint
            metrics: Current metrics
            step: Step number
        """
        checkpoint_info = {
            'path': checkpoint_path,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        checkpoints_file = os.path.join(self.experiment_dir, 'checkpoints.json')
        
        # Load existing checkpoints
        if os.path.exists(checkpoints_file):
            with open(checkpoints_file, 'r') as f:
                checkpoints = json.load(f)
        else:
            checkpoints = []
        
        checkpoints.append(checkpoint_info)
        
        # Save
        with open(checkpoints_file, 'w') as f:
            json.dump(checkpoints, f, indent=2)
        
        self.log_text('checkpoint', f'Checkpoint saved: {checkpoint_path}', step)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since logger creation."""
        return time.time() - self.start_time
    
    def save_metrics(self):
        """Save all metrics to JSON file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.log_text('save', f'Metrics saved to {self.metrics_file}')
    
    def close(self):
        """Close logger and save all data."""
        # Save metrics
        self.save_metrics()
        
        # Close TensorBoard
        if self.tb_writer:
            self.tb_writer.close()
        
        # Log completion
        elapsed = self.get_elapsed_time()
        self.log_text('close', f'Experiment completed. Total time: {elapsed/3600:.2f} hours')


class MetricTracker:
    """
    Track metrics with statistics.
    
    Computes running mean, std, min, max for metrics.
    
    Args:
        window_size: Window size for moving statistics (default: 100)
    
    Example:
        >>> tracker = MetricTracker(window_size=100)
        >>> tracker.update('reward', 150.5)
        >>> stats = tracker.get_stats('reward')
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}
    
    def update(self, name: str, value: float):
        """Update metric with new value."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(value)
        
        # Keep only recent values
        if len(self.metrics[name]) > self.window_size:
            self.metrics[name].pop(0)
    
    def get_stats(self, name: str) -> Dict:
        """Get statistics for metric."""
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = self.metrics[name]
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'last': values[-1],
            'count': len(values)
        }
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all metrics."""
        return {name: self.get_stats(name) for name in self.metrics}


if __name__ == '__main__':
    # Test logger
    print("Testing Logger...")
    
    logger = Logger(log_dir='test_logs/', experiment_name='test_run')
    
    # Test scalar logging
    for step in range(10):
        logger.log_scalar('train/loss', 1.0 / (step + 1), step)
        logger.log_scalar('train/reward', step * 10, step)
    
    print("✓ Scalar logging tested")
    
    # Test text logging
    logger.log_text('info', 'Test message')
    print("✓ Text logging tested")
    
    # Test config
    config = {'lr': 0.001, 'batch_size': 256}
    logger.log_config(config)
    print("✓ Config logging tested")
    
    # Test metric tracker
    tracker = MetricTracker(window_size=5)
    for i in range(10):
        tracker.update('cost', 1000 - i * 10)
    
    stats = tracker.get_stats('cost')
    print(f"✓ Metric tracker: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    
    logger.close()
    print("\n✓ All logger tests passed!")
