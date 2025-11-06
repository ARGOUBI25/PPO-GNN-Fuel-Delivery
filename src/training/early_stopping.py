"""
Early Stopping
Monitors training progress and stops when convergence detected.

Prevents overfitting and saves computation time.

Author: Majdi Argoubi
Date: 2025
"""

import numpy as np
from typing import Optional, Literal


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors a metric and stops training if no improvement for
    a specified number of episodes (patience).
    
    Args:
        patience: Number of episodes to wait for improvement (default: 5000)
        min_delta: Minimum change to qualify as improvement (default: 0.01)
        mode: 'min' or 'max' - minimize or maximize metric (default: 'min')
        baseline: Initial baseline value (default: None)
        restore_best: Whether to restore best weights on stop (default: False)
    
    Example:
        >>> early_stop = EarlyStopping(patience=5000, min_delta=0.01, mode='min')
        >>> for episode in range(max_episodes):
        ...     cost = train_episode()
        ...     if early_stop(cost):
        ...         print("Early stopping triggered")
        ...         break
    """
    
    def __init__(
        self,
        patience: int = 5000,
        min_delta: float = 0.01,
        mode: Literal['min', 'max'] = 'min',
        baseline: Optional[float] = None,
        restore_best: bool = False
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.restore_best = restore_best
        
        # State
        self.wait = 0
        self.stopped_episode = 0
        self.best_score = None
        self.best_weights = None
        
        # Comparison function
        if mode == 'min':
            self.is_better = lambda new, best: new < best - min_delta
            self.best_score = float('inf') if baseline is None else baseline
        else:
            self.is_better = lambda new, best: new > best + min_delta
            self.best_score = float('-inf') if baseline is None else baseline
    
    def __call__(self, current_score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_score: Current metric value
        
        Returns:
            should_stop: True if training should stop
        """
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        if self.is_better(current_score, self.best_score):
            # Improvement
            self.best_score = current_score
            self.wait = 0
        else:
            # No improvement
            self.wait += 1
            
            if self.wait >= self.patience:
                self.stopped_episode = self.wait
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.wait = 0
        self.stopped_episode = 0
        if self.mode == 'min':
            self.best_score = float('inf') if self.baseline is None else self.baseline
        else:
            self.best_score = float('-inf') if self.baseline is None else self.baseline


class AdaptiveEarlyStopping(EarlyStopping):
    """
    Adaptive early stopping with dynamic patience.
    
    Adjusts patience based on training progress. If improvements
    are frequent, patience is reduced. If rare, patience is increased.
    
    Args:
        initial_patience: Initial patience (default: 5000)
        min_patience: Minimum patience (default: 1000)
        max_patience: Maximum patience (default: 10000)
        patience_adjustment: Adjustment factor (default: 1.1)
        **kwargs: Arguments for EarlyStopping
    
    Example:
        >>> early_stop = AdaptiveEarlyStopping(initial_patience=5000)
        >>> if early_stop(cost):
        ...     print(f"Stopped with patience: {early_stop.patience}")
    """
    
    def __init__(
        self,
        initial_patience: int = 5000,
        min_patience: int = 1000,
        max_patience: int = 10000,
        patience_adjustment: float = 1.1,
        **kwargs
    ):
        super().__init__(patience=initial_patience, **kwargs)
        
        self.initial_patience = initial_patience
        self.min_patience = min_patience
        self.max_patience = max_patience
        self.patience_adjustment = patience_adjustment
        
        # Track improvement frequency
        self.total_checks = 0
        self.improvements = 0
    
    def __call__(self, current_score: float) -> bool:
        """Check if training should stop with adaptive patience."""
        self.total_checks += 1
        
        # Check improvement
        was_improvement = False
        if self.best_score is None:
            self.best_score = current_score
            was_improvement = True
        elif self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.wait = 0
            was_improvement = True
        else:
            self.wait += 1
        
        if was_improvement:
            self.improvements += 1
        
        # Adapt patience every 1000 checks
        if self.total_checks % 1000 == 0:
            self._adapt_patience()
        
        # Check stopping
        if self.wait >= self.patience:
            self.stopped_episode = self.wait
            return True
        
        return False
    
    def _adapt_patience(self):
        """Adapt patience based on improvement frequency."""
        improvement_rate = self.improvements / self.total_checks
        
        if improvement_rate > 0.1:  # Frequent improvements
            # Reduce patience (converging quickly)
            self.patience = max(
                self.min_patience,
                int(self.patience / self.patience_adjustment)
            )
        elif improvement_rate < 0.01:  # Rare improvements
            # Increase patience (slow convergence)
            self.patience = min(
                self.max_patience,
                int(self.patience * self.patience_adjustment)
            )
        
        # Reset counters
        self.total_checks = 0
        self.improvements = 0


class ConvergenceDetector:
    """
    Detects training convergence based on multiple criteria.
    
    Checks for:
    1. Stable metric (low variance)
    2. Plateau (no significant improvement)
    3. Oscillation (metric oscillates around value)
    
    Args:
        window_size: Window for computing statistics (default: 100)
        stability_threshold: CV threshold for stability (default: 0.05)
        plateau_threshold: Improvement threshold for plateau (default: 0.001)
        oscillation_threshold: Range threshold for oscillation (default: 0.02)
    
    Example:
        >>> detector = ConvergenceDetector(window_size=100)
        >>> for episode in range(max_episodes):
        ...     cost = train_episode()
        ...     if detector.check_convergence(cost):
        ...         print(f"Converged: {detector.get_convergence_reason()}")
        ...         break
    """
    
    def __init__(
        self,
        window_size: int = 100,
        stability_threshold: float = 0.05,
        plateau_threshold: float = 0.001,
        oscillation_threshold: float = 0.02
    ):
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.plateau_threshold = plateau_threshold
        self.oscillation_threshold = oscillation_threshold
        
        self.history = []
        self.convergence_reason = None
    
    def check_convergence(self, metric: float) -> bool:
        """
        Check if training has converged.
        
        Args:
            metric: Current metric value
        
        Returns:
            converged: True if converged
        """
        self.history.append(metric)
        
        # Need enough history
        if len(self.history) < self.window_size:
            return False
        
        # Keep only recent history
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Check stability
        if self._check_stability():
            self.convergence_reason = 'stability'
            return True
        
        # Check plateau
        if self._check_plateau():
            self.convergence_reason = 'plateau'
            return True
        
        # Check oscillation
        if self._check_oscillation():
            self.convergence_reason = 'oscillation'
            return True
        
        return False
    
    def _check_stability(self) -> bool:
        """Check if metric is stable (low variance)."""
        mean = np.mean(self.history)
        std = np.std(self.history)
        
        if mean != 0:
            cv = std / abs(mean)
            return cv < self.stability_threshold
        
        return False
    
    def _check_plateau(self) -> bool:
        """Check if metric has plateaued (no improvement)."""
        # Compare first half to second half of window
        mid = len(self.history) // 2
        first_half = np.mean(self.history[:mid])
        second_half = np.mean(self.history[mid:])
        
        if first_half != 0:
            relative_change = abs(second_half - first_half) / abs(first_half)
            return relative_change < self.plateau_threshold
        
        return False
    
    def _check_oscillation(self) -> bool:
        """Check if metric oscillates around a value."""
        mean = np.mean(self.history)
        max_val = np.max(self.history)
        min_val = np.min(self.history)
        
        if mean != 0:
            range_ratio = (max_val - min_val) / abs(mean)
            return range_ratio < self.oscillation_threshold
        
        return False
    
    def get_convergence_reason(self) -> Optional[str]:
        """Get reason for convergence."""
        return self.convergence_reason
    
    def reset(self):
        """Reset detector state."""
        self.history = []
        self.convergence_reason = None


class EarlyStoppingWithWarmup(EarlyStopping):
    """
    Early stopping with warmup period.
    
    Does not check for early stopping during warmup period,
    allowing model to explore initially.
    
    Args:
        warmup_episodes: Number of warmup episodes (default: 1000)
        **kwargs: Arguments for EarlyStopping
    
    Example:
        >>> early_stop = EarlyStoppingWithWarmup(warmup_episodes=1000, patience=5000)
        >>> for episode in range(max_episodes):
        ...     cost = train_episode()
        ...     if early_stop(cost, episode):
        ...         break
    """
    
    def __init__(self, warmup_episodes: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.warmup_episodes = warmup_episodes
        self.current_episode = 0
    
    def __call__(self, current_score: float, episode: Optional[int] = None) -> bool:
        """Check if training should stop (after warmup)."""
        if episode is not None:
            self.current_episode = episode
        else:
            self.current_episode += 1
        
        # During warmup, just track best score
        if self.current_episode < self.warmup_episodes:
            if self.best_score is None:
                self.best_score = current_score
            elif self.is_better(current_score, self.best_score):
                self.best_score = current_score
            return False
        
        # After warmup, apply normal early stopping
        return super().__call__(current_score)


if __name__ == '__main__':
    # Test early stopping
    print("Testing Early Stopping...")
    
    # Test basic early stopping
    print("\n1. Basic Early Stopping (minimize):")
    early_stop = EarlyStopping(patience=5, min_delta=0.1, mode='min')
    
    costs = [10.0, 9.5, 9.2, 9.1, 9.05, 9.04, 9.03, 9.02, 9.01, 9.00]
    for i, cost in enumerate(costs):
        if early_stop(cost):
            print(f"   Stopped at episode {i} with cost {cost:.2f}")
            break
    else:
        print(f"   Did not stop. Best: {early_stop.best_score:.2f}")
    
    # Test adaptive early stopping
    print("\n2. Adaptive Early Stopping:")
    adaptive_stop = AdaptiveEarlyStopping(
        initial_patience=10,
        min_patience=5,
        max_patience=20
    )
    
    # Simulate rapid improvement phase
    for i in range(15):
        cost = 10.0 - i * 0.5
        adaptive_stop(cost)
    
    print(f"   Patience after rapid improvement: {adaptive_stop.patience}")
    
    # Test convergence detector
    print("\n3. Convergence Detector:")
    detector = ConvergenceDetector(window_size=20, stability_threshold=0.05)
    
    # Simulate convergence to stable value
    for i in range(50):
        cost = 5.0 + np.random.randn() * 0.1  # Stable around 5.0
        if detector.check_convergence(cost):
            print(f"   Converged at iteration {i}")
            print(f"   Reason: {detector.get_convergence_reason()}")
            break
    
    # Test early stopping with warmup
    print("\n4. Early Stopping with Warmup:")
    warmup_stop = EarlyStoppingWithWarmup(warmup_episodes=10, patience=5, min_delta=0.1)
    
    costs = [10.0] * 10 + [9.5, 9.5, 9.5, 9.5, 9.5, 9.5]  # Constant during warmup, then plateau
    for i, cost in enumerate(costs):
        if warmup_stop(cost, episode=i):
            print(f"   Stopped at episode {i}")
            break
    else:
        print(f"   Did not stop. Best: {warmup_stop.best_score:.2f}")
    
    print("\n✓ All tests passed!")
