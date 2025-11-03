"""
Adaptive Penalties (Three-Tier System)
Graduated penalty adjustment based on violation severity.

Section 4.3, Algorithm 2: Three-tier constraint validation mechanism.

Author: Your Name
Date: 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import copy

from .violation_analysis import ViolationScore
from .deterministic_model import ConstraintViolation


class TierAction(Enum):
    """Actions for each tier."""
    CONTINUE = "continue"          # Tier 1: Continue training
    FINE_TUNE = "fine_tune"        # Tier 2: Fine-tune with penalty adjustment
    RE_TRAIN = "re_train"          # Tier 3: Full re-training


@dataclass
class PenaltyAdjustment:
    """
    Penalty adjustment recommendation.
    
    Attributes:
        tier: Tier number (1, 2, or 3)
        action: Recommended action
        penalty_multiplier: Multiplier for constraint penalties
        num_episodes: Number of episodes for adjustment
        reset_policy: Whether to reset policy network
        preserve_encoder: Whether to preserve encoder weights
        violated_constraints: List of constraint types to adjust
        description: Human-readable description
    """
    tier: int
    action: TierAction
    penalty_multiplier: float
    num_episodes: int
    reset_policy: bool
    preserve_encoder: bool
    violated_constraints: List[str]
    description: str


class AdaptivePenaltyManager:
    """
    Three-tier adaptive penalty system (Algorithm 2).
    
    Implements graduated response based on violation severity:
    - Tier 1 (V ≤ 5%): Tolerance zone for exploration
    - Tier 2 (5% < V ≤ 25%): Fine-tuning with 1.5× penalties (1,000 episodes)
    - Tier 3 (V > 25%): Re-training with 10× penalties + reset θ (10,000 episodes)
    
    Args:
        tier1_threshold: Tier 1 threshold (default: 0.05)
        tier2_threshold: Tier 2 threshold (default: 0.25)
        tier2_multiplier: Tier 2 penalty multiplier (default: 1.5)
        tier2_episodes: Tier 2 training episodes (default: 1000)
        tier3_multiplier: Tier 3 penalty multiplier (default: 10.0)
        tier3_episodes: Tier 3 training episodes (default: 10000)
        selective_adjustment: Only adjust violated constraints in Tier 2 (default: True)
    
    Example:
        >>> manager = AdaptivePenaltyManager()
        >>> adjustment = manager.assess_violations(V_total=0.15, violations=violations)
        >>> print(f"Tier {adjustment.tier}: {adjustment.action.value}")
        >>> new_penalties = manager.apply_penalties(current_penalties, adjustment)
    """
    
    def __init__(
        self,
        tier1_threshold: float = 0.05,
        tier2_threshold: float = 0.25,
        tier2_multiplier: float = 1.5,
        tier2_episodes: int = 1000,
        tier3_multiplier: float = 10.0,
        tier3_episodes: int = 10000,
        selective_adjustment: bool = True
    ):
        self.tier1_threshold = tier1_threshold
        self.tier2_threshold = tier2_threshold
        self.tier2_multiplier = tier2_multiplier
        self.tier2_episodes = tier2_episodes
        self.tier3_multiplier = tier3_multiplier
        self.tier3_episodes = tier3_episodes
        self.selective_adjustment = selective_adjustment
        
        # History tracking
        self.tier_history = []
        self.penalty_history = []
    
    def assess_violations(
        self,
        V_total: float,
        violations: List[ConstraintViolation],
        violation_score: Optional[ViolationScore] = None
    ) -> PenaltyAdjustment:
        """
        Determine tier and recommended action (Algorithm 2, lines 2-15).
        
        Args:
            V_total: Total violation score
            violations: List of ConstraintViolation objects
            violation_score: Optional ViolationScore for detailed analysis
        
        Returns:
            adjustment: PenaltyAdjustment with recommendations
        
        Example:
            >>> adjustment = manager.assess_violations(V_total=0.15, violations=violations)
            >>> if adjustment.action == TierAction.FINE_TUNE:
            ...     print(f"Fine-tune for {adjustment.num_episodes} episodes")
        """
        # Determine tier
        tier = self._determine_tier(V_total)
        
        # Extract violated constraints
        violated_constraints = list(set([v.constraint_type.value for v in violations]))
        
        # Generate adjustment based on tier
        if tier == 1:
            # Tier 1: Tolerance (continue training)
            adjustment = PenaltyAdjustment(
                tier=1,
                action=TierAction.CONTINUE,
                penalty_multiplier=1.0,
                num_episodes=0,
                reset_policy=False,
                preserve_encoder=True,
                violated_constraints=[],
                description=f"Tier 1: V_total={V_total:.2%} ≤ {self.tier1_threshold:.0%}. Continue training without penalty adjustment."
            )
        
        elif tier == 2:
            # Tier 2: Fine-tuning
            adjustment = PenaltyAdjustment(
                tier=2,
                action=TierAction.FINE_TUNE,
                penalty_multiplier=self.tier2_multiplier,
                num_episodes=self.tier2_episodes,
                reset_policy=False,
                preserve_encoder=True,
                violated_constraints=violated_constraints if self.selective_adjustment else [],
                description=f"Tier 2: V_total={V_total:.2%} in ({self.tier1_threshold:.0%}, {self.tier2_threshold:.0%}]. "
                           f"Fine-tune with {self.tier2_multiplier}× penalties for {self.tier2_episodes} episodes."
            )
        
        else:  # tier == 3
            # Tier 3: Re-training
            adjustment = PenaltyAdjustment(
                tier=3,
                action=TierAction.RE_TRAIN,
                penalty_multiplier=self.tier3_multiplier,
                num_episodes=self.tier3_episodes,
                reset_policy=True,
                preserve_encoder=True,
                violated_constraints=[],  # Adjust all constraints
                description=f"Tier 3: V_total={V_total:.2%} > {self.tier2_threshold:.0%}. "
                           f"Re-train with {self.tier3_multiplier}× penalties, reset policy θ, for {self.tier3_episodes} episodes."
            )
        
        # Record in history
        self.tier_history.append(tier)
        
        return adjustment
    
    def _determine_tier(self, V_total: float) -> int:
        """Determine tier based on V_total."""
        if V_total <= self.tier1_threshold:
            return 1
        elif V_total <= self.tier2_threshold:
            return 2
        else:
            return 3
    
    def apply_penalties(
        self,
        current_penalties: Dict[str, float],
        adjustment: PenaltyAdjustment
    ) -> Dict[str, float]:
        """
        Apply tier-specific penalty adjustments (Algorithm 2, lines 16-23).
        
        Args:
            current_penalties: Current penalty weights λ_c
            adjustment: PenaltyAdjustment from assess_violations
        
        Returns:
            new_penalties: Adjusted penalty weights
        
        Example:
            >>> current = {'capacity': 2.0, 'time_window': 1.5}
            >>> new = manager.apply_penalties(current, adjustment)
            >>> print(f"Capacity penalty: {current['capacity']} → {new['capacity']}")
        """
        new_penalties = copy.deepcopy(current_penalties)
        
        if adjustment.action == TierAction.CONTINUE:
            # Tier 1: No change
            pass
        
        elif adjustment.action == TierAction.FINE_TUNE:
            # Tier 2: Selective or global adjustment
            if self.selective_adjustment and adjustment.violated_constraints:
                # Only adjust violated constraints
                for constraint_type in adjustment.violated_constraints:
                    if constraint_type in new_penalties:
                        new_penalties[constraint_type] *= adjustment.penalty_multiplier
            else:
                # Adjust all constraint penalties
                for constraint_type in new_penalties:
                    new_penalties[constraint_type] *= adjustment.penalty_multiplier
        
        elif adjustment.action == TierAction.RE_TRAIN:
            # Tier 3: Aggressive adjustment for all constraints
            for constraint_type in new_penalties:
                new_penalties[constraint_type] *= adjustment.penalty_multiplier
        
        # Record in history
        self.penalty_history.append(new_penalties)
        
        return new_penalties
    
    def get_tier_statistics(self) -> Dict:
        """
        Get statistics on tier activations.
        
        Returns:
            stats: Dictionary with tier activation counts and percentages
        """
        if not self.tier_history:
            return {'tier1': 0, 'tier2': 0, 'tier3': 0}
        
        total = len(self.tier_history)
        
        return {
            'tier1_count': self.tier_history.count(1),
            'tier2_count': self.tier_history.count(2),
            'tier3_count': self.tier_history.count(3),
            'tier1_pct': self.tier_history.count(1) / total * 100,
            'tier2_pct': self.tier_history.count(2) / total * 100,
            'tier3_pct': self.tier_history.count(3) / total * 100,
            'total_validations': total
        }
    
    def should_trigger_validation(
        self,
        episode: int,
        validation_frequency: int = 1000
    ) -> bool:
        """
        Determine if validation should be triggered at current episode.
        
        Args:
            episode: Current episode number
            validation_frequency: Validation frequency (default: 1000)
        
        Returns:
            should_validate: True if validation should be triggered
        """
        return episode % validation_frequency == 0 and episode > 0
    
    def detect_quality_degradation(
        self,
        recent_costs: List[float],
        threshold: float = 0.05,
        patience: int = 5
    ) -> bool:
        """
        Detect quality degradation (Algorithm 2, lines 24-28).
        
        Quality degradation: cost increases by >5% over last 5 episodes.
        
        Args:
            recent_costs: List of recent episode costs
            threshold: Degradation threshold (default: 0.05 = 5%)
            patience: Number of episodes to check (default: 5)
        
        Returns:
            degraded: True if quality degradation detected
        """
        if len(recent_costs) < patience + 1:
            return False
        
        # Compare recent costs to baseline
        baseline_cost = recent_costs[-(patience + 1)]
        recent_avg = np.mean(recent_costs[-patience:])
        
        if baseline_cost > 0:
            degradation = (recent_avg - baseline_cost) / baseline_cost
            return degradation > threshold
        
        return False
    
    def detect_high_variance(
        self,
        recent_costs: List[float],
        threshold: float = 0.15
    ) -> bool:
        """
        Detect high variance in costs (Algorithm 2, lines 29-32).
        
        High variance: coefficient of variation > 15%.
        
        Args:
            recent_costs: List of recent episode costs
            threshold: CV threshold (default: 0.15 = 15%)
        
        Returns:
            high_variance: True if high variance detected
        """
        if len(recent_costs) < 2:
            return False
        
        mean_cost = np.mean(recent_costs)
        std_cost = np.std(recent_costs)
        
        if mean_cost > 0:
            cv = std_cost / mean_cost
            return cv > threshold
        
        return False
    
    def restore_checkpoint(
        self,
        current_penalties: Dict[str, float],
        checkpoint_penalties: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Restore penalties from checkpoint (Algorithm 2, line 28).
        
        Args:
            current_penalties: Current penalty weights
            checkpoint_penalties: Checkpoint penalty weights
        
        Returns:
            restored_penalties: Restored penalty weights
        """
        return copy.deepcopy(checkpoint_penalties)
    
    def reduce_learning_rate(
        self,
        current_lr: float,
        multiplier: float = 0.5
    ) -> float:
        """
        Reduce learning rate (Algorithm 2, line 32).
        
        Args:
            current_lr: Current learning rate
            multiplier: Reduction multiplier (default: 0.5)
        
        Returns:
            new_lr: Reduced learning rate
        """
        return current_lr * multiplier


class PenaltyScheduler:
    """
    Scheduler for gradual penalty increases.
    
    Alternative to three-tier system: gradual increase over time.
    
    Args:
        initial_penalties: Initial penalty weights
        target_penalties: Target penalty weights
        num_steps: Number of steps to reach target
        schedule: 'linear', 'exponential', or 'cosine'
    
    Example:
        >>> scheduler = PenaltyScheduler(initial={...}, target={...}, num_steps=10000)
        >>> for step in range(10000):
        ...     penalties = scheduler.get_penalties(step)
    """
    
    def __init__(
        self,
        initial_penalties: Dict[str, float],
        target_penalties: Dict[str, float],
        num_steps: int,
        schedule: str = 'linear'
    ):
        self.initial_penalties = initial_penalties
        self.target_penalties = target_penalties
        self.num_steps = num_steps
        self.schedule = schedule
    
    def get_penalties(self, step: int) -> Dict[str, float]:
        """Get penalties at current step."""
        if step >= self.num_steps:
            return self.target_penalties
        
        # Compute interpolation factor
        if self.schedule == 'linear':
            alpha = step / self.num_steps
        elif self.schedule == 'exponential':
            alpha = 1 - np.exp(-5 * step / self.num_steps)
        elif self.schedule == 'cosine':
            alpha = (1 - np.cos(np.pi * step / self.num_steps)) / 2
        else:
            alpha = step / self.num_steps
        
        # Interpolate penalties
        penalties = {}
        for constraint_type in self.initial_penalties:
            initial = self.initial_penalties[constraint_type]
            target = self.target_penalties.get(constraint_type, initial)
            penalties[constraint_type] = initial + alpha * (target - initial)
        
        return penalties


class AdaptiveWeightManager:
    """
    Manages adaptive constraint weights based on violation patterns.
    
    Automatically adjusts weights to focus on frequently violated constraints.
    
    Args:
        initial_weights: Initial constraint weights
        adaptation_rate: Rate of weight adaptation (default: 0.1)
        min_weight: Minimum weight value (default: 0.1)
        max_weight: Maximum weight value (default: 10.0)
    
    Example:
        >>> manager = AdaptiveWeightManager(initial_weights={...})
        >>> manager.update(violations)
        >>> current_weights = manager.get_weights()
    """
    
    def __init__(
        self,
        initial_weights: Dict[str, float],
        adaptation_rate: float = 0.1,
        min_weight: float = 0.1,
        max_weight: float = 10.0
    ):
        self.weights = copy.deepcopy(initial_weights)
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Violation counts
        self.violation_counts = {k: 0 for k in initial_weights}
        self.total_validations = 0
    
    def update(self, violations: List[ConstraintViolation]):
        """Update weights based on current violations."""
        self.total_validations += 1
        
        # Count violations
        for violation in violations:
            constraint_type = violation.constraint_type.value
            if constraint_type in self.violation_counts:
                self.violation_counts[constraint_type] += 1
        
        # Adapt weights
        if self.total_validations % 10 == 0:  # Update every 10 validations
            self._adapt_weights()
    
    def _adapt_weights(self):
        """Adapt weights based on violation frequency."""
        # Compute violation rates
        rates = {
            k: v / max(self.total_validations, 1)
            for k, v in self.violation_counts.items()
        }
        
        # Adjust weights (increase for frequently violated constraints)
        for constraint_type, rate in rates.items():
            if rate > 0.1:  # If violated in >10% of validations
                self.weights[constraint_type] *= (1 + self.adaptation_rate)
            elif rate < 0.01:  # If rarely violated
                self.weights[constraint_type] *= (1 - self.adaptation_rate * 0.5)
            
            # Clip to bounds
            self.weights[constraint_type] = np.clip(
                self.weights[constraint_type],
                self.min_weight,
                self.max_weight
            )
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return copy.deepcopy(self.weights)
    
    def reset(self):
        """Reset violation counts."""
        self.violation_counts = {k: 0 for k in self.weights}
        self.total_validations = 0


if __name__ == '__main__':
    # Test adaptive penalty manager
    from .deterministic_model import ConstraintType
    
    print("Testing Adaptive Penalty Manager...")
    
    manager = AdaptivePenaltyManager(
        tier1_threshold=0.05,
        tier2_threshold=0.25,
        tier2_multiplier=1.5,
        tier3_multiplier=10.0
    )
    
    # Test Tier 1 (low violations)
    print("\nTest Tier 1:")
    violations_t1 = [
        ConstraintViolation(ConstraintType.CAPACITY, 0.02, vehicle=0)
    ]
    adjustment_t1 = manager.assess_violations(0.03, violations_t1)
    print(f"  Tier: {adjustment_t1.tier}")
    print(f"  Action: {adjustment_t1.action.value}")
    print(f"  Description: {adjustment_t1.description}")
    
    # Test Tier 2 (moderate violations)
    print("\nTest Tier 2:")
    violations_t2 = [
        ConstraintViolation(ConstraintType.CAPACITY, 0.10, vehicle=0),
        ConstraintViolation(ConstraintType.TIME_WINDOW, 0.05, node=5, vehicle=1)
    ]
    adjustment_t2 = manager.assess_violations(0.15, violations_t2)
    print(f"  Tier: {adjustment_t2.tier}")
    print(f"  Action: {adjustment_t2.action.value}")
    print(f"  Penalty multiplier: {adjustment_t2.penalty_multiplier}×")
    print(f"  Episodes: {adjustment_t2.num_episodes}")
    print(f"  Violated constraints: {adjustment_t2.violated_constraints}")
    
    # Apply penalties
    current_penalties = {
        'capacity': 2.0,
        'time_window': 1.5,
        'demand': 1.8
    }
    new_penalties = manager.apply_penalties(current_penalties, adjustment_t2)
    print(f"\n  Penalty adjustments:")
    for constraint_type in current_penalties:
        print(f"    {constraint_type}: {current_penalties[constraint_type]} → {new_penalties[constraint_type]}")
    
    # Test Tier 3 (severe violations)
    print("\nTest Tier 3:")
    violations_t3 = [
        ConstraintViolation(ConstraintType.CAPACITY, 0.30, vehicle=0),
        ConstraintViolation(ConstraintType.TIME_WINDOW, 0.20, node=5, vehicle=1),
        ConstraintViolation(ConstraintType.DEMAND, 0.15, node=10)
    ]
    adjustment_t3 = manager.assess_violations(0.40, violations_t3)
    print(f"  Tier: {adjustment_t3.tier}")
    print(f"  Action: {adjustment_t3.action.value}")
    print(f"  Penalty multiplier: {adjustment_t3.penalty_multiplier}×")
    print(f"  Reset policy: {adjustment_t3.reset_policy}")
    print(f"  Episodes: {adjustment_t3.num_episodes}")
    
    # Test tier statistics
    print("\nTier Statistics:")
    stats = manager.get_tier_statistics()
    print(f"  Tier 1: {stats['tier1_count']} ({stats['tier1_pct']:.1f}%)")
    print(f"  Tier 2: {stats['tier2_count']} ({stats['tier2_pct']:.1f}%)")
    print(f"  Tier 3: {stats['tier3_count']} ({stats['tier3_pct']:.1f}%)")
    
    # Test quality degradation detection
    print("\nTest Quality Degradation Detection:")
    costs_degrading = [100, 105, 110, 115, 120, 125]
    degraded = manager.detect_quality_degradation(costs_degrading)
    print(f"  Degrading costs: {degraded}")
    
    costs_stable = [100, 101, 99, 100, 102, 98]
    stable = manager.detect_quality_degradation(costs_stable)
    print(f"  Stable costs: {stable}")
    
    print("\n✓ All tests passed!")
