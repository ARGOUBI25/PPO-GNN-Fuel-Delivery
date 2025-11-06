"""
Violation Analysis
Compute total violation score V_total for three-tier validation.

Section 4.3, Equation 11: V_total computation.

Author: Majdi Argoubi
Date: 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .deterministic_model import ConstraintViolation, ConstraintType


@dataclass
class ViolationScore:
    """
    Complete violation score breakdown.
    
    Attributes:
        V_total: Total violation score (Equation 11)
        violations_by_type: Violation scores per constraint type
        num_violations: Total number of violations
        max_violation: Maximum single violation
        tier: Recommended tier (1, 2, or 3)
    """
    V_total: float
    violations_by_type: Dict[str, float]
    num_violations: int
    max_violation: float
    tier: int


def compute_violation_score(
    solution: Dict,
    constraints: Dict,
    weights: Dict,
    violations: List[ConstraintViolation]
) -> ViolationScore:
    """
    Compute total violation score V_total (Equation 11).
    
    Equation 11:
        V_total = Σ_{c∈C} w_c · max(0, (actual_c - limit_c) / limit_c)
    
    where:
        - C: set of constraint types
        - w_c: importance weight for constraint c
        - actual_c: actual value (e.g., load, time)
        - limit_c: constraint limit (e.g., capacity, time window)
    
    Args:
        solution: Solution to evaluate
        constraints: Constraint limits
        weights: Importance weights w_c per constraint type
        violations: List of ConstraintViolation objects
    
    Returns:
        violation_score: ViolationScore object with breakdown
    
    Example:
        >>> V_score = compute_violation_score(solution, constraints, weights, violations)
        >>> print(f"Total Violations: {V_score.V_total:.2%}")
        >>> print(f"Tier: {V_score.tier}")
    """
    # Initialize violation scores by type
    violations_by_type = {}
    
    for constraint_type in ConstraintType:
        violations_by_type[constraint_type.value] = 0.0
    
    # Compute weighted violation scores
    for violation in violations:
        constraint_name = violation.constraint_type.value
        weight = weights.get(constraint_name, 1.0)
        
        # Weighted severity
        weighted_severity = weight * violation.severity
        violations_by_type[constraint_name] += weighted_severity
    
    # Total violation score
    V_total = sum(violations_by_type.values())
    
    # Determine tier based on V_total
    tier = determine_tier(V_total)
    
    # Maximum single violation
    max_violation = max([v.severity for v in violations]) if violations else 0.0
    
    return ViolationScore(
        V_total=V_total,
        violations_by_type=violations_by_type,
        num_violations=len(violations),
        max_violation=max_violation,
        tier=tier
    )


def determine_tier(V_total: float) -> int:
    """
    Determine tier based on total violation score.
    
    Tier thresholds (Section 4.3):
    - Tier 1: V_total ≤ 0.05 (5%)
    - Tier 2: 0.05 < V_total ≤ 0.25 (5-25%)
    - Tier 3: V_total > 0.25 (>25%)
    
    Args:
        V_total: Total violation score
    
    Returns:
        tier: 1, 2, or 3
    """
    if V_total <= 0.05:
        return 1
    elif V_total <= 0.25:
        return 2
    else:
        return 3


def compute_constraint_importance(
    violations: List[ConstraintViolation],
    historical_violations: Optional[List[List[ConstraintViolation]]] = None
) -> Dict[str, float]:
    """
    Compute constraint importance based on violation frequency and severity.
    
    Used to adaptively adjust constraint weights over time.
    
    Args:
        violations: Current violations
        historical_violations: List of violation lists from previous episodes
    
    Returns:
        importance: Dictionary of importance scores per constraint type
    """
    importance = {}
    
    # Count frequency and severity
    frequency = {}
    severity_sum = {}
    
    for constraint_type in ConstraintType:
        frequency[constraint_type.value] = 0
        severity_sum[constraint_type.value] = 0.0
    
    # Current violations
    for violation in violations:
        constraint_name = violation.constraint_type.value
        frequency[constraint_name] += 1
        severity_sum[constraint_name] += violation.severity
    
    # Historical violations (if available)
    if historical_violations:
        for past_violations in historical_violations:
            for violation in past_violations:
                constraint_name = violation.constraint_type.value
                frequency[constraint_name] += 1
                severity_sum[constraint_name] += violation.severity
    
    # Compute importance (frequency × average severity)
    total_violations = len(violations) + (
        sum(len(v) for v in historical_violations) if historical_violations else 0
    )
    
    for constraint_name in frequency:
        if frequency[constraint_name] > 0:
            avg_severity = severity_sum[constraint_name] / frequency[constraint_name]
            freq_ratio = frequency[constraint_name] / max(total_violations, 1)
            importance[constraint_name] = freq_ratio * avg_severity
        else:
            importance[constraint_name] = 0.0
    
    # Normalize to [0, 1]
    max_importance = max(importance.values()) if importance else 1.0
    if max_importance > 0:
        importance = {k: v / max_importance for k, v in importance.items()}
    
    return importance


def analyze_violation_trends(
    violation_history: List[ViolationScore],
    window_size: int = 10
) -> Dict:
    """
    Analyze trends in violation scores over time.
    
    Args:
        violation_history: List of ViolationScore objects over time
        window_size: Window size for moving average
    
    Returns:
        trends: Dictionary with trend statistics
            - 'improving': True if violations decreasing
            - 'stable': True if violations stable
            - 'deteriorating': True if violations increasing
            - 'moving_avg': Moving average of V_total
            - 'trend_slope': Slope of linear regression
    """
    if len(violation_history) < 2:
        return {
            'improving': False,
            'stable': True,
            'deteriorating': False,
            'moving_avg': [],
            'trend_slope': 0.0
        }
    
    # Extract V_total values
    V_totals = np.array([score.V_total for score in violation_history])
    
    # Compute moving average
    if len(V_totals) >= window_size:
        moving_avg = np.convolve(V_totals, np.ones(window_size)/window_size, mode='valid')
    else:
        moving_avg = V_totals
    
    # Linear regression for trend
    x = np.arange(len(V_totals))
    if len(x) > 1:
        coeffs = np.polyfit(x, V_totals, 1)
        trend_slope = coeffs[0]
    else:
        trend_slope = 0.0
    
    # Determine trend
    improving = trend_slope < -0.01  # Decreasing violations
    deteriorating = trend_slope > 0.01  # Increasing violations
    stable = not (improving or deteriorating)
    
    return {
        'improving': improving,
        'stable': stable,
        'deteriorating': deteriorating,
        'moving_avg': moving_avg.tolist(),
        'trend_slope': trend_slope
    }


def compute_normalized_violation(
    actual: float,
    limit: float,
    tolerance: float = 0.0
) -> float:
    """
    Compute normalized violation for a single constraint.
    
    Violation = max(0, (actual - limit * (1 + tolerance)) / limit)
    
    Args:
        actual: Actual value
        limit: Constraint limit
        tolerance: Tolerance factor (e.g., 0.05 for 5%)
    
    Returns:
        violation: Normalized violation [0, ∞)
    """
    if limit <= 0:
        return 0.0
    
    threshold = limit * (1 + tolerance)
    
    if actual <= threshold:
        return 0.0
    else:
        return (actual - threshold) / limit


def compute_violation_gradient(
    solution: Dict,
    constraints: Dict,
    perturbation: float = 1e-5
) -> Dict:
    """
    Compute gradient of violation score with respect to solution variables.
    
    Useful for gradient-based constraint repair.
    
    Args:
        solution: Current solution
        constraints: Constraint limits
        perturbation: Finite difference step size
    
    Returns:
        gradient: Dictionary of gradients per solution variable
    """
    # This is a placeholder for gradient computation
    # In practice, would compute numerical or analytical gradients
    
    gradient = {}
    
    # Example: gradient with respect to route assignments
    routes = solution.get('routes', [])
    
    for route_idx, route in enumerate(routes):
        gradient[f'route_{route_idx}_load'] = 0.0
        gradient[f'route_{route_idx}_time'] = 0.0
    
    return gradient


class ViolationTracker:
    """
    Tracks violation history for adaptive penalty adjustment.
    
    Args:
        max_history: Maximum number of episodes to track
    
    Example:
        >>> tracker = ViolationTracker(max_history=100)
        >>> tracker.record(violation_score)
        >>> trends = tracker.get_trends()
        >>> importance = tracker.get_constraint_importance()
    """
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.history = []
        self.violation_history = []
    
    def record(
        self,
        violation_score: ViolationScore,
        violations: List[ConstraintViolation]
    ):
        """Record violation score and violations."""
        self.history.append(violation_score)
        self.violation_history.append(violations)
        
        # Maintain max history
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.violation_history.pop(0)
    
    def get_trends(self, window_size: int = 10) -> Dict:
        """Get violation trends."""
        return analyze_violation_trends(self.history, window_size)
    
    def get_constraint_importance(self) -> Dict[str, float]:
        """Get constraint importance based on history."""
        if not self.violation_history:
            return {}
        
        current_violations = self.violation_history[-1]
        historical_violations = self.violation_history[:-1] if len(self.violation_history) > 1 else None
        
        return compute_constraint_importance(current_violations, historical_violations)
    
    def get_recent_average(self, n: int = 10) -> float:
        """Get average V_total over last n episodes."""
        if not self.history:
            return 0.0
        
        recent = self.history[-n:]
        return np.mean([score.V_total for score in recent])
    
    def get_violation_rate(self, constraint_type: str, n: int = 10) -> float:
        """Get violation rate for specific constraint type over last n episodes."""
        if not self.violation_history:
            return 0.0
        
        recent = self.violation_history[-n:]
        
        total_episodes = len(recent)
        episodes_with_violation = sum(
            1 for violations in recent
            if any(v.constraint_type.value == constraint_type for v in violations)
        )
        
        return episodes_with_violation / total_episodes if total_episodes > 0 else 0.0


if __name__ == '__main__':
    # Test violation analysis
    from .deterministic_model import ConstraintViolation, ConstraintType
    
    print("Testing Violation Analysis...")
    
    # Create dummy violations
    violations = [
        ConstraintViolation(
            constraint_type=ConstraintType.CAPACITY,
            severity=0.15,
            vehicle=0,
            description="Capacity exceeded"
        ),
        ConstraintViolation(
            constraint_type=ConstraintType.TIME_WINDOW,
            severity=0.08,
            node=5,
            vehicle=1,
            description="Late arrival"
        ),
        ConstraintViolation(
            constraint_type=ConstraintType.CAPACITY,
            severity=0.05,
            vehicle=2,
            description="Capacity exceeded"
        )
    ]
    
    # Constraint weights (from config)
    weights = {
        'capacity': 2.0,
        'time_window': 1.5,
        'demand': 1.8,
        'sequencing': 1.0,
        'station_limit': 1.2,
        'max_route_time': 1.0
    }
    
    # Compute violation score
    score = compute_violation_score({}, {}, weights, violations)
    
    print(f"✓ Total violation score: {score.V_total:.4f}")
    print(f"  Number of violations: {score.num_violations}")
    print(f"  Max violation: {score.max_violation:.2%}")
    print(f"  Recommended tier: {score.tier}")
    print(f"\n  Violations by type:")
    for constraint_type, value in score.violations_by_type.items():
        if value > 0:
            print(f"    {constraint_type}: {value:.4f}")
    
    # Test violation tracker
    print("\nTesting Violation Tracker...")
    tracker = ViolationTracker(max_history=20)
    
    # Simulate multiple episodes
    for i in range(15):
        # Decreasing violations over time
        dummy_violations = [
            ConstraintViolation(
                constraint_type=ConstraintType.CAPACITY,
                severity=0.2 / (i + 1),
                vehicle=0
            )
        ]
        
        dummy_score = ViolationScore(
            V_total=0.2 / (i + 1),
            violations_by_type={'capacity': 0.2 / (i + 1)},
            num_violations=1,
            max_violation=0.2 / (i + 1),
            tier=1
        )
        
        tracker.record(dummy_score, dummy_violations)
    
    trends = tracker.get_trends()
    print(f"✓ Trend analysis:")
    print(f"  Improving: {trends['improving']}")
    print(f"  Stable: {trends['stable']}")
    print(f"  Deteriorating: {trends['deteriorating']}")
    print(f"  Trend slope: {trends['trend_slope']:.6f}")
    
    recent_avg = tracker.get_recent_average(n=5)
    print(f"  Recent average (last 5): {recent_avg:.4f}")
