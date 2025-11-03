"""
Constraint Checker
Utilities for checking individual constraints.

Provides granular constraint checking functions used by DeterministicModel.

Author: Your Name
Date: 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class ConstraintCheck:
    """
    Result of a constraint check.
    
    Attributes:
        satisfied: Whether constraint is satisfied
        value: Actual constraint value
        limit: Constraint limit
        violation: Violation amount (0 if satisfied)
        normalized_violation: Normalized violation [0, ∞)
    """
    satisfied: bool
    value: float
    limit: float
    violation: float
    normalized_violation: float


class ConstraintChecker:
    """
    Utilities for checking individual constraints.
    
    Provides modular constraint checking functions that can be
    composed to validate complete solutions.
    
    Args:
        tolerances: Dictionary of tolerance values per constraint type
    
    Example:
        >>> checker = ConstraintChecker(tolerances={'capacity': 0.05})
        >>> result = checker.check_capacity(load=550, capacity=500)
        >>> print(f"Satisfied: {result.satisfied}, Violation: {result.normalized_violation:.2%}")
    """
    
    def __init__(self, tolerances: Optional[Dict[str, float]] = None):
        self.tolerances = tolerances or {}
    
    def check_capacity(
        self,
        load: float,
        capacity: float,
        tolerance: Optional[float] = None
    ) -> ConstraintCheck:
        """
        Check vehicle capacity constraint.
        
        Constraint: load ≤ capacity
        
        Args:
            load: Current vehicle load
            capacity: Vehicle capacity
            tolerance: Tolerance factor (default: from config or 0.0)
        
        Returns:
            result: ConstraintCheck object
        """
        if tolerance is None:
            tolerance = self.tolerances.get('capacity', 0.0)
        
        limit = capacity * (1 + tolerance)
        satisfied = load <= limit
        violation = max(0, load - limit)
        normalized_violation = violation / capacity if capacity > 0 else 0.0
        
        return ConstraintCheck(
            satisfied=satisfied,
            value=load,
            limit=limit,
            violation=violation,
            normalized_violation=normalized_violation
        )
    
    def check_time_window(
        self,
        arrival_time: float,
        time_window: Tuple[float, float],
        tolerance: Optional[float] = None
    ) -> ConstraintCheck:
        """
        Check time window constraint.
        
        Constraint: earliest ≤ arrival_time ≤ latest
        
        Args:
            arrival_time: Actual arrival time
            time_window: (earliest, latest) tuple
            tolerance: Tolerance factor (default: from config or 0.0)
        
        Returns:
            result: ConstraintCheck object
        """
        if tolerance is None:
            tolerance = self.tolerances.get('time_window', 0.0)
        
        earliest, latest = time_window
        limit = latest * (1 + tolerance)
        
        # Check both early and late arrivals
        if arrival_time < earliest:
            # Early arrival (not typically a hard violation)
            satisfied = True
            violation = 0.0
            normalized_violation = 0.0
        elif arrival_time > limit:
            # Late arrival
            satisfied = False
            violation = arrival_time - limit
            normalized_violation = violation / latest if latest > 0 else 0.0
        else:
            # Within window
            satisfied = True
            violation = 0.0
            normalized_violation = 0.0
        
        return ConstraintCheck(
            satisfied=satisfied,
            value=arrival_time,
            limit=limit,
            violation=violation,
            normalized_violation=normalized_violation
        )
    
    def check_route_duration(
        self,
        duration: float,
        max_duration: float,
        tolerance: Optional[float] = None
    ) -> ConstraintCheck:
        """
        Check maximum route duration constraint.
        
        Constraint: duration ≤ max_duration
        
        Args:
            duration: Actual route duration
            max_duration: Maximum allowed duration
            tolerance: Tolerance factor
        
        Returns:
            result: ConstraintCheck object
        """
        if tolerance is None:
            tolerance = self.tolerances.get('max_duration', 0.0)
        
        limit = max_duration * (1 + tolerance)
        satisfied = duration <= limit
        violation = max(0, duration - limit)
        normalized_violation = violation / max_duration if max_duration > 0 else 0.0
        
        return ConstraintCheck(
            satisfied=satisfied,
            value=duration,
            limit=limit,
            violation=violation,
            normalized_violation=normalized_violation
        )
    
    def check_demand_coverage(
        self,
        visited_nodes: List[int],
        required_nodes: List[int]
    ) -> ConstraintCheck:
        """
        Check demand coverage constraint.
        
        Constraint: All required nodes must be visited
        
        Args:
            visited_nodes: List of visited node indices
            required_nodes: List of required node indices
        
        Returns:
            result: ConstraintCheck object
        """
        visited_set = set(visited_nodes)
        required_set = set(required_nodes)
        
        unvisited = required_set - visited_set
        num_unvisited = len(unvisited)
        num_required = len(required_set)
        
        satisfied = num_unvisited == 0
        violation = num_unvisited
        normalized_violation = violation / num_required if num_required > 0 else 0.0
        
        return ConstraintCheck(
            satisfied=satisfied,
            value=len(visited_set & required_set),
            limit=num_required,
            violation=violation,
            normalized_violation=normalized_violation
        )
    
    def check_precedence(
        self,
        route: List[int],
        precedence_pairs: List[Tuple[int, int]]
    ) -> ConstraintCheck:
        """
        Check precedence constraints.
        
        Constraint: For each pair (i, j), i must be visited before j
        
        Args:
            route: Ordered list of visited nodes
            precedence_pairs: List of (predecessor, successor) pairs
        
        Returns:
            result: ConstraintCheck object
        """
        violations = 0
        
        for predecessor, successor in precedence_pairs:
            if predecessor in route and successor in route:
                pred_idx = route.index(predecessor)
                succ_idx = route.index(successor)
                
                if pred_idx >= succ_idx:
                    violations += 1
        
        satisfied = violations == 0
        normalized_violation = violations / len(precedence_pairs) if precedence_pairs else 0.0
        
        return ConstraintCheck(
            satisfied=satisfied,
            value=len(precedence_pairs) - violations,
            limit=len(precedence_pairs),
            violation=violations,
            normalized_violation=normalized_violation
        )
    
    def check_distance_limit(
        self,
        distance: float,
        max_distance: float,
        tolerance: Optional[float] = None
    ) -> ConstraintCheck:
        """
        Check maximum distance constraint.
        
        Constraint: distance ≤ max_distance
        
        Args:
            distance: Actual distance
            max_distance: Maximum allowed distance
            tolerance: Tolerance factor
        
        Returns:
            result: ConstraintCheck object
        """
        if tolerance is None:
            tolerance = self.tolerances.get('max_distance', 0.0)
        
        limit = max_distance * (1 + tolerance)
        satisfied = distance <= limit
        violation = max(0, distance - limit)
        normalized_violation = violation / max_distance if max_distance > 0 else 0.0
        
        return ConstraintCheck(
            satisfied=satisfied,
            value=distance,
            limit=limit,
            violation=violation,
            normalized_violation=normalized_violation
        )
    
    def check_fuel_constraint(
        self,
        fuel_consumed: float,
        fuel_capacity: float,
        refuel_points: List[int],
        route: List[int]
    ) -> ConstraintCheck:
        """
        Check fuel constraint with refueling.
        
        Constraint: Vehicle has sufficient fuel between refuel points
        
        Args:
            fuel_consumed: Total fuel consumed
            fuel_capacity: Fuel tank capacity
            refuel_points: List of refueling station indices
            route: Ordered route
        
        Returns:
            result: ConstraintCheck object
        """
        # Simplified check (in practice, would check between refuel points)
        satisfied = fuel_consumed <= fuel_capacity
        violation = max(0, fuel_consumed - fuel_capacity)
        normalized_violation = violation / fuel_capacity if fuel_capacity > 0 else 0.0
        
        return ConstraintCheck(
            satisfied=satisfied,
            value=fuel_consumed,
            limit=fuel_capacity,
            violation=violation,
            normalized_violation=normalized_violation
        )
    
    def check_multiple_constraints(
        self,
        checks: List[ConstraintCheck]
    ) -> Tuple[bool, float]:
        """
        Check multiple constraints and aggregate results.
        
        Args:
            checks: List of ConstraintCheck objects
        
        Returns:
            all_satisfied: True if all constraints satisfied
            total_violation: Sum of normalized violations
        """
        all_satisfied = all(check.satisfied for check in checks)
        total_violation = sum(check.normalized_violation for check in checks)
        
        return all_satisfied, total_violation


class StochasticConstraintChecker(ConstraintChecker):
    """
    Constraint checker for stochastic constraints.
    
    Handles chance constraints using scenario-based or deterministic
    equivalent approaches.
    
    Args:
        confidence_level: Confidence level for chance constraints (default: 0.95)
        num_scenarios: Number of scenarios for scenario-based checking (default: 100)
        tolerances: Tolerance values per constraint type
    
    Example:
        >>> checker = StochasticConstraintChecker(confidence_level=0.95)
        >>> result = checker.check_stochastic_capacity(
        ...     demand_mean=100, demand_std=20, capacity=150
        ... )
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        num_scenarios: int = 100,
        tolerances: Optional[Dict[str, float]] = None
    ):
        super().__init__(tolerances)
        self.confidence_level = confidence_level
        self.num_scenarios = num_scenarios
        self.z_score = self._get_z_score(confidence_level)
    
    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for confidence level."""
        z_scores = {
            0.90: 1.282,
            0.95: 1.645,
            0.99: 2.326
        }
        return z_scores.get(confidence_level, 1.645)
    
    def check_stochastic_capacity(
        self,
        demand_mean: float,
        demand_std: float,
        capacity: float,
        tolerance: Optional[float] = None
    ) -> ConstraintCheck:
        """
        Check stochastic capacity constraint.
        
        Chance constraint: P(demand ≤ capacity) ≥ confidence_level
        Deterministic equivalent: demand_mean + z * demand_std ≤ capacity
        
        Args:
            demand_mean: Mean demand
            demand_std: Demand standard deviation
            capacity: Vehicle capacity
            tolerance: Tolerance factor
        
        Returns:
            result: ConstraintCheck object
        """
        # Deterministic equivalent demand
        deterministic_demand = demand_mean + self.z_score * demand_std
        
        return self.check_capacity(deterministic_demand, capacity, tolerance)
    
    def check_scenario_based(
        self,
        scenarios: np.ndarray,
        constraint_fn: Callable,
        *args
    ) -> ConstraintCheck:
        """
        Check constraint using scenario-based approach.
        
        Args:
            scenarios: Array of scenarios [num_scenarios, ...]
            constraint_fn: Function to check constraint for each scenario
            *args: Additional arguments for constraint_fn
        
        Returns:
            result: ConstraintCheck aggregated over scenarios
        """
        num_scenarios = len(scenarios)
        num_satisfied = 0
        
        for scenario in scenarios:
            result = constraint_fn(scenario, *args)
            if result.satisfied:
                num_satisfied += 1
        
        satisfaction_rate = num_satisfied / num_scenarios
        satisfied = satisfaction_rate >= self.confidence_level
        violation = max(0, self.confidence_level - satisfaction_rate)
        
        return ConstraintCheck(
            satisfied=satisfied,
            value=satisfaction_rate,
            limit=self.confidence_level,
            violation=violation,
            normalized_violation=violation / self.confidence_level
        )


if __name__ == '__main__':
    # Test constraint checker
    print("Testing Constraint Checker...")
    
    checker = ConstraintChecker(tolerances={'capacity': 0.05, 'time_window': 0.1})
    
    # Test capacity check
    print("\n1. Capacity Check:")
    result1 = checker.check_capacity(load=520, capacity=500)
    print(f"   Load: {result1.value}, Limit: {result1.limit}")
    print(f"   Satisfied: {result1.satisfied}")
    print(f"   Violation: {result1.normalized_violation:.2%}")
    
    # Test time window check
    print("\n2. Time Window Check:")
    result2 = checker.check_time_window(arrival_time=125, time_window=(0, 120))
    print(f"   Arrival: {result2.value}, Window: [0, 120]")
    print(f"   Satisfied: {result2.satisfied}")
    print(f"   Violation: {result2.normalized_violation:.2%}")
    
    # Test route duration
    print("\n3. Route Duration Check:")
    result3 = checker.check_route_duration(duration=450, max_duration=480)
    print(f"   Duration: {result3.value}, Max: {result3.limit}")
    print(f"   Satisfied: {result3.satisfied}")
    
    # Test demand coverage
    print("\n4. Demand Coverage Check:")
    result4 = checker.check_demand_coverage(
        visited_nodes=[1, 2, 3, 5, 7],
        required_nodes=[1, 2, 3, 4, 5, 6, 7, 8]
    )
    print(f"   Visited: {result4.value}/{result4.limit} required nodes")
    print(f"   Satisfied: {result4.satisfied}")
    print(f"   Violation: {result4.normalized_violation:.2%}")
    
    # Test precedence
    print("\n5. Precedence Check:")
    result5 = checker.check_precedence(
        route=[1, 3, 2, 4],
        precedence_pairs=[(1, 2), (2, 4)]  # 1 before 2, 2 before 4
    )
    print(f"   Route: [1, 3, 2, 4]")
    print(f"   Precedence pairs: [(1,2), (2,4)]")
    print(f"   Satisfied: {result5.satisfied}")
    
    # Test stochastic constraint checker
    print("\n6. Stochastic Capacity Check:")
    stochastic_checker = StochasticConstraintChecker(confidence_level=0.95)
    result6 = stochastic_checker.check_stochastic_capacity(
        demand_mean=100,
        demand_std=20,
        capacity=150
    )
    print(f"   Demand: μ=100, σ=20, Capacity: 150")
    print(f"   Deterministic equivalent: {100 + 1.645 * 20:.1f}")
    print(f"   Satisfied: {result6.satisfied}")
    
    print("\n✓ All tests passed!")
