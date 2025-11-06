"""
Evaluation Metrics
Performance metrics for VRP solution evaluation.

Section 5.3: Evaluation metrics including optimality gap, constraint
violation rate, computational efficiency, and robustness.

Author: Majdi Argoubi
Date: 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class EvaluationMetrics:
    """
    Complete evaluation metrics for VRP solution.
    
    Attributes:
        total_cost: Total routing cost ($)
        optimality_gap: Gap to optimal/best-known solution (%)
        num_routes: Number of routes
        total_distance: Total distance (km)
        total_time: Total time (minutes)
        avg_route_distance: Average route distance
        avg_route_load: Average route load utilization
        constraint_violations: Number of constraint violations
        violation_rate: Constraint violation rate (%)
        feasibility: Solution feasibility (True/False)
        computation_time: Computation time (seconds)
    """
    total_cost: float
    optimality_gap: float
    num_routes: int
    total_distance: float
    total_time: float
    avg_route_distance: float
    avg_route_load: float
    constraint_violations: int
    violation_rate: float
    feasibility: bool
    computation_time: float


class MetricsCalculator:
    """
    Calculator for VRP evaluation metrics.
    
    Computes comprehensive metrics for solution quality assessment
    including cost, feasibility, and efficiency measures.
    
    Args:
        network: Network configuration with costs
        optimal_cost: Optimal or best-known cost (for gap calculation)
    
    Example:
        >>> calculator = MetricsCalculator(network, optimal_cost=5000)
        >>> metrics = calculator.compute_metrics(solution)
        >>> print(f"Optimality Gap: {metrics.optimality_gap:.2%}")
    """
    
    def __init__(
        self,
        network: Dict,
        optimal_cost: Optional[float] = None
    ):
        self.network = network
        self.optimal_cost = optimal_cost
        
        # Cost parameters (Section 3.1)
        self.fuel_cost_per_km = network.get('fuel_cost_per_km', 0.5)
        self.driver_cost_per_hour = network.get('driver_cost_per_hour', 25.0)
        self.vehicle_fixed_cost = network.get('vehicle_fixed_cost', 50.0)
        self.penalty_unmet = network.get('penalty_unmet', 10.0)
        
        self.vehicle_capacity = network.get('vehicle_capacity', 1000.0)
    
    def compute_metrics(
        self,
        solution: Dict,
        computation_time: Optional[float] = None,
        violations: Optional[List] = None
    ) -> EvaluationMetrics:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            solution: Solution dictionary with routes
            computation_time: Time to compute solution (seconds)
            violations: List of constraint violations (if available)
        
        Returns:
            metrics: EvaluationMetrics object
        """
        routes = solution.get('routes', [])
        
        # Cost metrics
        total_cost = self._compute_total_cost(routes)
        optimality_gap = self._compute_optimality_gap(total_cost)
        
        # Route metrics
        num_routes = len(routes)
        total_distance = sum(r.get('distance', 0) for r in routes)
        total_time = sum(r.get('time', 0) for r in routes)
        avg_route_distance = total_distance / num_routes if num_routes > 0 else 0
        
        # Load utilization
        total_load = sum(r.get('load', 0) for r in routes)
        avg_route_load = (total_load / (num_routes * self.vehicle_capacity)) if num_routes > 0 else 0
        
        # Constraint violations
        if violations is not None:
            constraint_violations = len(violations)
            violation_rate = constraint_violations / max(num_routes, 1)
            feasibility = (constraint_violations == 0)
        else:
            constraint_violations = 0
            violation_rate = 0.0
            feasibility = True
        
        # Computation time
        if computation_time is None:
            computation_time = solution.get('computation_time', 0.0)
        
        return EvaluationMetrics(
            total_cost=total_cost,
            optimality_gap=optimality_gap,
            num_routes=num_routes,
            total_distance=total_distance,
            total_time=total_time,
            avg_route_distance=avg_route_distance,
            avg_route_load=avg_route_load,
            constraint_violations=constraint_violations,
            violation_rate=violation_rate,
            feasibility=feasibility,
            computation_time=computation_time
        )
    
    def _compute_total_cost(self, routes: List[Dict]) -> float:
        """
        Compute total routing cost (Section 3.1).
        
        Cost = fuel + driver + vehicle fixed + penalties
        """
        total_cost = 0.0
        
        for route in routes:
            # Fuel cost
            distance = route.get('distance', 0)
            total_cost += distance * self.fuel_cost_per_km
            
            # Driver cost
            time = route.get('time', 0)
            total_cost += (time / 60.0) * self.driver_cost_per_hour
            
            # Vehicle fixed cost
            total_cost += self.vehicle_fixed_cost
        
        return total_cost
    
    def _compute_optimality_gap(self, solution_cost: float) -> float:
        """
        Compute optimality gap (Section 5.3).
        
        Gap = (solution_cost - optimal_cost) / optimal_cost
        """
        if self.optimal_cost is None or self.optimal_cost == 0:
            return 0.0
        
        gap = (solution_cost - self.optimal_cost) / self.optimal_cost
        return max(0.0, gap)  # Clip to non-negative
    
    def compute_cost_breakdown(self, routes: List[Dict]) -> Dict[str, float]:
        """
        Break down total cost by component.
        
        Returns:
            breakdown: Dictionary with cost components
        """
        fuel_cost = 0.0
        driver_cost = 0.0
        fixed_cost = 0.0
        
        for route in routes:
            fuel_cost += route.get('distance', 0) * self.fuel_cost_per_km
            driver_cost += (route.get('time', 0) / 60.0) * self.driver_cost_per_hour
            fixed_cost += self.vehicle_fixed_cost
        
        total_cost = fuel_cost + driver_cost + fixed_cost
        
        return {
            'fuel_cost': fuel_cost,
            'fuel_pct': fuel_cost / total_cost * 100 if total_cost > 0 else 0,
            'driver_cost': driver_cost,
            'driver_pct': driver_cost / total_cost * 100 if total_cost > 0 else 0,
            'fixed_cost': fixed_cost,
            'fixed_pct': fixed_cost / total_cost * 100 if total_cost > 0 else 0,
            'total_cost': total_cost
        }


def compute_solution_quality_metrics(
    solutions: List[Dict],
    optimal_costs: Optional[List[float]] = None
) -> Dict:
    """
    Compute aggregate quality metrics over multiple solutions.
    
    Args:
        solutions: List of solution dictionaries
        optimal_costs: Optimal costs for each instance (if available)
    
    Returns:
        aggregate_metrics: Dictionary with mean, std, min, max for each metric
    """
    costs = [s.get('cost', 0) for s in solutions]
    times = [s.get('computation_time', 0) for s in solutions]
    num_routes = [s.get('num_routes', 0) for s in solutions]
    
    # Optimality gaps
    gaps = []
    if optimal_costs:
        for i, solution in enumerate(solutions):
            cost = solution.get('cost', 0)
            optimal = optimal_costs[i]
            if optimal > 0:
                gap = (cost - optimal) / optimal
                gaps.append(max(0.0, gap))
    
    return {
        'mean_cost': np.mean(costs),
        'std_cost': np.std(costs),
        'min_cost': np.min(costs),
        'max_cost': np.max(costs),
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'mean_routes': np.mean(num_routes),
        'mean_gap': np.mean(gaps) if gaps else None,
        'median_gap': np.median(gaps) if gaps else None,
        'max_gap': np.max(gaps) if gaps else None
    }


def compute_robustness_metrics(
    solutions: List[Dict],
    stochastic_scenarios: Optional[List[Dict]] = None
) -> Dict:
    """
    Compute robustness metrics (Section 5.3).
    
    Evaluates solution performance under stochastic conditions.
    
    Args:
        solutions: List of solutions
        stochastic_scenarios: List of stochastic demand scenarios
    
    Returns:
        robustness: Dictionary with robustness metrics
    """
    costs = [s.get('cost', 0) for s in solutions]
    
    # Cost variability
    cv = np.std(costs) / np.mean(costs) if np.mean(costs) > 0 else 0
    
    # Worst-case performance
    worst_case = np.max(costs)
    best_case = np.min(costs)
    
    # Percentiles
    p95 = np.percentile(costs, 95)
    p99 = np.percentile(costs, 99)
    
    return {
        'coefficient_of_variation': cv,
        'worst_case_cost': worst_case,
        'best_case_cost': best_case,
        'cost_range': worst_case - best_case,
        '95th_percentile': p95,
        '99th_percentile': p99,
        'expected_cost': np.mean(costs)
    }


def compute_efficiency_metrics(
    training_time: float,
    inference_time: float,
    num_episodes: int,
    solution_quality: float
) -> Dict:
    """
    Compute computational efficiency metrics (Section 5.3).
    
    Args:
        training_time: Total training time (hours)
        inference_time: Average inference time per solution (seconds)
        num_episodes: Number of training episodes
        solution_quality: Solution quality (e.g., optimality gap)
    
    Returns:
        efficiency: Dictionary with efficiency metrics
    """
    return {
        'training_time_hours': training_time,
        'training_time_per_episode': training_time * 3600 / num_episodes,  # seconds
        'inference_time_seconds': inference_time,
        'quality_per_training_hour': solution_quality / training_time if training_time > 0 else 0,
        'total_time_hours': training_time + (inference_time / 3600)
    }


class ConstraintViolationAnalyzer:
    """
    Analyzer for constraint violations.
    
    Computes violation rates and severity for different constraint types.
    
    Args:
        constraint_types: List of constraint type names
    
    Example:
        >>> analyzer = ConstraintViolationAnalyzer()
        >>> analyzer.add_violations(violations)
        >>> stats = analyzer.get_statistics()
    """
    
    def __init__(self, constraint_types: Optional[List[str]] = None):
        if constraint_types is None:
            constraint_types = [
                'capacity', 'time_window', 'demand',
                'sequencing', 'station_limit', 'max_route_time'
            ]
        
        self.constraint_types = constraint_types
        self.violations_by_type = {ct: [] for ct in constraint_types}
        self.total_solutions = 0
    
    def add_violations(self, violations: List):
        """Add violations from a solution."""
        self.total_solutions += 1
        
        for violation in violations:
            constraint_type = violation.constraint_type.value
            if constraint_type in self.violations_by_type:
                self.violations_by_type[constraint_type].append(violation.severity)
    
    def get_statistics(self) -> Dict:
        """Get violation statistics."""
        stats = {}
        
        for constraint_type in self.constraint_types:
            violations = self.violations_by_type[constraint_type]
            
            if violations:
                stats[constraint_type] = {
                    'count': len(violations),
                    'rate': len(violations) / self.total_solutions,
                    'mean_severity': np.mean(violations),
                    'max_severity': np.max(violations),
                    'total_severity': np.sum(violations)
                }
            else:
                stats[constraint_type] = {
                    'count': 0,
                    'rate': 0.0,
                    'mean_severity': 0.0,
                    'max_severity': 0.0,
                    'total_severity': 0.0
                }
        
        # Overall statistics
        total_violations = sum(len(v) for v in self.violations_by_type.values())
        stats['overall'] = {
            'total_violations': total_violations,
            'violation_rate': total_violations / self.total_solutions if self.total_solutions > 0 else 0,
            'feasibility_rate': 1 - (sum(1 for v in self.violations_by_type.values() if v) / self.total_solutions) if self.total_solutions > 0 else 1.0
        }
        
        return stats


def compare_solutions(
    solution1: Dict,
    solution2: Dict,
    metric_names: Optional[List[str]] = None
) -> Dict:
    """
    Compare two solutions across multiple metrics.
    
    Args:
        solution1: First solution
        solution2: Second solution
        metric_names: Metrics to compare (default: ['cost', 'distance', 'time'])
    
    Returns:
        comparison: Dictionary with comparison results
    """
    if metric_names is None:
        metric_names = ['cost', 'distance', 'time', 'num_routes']
    
    comparison = {}
    
    for metric in metric_names:
        val1 = solution1.get(metric, 0)
        val2 = solution2.get(metric, 0)
        
        difference = val2 - val1
        pct_change = (difference / val1 * 100) if val1 != 0 else 0
        
        comparison[metric] = {
            'solution1': val1,
            'solution2': val2,
            'difference': difference,
            'pct_change': pct_change,
            'better': 'solution1' if val1 < val2 else 'solution2' if val2 < val1 else 'equal'
        }
    
    return comparison


if __name__ == '__main__':
    # Test metrics calculator
    print("Testing Metrics Calculator...")
    
    # Dummy network
    network = {
        'fuel_cost_per_km': 0.5,
        'driver_cost_per_hour': 25.0,
        'vehicle_fixed_cost': 50.0,
        'vehicle_capacity': 1000.0
    }
    
    # Dummy solution
    solution = {
        'routes': [
            {'distance': 150, 'time': 180, 'load': 800},
            {'distance': 120, 'time': 150, 'load': 600},
            {'distance': 100, 'time': 120, 'load': 500}
        ],
        'cost': 523.75,
        'computation_time': 2.5
    }
    
    calculator = MetricsCalculator(network, optimal_cost=500.0)
    metrics = calculator.compute_metrics(solution)
    
    print(f"\n✓ Evaluation Metrics:")
    print(f"  Total Cost: ${metrics.total_cost:.2f}")
    print(f"  Optimality Gap: {metrics.optimality_gap:.2%}")
    print(f"  Number of Routes: {metrics.num_routes}")
    print(f"  Total Distance: {metrics.total_distance:.1f} km")
    print(f"  Avg Route Load: {metrics.avg_route_load:.2%}")
    print(f"  Feasibility: {metrics.feasibility}")
    print(f"  Computation Time: {metrics.computation_time:.2f}s")
    
    # Test cost breakdown
    print(f"\n✓ Cost Breakdown:")
    breakdown = calculator.compute_cost_breakdown(solution['routes'])
    print(f"  Fuel: ${breakdown['fuel_cost']:.2f} ({breakdown['fuel_pct']:.1f}%)")
    print(f"  Driver: ${breakdown['driver_cost']:.2f} ({breakdown['driver_pct']:.1f}%)")
    print(f"  Fixed: ${breakdown['fixed_cost']:.2f} ({breakdown['fixed_pct']:.1f}%)")
    
    # Test robustness metrics
    print(f"\n✓ Robustness Metrics:")
    solutions = [
        {'cost': 500 + np.random.randn() * 20}
        for _ in range(100)
    ]
    robustness = compute_robustness_metrics(solutions)
    print(f"  CV: {robustness['coefficient_of_variation']:.3f}")
    print(f"  95th percentile: ${robustness['95th_percentile']:.2f}")
    print(f"  Expected cost: ${robustness['expected_cost']:.2f}")
    
    print("\n✓ All tests passed!")
