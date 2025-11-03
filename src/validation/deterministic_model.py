"""
Deterministic Model (Feasibility Oracle)
Serves as dynamic feasibility oracle for three-tier validation.

Section 3.2: Deterministic equivalent model.
Section 4.3: Three-tier closed-loop constraint validation.

Author: Your Name
Date: 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ConstraintType(Enum):
    """Types of constraints in the VRP."""
    CAPACITY = "capacity"
    TIME_WINDOW = "time_window"
    DEMAND = "demand"
    SEQUENCING = "sequencing"
    STATION_LIMIT = "station_limit"
    MAX_ROUTE_TIME = "max_route_time"


@dataclass
class ConstraintViolation:
    """
    Represents a constraint violation.
    
    Attributes:
        constraint_type: Type of constraint violated
        severity: Violation amount (normalized)
        node: Node index (if applicable)
        vehicle: Vehicle index (if applicable)
        description: Human-readable description
    """
    constraint_type: ConstraintType
    severity: float
    node: Optional[int] = None
    vehicle: Optional[int] = None
    description: str = ""


class DeterministicModel:
    """
    Deterministic equivalent model for feasibility checking.
    
    Serves as feasibility oracle in three-tier validation (Section 4.3).
    Evaluates solutions against deterministic constraints derived from
    stochastic chance constraints (Section 3.2).
    
    Constraints checked (Section 3.1):
    1. Vehicle capacity (Constraint 1)
    2. Demand satisfaction (Constraint 2)
    3. Time windows (Constraint 19)
    4. Sequencing (Constraints 11-17)
    5. Station vehicle limit
    6. Maximum route duration
    
    Args:
        network: Network topology and constraints
        config: Validation configuration
            - capacity_tolerance: Tolerance for capacity violations (default: 0.05)
            - time_window_tolerance: Tolerance for time window violations (default: 0.1)
            - confidence_level: Confidence level for chance constraints (default: 0.95)
    
    Example:
        >>> model = DeterministicModel(network, config)
        >>> is_feasible, violations = model.check_feasibility(solution)
        >>> print(f"Feasible: {is_feasible}, Violations: {len(violations)}")
    """
    
    def __init__(
        self,
        network: Dict,
        config: Optional[Dict] = None
    ):
        self.network = network
        self.config = config or {}
        
        # Tolerances
        self.capacity_tolerance = self.config.get('capacity_tolerance', 0.05)
        self.time_window_tolerance = self.config.get('time_window_tolerance', 0.1)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        
        # Network parameters
        self.nodes = network['nodes']
        self.num_nodes = len(self.nodes)
        self.depot_idx = 0
        
        # Vehicle parameters
        self.vehicle_capacity = network.get('vehicle_capacity', 1000.0)
        self.num_vehicles = network.get('num_vehicles', 20)
        self.max_route_time = network.get('max_route_time', 480.0)
        
        # Demand parameters (stochastic)
        self.demand_mean = np.array([node.get('demand_mean', 0) for node in self.nodes])
        self.demand_std = np.array([node.get('demand_std', 0) for node in self.nodes])
        
        # Deterministic demand (mean + safety stock)
        # For 95% confidence: demand_det = μ + 1.645σ
        z_score = self._get_z_score(self.confidence_level)
        self.demand_deterministic = self.demand_mean + z_score * self.demand_std
        
        # Time windows
        self.time_windows = np.array([
            [node.get('time_window_start', 0), node.get('time_window_end', 1440)]
            for node in self.nodes
        ])
        
        # Distance and time matrices
        self.distance_matrix = self._build_distance_matrix()
        self.time_matrix = self._build_time_matrix()
        
        # Service times
        self.service_time = network.get('service_time_per_unit', 0.1)  # minutes per unit
    
    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for given confidence level."""
        # Standard normal quantiles
        z_scores = {
            0.90: 1.282,
            0.95: 1.645,
            0.99: 2.326
        }
        return z_scores.get(confidence_level, 1.645)
    
    def _build_distance_matrix(self) -> np.ndarray:
        """Build distance matrix from network."""
        n = self.num_nodes
        dist_matrix = np.zeros((n, n))
        
        if 'coordinates' in self.nodes[0]:
            coords = np.array([node['coordinates'] for node in self.nodes])
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
        else:
            edges = self.network.get('edges', {})
            for (i, j), distance in edges.items():
                dist_matrix[i, j] = distance
                dist_matrix[j, i] = distance
        
        return dist_matrix
    
    def _build_time_matrix(self) -> np.ndarray:
        """Build time matrix (travel times)."""
        speed = self.network.get('vehicle_speed', 60.0)
        return self.distance_matrix / speed * 60.0  # Convert to minutes
    
    def check_feasibility(
        self,
        solution: Dict
    ) -> Tuple[bool, List[ConstraintViolation]]:
        """
        Check if solution satisfies all constraints.
        
        Evaluates constraints from Section 3.1:
        - Vehicle capacity (Constraint 1)
        - Demand satisfaction (Constraint 2)
        - Time windows (Constraint 19)
        - Sequencing (Constraints 11-17)
        
        Args:
            solution: Solution dictionary containing:
                - 'routes': List of routes
                - Each route: {'vehicle_id', 'nodes', 'load', 'time'}
        
        Returns:
            is_feasible: True if all constraints satisfied
            violations: List of ConstraintViolation objects
        
        Example:
            >>> is_feasible, violations = model.check_feasibility(solution)
            >>> if not is_feasible:
            ...     for v in violations:
            ...         print(f"{v.constraint_type}: {v.severity:.2%}")
        """
        violations = []
        
        # Extract routes
        routes = solution.get('routes', [])
        
        # 1. Check vehicle capacity constraints
        capacity_violations = self._check_capacity_constraints(routes)
        violations.extend(capacity_violations)
        
        # 2. Check demand satisfaction
        demand_violations = self._check_demand_satisfaction(routes)
        violations.extend(demand_violations)
        
        # 3. Check time window constraints
        time_violations = self._check_time_windows(routes)
        violations.extend(time_violations)
        
        # 4. Check sequencing constraints
        sequencing_violations = self._check_sequencing(routes)
        violations.extend(sequencing_violations)
        
        # 5. Check station vehicle limit
        station_violations = self._check_station_limits(routes)
        violations.extend(station_violations)
        
        # 6. Check maximum route time
        route_time_violations = self._check_max_route_time(routes)
        violations.extend(route_time_violations)
        
        # Solution is feasible if no violations
        is_feasible = len(violations) == 0
        
        return is_feasible, violations
    
    def _check_capacity_constraints(self, routes: List[Dict]) -> List[ConstraintViolation]:
        """
        Check vehicle capacity constraints (Constraint 1).
        
        Constraint: Σ d_i x_ik ≤ Q_k for all vehicles k
        """
        violations = []
        
        for route in routes:
            vehicle_id = route['vehicle_id']
            nodes = route['nodes']
            
            # Compute total demand
            total_demand = sum(self.demand_deterministic[node] for node in nodes)
            
            # Check capacity
            if total_demand > self.vehicle_capacity * (1 + self.capacity_tolerance):
                severity = (total_demand - self.vehicle_capacity) / self.vehicle_capacity
                
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.CAPACITY,
                    severity=severity,
                    vehicle=vehicle_id,
                    description=f"Vehicle {vehicle_id} exceeds capacity: {total_demand:.1f} > {self.vehicle_capacity:.1f}"
                ))
        
        return violations
    
    def _check_demand_satisfaction(self, routes: List[Dict]) -> List[ConstraintViolation]:
        """
        Check demand satisfaction constraints (Constraint 2).
        
        Constraint: Each station visited at least once if demand > 0
        """
        violations = []
        
        # Find visited nodes
        visited_nodes = set()
        for route in routes:
            visited_nodes.update(route['nodes'])
        
        # Check if all nodes with demand are visited
        for node_idx in range(1, self.num_nodes):  # Skip depot
            if self.demand_deterministic[node_idx] > 0 and node_idx not in visited_nodes:
                severity = self.demand_deterministic[node_idx] / self.vehicle_capacity
                
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.DEMAND,
                    severity=severity,
                    node=node_idx,
                    description=f"Node {node_idx} not visited (demand: {self.demand_deterministic[node_idx]:.1f})"
                ))
        
        return violations
    
    def _check_time_windows(self, routes: List[Dict]) -> List[ConstraintViolation]:
        """
        Check time window constraints (Constraint 19).
        
        Constraint: e_i ≤ t_ki ≤ l_i for all nodes i
        """
        violations = []
        
        for route in routes:
            vehicle_id = route['vehicle_id']
            nodes = route['nodes']
            
            if len(nodes) == 0:
                continue
            
            # Simulate route timing
            current_time = 0
            
            # Travel from depot to first node
            current_time += self.time_matrix[self.depot_idx, nodes[0]]
            
            for idx, node in enumerate(nodes):
                # Check time window
                tw_start, tw_end = self.time_windows[node]
                
                # Early arrival - must wait
                if current_time < tw_start:
                    current_time = tw_start
                
                # Late arrival - violation
                if current_time > tw_end * (1 + self.time_window_tolerance):
                    severity = (current_time - tw_end) / tw_end
                    
                    violations.append(ConstraintViolation(
                        constraint_type=ConstraintType.TIME_WINDOW,
                        severity=severity,
                        node=node,
                        vehicle=vehicle_id,
                        description=f"Vehicle {vehicle_id} arrives late at node {node}: {current_time:.1f} > {tw_end:.1f}"
                    ))
                
                # Service time
                service_time = self.demand_deterministic[node] * self.service_time
                current_time += service_time
                
                # Travel to next node
                if idx < len(nodes) - 1:
                    current_time += self.time_matrix[node, nodes[idx + 1]]
        
        return violations
    
    def _check_sequencing(self, routes: List[Dict]) -> List[ConstraintViolation]:
        """
        Check sequencing constraints (Constraints 11-17).
        
        Ensures routes form valid tours: depot → nodes → depot
        """
        violations = []
        
        for route in routes:
            vehicle_id = route['vehicle_id']
            nodes = route['nodes']
            
            # Check for duplicate visits
            if len(nodes) != len(set(nodes)):
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.SEQUENCING,
                    severity=1.0,
                    vehicle=vehicle_id,
                    description=f"Vehicle {vehicle_id} has duplicate node visits"
                ))
            
            # Check for depot in route (should not be)
            if self.depot_idx in nodes:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.SEQUENCING,
                    severity=1.0,
                    vehicle=vehicle_id,
                    description=f"Vehicle {vehicle_id} has depot in middle of route"
                ))
        
        return violations
    
    def _check_station_limits(self, routes: List[Dict]) -> List[ConstraintViolation]:
        """
        Check station vehicle limits.
        
        Some stations may have limits on number of vehicles simultaneously.
        """
        violations = []
        
        # Count visits per station
        station_visits = {}
        for route in routes:
            for node in route['nodes']:
                if node not in station_visits:
                    station_visits[node] = 0
                station_visits[node] += 1
        
        # Check limits (if specified in network)
        for node, count in station_visits.items():
            max_vehicles = self.nodes[node].get('max_vehicles', float('inf'))
            
            if count > max_vehicles:
                severity = (count - max_vehicles) / max_vehicles
                
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.STATION_LIMIT,
                    severity=severity,
                    node=node,
                    description=f"Node {node} visited by {count} vehicles (max: {max_vehicles})"
                ))
        
        return violations
    
    def _check_max_route_time(self, routes: List[Dict]) -> List[ConstraintViolation]:
        """Check maximum route time constraint."""
        violations = []
        
        for route in routes:
            vehicle_id = route['vehicle_id']
            nodes = route['nodes']
            
            if len(nodes) == 0:
                continue
            
            # Compute total route time
            total_time = 0
            
            # Depot to first node
            total_time += self.time_matrix[self.depot_idx, nodes[0]]
            
            # Between nodes
            for i in range(len(nodes) - 1):
                total_time += self.time_matrix[nodes[i], nodes[i + 1]]
                # Service time
                total_time += self.demand_deterministic[nodes[i]] * self.service_time
            
            # Last node service
            total_time += self.demand_deterministic[nodes[-1]] * self.service_time
            
            # Return to depot
            total_time += self.time_matrix[nodes[-1], self.depot_idx]
            
            # Check limit
            if total_time > self.max_route_time:
                severity = (total_time - self.max_route_time) / self.max_route_time
                
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.MAX_ROUTE_TIME,
                    severity=severity,
                    vehicle=vehicle_id,
                    description=f"Vehicle {vehicle_id} route time {total_time:.1f} exceeds max {self.max_route_time:.1f}"
                ))
        
        return violations
    
    def get_constraint_violations_summary(
        self,
        violations: List[ConstraintViolation]
    ) -> Dict:
        """
        Summarize violations by type.
        
        Returns:
            summary: Dictionary with counts and severities per constraint type
        """
        summary = {}
        
        for constraint_type in ConstraintType:
            type_violations = [v for v in violations if v.constraint_type == constraint_type]
            
            if type_violations:
                summary[constraint_type.value] = {
                    'count': len(type_violations),
                    'total_severity': sum(v.severity for v in type_violations),
                    'max_severity': max(v.severity for v in type_violations),
                    'avg_severity': np.mean([v.severity for v in type_violations])
                }
        
        return summary


if __name__ == '__main__':
    # Test deterministic model
    print("Testing Deterministic Model...")
    
    # Dummy network
    np.random.seed(42)
    num_nodes = 20
    
    network = {
        'nodes': [
            {
                'id': i,
                'coordinates': np.random.rand(2) * 100,
                'demand_mean': np.random.randint(50, 150) if i > 0 else 0,
                'demand_std': np.random.randint(10, 30) if i > 0 else 0,
                'time_window_start': 0,
                'time_window_end': 480
            }
            for i in range(num_nodes)
        ],
        'vehicle_capacity': 500,
        'num_vehicles': 5,
        'max_route_time': 480,
        'vehicle_speed': 60.0,
        'service_time_per_unit': 0.1
    }
    
    model = DeterministicModel(network)
    
    # Test feasible solution
    feasible_solution = {
        'routes': [
            {'vehicle_id': 0, 'nodes': [1, 2, 3], 'load': 300, 'time': 120},
            {'vehicle_id': 1, 'nodes': [4, 5], 'load': 200, 'time': 90}
        ]
    }
    
    is_feasible, violations = model.check_feasibility(feasible_solution)
    print(f"✓ Feasible solution: {is_feasible}")
    print(f"  Violations: {len(violations)}")
    
    # Test infeasible solution (capacity violation)
    infeasible_solution = {
        'routes': [
            {'vehicle_id': 0, 'nodes': list(range(1, 15)), 'load': 2000, 'time': 400}
        ]
    }
    
    is_feasible, violations = model.check_feasibility(infeasible_solution)
    print(f"\n✓ Infeasible solution: {is_feasible}")
    print(f"  Violations: {len(violations)}")
    
    if violations:
        summary = model.get_constraint_violations_summary(violations)
        print(f"\n  Violations by type:")
        for constraint_type, stats in summary.items():
            print(f"    {constraint_type}: {stats['count']} violations, max severity: {stats['max_severity']:.2%}")
