"""
Clarke-Wright Savings Algorithm
Classical heuristic for Vehicle Routing Problem (VRP).

Section 5.1.3: Heuristic baseline for comparison.
Constructs routes by iteratively merging pairs with maximum savings.

Reference: Clarke, G., & Wright, J. W. (1964). Scheduling of vehicles from a 
central depot to a number of delivery points. Operations Research, 12(4), 568-581.

Author: Majdi Argoubi
Date: 2025
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import heapq


@dataclass
class Route:
    """
    Represents a vehicle route.
    
    Attributes:
        nodes: List of node indices in order
        vehicle_id: Vehicle assigned to this route
        load: Current load on vehicle
        distance: Total route distance
        time: Total route time
        feasible: Whether route satisfies all constraints
    """
    nodes: List[int]
    vehicle_id: int
    load: float = 0.0
    distance: float = 0.0
    time: float = 0.0
    feasible: bool = True
    
    def __len__(self):
        return len(self.nodes)


class ClarkeWrightSolver:
    """
    Clarke-Wright Savings Algorithm for VRP.
    
    Constructs routes by iteratively merging pairs with maximum savings:
        s_ij = d_0i + d_0j - d_ij
    
    where:
        - d_0i: distance from depot to node i
        - d_0j: distance from depot to node j
        - d_ij: distance between nodes i and j
    
    Section 5.1.3: Classical heuristic baseline achieving 13.1% gap.
    
    Args:
        network: Network data containing:
            - 'nodes': List of node dictionaries with coordinates, demands
            - 'edges': Distance/time matrix
            - 'depot': Depot information
        vehicle_capacity: Vehicle capacity Q_k
        max_route_time: Maximum route duration (minutes)
        time_windows: Enable time window constraints (default: True)
        num_vehicles: Maximum number of vehicles available
    
    Example:
        >>> solver = ClarkeWrightSolver(network, vehicle_capacity=1000)
        >>> routes, cost, solve_time = solver.solve()
        >>> print(f"Total Cost: ${cost:.2f}, Time: {solve_time:.2f}s")
    """
    
    def __init__(
        self,
        network: Dict,
        vehicle_capacity: float = 1000.0,
        max_route_time: float = 480.0,  # 8 hours
        time_windows: bool = True,
        num_vehicles: int = 20
    ):
        self.network = network
        self.vehicle_capacity = vehicle_capacity
        self.max_route_time = max_route_time
        self.time_windows = time_windows
        self.num_vehicles = num_vehicles
        
        # Extract network components
        self.nodes = network['nodes']
        self.num_nodes = len(self.nodes)
        self.depot = network['depot']
        
        # Distance matrix
        self.distance_matrix = self._build_distance_matrix()
        
        # Time matrix (can differ from distance if speeds vary)
        self.time_matrix = self._build_time_matrix()
        
        # Node demands
        self.demands = np.array([node.get('demand', 0) for node in self.nodes])
        
        # Time windows [earliest, latest]
        if time_windows:
            self.time_windows_data = np.array([
                [node.get('time_window_start', 0), node.get('time_window_end', 1440)]
                for node in self.nodes
            ])
        else:
            self.time_windows_data = None
    
    def _build_distance_matrix(self) -> np.ndarray:
        """Build distance matrix from network edges."""
        n = self.num_nodes
        dist_matrix = np.zeros((n, n))
        
        # Euclidean distance if coordinates available
        if 'coordinates' in self.nodes[0]:
            coords = np.array([node['coordinates'] for node in self.nodes])
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
        else:
            # Use provided edge distances
            edges = self.network.get('edges', {})
            for (i, j), distance in edges.items():
                dist_matrix[i, j] = distance
                dist_matrix[j, i] = distance
        
        return dist_matrix
    
    def _build_time_matrix(self) -> np.ndarray:
        """Build time matrix (travel times)."""
        # Assume speed = 60 km/h for simplicity
        # Time in minutes = distance / speed * 60
        speed = self.network.get('vehicle_speed', 60.0)  # km/h
        return self.distance_matrix / speed * 60.0
    
    def _compute_savings(self) -> List[Tuple[float, int, int]]:
        """
        Compute savings for all node pairs.
        
        Savings: s_ij = d_0i + d_0j - d_ij
        
        Returns:
            List of (savings, i, j) sorted by savings (descending)
        """
        depot_idx = 0  # Assume depot is node 0
        savings_list = []
        
        for i in range(1, self.num_nodes):  # Skip depot
            for j in range(i + 1, self.num_nodes):
                # Savings from combining routes (0->i->0) and (0->j->0)
                # into route (0->i->j->0)
                savings = (
                    self.distance_matrix[depot_idx, i] +
                    self.distance_matrix[depot_idx, j] -
                    self.distance_matrix[i, j]
                )
                savings_list.append((savings, i, j))
        
        # Sort by savings (descending)
        savings_list.sort(reverse=True, key=lambda x: x[0])
        
        return savings_list
    
    def _check_capacity(self, route: Route, additional_demand: float) -> bool:
        """Check if adding demand violates capacity constraint."""
        return (route.load + additional_demand) <= self.vehicle_capacity
    
    def _check_time_windows(self, route: Route, new_node: int, position: int) -> bool:
        """Check if inserting node at position satisfies time windows."""
        if not self.time_windows:
            return True
        
        # Simulate route with new node
        nodes = route.nodes[:position] + [new_node] + route.nodes[position:]
        current_time = 0
        depot_idx = 0
        
        for idx, node in enumerate(nodes):
            if idx == 0:
                # Travel from depot to first node
                current_time += self.time_matrix[depot_idx, node]
            else:
                # Travel from previous node
                current_time += self.time_matrix[nodes[idx-1], node]
            
            # Check time window
            tw_start, tw_end = self.time_windows_data[node]
            
            if current_time > tw_end:
                return False  # Too late
            
            # Wait if arriving early
            if current_time < tw_start:
                current_time = tw_start
            
            # Service time (simplified: 10 minutes per node)
            current_time += 10
        
        # Return to depot
        current_time += self.time_matrix[nodes[-1], depot_idx]
        
        return current_time <= self.max_route_time
    
    def _merge_routes(
        self,
        route1: Route,
        route2: Route,
        i: int,
        j: int
    ) -> Optional[Route]:
        """
        Attempt to merge two routes by connecting nodes i and j.
        
        Args:
            route1: First route
            route2: Second route
            i: Node from route1
            j: Node from route2
        
        Returns:
            Merged route if feasible, None otherwise
        """
        # Check which ends the nodes are at
        i_at_start = (route1.nodes[0] == i)
        i_at_end = (route1.nodes[-1] == i)
        j_at_start = (route2.nodes[0] == j)
        j_at_end = (route2.nodes[-1] == j)
        
        # Only merge if nodes are at route ends
        if not ((i_at_start or i_at_end) and (j_at_start or j_at_end)):
            return None
        
        # Determine merge configuration
        if i_at_end and j_at_start:
            # route1 -> route2
            merged_nodes = route1.nodes + route2.nodes
        elif i_at_start and j_at_end:
            # route2 -> route1
            merged_nodes = route2.nodes + route1.nodes
        elif i_at_end and j_at_end:
            # route1 -> reversed(route2)
            merged_nodes = route1.nodes + list(reversed(route2.nodes))
        elif i_at_start and j_at_start:
            # reversed(route1) -> route2
            merged_nodes = list(reversed(route1.nodes)) + route2.nodes
        else:
            return None
        
        # Create merged route
        merged_route = Route(
            nodes=merged_nodes,
            vehicle_id=route1.vehicle_id,
            load=route1.load + route2.load
        )
        
        # Check capacity
        if merged_route.load > self.vehicle_capacity:
            return None
        
        # Check time windows
        if self.time_windows and not self._check_time_windows(merged_route, merged_nodes[0], 0):
            return None
        
        # Compute route metrics
        self._compute_route_metrics(merged_route)
        
        # Check time constraint
        if merged_route.time > self.max_route_time:
            return None
        
        return merged_route
    
    def _compute_route_metrics(self, route: Route):
        """Compute distance and time for route."""
        depot_idx = 0
        distance = 0
        time = 0
        
        # Depot to first node
        if len(route.nodes) > 0:
            distance += self.distance_matrix[depot_idx, route.nodes[0]]
            time += self.time_matrix[depot_idx, route.nodes[0]]
            time += 10  # Service time
        
        # Between nodes
        for i in range(len(route.nodes) - 1):
            distance += self.distance_matrix[route.nodes[i], route.nodes[i+1]]
            time += self.time_matrix[route.nodes[i], route.nodes[i+1]]
            time += 10  # Service time
        
        # Last node to depot
        if len(route.nodes) > 0:
            distance += self.distance_matrix[route.nodes[-1], depot_idx]
            time += self.time_matrix[route.nodes[-1], depot_idx]
        
        route.distance = distance
        route.time = time
    
    def solve(self) -> Tuple[List[Route], float, float]:
        """
        Solve VRP using Clarke-Wright algorithm.
        
        Returns:
            routes: List of Route objects
            total_cost: Total routing cost ($)
            solve_time: Computation time (seconds)
        
        Algorithm:
            1. Initialize: each customer in separate route
            2. Compute savings for all pairs
            3. Sort savings (descending)
            4. For each pair (i,j) with highest savings:
                - If i and j in different routes at route ends:
                  - Merge routes if feasible
            5. Return final routes
        """
        start_time = time.time()
        
        # Initialize: each node in separate route
        routes = []
        node_to_route = {}  # Map node -> route index
        
        for node_idx in range(1, self.num_nodes):  # Skip depot
            route = Route(
                nodes=[node_idx],
                vehicle_id=len(routes),
                load=self.demands[node_idx]
            )
            self._compute_route_metrics(route)
            routes.append(route)
            node_to_route[node_idx] = len(routes) - 1
        
        # Compute savings
        savings_list = self._compute_savings()
        
        # Merge routes based on savings
        for savings_value, i, j in savings_list:
            # Check if nodes are in different routes
            if i not in node_to_route or j not in node_to_route:
                continue
            
            route1_idx = node_to_route[i]
            route2_idx = node_to_route[j]
            
            if route1_idx == route2_idx:
                continue  # Already in same route
            
            route1 = routes[route1_idx]
            route2 = routes[route2_idx]
            
            # Attempt merge
            merged_route = self._merge_routes(route1, route2, i, j)
            
            if merged_route is not None:
                # Update routes
                routes[route1_idx] = merged_route
                routes[route2_idx] = None  # Mark as removed
                
                # Update node_to_route mapping
                for node in merged_route.nodes:
                    node_to_route[node] = route1_idx
        
        # Remove empty routes
        routes = [r for r in routes if r is not None]
        
        # Reassign vehicle IDs
        for idx, route in enumerate(routes):
            route.vehicle_id = idx
        
        # Compute total cost
        total_cost = self._compute_total_cost(routes)
        
        solve_time = time.time() - start_time
        
        return routes, total_cost, solve_time
    
    def _compute_total_cost(self, routes: List[Route]) -> float:
        """
        Compute total routing cost.
        
        Cost components (from Section 3.1):
        - Fuel cost: $0.50/km
        - Driver cost: $25/hour
        - Vehicle fixed cost: $50 per vehicle
        - Penalty for unmet demand: $10 per unit
        """
        fuel_cost_per_km = 0.5
        driver_cost_per_hour = 25.0
        vehicle_fixed_cost = 50.0
        penalty_unmet = 10.0
        
        total_cost = 0.0
        
        for route in routes:
            # Fuel cost
            total_cost += route.distance * fuel_cost_per_km
            
            # Driver cost
            total_cost += (route.time / 60.0) * driver_cost_per_hour
            
            # Vehicle fixed cost
            total_cost += vehicle_fixed_cost
        
        # Check for unmet demand (if not all nodes covered)
        covered_nodes = set()
        for route in routes:
            covered_nodes.update(route.nodes)
        
        for node_idx in range(1, self.num_nodes):
            if node_idx not in covered_nodes:
                total_cost += self.demands[node_idx] * penalty_unmet
        
        return total_cost
    
    def get_solution_dict(self, routes: List[Route]) -> Dict:
        """
        Convert routes to solution dictionary format.
        
        Returns:
            solution: Dictionary with routes, cost, metrics
        """
        return {
            'routes': [
                {
                    'vehicle_id': route.vehicle_id,
                    'nodes': route.nodes,
                    'load': route.load,
                    'distance': route.distance,
                    'time': route.time
                }
                for route in routes
            ],
            'num_routes': len(routes),
            'total_distance': sum(r.distance for r in routes),
            'total_time': sum(r.time for r in routes),
            'max_route_time': max(r.time for r in routes) if routes else 0
        }


class ParallelClarkeWright(ClarkeWrightSolver):
    """
    Parallel version of Clarke-Wright using route-first, cluster-second.
    
    Builds one giant tour ignoring capacity, then splits into feasible routes.
    Often produces better solutions than sequential Clarke-Wright.
    """
    
    def solve(self) -> Tuple[List[Route], float, float]:
        """Solve using parallel Clarke-Wright."""
        start_time = time.time()
        
        # Step 1: Build giant tour (ignore capacity)
        giant_tour = self._build_giant_tour()
        
        # Step 2: Split into feasible routes
        routes = self._split_giant_tour(giant_tour)
        
        # Compute total cost
        total_cost = self._compute_total_cost(routes)
        
        solve_time = time.time() - start_time
        
        return routes, total_cost, solve_time
    
    def _build_giant_tour(self) -> List[int]:
        """Build giant tour using nearest neighbor."""
        unvisited = set(range(1, self.num_nodes))
        tour = []
        current = 0  # Start at depot
        
        while unvisited:
            # Find nearest unvisited node
            nearest = min(unvisited, key=lambda x: self.distance_matrix[current, x])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return tour
    
    def _split_giant_tour(self, tour: List[int]) -> List[Route]:
        """Split giant tour into feasible routes using dynamic programming."""
        n = len(tour)
        
        # DP: best_cost[i] = minimum cost to serve first i customers
        best_cost = [float('inf')] * (n + 1)
        best_cost[0] = 0
        pred = [-1] * (n + 1)
        
        depot_idx = 0
        
        for i in range(n):
            load = 0
            time = 0
            distance = 0
            
            # Try routes starting at position i
            for j in range(i, n):
                # Add customer j to current route
                load += self.demands[tour[j]]
                
                if load > self.vehicle_capacity:
                    break
                
                # Update distance and time
                if j == i:
                    distance += self.distance_matrix[depot_idx, tour[j]]
                    time += self.time_matrix[depot_idx, tour[j]]
                else:
                    distance += self.distance_matrix[tour[j-1], tour[j]]
                    time += self.time_matrix[tour[j-1], tour[j]]
                
                time += 10  # Service time
                
                # Return to depot
                return_distance = self.distance_matrix[tour[j], depot_idx]
                return_time = self.time_matrix[tour[j], depot_idx]
                
                total_distance = distance + return_distance
                total_time = time + return_time
                
                if total_time > self.max_route_time:
                    break
                
                # Route cost
                route_cost = (
                    total_distance * 0.5 +  # Fuel
                    (total_time / 60.0) * 25.0 +  # Driver
                    50.0  # Fixed cost
                )
                
                # Update DP
                if best_cost[i] + route_cost < best_cost[j + 1]:
                    best_cost[j + 1] = best_cost[i] + route_cost
                    pred[j + 1] = i
        
        # Reconstruct routes
        routes = []
        j = n
        vehicle_id = 0
        
        while j > 0:
            i = pred[j]
            route_nodes = tour[i:j]
            
            route = Route(
                nodes=route_nodes,
                vehicle_id=vehicle_id,
                load=sum(self.demands[node] for node in route_nodes)
            )
            self._compute_route_metrics(route)
            routes.append(route)
            
            vehicle_id += 1
            j = i
        
        routes.reverse()
        return routes


if __name__ == '__main__':
    # Test Clarke-Wright
    print("Testing Clarke-Wright Savings Algorithm...")
    
    # Dummy network
    np.random.seed(42)
    num_nodes = 20
    
    network = {
        'nodes': [
            {
                'id': i,
                'coordinates': np.random.rand(2) * 100,
                'demand': np.random.randint(50, 150) if i > 0 else 0,
                'time_window_start': 0,
                'time_window_end': 480
            }
            for i in range(num_nodes)
        ],
        'depot': {'id': 0, 'coordinates': [0, 0]},
        'vehicle_speed': 60.0
    }
    
    # Solve
    solver = ClarkeWrightSolver(network, vehicle_capacity=500, time_windows=False)
    routes, cost, solve_time = solver.solve()
    
    print(f"✓ Solution found in {solve_time:.2f}s")
    print(f"  Number of routes: {len(routes)}")
    print(f"  Total cost: ${cost:.2f}")
    print(f"  Total distance: {sum(r.distance for r in routes):.1f} km")
    
    # Test parallel version
    print("\nTesting Parallel Clarke-Wright...")
    parallel_solver = ParallelClarkeWright(network, vehicle_capacity=500, time_windows=False)
    routes_p, cost_p, time_p = parallel_solver.solve()
    
    print(f"✓ Parallel solution found in {time_p:.2f}s")
    print(f"  Number of routes: {len(routes_p)}")
    print(f"  Total cost: ${cost_p:.2f}")
    print(f"  Improvement: {(cost - cost_p) / cost * 100:.1f}%")
