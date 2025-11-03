"""
Gurobi MILP Solver
Exact solver using Gurobi commercial optimizer.

Section 3.2: Deterministic equivalent model.
Section 5.4: Detailed comparison with exact solver.

Requires Gurobi license (academic or commercial).
Install: pip install gurobipy

Author: Majdi Argooubi
Date: 2025
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import warnings

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    warnings.warn(
        "Gurobi not installed. Install with: pip install gurobipy\n"
        "Requires license: https://www.gurobi.com/downloads/"
    )


class GurobiSolver:
    """
    Exact MILP solver using Gurobi commercial optimizer.
    
    Implements deterministic equivalent model (Section 3.2).
    
    Formulation:
        Minimize: Σ c_ij x_ij + fixed costs + penalty costs
        Subject to:
            - Vehicle capacity constraints
            - Flow conservation
            - Time window constraints
            - Demand satisfaction
            - Subtour elimination
    
    Section 5.4: Comparison shows:
        - 10 nodes: Optimal in ~285s
        - 50 nodes: Optimal in ~4,580s
        - 100 nodes: Best known in 7,200s (MIP gap 1.2%)
        - 200 nodes: No solution within 4 hours
    
    Args:
        network: Network data
        config: Solver configuration
            - time_limit: Time limit in seconds (default: 7200)
            - mip_gap: MIP gap tolerance (default: 0.01 = 1%)
            - threads: Number of threads (default: 8)
            - presolve: Presolve level (default: 2)
            - method: LP method (-1=auto, 0=primal, 1=dual, 2=barrier)
    
    Example:
        >>> solver = GurobiSolver(network, config={'time_limit': 7200})
        >>> solution = solver.solve()
        >>> print(f"Optimal Cost: ${solution['cost']:.2f}")
        >>> print(f"MIP Gap: {solution['mip_gap']:.2%}")
    """
    
    def __init__(
        self,
        network: Dict,
        config: Optional[Dict] = None
    ):
        if not GUROBI_AVAILABLE:
            raise ImportError(
                "Gurobi not available. Install with: pip install gurobipy\n"
                "Requires license: https://www.gurobi.com/downloads/"
            )
        
        self.network = network
        self.config = config or {}
        
        # Default configuration
        self.time_limit = self.config.get('time_limit', 7200)  # 2 hours
        self.mip_gap = self.config.get('mip_gap', 0.01)  # 1%
        self.threads = self.config.get('threads', 8)
        self.presolve = self.config.get('presolve', 2)
        self.method = self.config.get('method', -1)  # Auto
        
        # Extract network data
        self.nodes = network['nodes']
        self.num_nodes = len(self.nodes)
        self.depot_idx = 0
        
        # Build matrices
        self.distance_matrix = self._build_distance_matrix()
        self.time_matrix = self._build_time_matrix()
        
        # Parameters
        self.demands = np.array([node.get('demand', 0) for node in self.nodes])
        self.vehicle_capacity = network.get('vehicle_capacity', 1000.0)
        self.num_vehicles = network.get('num_vehicles', 20)
        self.max_route_time = network.get('max_route_time', 480.0)
        
        # Time windows
        self.time_windows = np.array([
            [node.get('time_window_start', 0), node.get('time_window_end', 1440)]
            for node in self.nodes
        ])
        
        # Cost parameters
        self.fuel_cost_per_km = 0.5
        self.driver_cost_per_hour = 25.0
        self.vehicle_fixed_cost = 50.0
        self.penalty_unmet = 10.0
    
    def _build_distance_matrix(self) -> np.ndarray:
        """Build distance matrix."""
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
        """Build time matrix."""
        speed = self.network.get('vehicle_speed', 60.0)
        return self.distance_matrix / speed * 60.0
    
    def solve(self) -> Dict:
        """
        Solve VRP using Gurobi MILP.
        
        Returns:
            solution: Dictionary containing:
                - 'cost': Total cost
                - 'routes': Optimized routes
                - 'time': Solve time (seconds)
                - 'mip_gap': Final MIP gap
                - 'status': 'optimal', 'feasible', 'infeasible', 'time_limit'
                - 'obj_bound': Objective bound (lower bound for minimization)
                - 'nodes_explored': Number of branch-and-bound nodes explored
        """
        start_time = time.time()
        
        try:
            # Create model
            model = gp.Model("VRP")
            
            # Set parameters
            model.setParam('TimeLimit', self.time_limit)
            model.setParam('MIPGap', self.mip_gap)
            model.setParam('Threads', self.threads)
            model.setParam('Presolve', self.presolve)
            model.setParam('Method', self.method)
            model.setParam('OutputFlag', 1)  # Enable output
            
            # Decision variables
            x, y, t, u = self._create_variables(model)
            
            # Objective function
            self._set_objective(model, x, y)
            
            # Constraints
            self._add_constraints(model, x, y, t, u)
            
            # Optimize
            model.optimize()
            
            solve_time = time.time() - start_time
            
            # Extract solution
            if model.status == GRB.OPTIMAL:
                status = 'optimal'
                mip_gap = 0.0
            elif model.status == GRB.TIME_LIMIT:
                status = 'time_limit'
                mip_gap = model.MIPGap
            elif model.status == GRB.INFEASIBLE:
                status = 'infeasible'
                return {
                    'status': status,
                    'time': solve_time,
                    'cost': float('inf'),
                    'routes': [],
                    'mip_gap': None
                }
            else:
                status = 'feasible'
                mip_gap = model.MIPGap if hasattr(model, 'MIPGap') else None
            
            # Extract routes
            routes = self._extract_routes(x, y)
            
            # Compute cost
            total_cost = model.objVal if hasattr(model, 'objVal') else float('inf')
            
            return {
                'status': status,
                'cost': total_cost,
                'routes': routes,
                'time': solve_time,
                'mip_gap': mip_gap,
                'obj_bound': model.ObjBound if hasattr(model, 'ObjBound') else None,
                'nodes_explored': model.NodeCount if hasattr(model, 'NodeCount') else 0,
                'num_routes': len(routes),
                'total_distance': sum(r['distance'] for r in routes),
            }
            
        except gp.GurobiError as e:
            print(f"Gurobi error: {e}")
            return {
                'status': 'error',
                'cost': float('inf'),
                'routes': [],
                'time': time.time() - start_time,
                'error': str(e)
            }
    
    def _create_variables(self, model: gp.Model) -> Tuple:
        """
        Create decision variables.
        
        Returns:
            x: Binary variables x[i,j,k] = 1 if vehicle k travels from i to j
            y: Binary variables y[i,k] = 1 if node i is served by vehicle k
            t: Continuous variables t[i,k] = arrival time at node i by vehicle k
            u: Continuous variables u[i,k] = load of vehicle k when leaving i
        """
        n = self.num_nodes
        K = self.num_vehicles
        
        # x[i,j,k]: vehicle k travels from i to j
        x = model.addVars(n, n, K, vtype=GRB.BINARY, name='x')
        
        # y[i,k]: node i served by vehicle k
        y = model.addVars(n, K, vtype=GRB.BINARY, name='y')
        
        # t[i,k]: arrival time at node i by vehicle k
        t = model.addVars(n, K, vtype=GRB.CONTINUOUS, lb=0, name='t')
        
        # u[i,k]: load when leaving node i
        u = model.addVars(n, K, vtype=GRB.CONTINUOUS, lb=0, ub=self.vehicle_capacity, name='u')
        
        return x, y, t, u
    
    def _set_objective(self, model: gp.Model, x, y):
        """
        Set objective function: minimize total cost.
        
        Cost = fuel cost + driver cost + vehicle fixed cost + unmet demand penalty
        """
        n = self.num_nodes
        K = self.num_vehicles
        depot = self.depot_idx
        
        # Distance cost (fuel)
        distance_cost = gp.quicksum(
            self.distance_matrix[i, j] * self.fuel_cost_per_km * x[i, j, k]
            for i in range(n) for j in range(n) for k in range(K) if i != j
        )
        
        # Time cost (driver) - approximation
        time_cost = gp.quicksum(
            self.time_matrix[i, j] / 60.0 * self.driver_cost_per_hour * x[i, j, k]
            for i in range(n) for j in range(n) for k in range(K) if i != j
        )
        
        # Fixed vehicle cost
        vehicle_cost = gp.quicksum(
            self.vehicle_fixed_cost * x[depot, j, k]
            for j in range(1, n) for k in range(K)
        )
        
        # Unmet demand penalty
        unmet_penalty = gp.quicksum(
            self.penalty_unmet * self.demands[i] * (1 - gp.quicksum(y[i, k] for k in range(K)))
            for i in range(1, n)
        )
        
        total_cost = distance_cost + time_cost + vehicle_cost + unmet_penalty
        
        model.setObjective(total_cost, GRB.MINIMIZE)
    
    def _add_constraints(self, model: gp.Model, x, y, t, u):
        """Add all constraints to the model."""
        n = self.num_nodes
        K = self.num_vehicles
        depot = self.depot_idx
        
        # 1. Each customer visited at most once
        for i in range(1, n):
            model.addConstr(
                gp.quicksum(y[i, k] for k in range(K)) <= 1,
                name=f'visit_{i}'
            )
        
        # 2. Flow conservation
        for k in range(K):
            for i in range(n):
                if i == depot:
                    # Depot: outflow - inflow = 0 or 1 (vehicle used)
                    model.addConstr(
                        gp.quicksum(x[depot, j, k] for j in range(1, n)) -
                        gp.quicksum(x[j, depot, k] for j in range(1, n)) == 0,
                        name=f'depot_flow_{k}'
                    )
                else:
                    # Customer: inflow = outflow = y[i,k]
                    model.addConstr(
                        gp.quicksum(x[j, i, k] for j in range(n) if j != i) == y[i, k],
                        name=f'inflow_{i}_{k}'
                    )
                    model.addConstr(
                        gp.quicksum(x[i, j, k] for j in range(n) if j != i) == y[i, k],
                        name=f'outflow_{i}_{k}'
                    )
        
        # 3. Vehicle capacity
        for k in range(K):
            model.addConstr(
                gp.quicksum(self.demands[i] * y[i, k] for i in range(1, n)) <= self.vehicle_capacity,
                name=f'capacity_{k}'
            )
        
        # 4. Time window constraints
        M = 10000  # Big M
        for k in range(K):
            for i in range(n):
                for j in range(1, n):
                    if i != j:
                        # If vehicle k travels from i to j, update arrival time
                        model.addConstr(
                            t[j, k] >= t[i, k] + self.time_matrix[i, j] + 10 - M * (1 - x[i, j, k]),
                            name=f'time_{i}_{j}_{k}'
                        )
                
                # Time window bounds
                tw_start, tw_end = self.time_windows[i]
                model.addConstr(t[i, k] >= tw_start * y[i, k], name=f'tw_start_{i}_{k}')
                model.addConstr(t[i, k] <= tw_end + M * (1 - y[i, k]), name=f'tw_end_{i}_{k}')
        
        # 5. Maximum route time
        for k in range(K):
            model.addConstr(
                t[depot, k] <= self.max_route_time,
                name=f'max_time_{k}'
            )
        
        # 6. Subtour elimination (MTZ constraints)
        for k in range(K):
            for i in range(1, n):
                for j in range(1, n):
                    if i != j:
                        model.addConstr(
                            u[i, k] - u[j, k] + self.vehicle_capacity * x[i, j, k] <= 
                            self.vehicle_capacity - self.demands[j],
                            name=f'subtour_{i}_{j}_{k}'
                        )
    
    def _extract_routes(self, x, y) -> List[Dict]:
        """Extract routes from solution."""
        n = self.num_nodes
        K = self.num_vehicles
        depot = self.depot_idx
        
        routes = []
        
        for k in range(K):
            # Check if vehicle is used
            if sum(x[depot, j, k].X for j in range(1, n)) < 0.5:
                continue
            
            # Build route
            route_nodes = []
            current = depot
            visited = set([depot])
            
            while True:
                # Find next node
                next_node = None
                for j in range(n):
                    if j not in visited and x[current, j, k].X > 0.5:
                        next_node = j
                        break
                
                if next_node is None or next_node == depot:
                    break
                
                route_nodes.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if route_nodes:
                # Compute route metrics
                distance = self.distance_matrix[depot, route_nodes[0]]
                for i in range(len(route_nodes) - 1):
                    distance += self.distance_matrix[route_nodes[i], route_nodes[i+1]]
                distance += self.distance_matrix[route_nodes[-1], depot]
                
                load = sum(self.demands[i] for i in route_nodes)
                
                routes.append({
                    'vehicle_id': k,
                    'nodes': route_nodes,
                    'distance': distance,
                    'load': load
                })
        
        return routes


if __name__ == '__main__':
    if not GUROBI_AVAILABLE:
        print("❌ Gurobi not available")
        print("Install: pip install gurobipy")
        print("License: https://www.gurobi.com/downloads/")
        exit(1)
    
    # Test with small instance
    print("Testing Gurobi Solver (small instance)...")
    
    np.random.seed(42)
    num_nodes = 10
    
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
        'vehicle_capacity': 500,
        'num_vehicles': 5,
        'max_route_time': 480,
        'vehicle_speed': 60.0
    }
    
    solver = GurobiSolver(network, config={'time_limit': 300, 'mip_gap': 0.01})
    solution = solver.solve()
    
    print(f"\n✓ Solution status: {solution['status']}")
    print(f"  Total cost: ${solution['cost']:.2f}")
    print(f"  Number of routes: {solution.get('num_routes', 0)}")
    print(f"  Solve time: {solution['time']:.2f}s")
    print(f"  MIP gap: {solution.get('mip_gap', 0):.2%}")
    print(f"  Nodes explored: {solution.get('nodes_explored', 0):,}")
