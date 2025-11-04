"""
Visualization
Plotting utilities for results and analysis.

Creates publication-quality figures for the paper.

Author: Your Name
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12


class ResultVisualizer:
    """
    Visualizer for creating publication-quality figures.
    
    Creates figures matching paper style (Figures 5.1-5.6).
    
    Args:
        output_dir: Directory for saving figures
        dpi: DPI for saved figures (default: 300)
        format: Format for figures ('png', 'pdf', 'svg')
    
    Example:
        >>> viz = ResultVisualizer(output_dir='figures/')
        >>> viz.plot_training_curve(rewards, costs)
        >>> viz.save_figure('training_curve.pdf')
    """
    
    def __init__(
        self,
        output_dir: str = 'figures/',
        dpi: int = 300,
        format: str = 'pdf'
    ):
        self.output_dir = output_dir
        self.dpi = dpi
        self.format = format
        
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_training_curve(
        self,
        rewards: List[float],
        costs: List[float],
        window: int = 100,
        title: str = "Training Progress"
    ):
        """
        Plot training curves (Figure 5.1 style).
        
        Args:
            rewards: Episode rewards
            costs: Episode costs
            window: Window for moving average
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        episodes = np.arange(len(rewards))
        
        # Rewards
        ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
        
        if len(rewards) >= window:
            smooth_rewards = self._moving_average(rewards, window)
            ax1.plot(episodes[window-1:], smooth_rewards, color='darkblue', 
                    linewidth=2, label=f'{window}-episode MA')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Costs
        ax2.plot(episodes, costs, alpha=0.3, color='red', label='Raw')
        
        if len(costs) >= window:
            smooth_costs = self._moving_average(costs, window)
            ax2.plot(episodes[window-1:], smooth_costs, color='darkred',
                    linewidth=2, label=f'{window}-episode MA')
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cost ($)')
        ax2.set_title('Solution Cost')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_method_comparison(
        self,
        method_names: List[str],
        mean_costs: List[float],
        std_costs: List[float],
        title: str = "Method Comparison"
    ):
        """
        Plot method comparison bar chart (Table 5.2 as figure).
        
        Args:
            method_names: List of method names
            mean_costs: Mean costs for each method
            std_costs: Standard deviations
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(method_names))
        colors = plt.cm.Set3(np.linspace(0, 1, len(method_names)))
        
        bars = ax.bar(x, mean_costs, yerr=std_costs, capsize=5,
                     color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Method', fontweight='bold')
        ax.set_ylabel('Cost ($)', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=15, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:.0f}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_optimality_gap_boxplot(
        self,
        method_names: List[str],
        gaps_by_method: Dict[str, List[float]],
        title: str = "Optimality Gap Distribution"
    ):
        """
        Plot optimality gap boxplot (Figure 5.2 style).
        
        Args:
            method_names: List of method names
            gaps_by_method: Dictionary of {method: [gaps]}
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data = [gaps_by_method[method] for method in method_names]
        
        bp = ax.boxplot(data, labels=method_names, patch_artist=True,
                       showmeans=True, meanline=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(method_names)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Method', fontweight='bold')
        ax.set_ylabel('Optimality Gap (%)', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
        
        plt.tight_layout()
        return fig
    
    def plot_scalability(
        self,
        sizes: List[int],
        times_by_method: Dict[str, List[float]],
        title: str = "Computational Scalability"
    ):
        """
        Plot scalability analysis (Figure 5.3 style).
        
        Args:
            sizes: Problem sizes
            times_by_method: Dictionary of {method: [times]}
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        markers = ['o', 's', '^', 'v', 'D']
        
        for i, (method, times) in enumerate(times_by_method.items()):
            marker = markers[i % len(markers)]
            ax.plot(sizes, times, marker=marker, linewidth=2, 
                   markersize=8, label=method)
        
        ax.set_xlabel('Problem Size (nodes)', fontweight='bold')
        ax.set_ylabel('Computation Time (s)', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        return fig
    
    def plot_violation_analysis(
        self,
        episodes: List[int],
        v_totals: List[float],
        tiers: List[int],
        title: str = "Constraint Violation Analysis"
    ):
        """
        Plot violation and tier activation (Figure 5.4 style).
        
        Args:
            episodes: Episode numbers
            v_totals: V_total values
            tiers: Tier activations
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # V_total over time
        ax1.plot(episodes, v_totals, linewidth=2, color='purple')
        ax1.axhline(y=0.05, color='green', linestyle='--', 
                   label='Tier 1 threshold (5%)')
        ax1.axhline(y=0.25, color='orange', linestyle='--',
                   label='Tier 2 threshold (25%)')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('V_total', fontweight='bold')
        ax1.set_title('Total Violation Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Tier activations
        tier_colors = {1: 'green', 2: 'orange', 3: 'red'}
        colors = [tier_colors[t] for t in tiers]
        
        ax2.scatter(episodes, tiers, c=colors, s=50, alpha=0.6)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Tier', fontweight='bold')
        ax2.set_title('Tier Activations')
        ax2.set_yticks([1, 2, 3])
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_route_visualization(
        self,
        nodes: np.ndarray,
        routes: List[List[int]],
        depot_idx: int = 0,
        title: str = "Route Visualization"
    ):
        """
        Visualize routes on 2D plane (Figure 5.5 style).
        
        Args:
            nodes: Node coordinates [num_nodes, 2]
            routes: List of routes (each route is list of node indices)
            depot_idx: Depot node index
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot depot
        ax.scatter(nodes[depot_idx, 0], nodes[depot_idx, 1],
                  c='red', s=300, marker='s', label='Depot',
                  edgecolors='black', linewidths=2, zorder=3)
        
        # Plot customer nodes
        customer_nodes = np.delete(nodes, depot_idx, axis=0)
        ax.scatter(customer_nodes[:, 0], customer_nodes[:, 1],
                  c='blue', s=100, alpha=0.6, label='Customers',
                  edgecolors='black', linewidths=1, zorder=2)
        
        # Plot routes
        colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))
        
        for route_idx, route in enumerate(routes):
            color = colors[route_idx]
            
            # Add depot at start and end
            full_route = [depot_idx] + route + [depot_idx]
            route_coords = nodes[full_route]
            
            ax.plot(route_coords[:, 0], route_coords[:, 1],
                   color=color, linewidth=2, alpha=0.7,
                   label=f'Route {route_idx+1}', zorder=1)
            
            # Add arrows
            for i in range(len(route_coords) - 1):
                dx = route_coords[i+1, 0] - route_coords[i, 0]
                dy = route_coords[i+1, 1] - route_coords[i, 1]
                ax.arrow(route_coords[i, 0], route_coords[i, 1],
                        dx * 0.9, dy * 0.9,
                        head_width=2, head_length=1.5,
                        fc=color, ec=color, alpha=0.5, zorder=1)
        
        ax.set_xlabel('X Coordinate', fontweight='bold')
        ax.set_ylabel('Y Coordinate', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig
    
    def plot_ablation_study(
        self,
        component_names: List[str],
        performance_with: List[float],
        performance_without: List[float],
        title: str = "Ablation Study"
    ):
        """
        Plot ablation study results (Figure 5.6 style).
        
        Args:
            component_names: Names of components
            performance_with: Performance with component
            performance_without: Performance without component
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(component_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, performance_with, width,
                      label='With Component', color='green', alpha=0.7)
        bars2 = ax.bar(x + width/2, performance_without, width,
                      label='Without Component', color='red', alpha=0.7)
        
        ax.set_xlabel('Component', fontweight='bold')
        ax.set_ylabel('Cost ($)', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(component_names, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:.0f}',
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def _moving_average(self, data: List[float], window: int) -> np.ndarray:
        """Compute moving average."""
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def save_figure(self, filename: str, fig=None):
        """Save current or specified figure."""
        if fig is None:
            fig = plt.gcf()
        
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        print(f"Figure saved to {filepath}")
    
    def close_all(self):
        """Close all figures."""
        plt.close('all')


def create_paper_figures(
    training_data: Dict,
    benchmark_data: Dict,
    validation_data: Dict,
    output_dir: str = 'paper_figures/'
):
    """
    Create all figures for the paper.
    
    Generates Figures 5.1-5.6 matching paper style.
    
    Args:
        training_data: Training statistics
        benchmark_data: Benchmark results
        validation_data: Validation statistics
        output_dir: Output directory
    """
    viz = ResultVisualizer(output_dir=output_dir)
    
    print("Creating paper figures...")
    
    # Figure 5.1: Training curves
    fig1 = viz.plot_training_curve(
        rewards=training_data['rewards'],
        costs=training_data['costs'],
        title="Figure 5.1: Training Progress"
    )
    viz.save_figure('fig5_1_training_curves.pdf', fig1)
    
    # Figure 5.2: Method comparison
    fig2 = viz.plot_method_comparison(
        method_names=benchmark_data['method_names'],
        mean_costs=benchmark_data['mean_costs'],
        std_costs=benchmark_data['std_costs'],
        title="Figure 5.2: Method Comparison"
    )
    viz.save_figure('fig5_2_method_comparison.pdf', fig2)
    
    # Figure 5.3: Scalability
    fig3 = viz.plot_scalability(
        sizes=benchmark_data['sizes'],
        times_by_method=benchmark_data['times_by_method'],
        title="Figure 5.3: Computational Scalability"
    )
    viz.save_figure('fig5_3_scalability.pdf', fig3)
    
    # Figure 5.4: Violation analysis
    fig4 = viz.plot_violation_analysis(
        episodes=validation_data['episodes'],
        v_totals=validation_data['v_totals'],
        tiers=validation_data['tiers'],
        title="Figure 5.4: Constraint Violation Analysis"
    )
    viz.save_figure('fig5_4_violations.pdf', fig4)
    
    print(f"All figures saved to {output_dir}")
    
    viz.close_all()


if __name__ == '__main__':
    # Test visualization
    print("Testing Visualization...")
    
    viz = ResultVisualizer(output_dir='test_figures/')
    
    # Test training curve
    print("\n1. Training Curve:")
    rewards = 100 + np.cumsum(np.random.randn(1000) * 10)
    costs = 1000 - np.cumsum(np.random.randn(1000) * 5)
    
    fig1 = viz.plot_training_curve(rewards, costs)
    viz.save_figure('test_training.png', fig1)
    print("   ✓ Saved training curve")
    
    # Test method comparison
    print("\n2. Method Comparison:")
    methods = ['PPO-GNN', 'Classical-PPO', 'Clarke-Wright', 'Gurobi']
    means = [1050, 1150, 1250, 1000]
    stds = [50, 60, 80, 20]
    
    fig2 = viz.plot_method_comparison(methods, means, stds)
    viz.save_figure('test_comparison.png', fig2)
    print("   ✓ Saved method comparison")
    
    # Test route visualization
    print("\n3. Route Visualization:")
    nodes = np.random.rand(20, 2) * 100
    routes = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]
    
    fig3 = viz.plot_route_visualization(nodes, routes)
    viz.save_figure('test_routes.png', fig3)
    print("   ✓ Saved route visualization")
    
    viz.close_all()
    
    print("\n✓ All visualization tests passed!")
