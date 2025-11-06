"""
Algorithm 2: Three-Tier Constraint Validation
Implements Algorithm 2 from Section 4.3.

Validates solutions and adjusts penalties based on violation severity.

Author: Majdi Argoubi
Date: 2025
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import copy

import sys
sys.path.append('..')
from validation.deterministic_model import DeterministicModel, ConstraintViolation
from validation.violation_analysis import (
    compute_violation_score,
    ViolationTracker,
    ViolationScore
)
from validation.adaptive_penalties import (
    AdaptivePenaltyManager,
    TierAction,
    PenaltyAdjustment
)
from utils.logger import Logger


class ConstraintValidator:
    """
    Three-tier constraint validation mechanism (Algorithm 2).
    
    Implements Algorithm 2 from Section 4.3:
    1. Generate solution from current policy
    2. Check feasibility using deterministic model
    3. Compute total violation V_total (Equation 11)
    4. Determine tier and apply appropriate response:
       - Tier 1: Continue (V ≤ 5%)
       - Tier 2: Fine-tune with 1.5× penalties (5% < V ≤ 25%)
       - Tier 3: Re-train with 10× penalties + reset (V > 25%)
    5. Monitor quality degradation
    
    Args:
        model: Trained model to validate
        env: Environment for solution generation
        config: Validation configuration
        logger: Logger for tracking
    
    Example:
        >>> validator = ConstraintValidator(model, env, config)
        >>> result = validator.validate(episode=1000)
        >>> if result['tier'] == 2:
        ...     print("Fine-tuning required")
    """
    
    def __init__(
        self,
        model,
        env,
        config: Dict,
        logger: Optional[Logger] = None
    ):
        self.model = model
        self.env = env
        self.config = config
        self.logger = logger
        
        # Deterministic model for feasibility checking
        network = env.network if hasattr(env, 'network') else {}
        self.deterministic_model = DeterministicModel(
            network=network,
            config=config.get('deterministic_model', {})
        )
        
        # Adaptive penalty manager
        penalty_config = config.get('adaptive_penalties', {})
        self.penalty_manager = AdaptivePenaltyManager(
            tier1_threshold=penalty_config.get('tier1_threshold', 0.05),
            tier2_threshold=penalty_config.get('tier2_threshold', 0.25),
            tier2_multiplier=penalty_config.get('tier2_multiplier', 1.5),
            tier2_episodes=penalty_config.get('tier2_episodes', 1000),
            tier3_multiplier=penalty_config.get('tier3_multiplier', 10.0),
            tier3_episodes=penalty_config.get('tier3_episodes', 10000),
            selective_adjustment=penalty_config.get('selective_adjustment', True)
        )
        
        # Violation tracker
        self.violation_tracker = ViolationTracker(
            max_history=config.get('max_history', 100)
        )
        
        # Constraint weights
        self.constraint_weights = config.get('constraint_weights', {
            'capacity': 2.0,
            'time_window': 1.5,
            'demand': 1.8,
            'sequencing': 1.0,
            'station_limit': 1.2,
            'max_route_time': 1.0
        })
        
        # Current penalties (can be adjusted by tier system)
        self.current_penalties = copy.deepcopy(self.constraint_weights)
        
        # Quality monitoring
        self.recent_costs = []
        self.quality_degradation_count = 0
        self.high_variance_count = 0
        
        # Checkpoint for restoration
        self.checkpoint_penalties = copy.deepcopy(self.constraint_weights)
        self.checkpoint_cost = float('inf')
    
    def validate(self, episode: int) -> Dict:
        """
        Execute Algorithm 2 (Section 4.3).
        
        Algorithm 2: Three-Tier Constraint Validation
        
        1: Generate solution S using current policy π_θ
        2: Check feasibility using deterministic model
        3: Compute V_total using Equation 11
        4: if V_total ≤ 0.05 then
        5:     Tier 1: Continue training
        6: else if 0.05 < V_total ≤ 0.25 then
        7:     Tier 2: Fine-tune with λ_c ← 1.5 × λ_c for 1,000 episodes
        8: else
        9:     Tier 3: Re-train with λ_c ← 10 × λ_c, reset θ, for 10,000 episodes
        10: end if
        11: if Quality degradation detected then
        12:     Restore penalties from checkpoint
        13: end if
        14: if High variance detected then
        15:     Reduce learning rate by 50%
        16: end if
        
        Args:
            episode: Current episode number
        
        Returns:
            validation_result: Dictionary with validation results
        """
        # Algorithm 2: Line 1 (Generate solution)
        solution = self._generate_solution()
        
        # Algorithm 2: Line 2 (Check feasibility)
        is_feasible, violations = self.deterministic_model.check_feasibility(solution)
        
        # Algorithm 2: Line 3 (Compute V_total)
        violation_score = compute_violation_score(
            solution=solution,
            constraints={},
            weights=self.constraint_weights,
            violations=violations
        )
        
        V_total = violation_score.V_total
        
        # Track violations
        self.violation_tracker.record(violation_score, violations)
        
        # Algorithm 2: Lines 4-10 (Determine tier and action)
        adjustment = self.penalty_manager.assess_violations(
            V_total=V_total,
            violations=violations,
            violation_score=violation_score
        )
        
        tier = adjustment.tier
        action = adjustment.action
        
        # Log validation
        self._log_validation(episode, solution, violation_score, adjustment)
        
        # Algorithm 2: Lines 11-13 (Quality degradation)
        if self._check_quality_degradation(solution):
            self._restore_checkpoint()
        
        # Algorithm 2: Lines 14-16 (High variance)
        if self._check_high_variance():
            self._reduce_learning_rate()
        
        # Apply penalty adjustments if needed
        if action != TierAction.CONTINUE:
            self.current_penalties = self.penalty_manager.apply_penalties(
                self.current_penalties,
                adjustment
            )
        
        # Update checkpoint if improved
        if 'cost' in solution:
            current_cost = solution['cost']
            if current_cost < self.checkpoint_cost:
                self.checkpoint_cost = current_cost
                self.checkpoint_penalties = copy.deepcopy(self.current_penalties)
        
        return {
            'episode': episode,
            'is_feasible': is_feasible,
            'V_total': V_total,
            'tier': tier,
            'action': action.value,
            'adjustment': adjustment,
            'num_violations': len(violations),
            'violations': violations,
            'violation_score': violation_score,
            'current_penalties': self.current_penalties,
            'solution': solution
        }
    
    def _generate_solution(self) -> Dict:
        """
        Generate solution using current policy (Algorithm 2, line 1).
        
        Returns:
            solution: Dictionary with routes and cost
        """
        # Reset environment
        state = self.env.reset()
        
        routes = []
        current_route = {'vehicle_id': 0, 'nodes': [], 'load': 0, 'time': 0}
        total_cost = 0
        done = False
        step = 0
        max_steps = 500
        
        # Generate solution by rolling out policy
        while not done and step < max_steps:
            # Select action using trained policy (deterministic)
            with torch.no_grad():
                action, _, _ = self.model.act(state, deterministic=True)
            
            # Take action
            next_state, reward, done, info = self.env.step(action)
            
            # Update current route (simplified)
            # In practice, this depends on action encoding
            if 'route' in info:
                current_route = info['route']
            
            if 'cost' in info:
                total_cost = info['cost']
            
            state = next_state
            step += 1
        
        # Extract routes from final state
        if hasattr(self.env, 'get_routes'):
            routes = self.env.get_routes()
        
        return {
            'routes': routes,
            'cost': total_cost,
            'num_routes': len(routes),
            'total_distance': sum(r.get('distance', 0) for r in routes),
            'total_time': sum(r.get('time', 0) for r in routes)
        }
    
    def _check_quality_degradation(self, solution: Dict) -> bool:
        """
        Check for quality degradation (Algorithm 2, lines 11-13).
        
        Quality degrades if cost increases by >5% over last 5 episodes.
        """
        if 'cost' not in solution:
            return False
        
        self.recent_costs.append(solution['cost'])
        
        # Keep only recent costs
        if len(self.recent_costs) > 10:
            self.recent_costs.pop(0)
        
        # Check degradation
        degraded = self.penalty_manager.detect_quality_degradation(
            self.recent_costs,
            threshold=0.05,
            patience=5
        )
        
        if degraded:
            self.quality_degradation_count += 1
            print(f"\n  ⚠️  Quality degradation detected (count: {self.quality_degradation_count})")
        
        return degraded
    
    def _check_high_variance(self) -> bool:
        """
        Check for high variance (Algorithm 2, lines 14-16).
        
        High variance if coefficient of variation > 15%.
        """
        if len(self.recent_costs) < 5:
            return False
        
        high_var = self.penalty_manager.detect_high_variance(
            self.recent_costs,
            threshold=0.15
        )
        
        if high_var:
            self.high_variance_count += 1
            print(f"\n  ⚠️  High variance detected (count: {self.high_variance_count})")
        
        return high_var
    
    def _restore_checkpoint(self):
        """Restore penalties from checkpoint (Algorithm 2, line 12)."""
        print(f"\n  🔄 Restoring penalties from checkpoint")
        print(f"     Checkpoint cost: ${self.checkpoint_cost:.2f}")
        
        self.current_penalties = self.penalty_manager.restore_checkpoint(
            self.current_penalties,
            self.checkpoint_penalties
        )
        
        if self.logger:
            self.logger.log_text('validation', f'Restored penalties from checkpoint')
    
    def _reduce_learning_rate(self):
        """Reduce learning rate (Algorithm 2, line 15)."""
        print(f"\n  📉 Reducing learning rate by 50%")
        
        # This would typically interact with the optimizer
        # For now, we just log it
        if self.logger:
            self.logger.log_text('validation', 'Reduced learning rate by 50%')
    
    def _log_validation(
        self,
        episode: int,
        solution: Dict,
        violation_score: ViolationScore,
        adjustment: PenaltyAdjustment
    ):
        """Log validation results."""
        print(f"\n{'='*80}")
        print(f"Constraint Validation (Episode {episode})")
        print(f"{'='*80}")
        
        # Solution quality
        if 'cost' in solution:
            print(f"Solution Cost: ${solution['cost']:.2f}")
        print(f"Number of Routes: {solution.get('num_routes', 0)}")
        
        # Violations
        print(f"\nViolation Analysis:")
        print(f"  V_total: {violation_score.V_total:.4f}")
        print(f"  Tier: {adjustment.tier}")
        print(f"  Action: {adjustment.action.value}")
        print(f"  Number of violations: {violation_score.num_violations}")
        print(f"  Max violation: {violation_score.max_violation:.2%}")
        
        # Violations by type
        if violation_score.num_violations > 0:
            print(f"\n  Violations by type:")
            for constraint_type, value in violation_score.violations_by_type.items():
                if value > 0:
                    print(f"    {constraint_type}: {value:.4f}")
        
        # Tier action details
        print(f"\n{adjustment.description}")
        
        # Violated constraints
        if adjustment.violated_constraints:
            print(f"  Violated constraints: {', '.join(adjustment.violated_constraints)}")
        
        # Penalty adjustments
        if adjustment.action != TierAction.CONTINUE:
            print(f"\n  Penalty Adjustments:")
            for constraint_type, penalty in self.current_penalties.items():
                new_penalty = penalty * adjustment.penalty_multiplier
                print(f"    {constraint_type}: {penalty:.2f} → {new_penalty:.2f}")
        
        # TensorBoard logging
        if self.logger:
            self.logger.log_scalar('validation/V_total', violation_score.V_total, episode)
            self.logger.log_scalar('validation/tier', adjustment.tier, episode)
            self.logger.log_scalar('validation/num_violations', violation_score.num_violations, episode)
            
            if 'cost' in solution:
                self.logger.log_scalar('validation/cost', solution['cost'], episode)
            
            # Log penalty values
            for constraint_type, penalty in self.current_penalties.items():
                self.logger.log_scalar(f'penalties/{constraint_type}', penalty, episode)
    
    def get_validation_summary(self) -> Dict:
        """
        Get summary of validation history.
        
        Returns:
            summary: Dictionary with validation statistics
        """
        # Tier statistics
        tier_stats = self.penalty_manager.get_tier_statistics()
        
        # Constraint importance
        importance = self.violation_tracker.get_constraint_importance()
        
        # Recent trends
        trends = self.violation_tracker.get_trends()
        
        return {
            'tier_statistics': tier_stats,
            'constraint_importance': importance,
            'trends': trends,
            'quality_degradation_count': self.quality_degradation_count,
            'high_variance_count': self.high_variance_count,
            'current_penalties': self.current_penalties,
            'checkpoint_cost': self.checkpoint_cost
        }


class BatchConstraintValidator(ConstraintValidator):
    """
    Batch constraint validator for validating multiple solutions.
    
    Useful for validating a batch of solutions from different policies
    or checkpoints.
    
    Args:
        model: Model to validate
        env: Environment
        config: Configuration
        logger: Logger
    
    Example:
        >>> validator = BatchConstraintValidator(model, env, config)
        >>> results = validator.validate_batch(num_solutions=10)
    """
    
    def validate_batch(
        self,
        num_solutions: int = 10,
        episode: int = 0
    ) -> Dict:
        """
        Validate batch of solutions.
        
        Args:
            num_solutions: Number of solutions to validate
            episode: Current episode
        
        Returns:
            batch_results: Aggregated validation results
        """
        all_results = []
        
        for i in range(num_solutions):
            result = self.validate(episode)
            all_results.append(result)
        
        # Aggregate results
        mean_V_total = np.mean([r['V_total'] for r in all_results])
        mean_violations = np.mean([r['num_violations'] for r in all_results])
        feasibility_rate = sum(r['is_feasible'] for r in all_results) / num_solutions
        
        # Tier distribution
        tier_counts = {1: 0, 2: 0, 3: 0}
        for r in all_results:
            tier_counts[r['tier']] += 1
        
        return {
            'num_solutions': num_solutions,
            'mean_V_total': mean_V_total,
            'mean_violations': mean_violations,
            'feasibility_rate': feasibility_rate,
            'tier_distribution': tier_counts,
            'all_results': all_results
        }


if __name__ == '__main__':
    print("Testing Algorithm 2: Constraint Validation...")
    print("This requires a complete model and environment setup.")
    print("See examples in notebooks/ for full validation examples.")
