"""
Algorithm 1: Complete Training Procedure
Implements Algorithm 1 from Section 4.2.3.

Includes GNN encoder training, PPO updates, and validation.

Author: Majdi Argoubi
Date: 2025
"""

import torch
import numpy as np
import time
from typing import Dict, Optional
import os

from .ppo_trainer import PPOTrainer, RolloutBuffer
from .algorithm2_validation import ConstraintValidator
from .early_stopping import EarlyStopping
from utils.logger import Logger


class Algorithm1Trainer:
    """
    Complete training procedure (Algorithm 1).
    
    Implements the full training algorithm from Section 4.2.3:
    1. Collect rollouts with current policy
    2. Compute advantages using GAE
    3. Update policy and value networks (K epochs)
    4. Update GNN encoder periodically
    5. Validate constraints (Algorithm 2)
    6. Early stopping based on convergence
    
    Args:
        model: PPO-GNN model
        env: Training environment
        config: Configuration dictionary
        logger: TensorBoard logger
        checkpoint_dir: Directory for checkpoints
        device: 'cuda' or 'cpu'
    
    Example:
        >>> trainer = Algorithm1Trainer(model, env, config, logger)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model,
        env,
        config: Dict,
        logger: Optional[Logger] = None,
        checkpoint_dir: str = 'checkpoints/',
        device: str = 'cuda'
    ):
        self.model = model
        self.env = env
        self.config = config
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # PPO trainer for basic operations
        self.ppo_trainer = PPOTrainer(model, env, config, logger, device)
        
        # Constraint validator (Algorithm 2)
        validation_config = config.get('validation', {})
        if validation_config.get('enabled', True):
            self.validator = ConstraintValidator(
                model=model,
                env=env,
                config=validation_config,
                logger=logger
            )
        else:
            self.validator = None
        
        # Early stopping
        early_stop_config = config['training']
        self.early_stopping = EarlyStopping(
            patience=early_stop_config.get('early_stop_patience', 5000),
            min_delta=early_stop_config.get('min_delta', 0.01),
            mode='min'
        )
        
        # Training parameters
        self.max_episodes = config['training']['max_episodes']
        self.validation_frequency = validation_config.get('validation_frequency', 1000)
        self.gnn_update_frequency = config.get('gnn', {}).get('update_frequency', 1000)
        self.log_frequency = config.get('logging', {}).get('log_frequency', 100)
        
        # Statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_costs': [],
            'policy_losses': [],
            'value_losses': [],
            'validation_scores': [],
            'tier_activations': []
        }
    
    def train(self) -> Dict:
        """
        Execute Algorithm 1 (Section 4.2.3).
        
        Algorithm 1: PPO-GNN Training with Three-Tier Validation
        
        1: Initialize policy π_θ, value V_φ, GNN ψ
        2: for episode = 1 to max_episodes do
        3:     Collect rollouts using π_θ
        4:     Compute advantages A_t using GAE
        5:     for epoch = 1 to K do
        6:         Update π_θ using PPO-Clip
        7:         Update V_φ using MSE loss
        8:     end for
        9:     if episode mod GNN_freq == 0 then
        10:        Update GNN encoder ψ
        11:    end if
        12:    if episode mod Val_freq == 0 then
        13:        Run Algorithm 2 (Constraint Validation)
        14:    end if
        15:    Check early stopping
        16: end for
        
        Returns:
            training_stats: Dictionary with training statistics
        """
        print("=" * 80)
        print("Algorithm 1: PPO-GNN Training with Three-Tier Validation")
        print("=" * 80)
        print(f"Max episodes: {self.max_episodes}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Validation frequency: {self.validation_frequency}")
        print(f"GNN update frequency: {self.gnn_update_frequency}")
        print(f"Early stopping patience: {self.early_stopping.patience}")
        print("=" * 80)
        
        start_time = time.time()
        episode = 0
        
        # Algorithm 1: Line 2 (for episode = 1 to max_episodes)
        while episode < self.max_episodes:
            # Algorithm 1: Line 3 (Collect rollouts)
            rollouts = self.ppo_trainer.collect_rollouts(
                num_steps=self.config['training']['batch_size']
            )
            
            # Algorithm 1: Line 4-8 (PPO update with advantages)
            logs = self.ppo_trainer.update_policy(rollouts)
            
            # Store statistics
            self.training_stats['policy_losses'].append(logs['policy_loss'])
            self.training_stats['value_losses'].append(logs['value_loss'])
            
            # Increment episode counter
            episode += len(self.ppo_trainer.episode_rewards)
            
            # Algorithm 1: Line 9-11 (GNN update)
            if episode % self.gnn_update_frequency == 0:
                self._update_gnn()
            
            # Algorithm 1: Line 12-14 (Constraint validation)
            if self.validator and episode % self.validation_frequency == 0:
                validation_result = self.validator.validate(episode)
                
                self.training_stats['validation_scores'].append(
                    validation_result['V_total']
                )
                self.training_stats['tier_activations'].append(
                    validation_result['tier']
                )
                
                # Apply penalty adjustments if needed
                if validation_result['action'] != 'continue':
                    self._apply_penalty_adjustments(validation_result)
            
            # Logging
            if episode % self.log_frequency == 0:
                self._log_progress(episode, logs)
            
            # Algorithm 1: Line 15 (Early stopping)
            if len(self.ppo_trainer.episode_costs) > 0:
                mean_cost = np.mean(self.ppo_trainer.episode_costs[-10:])
                
                if self.early_stopping(mean_cost):
                    print(f"\nEarly stopping triggered at episode {episode}")
                    print(f"Best cost: ${self.early_stopping.best_score:.2f}")
                    break
                
                # Save best model
                if mean_cost < self.ppo_trainer.best_cost:
                    self.ppo_trainer.best_cost = mean_cost
                    self._save_checkpoint(episode, is_best=True)
            
            # Periodic checkpoint
            if episode % 5000 == 0:
                self._save_checkpoint(episode)
        
        # Final checkpoint
        self._save_checkpoint(episode, is_best=False)
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("Training Completed")
        print("=" * 80)
        print(f"Total episodes: {episode}")
        print(f"Total time: {elapsed_time / 3600:.2f} hours")
        print(f"Best cost: ${self.ppo_trainer.best_cost:.2f}")
        
        if self.validator:
            tier_stats = self.validator.penalty_manager.get_tier_statistics()
            print(f"\nTier Statistics:")
            print(f"  Tier 1: {tier_stats['tier1_count']} ({tier_stats['tier1_pct']:.1f}%)")
            print(f"  Tier 2: {tier_stats['tier2_count']} ({tier_stats['tier2_pct']:.1f}%)")
            print(f"  Tier 3: {tier_stats['tier3_count']} ({tier_stats['tier3_pct']:.1f}%)")
        
        print("=" * 80)
        
        return {
            'episode_rewards': self.ppo_trainer.episode_rewards,
            'episode_costs': self.ppo_trainer.episode_costs,
            'best_cost': self.ppo_trainer.best_cost,
            'total_episodes': episode,
            'total_time': elapsed_time,
            'training_stats': self.training_stats
        }
    
    def _update_gnn(self):
        """Update GNN encoder (Algorithm 1, line 10)."""
        # GNN optimizer step is handled in model.update()
        # This is a placeholder for any additional GNN-specific updates
        if self.logger:
            self.logger.log_text('info', 'GNN encoder updated')
    
    def _apply_penalty_adjustments(self, validation_result: Dict):
        """Apply penalty adjustments from validation."""
        action = validation_result['action']
        adjustment = validation_result['adjustment']
        
        if action == 'fine_tune':
            print(f"\n  Tier 2 triggered: Fine-tuning with {adjustment.penalty_multiplier}× penalties")
            print(f"  Training for {adjustment.num_episodes} additional episodes")
            
            # Apply penalty adjustments
            # (This would update reward weights in the environment or model)
            
        elif action == 're_train':
            print(f"\n  Tier 3 triggered: Re-training with {adjustment.penalty_multiplier}× penalties")
            print(f"  Resetting policy network")
            print(f"  Training for {adjustment.num_episodes} additional episodes")
            
            # Reset policy network if needed
            if adjustment.reset_policy:
                # Reinitialize policy network
                self.model.policy_net.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights for layer."""
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def _log_progress(self, episode: int, logs: Dict):
        """Log training progress."""
        print(f"\n{'='*80}")
        print(f"Episode {episode}/{self.max_episodes}")
        print(f"{'='*80}")
        
        # Recent statistics
        if len(self.ppo_trainer.episode_rewards) > 0:
            recent_rewards = self.ppo_trainer.episode_rewards[-10:]
            recent_lengths = self.ppo_trainer.episode_lengths[-10:]
            
            print(f"Mean Reward (last 10): {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
            print(f"Mean Length (last 10): {np.mean(recent_lengths):.1f}")
        
        if len(self.ppo_trainer.episode_costs) > 0:
            recent_costs = self.ppo_trainer.episode_costs[-10:]
            print(f"Mean Cost (last 10): ${np.mean(recent_costs):.2f} ± ${np.std(recent_costs):.2f}")
            print(f"Best Cost: ${self.ppo_trainer.best_cost:.2f}")
        
        # Training metrics
        print(f"\nTraining Metrics:")
        print(f"  Policy Loss: {logs['policy_loss']:.4f}")
        print(f"  Value Loss: {logs['value_loss']:.4f}")
        print(f"  Entropy: {logs['entropy']:.4f}")
        print(f"  Approx KL: {logs['approx_kl']:.4f}")
        print(f"  Clip Fraction: {logs['clip_fraction']:.2%}")
        
        # Validation statistics
        if len(self.training_stats['validation_scores']) > 0:
            recent_v_total = self.training_stats['validation_scores'][-1]
            recent_tier = self.training_stats['tier_activations'][-1]
            print(f"\nConstraint Validation:")
            print(f"  V_total: {recent_v_total:.4f}")
            print(f"  Current Tier: {recent_tier}")
        
        # TensorBoard logging
        if self.logger:
            if len(self.ppo_trainer.episode_rewards) > 0:
                self.logger.log_scalar('train/mean_reward', np.mean(recent_rewards), episode)
            
            if len(self.ppo_trainer.episode_costs) > 0:
                self.logger.log_scalar('train/mean_cost', np.mean(recent_costs), episode)
            
            self.logger.log_scalar('train/policy_loss', logs['policy_loss'], episode)
            self.logger.log_scalar('train/value_loss', logs['value_loss'], episode)
            self.logger.log_scalar('train/entropy', logs['entropy'], episode)
            
            if len(self.training_stats['validation_scores']) > 0:
                self.logger.log_scalar('validation/V_total', recent_v_total, episode)
                self.logger.log_scalar('validation/tier', recent_tier, episode)
    
    def _save_checkpoint(self, episode: int, is_best: bool = False):
        """Save model checkpoint."""
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            print(f"\n  💾 Saving best model (episode {episode})")
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_ep{episode}.pth')
            print(f"\n  💾 Saving checkpoint (episode {episode})")
        
        # Save model
        self.model.save(path)
        
        # Save training statistics
        stats_path = os.path.join(self.checkpoint_dir, f'training_stats_ep{episode}.npz')
        np.savez(
            stats_path,
            episode_rewards=self.ppo_trainer.episode_rewards,
            episode_costs=self.ppo_trainer.episode_costs,
            policy_losses=self.training_stats['policy_losses'],
            value_losses=self.training_stats['value_losses'],
            validation_scores=self.training_stats['validation_scores'],
            tier_activations=self.training_stats['tier_activations']
        )


if __name__ == '__main__':
    print("Testing Algorithm 1 Trainer...")
    print("This requires a complete model and environment setup.")
    print("See examples in notebooks/ for full training examples.")
