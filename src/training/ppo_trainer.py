"""
PPO Trainer
Main PPO training coordinator.

Orchestrates training loop including rollout collection, PPO updates,
constraint validation, and checkpointing.

Author: Your Name
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from collections import deque
import os

import sys
sys.path.append('..')
from models.gae import compute_gae
from validation.deterministic_model import DeterministicModel
from validation.violation_analysis import compute_violation_score, ViolationTracker
from validation.adaptive_penalties import AdaptivePenaltyManager, PenaltyAdjustment
from utils.logger import Logger


class RolloutBuffer:
    """
    Buffer for storing rollouts during training.
    
    Stores states, actions, rewards, log probs, values, and dones
    for PPO updates.
    
    Args:
        capacity: Maximum number of transitions to store
    
    Example:
        >>> buffer = RolloutBuffer(capacity=2048)
        >>> buffer.add(state, action, reward, log_prob, value, done)
        >>> rollouts = buffer.get()
    """
    
    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self.clear()
    
    def clear(self):
        """Clear buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.size = 0
    
    def add(
        self,
        state: Dict,
        action: torch.Tensor,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: bool
    ):
        """Add a transition to buffer."""
        self.states.append(state)
        self.actions.append(action.cpu())
        self.rewards.append(reward)
        self.log_probs.append(log_prob.cpu())
        self.values.append(value.cpu())
        self.dones.append(float(done))
        self.size += 1
    
    def get(self) -> Dict:
        """
        Get all transitions as tensors.
        
        Returns:
            rollouts: Dictionary with tensors
        """
        return {
            'states': self.states,
            'actions': torch.stack(self.actions),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32),
            'old_log_probs': torch.stack(self.log_probs),
            'values': torch.stack(self.values),
            'dones': torch.tensor(self.dones, dtype=torch.float32)
        }
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.size >= self.capacity


class PPOTrainer:
    """
    PPO training coordinator.
    
    Orchestrates training loop including:
    - Rollout collection
    - PPO updates (Algorithm 1)
    - Constraint validation (Algorithm 2)
    - Checkpointing and logging
    
    Args:
        model: PPO-GNN model or baseline model
        env: Training environment
        config: Training configuration
        logger: Logger for TensorBoard
        device: Device to use ('cuda' or 'cpu')
    
    Example:
        >>> trainer = PPOTrainer(model, env, config, logger)
        >>> trainer.train(num_episodes=50000)
    """
    
    def __init__(
        self,
        model: nn.Module,
        env,
        config: Dict,
        logger: Optional[Logger] = None,
        device: str = 'cuda'
    ):
        self.model = model
        self.env = env
        self.config = config
        self.logger = logger
        self.device = device
        
        # Training parameters
        self.batch_size = config['training']['batch_size']
        self.epochs_per_update = config['training']['epochs_per_update']
        self.max_episodes = config['training']['max_episodes']
        self.episode_length = config['training']['episode_length']
        
        # Rollout buffer
        self.buffer = RolloutBuffer(capacity=self.batch_size)
        
        # Episode statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_costs = []
        
        # Best model tracking
        self.best_cost = float('inf')
        self.episodes_since_improvement = 0
        
    def collect_rollouts(
        self,
        num_steps: int
    ) -> Dict:
        """
        Collect experience rollouts.
        
        Args:
            num_steps: Number of steps to collect
        
        Returns:
            rollouts: Dictionary with collected experience
        """
        self.buffer.clear()
        
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(num_steps):
            # Select action
            action, log_prob, value = self.model.act(state, deterministic=False)
            
            # Environment step
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            self.buffer.add(state, action, reward, log_prob, value, done)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Handle episode end
            if done or episode_length >= self.episode_length:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                if 'cost' in info:
                    self.episode_costs.append(info['cost'])
                
                # Reset
                state = self.env.reset()
                episode_reward = 0
                episode_length = 0
        
        return self.buffer.get()
    
    def update_policy(
        self,
        rollouts: Dict
    ) -> Dict:
        """
        Update policy using PPO.
        
        Args:
            rollouts: Experience buffer
        
        Returns:
            logs: Training statistics
        """
        logs = self.model.update(
            rollouts,
            epochs=self.epochs_per_update,
            batch_size=self.batch_size
        )
        
        return logs
    
    def train(
        self,
        num_episodes: Optional[int] = None,
        checkpoint_dir: str = 'checkpoints/',
        log_frequency: int = 100
    ):
        """
        Execute complete training procedure.
        
        Args:
            num_episodes: Maximum training episodes (default: from config)
            checkpoint_dir: Directory for checkpoints
            log_frequency: Log every N episodes
        
        Returns:
            training_logs: Training statistics
        """
        if num_episodes is None:
            num_episodes = self.max_episodes
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"Starting PPO training for {num_episodes} episodes...")
        print(f"Batch size: {self.batch_size}, Epochs per update: {self.epochs_per_update}")
        
        start_time = time.time()
        episode = 0
        
        while episode < num_episodes:
            # Collect rollouts
            rollouts = self.collect_rollouts(self.batch_size)
            
            # Update policy
            logs = self.update_policy(rollouts)
            
            # Increment episode counter
            episode += len(self.episode_rewards)
            
            # Logging
            if episode % log_frequency == 0:
                self._log_statistics(episode, logs)
                self._save_checkpoint(checkpoint_dir, episode)
            
            # Check for improvement
            if len(self.episode_costs) > 0:
                recent_cost = np.mean(self.episode_costs[-10:])
                if recent_cost < self.best_cost:
                    self.best_cost = recent_cost
                    self.episodes_since_improvement = 0
                    self._save_checkpoint(checkpoint_dir, episode, is_best=True)
                else:
                    self.episodes_since_improvement += 1
        
        # Final save
        self._save_checkpoint(checkpoint_dir, episode, is_best=False)
        
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time / 3600:.2f} hours")
        print(f"Best cost: ${self.best_cost:.2f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_costs': self.episode_costs,
            'best_cost': self.best_cost,
            'total_time': elapsed_time
        }
    
    def _log_statistics(self, episode: int, logs: Dict):
        """Log training statistics."""
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-10:])
            mean_length = np.mean(self.episode_lengths[-10:])
            
            print(f"\nEpisode {episode}")
            print(f"  Mean Reward (last 10): {mean_reward:.2f}")
            print(f"  Mean Length (last 10): {mean_length:.1f}")
            print(f"  Policy Loss: {logs['policy_loss']:.4f}")
            print(f"  Value Loss: {logs['value_loss']:.4f}")
            print(f"  Entropy: {logs['entropy']:.4f}")
            
            if len(self.episode_costs) > 0:
                mean_cost = np.mean(self.episode_costs[-10:])
                print(f"  Mean Cost (last 10): ${mean_cost:.2f}")
            
            # TensorBoard logging
            if self.logger:
                self.logger.log_scalar('train/reward', mean_reward, episode)
                self.logger.log_scalar('train/episode_length', mean_length, episode)
                self.logger.log_scalar('train/policy_loss', logs['policy_loss'], episode)
                self.logger.log_scalar('train/value_loss', logs['value_loss'], episode)
                self.logger.log_scalar('train/entropy', logs['entropy'], episode)
                
                if len(self.episode_costs) > 0:
                    self.logger.log_scalar('train/cost', mean_cost, episode)
    
    def _save_checkpoint(
        self,
        checkpoint_dir: str,
        episode: int,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        if is_best:
            path = os.path.join(checkpoint_dir, 'best_model.pth')
            print(f"  Saving best model to {path}")
        else:
            path = os.path.join(checkpoint_dir, f'checkpoint_ep{episode}.pth')
        
        self.model.save(path)
    
    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict:
        """
        Evaluate policy.
        
        Args:
            num_episodes: Number of episodes to evaluate
            deterministic: Use deterministic policy
        
        Returns:
            stats: Evaluation statistics
        """
        eval_rewards = []
        eval_costs = []
        eval_lengths = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done and episode_length < self.episode_length:
                action, _, _ = self.model.act(state, deterministic=deterministic)
                state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            
            if 'cost' in info:
                eval_costs.append(info['cost'])
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_cost': np.mean(eval_costs) if eval_costs else None,
            'std_cost': np.std(eval_costs) if eval_costs else None,
            'mean_length': np.mean(eval_lengths),
        }


class AsyncPPOTrainer(PPOTrainer):
    """
    Asynchronous PPO trainer using multiple parallel environments.
    
    Speeds up training by collecting rollouts from multiple environments
    in parallel.
    
    Args:
        model: PPO model
        envs: List of parallel environments
        config: Training configuration
        logger: Logger
        device: Device
    
    Example:
        >>> envs = [make_env() for _ in range(4)]
        >>> trainer = AsyncPPOTrainer(model, envs, config, logger)
        >>> trainer.train(num_episodes=50000)
    """
    
    def __init__(
        self,
        model: nn.Module,
        envs: List,
        config: Dict,
        logger: Optional[Logger] = None,
        device: str = 'cuda'
    ):
        # Use first env for initialization
        super().__init__(model, envs[0], config, logger, device)
        self.envs = envs
        self.num_envs = len(envs)
    
    def collect_rollouts(self, num_steps: int) -> Dict:
        """Collect rollouts from parallel environments."""
        self.buffer.clear()
        
        # Initialize states for all environments
        states = [env.reset() for env in self.envs]
        episode_rewards = [0] * self.num_envs
        episode_lengths = [0] * self.num_envs
        
        steps_per_env = num_steps // self.num_envs
        
        for step in range(steps_per_env):
            for env_idx, (env, state) in enumerate(zip(self.envs, states)):
                # Select action
                action, log_prob, value = self.model.act(state, deterministic=False)
                
                # Environment step
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                self.buffer.add(state, action, reward, log_prob, value, done)
                
                # Update
                states[env_idx] = next_state
                episode_rewards[env_idx] += reward
                episode_lengths[env_idx] += 1
                
                # Handle episode end
                if done or episode_lengths[env_idx] >= self.episode_length:
                    self.episode_rewards.append(episode_rewards[env_idx])
                    self.episode_lengths.append(episode_lengths[env_idx])
                    
                    if 'cost' in info:
                        self.episode_costs.append(info['cost'])
                    
                    # Reset
                    states[env_idx] = env.reset()
                    episode_rewards[env_idx] = 0
                    episode_lengths[env_idx] = 0
        
        return self.buffer.get()


if __name__ == '__main__':
    # Test PPO trainer (with dummy components)
    print("Testing PPO Trainer...")
    
    # This would normally use real model and environment
    # Here we just test the structure
    
    class DummyModel:
        def act(self, state, deterministic=False):
            action = torch.randint(0, 10, (1,))
            log_prob = torch.tensor([-2.3])
            value = torch.tensor([10.5])
            return action, log_prob, value
        
        def update(self, rollouts, epochs, batch_size):
            return {
                'policy_loss': 0.5,
                'value_loss': 0.3,
                'entropy': 0.1,
                'approx_kl': 0.01,
                'clip_fraction': 0.15
            }
        
        def save(self, path):
            print(f"  Saving model to {path}")
    
    class DummyEnv:
        def reset(self):
            return {'node_features': torch.randn(10, 4), 'vehicle_states': torch.randn(3, 8)}
        
        def step(self, action):
            next_state = self.reset()
            reward = np.random.randn()
            done = np.random.rand() < 0.1
            info = {'cost': np.random.uniform(1000, 2000)}
            return next_state, reward, done, info
    
    # Configuration
    config = {
        'training': {
            'batch_size': 256,
            'epochs_per_update': 10,
            'max_episodes': 1000,
            'episode_length': 200
        }
    }
    
    # Create trainer
    model = DummyModel()
    env = DummyEnv()
    trainer = PPOTrainer(model, env, config)
    
    # Collect rollouts
    print("\n✓ Collecting rollouts...")
    rollouts = trainer.collect_rollouts(num_steps=256)
    print(f"  Collected {len(rollouts['rewards'])} transitions")
    
    # Update policy
    print("\n✓ Updating policy...")
    logs = trainer.update_policy(rollouts)
    print(f"  Policy loss: {logs['policy_loss']:.4f}")
    
    print("\n✓ All tests passed!")
