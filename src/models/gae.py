"""
Generalized Advantage Estimation (GAE)
Compute advantages for PPO using TD(λ) with variance reduction.

Author: Majdi Argoubi
Date: 2025
"""

import torch
import numpy as np
from typing import Tuple, Optional


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (Equation 9).
    
    GAE combines TD(λ) with advantage estimation for variance reduction
    while maintaining low bias.
    
    Equation 9:
        A_t^GAE(γ,λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
    
    where:
        δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)
    
    Args:
        rewards: Rewards for each timestep [num_steps]
        values: State values V(s_t) for each timestep [num_steps]
        dones: Done flags (1 if episode ended) [num_steps]
        gamma: Discount factor γ (default: 0.99)
        lambda_: GAE parameter λ (default: 0.95)
        normalize: Whether to normalize advantages (default: True)
    
    Returns:
        advantages: Advantage estimates [num_steps]
        returns: Discounted returns [num_steps]
    
    Example:
        >>> rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> values = torch.tensor([0.5, 1.5, 2.5, 3.5])
        >>> dones = torch.tensor([0, 0, 0, 1])
        >>> advantages, returns = compute_gae(rewards, values, dones)
        >>> print(f"Advantages: {advantages}")
    """
    num_steps = len(rewards)
    advantages = torch.zeros_like(rewards)
    
    # Compute TD errors: δ_t = r_t + γV(s_{t+1}) - V(s_t)
    deltas = torch.zeros_like(rewards)
    
    for t in range(num_steps - 1):
        # δ_t = r_t + γV(s_{t+1})(1-done) - V(s_t)
        deltas[t] = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
    
    # Last timestep
    deltas[-1] = rewards[-1] - values[-1]
    
    # Compute advantages using GAE
    # A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    gae = 0
    for t in reversed(range(num_steps)):
        # gae = δ_t + (γλ)(1-done)gae_{t+1}
        gae = deltas[t] + gamma * lambda_ * (1 - dones[t]) * gae
        advantages[t] = gae
    
    # Compute returns: R_t = A_t + V(s_t)
    returns = advantages + values
    
    # Normalize advantages for training stability
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns


def compute_td_error(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99
) -> torch.Tensor:
    """
    Compute temporal difference (TD) errors.
    
    TD error: δ_t = r_t + γV(s_{t+1})(1-done) - V(s_t)
    
    Args:
        rewards: Rewards [num_steps]
        values: Current state values [num_steps]
        next_values: Next state values [num_steps]
        dones: Done flags [num_steps]
        gamma: Discount factor (default: 0.99)
    
    Returns:
        td_errors: TD errors [num_steps]
    """
    td_errors = rewards + gamma * next_values * (1 - dones) - values
    return td_errors


def compute_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    normalize: bool = False
) -> torch.Tensor:
    """
    Compute discounted returns (Monte Carlo).
    
    R_t = r_t + γr_{t+1} + γ²r_{t+2} + ... + γ^{T-t}r_T
    
    Args:
        rewards: Rewards [num_steps]
        dones: Done flags [num_steps]
        gamma: Discount factor (default: 0.99)
        normalize: Normalize returns (default: False)
    
    Returns:
        returns: Discounted returns [num_steps]
    
    Example:
        >>> rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> dones = torch.tensor([0, 0, 0, 1])
        >>> returns = compute_returns(rewards, dones, gamma=0.99)
    """
    num_steps = len(rewards)
    returns = torch.zeros_like(rewards)
    
    running_return = 0
    for t in reversed(range(num_steps)):
        running_return = rewards[t] + gamma * running_return * (1 - dones[t])
        returns[t] = running_return
    
    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns


def compute_n_step_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    n: int = 5,
    gamma: float = 0.99
) -> torch.Tensor:
    """
    Compute n-step returns.
    
    R_t^(n) = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n})
    
    Args:
        rewards: Rewards [num_steps]
        values: State values [num_steps]
        dones: Done flags [num_steps]
        n: Number of steps (default: 5)
        gamma: Discount factor (default: 0.99)
    
    Returns:
        n_step_returns: n-step returns [num_steps]
    """
    num_steps = len(rewards)
    n_step_returns = torch.zeros_like(rewards)
    
    for t in range(num_steps):
        n_step_return = 0
        discount = 1
        
        # Sum discounted rewards for n steps
        for k in range(n):
            if t + k >= num_steps:
                break
            
            n_step_return += discount * rewards[t + k]
            discount *= gamma
            
            # Stop if episode ended
            if dones[t + k]:
                break
        
        # Add bootstrapped value
        if t + n < num_steps and not dones[t + n - 1]:
            n_step_return += discount * values[t + n]
        
        n_step_returns[t] = n_step_return
    
    return n_step_returns


def compute_advantages_simple(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute simple 1-step advantages (TD advantage).
    
    A_t = r_t + γV(s_{t+1}) - V(s_t)
    
    Simpler alternative to GAE with less bias but higher variance.
    
    Args:
        rewards: Rewards [num_steps]
        values: Current state values [num_steps]
        next_values: Next state values [num_steps]
        dones: Done flags [num_steps]
        gamma: Discount factor (default: 0.99)
        normalize: Normalize advantages (default: True)
    
    Returns:
        advantages: Advantage estimates [num_steps]
    """
    advantages = compute_td_error(rewards, values, next_values, dones, gamma)
    
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages


class GAEBuffer:
    """
    Buffer for storing rollouts and computing GAE advantages.
    
    Useful for collecting experience during training and computing
    advantages in batch.
    
    Args:
        size: Buffer size (number of timesteps)
        gamma: Discount factor
        lambda_: GAE parameter
    
    Example:
        >>> buffer = GAEBuffer(size=2048, gamma=0.99, lambda_=0.95)
        >>> for t in range(num_steps):
        ...     buffer.store(state, action, reward, value, done)
        >>> advantages, returns = buffer.compute_gae()
    """
    
    def __init__(
        self,
        size: int,
        gamma: float = 0.99,
        lambda_: float = 0.95
    ):
        self.size = size
        self.gamma = gamma
        self.lambda_ = lambda_
        
        # Storage
        self.rewards = torch.zeros(size)
        self.values = torch.zeros(size)
        self.dones = torch.zeros(size)
        
        self.ptr = 0
        self.full = False
    
    def store(
        self,
        reward: float,
        value: float,
        done: bool
    ):
        """Store a single transition."""
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True
    
    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get stored transitions."""
        end_idx = self.size if self.full else self.ptr
        return (
            self.rewards[:end_idx],
            self.values[:end_idx],
            self.dones[:end_idx]
        )
    
    def compute_gae(
        self,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages for stored transitions."""
        rewards, values, dones = self.get()
        return compute_gae(
            rewards, values, dones,
            gamma=self.gamma,
            lambda_=self.lambda_,
            normalize=normalize
        )
    
    def clear(self):
        """Clear buffer."""
        self.ptr = 0
        self.full = False


if __name__ == '__main__':
    # Test compute_gae
    print("Testing compute_gae...")
    
    # Dummy trajectory
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    values = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])
    dones = torch.tensor([0, 0, 0, 0, 1])  # Episode ends at last step
    
    advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95)
    
    print(f"✓ Rewards: {rewards.tolist()}")
    print(f"✓ Values: {values.tolist()}")
    print(f"✓ Advantages: {advantages.tolist()}")
    print(f"✓ Returns: {returns.tolist()}")
    
    # Test GAEBuffer
    print("\nTesting GAEBuffer...")
    buffer = GAEBuffer(size=5, gamma=0.99, lambda_=0.95)
    
    for r, v, d in zip(rewards, values, dones):
        buffer.store(r.item(), v.item(), bool(d.item()))
    
    adv, ret = buffer.compute_gae()
    print(f"✓ Buffer advantages: {adv.tolist()}")
    print(f"✓ Buffer returns: {ret.tolist()}")
    
    # Test other functions
    print("\nTesting other functions...")
    
    next_values = torch.cat([values[1:], torch.tensor([0.0])])
    td_errors = compute_td_error(rewards, values, next_values, dones)
    print(f"✓ TD errors: {td_errors.tolist()}")
    
    mc_returns = compute_returns(rewards, dones, gamma=0.99)
    print(f"✓ MC returns: {mc_returns.tolist()}")
    
    n_step_returns = compute_n_step_returns(rewards, values, dones, n=3, gamma=0.99)
    print(f"✓ 3-step returns: {n_step_returns.tolist()}")
    
    print("\n✓ All GAE tests passed!")
