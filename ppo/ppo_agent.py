"""
Custom PPO Agent Implementation (Priority 1, Item 1)

Implements Proximal Policy Optimization algorithm with:
- Actor-Critic architecture with 128-dim hidden layers
- Clipped surrogate objective (Equation 1 from paper)
- Generalized Advantage Estimation (GAE)
- Custom reward function (Equation 2 from paper)

Hyperparameters:
- Learning rate: 3e-4
- Gamma: 0.99
- Epsilon (clip): 0.2
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class Transition:
    """Single transition in trajectory"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


@dataclass
class Trajectory:
    """Complete trajectory (episode or rollout)"""
    states: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    dones: List[bool]
    log_probs: List[float]
    values: List[float]

    def __len__(self):
        return len(self.states)


class PolicyNetwork(nn.Module):
    """
    Actor-Critic Policy Network

    Architecture:
    - Input: state vector
    - Hidden: 2 layers x 128 units
    - Output: action probabilities (actor) + state value (critic)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize policy network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension (default: 128)
        """
        super(PolicyNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            Tuple of (action_logits, state_value)
        """
        features = self.shared(state)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        return action_logits, state_value

    def get_action_and_value(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple:
        """
        Get action, log probability, entropy, and value.

        Args:
            state: State tensor
            action: Optional action tensor (for evaluation)

        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


class PPOAgent:
    """
    Proximal Policy Optimization Agent

    Implements:
    - PPO clipped surrogate objective (Equation 1)
    - Generalized Advantage Estimation (GAE)
    - Custom reward function (Equation 2)
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 gae_lambda: float = 0.95,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize PPO agent.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate (default: 3e-4)
            gamma: Discount factor (default: 0.99)
            epsilon: PPO clip parameter (default: 0.2)
            gae_lambda: GAE lambda (default: 0.95)
            value_coef: Value loss coefficient (default: 0.5)
            entropy_coef: Entropy bonus coefficient (default: 0.01)
            max_grad_norm: Max gradient norm for clipping
            device: Device to use (cuda/cpu)
        """
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Create policy network
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Training statistics
        self.training_iterations = 0
        self.total_steps = 0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Select action from policy.

        Args:
            state: Current state
            deterministic: If True, select argmax action

        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            if deterministic:
                logits, value = self.policy(state_tensor)
                action = torch.argmax(logits, dim=-1)
                log_prob = torch.tensor(0.0)
            else:
                action, log_prob, _, value = self.policy.get_action_and_value(state_tensor)

            return int(action.item()), float(log_prob.item()), float(value.item())

    def compute_gae(self, trajectory: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.

        GAE formula:
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)

        Args:
            trajectory: Complete trajectory

        Returns:
            Tuple of (advantages, returns)
        """
        advantages_list: List[float] = []
        returns_list: List[float] = []

        gae = 0.0
        next_value = 0.0

        # Process trajectory in reverse
        for t in reversed(range(len(trajectory))):
            if t == len(trajectory) - 1:
                next_non_terminal = 1.0 - float(trajectory.dones[t])
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - float(trajectory.dones[t])
                next_value = float(trajectory.values[t + 1])

            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = float(trajectory.rewards[t]) + self.gamma * next_value * next_non_terminal - float(trajectory.values[t])

            # GAE: A_t = δ_t + (γλ)A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae

            advantages_list.insert(0, gae)
            returns_list.insert(0, gae + float(trajectory.values[t]))

        advantages_arr = np.array(advantages_list, dtype=np.float32)
        returns_arr = np.array(returns_list, dtype=np.float32)

        # Normalize advantages
        advantages_arr = (advantages_arr - advantages_arr.mean()) / (advantages_arr.std() + 1e-8)

        return advantages_arr, returns_arr

    def update(self, trajectories: List[Trajectory], n_epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        """
        Update policy using PPO clipped objective.

        PPO objective (Equation 1):
        L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
        where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

        Args:
            trajectories: List of trajectories to train on
            n_epochs: Number of training epochs
            batch_size: Minibatch size

        Returns:
            Dictionary of training metrics
        """
        # Flatten trajectories into single dataset
        all_states: List[np.ndarray] = []
        all_actions: List[int] = []
        all_old_log_probs: List[float] = []
        all_advantages: List[float] = []
        all_returns: List[float] = []

        for traj in trajectories:
            advs, rets = self.compute_gae(traj)
            all_states.extend(traj.states)
            all_actions.extend(traj.actions)
            all_old_log_probs.extend(traj.log_probs)
            all_advantages.extend(advs.tolist())
            all_returns.extend(rets.tolist())

        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(all_states)).to(self.device)
        actions_tensor = torch.LongTensor(all_actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(all_old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(all_advantages).to(self.device)
        returns_tensor = torch.FloatTensor(all_returns).to(self.device)

        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clipfrac = 0.0
        n_updates = 0

        # PPO epochs
        for epoch in range(n_epochs):
            # Mini-batch training
            indices = np.arange(len(states_tensor))
            np.random.shuffle(indices)

            for start in range(0, len(states_tensor), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Get current policy predictions
                _, new_log_probs, entropy, values = self.policy.get_action_and_value(
                    batch_states, batch_actions
                )

                # Compute ratio: π_θ(a|s) / π_θ_old(a|s)
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)

                # Compute clipped surrogate objective
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

                # Compute clip fraction (how often we're clipping)
                with torch.no_grad():
                    clipfrac = ((ratio - 1.0).abs() > self.epsilon).float().mean().item()
                    total_clipfrac += clipfrac

                n_updates += 1

        self.training_iterations += 1
        self.total_steps += len(states_tensor)

        # Return metrics
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'clipfrac': total_clipfrac / n_updates,
            'n_updates': n_updates,
            'training_iterations': self.training_iterations,
            'total_steps': self.total_steps
        }

    def compute_reward(self,
                      got_root: bool = False,
                      new_host_owned: bool = False,
                      credentials_found: int = 0,
                      safety_violation: bool = False,
                      step_penalty: float = -1.0) -> float:
        """
        Compute reward based on paper's Equation 2:
        R = +100 (root) + 10 (new host) + 5 (per credential) - 1 (step) - 100 (safety violation)

        Args:
            got_root: Whether root access was obtained
            new_host_owned: Whether a new host was compromised
            credentials_found: Number of credentials found
            safety_violation: Whether a safety constraint was violated
            step_penalty: Penalty per step (default: -1)

        Returns:
            Total reward value
        """
        reward = step_penalty

        if got_root:
            reward += 100.0

        if new_host_owned:
            reward += 10.0

        reward += credentials_found * 5.0

        if safety_violation:
            reward -= 100.0

        return reward

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_iterations': self.training_iterations,
            'total_steps': self.total_steps
        }, path)

    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_iterations = checkpoint['training_iterations']
        self.total_steps = checkpoint['total_steps']


class RolloutBuffer:
    """Buffer for storing trajectories during rollout"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.trajectories: deque = deque(maxlen=max_size)

    def add_trajectory(self, trajectory: Trajectory):
        """Add a complete trajectory"""
        self.trajectories.append(trajectory)

    def get_trajectories(self) -> List[Trajectory]:
        """Get all stored trajectories"""
        return list(self.trajectories)

    def clear(self):
        """Clear buffer"""
        self.trajectories.clear()

    def __len__(self):
        return len(self.trajectories)
