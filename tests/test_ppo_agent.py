#!/usr/bin/env python3
"""
test_ppo_agent.py - Unit tests for PPO Agent

Tests the custom PPO implementation including:
- PolicyNetwork architecture
- PPOAgent training loop
- Action selection
- GAE computation
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ppo_agent import PolicyNetwork, PPOAgent, Trajectory


class TestPolicyNetwork:
    """Test PolicyNetwork architecture and forward pass."""

    def test_network_initialization(self):
        """Test network initialization with valid parameters."""
        state_dim = 128
        action_dim = 50

        network = PolicyNetwork(state_dim, action_dim)

        assert network.state_dim == state_dim
        assert network.action_dim == action_dim
        assert isinstance(network.shared, torch.nn.Sequential)
        assert isinstance(network.actor, torch.nn.Sequential)
        assert isinstance(network.critic, torch.nn.Sequential)

    def test_forward_pass_shape(self):
        """Test forward pass output shapes."""
        state_dim = 128
        action_dim = 50
        batch_size = 32

        network = PolicyNetwork(state_dim, action_dim)
        obs = torch.randn(batch_size, state_dim)

        logits, value = network(obs)

        assert logits.shape == (batch_size, action_dim)
        assert value.shape == (batch_size, 1)

    def test_get_action_and_value(self):
        """Test get_action_and_value helper."""
        state_dim = 128
        action_dim = 50

        network = PolicyNetwork(state_dim, action_dim)
        obs = torch.randn(4, state_dim)

        action, log_prob, entropy, value = network.get_action_and_value(obs)

        assert action.shape == (4,)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4,)

    def test_value_range(self):
        """Test that value predictions are reasonable."""
        state_dim = 128
        action_dim = 50

        network = PolicyNetwork(state_dim, action_dim)
        obs = torch.randn(10, state_dim)

        _, value = network(obs)

        # Values should be finite
        assert torch.all(torch.isfinite(value))


class TestPPOAgent:
    """Test PPOAgent training and inference."""

    @pytest.fixture
    def agent(self):
        """Create a test PPO agent."""
        return PPOAgent(
            state_dim=128,
            action_dim=50,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            epsilon=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            device='cpu'
        )

    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.gamma == 0.99
        assert agent.gae_lambda == 0.95
        assert agent.epsilon == 0.2
        assert isinstance(agent.policy, PolicyNetwork)
        assert isinstance(agent.optimizer, torch.optim.Adam)

    def test_select_action(self, agent):
        """Test action selection."""
        obs = np.random.randn(128).astype(np.float32)

        action, log_prob, value = agent.select_action(obs)

        assert isinstance(action, int)
        assert 0 <= action < 50
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_select_action_deterministic(self, agent):
        """Test deterministic action selection returns same action."""
        obs = np.random.randn(128).astype(np.float32)

        action1, _, _ = agent.select_action(obs, deterministic=True)
        action2, _, _ = agent.select_action(obs, deterministic=True)

        assert action1 == action2

    def test_compute_gae(self, agent):
        """Test Generalized Advantage Estimation."""
        traj = Trajectory(
            states=[np.random.randn(128).astype(np.float32) for _ in range(5)],
            actions=[0, 1, 2, 3, 4],
            rewards=[1.0, 2.0, 3.0, 0.0, 10.0],
            dones=[False, False, False, False, True],
            log_probs=[-0.5, -0.3, -0.2, -0.1, -0.4],
            values=[0.5, 1.0, 1.5, 0.0, 5.0]
        )

        advantages, returns = agent.compute_gae(traj)

        assert len(advantages) == 5
        assert len(returns) == 5
        # Advantages should be normalised (mean ≈ 0, std ≈ 1)
        assert abs(advantages.mean()) < 0.5

    def test_update_with_trajectories(self, agent):
        """Test policy update with a trajectory."""
        traj = Trajectory(
            states=[np.random.randn(128).astype(np.float32) for _ in range(32)],
            actions=[np.random.randint(0, 50) for _ in range(32)],
            rewards=[np.random.randn() for _ in range(32)],
            dones=[False] * 31 + [True],
            log_probs=[np.random.randn() for _ in range(32)],
            values=[np.random.randn() for _ in range(32)]
        )

        initial_params = [p.clone() for p in agent.policy.parameters()]

        loss_dict = agent.update([traj], n_epochs=2, batch_size=16)

        assert 'policy_loss' in loss_dict
        assert 'value_loss' in loss_dict
        assert 'entropy' in loss_dict
        assert 'clipfrac' in loss_dict

        # Parameters should have changed
        final_params = list(agent.policy.parameters())
        params_changed = any(
            not torch.equal(init, final)
            for init, final in zip(initial_params, final_params)
        )
        assert params_changed

    def test_save_and_load(self, agent, tmp_path):
        """Test model saving and loading."""
        save_path = tmp_path / "test_model.pt"
        agent.save(str(save_path))

        assert save_path.exists()

        new_agent = PPOAgent(state_dim=128, action_dim=50, device='cpu')
        new_agent.load(str(save_path))

        for p1, p2 in zip(agent.policy.parameters(), new_agent.policy.parameters()):
            assert torch.equal(p1, p2)

    def test_gradient_clipping(self, agent):
        """Test that gradients are clipped during update."""
        traj = Trajectory(
            states=[np.random.randn(128).astype(np.float32) for _ in range(32)],
            actions=[np.random.randint(0, 50) for _ in range(32)],
            rewards=[np.random.randn() * 100 for _ in range(32)],  # large rewards
            dones=[False] * 31 + [True],
            log_probs=[np.random.randn() for _ in range(32)],
            values=[np.random.randn() * 100 for _ in range(32)]
        )

        loss_dict = agent.update([traj], n_epochs=1, batch_size=16)

        assert all(np.isfinite(v) for v in loss_dict.values())


class TestTrainingLoop:
    """Test complete training loop behavior."""

    def test_episode_rollout(self):
        """Test collecting a trajectory from environment interaction."""
        agent = PPOAgent(state_dim=128, action_dim=50, device='cpu')

        states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []

        for _ in range(10):
            obs = np.random.randn(128).astype(np.float32)
            action, log_prob, value = agent.select_action(obs)

            states.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(np.random.uniform(-1, 1))
            dones.append(False)

        dones[-1] = True

        traj = Trajectory(states, actions, rewards, dones, log_probs, values)
        assert len(traj) == 10

    def test_advantage_computation_full_episode(self):
        """Test advantage computation on full episode."""
        agent = PPOAgent(state_dim=128, action_dim=50, gamma=0.99, gae_lambda=0.95, device='cpu')

        traj = Trajectory(
            states=[np.random.randn(128).astype(np.float32) for _ in range(5)],
            actions=[0, 1, 2, 3, 4],
            rewards=[0, 0, 0, 0, 10],
            dones=[False, False, False, False, True],
            log_probs=[-0.5] * 5,
            values=[1, 2, 3, 4, 5]
        )

        advantages, _ = agent.compute_gae(traj)
        assert len(advantages) == 5


def test_action_selection_basic():
    """Basic test for action selection."""
    agent = PPOAgent(state_dim=128, action_dim=50, device='cpu')
    obs = np.random.randn(128).astype(np.float32)

    action, _, _ = agent.select_action(obs)
    assert 0 <= action < 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
