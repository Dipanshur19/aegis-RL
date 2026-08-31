"""
CyberBattleSim Environment Wrapper for AUVAP-PPO Integration

This module provides a Gymnasium-compatible wrapper around Microsoft's CyberBattleSim
that integrates with the existing AUVAP vulnerability assessment pipeline.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import networkx as nx

# AUVAP imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import VAFinding
from task_manager import ExploitTask, TaskState
from feasibility_filter import compute_risk_score


@dataclass
class AUVAPConfig:
    """Configuration for AUVAP-CyberBattleSim integration"""
    max_steps: int = 100
    reward_success: float = 10.0
    reward_failure: float = -1.0
    reward_step: float = -0.1
    use_risk_score_reward: bool = True
    normalize_observations: bool = True


class AUVAPCyberBattleEnv(gym.Env):
    """
    Custom Gymnasium environment that bridges AUVAP vulnerability assessment
    with CyberBattleSim for reinforcement learning training.

    This environment:
    - Uses vulnerability findings from AUVAP as the basis for network configuration
    - Provides observations based on network state and available exploits
    - Rewards the agent based on successful exploitation and risk scores
    - Integrates with the existing ExploitTask system

    Observation Space:
        - Network topology features
        - Discovered nodes and their vulnerabilities
        - Available actions (exploits)
        - Current attacker position

    Action Space:
        - Discrete actions corresponding to available exploits
        - Each action maps to an ExploitTask

    Reward:
        - Positive reward for successful exploitation
        - Negative reward for failed attempts
        - Optional reward shaping using CVSS-based risk scores
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 vulnerability_findings: List[VAFinding] = None,
                 config: AUVAPConfig = None,
                 cyberbattle_env = None):
        """
        Initialize the AUVAP-CyberBattleSim environment.

        Args:
            vulnerability_findings: List of vulnerability findings from AUVAP parser
            config: Configuration parameters
            cyberbattle_env: Optional pre-configured CyberBattleSim environment
        """
        super().__init__()

        self.config = config or AUVAPConfig()
        self.vulnerability_findings = vulnerability_findings or []

        # Initialize the underlying CyberBattleSim environment
        if cyberbattle_env is not None:
            self.cyberbattle_env = cyberbattle_env
        else:
            self.cyberbattle_env = self._create_cyberbattle_env()

        # State tracking
        self.current_step = 0
        self.total_reward = 0.0
        self.exploited_vulns = set()
        self.available_tasks: List[ExploitTask] = []

        # Define action and observation spaces
        # Action space: discrete actions for each possible exploit
        self.action_space = gym.spaces.Discrete(self._get_max_actions())

        # Observation space: feature vector representing network state
        obs_dim = self._get_observation_dimension()
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )

    def _create_cyberbattle_env(self):
        """
        Create a CyberBattleSim environment from vulnerability findings.

        This method converts AUVAP vulnerability data into a CyberBattleSim
        network topology with appropriate vulnerabilities.
        """
        # For now, create a simple chain network
        # In production, this would be derived from actual vulnerability findings
        try:
            from cyberbattle.simulation import model, actions
            from cyberbattle._env.cyberbattle_env import CyberBattleEnv

            # Create a simple network topology
            # TODO: Build this from actual vulnerability findings
            network = nx.DiGraph()
            network.add_nodes_from(['client', 'web_server', 'db_server', 'file_server'])
            network.add_edges_from([
                ('client', 'web_server'),
                ('web_server', 'db_server'),
                ('web_server', 'file_server')
            ])

            # Create vulnerability library
            # TODO: Map VAFinding objects to CyberBattleSim vulnerabilities
            vulnerability_library = {}

            # Create and return the environment
            # This is a placeholder - actual implementation will use CyberBattleSim's API
            return None  # Placeholder for actual CyberBattleSim environment

        except ImportError:
            # Fallback if CyberBattleSim is not installed
            print("Warning: CyberBattleSim not installed. Using mock environment.")
            return None

    def _get_max_actions(self) -> int:
        """Get the maximum number of possible actions."""
        # Start with a reasonable default
        # This will be dynamically updated based on available exploits
        return 50

    def _get_observation_dimension(self) -> int:
        """Calculate the dimension of the observation space."""
        # Base features:
        # - Network topology features (adjacency, node features)
        # - Current position
        # - Available exploits
        # - Vulnerability states
        base_features = 20  # Basic network state
        vuln_features = len(self.vulnerability_findings) if self.vulnerability_findings else 10
        return base_features + vuln_features

    def _build_observation(self) -> np.ndarray:
        """
        Build the observation vector from the current environment state.

        Returns:
            numpy array representing the current state
        """
        obs_dim = self._get_observation_dimension()
        observation = np.zeros(obs_dim, dtype=np.float32)

        # TODO: Populate with actual state information
        # - Current node features
        # - Network topology information
        # - Available actions
        # - Discovered vulnerabilities

        # For now, return a placeholder observation
        observation[0] = self.current_step / self.config.max_steps  # Normalized step count

        if self.config.normalize_observations:
            observation = np.clip(observation, 0, 1)

        return observation

    def _calculate_reward(self, action_success: bool, exploit_info: Dict = None) -> float:
        """
        Calculate the reward for the current step.

        Args:
            action_success: Whether the action succeeded
            exploit_info: Information about the exploit attempt

        Returns:
            The calculated reward value
        """
        if action_success:
            reward = self.config.reward_success

            # Add risk-based reward shaping if enabled
            if self.config.use_risk_score_reward and exploit_info:
                # Use AUVAP's risk scoring to shape rewards
                finding_dict = {
                    'cvss': exploit_info.get('cvss_score', 5.0),
                    'attack_vector': exploit_info.get('attack_vector', 'Network'),
                    'automation_candidate': exploit_info.get('automation_candidate', True)
                }
                risk_score = compute_risk_score(finding_dict)
                # Scale risk score to reward magnitude
                reward += risk_score * 0.5
        else:
            reward = self.config.reward_failure

        # Add step penalty to encourage efficiency
        reward += self.config.reward_step

        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            Tuple of (observation, info dict)
        """
        super().reset(seed=seed)

        # Reset internal state
        self.current_step = 0
        self.total_reward = 0.0
        self.exploited_vulns = set()
        self.available_tasks = []

        # Reset CyberBattleSim environment if it exists
        if self.cyberbattle_env is not None:
            # Call CyberBattleSim's reset
            pass

        observation = self._build_observation()
        info = {
            'step': self.current_step,
            'exploited_count': 0,
            'available_actions': len(self.available_tasks)
        }

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: The action to take (discrete action index)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1

        # Execute the action in CyberBattleSim
        action_success = False
        exploit_info = {}

        if self.cyberbattle_env is not None:
            # Execute action in CyberBattleSim
            # obs, reward, done, info = self.cyberbattle_env.step(action)
            pass
        else:
            # Mock execution for testing
            action_success = np.random.random() > 0.5  # 50% success rate
            exploit_info = {'cvss_score': 7.5, 'attack_surface': 'remote'}

        # Calculate reward
        reward = self._calculate_reward(action_success, exploit_info)
        self.total_reward += reward

        # Build new observation
        observation = self._build_observation()

        # Check termination conditions
        terminated = False  # Task complete (e.g., all critical nodes owned)
        truncated = self.current_step >= self.config.max_steps  # Max steps reached

        # Build info dict
        info = {
            'step': self.current_step,
            'action_success': action_success,
            'total_reward': self.total_reward,
            'exploited_count': len(self.exploited_vulns),
            'exploit_info': exploit_info
        }

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Render the environment state."""
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.config.max_steps}")
            print(f"Total Reward: {self.total_reward:.2f}")
            print(f"Exploited Vulnerabilities: {len(self.exploited_vulns)}")
        # TODO: Add visualization using CyberBattleSim's rendering

    def close(self):
        """Clean up resources."""
        if self.cyberbattle_env is not None:
            # Close CyberBattleSim environment
            pass


def create_env_from_nessus(nessus_file: str, config: AUVAPConfig = None) -> AUVAPCyberBattleEnv:
    """
    Helper function to create an environment from a Nessus scan file.

    Args:
        nessus_file: Path to Nessus XML file
        config: Optional configuration

    Returns:
        Configured AUVAPCyberBattleEnv instance
    """
    from parser import parse_nessus_xml

    # Parse vulnerability findings
    findings = parse_nessus_xml(nessus_file)

    # Create and return environment
    return AUVAPCyberBattleEnv(
        vulnerability_findings=findings,
        config=config
    )
