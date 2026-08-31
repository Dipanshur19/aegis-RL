"""
Reward Shaping for AUVAP-PPO Integration

This module provides reward calculation and shaping based on AUVAP's
risk scoring and vulnerability assessment.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

# AUVAP imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feasibility_filter import compute_risk_score
from parser import VAFinding


@dataclass
class RewardConfig:
    """Configuration for reward calculation"""
    success_reward: float = 10.0
    failure_penalty: float = -1.0
    step_penalty: float = -0.1
    use_risk_shaping: bool = True
    risk_weight: float = 0.5
    critical_node_bonus: float = 5.0
    discovery_reward: float = 0.5
    lateral_movement_bonus: float = 2.0


class RewardShaper:
    """
    Calculates and shapes rewards for the RL agent.

    Reward components:
    1. Base reward: Success/failure of exploitation
    2. Risk-based shaping: Uses CVSS and automation feasibility
    3. Strategic bonuses: Critical nodes, lateral movement, discovery
    4. Time penalty: Encourages efficiency
    """

    def __init__(self, config: RewardConfig = None):
        """
        Initialize the reward shaper.

        Args:
            config: Reward configuration
        """
        self.config = config or RewardConfig()

    def calculate_reward(self,
                        action_success: bool,
                        exploit_info: Dict = None,
                        strategic_info: Dict = None) -> float:
        """
        Calculate the total reward for an action.

        Args:
            action_success: Whether the exploitation succeeded
            exploit_info: Information about the exploit (CVSS, target, etc.)
            strategic_info: Strategic information (node type, network position, etc.)

        Returns:
            Total reward value
        """
        reward = 0.0

        # Base reward/penalty
        if action_success:
            reward += self.config.success_reward
        else:
            reward += self.config.failure_penalty

        # Risk-based reward shaping
        if self.config.use_risk_shaping and exploit_info:
            risk_reward = self._calculate_risk_reward(exploit_info)
            reward += risk_reward * self.config.risk_weight

        # Strategic bonuses
        if action_success and strategic_info:
            strategic_bonus = self._calculate_strategic_bonus(strategic_info)
            reward += strategic_bonus

        # Step penalty (encourages efficiency)
        reward += self.config.step_penalty

        return reward

    def _calculate_risk_reward(self, exploit_info: Dict) -> float:
        """
        Calculate reward based on AUVAP's risk scoring.

        Uses the existing compute_risk_score function to determine
        the value of successfully exploiting a vulnerability.

        Args:
            exploit_info: Information about the exploit

        Returns:
            Risk-based reward component
        """
        cvss_score = exploit_info.get('cvss_score', 5.0)
        attack_surface = exploit_info.get('attack_surface', 'remote')
        automation_level = exploit_info.get('automation_level', 'medium')

        # Use AUVAP's risk scoring
        finding_dict = {
            'cvss': cvss_score,
            'attack_vector': 'Network' if str(attack_surface).lower() == 'remote' else 'Local',
            'automation_candidate': (str(automation_level).lower() in ['high', 'medium'])
        }
        risk_score = compute_risk_score(finding_dict)

        # Normalize risk score to reasonable reward range
        # risk_score typically ranges from 0-10
        risk_reward = risk_score  # Scale directly

        return risk_reward

    def _calculate_strategic_bonus(self, strategic_info: Dict) -> float:
        """
        Calculate bonus rewards for strategically valuable actions.

        Args:
            strategic_info: Strategic information about the action

        Returns:
            Strategic bonus reward
        """
        bonus = 0.0

        # Critical node bonus
        if strategic_info.get('is_critical_node', False):
            bonus += self.config.critical_node_bonus

        # Discovery bonus (for finding new nodes)
        if strategic_info.get('discovered_new_node', False):
            bonus += self.config.discovery_reward

        # Lateral movement bonus (for moving deeper into network)
        if strategic_info.get('lateral_movement', False):
            bonus += self.config.lateral_movement_bonus

        # Network position bonus (deeper = better)
        network_depth = strategic_info.get('network_depth', 0)
        bonus += network_depth * 0.5

        return bonus

    def calculate_episode_reward(self,
                                 total_nodes_owned: int,
                                 total_nodes: int,
                                 critical_nodes_owned: int,
                                 steps_taken: int,
                                 max_steps: int) -> float:
        """
        Calculate cumulative episode reward/bonus.

        This can be used as a final episode bonus based on overall performance.

        Args:
            total_nodes_owned: Number of nodes successfully compromised
            total_nodes: Total number of nodes in network
            critical_nodes_owned: Number of critical nodes owned
            steps_taken: Number of steps taken
            max_steps: Maximum allowed steps

        Returns:
            Episode completion bonus
        """
        bonus = 0.0

        # Completion bonus
        completion_ratio = total_nodes_owned / max(total_nodes, 1)
        bonus += completion_ratio * 20.0

        # Critical node bonus
        bonus += critical_nodes_owned * self.config.critical_node_bonus

        # Efficiency bonus (fewer steps = better)
        efficiency = 1.0 - (steps_taken / max(max_steps, 1))
        bonus += efficiency * 10.0

        return bonus

    def shape_reward_from_finding(self, finding: VAFinding, success: bool = True) -> float:
        """
        Calculate reward directly from a VAFinding object.

        Args:
            finding: Vulnerability finding from AUVAP
            success: Whether the exploitation was successful

        Returns:
            Shaped reward value
        """
        exploit_info = {
            'cvss_score': finding.cvss_base_score or 5.0,
            'attack_surface': 'remote' if finding.protocol in ['tcp', 'udp'] else 'local',
            'automation_level': self._infer_automation_level(finding)
        }

        strategic_info = {
            'is_critical_node': finding.severity in ['Critical', 'High'],
            'network_depth': 1  # Default, would be determined by network position
        }

        return self.calculate_reward(success, exploit_info, strategic_info)

    def _infer_automation_level(self, finding: VAFinding) -> str:
        """
        Infer automation feasibility from vulnerability characteristics.

        Args:
            finding: Vulnerability finding

        Returns:
            Automation level ('high', 'medium', 'low')
        """
        # Heuristic based on CVSS and exploit availability
        cvss = finding.cvss_base_score or 5.0

        if cvss >= 9.0:
            return 'high'
        elif cvss >= 7.0:
            return 'medium'
        else:
            return 'low'

    def get_reward_statistics(self, episode_rewards: list) -> Dict:
        """
        Calculate statistics from episode rewards.

        Args:
            episode_rewards: List of rewards from an episode

        Returns:
            Dictionary with reward statistics
        """
        if not episode_rewards:
            return {
                'total': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'positive_count': 0,
                'negative_count': 0
            }

        rewards_array = np.array(episode_rewards)

        return {
            'total': np.sum(rewards_array),
            'mean': np.mean(rewards_array),
            'std': np.std(rewards_array),
            'min': np.min(rewards_array),
            'max': np.max(rewards_array),
            'positive_count': np.sum(rewards_array > 0),
            'negative_count': np.sum(rewards_array < 0)
        }
