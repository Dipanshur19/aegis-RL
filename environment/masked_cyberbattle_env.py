"""
Masked CyberBattle Environment with Masking Sensor Integration

This environment integrates the masking sensor algorithm with the CyberBattle
environment to provide controlled, safe task exposure to the DRL agent.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cyberbattle_wrapper import AUVAPCyberBattleEnv, AUVAPConfig
from environment.masking_sensor import MaskingSensor, TaskExposure
from environment.action_mapper import ActionMapper
from environment.reward_shaper import RewardShaper, RewardConfig
from parser import VAFinding
from policy_engine import PolicyEngine


class MaskedCyberBattleEnv(gym.Env):
    """
    Masked environment that exposes one task at a time via masking sensor.

    This environment:
    - Uses masking sensor to control task exposure
    - Enforces safety constraints
    - Tracks network state across episodes
    - Provides action masking for invalid actions
    - Logs all execution for replay
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 vulnerability_findings: List[VAFinding],
                 config: AUVAPConfig = None,
                 policy_engine: Optional[PolicyEngine] = None,
                 max_attempts_per_task: int = 3,
                 enable_safety_constraints: bool = True,
                 log_file: Optional[str] = None):
        """
        Initialize the masked environment.

        Args:
            vulnerability_findings: List of vulnerability findings
            config: Environment configuration
            policy_engine: Optional policy engine for constraints
            max_attempts_per_task: Max retry attempts per task
            enable_safety_constraints: Enable safety constraints
            log_file: Path to execution log file
        """
        super().__init__()

        self.config = config or AUVAPConfig()
        self.vulnerability_findings = vulnerability_findings
        self.policy_engine = policy_engine

        # Initialize components
        self.action_mapper = ActionMapper(max_actions=100)

        self.masking_sensor = MaskingSensor(
            findings=vulnerability_findings,
            action_mapper=self.action_mapper,
            policy_engine=policy_engine,
            max_attempts_per_task=max_attempts_per_task,
            enable_safety_constraints=enable_safety_constraints,
            log_file=log_file
        )

        self.reward_shaper = RewardShaper(RewardConfig(
            success_reward=self.config.reward_success,
            failure_penalty=self.config.reward_failure,
            step_penalty=self.config.reward_step,
            use_risk_shaping=self.config.use_risk_score_reward
        ))

        # Current task tracking
        self.current_exposure: Optional[TaskExposure] = None
        self.current_step = 0
        self.episode_start_time = 0

        # Define spaces
        self.action_space = gym.spaces.Discrete(100)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(50,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset to next task from masking sensor.

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Get next task from masking sensor
        self.current_exposure = self.masking_sensor.get_current_task()
        self.current_step = 0

        if self.current_exposure is None:
            # No more tasks - return dummy observation
            return self._build_terminal_observation(), {'terminal': True, 'sensor_complete': True}

        # Build observation for current task
        observation = self._build_observation()

        info = {
            'task_id': self.current_exposure.task.task_id,
            'target': self.current_exposure.task.target_host,
            'cvss': self.current_exposure.task.cvss_score,
            'allowed_actions': len(self.current_exposure.allowed_actions),
            'attempt': self.current_exposure.attempt_number,
            'max_attempts': self.current_exposure.max_attempts,
            'safety_constraints': len(self.current_exposure.safety_constraints),
            'terminal': False
        }

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action in the environment.

        Args:
            action: Action ID to execute

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1

        # Check if we have a current task
        if self.current_exposure is None:
            return self._build_terminal_observation(), 0.0, True, False, {'error': 'No current task'}

        # Check if action is allowed
        if action not in self.current_exposure.allowed_actions:
            # Invalid action - large penalty
            return self._build_observation(), -10.0, False, False, {
                'invalid_action': True,
                'action': action,
                'allowed_actions': self.current_exposure.allowed_actions
            }

        # Execute action (simulate exploitation)
        execution_result = self._execute_action(action)

        # Calculate reward
        reward = self._calculate_reward(execution_result)

        # Check if task is complete
        terminated = execution_result['success'] or \
                    self.current_step >= self.config.max_steps or \
                    self.current_exposure.attempt_number >= self.current_exposure.max_attempts - 1

        # Advance sensor if task complete
        if terminated:
            self.masking_sensor.advance(execution_result)

        # Build info
        info = {
            'task_id': self.current_exposure.task.task_id,
            'action': action,
            'action_success': execution_result['success'],
            'reward': reward,
            'step': self.current_step,
            'execution_result': execution_result,
            'sensor_stats': self.masking_sensor.get_statistics()
        }

        # Next observation
        observation = self._build_observation()

        return observation, reward, terminated, False, info

    def _execute_action(self, action: int) -> Dict:
        """
        Simulate action execution.

        In production, this would:
        1. Generate exploit script via LLM
        2. Execute in sandbox with safety constraints
        3. Parse results and update network state

        For now, we simulate with probabilistic success.
        """
        import time
        import random

        start_time = time.time()

        # Get task info
        task = self.current_exposure.task

        # Success probability based on CVSS and attempts
        base_success_prob = 0.6

        # Higher CVSS = easier to exploit
        cvss_bonus = (task.cvss_score - 5.0) / 10.0  # -0.5 to +0.5

        # More attempts = lower success (target may be patched/detected)
        attempt_penalty = self.current_exposure.attempt_number * 0.1

        success_prob = base_success_prob + cvss_bonus - attempt_penalty
        success_prob = np.clip(success_prob, 0.1, 0.9)

        success = random.random() < success_prob

        duration = time.time() - start_time

        result = {
            'success': success,
            'action': action,
            'duration': duration,
            'safety_violations': [],
            'error': '' if success else 'Exploitation failed',
            'artifacts': {}
        }

        if success:
            # Generate mock credentials
            result['artifacts'] = {
                'credentials': [f'user{random.randint(1,100)}:pass{random.randint(1000,9999)}']
            }

        return result

    def _calculate_reward(self, execution_result: Dict) -> float:
        """Calculate reward using reward shaper."""
        task = self.current_exposure.task

        # Build exploit info for reward shaper
        exploit_info = {
            'cvss_score': task.cvss_score,
            'attack_surface': 'remote' if task.protocol in ['tcp', 'udp'] else 'local',
            'automation_level': 'high' if task.cvss_score >= 7.0 else 'medium'
        }

        # Build strategic info
        strategic_info = {
            'is_critical_node': task.cvss_score >= 9.0,
            'discovered_new_node': execution_result['success'],
            'lateral_movement': len(self.masking_sensor.owned_nodes) > 0,
            'network_depth': len(self.masking_sensor.owned_nodes)
        }

        return self.reward_shaper.calculate_reward(
            action_success=execution_result['success'],
            exploit_info=exploit_info,
            strategic_info=strategic_info
        )

    def _build_observation(self) -> np.ndarray:
        """
        Build observation vector from current state.

        Observation includes:
        - Task features (CVSS, severity, port, protocol)
        - Network state (owned nodes, credentials)
        - Attempt information
        - Context features
        """
        obs = np.zeros(50, dtype=np.float32)

        if self.current_exposure is None:
            return obs

        task = self.current_exposure.task

        # Task features (0-9)
        obs[0] = task.cvss_score / 10.0  # Normalized CVSS
        obs[1] = task.port / 65535.0  # Normalized port
        obs[2] = 1.0 if task.protocol == 'tcp' else 0.5 if task.protocol == 'udp' else 0.0
        obs[3] = task.priority / 100.0  # Normalized priority
        obs[4] = self.current_exposure.attempt_number / self.current_exposure.max_attempts
        obs[5] = len(self.current_exposure.allowed_actions) / 10.0
        obs[6] = len(self.current_exposure.safety_constraints) / 10.0
        obs[7] = self.current_step / self.config.max_steps

        # Network state (10-19)
        obs[10] = len(self.masking_sensor.owned_nodes) / 20.0
        obs[11] = sum(len(creds) for creds in self.masking_sensor.discovered_credentials.values()) / 50.0

        # Context features (20-29)
        context = self.current_exposure.context
        obs[20] = context['owned_count'] / 20.0
        obs[21] = len(context['available_credentials']) / 10.0
        obs[22] = context['previous_attempts'] / self.current_exposure.max_attempts

        # Sensor statistics (30-39)
        stats = self.masking_sensor.get_statistics()
        obs[30] = stats['success_rate']
        obs[31] = stats['completed'] / max(stats['total_tasks'], 1)
        obs[32] = stats['failed'] / max(stats['total_tasks'], 1)

        return obs

    def _build_terminal_observation(self) -> np.ndarray:
        """Build observation when no more tasks available."""
        return np.zeros(50, dtype=np.float32)

    def action_masks(self) -> np.ndarray:
        """
        Get action mask for current state.
        Required for MaskablePPO.

        Returns:
            Boolean array where True = valid action
        """
        mask = np.zeros(100, dtype=bool)

        if self.current_exposure is not None:
            for action_id in self.current_exposure.allowed_actions:
                if action_id < 100:
                    mask[action_id] = True

        return mask

    def is_complete(self) -> bool:
        """Check if all tasks are complete."""
        return self.masking_sensor.is_complete()

    def get_sensor_statistics(self) -> Dict:
        """Get masking sensor statistics."""
        return self.masking_sensor.get_statistics()

    def save_execution_log(self, output_file: str):
        """Save complete execution log."""
        self.masking_sensor.save_execution_log(output_file)

    def render(self, mode='human'):
        """Render the environment state."""
        if mode == 'human' and self.current_exposure is not None:
            task = self.current_exposure.task
            print(f"\n{'='*60}")
            print(f"Current Task: {task.task_id}")
            print(f"Target: {task.target_host}:{task.target_port}")
            print(f"CVSS: {task.cvss_score}")
            print(f"Allowed actions: {self.current_exposure.allowed_actions}")
            print(f"Attempt: {self.current_exposure.attempt_number + 1}/{self.current_exposure.max_attempts}")
            print(f"Owned nodes: {len(self.masking_sensor.owned_nodes)}")
            print(f"{'='*60}\n")

    def close(self):
        """Clean up resources."""
        pass


def create_masked_env_from_nessus(nessus_file: str,
                                   config: AUVAPConfig = None,
                                   policy_engine: Optional[PolicyEngine] = None) -> MaskedCyberBattleEnv:
    """
    Helper function to create masked environment from Nessus file.

    Args:
        nessus_file: Path to Nessus XML file
        config: Environment configuration
        policy_engine: Optional policy engine

    Returns:
        Configured MaskedCyberBattleEnv
    """
    from parser import parse_nessus_xml

    findings = parse_nessus_xml(nessus_file)

    return MaskedCyberBattleEnv(
        vulnerability_findings=findings,
        config=config,
        policy_engine=policy_engine
    )
