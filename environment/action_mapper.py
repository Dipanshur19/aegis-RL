"""
Action Space Mapping for AUVAP-PPO Integration

This module provides mappings between AUVAP ExploitTask objects and
discrete action indices for reinforcement learning.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# AUVAP imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_manager import ExploitTask, TaskState
from parser import VAFinding


@dataclass
class ActionInfo:
    """Information about a discrete action"""
    action_id: int
    exploit_task: ExploitTask
    target_host: str
    target_port: int
    vulnerability_id: str
    cvss_score: float
    description: str


class ActionMapper:
    """
    Maps between ExploitTask objects and discrete action indices.

    This class maintains a bidirectional mapping between:
    - Discrete action IDs (used by RL agent)
    - ExploitTask objects (from AUVAP pipeline)

    It also handles:
    - Action masking (filtering out invalid actions)
    - Action prioritization based on CVSS scores
    - Dynamic action space updates as new vulnerabilities are discovered
    """

    def __init__(self, max_actions: int = 100):
        """
        Initialize the action mapper.

        Args:
            max_actions: Maximum number of actions in the action space
        """
        self.max_actions = max_actions
        self.action_to_task: Dict[int, ExploitTask] = {}
        self.task_to_action: Dict[str, int] = {}  # task_id -> action_id
        self.action_info: Dict[int, ActionInfo] = {}
        self.next_action_id = 0

    def register_task(self, task: ExploitTask) -> Optional[int]:
        """
        Register a new exploit task and assign it an action ID.

        Args:
            task: The ExploitTask to register

        Returns:
            The assigned action ID, or None if max actions reached
        """
        # Check if task already registered
        if task.task_id in self.task_to_action:
            return self.task_to_action[task.task_id]

        # Check if we've reached max actions
        if self.next_action_id >= self.max_actions:
            return None

        action_id = self.next_action_id
        self.action_to_task[action_id] = task
        self.task_to_action[task.task_id] = action_id

        # Store action info
        self.action_info[action_id] = ActionInfo(
            action_id=action_id,
            exploit_task=task,
            target_host=task.target_host,
            target_port=task.target_port,
            vulnerability_id=task.vulnerability_id,
            cvss_score=task.cvss_score,
            description=f"Exploit {task.vulnerability_id} on {task.target_host}:{task.target_port}"
        )

        self.next_action_id += 1
        return action_id

    def register_tasks_from_findings(self, findings: List[VAFinding]) -> List[int]:
        """
        Create and register exploit tasks from vulnerability findings.

        Args:
            findings: List of vulnerability findings

        Returns:
            List of registered action IDs
        """
        action_ids = []

        for finding in findings:
            # Create an ExploitTask from the finding
            host = getattr(finding, 'host_ip', '')
            port = getattr(finding, 'port', 0)
            finding_id = getattr(finding, 'finding_id', '')
            cvss = getattr(finding, 'cvss_base_score', 0.0) or 0.0
            task = ExploitTask(
                task_id=finding_id,
                finding_id=finding_id,
                state=TaskState.PLANNED,
                target=f"{host}:{port}",
                config={
                    'host': host,
                    'port': port,
                    'protocol': getattr(finding, 'protocol', 'tcp'),
                    'service': getattr(finding, 'service', 'tcp'),
                    'vulnerability_id': getattr(finding, 'plugin_id', ''),
                    'description': getattr(finding, 'plugin_name', ''),
                    'cvss_score': cvss,
                },
                risk_score=cvss,
                priority=self._calculate_priority(finding)
            )

            action_id = self.register_task(task)
            if action_id is not None:
                action_ids.append(action_id)

        return action_ids

    def _calculate_priority(self, finding: VAFinding) -> float:
        """
        Calculate task priority based on vulnerability characteristics.

        Args:
            finding: The vulnerability finding

        Returns:
            Priority score (higher = more important)
        """
        # Base priority on CVSS score
        priority = finding.cvss_base_score or 5.0

        # Adjust based on severity
        severity_weights = {
            'Critical': 2.0,
            'High': 1.5,
            'Medium': 1.0,
            'Low': 0.5,
            'Info': 0.1
        }
        severity_weight = severity_weights.get(finding.severity, 1.0)
        priority *= severity_weight

        return priority

    def get_task_from_action(self, action_id: int) -> Optional[ExploitTask]:
        """
        Get the ExploitTask corresponding to an action ID.

        Args:
            action_id: The discrete action ID

        Returns:
            The corresponding ExploitTask, or None if not found
        """
        return self.action_to_task.get(action_id)

    def get_action_from_task(self, task_id: str) -> Optional[int]:
        """
        Get the action ID corresponding to a task ID.

        Args:
            task_id: The task identifier

        Returns:
            The corresponding action ID, or None if not found
        """
        return self.task_to_action.get(task_id)

    def get_action_info(self, action_id: int) -> Optional[ActionInfo]:
        """
        Get detailed information about an action.

        Args:
            action_id: The action ID

        Returns:
            ActionInfo object with details
        """
        return self.action_info.get(action_id)

    def get_valid_actions(self, exclude_completed: bool = True) -> List[int]:
        """
        Get list of valid action IDs.

        Args:
            exclude_completed: Whether to exclude completed tasks

        Returns:
            List of valid action IDs
        """
        valid_actions = []

        for action_id, task in self.action_to_task.items():
            if exclude_completed and task.state in [TaskState.SUCCEEDED, TaskState.ABORTED]:
                continue
            valid_actions.append(action_id)

        return valid_actions

    def get_action_mask(self) -> np.ndarray:
        """
        Get a binary mask indicating valid actions.

        Returns:
            Boolean numpy array of shape (max_actions,) where True = valid action
        """
        mask = np.zeros(self.max_actions, dtype=bool)
        valid_actions = self.get_valid_actions()

        for action_id in valid_actions:
            if action_id < self.max_actions:
                mask[action_id] = True

        return mask

    def get_action_priorities(self) -> np.ndarray:
        """
        Get priority scores for all actions.

        Returns:
            Numpy array of priority scores
        """
        priorities = np.zeros(self.max_actions, dtype=np.float32)

        for action_id, info in self.action_info.items():
            if action_id < self.max_actions:
                priorities[action_id] = info.exploit_task.priority

        return priorities

    def sample_action(self, valid_only: bool = True) -> int:
        """
        Sample a random action.

        Args:
            valid_only: Whether to sample only from valid actions

        Returns:
            Sampled action ID
        """
        if valid_only:
            valid_actions = self.get_valid_actions()
            if not valid_actions:
                return 0  # Default action
            return np.random.choice(valid_actions)
        else:
            return np.random.randint(0, self.next_action_id)

    def reset(self):
        """Reset the action mapper to initial state."""
        self.action_to_task.clear()
        self.task_to_action.clear()
        self.action_info.clear()
        self.next_action_id = 0

    def get_statistics(self) -> Dict:
        """
        Get statistics about the action space.

        Returns:
            Dictionary with statistics
        """
        states: Dict[str, int] = {}
        for task in self.action_to_task.values():
            state_name = task.state.name
            states[state_name] = states.get(state_name, 0) + 1

        return {
            'total_actions': self.next_action_id,
            'registered_tasks': len(self.action_to_task),
            'valid_actions': len(self.get_valid_actions()),
            'task_states': states,
            'max_cvss': max((info.cvss_score for info in self.action_info.values()), default=0.0),
            'avg_cvss': np.mean([info.cvss_score for info in self.action_info.values()]) if self.action_info else 0.0
        }

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return f"ActionMapper(actions={stats['total_actions']}, valid={stats['valid_actions']})"
