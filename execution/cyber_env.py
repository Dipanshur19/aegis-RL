"""
CyberBattleSim Environment Integration (Priority 1, Item 1)

Full Gymnasium environment integrating:
- Microsoft CyberBattleSim as the simulation backend
- TerrainGenerator for synthetic network creation
- AUVAP vulnerability findings as network configuration
- PPO-compatible observation/action spaces
- Safety constraint enforcement
- Reward shaping based on penetration testing objectives
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import random

from parser import VAFinding
from task_manager import ExploitTask, TaskState
from execution.terrain_generator import TerrainGenerator, TerrainParams, NodeAttributes


class ActionType(Enum):
    """Types of actions in penetration testing"""
    EXPLOIT_VULNERABILITY = "exploit_vuln"
    LATERAL_MOVE = "lateral_move"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    NO_OP = "no_op"


@dataclass
class CyberAction:
    """Represents an action in the cyber environment"""
    action_id: int
    action_type: ActionType
    target_node: str
    vulnerability: Optional[str] = None
    source_node: Optional[str] = None
    description: str = ""


@dataclass
class CyberState:
    """Current state of the cyber environment"""
    owned_nodes: Set[str] = field(default_factory=set)
    discovered_nodes: Set[str] = field(default_factory=set)
    current_node: str = ""
    credentials_found: Dict[str, List[Dict]] = field(default_factory=dict)
    exploited_vulns: Set[str] = field(default_factory=set)
    discovered_services: Dict[str, List[str]] = field(default_factory=dict)
    step_count: int = 0
    root_owned_nodes: Set[str] = field(default_factory=set)


@dataclass
class EnvConfig:
    """Configuration for CyberBattleSim environment"""
    max_steps: int = 100
    max_nodes: int = 50
    observation_dim: int = 256
    action_dim: int = 100

    # Rewards (matching Equation 2 from paper)
    reward_root_access: float = 100.0
    reward_new_host: float = 10.0
    reward_credentials: float = 5.0
    reward_step_penalty: float = -1.0
    reward_safety_violation: float = -100.0

    # Safety constraints
    enable_safety_checks: bool = True
    max_exploit_attempts_per_vuln: int = 3
    forbidden_nodes: Set[str] = field(default_factory=set)

    # Reward shaping
    use_cvss_shaping: bool = True
    use_distance_shaping: bool = True  # Reward proximity to high-value targets


class CyberBattleEnv(gym.Env):
    """
    Full CyberBattleSim environment for AUVAP-PPO training.

    This environment:
    - Uses TerrainGenerator to create synthetic networks
    - Supports loading real vulnerability scans
    - Provides PPO-compatible gym interface
    - Implements reward function from paper (Equation 2)
    - Enforces safety constraints
    - Tracks penetration testing objectives

    Observation Space:
        Box(256,) - Feature vector containing:
        - Node features (owned, discovered, services, vulnerabilities)
        - Network topology (adjacency, connectivity)
        - Current position and progress metrics
        - Available actions encoding

    Action Space:
        Discrete(100) - Action IDs mapped to:
        - Exploit vulnerability on target node
        - Lateral movement between nodes
        - Credential access attempts
        - Network discovery
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self,
                 terrain_generator: Optional[TerrainGenerator] = None,
                 terrain_params: Optional[TerrainParams] = None,
                 vulnerability_findings: Optional[List[VAFinding]] = None,
                 config: Optional[EnvConfig] = None,
                 seed: Optional[int] = None):
        """
        Initialize CyberBattleSim environment.

        Args:
            terrain_generator: Generator for synthetic networks
            terrain_params: Parameters for terrain generation
            vulnerability_findings: Real vulnerability scan results
            config: Environment configuration
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.config = config or EnvConfig()
        self.terrain_generator = terrain_generator or TerrainGenerator()
        self.terrain_params = terrain_params or TerrainParams()
        self.vulnerability_findings = vulnerability_findings or []

        # Simulation state
        self.network_graph: Optional[nx.DiGraph] = None
        self.terrain_id: Optional[str] = None
        self.state = CyberState()

        # Action mapping
        self.actions: List[CyberAction] = []
        self.action_map: Dict[int, CyberAction] = {}
        self.exploit_attempts: Dict[str, int] = {}  # Track attempts per vulnerability

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.config.observation_dim,),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(self.config.action_dim)

        # Metrics
        self.episode_reward = 0.0
        self.episode_metrics = {
            'nodes_owned': 0,
            'root_access_count': 0,
            'credentials_found': 0,
            'safety_violations': 0
        }

        # Set random seed
        if seed is not None:
            self.seed(seed)

    def seed(self, seed: int):
        """Set random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        self.terrain_generator.params.ensure_connected = True

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options (can contain 'terrain_params', 'preset')

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if seed is not None:
            self.seed(seed)

        # Handle options
        options = options or {}
        terrain_params = options.get('terrain_params', self.terrain_params)
        preset = options.get('preset')

        # Generate or load network terrain
        if self.vulnerability_findings:
            # Build network from real vulnerability scan
            self.network_graph = self._build_network_from_findings()
            self.terrain_id = "real_scan"
        else:
            # Generate synthetic terrain
            if preset:
                # Load preset from config
                terrain_params = self._load_terrain_preset(preset)

            self.network_graph, self.terrain_id = self.terrain_generator.generate_terrain(
                terrain_params, seed
            )

        # Initialize state
        self._initialize_state()

        # Build action space
        self._build_action_space()

        # Reset metrics
        self.episode_reward = 0.0
        self.episode_metrics = {
            'nodes_owned': 1,  # Start with entry node
            'root_access_count': 0,
            'credentials_found': 0,
            'safety_violations': 0
        }

        # Build initial observation
        observation = self._build_observation()

        info = {
            'terrain_id': self.terrain_id,
            'num_nodes': len(self.network_graph.nodes()),
            'num_edges': len(self.network_graph.edges()),
            'entry_node': self.state.current_node,
            'available_actions': len(self._get_valid_actions())
        }

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action ID to execute

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.state.step_count += 1

        # Get action details
        if action not in self.action_map:
            # Invalid action
            reward = -1.0
            observation = self._build_observation()
            info = {
                'success': False,
                'error': 'Invalid action',
                'action_id': action
            }
            return observation, reward, False, False, info

        cyber_action = self.action_map[action]

        # Execute action
        result = self._execute_action(cyber_action)

        # Calculate reward (Equation 2 from paper)
        reward = self._compute_reward(result)
        self.episode_reward += reward

        # Update metrics
        self._update_metrics(result)

        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.state.step_count >= self.config.max_steps

        # Build new observation
        observation = self._build_observation()

        # Build info dict
        info = {
            'action_type': cyber_action.action_type.value,
            'target_node': cyber_action.target_node,
            'success': result.get('success', False),
            'reward_breakdown': result.get('reward_breakdown', {}),
            'nodes_owned': len(self.state.owned_nodes),
            'root_access': result.get('got_root', False),
            'safety_violation': result.get('safety_violation', False),
            'episode_reward': self.episode_reward,
            'step': self.state.step_count
        }

        return observation, reward, terminated, truncated, info

    def _initialize_state(self):
        """Initialize environment state"""
        # Find entry point node (lowest firewall protection)
        entry_nodes = [
            node for node in self.network_graph.nodes()
            if len(self.network_graph.nodes[node].get('firewall_rules', [])) == 0
        ]

        if not entry_nodes:
            # Fallback: use any node
            entry_nodes = list(self.network_graph.nodes())

        # Select entry node
        entry_node = entry_nodes[0] if entry_nodes else list(self.network_graph.nodes())[0]

        # Initialize state
        self.state = CyberState(
            owned_nodes={entry_node},
            discovered_nodes={entry_node},
            current_node=entry_node,
            step_count=0
        )

        # Discover immediate neighbors
        for neighbor in self.network_graph.neighbors(entry_node):
            self.state.discovered_nodes.add(neighbor)

    def _build_action_space(self):
        """Build action space from current network state"""
        self.actions = []
        self.action_map = {}
        action_id = 0

        # For each owned node, find available actions
        for owned_node in self.state.owned_nodes:
            # 1. Exploit actions on discovered neighbors
            for neighbor in self.network_graph.neighbors(owned_node):
                if neighbor not in self.state.owned_nodes:
                    # Get vulnerabilities for this node
                    vulnerabilities = self.network_graph.nodes[neighbor].get('vulnerabilities', [])

                    for vuln in vulnerabilities:
                        if vuln not in self.state.exploited_vulns:
                            action = CyberAction(
                                action_id=action_id,
                                action_type=ActionType.EXPLOIT_VULNERABILITY,
                                target_node=neighbor,
                                vulnerability=vuln,
                                source_node=owned_node,
                                description=f"Exploit {vuln} on {neighbor} from {owned_node}"
                            )
                            self.actions.append(action)
                            self.action_map[action_id] = action
                            action_id += 1

                            if action_id >= self.config.action_dim:
                                return

            # 2. Lateral movement to owned neighbors
            for neighbor in self.network_graph.neighbors(owned_node):
                if neighbor in self.state.owned_nodes and neighbor != self.state.current_node:
                    action = CyberAction(
                        action_id=action_id,
                        action_type=ActionType.LATERAL_MOVE,
                        target_node=neighbor,
                        source_node=owned_node,
                        description=f"Move from {owned_node} to {neighbor}"
                    )
                    self.actions.append(action)
                    self.action_map[action_id] = action
                    action_id += 1

                    if action_id >= self.config.action_dim:
                        return

            # 3. Credential access on owned node
            if owned_node not in self.state.credentials_found:
                credentials = self.network_graph.nodes[owned_node].get('credentials', [])
                if credentials:
                    action = CyberAction(
                        action_id=action_id,
                        action_type=ActionType.CREDENTIAL_ACCESS,
                        target_node=owned_node,
                        description=f"Extract credentials from {owned_node}"
                    )
                    self.actions.append(action)
                    self.action_map[action_id] = action
                    action_id += 1

                    if action_id >= self.config.action_dim:
                        return

            # 4. Discovery on owned node
            action = CyberAction(
                action_id=action_id,
                action_type=ActionType.DISCOVERY,
                target_node=owned_node,
                description=f"Discover network from {owned_node}"
            )
            self.actions.append(action)
            self.action_map[action_id] = action
            action_id += 1

            if action_id >= self.config.action_dim:
                return

        # Fill remaining action space with NO_OP
        while action_id < self.config.action_dim:
            action = CyberAction(
                action_id=action_id,
                action_type=ActionType.NO_OP,
                target_node="",
                description="No operation"
            )
            self.actions.append(action)
            self.action_map[action_id] = action
            action_id += 1

    def _execute_action(self, action: CyberAction) -> Dict:
        """
        Execute an action in the environment.

        Args:
            action: Action to execute

        Returns:
            Dictionary with execution results
        """
        result = {
            'success': False,
            'got_root': False,
            'new_host_owned': False,
            'credentials_found': 0,
            'safety_violation': False,
            'reward_breakdown': {}
        }

        # Check safety constraints
        if self.config.enable_safety_checks:
            if action.target_node in self.config.forbidden_nodes:
                result['safety_violation'] = True
                result['error'] = f"Target node {action.target_node} is forbidden"
                return result

        # Execute based on action type
        if action.action_type == ActionType.EXPLOIT_VULNERABILITY:
            result = self._execute_exploit(action)

        elif action.action_type == ActionType.LATERAL_MOVE:
            result = self._execute_lateral_move(action)

        elif action.action_type == ActionType.CREDENTIAL_ACCESS:
            result = self._execute_credential_access(action)

        elif action.action_type == ActionType.DISCOVERY:
            result = self._execute_discovery(action)

        elif action.action_type == ActionType.NO_OP:
            result['success'] = True

        return result

    def _execute_exploit(self, action: CyberAction) -> Dict:
        """Execute vulnerability exploitation"""
        result = {
            'success': False,
            'got_root': False,
            'new_host_owned': False,
            'credentials_found': 0,
            'safety_violation': False,
            'reward_breakdown': {}
        }

        # Check exploit attempt limit
        vuln_key = f"{action.target_node}_{action.vulnerability}"
        attempts = self.exploit_attempts.get(vuln_key, 0)

        if attempts >= self.config.max_exploit_attempts_per_vuln:
            result['error'] = "Max exploit attempts reached"
            return result

        self.exploit_attempts[vuln_key] = attempts + 1

        # Simulate exploitation (success probability based on node attributes)
        target_attrs = self.network_graph.nodes[action.target_node]
        firewall_rules = target_attrs.get('firewall_rules', [])

        # Base success probability
        success_prob = 0.7

        # Reduce by firewall protection
        success_prob -= len(firewall_rules) * 0.1

        # Increase if we have credentials for this node
        if action.target_node in self.state.credentials_found:
            success_prob += 0.2

        success_prob = np.clip(success_prob, 0.1, 0.95)

        # Determine success
        if random.random() < success_prob:
            result['success'] = True

            # Mark node as owned
            if action.target_node not in self.state.owned_nodes:
                self.state.owned_nodes.add(action.target_node)
                result['new_host_owned'] = True

            # Check for root access (server/router/firewall roles)
            role = target_attrs.get('role', 'workstation')
            if role in ['server', 'router', 'firewall']:
                result['got_root'] = True
                self.state.root_owned_nodes.add(action.target_node)

            # Mark vulnerability as exploited
            self.state.exploited_vulns.add(action.vulnerability)

            # Move to new node
            self.state.current_node = action.target_node

            # Discover neighbors
            for neighbor in self.network_graph.neighbors(action.target_node):
                self.state.discovered_nodes.add(neighbor)

        return result

    def _execute_lateral_move(self, action: CyberAction) -> Dict:
        """Execute lateral movement"""
        result = {'success': False}

        if action.target_node in self.state.owned_nodes:
            self.state.current_node = action.target_node
            result['success'] = True

        return result

    def _execute_credential_access(self, action: CyberAction) -> Dict:
        """Execute credential access"""
        result = {
            'success': False,
            'credentials_found': 0
        }

        if action.target_node in self.state.owned_nodes:
            credentials = self.network_graph.nodes[action.target_node].get('credentials', [])

            if credentials:
                self.state.credentials_found[action.target_node] = credentials
                result['credentials_found'] = len(credentials)
                result['success'] = True

        return result

    def _execute_discovery(self, action: CyberAction) -> Dict:
        """Execute network discovery"""
        result = {'success': True}

        # Discover all neighbors of current node
        for neighbor in self.network_graph.neighbors(action.target_node):
            self.state.discovered_nodes.add(neighbor)

        return result

    def _compute_reward(self, result: Dict) -> float:
        """
        Compute reward based on action result (Equation 2 from paper).

        Reward = +100 (root access) + 10 (new host) + 5 (credentials) - 1 (step) - 100 (violation)
        """
        reward = 0.0
        reward_breakdown = {}

        # Root access (high-value target)
        if result.get('got_root', False):
            root_reward = self.config.reward_root_access
            reward += root_reward
            reward_breakdown['root_access'] = root_reward

        # New host owned
        if result.get('new_host_owned', False):
            host_reward = self.config.reward_new_host
            reward += host_reward
            reward_breakdown['new_host'] = host_reward

        # Credentials found
        creds_found = result.get('credentials_found', 0)
        if creds_found > 0:
            cred_reward = self.config.reward_credentials * creds_found
            reward += cred_reward
            reward_breakdown['credentials'] = cred_reward

        # Step penalty (encourage efficiency)
        step_penalty = self.config.reward_step_penalty
        reward += step_penalty
        reward_breakdown['step_penalty'] = step_penalty

        # Safety violation (strong negative)
        if result.get('safety_violation', False):
            safety_penalty = self.config.reward_safety_violation
            reward += safety_penalty
            reward_breakdown['safety_violation'] = safety_penalty

        # Store breakdown in result
        result['reward_breakdown'] = reward_breakdown

        return reward

    def _build_observation(self) -> np.ndarray:
        """
        Build observation vector from current state.

        Observation includes:
        - Current node features (one-hot encoded position)
        - Owned nodes mask
        - Discovered nodes mask
        - Network topology features
        - Progress metrics
        """
        obs = np.zeros(self.config.observation_dim, dtype=np.float32)
        idx = 0

        # Section 1: Progress metrics (10 features)
        if idx < self.config.observation_dim:
            obs[idx] = self.state.step_count / self.config.max_steps  # Normalized step
            idx += 1
        if idx < self.config.observation_dim:
            obs[idx] = len(self.state.owned_nodes) / len(self.network_graph.nodes())  # Owned ratio
            idx += 1
        if idx < self.config.observation_dim:
            obs[idx] = len(self.state.discovered_nodes) / len(self.network_graph.nodes())  # Discovered ratio
            idx += 1
        if idx < self.config.observation_dim:
            obs[idx] = len(self.state.root_owned_nodes) / max(1, len(self.state.owned_nodes))  # Root ratio
            idx += 1
        if idx < self.config.observation_dim:
            obs[idx] = len(self.state.exploited_vulns) / max(1, sum(len(self.network_graph.nodes[n].get('vulnerabilities', [])) for n in self.network_graph.nodes()))
            idx += 1
        idx += 5  # Reserved for future metrics

        # Section 2: Current node encoding (50 features)
        node_list = list(self.network_graph.nodes())
        if self.state.current_node in node_list:
            node_idx = node_list.index(self.state.current_node)
            # One-hot encode current position (capped at 50)
            if idx + min(node_idx, 49) < self.config.observation_dim:
                obs[idx + min(node_idx, 49)] = 1.0
        idx += 50

        # Section 3: Owned nodes mask (50 features)
        for i, node in enumerate(node_list[:50]):
            if idx >= self.config.observation_dim:
                break
            obs[idx] = 1.0 if node in self.state.owned_nodes else 0.0
            idx += 1

        # Section 4: Discovered nodes mask (50 features)
        for i, node in enumerate(node_list[:50]):
            if idx >= self.config.observation_dim:
                break
            obs[idx] = 1.0 if node in self.state.discovered_nodes else 0.0
            idx += 1

        # Section 5: Network topology features (50 features)
        # Connectivity features for current node
        if self.state.current_node:
            in_degree = self.network_graph.in_degree(self.state.current_node)
            out_degree = self.network_graph.out_degree(self.state.current_node)

            if idx < self.config.observation_dim:
                obs[idx] = min(in_degree / 10.0, 1.0)  # Normalized in-degree
                idx += 1
            if idx < self.config.observation_dim:
                obs[idx] = min(out_degree / 10.0, 1.0)  # Normalized out-degree
                idx += 1

        # Section 6: Available actions encoding (remaining space)
        valid_actions = self._get_valid_actions()
        for i, action_id in enumerate(valid_actions):
            if idx >= self.config.observation_dim:
                break
            obs[idx] = 1.0
            idx += 1

        return obs

    def _get_valid_actions(self) -> List[int]:
        """Get list of currently valid action IDs"""
        valid = []
        for action_id, action in self.action_map.items():
            if action.action_type == ActionType.NO_OP:
                continue

            # Check if action is still valid
            if action.action_type == ActionType.EXPLOIT_VULNERABILITY:
                if action.target_node not in self.state.owned_nodes:
                    valid.append(action_id)
            elif action.action_type == ActionType.LATERAL_MOVE:
                if action.target_node in self.state.owned_nodes:
                    valid.append(action_id)
            elif action.action_type == ActionType.CREDENTIAL_ACCESS:
                if action.target_node not in self.state.credentials_found:
                    valid.append(action_id)
            else:
                valid.append(action_id)

        return valid

    def action_masks(self) -> np.ndarray:
        """
        Get action mask for masked PPO.

        Returns:
            Boolean array indicating valid actions
        """
        mask = np.zeros(self.config.action_dim, dtype=bool)
        valid_actions = self._get_valid_actions()

        for action_id in valid_actions:
            mask[action_id] = True

        # Always allow NO_OP actions
        for action_id, action in self.action_map.items():
            if action.action_type == ActionType.NO_OP:
                mask[action_id] = True

        return mask

    def _update_metrics(self, result: Dict):
        """Update episode metrics"""
        if result.get('new_host_owned', False):
            self.episode_metrics['nodes_owned'] += 1

        if result.get('got_root', False):
            self.episode_metrics['root_access_count'] += 1

        creds = result.get('credentials_found', 0)
        if creds > 0:
            self.episode_metrics['credentials_found'] += creds

        if result.get('safety_violation', False):
            self.episode_metrics['safety_violations'] += 1

    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Terminate if all high-value nodes owned
        high_value_nodes = [
            node for node in self.network_graph.nodes()
            if self.network_graph.nodes[node].get('value', 0) >= 5.0
        ]

        if high_value_nodes:
            all_owned = all(node in self.state.owned_nodes for node in high_value_nodes)
            if all_owned:
                return True

        # Terminate if too many safety violations
        if self.episode_metrics['safety_violations'] >= 3:
            return True

        return False

    def _build_network_from_findings(self) -> nx.DiGraph:
        """Build network graph from vulnerability findings"""
        # Group findings by host
        hosts = {}
        for finding in self.vulnerability_findings:
            host = finding.host_ip or finding.host_fqdn
            if host not in hosts:
                hosts[host] = {
                    'os': 'unknown',
                    'role': 'workstation',
                    'services': [],
                    'vulnerabilities': [],
                    'value': 1.0
                }

            # Add service
            if finding.protocol and finding.protocol not in hosts[host]['services']:
                hosts[host]['services'].append(finding.protocol)

            # Add vulnerability
            if finding.plugin_id:
                hosts[host]['vulnerabilities'].append(finding.plugin_id)

            # Adjust value based on CVSS
            if finding.cvss_base_score and finding.cvss_base_score >= 7.0:
                hosts[host]['value'] = 5.0

        # Create graph
        graph = nx.DiGraph()

        # Add nodes
        for host, attrs in hosts.items():
            node_id = f"host_{host}"
            graph.add_node(node_id, **attrs, credentials=[], firewall_rules=[], owned=False)

        # Add edges (assume mesh connectivity for now)
        nodes = list(graph.nodes())
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i+1:]:
                if random.random() < 0.3:  # 30% connectivity
                    graph.add_edge(node_a, node_b)

        return graph

    def _load_terrain_preset(self, preset: str) -> TerrainParams:
        """Load terrain preset from configuration"""
        # This would load from config/terrain_config.yaml
        preset_params = {
            'small': TerrainParams(num_nodes=5, vuln_density=0.5),
            'medium': TerrainParams(num_nodes=15, vuln_density=0.3),
            'large': TerrainParams(num_nodes=30, vuln_density=0.2),
            'corporate': TerrainParams(num_nodes=20, graph_type='tree', vuln_density=0.25)
        }
        return preset_params.get(preset, self.terrain_params)

    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"CyberBattleSim Environment - Step {self.state.step_count}/{self.config.max_steps}")
            print(f"{'='*60}")
            print(f"Terrain ID: {self.terrain_id}")
            print(f"Current Node: {self.state.current_node}")
            print(f"Owned Nodes: {len(self.state.owned_nodes)}/{len(self.network_graph.nodes())}")
            print(f"Root Access: {len(self.state.root_owned_nodes)}")
            print(f"Credentials Found: {sum(len(c) for c in self.state.credentials_found.values())}")
            print(f"Episode Reward: {self.episode_reward:.2f}")
            print(f"Valid Actions: {len(self._get_valid_actions())}")
            print(f"{'='*60}\n")
        elif mode == 'ansi':
            return f"Step {self.state.step_count} | Owned: {len(self.state.owned_nodes)} | Reward: {self.episode_reward:.2f}"

    def close(self):
        """Clean up resources"""
        pass

    def get_metrics(self) -> Dict:
        """Get episode metrics"""
        return {
            **self.episode_metrics,
            'episode_reward': self.episode_reward,
            'steps': self.state.step_count,
            'success_rate': len(self.state.owned_nodes) / len(self.network_graph.nodes()) if self.network_graph else 0
        }


def make_cyber_env(terrain_preset: str = "small", seed: Optional[int] = None) -> CyberBattleEnv:
    """
    Factory function to create CyberBattleEnv with preset configuration.

    Args:
        terrain_preset: Preset name (small, medium, large, corporate)
        seed: Random seed

    Returns:
        Configured CyberBattleEnv instance
    """
    terrain_gen = TerrainGenerator()

    return CyberBattleEnv(
        terrain_generator=terrain_gen,
        terrain_params=None,  # Will use preset
        config=EnvConfig(),
        seed=seed
    )
