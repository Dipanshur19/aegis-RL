"""
Observation Space Builder for AUVAP-PPO Integration

This module handles the conversion of network state and vulnerability data
into observation vectors for reinforcement learning.
"""

import numpy as np
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import networkx as nx

# AUVAP imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import VAFinding
from task_manager import ExploitTask


@dataclass
class ObservationConfig:
    """Configuration for observation space"""
    max_nodes: int = 20
    max_vulns_per_node: int = 10
    include_network_topology: bool = True
    include_vuln_features: bool = True
    include_temporal_features: bool = True
    normalize: bool = True


class ObservationBuilder:
    """
    Builds observation vectors from environment state.

    The observation includes:
    - Network topology features (adjacency, connectivity)
    - Node features (owned, discovered, vulnerable)
    - Vulnerability features (CVSS, severity, exploitability)
    - Temporal features (steps, progress)
    """

    def __init__(self, config: ObservationConfig = None):
        """
        Initialize the observation builder.

        Args:
            config: Configuration for observation space
        """
        self.config = config or ObservationConfig()
        self.observation_dim = self._calculate_observation_dim()

    def _calculate_observation_dim(self) -> int:
        """Calculate the total dimension of observation space."""
        dim = 0

        # Network topology features
        if self.config.include_network_topology:
            # Adjacency matrix (flattened)
            dim += self.config.max_nodes * self.config.max_nodes
            # Node ownership vector
            dim += self.config.max_nodes
            # Node discovery vector
            dim += self.config.max_nodes

        # Vulnerability features
        if self.config.include_vuln_features:
            # Per-node vulnerability count
            dim += self.config.max_nodes
            # Max CVSS per node
            dim += self.config.max_nodes
            # Exploitability scores
            dim += self.config.max_nodes

        # Temporal features
        if self.config.include_temporal_features:
            # Current step (normalized)
            dim += 1
            # Success rate
            dim += 1
            # Total exploited count
            dim += 1

        return dim

    def build_observation(self,
                         network: nx.DiGraph = None,
                         owned_nodes: Set[str] = None,
                         discovered_nodes: Set[str] = None,
                         vulnerabilities: Dict[str, List[VAFinding]] = None,
                         current_step: int = 0,
                         max_steps: int = 100,
                         exploited_count: int = 0) -> np.ndarray:
        """
        Build an observation vector from the current state.

        Args:
            network: Network topology graph
            owned_nodes: Set of nodes owned by the attacker
            discovered_nodes: Set of discovered nodes
            vulnerabilities: Map of node -> list of vulnerabilities
            current_step: Current step number
            max_steps: Maximum steps allowed
            exploited_count: Number of successfully exploited vulnerabilities

        Returns:
            Observation vector as numpy array
        """
        observation = np.zeros(self.observation_dim, dtype=np.float32)
        idx = 0

        # Initialize defaults
        owned_nodes = owned_nodes or set()
        discovered_nodes = discovered_nodes or set()
        vulnerabilities = vulnerabilities or {}

        # Network topology features
        if self.config.include_network_topology:
            # Build adjacency matrix
            adjacency = self._build_adjacency_matrix(network)
            adjacency_flat = adjacency.flatten()
            observation[idx:idx+len(adjacency_flat)] = adjacency_flat
            idx += len(adjacency_flat)

            # Node ownership
            ownership = self._build_ownership_vector(network, owned_nodes)
            observation[idx:idx+len(ownership)] = ownership
            idx += len(ownership)

            # Node discovery
            discovery = self._build_discovery_vector(network, discovered_nodes)
            observation[idx:idx+len(discovery)] = discovery
            idx += len(discovery)

        # Vulnerability features
        if self.config.include_vuln_features:
            vuln_counts, max_cvss, exploitability = self._build_vuln_features(
                network, vulnerabilities
            )
            observation[idx:idx+len(vuln_counts)] = vuln_counts
            idx += len(vuln_counts)
            observation[idx:idx+len(max_cvss)] = max_cvss
            idx += len(max_cvss)
            observation[idx:idx+len(exploitability)] = exploitability
            idx += len(exploitability)

        # Temporal features
        if self.config.include_temporal_features:
            # Normalized step count
            observation[idx] = current_step / max(max_steps, 1)
            idx += 1

            # Success rate
            success_rate = exploited_count / max(current_step, 1)
            observation[idx] = success_rate
            idx += 1

            # Normalized exploit count
            observation[idx] = exploited_count / self.config.max_nodes
            idx += 1

        # Normalize if configured
        if self.config.normalize:
            observation = np.clip(observation, 0, 1)

        return observation

    def _build_adjacency_matrix(self, network: nx.DiGraph = None) -> np.ndarray:
        """
        Build adjacency matrix from network graph.

        Args:
            network: Network topology graph

        Returns:
            Adjacency matrix of shape (max_nodes, max_nodes)
        """
        matrix = np.zeros((self.config.max_nodes, self.config.max_nodes), dtype=np.float32)

        if network is None:
            return matrix

        # Create node-to-index mapping
        nodes = list(network.nodes())[:self.config.max_nodes]
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        # Fill adjacency matrix
        for u, v in network.edges():
            if u in node_to_idx and v in node_to_idx:
                matrix[node_to_idx[u], node_to_idx[v]] = 1.0

        return matrix

    def _build_ownership_vector(self,
                                network: nx.DiGraph = None,
                                owned_nodes: Set[str] = None) -> np.ndarray:
        """
        Build vector indicating which nodes are owned.

        Args:
            network: Network topology graph
            owned_nodes: Set of owned node IDs

        Returns:
            Binary vector of shape (max_nodes,)
        """
        vector = np.zeros(self.config.max_nodes, dtype=np.float32)

        if network is None or owned_nodes is None:
            return vector

        nodes = list(network.nodes())[:self.config.max_nodes]
        for i, node in enumerate(nodes):
            if node in owned_nodes:
                vector[i] = 1.0

        return vector

    def _build_discovery_vector(self,
                               network: nx.DiGraph = None,
                               discovered_nodes: Set[str] = None) -> np.ndarray:
        """
        Build vector indicating which nodes are discovered.

        Args:
            network: Network topology graph
            discovered_nodes: Set of discovered node IDs

        Returns:
            Binary vector of shape (max_nodes,)
        """
        vector = np.zeros(self.config.max_nodes, dtype=np.float32)

        if network is None or discovered_nodes is None:
            return vector

        nodes = list(network.nodes())[:self.config.max_nodes]
        for i, node in enumerate(nodes):
            if node in discovered_nodes:
                vector[i] = 1.0

        return vector

    def _build_vuln_features(self,
                            network: nx.DiGraph = None,
                            vulnerabilities: Dict[str, List[VAFinding]] = None) -> tuple:
        """
        Build vulnerability feature vectors.

        Args:
            network: Network topology graph
            vulnerabilities: Map of node -> vulnerabilities

        Returns:
            Tuple of (vuln_counts, max_cvss, exploitability) vectors
        """
        vuln_counts = np.zeros(self.config.max_nodes, dtype=np.float32)
        max_cvss = np.zeros(self.config.max_nodes, dtype=np.float32)
        exploitability = np.zeros(self.config.max_nodes, dtype=np.float32)

        if network is None or vulnerabilities is None:
            return vuln_counts, max_cvss, exploitability

        nodes = list(network.nodes())[:self.config.max_nodes]

        for i, node in enumerate(nodes):
            node_vulns = vulnerabilities.get(node, [])

            # Vulnerability count
            vuln_counts[i] = min(len(node_vulns), self.config.max_vulns_per_node) / self.config.max_vulns_per_node

            if node_vulns:
                # Max CVSS score (normalized)
                max_cvss[i] = max(v.cvss_base_score or 0.0 for v in node_vulns) / 10.0

                # Average exploitability (if available)
                exploitability_scores = [
                    v.cvss_base_score or 0.0 for v in node_vulns
                ]
                if exploitability_scores:
                    exploitability[i] = np.mean(exploitability_scores) / 10.0

        return vuln_counts, max_cvss, exploitability

    def get_observation_dimension(self) -> int:
        """Get the dimension of the observation space."""
        return self.observation_dim

    def get_observation_space_dict(self) -> Dict:
        """
        Get a dictionary describing the observation space structure.

        Returns:
            Dictionary with observation space metadata
        """
        return {
            'dimension': self.observation_dim,
            'network_topology_dim': (self.config.max_nodes * self.config.max_nodes +
                                    self.config.max_nodes * 2) if self.config.include_network_topology else 0,
            'vulnerability_dim': self.config.max_nodes * 3 if self.config.include_vuln_features else 0,
            'temporal_dim': 3 if self.config.include_temporal_features else 0,
            'normalized': self.config.normalize,
            'max_nodes': self.config.max_nodes
        }
