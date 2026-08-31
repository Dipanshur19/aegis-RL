"""
Terrain Generator (Priority 1, Item 2 - Algorithm 5)

Generates synthetic network environments (terrains) for training with:
- NetworkX graph generation (multiple topologies)
- Node attribute assignment (OS, services, vulnerabilities)
- Credential generation
- Firewall rules
- Attack path validation
- Deterministic replay via seed
"""

import networkx as nx
import numpy as np
import hashlib
import yaml
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import random


@dataclass
class NodeAttributes:
    """Attributes for a network node"""
    node_id: str
    os: str  # "windows", "linux", "macos"
    role: str  # "workstation", "server", "router", "firewall"
    services: List[str]  # Running services
    vulnerabilities: List[str]  # CVE IDs
    credentials: List[Dict[str, str]]  # username/password pairs
    firewall_rules: List[str]  # Firewall rules
    value: float  # Node value (for reward calculation)


@dataclass
class TerrainParams:
    """Parameters for terrain generation"""
    num_nodes: int = 10
    graph_type: str = "erdos_renyi"  # "erdos_renyi", "barabasi_albert", "scale_free", "tree"
    edge_probability: float = 0.3  # For erdos_renyi
    attachment: int = 2  # For barabasi_albert

    # OS distribution
    os_distribution: Dict[str, float] = field(default_factory=lambda: {
        "linux": 0.5,
        "windows": 0.4,
        "macos": 0.1
    })

    # Role distribution
    role_distribution: Dict[str, float] = field(default_factory=lambda: {
        "workstation": 0.5,
        "server": 0.3,
        "router": 0.1,
        "firewall": 0.1
    })

    # Vulnerability parameters
    vuln_density: float = 0.3  # Probability of vulnerability per service
    max_vulns_per_node: int = 5

    # Credential parameters
    cred_density: float = 0.2  # Probability of credentials per node
    max_creds_per_node: int = 3

    # Firewall parameters
    firewall_probability: float = 0.2

    # Network parameters
    ensure_connected: bool = True
    require_attack_path: bool = True
    entry_points: int = 2  # Number of entry point nodes


class TerrainGenerator:
    """
    Generates synthetic network terrains for RL training.

    Implements Algorithm 5 from paper.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize terrain generator.

        Args:
            config_path: Optional path to YAML configuration file
        """
        self.config_path = config_path
        self.params = self._load_config(config_path) if config_path else TerrainParams()

    def _load_config(self, path: str) -> TerrainParams:
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return TerrainParams(**config_dict)

    def generate_terrain(self, params: Optional[TerrainParams] = None, seed: Optional[int] = None) -> Tuple[nx.DiGraph, str]:
        """
        Generate a synthetic network terrain.

        Args:
            params: Terrain parameters (uses default if None)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (terrain_graph, terrain_id)
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        params = params or self.params

        # Step 1: Generate network topology
        graph = self._generate_topology(params)

        # Step 2: Assign node attributes
        self._assign_node_attributes(graph, params)

        # Step 3: Generate credentials
        self._generate_credentials(graph, params)

        # Step 4: Apply firewall rules
        self._apply_firewall_rules(graph, params)

        # Step 5: Validate attack paths exist
        if params.require_attack_path:
            if not self._validate_attack_paths(graph, params):
                # Regenerate if no valid attack path
                return self.generate_terrain(params, seed)

        # Step 6: Calculate terrain ID (deterministic hash)
        terrain_id = self._calculate_terrain_id(graph, seed)

        return graph, terrain_id

    def _generate_topology(self, params: TerrainParams) -> nx.DiGraph:
        """
        Generate network topology based on graph type.

        Supported types:
        - erdos_renyi: Random graph with edge probability
        - barabasi_albert: Scale-free preferential attachment
        - scale_free: Scale-free network
        - tree: Hierarchical tree structure
        """
        if params.graph_type == "erdos_renyi":
            # Random graph
            G = nx.erdos_renyi_graph(params.num_nodes, params.edge_probability, directed=True)

        elif params.graph_type == "barabasi_albert":
            # Scale-free network
            G_undirected = nx.barabasi_albert_graph(params.num_nodes, params.attachment)
            G = nx.DiGraph(G_undirected)

        elif params.graph_type == "scale_free":
            # Scale-free directed graph
            G = nx.scale_free_graph(params.num_nodes)

        elif params.graph_type == "tree":
            # Tree structure
            G_undirected = nx.random_labeled_tree(params.num_nodes)
            G = nx.DiGraph(G_undirected)

        else:
            raise ValueError(f"Unknown graph type: {params.graph_type}")

        # Ensure connectivity if required
        if params.ensure_connected:
            G = self._ensure_connected(G)

        # Relabel nodes to strings
        mapping = {i: f"node_{i}" for i in G.nodes()}
        G = nx.relabel_nodes(G, mapping)

        return G

    def _ensure_connected(self, G: nx.DiGraph) -> nx.DiGraph:
        """Ensure graph is weakly connected"""
        if nx.is_weakly_connected(G):
            return G

        # Connect components
        components = list(nx.weakly_connected_components(G))
        for i in range(len(components) - 1):
            node_a = list(components[i])[0]
            node_b = list(components[i + 1])[0]
            G.add_edge(node_a, node_b)

        return G

    def _assign_node_attributes(self, graph: nx.DiGraph, params: TerrainParams):
        """Assign attributes to each node"""
        for node in graph.nodes():
            # Sample OS
            node_os = self._sample_from_distribution(params.os_distribution)

            # Sample role
            role = self._sample_from_distribution(params.role_distribution)

            # Generate services based on role
            services = self._generate_services(role, node_os)

            # Generate vulnerabilities
            vulnerabilities = self._generate_vulnerabilities(services, params)

            # Calculate node value (for rewards)
            value = self._calculate_node_value(role)

            # Store attributes
            graph.nodes[node]['os'] = node_os
            graph.nodes[node]['role'] = role
            graph.nodes[node]['services'] = services
            graph.nodes[node]['vulnerabilities'] = vulnerabilities
            graph.nodes[node]['credentials'] = []
            graph.nodes[node]['firewall_rules'] = []
            graph.nodes[node]['value'] = value
            graph.nodes[node]['owned'] = False

    def _sample_from_distribution(self, distribution: Dict[str, float]) -> str:
        """Sample item from probability distribution"""
        items = list(distribution.keys())
        probabilities = list(distribution.values())
        return np.random.choice(items, p=probabilities)

    def _generate_services(self, role: str, node_os: str) -> List[str]:
        """Generate services based on role and OS"""
        service_map = {
            "workstation": ["ssh", "rdp", "vnc"],
            "server": ["ssh", "http", "https", "smb", "ftp"],
            "router": ["ssh", "telnet", "snmp"],
            "firewall": ["ssh", "https"]
        }

        os_specific = {
            "linux": ["ssh", "apache", "mysql"],
            "windows": ["rdp", "smb", "mssql", "iis"],
            "macos": ["ssh", "vnc", "afp"]
        }

        services = service_map.get(role, ["ssh"])
        services.extend(random.sample(os_specific[node_os], min(2, len(os_specific[node_os]))))

        return list(set(services))  # Remove duplicates

    def _generate_vulnerabilities(self, services: List[str], params: TerrainParams) -> List[str]:
        """Generate vulnerabilities for services"""
        vulns = []

        # Vulnerability database (simplified)
        vuln_database = {
            "ssh": ["CVE-2020-15778", "CVE-2021-28041"],
            "http": ["CVE-2021-41773", "CVE-2021-42013"],
            "https": ["CVE-2021-44228", "CVE-2022-22965"],
            "smb": ["CVE-2017-0144", "CVE-2020-0796"],
            "ftp": ["CVE-2019-12815", "CVE-2021-3711"],
            "rdp": ["CVE-2019-0708", "CVE-2020-0609"],
            "mysql": ["CVE-2021-2471", "CVE-2022-21245"]
        }

        for service in services:
            if random.random() < params.vuln_density:
                service_vulns = vuln_database.get(service, ["CVE-XXXX-XXXX"])
                num_vulns = min(random.randint(1, params.max_vulns_per_node), len(service_vulns))
                vulns.extend(random.sample(service_vulns, num_vulns))

        return vulns[:params.max_vulns_per_node]

    def _generate_credentials(self, graph: nx.DiGraph, params: TerrainParams):
        """Generate credentials for nodes"""
        usernames = ["admin", "root", "user", "guest", "administrator", "service"]
        passwords = ["password123", "admin", "letmein", "welcome", "P@ssw0rd", "12345"]

        for node in graph.nodes():
            if random.random() < params.cred_density:
                num_creds = random.randint(1, params.max_creds_per_node)
                creds = []

                for _ in range(num_creds):
                    creds.append({
                        "username": random.choice(usernames),
                        "password": random.choice(passwords)
                    })

                graph.nodes[node]['credentials'] = creds

    def _apply_firewall_rules(self, graph: nx.DiGraph, params: TerrainParams):
        """Apply firewall rules to nodes"""
        for node in graph.nodes():
            if random.random() < params.firewall_probability:
                rules = [
                    "ALLOW SSH from 192.168.1.0/24",
                    "DENY all from 0.0.0.0/0",
                    "ALLOW HTTPS from any"
                ]
                graph.nodes[node]['firewall_rules'] = random.sample(rules, random.randint(1, len(rules)))

    def _calculate_node_value(self, role: str) -> float:
        """Calculate node value for reward calculation"""
        value_map = {
            "workstation": 1.0,
            "server": 5.0,
            "router": 10.0,
            "firewall": 15.0
        }
        return value_map.get(role, 1.0)

    def _validate_attack_paths(self, graph: nx.DiGraph, params: TerrainParams) -> bool:
        """Validate that attack paths exist from entry points to high-value targets"""
        # Identify entry points (nodes with no incoming edges or low firewall)
        entry_points = [n for n in graph.nodes() if graph.in_degree(n) == 0 or len(graph.nodes[n]['firewall_rules']) == 0]

        if len(entry_points) < params.entry_points:
            return False

        # Identify high-value targets
        high_value_nodes = [n for n in graph.nodes() if graph.nodes[n]['value'] >= 5.0]

        if not high_value_nodes:
            return False

        # Check if paths exist
        for entry in entry_points[:params.entry_points]:
            for target in high_value_nodes:
                if nx.has_path(graph, entry, target):
                    return True

        return False

    def _calculate_terrain_id(self, graph: nx.DiGraph, seed: Optional[int]) -> str:
        """Calculate deterministic terrain ID for replay"""
        # Create hash from graph structure and seed
        data = {
            'nodes': list(graph.nodes()),
            'edges': list(graph.edges()),
            'seed': seed,
            'num_nodes': len(graph.nodes())
        }

        data_str = str(sorted(data.items()))
        terrain_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]

        return f"terrain_{terrain_hash}"

    def save_terrain(self, graph: nx.DiGraph, terrain_id: str, output_dir: str = "terrains"):
        """Save terrain to file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as YAML
        terrain_data: Dict[str, Any] = {
            'terrain_id': terrain_id,
            'nodes': {},
            'edges': list(graph.edges())
        }

        for node in graph.nodes():
            terrain_data['nodes'][node] = dict(graph.nodes[node])

        filepath = output_path / f"{terrain_id}.yaml"
        with open(filepath, 'w') as f:
            yaml.dump(terrain_data, f)

        print(f"[OK] Terrain saved to: {filepath}")

    def load_terrain(self, terrain_id: str, input_dir: str = "terrains") -> nx.DiGraph:
        """Load terrain from file"""
        filepath = Path(input_dir) / f"{terrain_id}.yaml"

        with open(filepath, 'r') as f:
            terrain_data = yaml.safe_load(f)

        # Reconstruct graph
        graph = nx.DiGraph()
        graph.add_edges_from(terrain_data['edges'])

        for node, attrs in terrain_data['nodes'].items():
            for key, value in attrs.items():
                graph.nodes[node][key] = value

        return graph
