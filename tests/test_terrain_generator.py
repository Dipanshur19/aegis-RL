#!/usr/bin/env python3
"""
test_terrain_generator.py - Unit tests for Terrain Generator

Tests synthetic network terrain generation including:
- Topology generation (multiple graph types)
- Node attribute assignment
- Credential generation
- Firewall rules
- Attack path validation
- Deterministic generation via seeds
"""

import pytest
import networkx as nx
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "execution"))

from terrain_generator import TerrainGenerator, TerrainParams, NodeAttributes


class TestTerrainParams:
    """Test TerrainParams dataclass."""

    def test_default_params(self):
        """Test default parameter initialization."""
        params = TerrainParams()

        assert params.num_nodes == 10
        assert params.graph_type == "erdos_renyi"
        assert params.edge_probability == 0.3
        assert params.ensure_connected == True
        assert params.require_attack_path == True

    def test_custom_params(self):
        """Test custom parameter initialization."""
        params = TerrainParams(
            num_nodes=20,
            graph_type="barabasi_albert",
            edge_probability=0.5
        )

        assert params.num_nodes == 20
        assert params.graph_type == "barabasi_albert"
        assert params.edge_probability == 0.5

    def test_os_distribution_sum(self):
        """Test that OS distribution sums to 1.0."""
        params = TerrainParams()
        total = sum(params.os_distribution.values())
        assert abs(total - 1.0) < 1e-6

    def test_role_distribution_sum(self):
        """Test that role distribution sums to 1.0."""
        params = TerrainParams()
        total = sum(params.role_distribution.values())
        assert abs(total - 1.0) < 1e-6


class TestTerrainGenerator:
    """Test TerrainGenerator functionality."""

    @pytest.fixture
    def generator(self):
        """Create a test terrain generator."""
        return TerrainGenerator()

    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.params is not None
        assert isinstance(generator.params, TerrainParams)

    def test_generate_terrain_basic(self, generator):
        """Test basic terrain generation."""
        params = TerrainParams(num_nodes=5)
        graph, terrain_id = generator.generate_terrain(params, seed=42)

        assert isinstance(graph, nx.DiGraph)
        assert len(graph.nodes()) == 5
        assert isinstance(terrain_id, str)
        assert len(terrain_id) > 0

    def test_generate_terrain_deterministic(self, generator):
        """Test that same seed produces same terrain."""
        params = TerrainParams(num_nodes=10)

        graph1, id1 = generator.generate_terrain(params, seed=42)
        graph2, id2 = generator.generate_terrain(params, seed=42)

        # Same terrain ID
        assert id1 == id2

        # Same number of nodes and edges
        assert len(graph1.nodes()) == len(graph2.nodes())
        assert len(graph1.edges()) == len(graph2.edges())

    def test_generate_terrain_different_seeds(self, generator):
        """Test that different seeds produce different terrains."""
        params = TerrainParams(num_nodes=10)

        graph1, id1 = generator.generate_terrain(params, seed=42)
        graph2, id2 = generator.generate_terrain(params, seed=99)

        # Different terrain IDs
        assert id1 != id2


class TestTopologyGeneration:
    """Test different topology generation methods."""

    @pytest.fixture
    def generator(self):
        return TerrainGenerator()

    def test_erdos_renyi_topology(self, generator):
        """Test Erdos-Renyi random graph generation."""
        params = TerrainParams(
            num_nodes=10,
            graph_type="erdos_renyi",
            edge_probability=0.3
        )

        graph, _ = generator.generate_terrain(params, seed=42)

        assert len(graph.nodes()) == 10
        assert isinstance(graph, nx.DiGraph)

    def test_barabasi_albert_topology(self, generator):
        """Test Barabasi-Albert scale-free graph generation."""
        params = TerrainParams(
            num_nodes=10,
            graph_type="barabasi_albert",
            attachment=2
        )

        graph, _ = generator.generate_terrain(params, seed=42)

        assert len(graph.nodes()) == 10
        assert isinstance(graph, nx.DiGraph)

    def test_scale_free_topology(self, generator):
        """Test scale-free graph generation."""
        params = TerrainParams(
            num_nodes=10,
            graph_type="scale_free"
        )

        graph, _ = generator.generate_terrain(params, seed=42)

        assert len(graph.nodes()) == 10
        assert isinstance(graph, nx.DiGraph)

    def test_tree_topology(self, generator):
        """Test tree topology generation."""
        params = TerrainParams(
            num_nodes=10,
            graph_type="tree"
        )

        graph, _ = generator.generate_terrain(params, seed=42)

        assert len(graph.nodes()) == 10
        assert isinstance(graph, nx.DiGraph)
        # Tree should have n-1 edges (considering it's converted to directed)
        assert len(graph.edges()) >= 9

    def test_invalid_topology_type(self, generator):
        """Test that invalid topology type raises error."""
        params = TerrainParams(graph_type="invalid_type")

        with pytest.raises(ValueError, match="Unknown graph type"):
            generator.generate_terrain(params, seed=42)

    def test_connectivity(self, generator):
        """Test that generated graph is connected when required."""
        params = TerrainParams(
            num_nodes=10,
            graph_type="erdos_renyi",
            ensure_connected=True
        )

        graph, _ = generator.generate_terrain(params, seed=42)

        # Graph should be weakly connected
        assert nx.is_weakly_connected(graph)


class TestNodeAttributes:
    """Test node attribute assignment."""

    @pytest.fixture
    def generator(self):
        return TerrainGenerator()

    @pytest.fixture
    def sample_graph(self, generator):
        """Generate a small sample graph."""
        params = TerrainParams(num_nodes=5)
        graph, _ = generator.generate_terrain(params, seed=42)
        return graph

    def test_node_has_required_attributes(self, sample_graph):
        """Test that nodes have all required attributes."""
        required_attrs = ['os', 'role', 'services', 'vulnerabilities',
                         'credentials', 'firewall_rules', 'value', 'owned']

        for node in sample_graph.nodes():
            node_data = sample_graph.nodes[node]
            for attr in required_attrs:
                assert attr in node_data

    def test_os_values(self, sample_graph):
        """Test that OS values are valid."""
        valid_os = ['linux', 'windows', 'macos']

        for node in sample_graph.nodes():
            os = sample_graph.nodes[node]['os']
            assert os in valid_os

    def test_role_values(self, sample_graph):
        """Test that role values are valid."""
        valid_roles = ['workstation', 'server', 'router', 'firewall']

        for node in sample_graph.nodes():
            role = sample_graph.nodes[node]['role']
            assert role in valid_roles

    def test_services_not_empty(self, sample_graph):
        """Test that services list is not empty."""
        for node in sample_graph.nodes():
            services = sample_graph.nodes[node]['services']
            assert isinstance(services, list)
            assert len(services) > 0

    def test_vulnerabilities_type(self, sample_graph):
        """Test that vulnerabilities is a list."""
        for node in sample_graph.nodes():
            vulns = sample_graph.nodes[node]['vulnerabilities']
            assert isinstance(vulns, list)

    def test_credentials_type(self, sample_graph):
        """Test that credentials is a list."""
        for node in sample_graph.nodes():
            creds = sample_graph.nodes[node]['credentials']
            assert isinstance(creds, list)

    def test_firewall_rules_type(self, sample_graph):
        """Test that firewall_rules is a list."""
        for node in sample_graph.nodes():
            rules = sample_graph.nodes[node]['firewall_rules']
            assert isinstance(rules, list)

    def test_node_value_positive(self, sample_graph):
        """Test that node value is positive."""
        for node in sample_graph.nodes():
            value = sample_graph.nodes[node]['value']
            assert isinstance(value, (int, float))
            assert value >= 0

    def test_owned_initially_false(self, sample_graph):
        """Test that nodes are not owned initially."""
        for node in sample_graph.nodes():
            owned = sample_graph.nodes[node]['owned']
            assert owned == False


class TestCredentialGeneration:
    """Test credential generation."""

    @pytest.fixture
    def generator(self):
        return TerrainGenerator()

    def test_credentials_structure(self, generator):
        """Test that credentials have correct structure."""
        params = TerrainParams(
            num_nodes=10,
            cred_density=0.5  # High density to ensure some credentials
        )

        graph, _ = generator.generate_terrain(params, seed=42)

        # Find nodes with credentials
        nodes_with_creds = [n for n in graph.nodes()
                           if len(graph.nodes[n]['credentials']) > 0]

        # Should have at least some nodes with credentials
        assert len(nodes_with_creds) > 0

        # Check credential structure
        for node in nodes_with_creds:
            creds = graph.nodes[node]['credentials']
            for cred in creds:
                assert isinstance(cred, dict)
                assert 'username' in cred
                assert 'password' in cred

    def test_max_credentials_per_node(self, generator):
        """Test that max credentials limit is respected."""
        params = TerrainParams(
            num_nodes=10,
            cred_density=1.0,  # Force credentials
            max_creds_per_node=3
        )

        graph, _ = generator.generate_terrain(params, seed=42)

        for node in graph.nodes():
            creds = graph.nodes[node]['credentials']
            assert len(creds) <= params.max_creds_per_node


class TestFirewallRules:
    """Test firewall rule generation."""

    @pytest.fixture
    def generator(self):
        return TerrainGenerator()

    def test_firewall_rules_type(self, generator):
        """Test that firewall rules are strings."""
        params = TerrainParams(
            num_nodes=10,
            firewall_probability=0.5
        )

        graph, _ = generator.generate_terrain(params, seed=42)

        for node in graph.nodes():
            rules = graph.nodes[node]['firewall_rules']
            assert isinstance(rules, list)
            for rule in rules:
                assert isinstance(rule, str)


class TestTerrainScaling:
    """Test terrain generation at different scales."""

    @pytest.fixture
    def generator(self):
        return TerrainGenerator()

    def test_small_terrain(self, generator):
        """Test generation of small terrain (5 nodes)."""
        params = TerrainParams(num_nodes=5)
        graph, _ = generator.generate_terrain(params, seed=42)

        assert len(graph.nodes()) == 5

    def test_medium_terrain(self, generator):
        """Test generation of medium terrain (50 nodes)."""
        params = TerrainParams(num_nodes=50)
        graph, _ = generator.generate_terrain(params, seed=42)

        assert len(graph.nodes()) == 50

    def test_large_terrain(self, generator):
        """Test generation of large terrain (100 nodes)."""
        params = TerrainParams(num_nodes=100)
        graph, _ = generator.generate_terrain(params, seed=42)

        assert len(graph.nodes()) == 100


class TestTerrainID:
    """Test terrain ID generation and uniqueness."""

    @pytest.fixture
    def generator(self):
        return TerrainGenerator()

    def test_terrain_id_format(self, generator):
        """Test that terrain ID is a valid hash."""
        params = TerrainParams(num_nodes=10)
        _, terrain_id = generator.generate_terrain(params, seed=42)

        assert isinstance(terrain_id, str)
        assert len(terrain_id) > 0
        # Should have terrain_ prefix followed by hexadecimal hash
        assert terrain_id.startswith("terrain_")
        hex_part = terrain_id[len("terrain_"):]
        assert all(c in '0123456789abcdef' for c in hex_part.lower())

    def test_terrain_id_uniqueness(self, generator):
        """Test that different terrains have different IDs."""
        params1 = TerrainParams(num_nodes=10)
        params2 = TerrainParams(num_nodes=20)

        _, id1 = generator.generate_terrain(params1, seed=42)
        _, id2 = generator.generate_terrain(params2, seed=42)

        assert id1 != id2


def test_integration_full_generation():
    """Integration test: Generate complete terrain and verify all aspects."""
    generator = TerrainGenerator()
    params = TerrainParams(
        num_nodes=15,
        graph_type="erdos_renyi",
        edge_probability=0.3,
        ensure_connected=True,
        vuln_density=0.4,
        cred_density=0.3
    )

    graph, terrain_id = generator.generate_terrain(params, seed=42)

    # Basic structure
    assert len(graph.nodes()) == 15
    assert nx.is_weakly_connected(graph)

    # All nodes have attributes
    for node in graph.nodes():
        assert 'os' in graph.nodes[node]
        assert 'role' in graph.nodes[node]
        assert 'services' in graph.nodes[node]
        assert len(graph.nodes[node]['services']) > 0

    # Terrain ID is valid
    assert isinstance(terrain_id, str)
    assert len(terrain_id) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
