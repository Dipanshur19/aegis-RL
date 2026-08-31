#!/usr/bin/env python3
"""
test_integration.py - Integration tests for AUVAP-PPO

Tests the integration of multiple components:
- Parser → Classifier → Policy → Task Manager
- PPO Agent → Terrain Generator
- LLM-DRL Bridge → Sandbox → Memory
- End-to-end pipeline execution
"""

import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "execution"))


class TestParserClassifierIntegration:
    """Test integration between parser and classifier."""

    def test_parser_to_classifier_flow(self):
        """Test parsing findings and passing to classifier."""
        from parser import VAFinding

        # Create sample finding using actual VAFinding constructor
        finding = VAFinding(
            host_ip="192.168.1.100",
            hostname="target-host",
            os="Linux",
            port=80,
            protocol="tcp",
            service="http",
            severity_text="High",
            cvss=7.5,
            cve="CVE-2023-1234",
            title="SQL Injection",
            description="SQL injection vulnerability",
            evidence="",
            remediation="Patch system",
            raw_plugin_id="12345",
            raw_plugin_family="Web"
        )

        # Verify finding structure
        assert finding.host_ip == "192.168.1.100"
        assert finding.port == 80
        assert finding.cvss == 7.5

        # Convert to dict for classifier
        finding_dict = {
            'host_ip': finding.host_ip,
            'port': finding.port,
            'service': finding.service,
            'cvss': finding.cvss,
            'severity_text': finding.severity_text,
            'title': finding.title
        }

        assert isinstance(finding_dict, dict)
        assert 'cvss' in finding_dict


class TestClassifierPolicyIntegration:
    """Test integration between classifier and policy engine."""

    def test_classified_findings_to_policy(self):
        """Test passing classified findings to policy engine."""
        from policy_engine import PolicyEngine

        # Create sample classified finding
        classified_finding = {
            'host_ip': '192.168.1.100',
            'port': 22,
            'service': 'ssh',
            'cvss': 9.8,
            'severity_bucket': 'Critical',
            'attack_vector': 'Network',
            'automation_candidate': True,
            'llm_confidence': 0.95
        }

        # Create policy engine
        engine = PolicyEngine()

        # Apply policy — evaluate returns tuple (action, reason, rule_id)
        result = engine.evaluate(classified_finding)

        # Should return a tuple
        assert isinstance(result, tuple)
        assert len(result) == 3


class TestPolicyTaskManagerIntegration:
    """Test integration between policy engine and task manager."""

    def test_policy_filtered_findings_to_tasks(self):
        """Test creating tasks from policy-approved findings."""
        from task_manager import initialize_tasks, ExploitTask

        # Create policy-approved findings
        approved_findings = [
            {
                'finding_id': 'find_001',
                'host_ip': '192.168.1.100',
                'port': 22,
                'service': 'ssh',
                'cvss': 9.8,
                'severity_bucket': 'Critical',
                'attack_vector': 'Network',
                'automation_candidate': True,
                'title': 'SSH RCE',
                'cve': 'CVE-2023-0001'
            },
            {
                'finding_id': 'find_002',
                'host_ip': '192.168.1.101',
                'port': 80,
                'service': 'http',
                'cvss': 7.5,
                'severity_bucket': 'High',
                'attack_vector': 'Network',
                'automation_candidate': True,
                'title': 'SQL Injection',
                'cve': 'CVE-2023-0002'
            }
        ]

        # Initialize tasks
        tasks = initialize_tasks(approved_findings)

        assert len(tasks) == 2
        assert all(isinstance(task, ExploitTask) for task in tasks)
        # Should be sorted by priority (risk_score)
        assert tasks[0].priority >= tasks[1].priority


class TestPPOAgentIntegration:
    """Test PPO agent action selection."""

    def test_ppo_agent_action_selection(self):
        """Test PPO agent action selection."""
        from ppo_agent import PPOAgent

        agent = PPOAgent(state_dim=128, action_dim=50, device='cpu')

        obs = np.random.randn(128).astype(np.float32)
        action, log_prob, value = agent.select_action(obs)

        assert 0 <= action < 50
        assert isinstance(log_prob, float)
        assert isinstance(value, float)


class TestTerrainGeneratorIntegration:
    """Test terrain generator with other components."""

    def test_terrain_generation(self):
        """Test generating terrain."""
        from terrain_generator import TerrainGenerator, TerrainParams
        import networkx as nx

        generator = TerrainGenerator()
        params = TerrainParams(num_nodes=10)
        graph, terrain_id = generator.generate_terrain(params, seed=42)

        assert isinstance(graph, nx.DiGraph)
        assert len(graph.nodes()) == 10
        assert nx.is_weakly_connected(graph)

        for node in graph.nodes():
            assert 'os' in graph.nodes[node]
            assert 'services' in graph.nodes[node]
            assert 'vulnerabilities' in graph.nodes[node]


class TestLLMDRLSandboxIntegration:
    """Test LLM-DRL bridge with sandbox executor."""

    def test_bridge_initialization(self):
        """Test LLM-DRL bridge can be created with mock sandbox."""
        from llm_drl_bridge import LLMDRLBridge
        from sandbox_executor import SandboxExecutor
        from persistent_memory import PersistentMemory

        mock_sandbox = Mock(spec=SandboxExecutor)
        mock_memory = Mock(spec=PersistentMemory)

        bridge = LLMDRLBridge(
            sandbox_executor=mock_sandbox,
            persistent_memory=mock_memory,
            verbose=False
        )

        assert bridge.sandbox == mock_sandbox
        assert bridge.memory == mock_memory


class TestEndToEndPipeline:
    """Test end-to-end pipeline integration."""

    def test_minimal_pipeline_flow(self):
        """Test minimal end-to-end flow through task creation."""
        from task_manager import initialize_tasks

        finding_dict = {
            'finding_id': 'e2e_001',
            'host_ip': '10.0.0.1',
            'port': 22,
            'service': 'ssh',
            'cvss': 9.8,
            'severity_text': 'Critical',
            'severity_bucket': 'Critical',
            'attack_vector': 'Network',
            'automation_candidate': True,
            'title': 'SSH RCE',
            'cve': 'CVE-2023-9999'
        }

        tasks = initialize_tasks([finding_dict])

        assert len(tasks) == 1
        assert tasks[0].risk_score > 0

    def test_pipeline_with_multiple_findings(self):
        """Test pipeline with multiple findings."""
        from task_manager import initialize_tasks

        findings = []
        for i in range(5):
            finding_dict = {
                'finding_id': f"multi_{i}",
                'host_ip': f"10.0.0.{i+1}",
                'port': 22 + i,
                'service': 'ssh',
                'cvss': 9.0 - i * 0.5,
                'severity_bucket': 'Critical',
                'attack_vector': 'Network',
                'automation_candidate': True,
                'title': f"Vulnerability {i}",
                'cve': f"CVE-2023-{1000+i}"
            }
            findings.append(finding_dict)

        tasks = initialize_tasks(findings)

        assert len(tasks) == 5
        assert tasks[0].priority >= tasks[1].priority
        assert tasks[1].priority >= tasks[2].priority


class TestMemoryPersistence:
    """Test persistent memory integration."""

    def test_memory_store_and_retrieve(self):
        """Test storing and retrieving execution history."""
        from persistent_memory import PersistentMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            memory = PersistentMemory(str(db_path))

            try:
                # Store attempt
                attempt_id = memory.store_outcome(
                    finding_type="sql_injection",
                    cve="CVE-2023-1234",
                    service="http",
                    target_os="linux",
                    script_content="print('test')",
                    success=True,
                    error_message="",
                    execution_trace="success",
                    cvss_score=7.5,
                    metadata={"test": True}
                )

                assert isinstance(attempt_id, str)
                assert len(attempt_id) > 0

                # Retrieve similar attempts
                attempts = memory.get_similar_attempts("sql_injection")
                assert len(attempts) > 0

                # Check success rate
                rate = memory.get_success_rate("sql_injection", "CVE-2023-1234", "http")
                assert rate == 1.0

            finally:
                memory.close()


def test_integration_stress_test():
    """Stress test with many findings."""
    from task_manager import initialize_tasks

    findings = []
    for i in range(100):
        finding = {
            'finding_id': f"stress_{i}",
            'host_ip': f"10.0.{i // 256}.{i % 256}",
            'port': 1000 + i,
            'service': ['ssh', 'http', 'ftp'][i % 3],
            'cvss': 5.0 + (i % 5),
            'severity_bucket': ['Medium', 'High', 'Critical'][i % 3],
            'attack_vector': 'Network',
            'automation_candidate': i % 2 == 0,
            'title': f"Vulnerability {i}",
            'cve': f"CVE-2023-{10000+i}"
        }
        findings.append(finding)

    tasks = initialize_tasks(findings)

    assert len(tasks) == 100
    assert all(task.risk_score > 0 for task in tasks)
    assert tasks[0].priority == max(t.priority for t in tasks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
