#!/usr/bin/env python3
"""
test_sandbox_executor.py - Unit tests for Sandbox Executor

Tests Docker-based sandbox execution including:
- Container initialization
- Script execution
- Resource limits
- Timeout handling
- Safety features
"""

import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "execution"))

from sandbox_executor import SandboxExecutor, ExecutionResult


class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_execution_result_basic(self):
        """Test basic ExecutionResult creation."""
        result = ExecutionResult(
            status="success",
            exit_code=0,
            stdout="test output",
            stderr="",
            duration=1.5,
            logs=["log1", "log2"]
        )

        assert result.status == "success"
        assert result.exit_code == 0
        assert result.stdout == "test output"
        assert result.duration == 1.5
        assert len(result.logs) == 2

    def test_execution_result_defaults(self):
        """Test ExecutionResult with default values."""
        result = ExecutionResult(
            status="success",
            exit_code=0,
            stdout="",
            stderr="",
            duration=0.0,
            logs=[]
        )

        assert result.artifacts == {}
        assert result.safety_violations == []

    def test_execution_result_with_artifacts(self):
        """Test ExecutionResult with artifacts."""
        artifacts = {"key": "value", "data": [1, 2, 3]}
        result = ExecutionResult(
            status="success",
            exit_code=0,
            stdout="",
            stderr="",
            duration=0.0,
            logs=[],
            artifacts=artifacts
        )

        assert result.artifacts == artifacts

    def test_execution_result_with_violations(self):
        """Test ExecutionResult with safety violations."""
        violations = ["network_access", "file_write"]
        result = ExecutionResult(
            status="failure",
            exit_code=1,
            stdout="",
            stderr="",
            duration=0.0,
            logs=[],
            safety_violations=violations
        )

        assert len(result.safety_violations) == 2
        assert "network_access" in result.safety_violations


class TestSandboxExecutorInitialization:
    """Test SandboxExecutor initialization."""

    @patch('sandbox_executor.docker.from_env')
    def test_executor_basic_init(self, mock_docker):
        """Test basic executor initialization."""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        mock_client.ping.return_value = True

        executor = SandboxExecutor()

        assert executor.docker_image == "python:3.10-slim"
        assert executor.default_timeout == 10
        assert executor.memory_limit == "512m"
        assert executor.cpu_count == 1.0
        assert executor.enable_network == False

    @patch('sandbox_executor.docker.from_env')
    def test_executor_custom_config(self, mock_docker):
        """Test executor with custom configuration."""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        mock_client.ping.return_value = True

        executor = SandboxExecutor(
            docker_image="python:3.11",
            default_timeout=30,
            memory_limit="1g",
            cpu_count=2.0,
            enable_network=True
        )

        assert executor.docker_image == "python:3.11"
        assert executor.default_timeout == 30
        assert executor.memory_limit == "1g"
        assert executor.cpu_count == 2.0
        assert executor.enable_network == True

    @patch('sandbox_executor.docker.from_env')
    def test_executor_docker_unavailable(self, mock_docker):
        """Test executor when Docker is unavailable."""
        mock_docker.side_effect = Exception("Docker not running")

        executor = SandboxExecutor()

        assert executor.client is None


class TestScriptExecution:
    """Test script execution in sandbox."""

    @pytest.fixture
    def test_script(self, tmp_path):
        """Create a simple test script."""
        script_path = tmp_path / "test_script.py"
        script_path.write_text("print('Hello from sandbox')\n")
        return str(script_path)

    @pytest.fixture
    def failing_script(self, tmp_path):
        """Create a script that fails."""
        script_path = tmp_path / "fail_script.py"
        script_path.write_text("import sys\nsys.exit(1)\n")
        return str(script_path)

    @pytest.fixture
    def timeout_script(self, tmp_path):
        """Create a script that times out."""
        script_path = tmp_path / "timeout_script.py"
        script_path.write_text("import time\ntime.sleep(100)\n")
        return str(script_path)

    @patch('sandbox_executor.docker.from_env')
    def test_execute_simple_script(self, mock_docker, test_script):
        """Test executing a simple successful script."""
        # Mock Docker client and container
        mock_client = Mock()
        mock_container = Mock()
        mock_docker.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.containers.run.return_value = mock_container

        # Mock container behavior
        mock_container.wait.return_value = {'StatusCode': 0}
        mock_container.logs.return_value = b"Hello from sandbox\n"

        executor = SandboxExecutor()
        result = executor.execute_task("test_task", test_script)

        # Verify container was configured correctly
        call_args = mock_client.containers.run.call_args[1]
        assert call_args['image'] == "python:3.10-slim"
        assert call_args['network_disabled'] == True
        assert call_args['mem_limit'] == "512m"

        # Verify result
        assert result.status == "success"
        assert result.exit_code == 0

    @patch('sandbox_executor.docker.from_env')
    def test_execute_with_custom_timeout(self, mock_docker, test_script):
        """Test executing with custom timeout."""
        mock_client = Mock()
        mock_container = Mock()
        mock_docker.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.containers.run.return_value = mock_container
        mock_container.wait.return_value = {'StatusCode': 0}
        mock_container.logs.return_value = b"output\n"

        executor = SandboxExecutor(default_timeout=10)
        result = executor.execute_task("test_task", test_script, timeout=30)

        # Verify timeout was passed to container.wait
        mock_container.wait.assert_called_once_with(timeout=30)

    @patch('sandbox_executor.docker.from_env')
    def test_execute_with_env_vars(self, mock_docker, test_script):
        """Test executing with environment variables."""
        mock_client = Mock()
        mock_container = Mock()
        mock_docker.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.containers.run.return_value = mock_container
        mock_container.wait.return_value = {'StatusCode': 0}
        mock_container.logs.return_value = b"output\n"

        executor = SandboxExecutor()
        env_vars = {"TEST_VAR": "test_value"}
        result = executor.execute_task("test_task", test_script, env_vars=env_vars)

        # Verify environment variables were passed
        call_args = mock_client.containers.run.call_args[1]
        assert call_args['environment'] == env_vars

    @patch('sandbox_executor.docker.from_env')
    def test_execute_failing_script(self, mock_docker, failing_script):
        """Test executing a script that exits with error."""
        mock_client = Mock()
        mock_container = Mock()
        mock_docker.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.containers.run.return_value = mock_container
        mock_container.wait.return_value = {'StatusCode': 1}
        mock_container.logs.return_value = b""

        executor = SandboxExecutor()
        result = executor.execute_task("test_task", failing_script)

        assert result.status == "failure"
        assert result.exit_code == 1

    @patch('sandbox_executor.docker.from_env')
    def test_execute_timeout_script(self, mock_docker, timeout_script):
        """Test executing a script that times out."""
        import requests

        mock_client = Mock()
        mock_container = Mock()
        mock_docker.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.containers.run.return_value = mock_container

        # Simulate timeout
        mock_container.wait.side_effect = requests.exceptions.ReadTimeout("Container timeout")
        mock_container.logs.return_value = b"partial output\n"

        executor = SandboxExecutor(default_timeout=5)
        result = executor.execute_task("test_task", timeout_script)

        assert result.status == "timeout"
        assert result.exit_code == -1
        # Verify container was killed
        mock_container.kill.assert_called_once()


class TestResourceLimits:
    """Test resource limit enforcement."""

    @patch('sandbox_executor.docker.from_env')
    def test_memory_limit_configuration(self, mock_docker, tmp_path):
        """Test that memory limit is properly configured."""
        mock_client = Mock()
        mock_container = Mock()
        mock_docker.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.containers.run.return_value = mock_container
        mock_container.wait.return_value = {'StatusCode': 0}
        mock_container.logs.return_value = b""

        script = tmp_path / "test.py"
        script.write_text("print('test')")

        executor = SandboxExecutor(memory_limit="256m")
        result = executor.execute_task("test_task", str(script))

        # Verify memory limit was set
        call_args = mock_client.containers.run.call_args[1]
        assert call_args['mem_limit'] == "256m"

    @patch('sandbox_executor.docker.from_env')
    def test_cpu_limit_configuration(self, mock_docker, tmp_path):
        """Test that CPU limit is properly configured."""
        mock_client = Mock()
        mock_container = Mock()
        mock_docker.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.containers.run.return_value = mock_container
        mock_container.wait.return_value = {'StatusCode': 0}
        mock_container.logs.return_value = b""

        script = tmp_path / "test.py"
        script.write_text("print('test')")

        executor = SandboxExecutor(cpu_count=2.0)
        result = executor.execute_task("test_task", str(script))

        # Verify CPU limit was set (nano_cpus = cpus * 1e9)
        call_args = mock_client.containers.run.call_args[1]
        assert call_args['nano_cpus'] == int(2.0 * 1e9)


class TestNetworkIsolation:
    """Test network isolation features."""

    @patch('sandbox_executor.docker.from_env')
    def test_network_disabled_by_default(self, mock_docker, tmp_path):
        """Test that network is disabled by default."""
        mock_client = Mock()
        mock_container = Mock()
        mock_docker.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.containers.run.return_value = mock_container
        mock_container.wait.return_value = {'StatusCode': 0}
        mock_container.logs.return_value = b""

        script = tmp_path / "test.py"
        script.write_text("print('test')")

        executor = SandboxExecutor()
        result = executor.execute_task("test_task", str(script))

        # Verify network is disabled
        call_args = mock_client.containers.run.call_args[1]
        assert call_args['network_disabled'] == True

    @patch('sandbox_executor.docker.from_env')
    def test_network_enabled_when_configured(self, mock_docker, tmp_path):
        """Test that network can be enabled when configured."""
        mock_client = Mock()
        mock_container = Mock()
        mock_docker.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.containers.run.return_value = mock_container
        mock_container.wait.return_value = {'StatusCode': 0}
        mock_container.logs.return_value = b""

        script = tmp_path / "test.py"
        script.write_text("print('test')")

        executor = SandboxExecutor(enable_network=True)
        result = executor.execute_task("test_task", str(script))

        # Verify network is enabled
        call_args = mock_client.containers.run.call_args[1]
        assert call_args['network_disabled'] == False


class TestFallbackExecution:
    """Test fallback execution when Docker is unavailable."""

    @patch('sandbox_executor.docker.from_env')
    def test_local_execution_fallback(self, mock_docker, tmp_path):
        """Test that local execution works as fallback."""
        mock_docker.side_effect = Exception("Docker not available")

        script = tmp_path / "test.py"
        script.write_text("print('Hello from fallback')")

        executor = SandboxExecutor()
        result = executor.execute_task("test_task", str(script))

        # Should still get a result (from local execution)
        assert isinstance(result, ExecutionResult)
        assert result.exit_code == 0


def test_integration_sandbox_execution(tmp_path):
    """Integration test for full sandbox execution flow."""
    # Create a simple script
    script = tmp_path / "integration_test.py"
    script.write_text("""
import sys
print("Integration test successful")
sys.exit(0)
""")

    # This test will use mock if Docker is unavailable
    with patch('sandbox_executor.docker.from_env') as mock_docker:
        mock_client = Mock()
        mock_container = Mock()
        mock_docker.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.containers.run.return_value = mock_container
        mock_container.wait.return_value = {'StatusCode': 0}
        mock_container.logs.return_value = b"Integration test successful\n"

        executor = SandboxExecutor(
            memory_limit="512m",
            cpu_count=1.0,
            default_timeout=30,
            enable_network=False
        )

        result = executor.execute_task(
            task_id="integration_test",
            script_path=str(script),
            timeout=60
        )

        assert result.status == "success"
        assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
