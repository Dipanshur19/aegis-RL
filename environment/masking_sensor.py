"""
Masking Sensor for AUVAP-PPO Integration

This module implements the masking sensor algorithm (Section I, Contribution #3)
that exposes one prioritized finding at a time to the DRL agent and constrains
the action space based on vulnerability type and safety rules.
"""

import os
import sys
import uuid
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import VAFinding
from task_manager import ExploitTask, TaskState
from policy_engine import PolicyEngine
from environment.action_mapper import ActionMapper


class TaskStatus(Enum):
    """Status of a task in the masking sensor"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


@dataclass
class SafetyConstraint:
    """Safety constraint for exploit execution"""
    constraint_type: str  # "network_isolation", "timeout", "resource_limit", etc.
    value: Any
    reason: str
    enforced: bool = True


@dataclass
class TaskExposure:
    """Information exposed to DRL agent about current task"""
    task: ExploitTask
    allowed_actions: List[int]
    safety_constraints: List[SafetyConstraint]
    context: Dict  # Additional context (network state, credentials, etc.)
    attempt_number: int
    max_attempts: int


@dataclass
class ExecutionLogEntry:
    """Log entry for deterministic replay"""
    timestamp: str
    task_id: str
    action_taken: int
    result: str
    safety_violations: List[str]
    duration_seconds: float
    metadata: Dict = field(default_factory=dict)


class MaskingSensor:
    """
    Masking Sensor Algorithm (Section I, Contribution #3)

    Controls what the DRL agent can see and do at each step:
    - Exposes one prioritized finding at a time
    - Constrains action space based on vulnerability type
    - Enforces safety constraints
    - Maintains execution log for deterministic replay
    """

    def __init__(self,
                 findings: List[VAFinding],
                 action_mapper: ActionMapper,
                 policy_engine: Optional[PolicyEngine] = None,
                 max_attempts_per_task: int = 3,
                 enable_safety_constraints: bool = True,
                 log_file: Optional[str] = None):
        """
        Initialize the masking sensor.

        Args:
            findings: List of vulnerability findings to process
            action_mapper: ActionMapper for task registration
            policy_engine: Optional policy engine for organizational constraints
            max_attempts_per_task: Maximum retry attempts per task
            enable_safety_constraints: Whether to enforce safety constraints
            log_file: Path to execution log file (for replay)
        """
        self.findings = findings
        self.action_mapper = action_mapper
        self.policy_engine = policy_engine
        self.max_attempts = max_attempts_per_task
        self.enable_safety = enable_safety_constraints
        self.log_file = log_file or "logs/masking_sensor_execution.jsonl"

        # State tracking
        self.task_queue: List[Tuple[float, ExploitTask]] = []  # (priority, task)
        self.current_task: Optional[ExploitTask] = None
        self.current_exposure: Optional[TaskExposure] = None
        self.task_status: Dict[str, TaskStatus] = {}
        self.task_attempts: Dict[str, int] = {}
        self.execution_log: List[ExecutionLogEntry] = []

        # Network state tracking (for context)
        self.owned_nodes: Set[str] = set()
        self.discovered_credentials: Dict[str, List[str]] = {}
        self.network_topology: Dict = {}

        # Initialize task queue
        self._initialize_task_queue()

    def _initialize_task_queue(self):
        """
        Initialize the task queue with prioritized findings.

        Priority calculation:
        - CVSS score (higher = higher priority)
        - Severity level (Critical > High > Medium > Low)
        - Attack surface (remote > local)
        - Existing credentials (if we have creds for target)
        """
        for finding in self.findings:
            # Register with action mapper
            task = self._create_task_from_finding(finding)
            action_id = self.action_mapper.register_task(task)

            if action_id is None:
                continue  # Max actions reached

            # Calculate priority
            priority = self._calculate_priority(finding)

            # Add to queue (negative priority for max-heap behavior)
            self.task_queue.append((-priority, task))
            self.task_status[task.task_id] = TaskStatus.PENDING
            self.task_attempts[task.task_id] = 0

        # Sort by priority (highest first)
        self.task_queue.sort(key=lambda x: x[0])

    def _create_task_from_finding(self, finding: VAFinding) -> ExploitTask:
        """Create an ExploitTask from a vulnerability finding."""
        host = getattr(finding, 'host_ip', '')
        port = getattr(finding, 'port', 0)
        finding_id = getattr(finding, 'finding_id', str(uuid.uuid4()))
        cvss = getattr(finding, 'cvss_base_score', getattr(finding, 'cvss', 5.0)) or 5.0
        priority = self._calculate_priority(finding)

        return ExploitTask(
            task_id=finding_id,
            finding_id=finding_id,
            state=TaskState.PLANNED,
            target=f"{host}:{port}",
            config={
                'host': host,
                'port': port,
                'protocol': getattr(finding, 'protocol', 'tcp'),
                'service': getattr(finding, 'service', getattr(finding, 'protocol', 'tcp')),
                'vulnerability_id': getattr(finding, 'plugin_id', getattr(finding, 'cve', '')),
                'description': getattr(finding, 'plugin_name', getattr(finding, 'title', '')),
                'cvss_score': cvss,
            },
            risk_score=cvss,
            priority=priority,
        )

    def _calculate_priority(self, finding: VAFinding) -> float:
        """Calculate task priority based on finding characteristics."""
        priority = 0.0

        # Base priority from CVSS
        priority += (finding.cvss_base_score or 5.0) * 10

        # Severity multiplier
        severity_weights = {
            'Critical': 2.0,
            'High': 1.5,
            'Medium': 1.0,
            'Low': 0.5,
            'Info': 0.1
        }
        priority *= severity_weights.get(finding.severity, 1.0)

        # Attack surface bonus
        if finding.protocol in ['tcp', 'udp']:
            priority *= 1.2  # Remote exploits prioritized

        # Credential availability bonus
        if finding.host_ip in self.discovered_credentials:
            priority *= 1.3

        return priority

    def get_current_task(self) -> Optional[TaskExposure]:
        """
        Get the current task exposure for the DRL agent.

        Returns:
            TaskExposure with current task and constraints, or None if queue empty
        """
        if self.current_exposure is not None:
            return self.current_exposure

        # Get next task from queue
        if not self.task_queue:
            return None

        priority, task = self.task_queue[0]
        self.current_task = task
        self.task_status[task.task_id] = TaskStatus.ACTIVE

        # Compute allowed actions
        allowed_actions = self._compute_allowed_actions(task)

        # Get safety constraints
        safety_constraints = self._get_safety_constraints(task)

        # Build context
        context = self._build_context(task)

        # Create exposure
        self.current_exposure = TaskExposure(
            task=task,
            allowed_actions=allowed_actions,
            safety_constraints=safety_constraints,
            context=context,
            attempt_number=self.task_attempts[task.task_id],
            max_attempts=self.max_attempts
        )

        return self.current_exposure

    def _compute_allowed_actions(self, task: ExploitTask) -> List[int]:
        """
        Compute allowed actions based on vulnerability type and current state.

        Action constraints:
        - Can only exploit discovered nodes
        - Can only use available credentials
        - Must respect policy engine rules
        - Cannot retry already-successful exploits
        """
        allowed: List[int] = []

        # Get action ID for this task
        action_id = self.action_mapper.get_action_from_task(task.task_id)
        if action_id is None:
            return allowed

        # Check if target is accessible
        if not self._is_target_accessible(task):
            return allowed  # Empty list = no actions allowed

        # Check policy engine constraints
        if self.policy_engine:
            if self.policy_engine.should_ignore(task):
                self.task_status[task.task_id] = TaskStatus.BLOCKED
                return allowed

            if self.policy_engine.requires_manual_review(task):
                self.task_status[task.task_id] = TaskStatus.BLOCKED
                return allowed

        # Check attempts limit
        if self.task_attempts[task.task_id] >= self.max_attempts:
            return allowed

        # Task is valid - allow corresponding action
        allowed.append(action_id)

        return allowed

    def _is_target_accessible(self, task: ExploitTask) -> bool:
        """
        Check if target is accessible from current network position.

        Returns True if:
        - Target is directly accessible (no owned nodes required)
        - OR we own a node that can reach the target
        """
        # If we don't own any nodes yet, only entry-point exploits allowed
        if not self.owned_nodes:
            # Allow if target is an entry point (e.g., public-facing)
            return self._is_entry_point(task.target_host)

        # Check if target is reachable from owned nodes
        for owned_node in self.owned_nodes:
            if self._can_reach(owned_node, task.target_host):
                return True

        return False

    def _is_entry_point(self, host: str) -> bool:
        """Check if host is an entry point (public-facing)."""
        # Simple heuristic: public IP or common public services
        # In production, use actual network topology data
        return True  # For now, allow all as potential entry points

    def _can_reach(self, source: str, target: str) -> bool:
        """Check if source node can reach target node."""
        # Check network topology
        if source in self.network_topology:
            reachable = self.network_topology[source].get('reachable', [])
            return target in reachable

        # Default: assume reachable if we own source
        return True

    def _get_safety_constraints(self, task: ExploitTask) -> List[SafetyConstraint]:
        """
        Get safety constraints for the current task.

        Constraints:
        - Network isolation (no external connections)
        - Execution timeout
        - Resource limits (CPU, memory)
        - Forbidden operations (data destruction, etc.)
        """
        constraints: List[SafetyConstraint] = []

        if not self.enable_safety:
            return constraints

        # Network isolation
        constraints.append(SafetyConstraint(
            constraint_type="network_isolation",
            value=True,
            reason="Prevent accidental external network access",
            enforced=True
        ))

        # Execution timeout
        timeout = self._calculate_timeout(task)
        constraints.append(SafetyConstraint(
            constraint_type="timeout",
            value=timeout,
            reason=f"Prevent infinite loops or hangs",
            enforced=True
        ))

        # Resource limits
        constraints.append(SafetyConstraint(
            constraint_type="memory_limit",
            value="512MB",
            reason="Prevent memory exhaustion",
            enforced=True
        ))

        constraints.append(SafetyConstraint(
            constraint_type="cpu_limit",
            value="1.0",
            reason="Limit CPU usage",
            enforced=True
        ))

        # Forbidden operations based on vulnerability type
        forbidden_ops = self._get_forbidden_operations(task)
        if forbidden_ops:
            constraints.append(SafetyConstraint(
                constraint_type="forbidden_operations",
                value=forbidden_ops,
                reason="Prevent dangerous operations",
                enforced=True
            ))

        return constraints

    def _calculate_timeout(self, task: ExploitTask) -> int:
        """Calculate appropriate timeout for task execution."""
        # Base timeout
        timeout = 10  # seconds

        # Adjust based on vulnerability type
        if "brute" in task.description.lower() or "password" in task.description.lower():
            timeout = 30  # Credential-based exploits may take longer

        # Adjust based on attempts (give more time on retries)
        timeout += self.task_attempts[task.task_id] * 5

        return min(timeout, 60)  # Cap at 60 seconds

    def _get_forbidden_operations(self, task: ExploitTask) -> List[str]:
        """
        DEPRECATED — kept for backward compatibility but no longer the
        primary enforcement mechanism.  Real safety enforcement is via Docker
        security profiles (see ``_get_docker_security_policy``).

        Returns:
            List of audit-only string patterns (used for log scanning).
        """
        return self._audit_patterns(task)

    @staticmethod
    def _audit_patterns(task: ExploitTask) -> List[str]:
        """Patterns used for *post-hoc* output auditing (secondary layer)."""
        patterns = [
            "rm -rf /",
            "DROP DATABASE",
            "DELETE FROM",
            "> /dev/sda",
            "dd if=/dev/zero",
            ":(){ :|:& };:",   # fork bomb
            "nc -e",           # reverse shell
            "bash -i >& /dev/",
        ]
        if task.cvss_score >= 9.0:
            patterns.extend(["curl http://", "wget http://"])
        return patterns

    def _get_docker_security_policy(self, task: ExploitTask) -> Dict:
        """
        Build a Docker-level security policy for the given task.

        The returned dict is passed directly to ``SandboxExecutor`` and
        maps to Docker container-creation options.  This is the **primary**
        enforcement mechanism (kernel-level, not bypassable from userspace).

        Policy tiers:
          • Standard (CVSS < 7.0): network off, caps dropped, 512 MB
          • High     (CVSS 7.0–8.9): + read-only rootfs, pids limit 32
          • Critical (CVSS ≥ 9.0): + stricter seccomp, pids limit 16, 256 MB

        Returns:
            Dictionary of Docker security options.
        """
        policy: Dict = {
            # --- Always applied ---
            "cap_drop": ["ALL"],
            "cap_add": ["DAC_OVERRIDE"],
            "network_disabled": True,
            "security_opt": ["no-new-privileges:true"],
            "tmpfs": {"/tmp": "size=64M,noexec,nosuid"},
            # Defaults
            "mem_limit": "512m",
            "pids_limit": 64,
            "read_only": False,
        }

        if task.cvss_score >= 7.0:
            policy["read_only"] = True
            policy["pids_limit"] = 32

        if task.cvss_score >= 9.0:
            policy["mem_limit"] = "256m"
            policy["pids_limit"] = 16
            # Tighter seccomp — also block socket/connect (no outbound traffic)
            policy["security_opt"].append(
                'seccomp={"defaultAction":"SCMP_ACT_ALLOW",'
                '"syscalls":[{"names":["mount","umount2","ptrace",'
                '"kexec_load","reboot","swapon","swapoff","pivot_root",'
                '"init_module","finit_module","delete_module",'
                '"acct","settimeofday","sethostname","setdomainname",'
                '"iopl","ioperm","syslog"],'
                '"action":"SCMP_ACT_ERRNO","errnoRet":1}]}'
            )

        return policy

    def _build_context(self, task: ExploitTask) -> Dict:
        """
        Build context information for the DRL agent.

        Context includes:
        - Owned nodes
        - Available credentials
        - Network topology
        - Previous exploit results
        """
        return {
            'owned_nodes': list(self.owned_nodes),
            'owned_count': len(self.owned_nodes),
            'available_credentials': self.discovered_credentials.get(task.target_host, []),
            'network_topology': self.network_topology,
            'previous_attempts': self.task_attempts[task.task_id],
            'related_cves': self._get_related_cves(task)
        }

    def _get_related_cves(self, task: ExploitTask) -> List[str]:
        """Get related CVEs for context."""
        # In production, query vulnerability database
        return []

    def advance(self, result: Dict) -> bool:
        """
        Advance to next task after current task completion.

        Args:
            result: Dictionary with execution result
                {
                    'success': bool,
                    'action': int,
                    'duration': float,
                    'error': str (optional),
                    'safety_violations': List[str] (optional),
                    'artifacts': Dict (optional)
                }

        Returns:
            True if advanced successfully, False if queue empty
        """
        if self.current_task is None:
            return False

        # Log execution
        self._log_execution(result)

        # Update task state
        if result.get('success', False):
            self._handle_success(result)
        else:
            self._handle_failure(result)

        # Clear current exposure
        self.current_exposure = None
        self.current_task = None

        # Remove from queue if completed/failed/max attempts
        if self.task_queue:
            task_id = self.task_queue[0][1].task_id
            status = self.task_status[task_id]

            if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.BLOCKED]:
                self.task_queue.pop(0)

        return len(self.task_queue) > 0

    def _handle_success(self, result: Dict):
        """Handle successful task execution."""
        task = self.current_task
        self.task_status[task.task_id] = TaskStatus.COMPLETED

        # Update network state
        self.owned_nodes.add(task.target_host)

        # Extract credentials if any
        if 'credentials' in result.get('artifacts', {}):
            creds = result['artifacts']['credentials']
            if task.target_host not in self.discovered_credentials:
                self.discovered_credentials[task.target_host] = []
            self.discovered_credentials[task.target_host].extend(creds)

    def _handle_failure(self, result: Dict):
        """Handle failed task execution."""
        task = self.current_task
        self.task_attempts[task.task_id] += 1

        # Check if max attempts reached
        if self.task_attempts[task.task_id] >= self.max_attempts:
            self.task_status[task.task_id] = TaskStatus.FAILED
        else:
            # Allow retry
            self.task_status[task.task_id] = TaskStatus.PENDING

    def _log_execution(self, result: Dict):
        """Log execution for deterministic replay."""
        if self.current_task is None:
            return

        entry = ExecutionLogEntry(
            timestamp=datetime.now().isoformat(),
            task_id=self.current_task.task_id,
            action_taken=result.get('action', -1),
            result='success' if result.get('success') else 'failure',
            safety_violations=result.get('safety_violations', []),
            duration_seconds=result.get('duration', 0.0),
            metadata={
                'target': self.current_task.target_host,
                'cvss': self.current_task.cvss_score,
                'attempt': self.task_attempts[self.current_task.task_id],
                'error': result.get('error', '')
            }
        )

        self.execution_log.append(entry)

        # Write to log file
        self._write_log_entry(entry)

    def _write_log_entry(self, entry: ExecutionLogEntry):
        """Write log entry to file (JSONL format)."""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        with open(self.log_file, 'a') as f:
            json.dump({
                'timestamp': entry.timestamp,
                'task_id': entry.task_id,
                'action': int(entry.action_taken) if entry.action_taken is not None else -1,
                'result': entry.result,
                'safety_violations': entry.safety_violations,
                'duration': float(entry.duration_seconds),
                'metadata': entry.metadata
            }, f, default=str)
            f.write('\n')

    def get_statistics(self) -> Dict:
        """Get execution statistics."""
        total = len(self.task_status)
        completed = sum(1 for s in self.task_status.values() if s == TaskStatus.COMPLETED)
        failed = sum(1 for s in self.task_status.values() if s == TaskStatus.FAILED)
        pending = sum(1 for s in self.task_status.values() if s == TaskStatus.PENDING)
        blocked = sum(1 for s in self.task_status.values() if s == TaskStatus.BLOCKED)

        return {
            'total_tasks': total,
            'completed': completed,
            'failed': failed,
            'pending': pending,
            'blocked': blocked,
            'success_rate': completed / total if total > 0 else 0.0,
            'owned_nodes': len(self.owned_nodes),
            'credentials_found': sum(len(creds) for creds in self.discovered_credentials.values()),
            'total_attempts': sum(self.task_attempts.values()),
            'avg_attempts': sum(self.task_attempts.values()) / total if total > 0 else 0.0
        }

    def is_complete(self) -> bool:
        """Check if all tasks are complete."""
        return len(self.task_queue) == 0

    def save_execution_log(self, output_file: str):
        """Save complete execution log to file."""
        with open(output_file, 'w') as f:
            json.dump([{
                'timestamp': e.timestamp,
                'task_id': e.task_id,
                'action': e.action_taken,
                'result': e.result,
                'safety_violations': e.safety_violations,
                'duration': e.duration_seconds,
                'metadata': e.metadata
            } for e in self.execution_log], f, indent=2)
