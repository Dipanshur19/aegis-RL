#!/usr/bin/env python3
"""
task_manager.py - Phase 4 Task Management

Implements:
1. Risk scoring for vulnerabilities
2. ExploitTask dataclass for task tracking with priority and attempt limits
3. Task initialization and state management
4. Priority-based task queue management
5. Attempt limit enforcement with retry logic
6. Asset grouping by host/service
7. Task manifest generation

Integrates with feasibility_filter.py and experiment.py for Phase 4.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class TaskState(Enum):
    """Exploit task execution states."""
    PLANNED = "PLANNED"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


@dataclass
class ExploitTask:
    """
    Represents a single exploit task for automated testing.

    Fields:
        task_id: Unique UUID for task tracking
        finding_id: SHA-1 hash from parser.py (links to original finding)
        state: Current execution state (PLANNED/EXECUTING/SUCCEEDED/FAILED/ABORTED)
        attempts: Number of execution attempts
        script_path: Path to exploit script (if generated)
        target: Target specification (host:port)
        config: Configuration dict for exploit parameters
        risk_score: Calculated risk score (CVSS × surface_weight × automation_weight)
        priority: Task priority (higher = more urgent, defaults to risk_score)
        max_attempts: Maximum allowed attempts (default: 3)
        timestamps: Dict of state transition timestamps
        error_message: Last error message (if FAILED)
    """
    task_id: str
    finding_id: str
    state: TaskState
    attempts: int = 0
    script_path: Optional[str] = None
    target: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    priority: float = 0.0  # Higher = more urgent (defaults to risk_score)
    max_attempts: int = 3  # Configurable attempt limit
    timestamps: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Initialize timestamps and priority on creation."""
        if not self.timestamps:
            self.timestamps = {"created": datetime.now().isoformat()}
        # Default priority to risk_score if not explicitly set
        if self.priority == 0.0 and self.risk_score > 0.0:
            self.priority = self.risk_score
    
    def update_state(self, new_state: TaskState, error: Optional[str] = None) -> None:
        """
        Update task state and record timestamp.
        
        Args:
            new_state: New TaskState to transition to
            error: Error message (if transitioning to FAILED)
        """
        self.state = new_state
        self.timestamps[new_state.value.lower()] = datetime.now().isoformat()
        
        if new_state == TaskState.FAILED and error:
            self.error_message = error
    
    def increment_attempts(self) -> None:
        """Increment attempt counter."""
        self.attempts += 1

    @property
    def target_host(self) -> str:
        """Extract target host."""
        if self.config and 'host' in self.config:
            return str(self.config['host'])
        if ':' in self.target:
            return self.target.split(':', 1)[0]
        return self.target

    @property
    def target_port(self) -> int:
        """Extract target port."""
        if self.config and 'port' in self.config:
            try:
                return int(self.config['port'])
            except (ValueError, TypeError):
                return 0
        if ':' in self.target:
            try:
                return int(self.target.split(':', 1)[1])
            except (ValueError, TypeError):
                return 0
        return 0

    @property
    def vulnerability_id(self) -> str:
        """Extract vulnerability ID (CVE or plugin ID)."""
        if self.config and 'vulnerability_id' in self.config:
            return str(self.config['vulnerability_id'])
        if self.config and 'cve' in self.config:
            return str(self.config['cve'])
        return self.finding_id

    @property
    def cvss_score(self) -> float:
        """Extract CVSS score."""
        if self.config and 'cvss_score' in self.config:
            return float(self.config['cvss_score'])
        if self.config and 'cvss' in self.config:
            return float(self.config['cvss'])
        return float(self.risk_score)

    @property
    def port(self) -> int:
        """Alias for target_port."""
        return self.target_port

    @property
    def host(self) -> str:
        """Alias for target_host."""
        return self.target_host

    @property
    def protocol(self) -> str:
        """Extract protocol."""
        if self.config and 'protocol' in self.config:
            return str(self.config['protocol'])
        if self.config and 'service' in self.config:
            return str(self.config['service'])
        return 'tcp'

    @property
    def description(self) -> str:
        """Extract task description."""
        if self.config and 'description' in self.config:
            return str(self.config['description'])
        return self.finding_id
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary with all task fields (state converted to string)
        """
        data = asdict(self)
        data['state'] = self.state.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExploitTask':
        """
        Create ExploitTask from dictionary.
        
        Args:
            data: Dictionary with task fields
            
        Returns:
            ExploitTask instance
        """
        data_copy = data.copy()
        if 'state' in data_copy:
            data_copy['state'] = TaskState(data_copy['state'])
        return cls(**data_copy)


def compute_risk_score(finding: Dict[str, Any]) -> float:
    """
    Calculate risk score for a vulnerability finding.
    
    Formula: r(f) = cvss × w_surface × w_auto
    
    Where:
    - cvss: CVSS base score (0.0-10.0)
    - w_surface: Attack surface weight based on attack_vector
        * Network: 1.0 (remotely accessible)
        * Adjacent: 0.7 (requires adjacent network)
        * Local: 0.4 (requires local access)
        * Physical: 0.2 (requires physical access)
    - w_auto: Automation feasibility weight
        * Automatable (automation_candidate=True): 1.0
        * Manual (automation_candidate=False): 0.3
    
    Args:
        finding: Dictionary with cvss, attack_vector, automation_candidate
        
    Returns:
        Risk score (0.0-10.0)
    """
    cvss = finding.get('cvss') or 0.0
    attack_vector = finding.get('attack_vector', 'Local')
    automation_candidate = finding.get('automation_candidate', False)
    
    # Attack surface weights
    surface_weights = {
        'Network': 1.0,
        'Adjacent': 0.7,
        'Local': 0.4,
        'Physical': 0.2
    }
    w_surface = surface_weights.get(attack_vector, 0.4)
    
    # Automation weights
    w_auto = 1.0 if automation_candidate else 0.3
    
    # Calculate risk score
    risk_score = cvss * w_surface * w_auto
    
    return round(risk_score, 2)


def create_exploit_task(finding: Dict[str, Any], max_attempts: int = 3,
                       priority: Optional[float] = None) -> ExploitTask:
    """
    Create an ExploitTask from a feasible vulnerability finding.

    Args:
        finding: Dictionary with enriched vulnerability data
        max_attempts: Maximum retry attempts (default: 3)
        priority: Task priority (defaults to risk_score if not provided)

    Returns:
        ExploitTask initialized with PLANNED state
    """
    task_id = str(uuid.uuid4())
    finding_id = finding.get('finding_id', 'unknown')
    
    # Build target string
    host_ip = finding.get('host_ip', 'unknown')
    port = finding.get('port', 0)
    target = f"{host_ip}:{port}"
    
    # Calculate risk score
    risk_score = compute_risk_score(finding)
    
    # Build config dict with exploit parameters
    config = {
        'host_ip': host_ip,
        'port': port,
        'service': finding.get('service', 'unknown'),
        'cvss': finding.get('cvss'),
        'cve': finding.get('cve'),
        'severity_bucket': finding.get('severity_bucket'),
        'attack_vector': finding.get('attack_vector'),
        'vuln_component': finding.get('vuln_component'),
        'title': finding.get('title')
    }
    
    return ExploitTask(
        task_id=task_id,
        finding_id=finding_id,
        state=TaskState.PLANNED,
        target=target,
        config=config,
        risk_score=risk_score,
        priority=priority if priority is not None else risk_score,
        max_attempts=max_attempts
    )


def initialize_tasks(feasible_findings: List[Dict[str, Any]]) -> List[ExploitTask]:
    """
    Convert feasible findings to ExploitTasks.

    Args:
        feasible_findings: List of vulnerability findings deemed feasible

    Returns:
        List of ExploitTask objects sorted by priority (descending)
    """
    tasks = [create_exploit_task(finding) for finding in feasible_findings]

    # Sort by priority (highest first)
    tasks.sort(key=lambda t: t.priority, reverse=True)

    return tasks


def sort_tasks_by_priority(tasks: List[ExploitTask]) -> List[ExploitTask]:
    """
    Sort tasks by priority (highest first).

    This is the recommended way to order tasks for execution.
    Priority defaults to risk_score but can be customized.

    Args:
        tasks: List of ExploitTask objects

    Returns:
        Sorted list (highest priority first)
    """
    return sorted(tasks, key=lambda t: t.priority, reverse=True)


def should_retry_task(task: ExploitTask) -> bool:
    """
    Check if a task should be retried based on attempt limit.

    A task should be retried if:
    1. It's in FAILED state
    2. Number of attempts is less than max_attempts

    Args:
        task: ExploitTask to check

    Returns:
        True if task should be retried, False otherwise
    """
    return task.state == TaskState.FAILED and task.attempts < task.max_attempts


def get_retryable_tasks(tasks: List[ExploitTask]) -> List[ExploitTask]:
    """
    Filter tasks that can be retried and sort by priority.

    Args:
        tasks: List of ExploitTask objects

    Returns:
        List of retryable tasks sorted by priority (highest first)
    """
    retryable = [task for task in tasks if should_retry_task(task)]
    return sorted(retryable, key=lambda t: t.priority, reverse=True)


def get_planned_tasks(tasks: List[ExploitTask]) -> List[ExploitTask]:
    """
    Filter tasks in PLANNED state and sort by priority.

    Args:
        tasks: List of ExploitTask objects

    Returns:
        List of planned tasks sorted by priority (highest first)
    """
    planned = [task for task in tasks if task.state == TaskState.PLANNED]
    return sorted(planned, key=lambda t: t.priority, reverse=True)


def get_next_task(tasks: List[ExploitTask]) -> Optional[ExploitTask]:
    """
    Get the next task to execute based on priority.

    Prioritizes PLANNED tasks first, then retryable FAILED tasks.

    Args:
        tasks: List of ExploitTask objects

    Returns:
        Highest priority task ready for execution, or None if no tasks available
    """
    # First, try to get a PLANNED task
    planned = get_planned_tasks(tasks)
    if planned:
        return planned[0]  # Return highest priority planned task

    # Otherwise, try to get a retryable FAILED task
    retryable = get_retryable_tasks(tasks)
    if retryable:
        return retryable[0]  # Return highest priority retryable task

    # No tasks available
    return None


def group_tasks_by_host(tasks: List[ExploitTask]) -> Dict[str, List[ExploitTask]]:
    """
    Group exploit tasks by host IP.

    Args:
        tasks: List of ExploitTask objects

    Returns:
        Dictionary keyed by host_ip with list of tasks per host
        Each host's tasks are sorted by priority (descending)
    """
    groups: Dict[str, List[ExploitTask]] = {}

    for task in tasks:
        host_ip = task.config.get('host_ip', 'unknown')

        if host_ip not in groups:
            groups[host_ip] = []

        groups[host_ip].append(task)

    # Sort within each group by priority
    for host_ip in groups:
        groups[host_ip].sort(key=lambda t: t.priority, reverse=True)

    return groups


def group_tasks_by_service(tasks: List[ExploitTask]) -> Dict[Tuple[str, str], List[ExploitTask]]:
    """
    Group exploit tasks by (host_ip, service) tuple.

    Useful for targeting same services across multiple hosts or
    multiple vulnerabilities in the same service on one host.

    Args:
        tasks: List of ExploitTask objects

    Returns:
        Dictionary keyed by (host_ip, service) tuple with list of tasks
        Each group's tasks are sorted by priority (descending)
    """
    groups: Dict[Tuple[str, str], List[ExploitTask]] = {}

    for task in tasks:
        host_ip = task.config.get('host_ip', 'unknown')
        service = task.config.get('service', 'unknown')
        key = (host_ip, service)

        if key not in groups:
            groups[key] = []

        groups[key].append(task)

    # Sort within each group by priority
    for key in groups:
        groups[key].sort(key=lambda t: t.priority, reverse=True)

    return groups


def generate_task_manifest(tasks: List[ExploitTask], 
                          output_path: Path = Path("tasks_manifest.json")) -> None:
    """
    Generate JSON manifest of all exploit tasks.
    
    Manifest includes:
    - Summary statistics
    - Grouped tasks by host
    - Full task details
    
    Args:
        tasks: List of ExploitTask objects
        output_path: Path to write manifest JSON
    """
    # Group by host
    host_groups = group_tasks_by_host(tasks)
    
    # Calculate statistics
    total_tasks = len(tasks)
    total_hosts = len(host_groups)
    
    risk_scores = [t.risk_score for t in tasks]
    avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
    max_risk = max(risk_scores) if risk_scores else 0.0
    
    state_counts: Dict[str, int] = {}
    for task in tasks:
        state_counts[task.state.value] = state_counts.get(task.state.value, 0) + 1
    
    # Build manifest structure
    manifest: Dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_tasks": total_tasks,
            "total_hosts": total_hosts,
            "avg_risk_score": round(avg_risk, 2),
            "max_risk_score": round(max_risk, 2),
            "state_distribution": state_counts
        },
        "hosts": [],
        "tasks": [task.to_dict() for task in tasks]
    }
    
    # Add per-host summaries
    for host_ip, host_tasks in sorted(host_groups.items(), 
                                     key=lambda x: sum(t.risk_score for t in x[1]),
                                     reverse=True):
        host_risk_scores = [t.risk_score for t in host_tasks]
        manifest["hosts"].append({
            "host_ip": host_ip,
            "task_count": len(host_tasks),
            "total_risk_score": round(sum(host_risk_scores), 2),
            "max_risk_score": round(max(host_risk_scores), 2),
            "task_ids": [t.task_id for t in host_tasks]
        })
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"[*] Task manifest written to: {output_path}")


def load_task_manifest(manifest_path: Path) -> List[ExploitTask]:
    """
    Load ExploitTask objects from manifest JSON.
    
    Args:
        manifest_path: Path to manifest file
        
    Returns:
        List of ExploitTask objects
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    tasks = [ExploitTask.from_dict(task_dict) for task_dict in manifest['tasks']]
    return tasks


def print_task_summary(tasks: List[ExploitTask]) -> None:
    """
    Print human-readable summary of tasks.
    
    Args:
        tasks: List of ExploitTask objects
    """
    print("\n" + "=" * 70)
    print("EXPLOIT TASK SUMMARY")
    print("=" * 70)
    print(f"Total Tasks:     {len(tasks)}")
    
    if not tasks:
        print("=" * 70 + "\n")
        return
    
    # State distribution
    state_counts: Dict[str, int] = {}
    for task in tasks:
        state_counts[task.state.value] = state_counts.get(task.state.value, 0) + 1
    
    print("\nState Distribution:")
    for state, count in sorted(state_counts.items()):
        print(f"  {state:12s}: {count}")
    
    # Risk statistics
    risk_scores = [t.risk_score for t in tasks]
    print(f"\nRisk Scores:")
    print(f"  Average:  {sum(risk_scores) / len(risk_scores):.2f}")
    print(f"  Maximum:  {max(risk_scores):.2f}")
    print(f"  Minimum:  {min(risk_scores):.2f}")
    
    # Host distribution
    host_groups = group_tasks_by_host(tasks)
    print(f"\nHost Distribution:")
    print(f"  Total Hosts: {len(host_groups)}")
    print(f"  Avg Tasks/Host: {len(tasks) / len(host_groups):.1f}")
    
    # Top 5 highest risk tasks
    top_tasks = sorted(tasks, key=lambda t: t.risk_score, reverse=True)[:5]
    print(f"\nTop 5 Highest Risk Tasks:")
    for i, task in enumerate(top_tasks, 1):
        title = task.config.get('title', 'Unknown')[:50]
        print(f"  {i}. [{task.risk_score:5.2f}] {task.target:20s} - {title}")
    
    print("=" * 70 + "\n")


# ============================================================================
# Integration Test / Demo
# ============================================================================

def test_risk_scoring():
    """Test risk score calculation."""
    print("[*] Testing risk score calculation\n")
    
    test_findings = [
        {
            'cvss': 9.8,
            'attack_vector': 'Network',
            'automation_candidate': True,
            'title': 'Remote Code Execution'
        },
        {
            'cvss': 7.5,
            'attack_vector': 'Adjacent',
            'automation_candidate': True,
            'title': 'Adjacent Network Vuln'
        },
        {
            'cvss': 8.4,
            'attack_vector': 'Local',
            'automation_candidate': False,
            'title': 'Local Privilege Escalation'
        }
    ]
    
    for finding in test_findings:
        score = compute_risk_score(finding)
        print(f"Title: {finding['title']}")
        print(f"  CVSS: {finding['cvss']}, Vector: {finding['attack_vector']}, "
              f"Auto: {finding['automation_candidate']}")
        print(f"  Risk Score: {score}\n")


def test_task_creation():
    """Test task creation and grouping."""
    print("[*] Testing task creation and management\n")
    
    # Sample findings
    sample_findings = [
        {
            'finding_id': 'abc123',
            'host_ip': '192.168.1.100',
            'port': 80,
            'service': 'http',
            'cvss': 9.8,
            'cve': 'CVE-2021-44228',
            'severity_bucket': 'Critical',
            'attack_vector': 'Network',
            'vuln_component': 'Log4j 2.14.1',
            'automation_candidate': True,
            'title': 'Log4Shell RCE'
        },
        {
            'finding_id': 'def456',
            'host_ip': '192.168.1.100',
            'port': 445,
            'service': 'microsoft-ds',
            'cvss': 9.8,
            'cve': 'MS17-010',
            'severity_bucket': 'Critical',
            'attack_vector': 'Network',
            'vuln_component': 'SMBv1',
            'automation_candidate': True,
            'title': 'EternalBlue'
        },
        {
            'finding_id': 'ghi789',
            'host_ip': '192.168.1.101',
            'port': 22,
            'service': 'ssh',
            'cvss': 5.3,
            'severity_bucket': 'Medium',
            'attack_vector': 'Network',
            'vuln_component': 'OpenSSH 7.4',
            'automation_candidate': True,
            'title': 'SSH User Enumeration'
        }
    ]
    
    # Create tasks
    tasks = initialize_tasks(sample_findings)
    
    print(f"Created {len(tasks)} tasks")
    print_task_summary(tasks)
    
    # Test grouping
    host_groups = group_tasks_by_host(tasks)
    print(f"Grouped into {len(host_groups)} hosts:")
    for host_ip, host_tasks in host_groups.items():
        print(f"  {host_ip}: {len(host_tasks)} tasks")
    print()
    
    # Generate manifest
    manifest_path = Path("test_tasks_manifest.json")
    generate_task_manifest(tasks, manifest_path)
    print(f"Manifest written to: {manifest_path}\n")
    
    # Test state updates
    print("[*] Testing state transitions")
    task = tasks[0]
    print(f"Initial state: {task.state.value}")
    
    task.update_state(TaskState.EXECUTING)
    print(f"After EXECUTING: {task.state.value}")
    
    task.update_state(TaskState.SUCCEEDED)
    print(f"After SUCCEEDED: {task.state.value}")
    
    print(f"Timestamps: {task.timestamps}")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 4 TASK MANAGEMENT TEST SUITE")
    print("=" * 70)
    print()
    
    try:
        test_risk_scoring()
        test_task_creation()
        print("[+] All tests completed successfully!")
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
