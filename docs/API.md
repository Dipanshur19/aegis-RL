# AUVAP-PPO API Documentation

**Version**: 1.0
**Last Updated**: 2025-11-10

---

## Table of Contents

1. [PPO Agent API](#ppo-agent-api)
2. [Environment API](#environment-api)
3. [Terrain Generator API](#terrain-generator-api)
4. [Sandbox Executor API](#sandbox-executor-api)
5. [LLM-DRL Bridge API](#llm-drl-bridge-api)
6. [Persistent Memory API](#persistent-memory-api)
7. [Task Manager API](#task-manager-api)
8. [Policy Engine API](#policy-engine-api)
9. [Parser API](#parser-api)

---

## PPO Agent API

### `PolicyNetwork`

Actor-Critic neural network for PPO.

```python
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        """
        Initialize PolicyNetwork.

        Args:
            obs_dim: Observation space dimension
            action_dim: Number of discrete actions
        """

    def forward(self, obs: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.

        Args:
            obs: Observations (batch_size, obs_dim)
            action_mask: Boolean mask for valid actions (batch_size, action_dim)

        Returns:
            logits: Action logits (batch_size, action_dim)
            value: State value estimates (batch_size, 1)
        """
```

**Example:**
```python
network = PolicyNetwork(obs_dim=128, action_dim=50)
obs = torch.randn(32, 128)
mask = torch.ones(32, 50, dtype=torch.bool)
mask[:, 40:] = False  # Disable last 10 actions

logits, value = network(obs, mask)
# logits[:, 40:] will be -inf
```

---

### `PPOAgent`

Complete PPO agent with training capabilities.

```python
class PPOAgent:
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5):
        """
        Initialize PPO agent.

        Args:
            obs_dim: Observation dimension
            action_dim: Action space size
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy regularization coefficient
            value_coef: Value loss coefficient
        """

    def select_action(self,
                     obs: torch.Tensor,
                     action_mask: Optional[torch.Tensor] = None
                    ) -> Tuple[int, float, float]:
        """
        Select action from policy.

        Args:
            obs: Single observation (obs_dim,)
            action_mask: Valid actions (action_dim,)

        Returns:
            action: Selected action index
            log_prob: Log probability of action
            value: Estimated state value
        """

    def compute_gae(self,
                   rewards: List[float],
                   values: List[float],
                   dones: List[bool]) -> List[float]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: Episode rewards
            values: Value estimates
            dones: Episode termination flags

        Returns:
            advantages: GAE advantages
        """

    def update(self,
              obs_batch: torch.Tensor,
              action_batch: torch.Tensor,
              log_prob_batch: torch.Tensor,
              advantage_batch: torch.Tensor,
              return_batch: torch.Tensor) -> Dict[str, float]:
        """
        Update policy using PPO.

        Args:
            obs_batch: Observations (batch_size, obs_dim)
            action_batch: Actions (batch_size,)
            log_prob_batch: Old log probs (batch_size,)
            advantage_batch: Advantages (batch_size,)
            return_batch: Returns (batch_size,)

        Returns:
            losses: Dict with policy_loss, value_loss, entropy, total_loss
        """

    def save(self, path: str):
        """Save model checkpoint."""

    def load(self, path: str):
        """Load model checkpoint."""
```

**Example:**
```python
agent = PPOAgent(obs_dim=128, action_dim=50)

# Training loop
for episode in range(1000):
    trajectory = collect_trajectory(env, agent)

    # Compute advantages
    advantages = agent.compute_gae(
        trajectory['rewards'],
        trajectory['values'],
        trajectory['dones']
    )

    # Update policy
    losses = agent.update(
        torch.stack(trajectory['observations']),
        torch.tensor(trajectory['actions']),
        torch.tensor(trajectory['log_probs']),
        torch.tensor(advantages),
        torch.tensor(returns)
    )

    if episode % 100 == 0:
        agent.save(f"checkpoint_{episode}.pt")
```

---

## Environment API

### `CyberBattleEnv`

Gymnasium wrapper for CyberBattleSim.

```python
class CyberBattleEnv(gym.Env):
    def __init__(self,
                 terrain_graph: nx.DiGraph,
                 max_steps: int = 200,
                 reward_scale: float = 1.0):
        """
        Initialize CyberBattle environment.

        Args:
            terrain_graph: Network topology from TerrainGenerator
            max_steps: Maximum episode length
            reward_scale: Reward scaling factor
        """

    def reset(self,
             seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment.

        Args:
            seed: Random seed for reproducibility

        Returns:
            observation: Initial state (obs_dim,)
            info: Episode info dict
        """

    def step(self,
            action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action in environment.

        Args:
            action: Action index [0, 49]

        Returns:
            observation: Next state
            reward: Step reward
            terminated: Episode end flag
            truncated: Timeout flag
            info: Step info
        """

    def get_action_mask(self) -> np.ndarray:
        """
        Get valid actions for current state.

        Returns:
            mask: Boolean array (action_dim,) where True = valid
        """
```

**Example:**
```python
from execution.terrain_generator import TerrainGenerator, TerrainParams
from execution.cyber_env import CyberBattleEnv

# Generate terrain
generator = TerrainGenerator()
params = TerrainParams(num_nodes=10)
graph, terrain_id = generator.generate_terrain(params, seed=42)

# Create environment
env = CyberBattleEnv(graph, max_steps=200)

# Episode loop
obs, info = env.reset(seed=42)
done = False

while not done:
    action_mask = env.get_action_mask()
    action = agent.select_action(obs, action_mask)[0]
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

---

## Terrain Generator API

### `TerrainGenerator`

Generates synthetic network environments.

```python
@dataclass
class TerrainParams:
    num_nodes: int = 10
    graph_type: str = "erdos_renyi"  # "barabasi_albert", "scale_free", "tree"
    edge_probability: float = 0.3
    attachment: int = 2
    os_distribution: Dict[str, float] = field(default_factory=lambda: {
        "linux": 0.5, "windows": 0.4, "macos": 0.1
    })
    role_distribution: Dict[str, float] = field(default_factory=lambda: {
        "workstation": 0.5, "server": 0.3, "router": 0.1, "firewall": 0.1
    })
    vuln_density: float = 0.3
    max_vulns_per_node: int = 5
    cred_density: float = 0.2
    max_creds_per_node: int = 3
    firewall_probability: float = 0.2
    ensure_connected: bool = True
    require_attack_path: bool = True
    entry_points: int = 2

class TerrainGenerator:
    def generate_terrain(self,
                        params: Optional[TerrainParams] = None,
                        seed: Optional[int] = None
                       ) -> Tuple[nx.DiGraph, str]:
        """
        Generate synthetic network terrain.

        Args:
            params: Terrain configuration
            seed: Random seed for reproducibility

        Returns:
            graph: NetworkX directed graph with node attributes
            terrain_id: Unique terrain identifier (SHA-256 hash)
        """
```

**Example:**
```python
from execution.terrain_generator import TerrainGenerator, TerrainParams

generator = TerrainGenerator()

# Generate small network
params = TerrainParams(
    num_nodes=5,
    graph_type="tree",
    vuln_density=0.5
)

graph, terrain_id = generator.generate_terrain(params, seed=42)

# Inspect nodes
for node in graph.nodes():
    attrs = graph.nodes[node]
    print(f"{node}: {attrs['os']}, {attrs['role']}")
    print(f"  Services: {attrs['services']}")
    print(f"  Vulnerabilities: {attrs['vulnerabilities']}")
```

---

## Sandbox Executor API

### `SandboxExecutor`

Docker-based exploit script executor.

```python
@dataclass
class ExecutionResult:
    status: str  # "success", "failure", "timeout", "error"
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    logs: List[str]
    artifacts: Dict = field(default_factory=dict)
    safety_violations: List[str] = field(default_factory=list)

class SandboxExecutor:
    def __init__(self,
                 docker_image: str = "python:3.10-slim",
                 default_timeout: int = 10,
                 memory_limit: str = "512m",
                 cpu_count: float = 1.0,
                 enable_network: bool = False,
                 work_dir: str = "/tmp/sandbox_workdir"):
        """
        Initialize sandbox executor.

        Args:
            docker_image: Base Docker image
            default_timeout: Execution timeout (seconds)
            memory_limit: Memory limit (e.g., "512m")
            cpu_count: CPU limit (cores)
            enable_network: Enable network access
            work_dir: Working directory for temporary files
        """

    def execute_task(self,
                    task_id: str,
                    script_path: str,
                    timeout: Optional[int] = None,
                    env_vars: Optional[Dict] = None) -> ExecutionResult:
        """
        Execute script in isolated container.

        Args:
            task_id: Unique task identifier
            script_path: Path to script file
            timeout: Override default timeout
            env_vars: Environment variables

        Returns:
            ExecutionResult with execution details
        """
```

**Example:**
```python
from execution.sandbox_executor import SandboxExecutor

executor = SandboxExecutor(
    memory_limit="256m",
    cpu_count=0.5,
    default_timeout=30,
    enable_network=False
)

# Execute exploit script
result = executor.execute_task(
    task_id="task_001",
    script_path="/path/to/exploit.py",
    timeout=60,
    env_vars={"TARGET_IP": "10.0.0.1", "TARGET_PORT": "22"}
)

if result.status == "success":
    print(f"Success! Output: {result.stdout}")
else:
    print(f"Failed: {result.stderr}")
```

---

## LLM-DRL Bridge API

### `LLMDRLBridge`

Orchestrates LLM↔DRL feedback loop.

```python
@dataclass
class ScriptGenerationRequest:
    finding: VAFinding
    task: ExploitTask
    similar_attempts: List = None
    refinement_iteration: int = 0
    previous_errors: List[str] = None
    execution_trace: str = ""

@dataclass
class ScriptGenerationResponse:
    script_content: str
    confidence: float
    reasoning: str
    metadata: Dict

class LLMDRLBridge:
    def __init__(self,
                 sandbox_executor: SandboxExecutor,
                 persistent_memory: PersistentMemory,
                 llm_provider: str = "openai",
                 max_refinement_iterations: int = 3,
                 use_memory_context: bool = True,
                 verbose: bool = True):
        """
        Initialize LLM-DRL bridge.

        Args:
            sandbox_executor: Sandbox for script execution
            persistent_memory: Memory for exploitation history
            llm_provider: LLM provider ("openai", "gemini", "local")
            max_refinement_iterations: Max refinement attempts
            use_memory_context: Use past attempts as context
            verbose: Print detailed progress
        """

    def plan_and_execute(self,
                        finding: VAFinding,
                        task: ExploitTask
                       ) -> Tuple[bool, ExecutionResult, str]:
        """
        Main LLM→DRL→LLM loop.

        Args:
            finding: Vulnerability finding
            task: Exploitation task

        Returns:
            success: Whether exploitation succeeded
            result: Final execution result
            script: Final exploit script
        """
```

**Example:**
```python
from execution.llm_drl_bridge import LLMDRLBridge
from execution.sandbox_executor import SandboxExecutor
from execution.persistent_memory import PersistentMemory

# Initialize components
sandbox = SandboxExecutor()
memory = PersistentMemory("exploits.db")

bridge = LLMDRLBridge(
    sandbox_executor=sandbox,
    persistent_memory=memory,
    llm_provider="openai",
    max_refinement_iterations=3
)

# Execute exploitation
success, result, script = bridge.plan_and_execute(finding, task)

if success:
    print(f"Exploitation successful!")
    print(f"Script:\n{script}")
else:
    print(f"Failed after {result.attempts} attempts")
```

---

## Persistent Memory API

### `PersistentMemory`

SQLite-based exploitation history storage.

```python
class PersistentMemory:
    def __init__(self, db_path: str = "exploitation_memory.db"):
        """
        Initialize persistent memory.

        Args:
            db_path: Path to SQLite database file
        """

    def store_attempt(self,
                     finding_id: str,
                     task_id: str,
                     script_content: str,
                     execution_result: Dict,
                     success: bool,
                     metadata: Optional[Dict] = None):
        """
        Store exploitation attempt.

        Args:
            finding_id: Vulnerability identifier
            task_id: Task identifier
            script_content: Exploit script code
            execution_result: Sandbox execution result
            success: Whether exploitation succeeded
            metadata: Additional metadata
        """

    def get_attempts_by_finding(self,
                               finding_id: str) -> List[Dict]:
        """
        Retrieve all attempts for a finding.

        Args:
            finding_id: Vulnerability identifier

        Returns:
            List of attempt records
        """

    def get_similar_attempts(self,
                            finding_id: str,
                            limit: int = 5) -> List[Dict]:
        """
        Find similar exploitation attempts.

        Args:
            finding_id: Reference finding
            limit: Max results

        Returns:
            List of similar attempts
        """

    def get_successful_attempts(self, limit: int = 100) -> List[Dict]:
        """Get all successful exploitations."""

    def clear_old_attempts(self, days: int = 30):
        """Delete attempts older than N days."""
```

**Example:**
```python
from execution.persistent_memory import PersistentMemory

memory = PersistentMemory("exploits.db")

# Store successful attempt
memory.store_attempt(
    finding_id="find_123",
    task_id="task_456",
    script_content=exploit_script,
    execution_result={"status": "success", "duration": 2.5},
    success=True,
    metadata={"cve": "CVE-2023-1234", "service": "ssh"}
)

# Retrieve similar attempts
similar = memory.get_similar_attempts("find_789", limit=3)
for attempt in similar:
    print(f"Found similar: {attempt['task_id']}")
    print(f"  Script: {attempt['script_content'][:100]}...")
```

---

## Task Manager API

### `ExploitTask`

Task tracking dataclass.

```python
@dataclass
class ExploitTask:
    task_id: str
    finding_id: str
    state: TaskState  # PLANNED/EXECUTING/SUCCEEDED/FAILED/ABORTED
    attempts: int = 0
    script_path: Optional[str] = None
    target: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    priority: float = 0.0
    max_attempts: int = 3
    timestamps: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None

    def update_state(self, new_state: TaskState, error: Optional[str] = None):
        """Update task state and record timestamp."""

    def increment_attempts(self):
        """Increment attempt counter."""
```

### Task Manager Functions

```python
def compute_risk_score(finding: Dict[str, Any]) -> float:
    """
    Calculate risk score: r(f) = cvss × w_surface × w_auto

    Args:
        finding: Vulnerability with cvss, attack_vector, automation_candidate

    Returns:
        Risk score [0.0, 10.0]
    """

def initialize_tasks(feasible_findings: List[Dict[str, Any]]) -> List[ExploitTask]:
    """
    Create tasks from findings and sort by priority.

    Args:
        feasible_findings: Vulnerabilities deemed feasible for automation

    Returns:
        Sorted list of ExploitTask objects (highest priority first)
    """

def get_next_task(tasks: List[ExploitTask]) -> Optional[ExploitTask]:
    """
    Get next task to execute (prioritizes PLANNED, then retryable FAILED).

    Args:
        tasks: List of tasks

    Returns:
        Highest priority ready task or None
    """

def should_retry_task(task: ExploitTask) -> bool:
    """Check if task should be retried (under attempt limit)."""
```

**Example:**
```python
from task_manager import initialize_tasks, get_next_task

# Initialize tasks from findings
tasks = initialize_tasks(feasible_findings)

# Execution loop
while True:
    task = get_next_task(tasks)
    if task is None:
        break

    task.update_state(TaskState.EXECUTING)
    task.increment_attempts()

    # Execute...
    success = execute_exploit(task)

    if success:
        task.update_state(TaskState.SUCCEEDED)
    else:
        task.update_state(TaskState.FAILED, error="Connection timeout")
```

---

## Policy Engine API

### `PolicyEngine`

Rule-based policy filtering.

```python
@dataclass
class PolicyRule:
    rule_id: str
    type: str  # "ignore", "force_manual", "prioritize"
    predicate: Callable[[dict], bool]
    reason: str
    precedence: int  # 0=user, 1=org, 2=baseline

class PolicyEngine:
    def add_rule(self, rule: PolicyRule):
        """Add policy rule (auto-sorts by precedence)."""

    def evaluate(self, finding: dict) -> PolicyAction:
        """
        Evaluate finding against policy rules.

        Args:
            finding: Vulnerability dict

        Returns:
            PolicyAction with rule_id, action, reason
        """

    def detect_conflicts(self,
                        test_samples: Optional[List[dict]] = None) -> Dict:
        """Detect rule conflicts (shadowing, unreachable, etc.)."""
```

### Policy Loader

```python
def load_policies_from_yaml(filepath: str) -> List[PolicyRule]:
    """
    Load policy rules from YAML configuration.

    Args:
        filepath: Path to YAML file

    Returns:
        List of PolicyRule objects
    """
```

**Example:**
```python
from policy_engine import PolicyEngine, create_default_policy_rules
from policy_loader import load_policies_from_yaml

engine = PolicyEngine()

# Add default rules
engine.add_rules(create_default_policy_rules())

# Load custom rules from YAML
custom_rules = load_policies_from_yaml("policy_config.yaml")
engine.add_rules(custom_rules)

# Evaluate finding
finding = {"cvss": 9.8, "severity_bucket": "Critical", ...}
action = engine.evaluate(finding)

if action.rule.type == "ignore":
    print(f"Ignored: {action.reason}")
elif action.rule.type == "force_manual":
    print(f"Manual review required: {action.reason}")
else:
    print(f"Prioritized: {action.reason}")
```

---

## Parser API

### `VAFinding`

Vulnerability finding dataclass.

```python
@dataclass
class VAFinding:
    finding_id: str  # Auto-computed SHA-1 hash
    host_ip: str
    port: Optional[int]
    protocol: str
    service: str
    cvss: float
    severity_text: str
    title: str
    description: str
    cve: Optional[str]
    # ... additional fields
```

### Parser Functions

```python
def parse_report(file_path: str,
                format: str = "auto",
                validate: bool = False) -> List[VAFinding]:
    """
    Parse vulnerability scan report.

    Args:
        file_path: Path to report file
        format: "auto", "xml", or "csv"
        validate: Enable schema validation

    Returns:
        List of VAFinding objects
    """

def validate_finding(finding: VAFinding) -> Tuple[bool, List[str]]:
    """
    Validate finding against schema.

    Args:
        finding: VAFinding to validate

    Returns:
        (is_valid, error_list)
    """
```

**Example:**
```python
from parser import parse_report, to_dict_list

# Parse report
findings = parse_report("nessus_scan.xml", format="auto", validate=True)

print(f"Parsed {len(findings)} findings")

# Convert to dicts for processing
findings_dicts = to_dict_list(findings)
```

---

## Error Handling

### Common Exceptions

```python
# Parser errors
FileNotFoundError: Report file not found
xml.etree.ElementTree.ParseError: Invalid XML
ValueError: Invalid format or schema

# Classifier errors
RuntimeError: LLM API failure (with retry exhausted)
KeyError: Missing required field in response

# Sandbox errors
docker.errors.ContainerError: Container execution failed
docker.errors.ImageNotFound: Docker image not found
TimeoutError: Execution timeout

# Memory errors
sqlite3.OperationalError: Database operation failed
```

### Error Handling Example

```python
from parser import parse_report

try:
    findings = parse_report("scan.xml", validate=True)
except FileNotFoundError:
    print("Report file not found")
except ValueError as e:
    print(f"Validation error: {e}")
```

---

**Document Version**: 1.0
**Maintained By**: AUVAP-PPO Development Team
