# Masking Sensor Integration Guide

## Overview

The **Masking Sensor Algorithm** is a critical component that controls what the DRL agent can see and do during training and execution. It implements the masking sensor described in Section I, Contribution #3 of the research paper.

## What is the Masking Sensor?

The masking sensor acts as an intelligent "task controller" between your vulnerability data and the RL agent:

```
Vulnerability Data → Masking Sensor → DRL Agent
                   (controls exposure)
```

### Key Responsibilities

1. **Task Exposure Control** - Exposes ONE prioritized task at a time
2. **Action Space Masking** - Constrains valid actions based on network state
3. **Safety Enforcement** - Applies safety constraints (timeouts, isolation, etc.)
4. **State Tracking** - Maintains network state (owned nodes, credentials)
5. **Execution Logging** - Records all actions for deterministic replay

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Masking Sensor                              │
│                                                                   │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │Priority Queue│   │Action Masker │   │Safety Engine │        │
│  │(CVSS-based)  │   │(Valid actions│   │(Constraints) │        │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘        │
│         │                   │                   │                │
│         └───────────────────┴───────────────────┘                │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │  Task Exposure  │                          │
│                    │  - Task info    │                          │
│                    │  - Allowed acts │                          │
│                    │  - Constraints  │                          │
│                    │  - Context      │                          │
│                    └────────┬────────┘                          │
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              ▼
                        ┌─────────────┐
                        │  DRL Agent  │
                        └─────────────┘
```

## Core Components

### 1. MaskingSensor Class

**Location:** `environment/masking_sensor.py`

```python
from environment.masking_sensor import MaskingSensor
from environment.action_mapper import ActionMapper

# Initialize
sensor = MaskingSensor(
    findings=vulnerability_findings,
    action_mapper=action_mapper,
    max_attempts_per_task=3,
    enable_safety_constraints=True,
    log_file="logs/execution.jsonl"
)

# Get current task exposure
exposure = sensor.get_current_task()

# After agent execution
result = {
    'success': True,
    'action': 42,
    'duration': 3.5,
    'artifacts': {'credentials': ['user:pass']}
}
sensor.advance(result)
```

### 2. MaskedCyberBattleEnv

**Location:** `environment/masked_cyberbattle_env.py`

A Gym environment that integrates the masking sensor:

```python
from environment.masked_cyberbattle_env import MaskedCyberBattleEnv

env = MaskedCyberBattleEnv(
    vulnerability_findings=findings,
    config=config,
    max_attempts_per_task=3
)

# Standard Gym interface
obs, info = env.reset()
action = agent.predict(obs)
obs, reward, done, truncated, info = env.step(action)
```

### 3. PPO Training Integration

**Location:** `training/train_ppo_masked.py`

Complete training script with masking sensor:

```bash
python3 training/train_ppo_masked.py \
    --nessus-file auvap_nessus_100_findings.xml \
    --timesteps 100000 \
    --lr 0.0003
```

## How It Works

### Task Exposure Flow

```
1. INITIALIZATION
   ├─ Load vulnerability findings
   ├─ Calculate priorities (CVSS × severity × attack surface)
   ├─ Build priority queue
   └─ Initialize network state (empty)

2. GET CURRENT TASK
   ├─ Pop highest-priority task
   ├─ Check if target accessible
   ├─ Compute allowed actions
   │  ├─ Check policy rules
   │  ├─ Check network reachability
   │  └─ Check retry limits
   ├─ Generate safety constraints
   └─ Build context

3. AGENT EXECUTES
   ├─ Receives TaskExposure
   ├─ Selects action from allowed_actions
   └─ Attempts exploitation

4. ADVANCE TO NEXT
   ├─ Log execution
   ├─ Update network state
   │  ├─ Add owned nodes
   │  └─ Store credentials
   ├─ Mark task complete/failed
   └─ Move to next task
```

### Priority Calculation

Tasks are automatically prioritized:

```python
priority = cvss_score * 10 * severity_weight * surface_multiplier

# Example priorities:
# Critical remote (CVSS 9.8): 235.2
# High local (CVSS 7.0):      105.0
# Medium (CVSS 5.0):          50.0
```

### Action Masking

Actions are blocked if:

- ❌ Target not discovered
- ❌ Target not accessible from owned nodes
- ❌ Policy engine forbids it
- ❌ Max retry attempts exceeded
- ❌ Already successfully exploited

### Safety Constraints

Every task has enforced constraints:

```python
SafetyConstraint(
    constraint_type="network_isolation",
    value=True,
    reason="Prevent external connections"
)

SafetyConstraint(
    constraint_type="timeout",
    value=10,  # seconds
    reason="Prevent infinite loops"
)

SafetyConstraint(
    constraint_type="memory_limit",
    value="512MB",
    reason="Prevent memory exhaustion"
)

SafetyConstraint(
    constraint_type="forbidden_operations",
    value=["rm -rf /", "DROP DATABASE", ...],
    reason="Prevent data destruction"
)
```

## Usage Examples

### Example 1: Basic Usage

```python
from parser import parse_nessus_xml
from environment.action_mapper import ActionMapper
from environment.masking_sensor import MaskingSensor

# Load data
findings = parse_nessus_xml("scan.xml")

# Create components
mapper = ActionMapper(max_actions=100)
sensor = MaskingSensor(findings=findings, action_mapper=mapper)

# Training loop
while not sensor.is_complete():
    # Get current task
    exposure = sensor.get_current_task()
    if not exposure or not exposure.allowed_actions:
        sensor.advance({'success': False, 'action': -1, 'duration': 0})
        continue

    # Agent acts
    action = agent.select_action(exposure)
    result = execute_action(action)

    # Advance
    sensor.advance(result)

# Get statistics
stats = sensor.get_statistics()
print(f"Success rate: {stats['success_rate']*100:.1f}%")
```

### Example 2: With PPO Training

```python
from environment.masked_cyberbattle_env import create_masked_env_from_nessus
from stable_baselines3 import PPO

# Create environment
env = create_masked_env_from_nessus("scan.xml")

# Create and train PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save results
model.save("trained_model")
env.save_execution_log("execution_log.json")
```

### Example 3: With Safety Constraints

```python
from environment.masking_sensor import MaskingSensor

sensor = MaskingSensor(
    findings=findings,
    action_mapper=mapper,
    enable_safety_constraints=True,  # Enable safety
    max_attempts_per_task=3
)

exposure = sensor.get_current_task()

# Check constraints before execution
for constraint in exposure.safety_constraints:
    print(f"{constraint.constraint_type}: {constraint.value}")
    print(f"Reason: {constraint.reason}")
```

## Configuration

### Environment Config

```python
from environment.cyberbattle_wrapper import AUVAPConfig

config = AUVAPConfig(
    max_steps=100,              # Max steps per episode
    reward_success=10.0,        # Reward for success
    reward_failure=-1.0,        # Penalty for failure
    reward_step=-0.1,           # Step penalty
    use_risk_score_reward=True  # Use CVSS-based rewards
)
```

### Sensor Config

```python
sensor = MaskingSensor(
    findings=findings,
    action_mapper=mapper,
    max_attempts_per_task=3,           # Max retries
    enable_safety_constraints=True,    # Enable safety
    log_file="logs/execution.jsonl"    # Log file path
)
```

## Monitoring & Logging

### Real-Time Statistics

```python
stats = sensor.get_statistics()

print(f"Total tasks:        {stats['total_tasks']}")
print(f"Completed:          {stats['completed']}")
print(f"Failed:             {stats['failed']}")
print(f"Success rate:       {stats['success_rate']*100:.1f}%")
print(f"Owned nodes:        {stats['owned_nodes']}")
print(f"Credentials found:  {stats['credentials_found']}")
```

### Execution Logs

Logs are saved in JSONL format:

```json
{
  "timestamp": "2025-11-10T15:45:23",
  "task_id": "finding_001",
  "action": 42,
  "result": "success",
  "safety_violations": [],
  "duration": 3.5,
  "metadata": {
    "target": "192.168.1.10",
    "cvss": 9.8,
    "attempt": 1
  }
}
```

### TensorBoard Integration

The training script logs sensor metrics:

```bash
tensorboard --logdir logs/
```

Metrics include:
- `sensor/total_tasks`
- `sensor/completed`
- `sensor/success_rate`
- `sensor/owned_nodes`
- `sensor/credentials_found`

## Testing

### Demo Script

```bash
python3 scripts/demo_masking_sensor.py --nessus-file scan.xml
```

### Complete Example

```bash
python3 scripts/example_masked_training.py --nessus-file scan.xml --train-steps 10000
```

### Unit Tests

```bash
pytest tests/test_masking_sensor.py -v
```

## Advanced Features

### Policy Engine Integration

```python
from policy_engine import PolicyEngine

# Load organizational policies
policy_engine = PolicyEngine()
policy_engine.load_from_file("policy_config.yaml")

# Create sensor with policies
sensor = MaskingSensor(
    findings=findings,
    action_mapper=mapper,
    policy_engine=policy_engine  # Enforces org policies
)
```

### Network Topology Tracking

```python
# Sensor tracks network state
sensor.owned_nodes = {'192.168.1.10', '192.168.1.20'}
sensor.discovered_credentials = {
    '192.168.1.10': ['admin:pass123'],
    '192.168.1.20': ['root:toor']
}
sensor.network_topology = {
    '192.168.1.10': {'reachable': ['192.168.1.20', '192.168.1.30']},
    '192.168.1.20': {'reachable': ['10.0.0.1']}
}
```

### Deterministic Replay

```python
# Save execution log
sensor.save_execution_log("replay_log.json")

# Replay from log
with open("replay_log.json") as f:
    log = json.load(f)

for entry in log:
    print(f"Task: {entry['task_id']}")
    print(f"Action: {entry['action']}")
    print(f"Result: {entry['result']}")
```

## Performance Considerations

### Memory Usage

- Sensor maintains task queue in memory
- Execution log grows with each action
- Consider periodic log flushing for long runs

### Task Queue Size

```python
# For large scans, limit queue size
if len(findings) > 1000:
    # Filter to high-priority only
    findings = [f for f in findings if f.cvss_base_score >= 7.0]
```

### Parallel Execution

The sensor is **not thread-safe**. For parallel training, create separate sensor instances per environment.

## Troubleshooting

### Issue: No allowed actions

**Cause:** Target not accessible or policy blocks action

**Solution:**
```python
exposure = sensor.get_current_task()
if not exposure.allowed_actions:
    print(f"Blocked: {exposure.task.target_host}")
    # Check why
    if not sensor._is_target_accessible(exposure.task):
        print("  Reason: Target not accessible")
    # Force skip
    sensor.advance({'success': False, 'action': -1, 'duration': 0})
```

### Issue: All tasks failed

**Cause:** Max attempts reached for all tasks

**Solution:**
```python
# Increase max attempts
sensor = MaskingSensor(
    findings=findings,
    action_mapper=mapper,
    max_attempts_per_task=5  # Increased from 3
)
```

### Issue: Training stalls

**Cause:** Agent stuck on hard tasks

**Solution:**
```python
# Add timeout for episodes
config = AUVAPConfig(max_steps=50)  # Force episode end

# Or skip hard tasks
if exposure.attempt_number >= 2:
    # Skip this task
    sensor.advance({'success': False, 'action': -1, 'duration': 0})
```

## Best Practices

1. **Always enable safety constraints** in production
2. **Monitor execution logs** for safety violations
3. **Set reasonable max_attempts** (3-5 typically)
4. **Filter findings** before training (high CVSS only)
5. **Use policy engine** for organizational constraints
6. **Save execution logs** for debugging and compliance
7. **Monitor sensor statistics** during training
8. **Test with demo script** before full training

## References

- **Paper:** Section I, Contribution #3 - Masking Sensor Algorithm
- **Implementation:** `environment/masking_sensor.py`
- **Integration:** `environment/masked_cyberbattle_env.py`
- **Training:** `training/train_ppo_masked.py`
- **Examples:** `scripts/demo_masking_sensor.py`, `scripts/example_masked_training.py`

## License

This implementation follows the licensing of the parent AUVAP project.
