# AUVAP-PPO: CyberBattleSim Integration

This directory contains the integration of **Proximal Policy Optimization (PPO)** with **Microsoft CyberBattleSim** for training reinforcement learning agents to perform automated penetration testing based on the AUVAP vulnerability assessment pipeline.

## Overview

The integration bridges three key components:

1. **AUVAP Pipeline** - Vulnerability assessment and task management
2. **CyberBattleSim** - Microsoft's cybersecurity simulation environment
3. **Stable-Baselines3 PPO** - State-of-the-art RL algorithm implementation

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  AUVAP Pipeline │────▶│  CyberBattleSim  │────▶│  PPO Agent  │
│  (Vulns/Tasks)  │     │   Environment    │     │  (Training) │
└─────────────────┘     └──────────────────┘     └─────────────┘
```

## Project Structure

```
AUVAP-PPO/
├── environment/                           # RL Environment
│   ├── cyberbattle_wrapper.py    # Main Gym environment wrapper
│   ├── action_mapper.py           # Action space mapping
│   ├── observation_builder.py    # Observation space builder
│   └── reward_shaper.py           # Reward calculation
│
├── training/                      # Training Scripts
│   ├── train_ppo.py              # Main training script
│   └── evaluate_ppo.py           # Evaluation script
│
├── config/                        # Configuration Files
│   ├── ppo_config.yaml           # PPO hyperparameters
│   └── training_config.yaml      # Training presets
│
├── checkpoints/                   # Model checkpoints (created during training)
├── logs/                         # TensorBoard logs (created during training)
│
└── [existing AUVAP files...]
```

## Installation

### 1. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install CyberBattleSim (from GitHub)
cd /tmp
git clone https://github.com/microsoft/CyberBattleSim.git
cd CyberBattleSim
pip install -e .
cd /home/user/AUVAP-PPO
```

### 2. Verify Installation

```bash
python3 -c "import gymnasium; import stable_baselines3; import cyberbattle; print('✓ All imports successful!')"
```

## Quick Start

### Training a PPO Agent

#### Basic Training (Default Settings)

```bash
python3 training/train_ppo.py
```

#### Training with Nessus Data

```bash
python3 training/train_ppo.py --nessus-file auvap_nessus_25_findings.xml
```

#### Custom Training Configuration

```bash
python3 training/train_ppo.py \
    --nessus-file auvap_nessus_100_findings.xml \
    --timesteps 200000 \
    --num-envs 8 \
    --lr 0.0003 \
    --batch-size 128 \
    --save-dir ./checkpoints \
    --log-dir ./logs
```

### Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir ./logs
```

Then open your browser to `http://localhost:6006`

### Evaluating a Trained Agent

#### Evaluate a Single Model

```bash
python3 training/evaluate_ppo.py checkpoints/ppo_cyberbattle_20250110_120000/final_model.zip \
    --episodes 100 \
    --nessus-file auvap_nessus_25_findings.xml
```

#### Compare Multiple Models

```bash
python3 training/evaluate_ppo.py \
    model1.zip,model2.zip,model3.zip \
    --compare \
    --episodes 50 \
    --output results.json
```

## Configuration

### PPO Hyperparameters

Edit `config/ppo_config.yaml` to customize:

- **Learning rate**: `ppo.learning_rate` (default: 0.0003)
- **Batch size**: `ppo.batch_size` (default: 64)
- **Entropy coefficient**: `ppo.ent_coef` (default: 0.01)
- **Discount factor**: `ppo.gamma` (default: 0.99)
- **Clip range**: `ppo.clip_range` (default: 0.2)

### Environment Settings

Edit `config/ppo_config.yaml` to customize environment:

- **Max steps**: `environment.max_steps` (default: 100)
- **Rewards**: `environment.reward_success`, `environment.reward_failure`
- **Risk shaping**: `environment.use_risk_score_reward` (default: true)

### Training Presets

Use predefined configurations from `config/training_config.yaml`:

- **quick_test**: 10K timesteps, 2 envs (fast debugging)
- **development**: 50K timesteps, 4 envs (iteration)
- **standard**: 200K timesteps, 8 envs (good performance)
- **long**: 1M timesteps, 16 envs (best performance)

## Architecture

### Environment (env/cyberbattle_wrapper.py)

The custom Gym environment that bridges AUVAP with CyberBattleSim:

- **Observation Space**: Network topology, vulnerability features, temporal info
- **Action Space**: Discrete actions mapped to exploit tasks
- **Reward Function**: Success/failure + CVSS-based risk shaping

```python
from environment.cyberbattle_wrapper import AUVAPCyberBattleEnv, AUVAPConfig

config = AUVAPConfig(max_steps=100, use_risk_score_reward=True)
env = AUVAPCyberBattleEnv(vulnerability_findings=findings, config=config)
```

### Action Mapping (env/action_mapper.py)

Maps AUVAP `ExploitTask` objects to discrete action IDs:

```python
from environment.action_mapper import ActionMapper

mapper = ActionMapper(max_actions=100)
action_ids = mapper.register_tasks_from_findings(findings)
task = mapper.get_task_from_action(action_id)
```

### Observation Building (env/observation_builder.py)

Converts network state to observation vectors:

```python
from environment.observation_builder import ObservationBuilder, ObservationConfig

config = ObservationConfig(max_nodes=20, include_vuln_features=True)
builder = ObservationBuilder(config)
obs = builder.build_observation(network, owned_nodes, discovered_nodes, vulnerabilities)
```

### Reward Shaping (env/reward_shaper.py)

Calculates rewards using AUVAP's risk scoring:

```python
from environment.reward_shaper import RewardShaper, RewardConfig

config = RewardConfig(use_risk_shaping=True, risk_weight=0.5)
shaper = RewardShaper(config)
reward = shaper.calculate_reward(action_success, exploit_info, strategic_info)
```

## Training Parameters

### Recommended Settings

#### Small Network (< 10 nodes)
```bash
python3 training/train_ppo.py \
    --timesteps 50000 \
    --num-envs 4 \
    --n-steps 1024 \
    --batch-size 64
```

#### Medium Network (10-20 nodes)
```bash
python3 training/train_ppo.py \
    --timesteps 200000 \
    --num-envs 8 \
    --n-steps 2048 \
    --batch-size 128
```

#### Large Network (> 20 nodes)
```bash
python3 training/train_ppo.py \
    --timesteps 500000 \
    --num-envs 16 \
    --n-steps 2048 \
    --batch-size 256
```

## Key Features

### 1. **AUVAP Integration**
- Uses real vulnerability data from Nessus scans
- Leverages existing risk scoring (`compute_risk_score`)
- Integrates with `ExploitTask` task management
- Applies organizational policies as action masking

### 2. **Risk-Based Reward Shaping**
- Rewards scaled by CVSS scores
- Bonus for critical nodes
- Bonus for lateral movement
- Efficiency incentives (step penalties)

### 3. **Flexible Configuration**
- YAML-based configuration
- Multiple training presets
- Customizable network architectures
- Adjustable observation/action spaces

### 4. **Advanced Training Features**
- Parallel environment execution
- Automatic checkpointing
- TensorBoard logging
- Evaluation callbacks
- Model comparison tools

## Evaluation Metrics

The evaluation script provides:

- **Mean Episode Reward**: Average cumulative reward
- **Success Rate**: Percentage of successful episodes
- **Episode Length**: Average steps per episode
- **Exploits per Episode**: Average successful exploitations
- **Reward Distribution**: Min, max, std deviation

## Example Workflow

### 1. Prepare Vulnerability Data

```bash
# Use existing Nessus scan
ls auvap_nessus_*.xml
```

### 2. Train Initial Model

```bash
python3 training/train_ppo.py \
    --nessus-file auvap_nessus_25_findings.xml \
    --timesteps 100000 \
    --num-envs 4 \
    --save-dir ./checkpoints/run1
```

### 3. Monitor Progress

```bash
tensorboard --logdir ./logs
```

### 4. Evaluate Performance

```bash
python3 training/evaluate_ppo.py \
    checkpoints/run1/final_model.zip \
    --episodes 100 \
    --nessus-file auvap_nessus_25_findings.xml
```

### 5. Iterate and Improve

Adjust hyperparameters in `config/ppo_config.yaml` and retrain:

```bash
# Try higher learning rate for faster convergence
python3 training/train_ppo.py \
    --nessus-file auvap_nessus_25_findings.xml \
    --lr 0.001 \
    --timesteps 150000 \
    --save-dir ./checkpoints/run2
```

### 6. Compare Models

```bash
python3 training/evaluate_ppo.py \
    checkpoints/run1/final_model.zip,checkpoints/run2/final_model.zip \
    --compare \
    --episodes 50 \
    --output comparison.json
```

## Troubleshooting

### Issue: Import errors

```bash
# Verify all packages are installed
pip install -r requirements.txt

# Check CyberBattleSim installation
python3 -c "import cyberbattle; print(cyberbattle.__file__)"
```

### Issue: Training crashes with OOM

Reduce the number of parallel environments:
```bash
python3 training/train_ppo.py --num-envs 2
```

Or reduce batch size:
```bash
python3 training/train_ppo.py --batch-size 32
```

### Issue: Poor training performance

Try these adjustments:
1. Increase entropy coefficient for more exploration: `--ent-coef 0.05`
2. Reduce learning rate for stability: `--lr 0.0001`
3. Increase training timesteps: `--timesteps 500000`
4. Adjust reward shaping in `config/ppo_config.yaml`

### Issue: CyberBattleSim not available

The environment will run in mock mode for testing. To use actual CyberBattleSim:
```bash
# Install from source
cd /tmp
git clone https://github.com/microsoft/CyberBattleSim.git
cd CyberBattleSim
pip install -e .
```

## Next Steps

1. **Enhance CyberBattleSim Integration**
   - Map VAFinding objects to CyberBattleSim vulnerabilities
   - Build network topologies from scan data
   - Implement realistic attack scenarios

2. **Advanced Features**
   - Multi-agent training (attacker vs defender)
   - Curriculum learning (start simple, increase complexity)
   - Transfer learning across different networks
   - Integration with actual exploitation tools

3. **Experiment Tracking**
   - Add Weights & Biases integration
   - Hyperparameter optimization (Optuna)
   - Automated benchmarking

4. **Production Deployment**
   - Model serving infrastructure
   - Real-time decision making
   - Safety constraints and guardrails

## References

- [Microsoft CyberBattleSim](https://github.com/microsoft/CyberBattleSim)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning Up - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

## Support

For issues or questions:
1. Check this README
2. Review configuration files in `config/`
3. Check existing AUVAP documentation
4. Review CyberBattleSim documentation

## License

This integration follows the licensing of the parent AUVAP project and CyberBattleSim (MIT License).
