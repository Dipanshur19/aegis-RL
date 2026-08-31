# 🚀 Quick Start: Real Script Execution

## TL;DR

```powershell
# Safe simulation mode (DEFAULT) - Recommended for training
python training/train_ppo_priority.py `
    --experiment-report results/experiment_report_20251111_023313.json `
    --exploits-manifest exploits/exploits_20251111_023313/exploits_manifest.json `
    --timesteps 50000

# Real execution mode (DANGEROUS) - Use only in isolated lab
python training/train_ppo_priority.py `
    --experiment-report results/experiment_report_20251111_023313.json `
    --exploits-manifest exploits/exploits_20251111_023313/exploits_manifest.json `
    --timesteps 1000 `
    --real-execution `
    --execution-timeout 30
```

## What Changed?

### Before (Old train_ppo.py)
- ❌ No real exploit execution
- ❌ No priority-based masking
- ❌ Random action selection

### After (New train_ppo_priority.py)
- ✅ **Real script execution** (optional, disabled by default)
- ✅ **Priority-based masking** (CVSS + feasibility + dependencies)
- ✅ **Sequential exploit execution** (ordered by priority)
- ✅ **Intelligent action masking** (blocks completed/unavailable tasks)
- ✅ **Success detection** (keyword-based output parsing)
- ✅ **Timeout protection** (prevents hanging)
- ✅ **Safe by default** (simulation mode)

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiment-report` | **Required** | Path to `experiment_report_*.json` |
| `--exploits-manifest` | **Required** | Path to `exploits_manifest.json` |
| `--timesteps` | 50000 | Total training timesteps |
| `--real-execution` | **False** | Enable real script execution |
| `--execution-timeout` | 30 | Script timeout (seconds) |
| `--lr` | 0.0003 | Learning rate |
| `--n-steps` | 2048 | Steps per PPO update |
| `--batch-size` | 64 | Minibatch size |
| `--n-epochs` | 10 | Training epochs |
| `--gamma` | 0.99 | Discount factor |
| `--seed` | 42 | Random seed |
| `--save-dir` | ./checkpoints | Model save directory |
| `--log-dir` | ./logs | TensorBoard log directory |

## File Structure

```
AUVAP-PPO-master/
├── training/
│   └── train_ppo_priority.py    # ⭐ Main training script with real execution
├── priority_masking.py           # Task prioritization and masking
├── results/
│   └── experiment_report_*.json  # Input: Classification results
├── exploits/
│   └── exploits_*/
│       ├── exploits_manifest.json # Input: Exploit metadata
│       ├── exploit_*.py           # Generated Python exploits
│       ├── exploit_*.ps1          # Generated PowerShell exploits
│       └── exploit_*.sh           # Generated Bash exploits
├── checkpoints/
│   └── ppo_masked_*/              # Output: Trained models
├── logs/
│   └── ppo_masked_*/              # Output: TensorBoard logs
├── REAL_EXECUTION_README.md       # 📖 Detailed usage guide
├── REAL_EXECUTION_SUMMARY.md      # 📊 Implementation overview
└── REAL_EXECUTION_QUICKSTART.md   # 🚀 This file
```

## Example Workflow

### 1️⃣ Generate Exploits (if not done)
```powershell
# Parse Nessus scan
python parser.py auvap_nessus_25_findings.xml

# Run full pipeline
python experiment.py auvap_nessus_25_findings.xml

# Generate exploit scripts
python exploit_generator.py results/tasks_manifest_*.json
```

### 2️⃣ Train with Simulation (Safe & Fast)
```powershell
python training/train_ppo_priority.py `
    --experiment-report results/experiment_report_20251111_023313.json `
    --exploits-manifest exploits/exploits_20251111_023313/exploits_manifest.json `
    --timesteps 50000
```

**Expected Output:**
```
============================================================
AUVAP-PPO PRIORITY-MASKED TRAINING
============================================================
Run name: ppo_masked_20251111_032500
Total timesteps: 50000
Creating priority-masked environment...
  - Real script execution: DISABLED (simulation)

[Priority Masker] Loaded 8 tasks
Starting training...
✓ Training complete! Model saved to: checkpoints/ppo_masked_*/final_model
```

### 3️⃣ Test with Real Execution (Optional, Isolated Lab Only)
```powershell
# ⚠️ WARNING: This actually executes exploit scripts!
python training/train_ppo_priority.py `
    --experiment-report results/experiment_report_20251111_023313.json `
    --exploits-manifest exploits/exploits_20251111_023313/exploits_manifest.json `
    --timesteps 1000 `
    --real-execution
```

**Expected Output:**
```
Creating priority-masked environment...
  - Real script execution: ENABLED
  - Execution timeout: 30s

[EXEC] Running python script: exploit_ghostcat_10_0_1_5_8009.py
[EXEC] Target: 10.0.1.5:8009 (CVE-2020-1938)
[EXEC] ✅ SUCCESS - Exploit completed
[EXEC] Output (first 500 chars):
Attempting Ghostcat exploit...
Connection established
Shell access granted

[Priority Masker] ✅ Task task_0 completed (success=True)
```

### 4️⃣ Monitor Training (Optional)
```powershell
tensorboard --logdir logs
# Open browser to http://localhost:6006
```

## How It Works

### Simulation Mode (Default)
1. Load prioritized task list from reports
2. Create Gym environment with action masking
3. PPO agent selects actions (0-7 = task indices)
4. Environment validates action via mask
5. **Simulate** exploit with probability model:
   - CVSS ≥ 9.0 → 90% success
   - CVSS ≥ 7.0 → 75% success
   - CVSS ≥ 4.0 → 60% success
   - CVSS < 4.0 → 40% success
6. Calculate reward (success + CVSS bonus - step penalty)
7. Update action mask (mark task completed)
8. Repeat until all tasks completed or max steps

### Real Execution Mode
1-4. Same as simulation mode
5. **Execute** actual exploit script:
   - Select executor (python/powershell/bash)
   - Run script with timeout (default 30s)
   - Capture stdout + stderr
   - Parse output for success keywords
   - Return success/failure based on keywords + exit code
6-8. Same as simulation mode

## Safety Checklist

Before using `--real-execution`:

- [ ] I am in an **isolated lab environment**
- [ ] I have **authorization** to test these targets
- [ ] Targets are **not production systems**
- [ ] Network is **isolated** from production
- [ ] I have **backups** of important data
- [ ] I understand exploits may **crash services**
- [ ] I have **logging** enabled
- [ ] I will **monitor** execution closely
- [ ] I know how to **stop training** (Ctrl+C)
- [ ] I have read the **legal warnings**

## Troubleshooting

### "Script not found"
```
[EXEC] Script not found: exploits/.../exploit_*.py
```
**Fix**: Regenerate exploits with `exploit_generator.py`

### "Timeout exceeded"
```
[EXEC] ⏱️ TIMEOUT - Script exceeded 30s
```
**Fix**: Increase timeout with `--execution-timeout 60`

### "All actions blocked"
```
[Priority Masker] All tasks blocked (dependencies not met)
```
**Fix**: Check network access in `priority_masking.py` (line ~130)

### Training too slow with real execution
**Fix**: Reduce `--timesteps` or switch to simulation mode

## Performance Tips

| Goal | Recommendation |
|------|----------------|
| Fast training | Use simulation mode |
| Realistic evaluation | Use real execution with low timesteps |
| Best of both | Train with simulation, evaluate with real execution |
| Debugging | Use `--timesteps 1000` for quick tests |
| Production | Use simulation with 50K-100K timesteps |

## Key Metrics

Monitor these in TensorBoard:

- **ep_rew_mean**: Average episode reward (target: increasing)
- **ep_len_mean**: Episode length (target: decreasing over time)
- **masked/invalid_actions_per_episode**: Invalid actions (target: 0)
- **policy_loss**: Policy gradient loss (target: stable)
- **value_loss**: Value function loss (target: decreasing)

## 🎯 Quick Decision Matrix

| Scenario | Mode | Timesteps | Timeout |
|----------|------|-----------|---------|
| 🏗️ Development | Simulation | 10K | N/A |
| 🧪 Testing | Real Execution | 1K | 30s |
| 🚀 Training | Simulation | 50K-100K | N/A |
| ✅ Validation | Real Execution | 5K | 30s |
| 🐛 Debugging | Simulation | 1K | N/A |

## Emergency Stop

If training with real execution causes problems:

1. **Ctrl+C** in terminal (saves interrupted model)
2. Check running processes: `Get-Process python`
3. Kill if needed: `Stop-Process -Name python -Force`
4. Check network activity: `netstat -an | findstr ESTABLISHED`

## Support

- 📖 Full documentation: `REAL_EXECUTION_README.md`
- 📊 Implementation details: `REAL_EXECUTION_SUMMARY.md`
- 💬 Issues: Check logs in `logs/ppo_masked_*/`
- 🔧 Code: `training/train_ppo_priority.py` (line 200+ for execution logic)

---

**Remember**: Real execution is **powerful but dangerous**. Always use simulation mode unless you specifically need real execution in a controlled environment! 🔒
