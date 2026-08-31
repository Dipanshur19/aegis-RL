# Real Script Execution Guide

## Overview

The PPO training system now supports **real exploit script execution** in addition to simulation mode. When enabled, the agent will actually run the generated Python/PowerShell/Bash exploit scripts against targets.

## Usage Modes

### 1. Simulation Mode (Default - Safe)
```powershell
python training/train_ppo_priority.py `
    --experiment-report results/experiment_report_20251111_023313.json `
    --exploits-manifest exploits/exploits_20251111_023313/exploits_manifest.json `
    --timesteps 50000
```

**Characteristics:**
- ✅ Safe: No actual network activity
- ✅ Fast: No I/O or network delays
- ✅ Probability-based: Higher CVSS = higher success rate
- ✅ Good for training and testing

### 2. Real Execution Mode (Dangerous)
```powershell
python training/train_ppo_priority.py `
    --experiment-report results/experiment_report_20251111_023313.json `
    --exploits-manifest exploits/exploits_20251111_023313/exploits_manifest.json `
    --timesteps 50000 `
    --real-execution `
    --execution-timeout 30
```

**Characteristics:**
- ⚠️ **DANGEROUS**: Actually executes exploit code
- 🐌 Slow: Network I/O, script execution overhead
- 🎯 Realistic: Real success/failure based on actual target state
- 📊 Uses output parsing to determine success

## Real Execution Details

### Script Execution

The system automatically selects the correct executor:

| Script Language | Executor | Command |
|----------------|----------|---------|
| Python | `python.exe` | `python script.py` |
| PowerShell | `powershell.exe` | `powershell -ExecutionPolicy Bypass -File script.ps1` |
| Bash/Shell | `bash` | `bash script.sh` |

### Success Detection

Scripts are considered successful if:

1. **Return Code**: Exit code 0
2. **Success Keywords Found**: 
   - "success", "successful", "exploited", "shell"
   - "access granted", "vulnerability confirmed"
   - "payload delivered", "connection established"
3. **No Failure Keywords**: 
   - "failed", "error", "denied", "timeout"
   - "refused", "not vulnerable", "unable to"

### Timeout Protection

- Default timeout: **30 seconds** per script
- Configurable via `--execution-timeout N`
- Scripts exceeding timeout are marked as failed
- Prevents hanging on unresponsive targets

### Safety Features

1. **Isolated Execution**: Scripts run in their own subprocess
2. **Timeout Enforcement**: Automatic termination after timeout
3. **Error Catching**: Exceptions don't crash training
4. **Output Logging**: First 500 chars of output logged
5. **Working Directory**: Scripts run from their exploit folder

## Example Training Logs

### Simulation Mode
```
[Priority Masker] ✅ Task task_2 completed (success=True)
                  Current access: 5 hosts
```

### Real Execution Mode
```
[EXEC] Running python script: exploit_ghostcat_10_0_1_5_8009.py
[EXEC] Target: 10.0.1.5:8009 (CVE-2020-1938)
[EXEC] ✅ SUCCESS - Exploit completed
[EXEC] Output (first 500 chars):
Attempting Ghostcat exploit...
Connection established to 10.0.1.5:8009
Payload delivered successfully
Shell access granted
[Priority Masker] ✅ Task task_2 completed (success=True)
                  Current access: 6 hosts
```

## Warnings

⚠️ **USE REAL EXECUTION ONLY IN CONTROLLED ENVIRONMENTS:**

1. **Legal**: Only run against targets you own or have permission to test
2. **Network**: May generate significant malicious traffic
3. **Safety**: Exploits can crash services or systems
4. **Ethics**: Unauthorized access is illegal in most jurisdictions
5. **Training Time**: Real execution is 10-100x slower than simulation

## Recommended Workflow

1. **Development Phase**: Use simulation mode
   ```powershell
   python training/train_ppo_priority.py --timesteps 10000
   ```

2. **Testing Phase**: Use real execution with short runs
   ```powershell
   python training/train_ppo_priority.py --timesteps 1000 --real-execution
   ```

3. **Production Training**: Use simulation for speed, real execution for validation
   ```powershell
   # Train with simulation
   python training/train_ppo_priority.py --timesteps 100000
   
   # Validate with real execution
   python training/train_ppo_priority.py --timesteps 1000 --real-execution
   ```

## Troubleshooting

### Script Not Found
```
[EXEC] Script not found: exploits/exploits_20251111_023313/exploit_script.py
```
**Solution**: Verify `exploit_script` paths in `exploits_manifest.json` are correct

### Unsupported Language
```
[EXEC] Unsupported language: javascript
```
**Solution**: Only Python, PowerShell, and Bash are supported

### Timeout Issues
```
[EXEC] ⏱️ TIMEOUT - Script exceeded 30s
```
**Solution**: Increase timeout with `--execution-timeout 60`

### Permission Errors (PowerShell)
```
[EXEC] ❌ ERROR - Access denied
```
**Solution**: Run PowerShell as Administrator or adjust execution policy

## Architecture

```
PriorityMaskedEnv
├── step(action) 
│   ├── task = masker.get_task_by_index(action)
│   ├── success = _simulate_exploit(task)
│   │   ├── if use_real_execution:
│   │   │   └── _execute_real_script(task)
│   │   │       ├── subprocess.run(cmd, timeout=30s)
│   │   │       ├── _parse_execution_result(output, return_code)
│   │   │       └── return success/failure
│   │   └── else:
│   │       └── probability-based simulation
│   └── masker.mark_completed(task_id, success)
└── action_masks()
    └── masker.get_action_mask()
```

## Performance Comparison

| Mode | Episodes/Hour | Training Time (50K steps) | Realism |
|------|---------------|---------------------------|---------|
| Simulation | ~500 | 10-15 minutes | Low |
| Real Execution | ~10-50 | 1-5 hours | High |

## Future Enhancements

- [ ] Script output structured parsing (JSON results)
- [ ] Resource limits (CPU, memory, disk)
- [ ] Sandbox integration (Docker containers)
- [ ] Real-time exploit telemetry
- [ ] Multi-threaded execution (parallel scripts)
- [ ] Exploit result caching (avoid re-running identical exploits)
