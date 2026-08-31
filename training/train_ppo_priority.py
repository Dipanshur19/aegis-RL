"""
Priority-Masked PPO Training Script for AUVAP

This script trains a PPO agent using the priority masking algorithm to:
1. Load prioritized exploit tasks from classification reports
2. Apply dynamic action masking based on dependencies
3. Execute exploits sequentially in priority order
4. Track network access and credentials
"""

import os
import sys
import argparse
import subprocess
import tempfile
import re
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

# AUVAP imports
from priority_masking import PriorityMasker


class PriorityMaskedEnv(gym.Env):
    """
    Gym environment that uses priority masking for action selection.
    
    The environment:
    - Presents prioritized vulnerability tasks
    - Uses action masks to block unavailable actions
    - Simulates exploit execution
    - Tracks network access and credentials
    """
    
    def __init__(self, 
                 priority_masker: PriorityMasker,
                 max_steps: int = 100,
                 reward_success: float = 100.0,
                 reward_failure: float = -10.0,
                 reward_step: float = -1.0,
                 use_real_execution: bool = False,
                 execution_timeout: int = 30):
        """
        Initialize priority-masked environment.
        
        Args:
            priority_masker: PriorityMasker instance with loaded tasks
            max_steps: Maximum steps per episode
            reward_success: Reward for successful exploit
            reward_failure: Penalty for failed exploit
            reward_step: Step penalty to encourage efficiency
            use_real_execution: If True, execute actual exploit scripts
            execution_timeout: Timeout for script execution (seconds)
        """
        super().__init__()
        
        self.masker = priority_masker
        self.max_steps = max_steps
        self.reward_success = reward_success
        self.reward_failure = reward_failure
        self.reward_step = reward_step
        self.use_real_execution = use_real_execution
        self.execution_timeout = execution_timeout
        
        # Action space: one action per task
        self.num_tasks = len(self.masker.tasks)
        self.action_space = spaces.Discrete(self.num_tasks)
        
        # Observation space: [task_features, progress, network_state]
        # Features per task: priority, cvss, completed, available
        obs_size = (
            self.num_tasks * 4 +  # Task features
            1 +                    # Progress ratio
            1 +                    # Accessible hosts count
            1                      # Current step
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.episode_tasks_completed = 0
        self.episode_successes = 0
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset masker state (but keep task list)
        self.masker.completed_tasks = set()
        self.masker.current_access = set()
        self.masker.compromised_credentials = {}
        
        for task in self.masker.tasks:
            task.is_completed = False
            task.is_available = False
            self.masker.current_access.add(task.host)  # External scanner position
        
        self.current_step = 0
        self.episode_tasks_completed = 0
        self.episode_successes = 0
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int):
        """
        Execute action (attempt exploit on selected task).
        
        Args:
            action: Task index to exploit
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Get task for this action
        task = self.masker.get_task_by_index(action)
        
        # Check if action is valid
        action_mask = self.masker.get_action_mask()
        
        if task is None or action_mask[action] == 0:
            # Invalid action: already completed or blocked
            reward = self.reward_failure * 2  # Extra penalty
            terminated = False
            truncated = False
            info = {
                'action_valid': False,
                'task_completed': False,
                'reason': 'Invalid action (blocked or completed)'
            }
        else:
            # Valid action: simulate exploit execution
            success = self._simulate_exploit(task)
            
            # Calculate reward
            if success:
                # Base reward + risk-based bonus
                risk_bonus = (task.cvss_score / 10.0) * 50  # 0-50 bonus
                reward = self.reward_success + risk_bonus + self.reward_step
                self.episode_successes += 1
            else:
                reward = self.reward_failure + self.reward_step
            
            # Mark task as completed
            self.masker.mark_completed(task.task_id, success)
            self.episode_tasks_completed += 1
            
            terminated = False
            truncated = False
            info = {
                'action_valid': True,
                'task_completed': True,
                'success': success,
                'task_id': task.task_id,
                'cve': task.cve,
                'cvss': task.cvss_score,
                'host': task.host
            }
        
        # Check termination conditions
        if self.episode_tasks_completed >= self.num_tasks:
            terminated = True
            info['episode_complete'] = True
        
        if self.current_step >= self.max_steps:
            truncated = True
            info['max_steps_reached'] = True
        
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, info
    
    def _simulate_exploit(self, task) -> bool:
        """
        Execute exploit (real or simulated).
        
        If use_real_execution=True:
        1. Execute the actual exploit script
        2. Parse the output for success indicators
        3. Return success/failure
        
        Otherwise, use probability-based simulation:
        - Higher CVSS = higher success probability
        """
        if self.use_real_execution:
            return self._execute_real_script(task)
        else:
            # Probability-based simulation
            cvss = task.cvss_score
            if cvss >= 9.0:
                success_prob = 0.90
            elif cvss >= 7.0:
                success_prob = 0.75
            elif cvss >= 4.0:
                success_prob = 0.60
            else:
                success_prob = 0.40
            
            return np.random.random() < success_prob
    
    def _execute_real_script(self, task) -> bool:
        """
        Execute the actual exploit script and parse results.
        
        Returns:
            True if exploit succeeded, False otherwise
        """
        # Convert to absolute path
        script_path = Path(task.exploit_script)
        if not script_path.is_absolute():
            script_path = Path.cwd() / script_path
        
        if not script_path.exists():
            print(f"[EXEC] Script not found: {script_path}")
            return False
        
        try:
            # Determine executor based on script language
            # Use absolute paths for scripts to avoid cwd issues
            if task.script_language.lower() == 'python':
                cmd = [sys.executable, str(script_path.absolute())]
            elif task.script_language.lower() == 'powershell':
                cmd = ['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', str(script_path.absolute())]
            elif task.script_language.lower() in ['bash', 'shell']:
                cmd = ['bash', str(script_path.absolute())]
            else:
                print(f"[EXEC] Unsupported language: {task.script_language}")
                return False
            
            print(f"\n[EXEC] Running {task.script_language} script: {script_path.name}")
            print(f"[EXEC] Target: {task.host}:{task.port} ({task.cve})")
            
            # Execute with timeout (run from project root, not script directory)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.execution_timeout
            )
            
            # Parse output for success indicators
            output = result.stdout + result.stderr
            success = self._parse_execution_result(output, result.returncode)
            
            if success:
                print(f"[EXEC] [SUCCESS] Exploit completed")
            else:
                print(f"[EXEC] [FAILED] Exploit unsuccessful")
            
            # Log output (truncated)
            if output:
                print(f"[EXEC] Output (first 500 chars):")
                print(output[:500])
            
            return success
            
        except subprocess.TimeoutExpired:
            print(f"[EXEC] [TIMEOUT] Script exceeded {self.execution_timeout}s")
            return False
        except Exception as e:
            print(f"[EXEC] [ERROR] {type(e).__name__}: {e}")
            return False
    
    def _parse_execution_result(self, output: str, return_code: int) -> bool:
        """
        Parse script output to determine success.
        
        Success indicators:
        - Return code 0
        - Keywords: "success", "exploited", "shell", "access granted"
        - No error keywords: "failed", "error", "denied", "timeout"
        
        Args:
            output: Combined stdout + stderr
            return_code: Process return code
            
        Returns:
            True if exploit appears successful
        """
        output_lower = output.lower()
        
        # Success keywords
        success_keywords = [
            'success', 'successful', 'exploited', 'shell', 'access granted',
            'vulnerability confirmed', 'payload delivered', 'connection established'
        ]
        
        # Failure keywords
        failure_keywords = [
            'failed', 'failure', 'error', 'denied', 'timeout', 'refused',
            'not vulnerable', 'connection closed', 'unable to'
        ]
        
        # Check for success indicators
        has_success = any(keyword in output_lower for keyword in success_keywords)
        has_failure = any(keyword in output_lower for keyword in failure_keywords)
        
        # Decision logic
        if return_code == 0 and has_success and not has_failure:
            return True
        elif has_success and not has_failure:
            return True
        elif return_code == 0 and not has_failure and len(output) > 0:
            # Script ran successfully with output but no clear indicators
            # Consider it a success if no failures detected
            return True
        else:
            return False
    
    def _get_observation(self) -> np.ndarray:
        """
        Build observation vector.
        
        Features:
        - Per-task: [priority_norm, cvss_norm, is_completed, is_available]
        - Global: [progress_ratio, accessible_hosts_norm, step_norm]
        """
        obs = []
        
        # Task features
        for task in self.masker.tasks:
            obs.extend([
                task.priority_score / 100.0,  # Normalize priority
                task.cvss_score / 10.0,        # Normalize CVSS
                1.0 if task.is_completed else 0.0,
                1.0 if task.is_available else 0.0
            ])
        
        # Global features
        progress = len(self.masker.completed_tasks) / max(self.num_tasks, 1)
        accessible_ratio = len(self.masker.current_access) / max(self.num_tasks, 1)
        step_norm = self.current_step / max(self.max_steps, 1)
        
        obs.extend([progress, accessible_ratio, step_norm])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> dict:
        """Get environment info"""
        return {
            'step': self.current_step,
            'tasks_completed': self.episode_tasks_completed,
            'tasks_succeeded': self.episode_successes,
            'accessible_hosts': len(self.masker.current_access)
        }
    
    def action_masks(self) -> np.ndarray:
        """Return current action mask for invalid action masking"""
        return self.masker.get_action_mask()


class MaskedActionCallback(BaseCallback):
    """Callback to log action masking statistics"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_invalid_actions = 0
        self.total_invalid_actions = 0
    
    def _on_step(self) -> bool:
        # Check if last action was invalid
        info = self.locals.get('infos', [{}])[0]
        if not info.get('action_valid', True):
            self.episode_invalid_actions += 1
            self.total_invalid_actions += 1
        
        # Log at episode end
        if self.locals.get('dones', [False])[0]:
            if self.episode_invalid_actions > 0:
                self.logger.record('masked/invalid_actions_per_episode', 
                                 self.episode_invalid_actions)
            self.episode_invalid_actions = 0
        
        return True


def train_ppo_masked(
    experiment_report: str,
    exploits_manifest: str,
    total_timesteps: int = 50000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    save_dir: str = "./checkpoints",
    log_dir: str = "./logs",
    seed: int = 42,
    use_real_execution: bool = False,
    execution_timeout: int = 30
):
    """
    Train PPO with priority masking.
    
    Args:
        experiment_report: Path to experiment_report_*.json
        exploits_manifest: Path to exploits_manifest.json
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        n_steps: Steps per update
        batch_size: Minibatch size
        n_epochs: Number of epochs
        gamma: Discount factor
        save_dir: Checkpoint directory
        log_dir: TensorBoard log directory
        seed: Random seed
        use_real_execution: If True, execute actual exploit scripts
        execution_timeout: Timeout for script execution (seconds)
    """
    # Create run name
    run_name = f"ppo_masked_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("="*60)
    print("AUVAP-PPO PRIORITY-MASKED TRAINING")
    print("="*60)
    print(f"Run name: {run_name}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Save directory: {save_dir}")
    print(f"Log directory: {log_dir}")
    print("="*60 + "\n")
    
    # Create directories
    save_dir = Path(save_dir)
    log_dir = Path(log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir = log_dir / run_name
    
    # Initialize priority masker
    print(f"Loading priority masker...")
    print(f"  - Experiment report: {experiment_report}")
    print(f"  - Exploits manifest: {exploits_manifest}")
    print(f"  - Mode: SEQUENTIAL (one-by-one, highest priority only)\n")
    
    masker = PriorityMasker(experiment_report, exploits_manifest, sequential_mode=True)
    
    # Show initial status
    masker.print_status()
    
    # Create environment
    print("\nCreating priority-masked environment...")
    print(f"  - Real script execution: {'ENABLED' if use_real_execution else 'DISABLED (simulation)'}")
    if use_real_execution:
        print(f"  - Execution timeout: {execution_timeout}s")
    
    def make_env():
        env = PriorityMaskedEnv(
            priority_masker=masker,
            max_steps=len(masker.tasks) * 2,  # Allow retries
            use_real_execution=use_real_execution,
            execution_timeout=execution_timeout
        )
        return env
    
    env = DummyVecEnv([make_env])
    
    # Set seed
    set_random_seed(seed)
    
    # Create PPO model
    print("\nInitializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        tensorboard_log=str(run_log_dir),
        seed=seed
    )
    
    print(f"\nModel Configuration:")
    print(f"  Policy: MlpPolicy")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Total parameters: {sum(p.numel() for p in model.policy.parameters())}")
    
    # Create callback
    callback = MaskedActionCallback()
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    print(f"Monitor with: tensorboard --logdir {log_dir}")
    print("="*60 + "\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,
            tb_log_name="ppo_masked",
            progress_bar=True
        )
        
        # Save final model
        final_model_path = save_dir / run_name / "final_model"
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(final_model_path)
        
        print(f"\n[OK] Training complete! Model saved to: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        interrupted_path = save_dir / run_name / "interrupted_model"
        interrupted_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(interrupted_path)
        print(f"[OK] Model saved to: {interrupted_path}")
    
    # Final status
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    masker.print_status()
    
    env.close()
    
    return model


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(
        description="Train PPO with priority masking"
    )
    
    # Required arguments
    parser.add_argument("--experiment-report", type=str, required=True,
                       help="Path to experiment_report_*.json")
    parser.add_argument("--exploits-manifest", type=str, required=True,
                       help="Path to exploits_manifest.json")
    
    # Training arguments
    parser.add_argument("--timesteps", type=int, default=50000,
                       help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048,
                       help="Steps per update")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Saving/logging
    parser.add_argument("--save-dir", type=str, default="./checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="./logs",
                       help="TensorBoard log directory")
    
    # Execution options
    parser.add_argument("--real-execution", action="store_true",
                       help="Execute actual exploit scripts (default: simulation)")
    parser.add_argument("--execution-timeout", type=int, default=30,
                       help="Timeout for script execution in seconds")
    
    args = parser.parse_args()
    
    # Train
    model = train_ppo_masked(
        experiment_report=args.experiment_report,
        exploits_manifest=args.exploits_manifest,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        use_real_execution=args.real_execution,
        execution_timeout=args.execution_timeout
    )


if __name__ == "__main__":
    main()
