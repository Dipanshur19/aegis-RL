"""
PPO Training Script for AUVAP-CyberBattleSim Integration

This script trains a PPO agent to learn optimal exploitation strategies
using the AUVAP vulnerability assessment pipeline and CyberBattleSim environment.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

# AUVAP-PPO imports
from environment.cyberbattle_wrapper import AUVAPCyberBattleEnv, AUVAPConfig, create_env_from_nessus
from parser import parse_nessus_xml


def make_env(env_config: AUVAPConfig, vulnerability_findings=None, rank: int = 0, seed: int = 0):
    """
    Create a single environment instance.

    Args:
        env_config: Environment configuration
        vulnerability_findings: Vulnerability data
        rank: Process rank for parallel environments
        seed: Random seed

    Returns:
        Function that creates the environment
    """
    def _init():
        env = AUVAPCyberBattleEnv(
            vulnerability_findings=vulnerability_findings,
            config=env_config
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def train_ppo(
    nessus_file: str = None,
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    num_envs: int = 4,
    save_dir: str = "./checkpoints",
    log_dir: str = "./logs",
    eval_freq: int = 10000,
    save_freq: int = 20000,
    seed: int = 42
):
    """
    Train a PPO agent on the AUVAP-CyberBattleSim environment.

    Args:
        nessus_file: Path to Nessus XML file (optional)
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for optimizer
        n_steps: Number of steps per environment per update
        batch_size: Minibatch size
        n_epochs: Number of epochs for optimization
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clipping range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm
        num_envs: Number of parallel environments
        save_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
        eval_freq: Evaluation frequency
        save_freq: Checkpoint save frequency
        seed: Random seed

    Returns:
        Trained PPO model
    """
    # Create directories
    save_dir = Path(save_dir)
    log_dir = Path(log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_cyberbattle_{timestamp}"
    run_log_dir = log_dir / run_name

    print("="*60)
    print("AUVAP-PPO Training")
    print("="*60)
    print(f"Run name: {run_name}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Num environments: {num_envs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Save directory: {save_dir}")
    print(f"Log directory: {run_log_dir}")
    print("="*60)

    # Load vulnerability findings if Nessus file provided
    vulnerability_findings = None
    if nessus_file and os.path.exists(nessus_file):
        print(f"\nLoading vulnerabilities from: {nessus_file}")
        vulnerability_findings = parse_nessus_xml(nessus_file)
        print(f"Loaded {len(vulnerability_findings)} vulnerability findings")
    else:
        print("\nNo Nessus file provided, using default environment")

    # Create environment configuration
    env_config = AUVAPConfig(
        max_steps=100,
        reward_success=10.0,
        reward_failure=-1.0,
        reward_step=-0.1,
        use_risk_score_reward=True
    )

    # Create vectorized environments
    print(f"\nCreating {num_envs} parallel environments...")
    if num_envs > 1:
        env = SubprocVecEnv([
            make_env(env_config, vulnerability_findings, i, seed)
            for i in range(num_envs)
        ])
    else:
        env = DummyVecEnv([make_env(env_config, vulnerability_findings, 0, seed)])

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(env_config, vulnerability_findings, 100, seed)])

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
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        tensorboard_log=str(run_log_dir),
        seed=seed
    )

    # Print model architecture
    print("\nModel Configuration:")
    print(f"  Policy: MlpPolicy")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Total parameters: {sum(p.numel() for p in model.policy.parameters())}")

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // num_envs,  # Adjust for multiple envs
        save_path=str(save_dir / run_name),
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / run_name),
        log_path=str(run_log_dir),
        eval_freq=eval_freq // num_envs,  # Adjust for multiple envs
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    callbacks = [checkpoint_callback, eval_callback]

    # Train the model
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    print(f"Monitor training progress with: tensorboard --logdir {log_dir}")
    print("="*60 + "\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
            tb_log_name="ppo",
            reset_num_timesteps=True,
            progress_bar=True
        )

        # Save final model
        final_model_path = save_dir / run_name / "final_model"
        model.save(final_model_path)
        print(f"\n[OK] Training complete! Final model saved to: {final_model_path}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        interrupted_path = save_dir / run_name / "interrupted_model"
        model.save(interrupted_path)
        print(f"[OK] Model saved to: {interrupted_path}")

    # Close environments
    env.close()
    eval_env.close()

    return model


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train PPO agent on AUVAP-CyberBattleSim")

    # Environment arguments
    parser.add_argument("--nessus-file", type=str, default=None,
                       help="Path to Nessus XML file")

    # Training arguments
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--num-envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048,
                       help="Steps per environment per update")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                       help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2,
                       help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                       help="Entropy coefficient")

    # Saving/logging
    parser.add_argument("--save-dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="./logs",
                       help="Directory for TensorBoard logs")
    parser.add_argument("--save-freq", type=int, default=20000,
                       help="Save frequency")
    parser.add_argument("--eval-freq", type=int, default=10000,
                       help="Evaluation frequency")

    args = parser.parse_args()

    # Train the model
    model = train_ppo(
        nessus_file=args.nessus_file,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        num_envs=args.num_envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        seed=args.seed
    )

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
