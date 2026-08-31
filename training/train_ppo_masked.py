"""
PPO Training with Masking Sensor Integration

This script trains a PPO agent using the masking sensor algorithm for
controlled, safe task exposure.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

# AUVAP-PPO imports
from environment.masked_cyberbattle_env import MaskedCyberBattleEnv, create_masked_env_from_nessus
from environment.cyberbattle_wrapper import AUVAPConfig
from parser import parse_nessus_xml


class MaskingSensorCallback(BaseCallback):
    """
    Callback to log masking sensor statistics during training.
    """

    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        """Called at each step."""
        if self.n_calls % self.log_freq == 0:
            # Log sensor statistics
            env = self.training_env.envs[0]
            if hasattr(env, 'get_sensor_statistics'):
                stats = env.get_sensor_statistics()

                # Log to TensorBoard
                self.logger.record("sensor/total_tasks", stats['total_tasks'])
                self.logger.record("sensor/completed", stats['completed'])
                self.logger.record("sensor/failed", stats['failed'])
                self.logger.record("sensor/success_rate", stats['success_rate'])
                self.logger.record("sensor/owned_nodes", stats['owned_nodes'])
                self.logger.record("sensor/credentials_found", stats['credentials_found'])

                if self.verbose > 0:
                    print(f"\n[Sensor Stats @ {self.num_timesteps} steps]")
                    print(f"  Success rate: {stats['success_rate']*100:.1f}%")
                    print(f"  Completed: {stats['completed']}/{stats['total_tasks']}")
                    print(f"  Owned nodes: {stats['owned_nodes']}")

        return True


class EpisodeEndCallback(BaseCallback):
    """
    Callback to handle episode completion and sensor advancement.
    """

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """Called at each step."""
        # Check for episode completion
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1

            # Get episode info
            if 'infos' in self.locals:
                info = self.locals['infos'][0]

                # Log episode metrics
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']

                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)

                    self.logger.record("episode/reward", ep_reward)
                    self.logger.record("episode/length", ep_length)
                    self.logger.record("episode/count", self.episode_count)

                    if self.verbose > 0 and self.episode_count % 10 == 0:
                        mean_reward = np.mean(self.episode_rewards[-10:])
                        print(f"\n[Episode {self.episode_count}]")
                        print(f"  Mean reward (last 10): {mean_reward:.2f}")
                        print(f"  This episode: {ep_reward:.2f}")

        return True


def make_masked_env(nessus_file: str, config: AUVAPConfig, seed: int = 0):
    """
    Create a masked environment instance.

    Args:
        nessus_file: Path to Nessus XML file
        config: Environment configuration
        seed: Random seed

    Returns:
        Function that creates the environment
    """
    def _init():
        env = create_masked_env_from_nessus(
            nessus_file=nessus_file,
            config=config
        )
        env.reset(seed=seed)
        env = Monitor(env)
        return env

    set_random_seed(seed)
    return _init


def train_ppo_masked(
    nessus_file: str,
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
    save_dir: str = "./checkpoints",
    log_dir: str = "./logs",
    eval_freq: int = 10000,
    save_freq: int = 20000,
    seed: int = 42,
    verbose: int = 1
):
    """
    Train a PPO agent with masking sensor integration.

    Args:
        nessus_file: Path to Nessus XML file
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        n_steps: Steps per update
        batch_size: Minibatch size
        n_epochs: Number of epochs
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm
        save_dir: Checkpoint directory
        log_dir: Log directory
        eval_freq: Evaluation frequency
        save_freq: Save frequency
        seed: Random seed
        verbose: Verbosity level

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
    run_name = f"ppo_masked_{timestamp}"
    run_log_dir = log_dir / run_name

    print("="*70)
    print("PPO Training with Masking Sensor")
    print("="*70)
    print(f"Run name: {run_name}")
    print(f"Nessus file: {nessus_file}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Save directory: {save_dir}")
    print(f"Log directory: {run_log_dir}")
    print("="*70)

    # Load vulnerability findings
    if not os.path.exists(nessus_file):
        raise FileNotFoundError(f"Nessus file not found: {nessus_file}")

    print(f"\nLoading vulnerabilities from: {nessus_file}")
    findings = parse_nessus_xml(nessus_file)
    print(f"Loaded {len(findings)} vulnerability findings")

    # Create environment configuration
    env_config = AUVAPConfig(
        max_steps=100,
        reward_success=10.0,
        reward_failure=-1.0,
        reward_step=-0.1,
        use_risk_score_reward=True
    )

    # Create environment
    print(f"\nCreating masked environment...")
    env = DummyVecEnv([make_masked_env(nessus_file, env_config, seed)])

    # Create evaluation environment
    eval_env = DummyVecEnv([make_masked_env(nessus_file, env_config, seed + 100)])

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
        verbose=verbose,
        tensorboard_log=str(run_log_dir),
        seed=seed
    )

    print("\nModel Configuration:")
    print(f"  Policy: MlpPolicy")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Total parameters: {sum(p.numel() for p in model.policy.parameters())}")

    # Create callbacks
    sensor_callback = MaskingSensorCallback(log_freq=100, verbose=verbose)
    episode_callback = EpisodeEndCallback(verbose=verbose)

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(save_dir / run_name),
        name_prefix="ppo_masked_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / run_name),
        log_path=str(run_log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    callbacks = [sensor_callback, episode_callback, checkpoint_callback, eval_callback]

    # Train the model
    print("\n" + "="*70)
    print("Starting training with masking sensor...")
    print("="*70)
    print(f"Monitor training progress with: tensorboard --logdir {log_dir}")
    print("="*70 + "\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
            tb_log_name="ppo_masked",
            reset_num_timesteps=True,
            progress_bar=False
        )

        # Save final model
        final_model_path = save_dir / run_name / "final_model"
        model.save(final_model_path)
        print(f"\n[OK] Training complete! Final model saved to: {final_model_path}")

        # Save execution log from environment
        log_output = save_dir / run_name / "execution_log.json"
        base_env = env.envs[0]
        if hasattr(base_env, 'save_execution_log'):
            base_env.save_execution_log(str(log_output))
            print(f"[OK] Execution log saved to: {log_output}")

        # Print final statistics
        if hasattr(base_env, 'get_sensor_statistics'):
            stats = base_env.get_sensor_statistics()
            print("\n" + "="*70)
            print("Final Sensor Statistics")
            print("="*70)
            print(f"Total tasks:        {stats['total_tasks']}")
            print(f"Completed:          {stats['completed']}")
            print(f"Failed:             {stats['failed']}")
            print(f"Success rate:       {stats['success_rate']*100:.1f}%")
            print(f"Owned nodes:        {stats['owned_nodes']}")
            print(f"Credentials found:  {stats['credentials_found']}")
            print("="*70)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        interrupted_path = save_dir / run_name / "interrupted_model"
        model.save(interrupted_path)
        print(f"✓ Model saved to: {interrupted_path}")

    # Close environments
    env.close()
    eval_env.close()

    return model


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train PPO with masking sensor")

    # Environment arguments
    parser.add_argument("--nessus-file", type=str, required=True,
                       help="Path to Nessus XML file")

    # Training arguments
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # PPO hyperparameters
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

    # Verbosity
    parser.add_argument("--verbose", type=int, default=1,
                       help="Verbosity level (0=quiet, 1=info, 2=debug)")

    args = parser.parse_args()

    # Train the model
    model = train_ppo_masked(
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
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        seed=args.seed,
        verbose=args.verbose
    )

    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
