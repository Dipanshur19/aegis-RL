"""
PPO Evaluation Script for AUVAP-CyberBattleSim Integration

This script evaluates a trained PPO agent and generates performance metrics.
"""

import os
import sys
import argparse
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# AUVAP-PPO imports
from environment.cyberbattle_wrapper import AUVAPCyberBattleEnv, AUVAPConfig
from parser import parse_nessus_xml


def evaluate_policy(
    model_path: str,
    nessus_file: str = None,
    n_eval_episodes: int = 100,
    render: bool = False,
    deterministic: bool = True,
    verbose: bool = True
):
    """
    Evaluate a trained PPO policy.

    Args:
        model_path: Path to saved model
        nessus_file: Path to Nessus XML file (optional)
        n_eval_episodes: Number of evaluation episodes
        render: Whether to render episodes
        deterministic: Use deterministic actions
        verbose: Print detailed information

    Returns:
        Dictionary with evaluation metrics
    """
    print("="*60)
    print("PPO Policy Evaluation")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_eval_episodes}")
    print(f"Deterministic: {deterministic}")
    print("="*60 + "\n")

    # Load vulnerability findings if provided
    vulnerability_findings = None
    if nessus_file and os.path.exists(nessus_file):
        print(f"Loading vulnerabilities from: {nessus_file}")
        vulnerability_findings = parse_nessus_xml(nessus_file)
        print(f"Loaded {len(vulnerability_findings)} findings\n")

    # Create environment
    env_config = AUVAPConfig(
        max_steps=100,
        reward_success=10.0,
        reward_failure=-1.0,
        reward_step=-0.1,
        use_risk_score_reward=True
    )

    env = AUVAPCyberBattleEnv(
        vulnerability_findings=vulnerability_findings,
        config=env_config
    )

    # Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    print("Model loaded successfully\n")

    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    total_exploits = 0

    # Run evaluation episodes
    print(f"Running {n_eval_episodes} evaluation episodes...")
    print("-" * 60)

    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        exploited_this_episode = 0

        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=deterministic)

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if info.get('action_success', False):
                exploited_this_episode += 1

            if render:
                env.render()

        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        total_exploits += exploited_this_episode

        # Consider episode successful if reward > 0
        if episode_reward > 0:
            success_count += 1

        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_eval_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Length: {episode_length} | "
                  f"Exploits: {exploited_this_episode}")

    env.close()

    # Calculate statistics
    episode_rewards = np.array(episode_rewards)
    episode_lengths = np.array(episode_lengths)

    metrics = {
        'n_episodes': n_eval_episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'success_rate': success_count / n_eval_episodes,
        'total_exploits': total_exploits,
        'avg_exploits_per_episode': total_exploits / n_eval_episodes
    }

    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Episodes:              {metrics['n_episodes']}")
    print(f"Mean Reward:           {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Reward Range:          [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
    print(f"Mean Episode Length:   {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    print(f"Success Rate:          {metrics['success_rate']*100:.1f}%")
    print(f"Avg Exploits/Episode:  {metrics['avg_exploits_per_episode']:.2f}")
    print(f"Total Exploits:        {metrics['total_exploits']}")
    print("="*60 + "\n")

    return metrics


def compare_policies(
    model_paths: list,
    nessus_file: str = None,
    n_eval_episodes: int = 50,
    output_file: str = None
):
    """
    Compare multiple trained policies.

    Args:
        model_paths: List of paths to saved models
        nessus_file: Path to Nessus XML file
        n_eval_episodes: Number of evaluation episodes per model
        output_file: Path to save comparison results (JSON)

    Returns:
        Dictionary with comparison results
    """
    print("="*60)
    print("Policy Comparison")
    print("="*60)
    print(f"Comparing {len(model_paths)} models")
    print(f"Episodes per model: {n_eval_episodes}")
    print("="*60 + "\n")

    results = {}

    for i, model_path in enumerate(model_paths):
        model_name = Path(model_path).stem
        print(f"\n[{i+1}/{len(model_paths)}] Evaluating: {model_name}")
        print("-" * 60)

        metrics = evaluate_policy(
            model_path=model_path,
            nessus_file=nessus_file,
            n_eval_episodes=n_eval_episodes,
            verbose=False
        )

        results[model_name] = metrics

    # Print comparison
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    print(f"{'Model':<30} {'Mean Reward':<15} {'Success Rate':<15}")
    print("-" * 60)

    for model_name, metrics in results.items():
        print(f"{model_name:<30} "
              f"{metrics['mean_reward']:>7.2f} ± {metrics['std_reward']:<5.2f} "
              f"{metrics['success_rate']*100:>6.1f}%")

    print("="*60 + "\n")

    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: {output_file}\n")

    return results


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")

    parser.add_argument("model_path", type=str,
                       help="Path to trained model (or comma-separated list for comparison)")
    parser.add_argument("--nessus-file", type=str, default=None,
                       help="Path to Nessus XML file")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                       help="Render episodes")
    parser.add_argument("--stochastic", action="store_true",
                       help="Use stochastic policy (default: deterministic)")
    parser.add_argument("--compare", action="store_true",
                       help="Compare multiple models")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results (JSON)")

    args = parser.parse_args()

    # Check if comparing multiple models
    if args.compare or ',' in args.model_path:
        model_paths = args.model_path.split(',')
        model_paths = [p.strip() for p in model_paths]

        results = compare_policies(
            model_paths=model_paths,
            nessus_file=args.nessus_file,
            n_eval_episodes=args.episodes,
            output_file=args.output
        )
    else:
        # Single model evaluation
        metrics = evaluate_policy(
            model_path=args.model_path,
            nessus_file=args.nessus_file,
            n_eval_episodes=args.episodes,
            render=args.render,
            deterministic=not args.stochastic,
            verbose=True
        )

        # Save results if output file specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"✓ Results saved to: {args.output}\n")


if __name__ == "__main__":
    main()
