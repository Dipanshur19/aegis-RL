"""
Complete Example: PPO Training with Masking Sensor

This script demonstrates the full integration of:
- AUVAP vulnerability assessment
- Masking sensor algorithm
- PPO reinforcement learning
- CyberBattleSim environment
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.masked_cyberbattle_env import MaskedCyberBattleEnv
from environment.cyberbattle_wrapper import AUVAPConfig
from parser import parse_nessus_xml
from stable_baselines3 import PPO
import time


def run_example(nessus_file: str, train_steps: int = 10000):
    """
    Run a complete training example with masking sensor.

    Args:
        nessus_file: Path to Nessus XML file
        train_steps: Number of training steps
    """
    print("="*70)
    print("Complete PPO + Masking Sensor Training Example")
    print("="*70)

    # Step 1: Load vulnerabilities
    print("\n[Step 1] Loading vulnerability data...")
    findings = parse_nessus_xml(nessus_file)
    print(f"  Loaded {len(findings)} findings from {nessus_file}")

    # Step 2: Create masked environment
    print("\n[Step 2] Creating masked environment with sensor...")
    config = AUVAPConfig(
        max_steps=50,
        reward_success=10.0,
        reward_failure=-1.0,
        use_risk_score_reward=True
    )

    env = MaskedCyberBattleEnv(
        vulnerability_findings=findings,
        config=config,
        max_attempts_per_task=3,
        enable_safety_constraints=True
    )
    print(f"  Environment created")
    print(f"  Task queue size: {len(env.masking_sensor.task_queue)}")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")

    # Step 3: Test environment
    print("\n[Step 3] Testing environment with random actions...")
    obs, info = env.reset()
    print(f"  Initial observation shape: {obs.shape}")
    print(f"  Task: {info.get('task_id', 'N/A')}")
    print(f"  Target: {info.get('target', 'N/A')}")

    # Run a few steps with random actions
    for i in range(5):
        # Get valid action
        allowed_actions = env.current_exposure.allowed_actions if env.current_exposure else []
        if not allowed_actions:
            obs, info = env.reset()
            allowed_actions = env.current_exposure.allowed_actions if env.current_exposure else []

        if allowed_actions:
            action = allowed_actions[0]  # Take first allowed action
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"  Step {i+1}: action={action}, reward={reward:.2f}, "
                  f"success={info.get('action_success', False)}")

            if terminated or truncated:
                obs, info = env.reset()

    # Step 4: Create and train PPO agent
    print(f"\n[Step 4] Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        verbose=1,
        seed=42
    )
    print(f"  PPO agent created")
    print(f"  Policy network parameters: {sum(p.numel() for p in model.policy.parameters())}")

    # Step 5: Train the agent
    print(f"\n[Step 5] Training PPO agent for {train_steps} steps...")
    start_time = time.time()

    model.learn(
        total_timesteps=train_steps,
        progress_bar=True,
        log_interval=10
    )

    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.1f} seconds")

    # Step 6: Evaluate trained agent
    print("\n[Step 6] Evaluating trained agent...")
    eval_episodes = 5
    episode_rewards = []

    for ep in range(eval_episodes):
        obs, info = env.reset()
        if info.get('terminal'):
            break

        episode_reward = 0
        done = False
        steps = 0

        while not done and steps < 50:
            # Use trained policy
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        print(f"  Episode {ep+1}: reward={episode_reward:.2f}, steps={steps}")

    mean_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
    print(f"  Mean reward: {mean_reward:.2f}")

    # Step 7: Show final statistics
    print("\n[Step 7] Final Statistics")
    print("-"*70)

    stats = env.get_sensor_statistics()
    print(f"  Task Statistics:")
    print(f"    Total tasks:        {stats['total_tasks']}")
    print(f"    Completed:          {stats['completed']}")
    print(f"    Failed:             {stats['failed']}")
    print(f"    Success rate:       {stats['success_rate']*100:.1f}%")

    print(f"\n  Network State:")
    print(f"    Owned nodes:        {stats['owned_nodes']}")
    print(f"    Credentials found:  {stats['credentials_found']}")

    print(f"\n  Execution Metrics:")
    print(f"    Total attempts:     {stats['total_attempts']}")
    print(f"    Avg attempts/task:  {stats['avg_attempts']:.2f}")

    # Step 8: Save results
    print("\n[Step 8] Saving results...")
    model.save("models/example_ppo_masked_model")
    print(f"  Model saved to: models/example_ppo_masked_model.zip")

    env.save_execution_log("logs/example_execution_log.json")
    print(f"  Execution log saved to: logs/example_execution_log.json")

    env.close()

    print("\n" + "="*70)
    print("Example Complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  ✓ Masking sensor controlled task exposure")
    print("  ✓ PPO agent learned from vulnerability data")
    print("  ✓ Safety constraints enforced throughout")
    print("  ✓ Complete execution log for replay")
    print(f"  ✓ Success rate: {stats['success_rate']*100:.1f}%")
    print("\nNext Steps:")
    print("  - Train longer for better performance (use train_ppo_masked.py)")
    print("  - Try different Nessus scan files")
    print("  - Tune PPO hyperparameters")
    print("  - Add policy engine for organizational constraints")
    print("="*70)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Complete training example")
    parser.add_argument("--nessus-file", type=str,
                       default="auvap_nessus_25_findings.xml",
                       help="Path to Nessus XML file")
    parser.add_argument("--train-steps", type=int, default=10000,
                       help="Number of training steps")

    args = parser.parse_args()

    if not os.path.exists(args.nessus_file):
        print(f"Error: File not found: {args.nessus_file}")
        print("\nAvailable Nessus files:")
        for f in os.listdir("."):
            if f.endswith(".xml") and "nessus" in f.lower():
                print(f"  - {f}")
        return 1

    try:
        run_example(args.nessus_file, args.train_steps)
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
