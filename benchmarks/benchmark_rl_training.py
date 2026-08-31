#!/usr/bin/env python3
"""
benchmark_rl_training.py - Performance Benchmarks for PPO Training

Benchmarks the RL training components including:
- PPO agent forward pass
- Action selection
- GAE computation
- Policy update
- Training loop iteration
- Memory efficiency
"""

import time
import sys
import statistics
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
import psutil
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ppo"))

from ppo_agent import PPOAgent, PolicyNetwork, Trajectory


class RLTrainingBenchmark:
    """Benchmark suite for PPO training."""

    def __init__(self, num_iterations: int = 100):
        """
        Initialize benchmark suite.

        Args:
            num_iterations: Number of iterations for each benchmark
        """
        self.num_iterations = num_iterations
        self.results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def benchmark_forward_pass(self, state_dim: int = 128, action_dim: int = 50,
                             batch_size: int = 32) -> Dict:
        """
        Benchmark PolicyNetwork forward pass.

        Args:
            state_dim: State/observation dimension
            action_dim: Action dimension
            batch_size: Batch size

        Returns:
            Dict with timing statistics
        """
        print(f"\n[Benchmark] PolicyNetwork Forward Pass")
        print("=" * 60)

        network = PolicyNetwork(state_dim, action_dim).to(self.device)
        network.eval()

        timings = []

        with torch.no_grad():
            for _ in range(self.num_iterations):
                obs = torch.randn(batch_size, state_dim).to(self.device)

                start_time = time.perf_counter()
                logits, value = network(obs)
                elapsed = time.perf_counter() - start_time

                timings.append(elapsed)

        results = {
            'mean': statistics.mean(timings),
            'median': statistics.median(timings),
            'stdev': statistics.stdev(timings) if len(timings) > 1 else 0,
            'min': min(timings),
            'max': max(timings),
            'throughput': batch_size / statistics.mean(timings)
        }

        print(f"  Observation dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {self.device}")
        print(f"  Mean time: {results['mean']*1000:.2f}ms")
        print(f"  Throughput: {results['throughput']:.0f} samples/s")

        return results

    def benchmark_action_selection(self, state_dim: int = 128,
                                  action_dim: int = 50) -> Dict:
        """
        Benchmark action selection.

        Args:
            state_dim: State/observation dimension
            action_dim: Action dimension

        Returns:
            Dict with timing statistics
        """
        print(f"\n[Benchmark] Action Selection")
        print("=" * 60)

        agent = PPOAgent(state_dim, action_dim, device=str(self.device))
        timings = []

        for _ in range(self.num_iterations):
            obs = np.random.randn(state_dim).astype(np.float32)

            start_time = time.perf_counter()
            action, log_prob, value = agent.select_action(obs)
            elapsed = time.perf_counter() - start_time

            timings.append(elapsed)

        results = {
            'mean': statistics.mean(timings),
            'median': statistics.median(timings),
            'stdev': statistics.stdev(timings) if len(timings) > 1 else 0,
            'actions_per_second': 1.0 / statistics.mean(timings)
        }

        print(f"  Mean time: {results['mean']*1000:.2f}ms")
        print(f"  Actions/s: {results['actions_per_second']:.0f}")

        return results

    def benchmark_gae_computation(self, episode_length: int = 100) -> Dict:
        """
        Benchmark GAE computation.

        Args:
            episode_length: Length of episode

        Returns:
            Dict with timing statistics
        """
        print(f"\n[Benchmark] GAE Computation")
        print("=" * 60)

        agent = PPOAgent(state_dim=128, action_dim=50, device=str(self.device))

        traj = Trajectory(
            states=[np.random.randn(128).astype(np.float32) for _ in range(episode_length)],
            actions=[np.random.randint(0, 50) for _ in range(episode_length)],
            rewards=[float(np.random.randn()) for _ in range(episode_length)],
            dones=[False] * (episode_length - 1) + [True],
            log_probs=[float(np.random.randn()) for _ in range(episode_length)],
            values=[float(np.random.randn()) for _ in range(episode_length)]
        )

        timings = []

        for _ in range(self.num_iterations):
            start_time = time.perf_counter()
            advantages, returns = agent.compute_gae(traj)
            elapsed = time.perf_counter() - start_time

            timings.append(elapsed)

        results = {
            'mean': statistics.mean(timings),
            'median': statistics.median(timings),
            'stdev': statistics.stdev(timings) if len(timings) > 1 else 0,
            'episode_length': episode_length
        }

        print(f"  Episode length: {episode_length}")
        print(f"  Mean time: {results['mean']*1000:.2f}ms")
        print(f"  Median time: {results['median']*1000:.2f}ms")

        return results

    def benchmark_policy_update(self, batch_size: int = 64) -> Dict:
        """
        Benchmark policy update step.

        Args:
            batch_size: Batch size for update

        Returns:
            Dict with timing statistics
        """
        print(f"\n[Benchmark] Policy Update")
        print("=" * 60)

        agent = PPOAgent(state_dim=128, action_dim=50, device=str(self.device))

        traj = Trajectory(
            states=[np.random.randn(128).astype(np.float32) for _ in range(batch_size)],
            actions=[np.random.randint(0, 50) for _ in range(batch_size)],
            rewards=[float(np.random.randn()) for _ in range(batch_size)],
            dones=[False] * (batch_size - 1) + [True],
            log_probs=[float(np.random.randn()) for _ in range(batch_size)],
            values=[float(np.random.randn()) for _ in range(batch_size)]
        )

        timings = []

        for _ in range(self.num_iterations):
            start_time = time.perf_counter()
            loss_dict = agent.update([traj], n_epochs=1, batch_size=batch_size)
            elapsed = time.perf_counter() - start_time
            timings.append(elapsed)

        results = {
            'mean': statistics.mean(timings),
            'median': statistics.median(timings),
            'stdev': statistics.stdev(timings) if len(timings) > 1 else 0,
            'updates_per_second': 1.0 / statistics.mean(timings)
        }

        print(f"  Batch size: {batch_size}")
        print(f"  Mean time: {results['mean']*1000:.2f}ms")
        print(f"  Updates/s: {results['updates_per_second']:.1f}")

        return results

    def benchmark_memory_usage(self, state_dim: int = 128, action_dim: int = 50) -> Dict:
        """
        Benchmark memory usage of PPO agent.

        Args:
            state_dim: State/observation dimension
            action_dim: Action dimension

        Returns:
            Dict with memory statistics
        """
        print(f"\n[Benchmark] Memory Usage")
        print("=" * 60)

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Create agent
        agent = PPOAgent(state_dim, action_dim, device=str(self.device))

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        agent_memory = mem_after - mem_before

        # Test with trajectory storage
        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []

        # Simulate 1000 steps
        for _ in range(1000):
            obs = np.random.randn(state_dim).astype(np.float32)
            action, log_prob, value = agent.select_action(obs)

            states.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(0.0)
            dones.append(False)

        traj = Trajectory(states, actions, rewards, dones, log_probs, values)

        mem_with_trajectory = process.memory_info().rss / 1024 / 1024  # MB
        trajectory_memory = mem_with_trajectory - mem_after

        results = {
            'agent_memory_mb': agent_memory,
            'trajectory_memory_mb': trajectory_memory,
            'total_memory_mb': mem_with_trajectory,
            'trajectory_steps': 1000
        }

        print(f"  Agent memory: {results['agent_memory_mb']:.2f} MB")
        print(f"  Trajectory memory (1000 steps): {results['trajectory_memory_mb']:.2f} MB")
        print(f"  Memory per step: {trajectory_memory/1000:.4f} MB")

        return results

    def benchmark_training_iteration(self, episode_length: int = 200,
                                    batch_size: int = 32) -> Dict:
        """
        Benchmark full training iteration.

        Args:
            episode_length: Length of episode
            batch_size: Batch size for update

        Returns:
            Dict with timing statistics
        """
        print(f"\n[Benchmark] Full Training Iteration")
        print("=" * 60)

        agent = PPOAgent(state_dim=128, action_dim=50, device=str(self.device))

        timings = {
            'rollout': [],
            'gae': [],
            'update': [],
            'total': []
        }

        for _ in range(10):  # Fewer iterations for full training
            # Rollout phase
            states = []
            actions = []
            log_probs = []
            values = []
            rewards = []
            dones = []

            rollout_start = time.perf_counter()
            for _ in range(episode_length):
                obs = np.random.randn(128).astype(np.float32)
                action, log_prob, value = agent.select_action(obs)

                states.append(obs)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(float(np.random.randn()))
                dones.append(False)

            dones[-1] = True
            rollout_time = time.perf_counter() - rollout_start

            traj = Trajectory(states, actions, rewards, dones, log_probs, values)

            # GAE phase
            gae_start = time.perf_counter()
            advantages, returns = agent.compute_gae(traj)
            gae_time = time.perf_counter() - gae_start

            # Update phase
            update_start = time.perf_counter()
            agent.update([traj], n_epochs=1, batch_size=batch_size)
            update_time = time.perf_counter() - update_start

            total_time = rollout_time + gae_time + update_time

            timings['rollout'].append(rollout_time)
            timings['gae'].append(gae_time)
            timings['update'].append(update_time)
            timings['total'].append(total_time)

        results = {
            'rollout_mean': statistics.mean(timings['rollout']),
            'gae_mean': statistics.mean(timings['gae']),
            'update_mean': statistics.mean(timings['update']),
            'total_mean': statistics.mean(timings['total']),
            'episode_length': episode_length,
            'iterations_per_minute': 60.0 / statistics.mean(timings['total'])
        }

        print(f"  Episode length: {episode_length}")
        print(f"  Rollout time: {results['rollout_mean']:.4f}s")
        print(f"  GAE time: {results['gae_mean']:.4f}s")
        print(f"  Update time: {results['update_mean']:.4f}s")
        print(f"  Total time: {results['total_mean']:.4f}s")
        print(f"  Iterations/minute: {results['iterations_per_minute']:.1f}")

        return results

    def run_all_benchmarks(self) -> Dict:
        """Run all RL training benchmarks."""
        print("\n" + "=" * 60)
        print("PPO TRAINING PERFORMANCE BENCHMARKS")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")

        all_results = {}

        all_results['forward_pass'] = self.benchmark_forward_pass()
        all_results['action_selection'] = self.benchmark_action_selection()
        all_results['gae_computation'] = self.benchmark_gae_computation()
        all_results['policy_update'] = self.benchmark_policy_update()
        all_results['memory_usage'] = self.benchmark_memory_usage()
        all_results['training_iteration'] = self.benchmark_training_iteration()

        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        print(f"\nForward pass: {all_results['forward_pass']['mean']*1000:.2f}ms")
        print(f"Action selection: {all_results['action_selection']['actions_per_second']:.0f} actions/s")
        print(f"Policy update: {all_results['policy_update']['updates_per_second']:.1f} updates/s")
        print(f"Training iteration: {all_results['training_iteration']['iterations_per_minute']:.1f} iter/min")
        print(f"Memory usage: {all_results['memory_usage']['agent_memory_mb']:.2f} MB")

        return all_results

    def save_results(self, results: Dict, output_file: str = "rl_benchmark_results.json"):
        """Save benchmark results to JSON file."""
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Results saved to: {output_path}")


def main():
    """Run RL training benchmarks."""
    benchmark = RLTrainingBenchmark(num_iterations=100)

    # Run all benchmarks
    results = benchmark.run_all_benchmarks()

    # Save results
    benchmark.save_results(results)

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
