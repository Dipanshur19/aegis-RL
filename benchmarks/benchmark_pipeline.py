#!/usr/bin/env python3
"""
benchmark_pipeline.py - Performance Benchmarks for AUVAP Pipeline

Benchmarks the full AUVAP pipeline including:
- Parsing performance (XML/CSV)
- Classification throughput
- Policy filtering performance
- Task initialization
- End-to-end pipeline execution time
"""

import time
import sys
import statistics
from pathlib import Path
from typing import List, Dict
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import parser
import task_manager
from policy_engine import PolicyEngine, create_default_policy_rules


class PipelineBenchmark:
    """Benchmark suite for AUVAP pipeline."""

    def __init__(self, num_iterations: int = 10):
        """
        Initialize benchmark suite.

        Args:
            num_iterations: Number of iterations for each benchmark
        """
        self.num_iterations = num_iterations
        self.results = {}

    def benchmark_parser(self, sample_file: str, file_format: str = "xml") -> Dict:
        """
        Benchmark report parsing performance.

        Args:
            sample_file: Path to sample report file
            file_format: Format of the report ("xml" or "csv")

        Returns:
            Dict with timing statistics
        """
        print(f"\n[Benchmark] Parsing {file_format.upper()} Report")
        print("=" * 60)

        timings = []

        for i in range(self.num_iterations):
            start_time = time.perf_counter()

            try:
                findings = parser.parse_report(sample_file)
                findings_list = parser.to_dict_list(findings)
                elapsed = time.perf_counter() - start_time
                timings.append(elapsed)

                if i == 0:
                    print(f"  Findings parsed: {len(findings_list)}")

            except Exception as e:
                print(f"  Error during iteration {i+1}: {e}")
                continue

        if timings:
            results = {
                'mean': statistics.mean(timings),
                'median': statistics.median(timings),
                'stdev': statistics.stdev(timings) if len(timings) > 1 else 0,
                'min': min(timings),
                'max': max(timings),
                'iterations': len(timings)
            }

            print(f"  Mean time: {results['mean']:.4f}s")
            print(f"  Median time: {results['median']:.4f}s")
            print(f"  Std dev: {results['stdev']:.4f}s")
            print(f"  Min/Max: {results['min']:.4f}s / {results['max']:.4f}s")

            return results
        else:
            return {}

    def benchmark_policy_filtering(self, findings: List[Dict]) -> Dict:
        """
        Benchmark policy filtering performance.

        Args:
            findings: List of findings to filter

        Returns:
            Dict with timing statistics
        """
        print(f"\n[Benchmark] Policy Filtering")
        print("=" * 60)

        # Initialize policy engine
        engine = PolicyEngine()
        engine.add_rules(create_default_policy_rules())

        timings = []

        for i in range(self.num_iterations):
            start_time = time.perf_counter()

            # Apply policy to all findings
            for finding in findings:
                engine.evaluate(finding)

            elapsed = time.perf_counter() - start_time
            timings.append(elapsed)

        results = {
            'mean': statistics.mean(timings),
            'median': statistics.median(timings),
            'stdev': statistics.stdev(timings) if len(timings) > 1 else 0,
            'min': min(timings),
            'max': max(timings),
            'findings_per_second': len(findings) / statistics.mean(timings)
        }

        print(f"  Findings processed: {len(findings)}")
        print(f"  Mean time: {results['mean']:.4f}s")
        print(f"  Throughput: {results['findings_per_second']:.0f} findings/s")
        print(f"  Std dev: {results['stdev']:.4f}s")

        return results

    def benchmark_task_initialization(self, findings: List[Dict]) -> Dict:
        """
        Benchmark task initialization performance.

        Args:
            findings: List of findings to create tasks from

        Returns:
            Dict with timing statistics
        """
        print(f"\n[Benchmark] Task Initialization")
        print("=" * 60)

        timings = []

        for i in range(self.num_iterations):
            start_time = time.perf_counter()

            tasks = task_manager.initialize_tasks(findings)

            elapsed = time.perf_counter() - start_time
            timings.append(elapsed)

            if i == 0:
                print(f"  Tasks created: {len(tasks)}")

        results = {
            'mean': statistics.mean(timings),
            'median': statistics.median(timings),
            'stdev': statistics.stdev(timings) if len(timings) > 1 else 0,
            'min': min(timings),
            'max': max(timings),
            'tasks_per_second': len(findings) / statistics.mean(timings)
        }

        print(f"  Mean time: {results['mean']:.4f}s")
        print(f"  Throughput: {results['tasks_per_second']:.0f} tasks/s")
        print(f"  Median time: {results['median']:.4f}s")

        return results

    def benchmark_risk_scoring(self, num_findings: int = 1000) -> Dict:
        """
        Benchmark risk score computation.

        Args:
            num_findings: Number of findings to compute risk for

        Returns:
            Dict with timing statistics
        """
        print(f"\n[Benchmark] Risk Score Computation")
        print("=" * 60)

        # Generate test findings
        test_findings = []
        for i in range(num_findings):
            finding = {
                'cvss': 5.0 + (i % 5),
                'attack_vector': ['Network', 'Adjacent', 'Local'][i % 3],
                'automation_candidate': i % 2 == 0
            }
            test_findings.append(finding)

        timings = []

        for i in range(self.num_iterations):
            start_time = time.perf_counter()

            for finding in test_findings:
                task_manager.compute_risk_score(finding)

            elapsed = time.perf_counter() - start_time
            timings.append(elapsed)

        results = {
            'mean': statistics.mean(timings),
            'median': statistics.median(timings),
            'stdev': statistics.stdev(timings) if len(timings) > 1 else 0,
            'scores_per_second': num_findings / statistics.mean(timings)
        }

        print(f"  Findings processed: {num_findings}")
        print(f"  Mean time: {results['mean']:.4f}s")
        print(f"  Throughput: {results['scores_per_second']:.0f} scores/s")

        return results

    def run_all_benchmarks(self, sample_file: str = None) -> Dict:
        """
        Run all pipeline benchmarks.

        Args:
            sample_file: Optional path to sample report file

        Returns:
            Dict with all benchmark results
        """
        print("\n" + "=" * 60)
        print("AEGIS-RL PIPELINE PERFORMANCE BENCHMARKS")
        print("=" * 60)
        print(f"Iterations per benchmark: {self.num_iterations}")

        all_results = {}

        # Benchmark risk scoring (no file needed)
        all_results['risk_scoring'] = self.benchmark_risk_scoring()

        # Generate synthetic findings for other benchmarks
        synthetic_findings = []
        for i in range(100):
            finding = {
                'finding_id': f"bench_{i}",
                'host_ip': f"192.168.1.{i}",
                'port': 22 + (i % 100),
                'service': ['ssh', 'http', 'ftp'][i % 3],
                'cvss': 5.0 + (i % 5),
                'severity_bucket': ['Medium', 'High', 'Critical'][i % 3],
                'attack_vector': 'Network',
                'automation_candidate': i % 2 == 0,
                'title': f"Vulnerability {i}",
                'cve': f"CVE-2023-{1000+i}"
            }
            synthetic_findings.append(finding)

        # Benchmark policy filtering
        all_results['policy_filtering'] = self.benchmark_policy_filtering(
            synthetic_findings
        )

        # Benchmark task initialization
        all_results['task_initialization'] = self.benchmark_task_initialization(
            synthetic_findings
        )

        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        for name, results in all_results.items():
            if results:
                print(f"\n{name}:")
                print(f"  Mean: {results.get('mean', 0):.4f}s")
                if 'per_second' in str(results):
                    for key, value in results.items():
                        if 'per_second' in key:
                            print(f"  {key.replace('_', ' ').title()}: {value:.0f}")

        return all_results

    def save_results(self, results: Dict, output_file: str = "benchmark_results.json"):
        """Save benchmark results to JSON file."""
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Results saved to: {output_path}")


def main():
    """Run pipeline benchmarks."""
    benchmark = PipelineBenchmark(num_iterations=10)

    # Run all benchmarks
    results = benchmark.run_all_benchmarks()

    # Save results
    benchmark.save_results(results)

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
