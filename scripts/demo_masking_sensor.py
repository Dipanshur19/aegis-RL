"""
Demonstration of the Masking Sensor Algorithm

This script shows how the masking sensor controls what the DRL agent can see and do.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import parse_nessus_xml
from environment.action_mapper import ActionMapper
from environment.masking_sensor import MaskingSensor
import time


def demo_masking_sensor(nessus_file: str):
    """
    Demonstrate the masking sensor algorithm.
    """
    print("="*70)
    print("Masking Sensor Algorithm Demonstration")
    print("="*70)

    # 1. Load vulnerability findings
    print(f"\n[1] Loading vulnerabilities from: {nessus_file}")
    findings = parse_nessus_xml(nessus_file)
    print(f"    Loaded {len(findings)} vulnerability findings")

    # 2. Create action mapper
    print("\n[2] Creating action mapper...")
    action_mapper = ActionMapper(max_actions=100)
    print(f"    Action mapper initialized with max_actions=100")

    # 3. Create masking sensor
    print("\n[3] Creating masking sensor...")
    sensor = MaskingSensor(
        findings=findings,
        action_mapper=action_mapper,
        max_attempts_per_task=3,
        enable_safety_constraints=True,
        log_file="logs/demo_masking_sensor.jsonl"
    )
    print(f"    Masking sensor initialized")
    print(f"    Task queue size: {len(sensor.task_queue)}")

    # 4. Simulate DRL agent execution
    print("\n[4] Simulating DRL agent execution...")
    print("-"*70)

    episode = 0
    max_episodes = 10

    while not sensor.is_complete() and episode < max_episodes:
        episode += 1

        # Get current task exposure
        exposure = sensor.get_current_task()

        if exposure is None:
            print("\n    No more tasks available")
            break

        task = exposure.task
        print(f"\n[Episode {episode}] Current Task:")
        print(f"    Target: {task.target_host}:{task.target_port}")
        print(f"    Vulnerability: {task.vulnerability_id}")
        print(f"    CVSS Score: {task.cvss_score}")
        print(f"    Priority: {task.priority:.2f}")
        print(f"    Attempt: {exposure.attempt_number + 1}/{exposure.max_attempts}")

        # Show allowed actions
        print(f"\n    Allowed Actions: {exposure.allowed_actions}")
        if not exposure.allowed_actions:
            print("    ⚠️  No actions allowed (blocked by policy or inaccessible)")

            # Advance with failure
            sensor.advance({
                'success': False,
                'action': -1,
                'duration': 0.0,
                'error': 'No allowed actions'
            })
            continue

        # Show safety constraints
        print(f"\n    Safety Constraints:")
        for constraint in exposure.safety_constraints:
            print(f"      - {constraint.constraint_type}: {constraint.value}")
            print(f"        Reason: {constraint.reason}")

        # Show context
        print(f"\n    Context:")
        print(f"      - Owned nodes: {len(exposure.context['owned_nodes'])}")
        print(f"      - Available credentials: {len(exposure.context['available_credentials'])}")
        print(f"      - Previous attempts: {exposure.context['previous_attempts']}")

        # Simulate exploitation attempt
        print(f"\n    Simulating exploitation...")
        time.sleep(0.1)  # Simulate work

        # Random success/failure (60% success rate)
        import random
        success = random.random() < 0.6

        # Simulate execution result
        result = {
            'success': success,
            'action': exposure.allowed_actions[0] if exposure.allowed_actions else -1,
            'duration': random.uniform(1.0, 5.0),
            'safety_violations': []
        }

        if success:
            print(f"    ✓ Exploitation SUCCESSFUL")
            result['artifacts'] = {
                'credentials': [f'user{random.randint(1,10)}:password{random.randint(1,100)}']
            }
        else:
            print(f"    ✗ Exploitation FAILED")
            result['error'] = "Exploit did not execute successfully"

        # Advance to next task
        has_more = sensor.advance(result)

        if not has_more:
            print("\n    ℹ️  Task queue exhausted")
            break

    # 5. Show final statistics
    print("\n" + "="*70)
    print("Execution Complete - Final Statistics")
    print("="*70)

    stats = sensor.get_statistics()
    print(f"\nTask Statistics:")
    print(f"  Total tasks:        {stats['total_tasks']}")
    print(f"  Completed:          {stats['completed']}")
    print(f"  Failed:             {stats['failed']}")
    print(f"  Pending:            {stats['pending']}")
    print(f"  Blocked:            {stats['blocked']}")
    print(f"  Success rate:       {stats['success_rate']*100:.1f}%")

    print(f"\nNetwork State:")
    print(f"  Owned nodes:        {stats['owned_nodes']}")
    print(f"  Credentials found:  {stats['credentials_found']}")

    print(f"\nExecution Metrics:")
    print(f"  Total attempts:     {stats['total_attempts']}")
    print(f"  Avg attempts/task:  {stats['avg_attempts']:.2f}")

    # 6. Show execution log
    print("\n" + "="*70)
    print("Execution Log Summary")
    print("="*70)

    for i, entry in enumerate(sensor.execution_log[:5], 1):  # Show first 5
        print(f"\n[{i}] {entry.timestamp}")
        print(f"    Task: {entry.task_id}")
        print(f"    Target: {entry.metadata.get('target', 'unknown')}")
        print(f"    Result: {entry.result}")
        print(f"    Duration: {entry.duration_seconds:.2f}s")
        if entry.safety_violations:
            print(f"    ⚠️  Safety violations: {entry.safety_violations}")

    if len(sensor.execution_log) > 5:
        print(f"\n... and {len(sensor.execution_log) - 5} more entries")

    # Save complete log
    log_output = "logs/demo_execution_log.json"
    sensor.save_execution_log(log_output)
    print(f"\n✓ Complete execution log saved to: {log_output}")

    print("\n" + "="*70)
    print("Demonstration Complete")
    print("="*70)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Demonstrate masking sensor algorithm")
    parser.add_argument("--nessus-file", type=str,
                       default="auvap_nessus_25_findings.xml",
                       help="Path to Nessus XML file")

    args = parser.parse_args()

    if not os.path.exists(args.nessus_file):
        print(f"Error: File not found: {args.nessus_file}")
        print("\nAvailable Nessus files:")
        for f in os.listdir("."):
            if f.endswith(".xml") and "nessus" in f.lower():
                print(f"  - {f}")
        return 1

    try:
        demo_masking_sensor(args.nessus_file)
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
