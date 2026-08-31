"""
Quick test script to verify PPO-CyberBattleSim integration setup
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")

    try:
        import gymnasium as gym
        print("  ✓ gymnasium")
    except ImportError as e:
        print(f"  ✗ gymnasium: {e}")
        return False

    try:
        import stable_baselines3
        print("  ✓ stable_baselines3")
    except ImportError as e:
        print(f"  ✗ stable_baselines3: {e}")
        return False

    try:
        import torch
        print("  ✓ torch")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False

    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False

    # CyberBattleSim is optional
    try:
        import cyberbattle
        print("  ✓ cyberbattle (optional)")
    except ImportError:
        print("  ⚠ cyberbattle (not installed - will use mock environment)")

    return True


def test_auvap_modules():
    """Test that AUVAP modules can be imported"""
    print("\nTesting AUVAP modules...")

    try:
        from parser import VAFinding, parse_nessus_xml
        print("  ✓ parser")
    except ImportError as e:
        print(f"  ✗ parser: {e}")
        return False

    try:
        from task_manager import ExploitTask, TaskState
        print("  ✓ task_manager")
    except ImportError as e:
        print(f"  ✗ task_manager: {e}")
        return False

    try:
        from feasibility_filter import compute_risk_score
        print("  ✓ feasibility_filter")
    except ImportError as e:
        print(f"  ✗ feasibility_filter: {e}")
        return False

    return True


def test_ppo_modules():
    """Test that PPO integration modules can be imported"""
    print("\nTesting PPO integration modules...")

    try:
        from environment.cyberbattle_wrapper import AUVAPCyberBattleEnv, AUVAPConfig
        print("  ✓ environment.cyberbattle_wrapper")
    except ImportError as e:
        print(f"  ✗ environment.cyberbattle_wrapper: {e}")
        return False

    try:
        from environment.action_mapper import ActionMapper
        print("  ✓ env.action_mapper")
    except ImportError as e:
        print(f"  ✗ env.action_mapper: {e}")
        return False

    try:
        from environment.observation_builder import ObservationBuilder
        print("  ✓ env.observation_builder")
    except ImportError as e:
        print(f"  ✗ env.observation_builder: {e}")
        return False

    try:
        from environment.reward_shaper import RewardShaper
        print("  ✓ env.reward_shaper")
    except ImportError as e:
        print(f"  ✗ env.reward_shaper: {e}")
        return False

    return True


def test_environment():
    """Test creating and using the environment"""
    print("\nTesting environment creation...")

    try:
        from environment.cyberbattle_wrapper import AUVAPCyberBattleEnv, AUVAPConfig

        config = AUVAPConfig(max_steps=10)
        env = AUVAPCyberBattleEnv(config=config)

        print("  ✓ Environment created")

        # Test reset
        obs, info = env.reset()
        print(f"  ✓ Environment reset (obs shape: {obs.shape})")

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  ✓ Environment step (reward: {reward:.2f})")

        env.close()
        print("  ✓ Environment closed")

        return True

    except Exception as e:
        print(f"  ✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_mapper():
    """Test action mapper functionality"""
    print("\nTesting action mapper...")

    try:
        from environment.action_mapper import ActionMapper
        from task_manager import ExploitTask, TaskState

        mapper = ActionMapper(max_actions=10)

        # Create a test task
        task = ExploitTask(
            task_id="test_1",
            vulnerability_id="TEST-001",
            target_host="192.168.1.1",
            target_port=80,
            protocol="tcp",
            cvss_score=7.5,
            state=TaskState.PLANNED,
            priority=7.5,
            description="Test exploit"
        )

        action_id = mapper.register_task(task)
        print(f"  ✓ Task registered with action_id: {action_id}")

        retrieved_task = mapper.get_task_from_action(action_id)
        assert retrieved_task.task_id == task.task_id
        print("  ✓ Task retrieval works")

        stats = mapper.get_statistics()
        print(f"  ✓ Mapper statistics: {stats}")

        return True

    except Exception as e:
        print(f"  ✗ Action mapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppo_creation():
    """Test creating a PPO model"""
    print("\nTesting PPO model creation...")

    try:
        from stable_baselines3 import PPO
        from environment.cyberbattle_wrapper import AUVAPCyberBattleEnv, AUVAPConfig

        config = AUVAPConfig(max_steps=10)
        env = AUVAPCyberBattleEnv(config=config)

        model = PPO("MlpPolicy", env, verbose=0)
        print("  ✓ PPO model created")

        # Test prediction
        obs, _ = env.reset()
        action, _states = model.predict(obs, deterministic=True)
        print(f"  ✓ Model prediction works (action: {action})")

        env.close()
        return True

    except Exception as e:
        print(f"  ✗ PPO model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("AUVAP-PPO Integration Setup Test")
    print("="*60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("AUVAP Modules", test_auvap_modules()))
    results.append(("PPO Modules", test_ppo_modules()))
    results.append(("Environment", test_environment()))
    results.append(("Action Mapper", test_action_mapper()))
    results.append(("PPO Model", test_ppo_creation()))

    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20} {status}")

    print("="*60)

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed! Setup is complete.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
