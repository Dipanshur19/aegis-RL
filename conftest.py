"""
Pytest configuration for AUVAP-PPO test suite.

Adds the project root and all subpackage directories to sys.path
so that tests can use the same import patterns as the source modules.
"""

import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ppo"))
sys.path.insert(0, str(ROOT / "execution"))
sys.path.insert(0, str(ROOT / "environment"))
sys.path.insert(0, str(ROOT / "training"))
