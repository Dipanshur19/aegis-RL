#!/usr/bin/env python3
"""
policy_loader.py - YAML Policy Configuration Loader

Loads policy rules from YAML configuration files for the AUVAP pipeline.

Supported Operators:
  Membership: in, not_in
  Equality: eq, ne
  String: contains, not_contains, starts_with, ends_with
  Regex: regex, not_regex
  Numeric: gt, gte, lt, lte, range
  Existence: exists, not_exists
  Emptiness: is_empty, not_empty

Usage:
    from policy_loader import load_policies_from_yaml

    rules = load_policies_from_yaml("policy_config.yaml")
    policy_engine.add_rules(rules)
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Callable

try:
    import yaml
except ImportError:
    print("[ERROR] PyYAML not installed. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

from policy_engine import PolicyRule


def create_predicate(condition: Dict[str, Any]) -> Callable[[dict], bool]:
    """
    Create a predicate function from a condition specification.

    Args:
        condition: Dictionary with 'field', 'operator', and 'value' keys

    Returns:
        Callable that takes a finding dict and returns bool

    Supported operators:
        - in: Field value is in list
        - not_in: Field value is not in list (NEW)
        - eq: Field value equals value
        - ne: Field value not equals value
        - contains: Value is substring of field (case-insensitive)
        - not_contains: Value is not substring of field (NEW)
        - starts_with: Field starts with value (case-insensitive) (NEW)
        - ends_with: Field ends with value (case-insensitive) (NEW)
        - regex: Field matches regular expression
        - not_regex: Field does not match regex (NEW)
        - gt: Field value greater than value
        - gte: Field value greater than or equal to value
        - lt: Field value less than value
        - lte: Field value less than or equal to value
        - range: Field value is within [min, max] range (NEW)
        - exists: Field exists and is not None (NEW)
        - not_exists: Field doesn't exist or is None (NEW)
        - is_empty: Field is empty string/list (NEW)
        - not_empty: Field has content (NEW)
    """
    field = condition['field']
    operator = condition['operator']
    value = condition.get('value')  # Some operators don't need value

    # Membership operators
    if operator == 'in':
        # Check if field value is in the list
        return lambda f: f.get(field) in value

    elif operator == 'not_in':
        # Check if field value is NOT in the list
        return lambda f: f.get(field) not in value

    # Equality operators
    elif operator == 'eq':
        # Check if field equals value
        return lambda f: f.get(field) == value

    elif operator == 'ne':
        # Check if field not equals value
        return lambda f: f.get(field) != value

    # String matching operators
    elif operator == 'contains':
        # Check if value is substring of field (case-insensitive)
        value_lower = str(value).lower()
        return lambda f: value_lower in str(f.get(field, '')).lower()

    elif operator == 'not_contains':
        # Check if value is NOT substring of field (case-insensitive)
        value_lower = str(value).lower()
        return lambda f: value_lower not in str(f.get(field, '')).lower()

    elif operator == 'starts_with':
        # Check if field starts with value (case-insensitive)
        value_lower = str(value).lower()
        return lambda f: str(f.get(field, '')).lower().startswith(value_lower)

    elif operator == 'ends_with':
        # Check if field ends with value (case-insensitive)
        value_lower = str(value).lower()
        return lambda f: str(f.get(field, '')).lower().endswith(value_lower)

    # Regex operators
    elif operator == 'regex':
        # Check if field matches regex pattern
        try:
            pattern = re.compile(value)
            return lambda f: bool(pattern.search(str(f.get(field, ''))))
        except re.error as e:
            print(f"[WARNING] Invalid regex pattern '{value}': {e}", file=sys.stderr)
            return lambda f: False

    elif operator == 'not_regex':
        # Check if field does NOT match regex pattern
        try:
            pattern = re.compile(value)
            return lambda f: not bool(pattern.search(str(f.get(field, ''))))
        except re.error as e:
            print(f"[WARNING] Invalid regex pattern '{value}': {e}", file=sys.stderr)
            return lambda f: True  # If regex invalid, match everything

    # Numeric comparison operators
    elif operator == 'gt':
        # Check if field value > value
        return lambda f: (f.get(field) or 0) > value

    elif operator == 'gte':
        # Check if field value >= value
        return lambda f: (f.get(field) or 0) >= value

    elif operator == 'lt':
        # Check if field value < value
        return lambda f: (f.get(field) or 0) < value

    elif operator == 'lte':
        # Check if field value <= value
        return lambda f: (f.get(field) or 0) <= value

    elif operator == 'range':
        # Check if field value is within [min, max] range (inclusive)
        # Value must be a list/tuple: [min, max]
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"'range' operator requires value as [min, max], got {value}")
        min_val, max_val = value
        return lambda f: min_val <= (f.get(field) or 0) <= max_val

    # Existence operators
    elif operator == 'exists':
        # Check if field exists and is not None
        return lambda f: field in f and f[field] is not None

    elif operator == 'not_exists':
        # Check if field doesn't exist or is None
        return lambda f: field not in f or f[field] is None

    # Emptiness operators
    elif operator == 'is_empty':
        # Check if field is empty string or empty list
        def is_empty_check(f):
            val = f.get(field)
            if val is None:
                return True
            if isinstance(val, str):
                return val.strip() == ''
            if isinstance(val, (list, tuple, dict)):
                return len(val) == 0
            return False
        return is_empty_check

    elif operator == 'not_empty':
        # Check if field has content (not empty)
        def not_empty_check(f):
            val = f.get(field)
            if val is None:
                return False
            if isinstance(val, str):
                return val.strip() != ''
            if isinstance(val, (list, tuple, dict)):
                return len(val) > 0
            return True
        return not_empty_check

    else:
        raise ValueError(f"Unknown operator: {operator}")


def parse_policy_rule(rule_dict: Dict[str, Any]) -> PolicyRule:
    """
    Convert YAML rule definition to PolicyRule object.
    
    Args:
        rule_dict: Dictionary from YAML with rule specification
        
    Returns:
        PolicyRule object
        
    Raises:
        KeyError: If required fields are missing
        ValueError: If operator is invalid
    """
    try:
        # Validate required fields
        required_fields = ['rule_id', 'type', 'condition', 'reason', 'precedence']
        for field in required_fields:
            if field not in rule_dict:
                raise KeyError(f"Missing required field: {field}")
        
        # Validate rule type
        valid_types = ['ignore', 'force_manual', 'prioritize']
        if rule_dict['type'] not in valid_types:
            raise ValueError(f"Invalid rule type: {rule_dict['type']}. Must be one of {valid_types}")
        
        # Validate condition structure
        condition = rule_dict['condition']
        if not isinstance(condition, dict):
            raise ValueError("Condition must be a dictionary")

        if 'field' not in condition or 'operator' not in condition:
            raise ValueError("Condition must have 'field' and 'operator' keys")

        # Some operators don't require 'value' (exists, not_exists, is_empty, not_empty)
        value_optional_operators = ['exists', 'not_exists', 'is_empty', 'not_empty']
        if condition['operator'] not in value_optional_operators and 'value' not in condition:
            raise ValueError(f"Operator '{condition['operator']}' requires a 'value' key")
        
        # Build predicate function
        predicate = create_predicate(condition)
        
        return PolicyRule(
            rule_id=rule_dict['rule_id'],
            type=rule_dict['type'],
            predicate=predicate,
            reason=rule_dict['reason'],
            precedence=rule_dict['precedence']
        )
    
    except Exception as e:
        print(f"[ERROR] Failed to parse rule {rule_dict.get('rule_id', 'unknown')}: {e}", 
              file=sys.stderr)
        raise


def load_policies_from_yaml(filepath: str) -> List[PolicyRule]:
    """
    Load policy rules from YAML configuration file.
    
    The YAML file should have three top-level keys:
        - user_policies: Precedence 0 (highest priority)
        - org_policies: Precedence 1 (organizational)
        - baseline_policies: Precedence 2 (framework defaults)
    
    Args:
        filepath: Path to YAML configuration file
        
    Returns:
        List of PolicyRule objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is invalid YAML
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Policy configuration file not found: {filepath}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"[ERROR] Invalid YAML in {filepath}: {e}", file=sys.stderr)
        raise
    
    if config is None:
        print(f"[WARNING] Empty configuration file: {filepath}", file=sys.stderr)
        return []
    
    rules = []
    
    # Load all policy levels
    policy_levels = ['user_policies', 'org_policies', 'baseline_policies']
    
    for level in policy_levels:
        if level not in config:
            continue
        
        level_rules = config[level]
        if not isinstance(level_rules, list):
            print(f"[WARNING] {level} is not a list, skipping", file=sys.stderr)
            continue
        
        for rule_dict in level_rules:
            try:
                rule = parse_policy_rule(rule_dict)
                rules.append(rule)
            except Exception as e:
                print(f"[WARNING] Skipping invalid rule in {level}: {e}", file=sys.stderr)
                continue
    
    return rules


def validate_policy_config(filepath: str) -> bool:
    """
    Validate a policy configuration file without loading rules.
    
    Args:
        filepath: Path to YAML configuration file
        
    Returns:
        True if valid, False otherwise (with error messages printed)
    """
    print(f"[*] Validating policy configuration: {filepath}")
    
    try:
        rules = load_policies_from_yaml(filepath)
        
        if not rules:
            print("[WARNING] No valid rules found in configuration", file=sys.stderr)
            return False
        
        print(f"[+] Successfully loaded {len(rules)} rules")
        
        # Check for duplicate rule IDs
        rule_ids = [r.rule_id for r in rules]
        duplicates = set([rid for rid in rule_ids if rule_ids.count(rid) > 1])
        
        if duplicates:
            print(f"[WARNING] Duplicate rule IDs found: {duplicates}", file=sys.stderr)
            return False
        
        # Print summary by type and precedence
        by_type: Dict[str, int] = {}
        by_precedence: Dict[int, int] = {}
        
        for rule in rules:
            by_type[rule.type] = by_type.get(rule.type, 0) + 1
            by_precedence[rule.precedence] = by_precedence.get(rule.precedence, 0) + 1
        
        print(f"\nRule breakdown:")
        print(f"  By type: {by_type}")
        print(f"  By precedence: {by_precedence}")
        
        print("\n[+] Configuration is valid")
        return True
    
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}", file=sys.stderr)
        return False


def main():
    """Test policy loader with sample configuration."""
    import sys
    
    config_file = "policy_config.yaml"
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    print("=" * 70)
    print("POLICY LOADER TEST")
    print("=" * 70)
    print()
    
    # Validate configuration
    if not validate_policy_config(config_file):
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("RULE DETAILS")
    print("=" * 70)
    
    # Load and display rules
    rules = load_policies_from_yaml(config_file)
    
    for rule in rules:
        print(f"\n{rule.rule_id} (precedence {rule.precedence}):")
        print(f"  Type: {rule.type}")
        print(f"  Reason: {rule.reason}")
    
    print("\n" + "=" * 70)
    print("TEST WITH SAMPLE FINDINGS")
    print("=" * 70)
    
    # Test rules with sample findings
    from policy_engine import PolicyEngine, apply_policy_filter
    
    engine = PolicyEngine()
    engine.add_rules(rules)
    
    test_findings = [
        {
            "host_ip": "10.0.1.5",
            "port": 8009,
            "service": "ajp13",
            "cvss": 9.8,
            "severity_text": "Critical",
            "title": "Apache Tomcat AJP Remote Code Execution",
            "cve": "CVE-2020-1938"
        },
        {
            "host_ip": "10.0.1.10",
            "port": 8080,
            "service": "http",
            "cvss": 7.5,
            "severity_text": "High",
            "title": "Unpatched Web Server",
            "cve": "CVE-2023-1234"
        },
        {
            "host_ip": "192.168.1.100",
            "port": 22,
            "service": "ssh",
            "cvss": 5.0,
            "severity_text": "Medium",
            "title": "SSH Weak Ciphers",
            "cve": None
        },
        {
            "host_ip": "10.0.1.20",
            "port": 3306,
            "service": "mysql-database",
            "cvss": 8.5,
            "severity_text": "High",
            "title": "MySQL Authentication Bypass",
            "cve": "CVE-2023-5678"
        },
        {
            "host_ip": "10.0.1.25",
            "port": 22,
            "service": "ssh",
            "cvss": 0.0,
            "severity_text": "None",
            "title": "SSH Server Banner Disclosure",
            "cve": None
        }
    ]
    
    selected, ignored = apply_policy_filter(test_findings, engine)
    
    print(f"\nSelected: {len(selected)} findings")
    for f in selected:
        hints_str = f" [HINTS: {f['hints']}]" if 'hints' in f else ""
        print(f"  ✓ {f['title']}")
        print(f"    → {f['policy_action']} (Rule: {f['policy_rule_id']}){hints_str}")
    
    print(f"\nIgnored: {len(ignored)} findings")
    for f in ignored:
        print(f"  ✗ {f['title']}")
        print(f"    → {f['policy_reason']} (Rule: {f['policy_rule_id']})")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
