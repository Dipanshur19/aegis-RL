#!/usr/bin/env python3
"""
Test Policy Loader Integration

Tests that policy_config.yaml loads correctly and integrates with policy_engine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from policy_loader import load_policies_from_yaml, validate_policy_config
from policy_engine import PolicyEngine, apply_policy_filter


def test_load_yaml_config():
    """Test loading policy_config.yaml"""
    print("\n=== Test: Load YAML Configuration ===")
    
    config_file = "policy_config.yaml"
    
    if not Path(config_file).exists():
        print(f"⚠️  {config_file} not found, skipping test")
        return True
    
    rules = load_policies_from_yaml(config_file)
    
    assert len(rules) > 0, "Should load at least one rule"
    
    # Check precedence levels are valid (0, 1, or 2)
    for rule in rules:
        assert rule.precedence in [0, 1, 2], \
            f"Rule {rule.rule_id} has invalid precedence: {rule.precedence}"
    
    # Check rule types are valid
    for rule in rules:
        assert rule.type in ['ignore', 'force_manual', 'prioritize'], \
            f"Rule {rule.rule_id} has invalid type: {rule.type}"
    
    print(f"✅ Loaded {len(rules)} rules from {config_file}")
    return True


def test_yaml_rule_precedence():
    """Test that YAML rules follow precedence hierarchy"""
    print("\n=== Test: YAML Rule Precedence ===")
    
    config_file = "policy_config.yaml"
    
    if not Path(config_file).exists():
        print(f"⚠️  {config_file} not found, skipping test")
        return True
    
    rules = load_policies_from_yaml(config_file)
    
    # Group by precedence
    by_precedence = {0: [], 1: [], 2: []}
    for rule in rules:
        by_precedence[rule.precedence].append(rule)
    
    print(f"Precedence 0 (User): {len(by_precedence[0])} rules")
    print(f"Precedence 1 (Org): {len(by_precedence[1])} rules")
    print(f"Precedence 2 (Baseline): {len(by_precedence[2])} rules")
    
    # Verify user rules start with USER-
    for rule in by_precedence[0]:
        assert rule.rule_id.startswith("USER-"), \
            f"User-level rule should have USER- prefix: {rule.rule_id}"
    
    # Verify org rules start with ORG-
    for rule in by_precedence[1]:
        assert rule.rule_id.startswith("ORG-"), \
            f"Org-level rule should have ORG- prefix: {rule.rule_id}"
    
    # Verify baseline rules start with BASE-
    for rule in by_precedence[2]:
        assert rule.rule_id.startswith("BASE-"), \
            f"Baseline rule should have BASE- prefix: {rule.rule_id}"
    
    print("✅ Rule precedence hierarchy is correct")
    return True


def test_yaml_operators():
    """Test different operators in YAML rules"""
    print("\n=== Test: YAML Operators ===")
    
    config_file = "policy_config.yaml"
    
    if not Path(config_file).exists():
        print(f"⚠️  {config_file} not found, skipping test")
        return True
    
    rules = load_policies_from_yaml(config_file)
    engine = PolicyEngine()
    engine.add_rules(rules)
    
    # Test 'in' operator (port in list)
    finding_port = {"host_ip": "10.0.1.5", "port": 8080, "title": "Test", "cvss": 7.5}
    action, reason, rule_id = engine.evaluate(finding_port)
    assert action == "ignore", "Port 8080 should be ignored by ORG-001"
    assert rule_id == "ORG-001"
    print(f"✓ 'in' operator works: {rule_id}")
    
    # Test 'eq' operator (cvss == 0.0)
    finding_info = {"host_ip": "10.0.1.10", "port": 22, "title": "Info", "cvss": 0.0}
    action, reason, rule_id = engine.evaluate(finding_info)
    assert action == "ignore", "CVSS 0.0 should be ignored"
    assert rule_id in ["ORG-006", "BASE-001"]  # Either org or baseline rule
    print(f"✓ 'eq' operator works: {rule_id}")
    
    # Test 'regex' operator (title matches RCE pattern)
    finding_rce = {
        "host_ip": "10.0.1.15",
        "port": 8009,
        "title": "Apache Tomcat AJP Remote Code Execution",
        "cvss": 9.8
    }
    action, reason, rule_id = engine.evaluate(finding_rce)
    assert action == "prioritize", "RCE should be prioritized"
    assert rule_id == "ORG-005"
    print(f"✓ 'regex' operator works: {rule_id}")
    
    # Test 'gte' operator (cvss >= 9.0)
    finding_critical = {
        "host_ip": "10.0.1.20",
        "port": 445,
        "title": "SMB Critical Vuln",
        "cvss": 9.5
    }
    action, reason, rule_id = engine.evaluate(finding_critical)
    assert action == "prioritize", "CVSS >= 9.0 should be prioritized"
    assert rule_id == "BASE-004"
    print(f"✓ 'gte' operator works: {rule_id}")
    
    print("✅ All YAML operators work correctly")
    return True


def test_integration_with_policy_engine():
    """Test full integration: YAML → PolicyEngine → apply_policy_filter"""
    print("\n=== Test: Full Integration ===")
    
    config_file = "policy_config.yaml"
    
    if not Path(config_file).exists():
        print(f"⚠️  {config_file} not found, skipping test")
        return True
    
    # Load YAML rules
    rules = load_policies_from_yaml(config_file)
    
    # Create engine and add rules
    engine = PolicyEngine()
    engine.add_rules(rules)
    
    # Sample findings
    findings = [
        {
            "host_ip": "192.168.1.100",  # USER-001: excluded host
            "port": 22,
            "service": "ssh",
            "cvss": 7.0,
            "title": "SSH Weak Config"
        },
        {
            "host_ip": "10.0.1.5",
            "port": 3306,
            "service": "mysql-database",  # USER-002: force manual for database
            "cvss": 8.5,
            "title": "MySQL Auth Bypass"
        },
        {
            "host_ip": "10.0.1.10",
            "port": 8009,
            "service": "ajp13",
            "cvss": 9.8,
            "cve": "CVE-2020-1938",  # ORG-002: prioritize critical CVE
            "title": "Tomcat Ghostcat"
        }
    ]
    
    # Apply policy filter
    selected, ignored = apply_policy_filter(findings, engine)
    
    print(f"Total findings: {len(findings)}")
    print(f"Selected: {len(selected)}")
    print(f"Ignored: {len(ignored)}")
    
    assert len(selected) == 2, "Should have 2 selected findings"
    assert len(ignored) == 1, "Should have 1 ignored finding"
    
    # Check that ignored finding is the excluded host
    assert ignored[0]["host_ip"] == "192.168.1.100"
    assert ignored[0]["policy_rule_id"] == "USER-001"
    
    # Check that database finding has force_manual hints
    db_finding = [f for f in selected if "database" in f["service"]][0]
    assert "hints" in db_finding
    assert db_finding["hints"]["force_manual"] == True
    assert db_finding["policy_rule_id"] == "USER-002"
    
    # Check that critical CVE has prioritize hints
    cve_finding = [f for f in selected if f.get("cve") == "CVE-2020-1938"][0]
    assert "hints" in cve_finding
    assert cve_finding["hints"]["prioritize"] == True
    assert cve_finding["policy_rule_id"] == "ORG-002"
    
    print("✅ Full integration works correctly")
    return True


def run_all_tests():
    """Run all policy loader tests"""
    print("=" * 70)
    print("POLICY LOADER INTEGRATION TESTS")
    print("=" * 70)
    
    tests = [
        ("Load YAML Configuration", test_load_yaml_config),
        ("YAML Rule Precedence", test_yaml_rule_precedence),
        ("YAML Operators", test_yaml_operators),
        ("Full Integration", test_integration_with_policy_engine),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ Test error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 All policy loader tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
