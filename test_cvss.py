#!/usr/bin/env python3
"""
Test Suite for CVSS Calculator Module
======================================

Tests CVSS computation, validation, and caching functionality.

Usage:
    python test_cvss.py
    pytest test_cvss.py -v
"""

import os
import sys
import json
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cvss_calculator import (
    compute_cvss,
    validate_cvss,
    enrich_finding_with_cvss,
    CVSSCalculator,
    CVSSMetrics,
    CVSSResult
)


def test_nvd_api_integration():
    """Test fetching CVSS from NVD API"""
    print("\n=== Test 1: NVD API Integration ===")
    
    # Test with known CVE (Ghostcat - Apache Tomcat AJP)
    result = compute_cvss(cve="CVE-2020-1938", force_refresh=True)
    
    print(f"CVE: CVE-2020-1938")
    print(f"Score: {result.cvss_score}")
    print(f"Severity: {result.severity}")
    print(f"Vector: {result.cvss_vector}")
    print(f"Source: {result.source}")
    print(f"Confidence: {result.confidence}")
    
    assert result.cvss_score > 0, "Score should be greater than 0"
    assert result.source in ["nvd_api", "cached"], f"Expected nvd_api or cached, got {result.source}"
    assert result.confidence == "high", f"Expected high confidence, got {result.confidence}"
    assert result.validated is True, "Should be validated"
    
    print("✅ NVD API test passed!")
    return result


def test_cache_functionality():
    """Test SQLite cache works"""
    print("\n=== Test 2: Cache Functionality ===")
    
    calc = CVSSCalculator()
    
    # First call - should hit API or already be cached
    print("First call (may hit API or cache)...")
    result1 = calc.compute_cvss(cve="CVE-2021-41773", force_refresh=True)
    source1 = result1.source
    
    print(f"First call source: {source1}")
    print(f"Score: {result1.cvss_score}")
    
    # Second call - should definitely hit cache
    print("Second call (should hit cache)...")
    time.sleep(0.5)
    result2 = calc.compute_cvss(cve="CVE-2021-41773")
    
    print(f"Second call source: {result2.source}")
    print(f"Score: {result2.cvss_score}")
    
    assert result2.source == "cached", f"Expected cached, got {result2.source}"
    assert result1.cvss_score == result2.cvss_score, "Scores should match"
    
    print("✅ Cache test passed!")
    return result2


def test_score_validation():
    """Test CVSS score validation"""
    print("\n=== Test 3: Score Validation ===")
    
    # Test with EternalBlue (MS17-010) - known Critical vulnerability
    validation = validate_cvss("CVE-2017-0144", reported_score=8.1)
    
    print(f"CVE: CVE-2017-0144 (EternalBlue)")
    print(f"Reported score: 8.1")
    print(f"Official score: {validation.get('official_score')}")
    print(f"Is accurate: {validation.get('is_accurate')}")
    print(f"Discrepancy: {validation.get('discrepancy')}")
    print(f"Recommendation: {validation.get('recommendation')}")
    
    if validation.get('official_score'):
        assert validation["official_score"] >= 8.0, "EternalBlue should be High/Critical"
        print("✅ Validation test passed!")
    else:
        print("⚠️  Could not validate (API unavailable)")
    
    return validation


def test_cvss_computation():
    """Test CVSS score computation from metrics"""
    print("\n=== Test 4: CVSS Computation ===")
    
    calc = CVSSCalculator()
    
    # Test metrics for a network-exploitable RCE
    metrics = CVSSMetrics(
        attack_vector='N',
        attack_complexity='L',
        privileges_required='N',
        user_interaction='N',
        scope='U',
        confidentiality='H',
        integrity='H',
        availability='H'
    )
    
    score = calc._compute_score_from_metrics(metrics)
    vector = metrics.to_vector_string()
    
    print(f"Test metrics: Network RCE, No auth required")
    print(f"Vector: {vector}")
    print(f"Computed score: {score}")
    print(f"Severity: {calc._score_to_severity(score)}")
    
    # Should be Critical (9.0+) for unauthenticated network RCE
    assert score >= 9.0, f"Expected Critical (9.0+), got {score}"
    assert calc._score_to_severity(score) == "Critical"
    
    print("✅ Computation test passed!")
    return score


def test_graceful_degradation():
    """Test fallback when all methods fail"""
    print("\n=== Test 5: Graceful Degradation ===")
    
    # Test with no CVE, minimal description, but existing score
    result = compute_cvss(
        cve=None,
        description="",
        existing_score=5.5
    )
    
    print(f"Input: No CVE, no description, existing_score=5.5")
    print(f"Score: {result.cvss_score}")
    print(f"Source: {result.source}")
    print(f"Confidence: {result.confidence}")
    
    assert result.cvss_score == 5.5, f"Expected 5.5, got {result.cvss_score}"
    assert result.source == "existing_score", f"Expected existing_score, got {result.source}"
    assert result.confidence == "low", f"Expected low confidence, got {result.confidence}"
    
    print("✅ Graceful degradation test passed!")
    return result


def test_finding_enrichment():
    """Test enriching vulnerability finding"""
    print("\n=== Test 6: Finding Enrichment ===")
    
    # Sample finding from parser
    finding = {
        "host_ip": "10.0.1.5",
        "port": 8009,
        "service": "ajp13",
        "cve": "CVE-2020-1938",
        "title": "Apache Tomcat AJP File Read Vulnerability",
        "description": "Apache Tomcat AJP Connector allows reading of arbitrary files",
        "cvss": 9.8,
        "severity": "Critical"
    }
    
    print("Original finding:")
    print(json.dumps(finding, indent=2))
    
    enriched = enrich_finding_with_cvss(finding)
    
    print("\nEnriched finding (new fields):")
    print(f"  cvss_computed: {enriched.get('cvss_computed')}")
    print(f"  cvss_vector: {enriched.get('cvss_vector')}")
    print(f"  cvss_severity: {enriched.get('cvss_severity')}")
    print(f"  cvss_confidence: {enriched.get('cvss_confidence')}")
    print(f"  cvss_source: {enriched.get('cvss_source')}")
    print(f"  cvss_validated: {enriched.get('cvss_validated')}")
    
    assert "cvss_computed" in enriched, "Missing cvss_computed"
    assert "cvss_confidence" in enriched, "Missing cvss_confidence"
    assert enriched["cvss_computed"] > 0, "Computed score should be > 0"
    
    print("✅ Enrichment test passed!")
    return enriched


def test_severity_mapping():
    """Test CVSS score to severity conversion"""
    print("\n=== Test 7: Severity Mapping ===")
    
    calc = CVSSCalculator()
    
    test_cases = [
        (0.0, "None"),
        (2.5, "Low"),
        (5.5, "Medium"),
        (8.0, "High"),
        (9.5, "Critical")
    ]
    
    all_passed = True
    for score, expected_severity in test_cases:
        severity = calc._score_to_severity(score)
        status = "✅" if severity == expected_severity else "❌"
        print(f"{status} Score {score} → {severity} (expected: {expected_severity})")
        
        if severity != expected_severity:
            all_passed = False
    
    assert all_passed, "Some severity mappings failed"
    print("✅ Severity mapping test passed!")


def test_unavailable_cve():
    """Test handling of non-existent CVE"""
    print("\n=== Test 8: Non-existent CVE ===")
    
    # Test with fake CVE
    result = compute_cvss(
        cve="CVE-9999-99999",
        description="This CVE does not exist",
        existing_score=None
    )
    
    print(f"Fake CVE: CVE-9999-99999")
    print(f"Score: {result.cvss_score}")
    print(f"Source: {result.source}")
    print(f"Confidence: {result.confidence}")
    
    # Should return unavailable or use LLM estimation
    assert result.source in ["unavailable", "llm_estimated", "cached"], \
        f"Unexpected source: {result.source}"
    
    print("✅ Non-existent CVE test passed!")
    return result


def run_all_tests():
    """Run all tests in sequence"""
    print("=" * 70)
    print("CVSS CALCULATOR TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("NVD API Integration", test_nvd_api_integration),
        ("Cache Functionality", test_cache_functionality),
        ("Score Validation", test_score_validation),
        ("CVSS Computation", test_cvss_computation),
        ("Graceful Degradation", test_graceful_degradation),
        ("Finding Enrichment", test_finding_enrichment),
        ("Severity Mapping", test_severity_mapping),
        ("Non-existent CVE", test_unavailable_cve),
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
            failed += 1
        
        # Rate limiting between API calls
        time.sleep(1)
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
