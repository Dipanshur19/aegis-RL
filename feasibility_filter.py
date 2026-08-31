#!/usr/bin/env python3
"""
feasibility_filter.py - Automated Pentest Feasibility Filter

This module triages classified vulnerability findings to determine which are suitable
candidates for automated penetration testing versus those requiring manual review.

Implements rule-based filtering on LLM-enriched vulnerability data to identify
high-value, remotely accessible, scriptable vulnerabilities.

NO EXPLOITATION IS PERFORMED - This is purely a triage/filtering module.
"""

import json
import sys
from typing import Any


def compute_risk_score(finding: dict) -> float:
    """
    Calculate risk score for a vulnerability finding (Phase 4).
    
    Formula: r(f) = cvss × w_surface × w_auto
    
    Where:
    - cvss: CVSS base score (0.0-10.0)
    - w_surface: Attack surface weight based on attack_vector
        * Network: 1.0, Adjacent: 0.7, Local: 0.4, Physical: 0.2
    - w_auto: Automation feasibility weight
        * Automatable (automation_candidate=True): 1.0
        * Manual (automation_candidate=False): 0.3
    
    Args:
        finding: Dictionary with cvss, attack_vector, automation_candidate
        
    Returns:
        Risk score (0.0-10.0)
    """
    cvss = finding.get('cvss') or 0.0
    attack_vector = finding.get('attack_vector', 'Local')
    automation_candidate = finding.get('automation_candidate', False)
    
    # Attack surface weights
    surface_weights = {
        'Network': 1.0,
        'Adjacent': 0.7,
        'Local': 0.4,
        'Physical': 0.2
    }
    w_surface = surface_weights.get(attack_vector, 0.4)
    
    # Automation weights
    w_auto = 1.0 if automation_candidate else 0.3
    
    # Calculate risk score
    risk_score = cvss * w_surface * w_auto
    
    return round(risk_score, 2)


def split_feasible(findings: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split findings into feasible automation candidates vs manual review required.
    
    Feasibility criteria (ALL must be true):
    1. automation_candidate == True (LLM suggests scriptable)
    2. attack_vector in ["Network", "Adjacent"] (remotely accessible)
    3. severity_bucket in ["High", "Critical"] (high impact)
    4. llm_confidence >= 0.5 (LLM is reasonably confident)
    
    Args:
        findings: List of enriched finding dictionaries from classifier
        
    Returns:
        Tuple of (feasible_findings, non_feasible_findings)
        Each finding gains "auto_feasibility_reason" key explaining the decision
    """
    feasible: list[dict] = []
    non_feasible: list[dict] = []
    
    for finding in findings:
        # Extract relevant fields with safe defaults
        auto_candidate = finding.get("automation_candidate", False)
        attack_vector = finding.get("attack_vector", "")
        severity = finding.get("severity_bucket", "")
        confidence = finding.get("llm_confidence", 0.0)
        
        # Make a copy to avoid mutating input
        enriched = finding.copy()
        
        # Evaluate feasibility criteria
        reasons = []
        is_feasible = True
        
        # Check criterion 1: automation candidate
        if not auto_candidate:
            reasons.append("not marked as automation candidate")
            is_feasible = False
        
        # Check criterion 2: attack vector
        if attack_vector not in ["Network", "Adjacent"]:
            reasons.append(f"attack vector is '{attack_vector}' (requires Network/Adjacent)")
            is_feasible = False
        
        # Check criterion 3: severity
        if severity not in ["High", "Critical"]:
            reasons.append(f"severity is '{severity}' (requires High/Critical)")
            is_feasible = False
        
        # Check criterion 4: confidence threshold
        if confidence < 0.5:
            reasons.append(f"LLM confidence too low ({confidence:.2f} < 0.5)")
            is_feasible = False
        
        # Calculate risk score (Phase 4)
        risk_score = compute_risk_score(enriched)
        enriched["risk_score"] = risk_score
        
        # Build reason string
        if is_feasible:
            enriched["auto_feasibility_reason"] = (
                f"Feasible: automation candidate, {attack_vector} accessible, "
                f"{severity} severity, {confidence:.2f} confidence, risk={risk_score}"
            )
            feasible.append(enriched)
        else:
            enriched["auto_feasibility_reason"] = (
                f"Manual review: {'; '.join(reasons)}, risk={risk_score}"
            )
            non_feasible.append(enriched)
    
    return feasible, non_feasible


def build_host_summary(findings: list[dict]) -> dict[str, dict[str, Any]]:
    """
    Build per-host summary statistics for vulnerability findings.
    
    Aggregates findings by host IP and computes:
    - Total finding count
    - Critical/High severity count
    - Feasible automation candidate count
    - Top 3 critical vulnerability titles
    
    Args:
        findings: List of enriched finding dictionaries (must include feasibility_reason)
        
    Returns:
        Dictionary keyed by host_ip with summary statistics
    """
    host_data: dict[str, dict[str, Any]] = {}
    
    for finding in findings:
        host_ip = finding.get("host_ip", "unknown")
        severity = finding.get("severity_bucket", "")
        title = finding.get("title", "Unknown vulnerability")
        
        # Determine if feasible based on presence of positive feasibility reason
        feasibility_reason = finding.get("auto_feasibility_reason", "")
        is_feasible = feasibility_reason.startswith("Feasible:")
        
        # Initialize host entry if not exists
        if host_ip not in host_data:
            host_data[host_ip] = {
                "total_findings": 0,
                "critical_or_high": 0,
                "feasible_candidates": 0,
                "top_critical_titles": []
            }
        
        # Update counters
        host_data[host_ip]["total_findings"] += 1
        
        if severity in ["High", "Critical"]:
            host_data[host_ip]["critical_or_high"] += 1
        
        if is_feasible:
            host_data[host_ip]["feasible_candidates"] += 1
        
        # Track critical titles (limit to 3)
        if severity == "Critical":
            critical_titles = host_data[host_ip]["top_critical_titles"]
            if len(critical_titles) < 3 and title not in critical_titles:
                critical_titles.append(title)
    
    return host_data


def main() -> None:
    """
    Test harness for feasibility filter module.
    
    Loads findings, applies classification (mocked), filters for feasibility,
    and displays triage results with host summaries.
    """
    print("[*] Testing feasibility filter module\n")
    
    # Import dependencies
    try:
        import parser
        import classifier
    except ImportError as e:
        print(f"[ERROR] Could not import required modules: {e}", file=sys.stderr)
        print("[ERROR] Ensure parser.py and classifier.py exist in the same directory",
              file=sys.stderr)
        sys.exit(1)
    
    # Load and parse Nessus XML
    xml_file = "auvap_nessus_100_findings.xml"
    try:
        print(f"[*] Parsing {xml_file}...")
        findings = parser.parse_nessus_xml(xml_file)
        findings_dicts = parser.to_dict_list(findings)
        print(f"[+] Loaded {len(findings_dicts)} findings\n")
    except Exception as e:
        print(f"[ERROR] Failed to parse XML: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not findings_dicts:
        print("[!] No findings to process")
        return
    
    # Monkey-patch classifier to avoid network calls
    def mock_classify_single(self: Any, finding: dict) -> dict:
        """Mock classification returning varied test data."""
        enriched = finding.copy()
        
        cvss = finding.get("cvss", 0.0) or 0.0
        title_lower = finding.get("title", "").lower()
        port = finding.get("port", 0)
        
        # Simulate diverse classification results for testing
        if cvss >= 9.0 or "remote code execution" in title_lower or "rce" in title_lower:
            enriched.update({
                "severity_bucket": "Critical",
                "attack_vector": "Network",
                "vuln_component": finding.get("service", "Unknown service"),
                "exploit_notes": "Remote code execution allowing arbitrary command execution.",
                "automation_candidate": True,
                "llm_confidence": 0.92
            })
        elif cvss >= 7.0 and ("disclosure" in title_lower or "unauthenticated" in title_lower):
            enriched.update({
                "severity_bucket": "High",
                "attack_vector": "Network",
                "vuln_component": f"{finding.get('service', 'Service')} on port {port}",
                "exploit_notes": "High-severity vulnerability with information disclosure risk.",
                "automation_candidate": True,
                "llm_confidence": 0.85
            })
        elif cvss >= 7.0:
            enriched.update({
                "severity_bucket": "High",
                "attack_vector": "Local",  # Not network-accessible - should be non-feasible
                "vuln_component": finding.get("title", "Unknown")[:40],
                "exploit_notes": "High-severity but requires local access.",
                "automation_candidate": False,
                "llm_confidence": 0.78
            })
        elif cvss >= 4.0:
            enriched.update({
                "severity_bucket": "Medium",
                "attack_vector": "Network",
                "vuln_component": finding.get("title", "Unknown")[:40],
                "exploit_notes": "Medium-severity issue requiring specific conditions.",
                "automation_candidate": False,  # Not auto candidate - should be non-feasible
                "llm_confidence": 0.72
            })
        else:
            enriched.update({
                "severity_bucket": "Low",
                "attack_vector": "Network",
                "vuln_component": "Configuration issue",
                "exploit_notes": "Low-severity informational finding.",
                "automation_candidate": False,
                "llm_confidence": 0.65
            })
        
        return enriched
    
    # Apply monkey patch
    import os
    original_method = classifier.LLMClassifierClient.classify_single
    classifier.LLMClassifierClient.classify_single = mock_classify_single
    
    try:
        # Set dummy API key for testing
        os.environ["GITHUB_API_KEY"] = "test_key_for_demo"
        
        # Classify findings (uses mocked method)
        print("[*] Classifying findings (using mock LLM)...")
        classified = classifier.classify_findings(findings_dicts)
        print(f"[+] Classified {len(classified)} findings\n")
        
        # Apply feasibility filter
        print("[*] Applying feasibility filter...")
        feasible, non_feasible = split_feasible(classified)
        
        print(f"[+] Feasibility triage complete:\n")
        print(f"    Feasible for automation:  {len(feasible)}")
        print(f"    Requires manual review:   {len(non_feasible)}")
        print(f"    Total:                    {len(feasible) + len(non_feasible)}\n")
        
        # Build host summary
        all_findings = feasible + non_feasible
        host_summary = build_host_summary(all_findings)
        
        print(f"[*] Host summary ({len(host_summary)} hosts):\n")
        print(json.dumps(host_summary, indent=2, default=str))
        
        # Show example feasible findings
        if feasible:
            print(f"\n[*] Example feasible finding:\n")
            example = {k: v for k, v in feasible[0].items() 
                      if k in ["host_ip", "port", "service", "title", "severity_bucket", 
                              "attack_vector", "automation_candidate", "llm_confidence",
                              "auto_feasibility_reason"]}
            print(json.dumps(example, indent=2, default=str))
        
        # Show example non-feasible finding
        if non_feasible:
            print(f"\n[*] Example non-feasible finding:\n")
            example = {k: v for k, v in non_feasible[0].items()
                      if k in ["host_ip", "port", "service", "title", "severity_bucket",
                              "attack_vector", "automation_candidate", "llm_confidence",
                              "auto_feasibility_reason"]}
            print(json.dumps(example, indent=2, default=str))
        
    finally:
        # Restore original method
        classifier.LLMClassifierClient.classify_single = original_method


if __name__ == "__main__":
    main()
