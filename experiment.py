#!/usr/bin/env python3
"""
experiment.py - AUVAP Vulnerability Assessment Triage Pipeline

This script orchestrates the complete offline vulnerability assessment pipeline:
1. Parse Nessus XML export
2. Classify and enrich findings using LLM
3. Filter for automation feasibility
4. Generate structured JSON report for management/audit

NO ACTIVE EXPLOITATION OR NETWORK SCANNING IS PERFORMED.
This is purely offline analysis of existing vulnerability scan data.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _prompt_local_model(default: str = "deepseek-r1:14b") -> str:
    """Prompt user to select a local LLM model for classification."""
    print("Select local model:")
    print("  1. deepseek-r1:14b (default)")
    print("  2. qwen3:14b")
    print("  3. Custom model name")
    print()

    choice = input("Choose model (1-3) [1]: ").strip()
    if choice == '2':
        return 'qwen3:14b'
    if choice == '3':
        custom = input("Enter local model name: ").strip()
        return custom or default
    return default


def build_host_report(
    host_ip: str,
    all_findings: list[dict],
    feasible_findings: list[dict]
) -> dict[str, Any]:
    """
    Build a summary report for a single host.
    
    Args:
        host_ip: Target host IP address
        all_findings: All classified findings (feasible + non-feasible)
        feasible_findings: Subset of findings deemed feasible for automation
        
    Returns:
        Dictionary with host summary and top risks
    """
    # Filter findings for this host
    host_findings = [f for f in all_findings if f.get("host_ip") == host_ip]
    
    if not host_findings:
        return {}
    
    # Extract host metadata from first finding
    first = host_findings[0]
    hostname = first.get("hostname", host_ip)
    host_os = first.get("os", "Unknown")
    
    # Count critical/high findings
    critical_high = [f for f in host_findings 
                     if f.get("severity_bucket") in ["Critical", "High"]]
    
    # Count feasible candidates for this host
    host_feasible = [f for f in feasible_findings if f.get("host_ip") == host_ip]
    
    # Select top risks (up to 3 Critical/High findings, sorted by severity then CVSS)
    def risk_sort_key(finding: dict) -> tuple[int, float]:
        """Sort key: Critical=0, High=1, then by CVSS descending."""
        severity = finding.get("severity_bucket", "")
        cvss = finding.get("cvss") or 0.0
        
        severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        return (severity_order.get(severity, 99), -cvss)
    
    top_risks_raw = sorted(critical_high, key=risk_sort_key)[:3]
    
    # Format top risks for report
    top_risks = []
    for finding in top_risks_raw:
        top_risks.append({
            "title": finding.get("title", "Unknown"),
            "severity_bucket": finding.get("severity_bucket", "Unknown"),
            "cvss": finding.get("cvss"),
            "port": finding.get("port", 0),
            "service": finding.get("service", "unknown"),
            "attack_vector": finding.get("attack_vector", "Unknown"),
            "exploit_notes": finding.get("exploit_notes", ""),
            "remediation": finding.get("remediation", "")
        })
    
    return {
        "host_ip": host_ip,
        "os": host_os,
        "hostname": hostname,
        "critical_high_findings": len(critical_high),
        "feasible_candidates": len(host_feasible),
        "top_risks": top_risks
    }


def build_feasible_findings_detail(feasible_findings: list[dict]) -> list[dict]:
    """
    Extract detailed information for all feasible findings.
    
    Args:
        feasible_findings: List of findings deemed feasible for automation
        
    Returns:
        List of dictionaries with essential fields for each feasible finding
    """
    detailed = []
    
    for finding in feasible_findings:
        detailed.append({
            "host_ip": finding.get("host_ip", "unknown"),
            "title": finding.get("title", "Unknown"),
            "severity_bucket": finding.get("severity_bucket", "Unknown"),
            "cvss": finding.get("cvss"),
            "cve": finding.get("cve"),
            "port": finding.get("port", 0),
            "service": finding.get("service", "unknown"),
            "attack_vector": finding.get("attack_vector", "Unknown"),
            "vuln_component": finding.get("vuln_component", "unknown"),
            "exploit_notes": finding.get("exploit_notes", ""),
            "auto_feasibility_reason": finding.get("auto_feasibility_reason", ""),
            "remediation": finding.get("remediation", "")
        })
    
    return detailed


def generate_report(
    source_file: str,
    all_findings: list[dict],
    feasible_findings: list[dict],
    non_feasible_findings: list[dict]
) -> dict[str, Any]:
    """
    Generate the complete experiment report structure.
    
    Args:
        source_file: Name of source Nessus XML file
        all_findings: All classified findings
        feasible_findings: Findings deemed feasible for automation
        non_feasible_findings: Findings requiring manual review
        
    Returns:
        Complete report dictionary ready for JSON serialization
    """
    # Build metadata section
    metadata = {
        "source_file": source_file,
        "total_findings": len(all_findings),
        "generated_by": "AUVAP experimental pipeline (offline triage only)",
        "disclaimer": "This report is generated from static VA data only. No active exploitation was performed."
    }
    
    # Get unique host IPs
    host_ips = sorted(set(f.get("host_ip", "unknown") for f in all_findings))
    
    # Build per-host reports
    hosts = []
    for host_ip in host_ips:
        host_report = build_host_report(host_ip, all_findings, feasible_findings)
        if host_report:
            hosts.append(host_report)
    
    # Sort hosts by critical/high findings (descending)
    hosts.sort(key=lambda h: h.get("critical_high_findings", 0), reverse=True)
    
    # Build detailed feasible findings
    feasible_detailed = build_feasible_findings_detail(feasible_findings)
    
    return {
        "metadata": metadata,
        "hosts": hosts,
        "feasible_findings_detailed": feasible_detailed
    }


def print_summary(report: dict[str, Any]) -> None:
    """
    Print human-readable summary to stdout.
    
    Args:
        report: Complete experiment report dictionary
    """
    print("=" * 70)
    print("AEGIS-RL VULNERABILITY ASSESSMENT TRIAGE REPORT")
    print("=" * 70)
    print()
    
    metadata = report.get("metadata", {})
    print(f"Source File: {metadata.get('source_file', 'Unknown')}")
    print(f"Total Findings: {metadata.get('total_findings', 0)}")
    print()
    
    feasible_count = len(report.get("feasible_findings_detailed", []))
    print(f"Feasible Automation Candidates: {feasible_count}")
    print()
    
    # List hosts sorted by critical/high count
    hosts = report.get("hosts", [])
    if hosts:
        print("Hosts (sorted by Critical/High findings):")
        print("-" * 70)
        for host in hosts:
            host_ip = host.get("host_ip", "unknown")
            hostname = host.get("hostname", "")
            crit_high = host.get("critical_high_findings", 0)
            feasible = host.get("feasible_candidates", 0)
            
            hostname_str = f" ({hostname})" if hostname and hostname != host_ip else ""
            print(f"  {host_ip}{hostname_str}")
            print(f"    Critical/High: {crit_high} | Feasible: {feasible}")
        print()
    
    print("=" * 70)
    print("DISCLAIMER:")
    print(metadata.get("disclaimer", ""))
    print("=" * 70)


def main() -> None:
    """
    Main orchestration function for AUVAP experiment pipeline.
    
    Loads vulnerability data, applies classification and feasibility filtering,
    generates structured report, and outputs summary and JSON.
    """
    # Import pipeline modules
    try:
        import parser
        import classifier_v2 as classifier
        import feasibility_filter
    except ImportError as e:
        print(f"[ERROR] Failed to import required modules: {e}", file=sys.stderr)
        print("[ERROR] Ensure parser.py, classifier_v2.py, and feasibility_filter.py exist",
              file=sys.stderr)
        sys.exit(1)
    
    # Configuration
    # Supports: .xml, .nessus (Nessus XML) or .csv (CSV format)
    input_file = "auvap_nessus_25_findings.xml"  # Change to .csv for CSV files
    
    # Create results folder if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate timestamped output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"experiment_report_{timestamp}.json"
    
    # ========================================================================
    # BUSINESS CONTEXT CONFIGURATION
    # ========================================================================
    # Interactive mode: Ask user for custom context
    print("=" * 70)
    print("Aegis-RL - Vulnerability Assessment & Pentesting Pipeline")
    print("=" * 70)
    print()
    
    # Ask if user wants to add custom context
    add_context = input("Add custom business context for LLM classification? (y/n) [n]: ").strip().lower()
    
    if add_context in ['y', 'yes']:
        print()
        print("Enter custom context (press Enter to skip any field):")
        print("-" * 70)
        
        # Excluded ports
        excluded_ports_input = input("Excluded ports (comma-separated, e.g., 33,8080): ").strip()
        excluded_ports = []
        if excluded_ports_input:
            try:
                excluded_ports = [int(p.strip()) for p in excluded_ports_input.split(',')]
            except ValueError:
                print("  [WARNING] Invalid port numbers, using defaults")
                excluded_ports = []
        
        # Critical services
        critical_services_input = input("Critical services (comma-separated, e.g., apache,nginx): ").strip()
        critical_services = []
        if critical_services_input:
            critical_services = [s.strip() for s in critical_services_input.split(',')]
        
        # Environment type
        environment = input("Environment type (production/staging/development/lab) [production]: ").strip().lower() or "production"
        
        # Custom notes
        print("\nCustom guidance for LLM (e.g., 'Port 33 is company standard, ignore it'):")
        custom_notes = input("> ").strip()
        
        # Build business context from user input
        business_context = {
            "excluded_ports": excluded_ports,
            "critical_services": critical_services,
            "environment": environment,
            "custom_notes": custom_notes or "No additional context provided.",
            "safe_configs": [],
            "compliance_requirements": []
        }
        
        print()
        print("[+] Custom context added!")
        print()
    else:
        # Use default context (minimal)
        business_context = {
            "excluded_ports": [],
            "critical_services": ["apache", "tomcat", "jenkins", "nginx"],
            "environment": "production",
            "custom_notes": "Standard vulnerability assessment with no custom exclusions.",
            "safe_configs": [],
            "compliance_requirements": []
        }
        print()
        print("Using default business context...")
        print()
    
    # Step 1: Parse Nessus XML
    print(f"[1/6] Parsing vulnerability report: {input_file}")
    try:
        findings = parser.parse_report(input_file)  # Auto-detects XML or CSV
        findings_dicts = parser.to_dict_list(findings)
        print(f"      Loaded {len(findings_dicts)} findings\n")
    except Exception as e:
        print(f"[ERROR] Failed to parse XML: {e}", file=sys.stderr)
        sys.exit(1)

    if not findings_dicts:
        print("[ERROR] No findings to process", file=sys.stderr)
        sys.exit(1)

    # Step 2: Enrich findings with CVSS scores (compute missing scores, validate existing)
    print("[2/6] Enriching findings with CVSS scores")
    try:
        from cvss_calculator import enrich_finding_with_cvss

        enriched_findings = []
        cvss_computed_count = 0
        cvss_validated_count = 0
        cvss_failed_count = 0

        for finding in findings_dicts:
            try:
                enriched = enrich_finding_with_cvss(finding)
                enriched_findings.append(enriched)

                # Track enrichment statistics
                if enriched.get('cvss_source') == 'computed':
                    cvss_computed_count += 1
                elif enriched.get('cvss_validated'):
                    cvss_validated_count += 1
            except Exception as e:
                # On error, keep original finding
                enriched_findings.append(finding)
                cvss_failed_count += 1

        findings_dicts = enriched_findings

        print(f"      [OK] CVSS enrichment complete:")
        print(f"         Computed: {cvss_computed_count} | Validated: {cvss_validated_count} | Failed: {cvss_failed_count}")
        print()

    except ImportError:
        print(f"      [WARNING] CVSS calculator not available", file=sys.stderr)
        print("      Continuing with original CVSS scores from scan...", file=sys.stderr)
        print()

    # Step 3: Apply Policy Filter BEFORE LLM classification
    print("[3/6] Applying organizational security policies")
    
    try:
        from policy_engine import PolicyEngine, apply_policy_filter, create_default_policy_rules, emit_policy_metrics
        from policy_loader import load_policies_from_yaml
        
        # Initialize policy engine with default rules
        policy_engine = PolicyEngine()
        policy_engine.add_rules(create_default_policy_rules())
        
        # Load policies from configuration file if exists
        policy_file = Path("policy_config.yaml")
        if policy_file.exists():
            print(f"      Loading policies from {policy_file}")
            try:
                yaml_rules = load_policies_from_yaml(str(policy_file))
                policy_engine.add_rules(yaml_rules)
                print(f"      [OK] Loaded {len(yaml_rules)} custom rules from YAML")
            except Exception as e:
                print(f"      [WARNING] Failed to load {policy_file}: {e}", file=sys.stderr)
        else:
            print(f"      Using default policies only (no {policy_file} found)")
        
        # Show policy summary
        rule_types = policy_engine.get_rules_by_type()
        print(f"      Total active rules: {policy_engine.get_rule_count()}")
        print(f"      - Ignore: {rule_types['ignore']}, Force-manual: {rule_types['force_manual']}, Prioritize: {rule_types['prioritize']}")
        print()
        
        # Apply policy filter
        selected_findings, ignored_findings = apply_policy_filter(findings_dicts, policy_engine)
        
        # Emit metrics
        emit_policy_metrics(selected_findings, ignored_findings)
        print()
        
        if not selected_findings:
            print("[ERROR] All findings were filtered out by policies. No findings to classify.", file=sys.stderr)
            sys.exit(1)
        
        # Replace findings_dicts with policy-approved findings
        findings_dicts = selected_findings
        
    except ImportError as e:
        print(f"      [WARNING] Policy engine not available: {e}", file=sys.stderr)
        print("      Continuing without policy filtering...", file=sys.stderr)
        print()
    
    # Step 4: Classify findings using LLM with business context
    print("[4/6] Classifying findings with LLM")
    print(f"      Environment: {business_context['environment']}")
    print(f"      Excluded ports: {business_context['excluded_ports']}")
    print(f"      Critical services: {business_context['critical_services']}")
    if business_context.get('custom_notes'):
        print(f"      Custom guidance: {business_context['custom_notes'][:80]}...")
    print()

    # Provider selection
    print("Select LLM Provider:")
    print("  1. Auto-detect (default)")
    print("  2. OpenAI (GPT-4o-mini)")
    print("  3. Google Gemini")
    print("  4. GitHub Models")
    print("  5. Local (Ollama/LM Studio)")
    print()

    provider_choice = input("Choose provider (1-5) [1]: ").strip()

    provider_map = {
        '1': 'auto',
        '2': 'openai',
        '3': 'gemini',
        '4': 'github',
        '5': 'local',
        '': 'auto'  # Default
    }

    PROVIDER = provider_map.get(provider_choice, 'auto')

    provider_names = {
        'auto': 'Auto-detect',
        'openai': 'OpenAI',
        'gemini': 'Google Gemini',
        'github': 'GitHub Models',
        'local': 'Local (Ollama/LM Studio)'
    }

    model_choice = None
    if PROVIDER == 'local':
        model_choice = _prompt_local_model()

    print(f"[+] Using: {provider_names[PROVIDER]}")
    if model_choice:
        print(f"      Local model: {model_choice}")
    print()

    # Initialize Phase 3 enhancements (calibrator and metrics tracking)
    try:
        from phase3_enhancements import ClassificationMetrics, ClassifierCalibrator

        # Initialize metrics tracker
        metrics = ClassificationMetrics()

        # Initialize calibrator (loads existing state if available)
        calibrator = ClassifierCalibrator(
            base_threshold=0.5,
            learning_rate=0.1,
            target_fpr=0.05,
            save_path=Path("calibration_state.json")
        )

        # Show calibrator status
        if Path("calibration_state.json").exists():
            print(f"      [METRICS] Calibrator loaded (adjusted threshold: {calibrator.adjusted_threshold:.3f})")
        else:
            print(f"      [METRICS] Calibrator initialized (base threshold: {calibrator.base_threshold:.3f})")
        print()

    except ImportError as e:
        print(f"      [WARNING] Phase 3 enhancements not available: {e}", file=sys.stderr)
        print("      Continuing without calibrator...", file=sys.stderr)
        metrics = None
        calibrator = None
        print()

    # Classify findings with metrics tracking
    classified = classifier.classify_findings(
        findings_dicts,
        provider=PROVIDER,
        model=model_choice,
        business_context=business_context,
        metrics=metrics  # Pass metrics tracker if available
    )
    print(f"      Classified {len(classified)} findings")

    # Print classification metrics summary (Phase 3)
    if metrics:
        metrics.print_summary()

        # Save metrics to file
        metrics_file = results_dir / f"classification_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics.get_summary(), f, indent=2)
        print(f"      Metrics saved to: {metrics_file}")

    print()
    
    # Step 5: Apply feasibility filter
    print("[5/6] Applying feasibility filter")
    feasible, non_feasible = feasibility_filter.split_feasible(classified)
    print(f"      Feasible: {len(feasible)} | Manual review: {len(non_feasible)}\n")
    
    # Step 6: Initialize exploit tasks (Phase 4)
    print("[6/6] Initializing exploit tasks")
    try:
        import task_manager
        
        # Create exploit tasks from feasible findings
        tasks = task_manager.initialize_tasks(feasible)
        print(f"      Created {len(tasks)} exploit tasks")
        
        # Print task summary
        task_manager.print_task_summary(tasks)
        
        # Generate task manifest
        manifest_path = results_dir / f"tasks_manifest_{timestamp}.json"
        task_manager.generate_task_manifest(tasks, manifest_path)
        print(f"      Task manifest: {manifest_path}\n")
        
    except ImportError as e:
        print(f"      [WARNING] Task manager not available: {e}", file=sys.stderr)
        print("      Skipping task initialization...", file=sys.stderr)
        print()
    
    # Generate final report
    print("[*] Generating experiment report")
    all_findings = feasible + non_feasible
    report = generate_report(input_file, all_findings, feasible, non_feasible)
    print(f"      Report generated successfully\n")
    
    # Print human-readable summary
    print_summary(report)
    print()
    
    # Write JSON report to file
    print(f"[*] Writing detailed report to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[+] Report saved successfully\n")


if __name__ == "__main__":
    main()
