#!/usr/bin/env python3
"""
parser.py - Vulnerability Scan Report Parser

This module parses vulnerability scan reports (Nessus XML or CSV) and normalizes findings
into structured Python dataclasses for downstream processing (LLM classification,
feasibility scoring, etc.).

Supported formats:
- Nessus XML (.nessus, .xml)
- CSV with standard columns

No network access or active scanning is performed - operates solely on local files.
"""

import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict


@dataclass
class DataQuality:
    """
    Tracks data quality and provenance for vulnerability findings.
    
    Attributes:
        missing_fields: List of field names that were empty in original data
        imputed_fields: List of field names that were enriched/computed
        source: Authoritative source of the data (scanner, nvd_api, llm, computed)
    """
    missing_fields: list[str] = field(default_factory=list)
    imputed_fields: list[str] = field(default_factory=list)
    source: str = "scanner"


@dataclass
class VAFinding:
    """
    Normalized vulnerability assessment finding from a Nessus report.
    
    Attributes:
        host_ip: Target host IP address
        hostname: Target hostname or FQDN
        os: Operating system detected on target
        port: Network port number
        protocol: Network protocol (tcp/udp/icmp)
        service: Service name running on port
        severity_text: Risk level (Critical/High/Medium/Low/None)
        cvss: CVSS base score (v3 preferred, falls back to v2)
        cve: Associated CVE identifier(s)
        title: Vulnerability title (plugin name)
        description: Detailed vulnerability description
        evidence: Plugin output showing proof of vulnerability
        remediation: Recommended solution/fix
        raw_plugin_id: Nessus plugin ID
        raw_plugin_family: Plugin family category
        finding_id: Deterministic hash for deduplication (computed in __post_init__)
        data_quality: Data quality and provenance tracking
        timestamp: When the finding was parsed/created
    """
    host_ip: str
    hostname: str
    os: str
    port: int
    protocol: str
    service: str
    severity_text: str
    cvss: Optional[float]
    cve: Optional[str]
    title: str
    description: str
    evidence: str
    remediation: str
    raw_plugin_id: str
    raw_plugin_family: str
    finding_id: str = field(init=False)
    data_quality: DataQuality = field(default_factory=DataQuality)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Compute deterministic finding_id using SHA-1 hash."""
        # Create deterministic ID: H(host_ip || service || port || cve)
        cve_value = self.cve if self.cve else "NOCVE"
        id_string = f"{self.host_ip}|{self.service}|{self.port}|{cve_value}"
        self.finding_id = hashlib.sha1(id_string.encode('utf-8')).hexdigest()

    @property
    def plugin_name(self) -> str:
        """Alias for title."""
        return self.title

    @property
    def plugin_id(self) -> str:
        """Alias for cve or raw_plugin_id."""
        return self.cve if self.cve else self.raw_plugin_id

    @property
    def cvss_base_score(self) -> Optional[float]:
        """Alias for cvss."""
        return self.cvss

    @property
    def severity(self) -> str:
        """Alias for severity_text."""
        return self.severity_text

    @property
    def synopsis(self) -> str:
        """Alias for description."""
        return self.description


def _safe_get_text(element: Optional[ET.Element], default: str = "") -> str:
    """Safely extract text from an XML element, returning default if None or empty."""
    if element is None:
        return default
    text = element.text
    return text.strip() if text else default


def _safe_get_float(element: Optional[ET.Element]) -> Optional[float]:
    """Safely convert element text to float, returning None if invalid."""
    if element is None:
        return None
    try:
        return float(element.text)
    except (ValueError, TypeError, AttributeError):
        return None


def _safe_get_int(value: str, default: int = 0) -> int:
    """Safely convert string to int, returning default if invalid."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _extract_host_properties(host_properties: Optional[ET.Element]) -> dict[str, str]:
    """
    Extract host metadata from HostProperties element.
    
    Returns:
        Dictionary with keys: host-ip, hostname, operating-system
    """
    props = {
        "host-ip": "",
        "hostname": "",
        "operating-system": ""
    }
    
    if host_properties is None:
        return props
    
    for tag in host_properties.findall("tag"):
        name = tag.get("name", "")
        if name in props:
            props[name] = _safe_get_text(tag)
    
    return props


def parse_nessus_xml(xml_path: str) -> list[VAFinding]:
    """
    Parse a Nessus XML report file and extract normalized vulnerability findings.
    
    Args:
        xml_path: Path to the Nessus .nessus XML export file
        
    Returns:
        List of VAFinding objects, one per vulnerability instance per host
        
    Raises:
        FileNotFoundError: If xml_path does not exist
        ET.ParseError: If XML is malformed
    """
    findings: list[VAFinding] = []
    
    # Validate file exists
    xml_file = Path(xml_path)
    if not xml_file.exists():
        raise FileNotFoundError(f"Nessus XML file not found: {xml_path}")
    
    # Parse XML
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ET.ParseError(f"Failed to parse XML file {xml_path}: {e}")
    
    # Iterate through each report host
    for report_host in root.findall(".//ReportHost"):
        # Extract host-level metadata
        host_properties = report_host.find("HostProperties")
        host_props = _extract_host_properties(host_properties)
        
        host_ip = host_props["host-ip"]
        hostname = host_props["hostname"] or host_ip  # Fallback to IP if no hostname
        os = host_props["operating-system"]
        
        # Iterate through each vulnerability finding for this host
        for item in report_host.findall("ReportItem"):
            # Extract port/protocol/service from attributes
            port = _safe_get_int(item.get("port", "0"))
            protocol = item.get("protocol", "")
            service = item.get("svc_name", "")
            
            # Extract plugin metadata
            plugin_id = item.get("pluginID", "")
            plugin_name = item.get("pluginName", "")
            plugin_family = item.get("pluginFamily", "")
            
            # Extract vulnerability details
            risk_factor = _safe_get_text(item.find("risk_factor"), "None")
            description = _safe_get_text(item.find("description"))
            solution = _safe_get_text(item.find("solution"))
            plugin_output = _safe_get_text(item.find("plugin_output"))
            
            # Extract CVSS scores (prefer v3 over v2)
            cvss3 = _safe_get_float(item.find("cvss3_base_score"))
            cvss2 = _safe_get_float(item.find("cvss_base_score"))
            cvss = cvss3 if cvss3 is not None else cvss2
            
            # Extract CVE (may be multiple, join with commas)
            cve_elements = item.findall("cve")
            cve = ", ".join(_safe_get_text(c) for c in cve_elements) if cve_elements else None
            
            # Create normalized finding
            finding = VAFinding(
                host_ip=host_ip,
                hostname=hostname,
                os=os,
                port=port,
                protocol=protocol,
                service=service,
                severity_text=risk_factor,
                cvss=cvss,
                cve=cve,
                title=plugin_name,
                description=description,
                evidence=plugin_output,
                remediation=solution,
                raw_plugin_id=plugin_id,
                raw_plugin_family=plugin_family
            )
            
            findings.append(finding)
    
    return findings


def parse_csv(csv_path: str) -> list[VAFinding]:
    """
    Parse a CSV vulnerability report into VAFinding objects.
    
    Expected CSV columns (case-insensitive, order doesn't matter):
    - host_ip, hostname, os, port, protocol, service
    - severity (or severity_text), cvss, cve
    - title (or name, vulnerability), description, solution (or remediation)
    - evidence (or plugin_output), plugin_id, plugin_family
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of VAFinding objects
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    findings = []
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Normalize column names (lowercase, strip spaces)
        if reader.fieldnames:
            reader.fieldnames = [col.lower().strip() for col in reader.fieldnames]
        
        for row_num, row in enumerate(reader, start=2):  # Start at 2 (1 is header)
            # Normalize all keys in row
            row = {k.lower().strip(): v.strip() if v else "" for k, v in row.items()}
            
            # Map flexible column names to standard fields
            def get_field(primary: str, *aliases: str) -> str:
                """Get field value trying primary name first, then aliases."""
                if primary in row and row[primary]:
                    return row[primary]
                for alias in aliases:
                    if alias in row and row[alias]:
                        return row[alias]
                return ""
            
            # Extract fields with flexible column mapping
            host_ip = get_field('host_ip', 'ip', 'host', 'target')
            hostname = get_field('hostname', 'host_name', 'fqdn', 'dns_name')
            os = get_field('os', 'operating_system', 'os_name')
            port_str = get_field('port', 'port_number')
            protocol = get_field('protocol', 'proto').lower() or "tcp"
            service = get_field('service', 'service_name', 'svc')
            severity = get_field('severity', 'severity_text', 'risk', 'risk_level')
            cvss_str = get_field('cvss', 'cvss_score', 'cvss_base_score')
            cve = get_field('cve', 'cve_id', 'cve_ids')
            title = get_field('title', 'name', 'vulnerability', 'vuln_name', 'plugin_name')
            description = get_field('description', 'desc', 'synopsis', 'summary')
            remediation = get_field('remediation', 'solution', 'fix', 'recommendation')
            evidence = get_field('evidence', 'plugin_output', 'output', 'proof')
            plugin_id = get_field('plugin_id', 'pluginid', 'id')
            plugin_family = get_field('plugin_family', 'family', 'category')
            
            # Validate required fields
            if not host_ip:
                print(f"[WARNING] Row {row_num}: Missing host_ip, skipping", file=sys.stderr)
                continue
            
            if not title:
                print(f"[WARNING] Row {row_num}: Missing title, skipping", file=sys.stderr)
                continue
            
            # Parse numeric fields
            port = _safe_get_int(port_str, default=0)
            cvss = None
            if cvss_str:
                try:
                    cvss = float(cvss_str)
                except ValueError:
                    pass
            
            # Create finding
            finding = VAFinding(
                host_ip=host_ip,
                hostname=hostname or host_ip,
                os=os or "Unknown",
                port=port,
                protocol=protocol,
                service=service or "unknown",
                severity_text=severity or "None",
                cvss=cvss,
                cve=cve or None,
                title=title,
                description=description or "No description provided",
                evidence=evidence or "N/A",
                remediation=remediation or "No remediation provided",
                raw_plugin_id=plugin_id or "unknown",
                raw_plugin_family=plugin_family or "unknown"
            )
            
            findings.append(finding)
    
    return findings


def validate_finding(finding: VAFinding) -> Tuple[bool, List[str]]:
    """
    Validate a VAFinding against expected schema and constraints.

    Validation checks:
    - Required fields are non-empty
    - Data types are correct
    - Values are within valid ranges
    - Format constraints (e.g., IP addresses, CVE format)

    Args:
        finding: VAFinding object to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Required string fields must not be empty
    required_str_fields = ['host_ip', 'title', 'severity_text']
    for field in required_str_fields:
        value = getattr(finding, field, None)
        if not value or not isinstance(value, str) or not value.strip():
            errors.append(f"Required field '{field}' is missing or empty")

    # Port must be valid range
    if finding.port is not None:
        if not isinstance(finding.port, int):
            errors.append(f"Port must be integer, got {type(finding.port)}")
        elif not (0 <= finding.port <= 65535):
            errors.append(f"Port {finding.port} out of valid range (0-65535)")

    # Protocol must be valid
    if finding.protocol:
        valid_protocols = ['tcp', 'udp', 'icmp', 'sctp', 'unknown']
        if finding.protocol.lower() not in valid_protocols:
            errors.append(f"Invalid protocol '{finding.protocol}', expected one of {valid_protocols}")

    # Severity must be valid
    if finding.severity_text:
        valid_severities = ['Critical', 'High', 'Medium', 'Low', 'None', 'Info']
        if finding.severity_text not in valid_severities:
            errors.append(f"Invalid severity '{finding.severity_text}', expected one of {valid_severities}")

    # CVSS score must be in valid range
    if finding.cvss is not None:
        if not isinstance(finding.cvss, (int, float)):
            errors.append(f"CVSS must be numeric, got {type(finding.cvss)}")
        elif not (0.0 <= finding.cvss <= 10.0):
            errors.append(f"CVSS score {finding.cvss} out of valid range (0.0-10.0)")

    # CVE format validation (if present)
    if finding.cve and finding.cve.strip():
        import re
        # CVE format: CVE-YYYY-NNNN+ (where YYYY is year, NNNN+ is 4+ digits)
        cve_pattern = r'CVE-\d{4}-\d{4,}'
        if not re.match(cve_pattern, finding.cve):
            # Could be multiple CVEs separated by commas
            cves = [c.strip() for c in finding.cve.split(',')]
            invalid_cves = [c for c in cves if not re.match(cve_pattern, c)]
            if invalid_cves:
                errors.append(f"Invalid CVE format: {', '.join(invalid_cves)}")

    # IP address format validation (basic check)
    if finding.host_ip:
        import re
        # Simple IPv4 pattern (basic validation)
        ipv4_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        if not re.match(ipv4_pattern, finding.host_ip):
            # Could be IPv6 or hostname - just warn, don't fail
            pass  # Allow hostnames and IPv6

    return (len(errors) == 0, errors)


def validate_findings_batch(findings: List[VAFinding], fail_on_invalid: bool = False) -> Tuple[List[VAFinding], Dict]:
    """
    Validate a batch of findings and return validation report.

    Args:
        findings: List of VAFinding objects
        fail_on_invalid: If True, raise exception on validation errors

    Returns:
        Tuple of (valid_findings, validation_report)

    Raises:
        ValueError: If fail_on_invalid=True and validation errors found
    """
    valid_findings = []
    invalid_findings = []
    validation_errors = {}

    for i, finding in enumerate(findings):
        is_valid, errors = validate_finding(finding)

        if is_valid:
            valid_findings.append(finding)
        else:
            invalid_findings.append(finding)
            validation_errors[i] = {
                'finding_id': finding.finding_id,
                'host_ip': finding.host_ip,
                'title': finding.title[:50] if finding.title else 'N/A',
                'errors': errors
            }

    report = {
        'total_findings': len(findings),
        'valid_findings': len(valid_findings),
        'invalid_findings': len(invalid_findings),
        'validation_errors': validation_errors
    }

    if fail_on_invalid and invalid_findings:
        error_msg = f"Validation failed: {len(invalid_findings)} invalid findings found\n"
        for idx, err_data in list(validation_errors.items())[:5]:  # Show first 5
            error_msg += f"  Finding {idx}: {err_data['title']} - {err_data['errors']}\n"
        raise ValueError(error_msg)

    return valid_findings, report


def detect_format(file_path: str) -> str:
    """
    Detect file format by inspecting content (not just extension).

    Detection logic:
    1. Check for XML declaration or root elements (Nessus XML)
    2. Check for CSV header patterns
    3. Fallback to extension if content is ambiguous

    Args:
        file_path: Path to report file

    Returns:
        Format string: 'xml', 'csv', or 'unknown'
    """
    path = Path(file_path)

    try:
        # Read first 2KB for format detection
        with open(file_path, 'rb') as f:
            header = f.read(2048)

        # Try to decode as UTF-8
        try:
            header_text = header.decode('utf-8', errors='ignore').strip()
        except:
            header_text = str(header, errors='ignore').strip()

        # Check for XML (Nessus format)
        if any(marker in header_text for marker in [
            '<?xml', '<NessusClientData', '<ReportHost', '<Report '
        ]):
            return 'xml'

        # Check for CSV format (look for common Nessus CSV headers)
        header_lower = header_text.lower()
        if any(marker in header_lower for marker in [
            'plugin id,cve,cvss',
            'host,port,protocol',
            'risk,host,protocol',
            'name,host,port'
        ]):
            # Additional check: does it have comma-separated values?
            first_line = header_text.split('\n')[0] if '\n' in header_text else header_text
            if ',' in first_line:
                return 'csv'

        # Fallback to extension-based detection
        suffix = path.suffix.lower()
        if suffix in ['.xml', '.nessus']:
            return 'xml'
        elif suffix == '.csv':
            return 'csv'

        return 'unknown'

    except Exception as e:
        # On error, fallback to extension
        suffix = path.suffix.lower()
        if suffix in ['.xml', '.nessus']:
            return 'xml'
        elif suffix == '.csv':
            return 'csv'
        return 'unknown'


def parse_report(file_path: str, deduplicate: bool = True, validate: bool = True) -> list[VAFinding]:
    """
    Auto-detect file format and parse vulnerability report.

    Supports:
    - Nessus XML (.nessus, .xml)
    - CSV (.csv)

    Format detection uses content inspection (not just file extension).

    Args:
        file_path: Path to report file
        deduplicate: Whether to deduplicate findings by finding_id (default: True)
        validate: Whether to validate findings against schema (default: True)

    Returns:
        List of VAFinding objects (deduplicated and validated if requested)

    Raises:
        ValueError: If file format is unsupported
    """
    # Detect format by content
    detected_format = detect_format(file_path)

    # Parse based on detected format
    if detected_format == 'xml':
        raw_findings = parse_nessus_xml(file_path)
    elif detected_format == 'csv':
        raw_findings = parse_csv(file_path)
    else:
        path = Path(file_path)
        raise ValueError(
            f"Unsupported or unrecognized file format: {path.suffix}. "
            f"Expected Nessus XML (.xml, .nessus) or CSV (.csv)"
        )

    raw_count = len(raw_findings)

    # Apply deduplication if requested
    if deduplicate:
        findings = deduplicate_findings(raw_findings)
    else:
        findings = raw_findings

    # Apply schema validation if requested
    if validate:
        findings, validation_report = validate_findings_batch(findings, fail_on_invalid=False)

        # Emit validation metrics
        if validation_report['invalid_findings'] > 0:
            print(f"[VALIDATION] Schema Validation:", file=sys.stderr)
            print(f"  Valid findings: {validation_report['valid_findings']}", file=sys.stderr)
            print(f"  Invalid findings: {validation_report['invalid_findings']}", file=sys.stderr)
            print(f"  Validation rate: {validation_report['valid_findings']/validation_report['total_findings']*100:.1f}%", file=sys.stderr)

            # Show sample errors (first 3)
            for idx, err_data in list(validation_report['validation_errors'].items())[:3]:
                print(f"  ⚠ Finding {idx} ({err_data['host_ip']}): {err_data['errors'][0]}", file=sys.stderr)

    # Calculate and emit normalization metrics
    imputed_count = sum(1 for f in findings if f.data_quality.imputed_fields)
    metrics = calculate_normalization_metrics(raw_count, len(findings), imputed_count)

    print(f"[METRICS] Phase 1 Normalization:", file=sys.stderr)
    print(f"  Raw findings: {metrics['raw_findings']}", file=sys.stderr)
    print(f"  Normalized findings: {metrics['normalized_findings']}", file=sys.stderr)
    print(f"  Deduplication removed: {metrics['deduplication_count']}", file=sys.stderr)
    print(f"  Normalization efficiency (η): {metrics['normalization_efficiency']:.3f}", file=sys.stderr)
    print(f"  Imputation rate (λ): {metrics['imputation_rate']:.3f}", file=sys.stderr)

    return findings


def to_dict_list(findings: list[VAFinding]) -> list[dict]:
    """
    Convert a list of VAFinding dataclasses to plain dictionaries.
    
    Useful for JSON serialization or DataFrame conversion.
    
    Args:
        findings: List of VAFinding objects
        
    Returns:
        List of dictionaries with the same structure
    """
    return [asdict(finding) for finding in findings]


def deduplicate_findings(findings: list[VAFinding]) -> list[VAFinding]:
    """
    Deduplicate findings by finding_id, keeping the "best" version per group.
    
    Selection criteria (in order of precedence):
    1. Most recent timestamp
    2. Highest CVSS score (if timestamps equal)
    3. Richest data_quality provenance (most imputed fields indicates more enrichment)
    
    Args:
        findings: List of VAFinding objects (may contain duplicates)
        
    Returns:
        Deduplicated list with one finding per unique finding_id
    """
    from collections import defaultdict
    
    # Group findings by finding_id
    groups: dict[str, list[VAFinding]] = defaultdict(list)
    for finding in findings:
        groups[finding.finding_id].append(finding)
    
    deduplicated = []
    
    for finding_id, group in groups.items():
        if len(group) == 1:
            deduplicated.append(group[0])
        else:
            # Multiple findings with same ID - select best
            best = group[0]
            for candidate in group[1:]:
                # 1. Most recent timestamp
                if candidate.timestamp > best.timestamp:
                    best = candidate
                    continue
                elif candidate.timestamp < best.timestamp:
                    continue
                
                # 2. Highest CVSS (if timestamps equal)
                candidate_cvss = candidate.cvss or 0.0
                best_cvss = best.cvss or 0.0
                if candidate_cvss > best_cvss:
                    best = candidate
                    continue
                elif candidate_cvss < best_cvss:
                    continue
                
                # 3. Richest provenance (more imputed fields = more enriched)
                if len(candidate.data_quality.imputed_fields) > len(best.data_quality.imputed_fields):
                    best = candidate
            
            deduplicated.append(best)
    
    return deduplicated


def calculate_normalization_metrics(raw_count: int, normalized_count: int, imputed_count: int) -> dict:
    """
    Calculate normalization and data quality metrics.
    
    Args:
        raw_count: Number of raw input findings
        normalized_count: Number of normalized output findings
        imputed_count: Number of findings with imputed/enriched fields
        
    Returns:
        Dictionary with keys:
            - normalization_efficiency (η_norm): Compression ratio |output| / |input|
            - imputation_rate (λ_impute): Proportion of findings with enrichment
            - deduplication_count: Number of duplicates removed
    """
    dedup_count = raw_count - normalized_count
    
    return {
        "normalization_efficiency": normalized_count / raw_count if raw_count > 0 else 0.0,
        "imputation_rate": imputed_count / normalized_count if normalized_count > 0 else 0.0,
        "deduplication_count": dedup_count,
        "raw_findings": raw_count,
        "normalized_findings": normalized_count,
        "imputed_findings": imputed_count
    }


def main() -> None:
    """
    Main entry point for testing the parser.
    
    Loads auvap_nessus_25_findings.xml, parses it, and displays summary statistics
    plus the first two findings as formatted JSON.
    """
    xml_file = "auvap_nessus_25_findings.xml"
    
    try:
        print(f"[*] Parsing Nessus XML report: {xml_file}")
        findings = parse_nessus_xml(xml_file)
        
        print(f"[+] Successfully parsed {len(findings)} findings")
        
        if not findings:
            print("[!] No findings found in report")
            return
        
        # Convert to dictionaries for JSON output
        findings_dicts = to_dict_list(findings)
        
        # Display first 2 findings
        print("\n[*] First 2 findings:\n")
        sample = findings_dicts[:2]
        print(json.dumps(sample, indent=2, default=str))
        
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except ET.ParseError as e:
        print(f"[ERROR] XML parsing failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
