# CVSS Calculator Module Documentation

## Overview

The CVSS Calculator module provides dynamic CVSS (Common Vulnerability Scoring System) v3.1 score computation, validation, and enrichment for the AUVAP pipeline. It integrates multiple data sources with intelligent fallback to ensure reliable scoring even when official data is unavailable.

## Features

- **NVD API Integration**: Fetches official CVSS scores from NIST National Vulnerability Database
- **CVSS v3.1 Computation**: Calculates scores from base metrics using official formula
- **LLM-Based Estimation**: Uses AI to estimate scores when CVE data is missing
- **SQLite Caching**: Local cache with 30-day expiry for performance and rate limiting
- **Score Validation**: Compare scanner-reported scores against official NVD data
- **Graceful Degradation**: Multi-tier fallback ensures scores are always available
- **Finding Enrichment**: Automatically add CVSS fields to vulnerability findings

## Installation

### Prerequisites

```powershell
# Install required Python package
pip install requests
```

### Optional: NVD API Key

For higher rate limits (50 req/30s), get a free API key:

1. Visit: https://nvd.nist.gov/developers/request-an-api-key
2. Register and receive key via email
3. Set environment variable:

```powershell
# Windows PowerShell
$env:NVD_API_KEY = "your-api-key-here"

# Or add to .env file
echo "NVD_API_KEY=your-api-key-here" >> .env
```

Without an API key, the module uses public access (10 req/min with 6s delay).

## Quick Start

### Basic Usage

```python
from cvss_calculator import compute_cvss, validate_cvss, enrich_finding_with_cvss

# Compute CVSS for a CVE
result = compute_cvss(cve="CVE-2020-1938")
print(f"Score: {result.cvss_score}")
print(f"Severity: {result.severity}")
print(f"Vector: {result.cvss_vector}")
print(f"Confidence: {result.confidence}")

# Validate a reported score
validation = validate_cvss("CVE-2017-0144", reported_score=8.1)
print(f"Official: {validation['official_score']}")
print(f"Accurate: {validation['is_accurate']}")
print(f"Recommendation: {validation['recommendation']}")

# Enrich a vulnerability finding
finding = {
    "cve": "CVE-2021-41773",
    "title": "Apache HTTP Server Path Traversal",
    "cvss": 7.5
}
enriched = enrich_finding_with_cvss(finding)
print(f"Computed: {enriched['cvss_computed']}")
print(f"Confidence: {enriched['cvss_confidence']}")
```

### Command-Line Interface

```powershell
# Compute CVSS for a CVE
python cvss_calculator.py --cve CVE-2020-1938

# Validate a reported score
python cvss_calculator.py --cve CVE-2017-0144 --validate --score 8.1

# Force refresh (bypass cache)
python cvss_calculator.py --cve CVE-2021-44228 --force-refresh

# Estimate from description (no CVE)
python cvss_calculator.py --description "Remote code execution without authentication"
```

## Integration with AUVAP

### Option 1: Enable During Classification

Modify `classifier_v2.py` to enrich findings before classification:

```python
from cvss_calculator import enrich_finding_with_cvss

def classify_findings(findings, llm_provider, enrich_cvss=True):
    """Classify vulnerability findings with optional CVSS enrichment"""
    
    if enrich_cvss:
        print("[*] Enriching findings with CVSS data...")
        findings = [enrich_finding_with_cvss(f) for f in findings]
    
    # Continue with existing classification logic...
```

### Option 2: Enable in Experiment Pipeline

Modify `experiment.py` to enable CVSS enrichment:

```python
# In main experiment pipeline
classified_findings = classify_findings(
    findings=parsed_findings,
    llm_provider=llm_provider,
    enrich_cvss=True  # Enable CVSS enrichment
)
```

### Output Fields Added

When enabled, the following fields are added to each finding:

| Field | Type | Description |
|-------|------|-------------|
| `cvss_computed` | float | Computed CVSS score (0.0-10.0) |
| `cvss_vector` | string | CVSS v3.1 vector string |
| `cvss_severity` | string | None/Low/Medium/High/Critical |
| `cvss_confidence` | string | high/medium/low/none |
| `cvss_source` | string | nvd_api/computed/llm_estimated/existing_score/cached/unavailable |
| `cvss_validated` | bool | Whether score was validated against NVD |
| `cvss_metrics` | dict | Base metrics (if computed) |

## Algorithm

The CVSS computation follows a 5-tier fallback strategy:

```
1. Cache          → Check SQLite cache (30-day expiry)
2. NVD API        → Query official CVE database
3. Computation    → Calculate from base metrics
4. LLM Estimation → AI-based estimation from description
5. Existing Score → Use scanner-reported score
6. Unavailable    → Return 0.0 with "none" confidence
```

### CVSS v3.1 Formula

```
Base Score = 
  IF Impact <= 0:
    0.0
  ELSE IF Scope Unchanged:
    Roundup(min[(Impact + Exploitability), 10])
  ELSE:
    Roundup(min[1.08 × (Impact + Exploitability), 10])

Impact (Scope Unchanged) = 
  6.42 × ISS

Impact (Scope Changed) = 
  7.52 × (ISS - 0.029) - 3.25 × (ISS - 0.02)^15

ISS = 
  1 - [(1 - Confidentiality) × (1 - Integrity) × (1 - Availability)]

Exploitability = 
  8.22 × AttackVector × AttackComplexity × PrivilegesRequired × UserInteraction
```

## Configuration

### Environment Variables

```bash
# NVD API Configuration
NVD_API_KEY=your-api-key-here      # Optional, increases rate limit

# LLM Provider (for estimation fallback)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:14b
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Cache Configuration

Cache is stored at `cache/cvss_cache.db`:

```python
from cvss_calculator import CVSSCalculator

# Initialize with custom cache path
calc = CVSSCalculator(cache_db_path="custom_cache/cvss.db")

# Clear expired cache entries (30+ days old)
calc._cleanup_cache()
```

### Rate Limiting

- **Without API Key**: 10 requests/minute (6 second delay)
- **With API Key**: 50 requests/30 seconds (0.6 second delay)

Delays are automatically enforced by the module.

## Testing

Run the comprehensive test suite:

```powershell
# Run all tests
python test_cvss.py

# Expected output:
# ✅ NVD API Integration
# ✅ Cache Functionality
# ✅ Score Validation
# ✅ CVSS Computation
# ✅ Graceful Degradation
# ✅ Finding Enrichment
# ✅ Severity Mapping
# ✅ Non-existent CVE
# 
# Passed: 8/8
```

### Test Coverage

- **NVD API Integration**: CVE-2020-1938 (Ghostcat)
- **Cache Hit/Miss**: CVE-2021-41773 (Apache Path Traversal)
- **Score Validation**: CVE-2017-0144 (EternalBlue)
- **Computation**: Network RCE metrics
- **Graceful Degradation**: No CVE with existing score
- **Finding Enrichment**: Sample vulnerability dict
- **Severity Mapping**: 0.0 → 10.0 range
- **Non-existent CVE**: CVE-9999-99999 (fake)

## Examples

### Example 1: Enrich Nmap NSE Findings

```python
from cvss_calculator import enrich_finding_with_cvss

nmap_findings = [
    {
        "host_ip": "10.0.1.5",
        "port": 8009,
        "service": "ajp13",
        "cve": "CVE-2020-1938",
        "cvss": 9.8
    },
    {
        "host_ip": "10.0.1.10",
        "port": 445,
        "service": "microsoft-ds",
        "cve": "CVE-2017-0144",
        "cvss": 8.1
    }
]

enriched = [enrich_finding_with_cvss(f) for f in nmap_findings]

for f in enriched:
    print(f"{f['cve']}: {f['cvss_computed']} ({f['cvss_confidence']})")
    if not f['cvss_validated']:
        print(f"  ⚠️  Discrepancy: Scanner={f['cvss']}, NVD={f['cvss_computed']}")
```

Output:
```
CVE-2020-1938: 9.8 (high)
CVE-2017-0144: 8.8 (high)
  ⚠️  Discrepancy: Scanner=8.1, NVD=8.8
```

### Example 2: Compute CVSS from Metrics

```python
from cvss_calculator import CVSSCalculator, CVSSMetrics

calc = CVSSCalculator()

# Unauthenticated SQL Injection
metrics = CVSSMetrics(
    attack_vector='N',       # Network
    attack_complexity='L',   # Low
    privileges_required='N', # None
    user_interaction='N',    # None
    scope='U',               # Unchanged
    confidentiality='H',     # High
    integrity='H',           # High
    availability='H'         # High
)

score = calc._compute_score_from_metrics(metrics)
vector = metrics.to_vector_string()

print(f"Vector: {vector}")
print(f"Score: {score}")
# Output:
# Vector: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H
# Score: 9.8
```

### Example 3: Batch Processing with Progress

```python
from cvss_calculator import CVSSCalculator
import time

calc = CVSSCalculator()

cves = [
    "CVE-2020-1938",
    "CVE-2021-44228",
    "CVE-2021-41773",
    "CVE-2017-0144"
]

print("Processing CVEs...")
for i, cve in enumerate(cves, 1):
    result = calc.compute_cvss(cve=cve)
    print(f"[{i}/{len(cves)}] {cve}: {result.cvss_score} ({result.source})")
    
    # Respect rate limiting
    if result.source == "nvd_api":
        time.sleep(6)

print("\nCache statistics:")
print(f"Cached entries: {calc._get_cache_size()}")
```

## API Reference

### Functions

#### `compute_cvss(cve, description, existing_score, force_refresh)`

Compute CVSS score with multi-source fallback.

**Parameters:**
- `cve` (str, optional): CVE identifier (e.g., "CVE-2020-1938")
- `description` (str, optional): Vulnerability description for LLM estimation
- `existing_score` (float, optional): Scanner-reported score as fallback
- `force_refresh` (bool, optional): Bypass cache (default: False)

**Returns:** `CVSSResult` object with score, vector, severity, confidence, source, validation status

#### `validate_cvss(cve, reported_score)`

Validate a reported CVSS score against NVD official data.

**Parameters:**
- `cve` (str): CVE identifier
- `reported_score` (float): Score to validate

**Returns:** dict with keys:
- `is_accurate` (bool): Whether scores match (±0.1 tolerance)
- `official_score` (float): NVD official score
- `reported_score` (float): Input score
- `discrepancy` (float): Absolute difference
- `recommendation` (str): Action to take

#### `enrich_finding_with_cvss(finding)`

Add CVSS fields to a vulnerability finding dictionary.

**Parameters:**
- `finding` (dict): Vulnerability finding (must have `cve` or `description`)

**Returns:** dict with added fields: `cvss_computed`, `cvss_vector`, `cvss_severity`, `cvss_confidence`, `cvss_source`, `cvss_validated`, `cvss_metrics`

### Classes

#### `CVSSMetrics`

Dataclass representing CVSS v3.1 base metrics.

**Attributes:**
- `attack_vector` (str): 'N' (Network), 'A' (Adjacent), 'L' (Local), 'P' (Physical)
- `attack_complexity` (str): 'L' (Low), 'H' (High)
- `privileges_required` (str): 'N' (None), 'L' (Low), 'H' (High)
- `user_interaction` (str): 'N' (None), 'R' (Required)
- `scope` (str): 'U' (Unchanged), 'C' (Changed)
- `confidentiality` (str): 'N' (None), 'L' (Low), 'H' (High)
- `integrity` (str): 'N' (None), 'L' (Low), 'H' (High)
- `availability` (str): 'N' (None), 'L' (Low), 'H' (High)

**Methods:**
- `to_vector_string()`: Generate CVSS v3.1 vector string

#### `CVSSResult`

Dataclass representing CVSS computation result.

**Attributes:**
- `cvss_score` (float): Computed score (0.0-10.0)
- `cvss_vector` (str): CVSS v3.1 vector string
- `severity` (str): None/Low/Medium/High/Critical
- `confidence` (str): high/medium/low/none
- `source` (str): nvd_api/computed/llm_estimated/existing_score/cached/unavailable
- `validated` (bool): Whether validated against NVD
- `metrics` (CVSSMetrics, optional): Base metrics if available
- `timestamp` (str): ISO format timestamp

## Troubleshooting

### Issue: "No API key provided"

**Cause:** Accessing NVD API without key (public access)  
**Impact:** Lower rate limit (10 req/min vs 50 req/30s)  
**Solution:** Get free API key from https://nvd.nist.gov/developers/request-an-api-key

### Issue: "Rate limit exceeded"

**Cause:** Too many requests to NVD API  
**Impact:** 403 Forbidden responses  
**Solution:** 
- Wait 60 seconds before retrying
- Enable caching to reduce API calls
- Get API key for higher limits

### Issue: "Cache database locked"

**Cause:** Multiple processes accessing SQLite database simultaneously  
**Impact:** OperationalError exception  
**Solution:**
- Use one CVSSCalculator instance per process
- Implement process-level locking if needed
- Consider Redis for multi-process caching

### Issue: "LLM estimation failed"

**Cause:** LLM provider unavailable or invalid API key  
**Impact:** Falls back to existing score or unavailable  
**Solution:**
- Verify LLM provider is running (check `ollama list`)
- Test LLM connection: `python classifier_v2.py --test`
- Check environment variables (OPENAI_API_KEY, etc.)

### Issue: "CVSS score mismatch"

**Cause:** Scanner uses different CVSS version or scoring method  
**Impact:** Validation shows discrepancy  
**Solution:**
- Check scanner's CVSS version (v2 vs v3.1)
- Use `validate_cvss()` to get official score
- Update scanner if using outdated database

## Performance

### Benchmarks

| Operation | Latency | Notes |
|-----------|---------|-------|
| Cache Hit | ~1ms | SQLite query |
| NVD API Call | ~900ms | Network + processing |
| CVSS Computation | ~0.1ms | Pure calculation |
| LLM Estimation | ~2-5s | Depends on model |
| Finding Enrichment | ~1-900ms | Cache hit vs API call |

### Optimization Tips

1. **Enable Caching**: Cache reduces API calls by 90%+ for repeated CVEs
2. **Batch Processing**: Process findings in groups to amortize delays
3. **Async Operations**: Use threading for independent CVE lookups
4. **API Key**: Reduces delay from 6s to 0.6s per request
5. **Cache Warmup**: Pre-populate cache with common CVEs

## Security Considerations

- **API Keys**: Store in `.env` file, never commit to Git
- **Cache Privacy**: Cache contains CVE metadata (no sensitive scan data)
- **Rate Limiting**: Respect NVD API terms of service
- **Validation**: Always validate critical vulnerabilities against official sources
- **Offline Mode**: System works without internet (uses cache + existing scores)

## Contributing

When modifying the CVSS calculator:

1. Update tests in `test_cvss.py`
2. Run full test suite before committing
3. Update this documentation
4. Follow CVSS v3.1 specification strictly
5. Maintain backward compatibility with existing pipelines

## References

- **CVSS v3.1 Specification**: https://www.first.org/cvss/v3.1/specification-document
- **NVD API Documentation**: https://nvd.nist.gov/developers/vulnerabilities
- **CVSS Calculator (Official)**: https://www.first.org/cvss/calculator/3.1
- **NVD Data Feeds**: https://nvd.nist.gov/vuln/data-feeds

## License

This module is part of the AUVAP project. See main `README.md` for license information.

## Changelog

### Version 1.0.0 (2025-01-30)

- Initial release with NVD API integration
- CVSS v3.1 computation from base metrics
- SQLite caching with 30-day expiry
- LLM-based estimation fallback
- Score validation against official database
- Finding enrichment API
- Comprehensive test suite
- CLI interface for manual testing

---

**For issues or questions, please open a GitHub issue or contact the AUVAP maintainers.**
