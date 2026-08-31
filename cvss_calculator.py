#!/usr/bin/env python3
"""
CVSS Calculator Module for AUVAP
=================================

Provides dynamic CVSS computation, validation, and enrichment using:
1. NVD (National Vulnerability Database) API v2.0
2. Mathematical CVSS v3.1 computation
3. LLM-based estimation for missing data
4. SQLite-based caching for performance

Usage:
    from cvss_calculator import compute_cvss, enrich_finding_with_cvss
    
    result = compute_cvss(
        cve="CVE-2020-1938",
        description="Tomcat AJP file read",
        existing_score=9.8
    )

Author: AUVAP Team
License: Educational Use Only
"""

import os
import sys
import time
import json
import sqlite3
import hashlib
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict

try:
    import requests
except ImportError:
    print("WARNING: requests library not installed. NVD API will not be available.")
    print("Install with: pip install requests")
    requests = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration
NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
CACHE_DB_PATH = Path("cache/cvss_cache.db")
CACHE_EXPIRY_DAYS = 30
RATE_LIMIT_DELAY = 6.0  # seconds between requests (safe limit)


@dataclass
class CVSSMetrics:
    """CVSS v3.1 Base Metrics"""
    attack_vector: str  # N/A/L/P (Network/Adjacent/Local/Physical)
    attack_complexity: str  # L/H (Low/High)
    privileges_required: str  # N/L/H (None/Low/High)
    user_interaction: str  # N/R (None/Required)
    scope: str  # U/C (Unchanged/Changed)
    confidentiality: str  # N/L/H (None/Low/High)
    integrity: str  # N/L/H
    availability: str  # N/L/H
    
    def to_vector_string(self) -> str:
        """Convert metrics to CVSS vector string"""
        return (
            f"CVSS:3.1/"
            f"AV:{self.attack_vector}/"
            f"AC:{self.attack_complexity}/"
            f"PR:{self.privileges_required}/"
            f"UI:{self.user_interaction}/"
            f"S:{self.scope}/"
            f"C:{self.confidentiality}/"
            f"I:{self.integrity}/"
            f"A:{self.availability}"
        )


@dataclass
class CVSSResult:
    """Complete CVSS computation result"""
    cvss_score: float
    cvss_vector: str
    severity: str  # None/Low/Medium/High/Critical
    confidence: str  # high/medium/low/none
    source: str  # nvd_api/computed/llm_estimated/cached/existing_score/unavailable
    validated: bool
    metrics: Optional[CVSSMetrics] = None
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        if self.metrics:
            result['metrics'] = asdict(self.metrics)
        return result


class CVSSCalculator:
    """Main CVSS calculation and validation engine"""
    
    def __init__(self, api_key: Optional[str] = None, llm_provider: str = "auto"):
        """
        Initialize CVSS Calculator
        
        Args:
            api_key: Optional NVD API key for higher rate limits
            llm_provider: LLM provider for estimation (auto/openai/gemini/local)
        """
        self.api_key = api_key or os.environ.get("NVD_API_KEY")
        self.llm_provider = llm_provider
        self.last_api_call = 0
        self._init_cache()
        
        logger.info(f"CVSSCalculator initialized (API key: {'Yes' if self.api_key else 'No'})")
    
    def _init_cache(self):
        """Initialize SQLite cache database"""
        try:
            CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(CACHE_DB_PATH))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cvss_cache (
                    cve TEXT PRIMARY KEY,
                    cvss_data TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()
            logger.info(f"Cache initialized at {CACHE_DB_PATH}")
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
    
    def compute_cvss(
        self,
        cve: Optional[str] = None,
        description: str = "",
        existing_score: Optional[float] = None,
        force_refresh: bool = False
    ) -> CVSSResult:
        """
        Main entry point for CVSS computation
        
        Priority order:
        1. Cache (if not expired and not force_refresh)
        2. NVD API (if CVE provided)
        3. LLM estimation from description
        4. Return existing_score with low confidence
        5. Return unavailable
        
        Args:
            cve: CVE identifier (e.g., "CVE-2020-1938")
            description: Vulnerability description for LLM estimation
            existing_score: Existing CVSS score from scanner
            force_refresh: Skip cache and fetch fresh data
            
        Returns:
            CVSSResult with computed score and metadata
        """
        logger.info(f"Computing CVSS for CVE={cve}, existing_score={existing_score}")
        
        # Check cache first
        if cve and not force_refresh:
            cached = self._get_from_cache(cve)
            if cached:
                logger.info(f"Cache hit for {cve}")
                return cached
        
        # Try NVD API
        if cve and requests:
            nvd_result = self._fetch_from_nvd(cve)
            if nvd_result:
                self._save_to_cache(cve, nvd_result)
                logger.info(f"NVD API success for {cve}: {nvd_result.cvss_score}")
                return nvd_result
        
        # Try LLM estimation
        if description:
            llm_result = self._estimate_with_llm(description, cve)
            if llm_result and llm_result.cvss_score > 0:
                logger.info(f"LLM estimation for {cve or 'unknown'}: {llm_result.cvss_score}")
                return llm_result
        
        # Fallback to existing score
        if existing_score:
            logger.info(f"Using existing score: {existing_score}")
            return CVSSResult(
                cvss_score=existing_score,
                cvss_vector="UNKNOWN",
                severity=self._score_to_severity(existing_score),
                confidence="low",
                source="existing_score",
                validated=False,
                timestamp=datetime.now().isoformat()
            )
        
        # Last resort: return unavailable
        logger.warning(f"Could not compute CVSS for {cve or 'unknown'}")
        return CVSSResult(
            cvss_score=0.0,
            cvss_vector="UNKNOWN",
            severity="None",
            confidence="none",
            source="unavailable",
            validated=False,
            timestamp=datetime.now().isoformat()
        )
    
    def validate_cvss(self, cve: str, reported_score: float) -> Dict[str, Any]:
        """
        Validate reported CVSS score against NVD
        
        Args:
            cve: CVE identifier
            reported_score: CVSS score from scanner
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating {cve}: reported={reported_score}")
        
        result = self.compute_cvss(cve=cve, existing_score=reported_score)
        
        if result.source in ["nvd_api", "cached"]:
            official_score = result.cvss_score
            discrepancy = abs(official_score - reported_score)
            is_accurate = discrepancy < 0.1  # Allow 0.1 point difference
            
            validation = {
                "is_accurate": is_accurate,
                "official_score": official_score,
                "reported_score": reported_score,
                "discrepancy": round(discrepancy, 1),
                "official_vector": result.cvss_vector,
                "recommendation": "Accurate" if is_accurate else f"Update to official NVD score ({official_score})"
            }
        else:
            validation = {
                "is_accurate": None,
                "official_score": None,
                "reported_score": reported_score,
                "discrepancy": None,
                "recommendation": "Unable to validate - NVD data unavailable"
            }
        
        logger.info(f"Validation result: {validation['recommendation']}")
        return validation
    
    def _get_from_cache(self, cve: str) -> Optional[CVSSResult]:
        """Retrieve CVSS data from cache if not expired"""
        try:
            conn = sqlite3.connect(str(CACHE_DB_PATH))
            cursor = conn.execute(
                "SELECT cvss_data, fetched_at FROM cvss_cache WHERE cve = ?",
                (cve,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            cvss_data, fetched_at = row
            fetch_time = datetime.fromisoformat(fetched_at)
            
            # Check if cache expired
            if datetime.now() - fetch_time > timedelta(days=CACHE_EXPIRY_DAYS):
                logger.info(f"Cache expired for {cve}")
                return None
            
            # Deserialize cached data
            data = json.loads(cvss_data)
            data['source'] = 'cached'
            
            # Reconstruct CVSSMetrics if present
            if data.get('metrics'):
                data['metrics'] = CVSSMetrics(**data['metrics'])
            
            return CVSSResult(**data)
            
        except Exception as e:
            logger.error(f"Cache read error for {cve}: {e}")
            return None
    
    def _save_to_cache(self, cve: str, result: CVSSResult):
        """Save CVSS result to cache"""
        try:
            conn = sqlite3.connect(str(CACHE_DB_PATH))
            cvss_data = json.dumps(result.to_dict())
            
            conn.execute(
                "INSERT OR REPLACE INTO cvss_cache (cve, cvss_data, fetched_at) VALUES (?, ?, ?)",
                (cve, cvss_data, datetime.now().isoformat())
            )
            conn.commit()
            conn.close()
            logger.debug(f"Cached {cve}")
        except Exception as e:
            logger.error(f"Cache write error for {cve}: {e}")
    
    def _fetch_from_nvd(self, cve: str) -> Optional[CVSSResult]:
        """
        Fetch CVE data from NVD API with rate limiting
        
        Args:
            cve: CVE identifier
            
        Returns:
            CVSSResult if successful, None otherwise
        """
        if not requests:
            logger.warning("requests library not available")
            return None
        
        try:
            # Rate limiting
            elapsed = time.time() - self.last_api_call
            if elapsed < RATE_LIMIT_DELAY:
                sleep_time = RATE_LIMIT_DELAY - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            
            # Build request
            headers = {}
            if self.api_key:
                headers['apiKey'] = self.api_key
            
            url = f"{NVD_API_BASE}?cveId={cve}"
            logger.debug(f"Fetching {url}")
            
            response = requests.get(url, headers=headers, timeout=10)
            self.last_api_call = time.time()
            
            if response.status_code != 200:
                logger.warning(f"NVD API returned {response.status_code} for {cve}")
                return None
            
            data = response.json()
            
            # Parse NVD response
            if 'vulnerabilities' not in data or len(data['vulnerabilities']) == 0:
                logger.warning(f"No data found for {cve}")
                return None
            
            vuln = data['vulnerabilities'][0]['cve']
            
            # Extract CVSS v3.1 (prefer v3.1 over v3.0)
            cvss_data = None
            if 'metrics' in vuln:
                if 'cvssMetricV31' in vuln['metrics']:
                    cvss_data = vuln['metrics']['cvssMetricV31'][0]['cvssData']
                elif 'cvssMetricV30' in vuln['metrics']:
                    cvss_data = vuln['metrics']['cvssMetricV30'][0]['cvssData']
            
            if not cvss_data:
                logger.warning(f"No CVSS v3.x data for {cve}")
                return None
            
            # Extract metrics
            metrics = CVSSMetrics(
                attack_vector=cvss_data.get('attackVector', 'N')[0],
                attack_complexity=cvss_data.get('attackComplexity', 'L')[0],
                privileges_required=cvss_data.get('privilegesRequired', 'N')[0],
                user_interaction=cvss_data.get('userInteraction', 'N')[0],
                scope=cvss_data.get('scope', 'U')[0],
                confidentiality=cvss_data.get('confidentialityImpact', 'N')[0],
                integrity=cvss_data.get('integrityImpact', 'N')[0],
                availability=cvss_data.get('availabilityImpact', 'N')[0]
            )
            
            return CVSSResult(
                cvss_score=cvss_data.get('baseScore', 0.0),
                cvss_vector=cvss_data.get('vectorString', 'UNKNOWN'),
                severity=cvss_data.get('baseSeverity', 'UNKNOWN'),
                confidence="high",
                source="nvd_api",
                validated=True,
                metrics=metrics,
                timestamp=datetime.now().isoformat()
            )
            
        except requests.exceptions.Timeout:
            logger.error(f"NVD API timeout for {cve}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"NVD API request failed for {cve}: {e}")
            return None
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse NVD response for {cve}: {e}")
            return None
    
    def _estimate_with_llm(self, description: str, cve: Optional[str]) -> Optional[CVSSResult]:
        """
        Estimate CVSS metrics using LLM reasoning
        
        Args:
            description: Vulnerability description
            cve: Optional CVE identifier for context
            
        Returns:
            CVSSResult with estimated score
        """
        if not description or len(description) < 20:
            logger.warning("Description too short for LLM estimation")
            return None
        
        try:
            # Import LLM modules (reuse from classifier_v2.py)
            prompt = self._build_cvss_estimation_prompt(description, cve)
            
            # Try to use existing LLM infrastructure
            try:
                import os
                from openai import OpenAI
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    return None
                client = OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.0
                )
                raw_text = resp.choices[0].message.content or "{}"
                metrics_dict = json.loads(raw_text)
            except Exception as e:
                logger.warning(f"LLM estimation failed: {e}")
                return None
            
            # Parse LLM response
            metrics = CVSSMetrics(
                attack_vector=metrics_dict.get('attack_vector', 'N'),
                attack_complexity=metrics_dict.get('attack_complexity', 'L'),
                privileges_required=metrics_dict.get('privileges_required', 'N'),
                user_interaction=metrics_dict.get('user_interaction', 'N'),
                scope=metrics_dict.get('scope', 'U'),
                confidentiality=metrics_dict.get('confidentiality', 'L'),
                integrity=metrics_dict.get('integrity', 'L'),
                availability=metrics_dict.get('availability', 'L')
            )
            
            # Compute score from metrics
            score = self._compute_score_from_metrics(metrics)
            
            return CVSSResult(
                cvss_score=score,
                cvss_vector=metrics.to_vector_string(),
                severity=self._score_to_severity(score),
                confidence="low",
                source="llm_estimated",
                validated=False,
                metrics=metrics,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"LLM estimation error: {e}")
            return None
    
    def _build_cvss_estimation_prompt(self, description: str, cve: Optional[str]) -> str:
        """Build LLM prompt for CVSS metric estimation"""
        cve_context = f" (CVE: {cve})" if cve else ""
        
        return f"""Analyze this vulnerability{cve_context} and estimate CVSS v3.1 metrics:

Vulnerability Description:
{description}

Return ONLY valid JSON with these exact keys (single letter values only):
{{
  "attack_vector": "N|A|L|P",
  "attack_complexity": "L|H",
  "privileges_required": "N|L|H",
  "user_interaction": "N|R",
  "scope": "U|C",
  "confidentiality": "N|L|H",
  "integrity": "N|L|H",
  "availability": "N|L|H"
}}

Guidelines:
- Network exploitable → AV:N
- Requires authentication → PR:L or PR:H
- User must click link → UI:R
- Data breach → C:H, Data modification → I:H, Service disruption → A:H
- Affects other components → S:C

Return only the JSON, no explanations."""
    
    def _compute_score_from_metrics(self, metrics: CVSSMetrics) -> float:
        """
        Compute CVSS v3.1 base score from metrics using official formula
        
        Reference: https://www.first.org/cvss/specification-document
        """
        # Metric value mappings
        AV_MAP = {'N': 0.85, 'A': 0.62, 'L': 0.55, 'P': 0.2}
        AC_MAP = {'L': 0.77, 'H': 0.44}
        PR_MAP = {
            'U': {'N': 0.85, 'L': 0.62, 'H': 0.27},
            'C': {'N': 0.85, 'L': 0.68, 'H': 0.50}
        }
        UI_MAP = {'N': 0.85, 'R': 0.62}
        IMPACT_MAP = {'N': 0.0, 'L': 0.22, 'H': 0.56}
        
        # Get values
        av = AV_MAP.get(metrics.attack_vector, 0.85)
        ac = AC_MAP.get(metrics.attack_complexity, 0.77)
        pr = PR_MAP[metrics.scope].get(metrics.privileges_required, 0.85)
        ui = UI_MAP.get(metrics.user_interaction, 0.85)
        
        c = IMPACT_MAP.get(metrics.confidentiality, 0.0)
        i = IMPACT_MAP.get(metrics.integrity, 0.0)
        a = IMPACT_MAP.get(metrics.availability, 0.0)
        
        # Compute ISS (Impact Sub-Score)
        iss = 1 - ((1 - c) * (1 - i) * (1 - a))
        
        # Compute Impact
        if metrics.scope == 'U':
            impact = 6.42 * iss
        else:  # Changed scope
            impact = 7.52 * (iss - 0.029) - 3.25 * ((iss - 0.02) ** 15)
        
        # Compute Exploitability
        exploitability = 8.22 * av * ac * pr * ui
        
        # Compute Base Score
        if impact <= 0:
            return 0.0
        
        if metrics.scope == 'U':
            base_score = min(impact + exploitability, 10)
        else:
            base_score = min(1.08 * (impact + exploitability), 10)
        
        # Round up to one decimal
        return round(base_score * 10 + 0.000001) / 10
    
    def _score_to_severity(self, score: float) -> str:
        """Convert CVSS score to severity rating"""
        if score == 0.0:
            return "None"
        elif score < 4.0:
            return "Low"
        elif score < 7.0:
            return "Medium"
        elif score < 9.0:
            return "High"
        else:
            return "Critical"


# Public API functions
def compute_cvss(
    cve: Optional[str] = None,
    description: str = "",
    existing_score: Optional[float] = None,
    force_refresh: bool = False
) -> CVSSResult:
    """
    Convenience function - creates calculator instance and computes
    
    Args:
        cve: CVE identifier
        description: Vulnerability description
        existing_score: Existing CVSS score
        force_refresh: Skip cache
        
    Returns:
        CVSSResult with computed score
    """
    calc = CVSSCalculator()
    return calc.compute_cvss(cve, description, existing_score, force_refresh)


def validate_cvss(cve: str, reported_score: float) -> Dict[str, Any]:
    """
    Convenience function - validates score against NVD
    
    Args:
        cve: CVE identifier
        reported_score: Score to validate
        
    Returns:
        Validation results dictionary
    """
    calc = CVSSCalculator()
    return calc.validate_cvss(cve, reported_score)


def enrich_finding_with_cvss(finding: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich a vulnerability finding with computed/validated CVSS
    
    Args:
        finding: Dictionary from parser.to_dict_list()
    
    Returns:
        Enhanced finding with cvss_computed, cvss_validated, cvss_confidence
    """
    calc = CVSSCalculator()
    
    result = calc.compute_cvss(
        cve=finding.get("cve"),
        description=finding.get("description", ""),
        existing_score=finding.get("cvss")
    )
    
    enriched = finding.copy()
    enriched.update({
        "cvss_computed": result.cvss_score,
        "cvss_vector": result.cvss_vector,
        "cvss_severity": result.severity,
        "cvss_confidence": result.confidence,
        "cvss_source": result.source,
        "cvss_validated": result.validated
    })
    
    return enriched


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="CVSS Calculator CLI")
    parser.add_argument("--cve", help="CVE identifier (e.g., CVE-2020-1938)")
    parser.add_argument("--description", help="Vulnerability description")
    parser.add_argument("--score", type=float, help="Existing CVSS score")
    parser.add_argument("--validate", action="store_true", help="Validate existing score")
    parser.add_argument("--force-refresh", action="store_true", help="Skip cache")
    
    args = parser.parse_args()
    
    if args.validate and args.cve and args.score:
        result = validate_cvss(args.cve, args.score)
        print(json.dumps(result, indent=2))
    elif args.cve or args.description:
        result = compute_cvss(
            cve=args.cve,
            description=args.description or "",
            existing_score=args.score,
            force_refresh=args.force_refresh
        )
        print(json.dumps(result.to_dict(), indent=2))
    else:
        parser.print_help()
