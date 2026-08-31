#!/usr/bin/env python3
"""
classifier_v2.py - Multi-Provider LLM Vulnerability Classifier

Improved classifier supporting Google Gemini, GitHub Models, and OpenAI.
Uses official SDKs for better reliability and automatic retry handling.

Adapted to work with VAFinding dataclass from parser.py.
"""

import json
import os
import sys
import time
from typing import Optional, Any
def _extract_json_object(text: str) -> str:
    """Extract first JSON object from text, raising ValueError if none found."""
    stripped = text.strip()
    
    # Try direct parse first
    if stripped.startswith('{') and stripped.endswith('}'):
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass
    
    # Find JSON object boundaries
    start = stripped.find('{')
    if start == -1:
        # Debug: print what we got
        print(f"\n[DEBUG] No JSON found in response. First 200 chars:", file=sys.stderr)
        print(f"[DEBUG] {stripped[:200]}", file=sys.stderr)
        raise ValueError("No JSON object found in response")
    
    # Find matching closing brace
    brace_count = 0
    end = -1
    for i in range(start, len(stripped)):
        if stripped[i] == '{':
            brace_count += 1
        elif stripped[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i
                break
    
    if end == -1:
        print(f"\n[DEBUG] Unclosed JSON object. First 200 chars:", file=sys.stderr)
        print(f"[DEBUG] {stripped[:200]}", file=sys.stderr)
        raise ValueError("No complete JSON object found in response")
    
    candidate = stripped[start:end + 1]
    
    # Validate it's valid JSON
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError as e:
        print(f"\n[DEBUG] Invalid JSON extracted. Error: {e}", file=sys.stderr)
        print(f"[DEBUG] Extracted: {candidate[:200]}", file=sys.stderr)
        raise ValueError(f"Invalid JSON in response: {e}")



def _truncate_few_shot_examples(examples: str, max_chars: int = 1500, max_examples: int = 3) -> str:
    """
    Truncate few-shot examples to fit within token budget.

    Strategy:
    1. Split examples by common delimiters (Example 1:, Example 2:, etc.)
    2. Keep only first N examples (max_examples)
    3. Truncate total length to max_chars if needed

    Args:
        examples: Formatted few-shot examples string
        max_chars: Maximum character limit (approx 375 tokens at 4 chars/token)
        max_examples: Maximum number of examples to include

    Returns:
        Truncated examples string
    """
    if not examples or len(examples) <= max_chars:
        return examples

    # Try to split by example markers
    import re
    # Common patterns: "Example 1:", "Example:", "---", etc.
    example_pattern = r'(?:Example\s+\d+:|Example:|---|\n\n)'
    parts = re.split(example_pattern, examples)

    # Filter empty parts
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) <= 1:
        # No clear delimiters, just truncate
        truncated = examples[:max_chars - 50] + "\n\n[...additional examples truncated for brevity]"
        return truncated

    # Take first max_examples
    selected_parts = parts[:max_examples]

    # Reconstruct with "Example N:" markers
    reconstructed = ""
    for i, part in enumerate(selected_parts, 1):
        reconstructed += f"Example {i}:\n{part}\n\n"

    # Final length check
    if len(reconstructed) > max_chars:
        reconstructed = reconstructed[:max_chars - 50] + "\n[...truncated]"

    return reconstructed.strip()


def build_classification_prompt(finding: dict, business_context: Optional[dict] = None,
                               few_shot_examples: Optional[str] = None,
                               max_example_chars: int = 1500) -> str:
    """
    Build a focused classification prompt for any LLM.

    Args:
        finding: Dictionary from parser.to_dict_list()
        business_context: Optional business rules and environment context
        few_shot_examples: Optional formatted few-shot examples
        max_example_chars: Maximum characters for few-shot examples (default: 1500)

    Returns:
        Prompt string requesting structured JSON response
    """
    cvss_str = f"{finding.get('cvss')}" if finding.get('cvss') is not None else "N/A"

    # Truncate description to 500 chars (Phase 3 requirement)
    description = finding.get('description', '')
    if len(description) > 500:
        description = description[:497] + "..."

    # Truncate evidence to 300 chars
    evidence = finding.get('evidence', '')
    if len(evidence) > 300:
        evidence = evidence[:297] + "..."

    # Build business context section
    context_rules = ""
    if business_context:
        context_rules = "\n\nBUSINESS CONTEXT & RULES:\n"
        if business_context.get('excluded_ports'):
            context_rules += f"- EXCLUDED PORTS: {', '.join(map(str, business_context['excluded_ports']))} (company standard ports, do NOT mark as automation candidates)\n"
        if business_context.get('critical_services'):
            context_rules += f"- CRITICAL SERVICES: {', '.join(business_context['critical_services'])} (prioritize these for automation)\n"
        if business_context.get('environment'):
            context_rules += f"- ENVIRONMENT: {business_context['environment']}\n"
        if business_context.get('custom_notes'):
            context_rules += f"- NOTES: {business_context['custom_notes']}\n"

    # Build few-shot examples section (Phase 3) with truncation
    examples_section = ""
    if few_shot_examples:
        truncated_examples = _truncate_few_shot_examples(few_shot_examples, max_example_chars)
        examples_section = "\n" + truncated_examples + "\n"
    
    prompt = f"""You are a security analyst performing vulnerability triage in a controlled lab environment.
You are NOT launching exploits - only classifying vulnerability data for automated testing feasibility.
{context_rules}
{examples_section}Analyze this vulnerability finding:

HOST: {finding.get('host_ip')} ({finding.get('hostname')})
PORT: {finding.get('port')}/{finding.get('protocol')}
SERVICE: {finding.get('service')}
CVSS: {cvss_str}
TITLE: {finding.get('title')}
DESCRIPTION: {description}
EVIDENCE: {evidence}

Respond with ONLY valid JSON (no markdown, no code blocks):

{{
  "severity_bucket": "<Low|Medium|High|Critical>",
  "attack_vector": "<Network|Adjacent|Local|Physical>",
  "vuln_component": "<brief component name>",
  "exploit_notes": "<1-2 sentence explanation>",
  "automation_candidate": <true|false>,
  "llm_confidence": <0.0-1.0>
}}

Rules:
- automation_candidate: true if remotely testable without credentials (RCE, info disclosure, weak crypto)
- automation_candidate: false if needs auth, social engineering, or is policy-only
- vuln_component: specific software/service vulnerable (e.g., "Apache 2.4.49", "OpenSSH 7.4")
"""
    return prompt


def _classify_with_openai_sdk(finding: dict, api_key: str,
                               base_url: str = "https://api.openai.com/v1",
                               model: str = "gpt-4o-mini",
                               business_context: Optional[dict] = None,
                               few_shot_examples: Optional[str] = None) -> dict:
    """
    Classify using OpenAI SDK (supports OpenAI, GitHub Models, Azure OpenAI).
    
    Args:
        finding: Dictionary from parser
        api_key: API key
        base_url: API endpoint (GitHub: https://models.inference.ai.azure.com)
        model: Model name
        business_context: Optional business rules and environment context
        few_shot_examples: Optional formatted few-shot examples
        
    Returns:
        Classification dict with all required fields
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = build_classification_prompt(finding, business_context, few_shot_examples)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a security analyst. Respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,  # Deterministic
        max_tokens=1200  # Increased further for verbose local models
    )
    
    raw_content = response.choices[0].message.content
    if not raw_content:
        # Empty response - possibly model timeout or context issue
        print(f"      [WARNING] Empty response from model, retrying with shorter prompt...", file=sys.stderr)
        raise ValueError("Empty response from LLM - will retry")
    response_text = raw_content.strip()
    
    # Remove markdown code blocks if present
    if response_text.startswith('```'):
        lines = response_text.split('\n')
        response_text = '\n'.join(lines[1:-1])
        if response_text.startswith('json'):
            response_text = response_text[4:].strip()
    
    try:
        classification = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"      [DEBUG] JSON parse failed: {e}", file=sys.stderr)
        print(f"      [DEBUG] Response preview: {response_text[:300]}", file=sys.stderr)
        print(f"      [DEBUG] Attempting to extract JSON object...", file=sys.stderr)
        cleaned = _extract_json_object(response_text)
        classification = json.loads(cleaned)

    # Validate and fix schema
    is_valid, errors = _validate_classification_schema(classification)
    if not is_valid:
        print(f"      [DEBUG] Schema validation errors: {errors}", file=sys.stderr)
        print(f"      [DEBUG] Attempting to fix schema...", file=sys.stderr)
        classification = _fix_classification_schema(classification, finding)

        # Re-validate after fix
        is_valid_after_fix, errors_after_fix = _validate_classification_schema(classification)
        if not is_valid_after_fix:
            print(f"      [WARNING] Schema still invalid after fix: {errors_after_fix}", file=sys.stderr)

    return classification


def _classify_with_gemini(finding: dict, api_key: str,
                          business_context: Optional[dict] = None,
                          few_shot_examples: Optional[str] = None) -> dict:
    """
    Classify using Google Gemini API.
    
    Args:
        finding: Dictionary from parser
        api_key: Google API key
        business_context: Optional business rules and environment context
        few_shot_examples: Optional formatted few-shot examples
        
    Returns:
        Classification dict with all required fields
    """
    try:
        from google import genai  # type: ignore[import]
        from google.genai import types  # type: ignore[import]
    except ImportError:
        raise ImportError("google-genai package not installed. Run: pip install google-genai")
    
    client = genai.Client(api_key=api_key)
    prompt = build_classification_prompt(finding, business_context, few_shot_examples)
    
    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=400
        )
    )
    
    raw_text = getattr(response, "text", None)
    if not raw_text:
        raise ValueError("Empty response from Gemini")
    response_text = raw_text.strip()
    
    # Remove markdown code blocks if present
    if response_text.startswith('```'):
        lines = response_text.split('\n')
        response_text = '\n'.join(lines[1:-1])
        if response_text.startswith('json'):
            response_text = response_text[4:].strip()
    
    try:
        classification = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"      [DEBUG] Gemini JSON parse failed: {e}", file=sys.stderr)
        print(f"      [DEBUG] Response preview: {response_text[:300]}", file=sys.stderr)
        print(f"      [DEBUG] Attempting to extract JSON object...", file=sys.stderr)
        cleaned = _extract_json_object(response_text)
        classification = json.loads(cleaned)

    # Validate and fix schema
    is_valid, errors = _validate_classification_schema(classification)
    if not is_valid:
        print(f"      [DEBUG] Schema validation errors: {errors}", file=sys.stderr)
        print(f"      [DEBUG] Attempting to fix schema...", file=sys.stderr)
        classification = _fix_classification_schema(classification, finding)

        # Re-validate after fix
        is_valid_after_fix, errors_after_fix = _validate_classification_schema(classification)
        if not is_valid_after_fix:
            print(f"      [WARNING] Schema still invalid after fix: {errors_after_fix}", file=sys.stderr)

    return classification


def _validate_classification_schema(classification: dict) -> tuple[bool, list[str]]:
    """
    Validate classification result against expected schema.

    Required fields and constraints:
    - severity_bucket: Must be one of [Low, Medium, High, Critical, None]
    - attack_vector: Must be one of [Network, Adjacent, Local, Physical]
    - vuln_component: String (required)
    - exploit_notes: String (required)
    - automation_candidate: Boolean (required)
    - llm_confidence: Float 0.0-1.0 (required)

    Args:
        classification: Dict from LLM response

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check required fields exist
    required_fields = [
        'severity_bucket',
        'attack_vector',
        'vuln_component',
        'exploit_notes',
        'automation_candidate',
        'llm_confidence'
    ]

    for field in required_fields:
        if field not in classification:
            errors.append(f"Missing required field: '{field}'")

    # Validate severity_bucket enum
    if 'severity_bucket' in classification:
        valid_severities = ['Low', 'Medium', 'High', 'Critical', 'None']
        if classification['severity_bucket'] not in valid_severities:
            errors.append(
                f"Invalid severity_bucket '{classification['severity_bucket']}', "
                f"must be one of {valid_severities}"
            )

    # Validate attack_vector enum
    if 'attack_vector' in classification:
        valid_vectors = ['Network', 'Adjacent', 'Local', 'Physical']
        if classification['attack_vector'] not in valid_vectors:
            errors.append(
                f"Invalid attack_vector '{classification['attack_vector']}', "
                f"must be one of {valid_vectors}"
            )

    # Validate automation_candidate is boolean
    if 'automation_candidate' in classification:
        if not isinstance(classification['automation_candidate'], bool):
            errors.append(
                f"automation_candidate must be boolean, got {type(classification['automation_candidate']).__name__}"
            )

    # Validate llm_confidence is float in range [0.0, 1.0]
    if 'llm_confidence' in classification:
        confidence = classification['llm_confidence']
        if not isinstance(confidence, (int, float)):
            errors.append(
                f"llm_confidence must be numeric, got {type(confidence).__name__}"
            )
        elif not (0.0 <= confidence <= 1.0):
            errors.append(
                f"llm_confidence must be in range [0.0, 1.0], got {confidence}"
            )

    # Validate string fields are not empty
    string_fields = ['vuln_component', 'exploit_notes']
    for field in string_fields:
        if field in classification:
            if not isinstance(classification[field], str):
                errors.append(f"{field} must be string, got {type(classification[field]).__name__}")
            elif not classification[field].strip():
                errors.append(f"{field} must not be empty")

    return (len(errors) == 0, errors)


def _fix_classification_schema(classification: dict, finding: dict) -> dict:
    """
    Attempt to fix common schema validation errors.

    Fixes applied:
    - Set missing fields to safe defaults
    - Normalize severity_bucket case and values
    - Normalize attack_vector case
    - Clamp llm_confidence to [0.0, 1.0]
    - Convert automation_candidate to boolean

    Args:
        classification: Dict from LLM (possibly invalid)
        finding: Original finding dict (for fallback values)

    Returns:
        Fixed classification dict
    """
    fixed = classification.copy()

    # Fix severity_bucket
    if 'severity_bucket' not in fixed or not fixed['severity_bucket']:
        # Use finding severity as fallback
        severity_text = finding.get('severity_text', 'Medium')
        fixed['severity_bucket'] = severity_text if severity_text in ['Low', 'Medium', 'High', 'Critical', 'None'] else 'Medium'
    else:
        # Normalize case and common variations
        severity = str(fixed['severity_bucket']).strip()
        severity_map = {
            'low': 'Low',
            'medium': 'Medium',
            'high': 'High',
            'critical': 'Critical',
            'none': 'None',
            'info': 'None',
            'informational': 'None'
        }
        fixed['severity_bucket'] = severity_map.get(severity.lower(), severity)

    # Fix attack_vector
    if 'attack_vector' not in fixed or not fixed['attack_vector']:
        fixed['attack_vector'] = 'Network'  # Safe default
    else:
        # Normalize case
        vector = str(fixed['attack_vector']).strip()
        vector_map = {
            'network': 'Network',
            'adjacent': 'Adjacent',
            'local': 'Local',
            'physical': 'Physical',
            'remote': 'Network',  # Common alias
            'local network': 'Adjacent'
        }
        fixed['attack_vector'] = vector_map.get(vector.lower(), vector)

    # Fix vuln_component
    if 'vuln_component' not in fixed or not fixed['vuln_component']:
        fixed['vuln_component'] = finding.get('service', 'unknown')

    # Fix exploit_notes
    if 'exploit_notes' not in fixed or not fixed['exploit_notes']:
        fixed['exploit_notes'] = "Classification from LLM"

    # Fix automation_candidate
    if 'automation_candidate' not in fixed:
        fixed['automation_candidate'] = False  # Conservative default
    else:
        # Convert to boolean if needed
        val = fixed['automation_candidate']
        if isinstance(val, str):
            fixed['automation_candidate'] = val.lower() in ['true', 'yes', '1']
        elif not isinstance(val, bool):
            fixed['automation_candidate'] = bool(val)

    # Fix llm_confidence
    if 'llm_confidence' not in fixed:
        fixed['llm_confidence'] = 0.5  # Neutral confidence
    else:
        # Ensure numeric and clamp to [0.0, 1.0]
        try:
            confidence = float(fixed['llm_confidence'])
            fixed['llm_confidence'] = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            fixed['llm_confidence'] = 0.5

    return fixed


def _heuristic_fallback(finding: dict) -> dict:
    """
    Fallback heuristic classification when LLM fails.
    
    Args:
        finding: Dictionary from parser
        
    Returns:
        Basic classification dict
    """
    cvss = finding.get("cvss")
    severity_text = finding.get("severity_text", "").lower()
    
    # Determine severity bucket
    if cvss is not None:
        if cvss >= 9.0 or severity_text == "critical":
            severity_bucket = "Critical"
        elif cvss >= 7.0:
            severity_bucket = "High"
        elif cvss >= 4.0:
            severity_bucket = "Medium"
        elif cvss > 0:
            severity_bucket = "Low"
        else:
            severity_bucket = "None"
    elif severity_text == "critical":
        severity_bucket = "Critical"
    elif severity_text == "high":
        severity_bucket = "High"
    elif severity_text == "medium":
        severity_bucket = "Medium"
    elif severity_text == "low":
        severity_bucket = "Low"
    else:
        severity_bucket = "None"
    
    return {
        "severity_bucket": severity_bucket,
        "attack_vector": "Network",
        "vuln_component": "unknown",
        "exploit_notes": "Heuristic fallback. Manual review required.",
        "automation_candidate": False,
        "llm_confidence": 0.2
    }


def _perform_classification(finding: dict, provider: str, api_key: str,
                           model: Optional[str], business_context: Optional[dict],
                           few_shot_examples: Optional[str]) -> dict:
    """
    Helper function to perform the actual classification call.

    Extracted to avoid code duplication in retry logic.
    """
    if provider == "github":
        model = model or "gpt-4o-mini"
        return _classify_with_openai_sdk(
            finding, api_key,
            base_url="https://models.inference.ai.azure.com",
            model=model,
            business_context=business_context,
            few_shot_examples=few_shot_examples
        )
    elif provider == "gemini":
        return _classify_with_gemini(finding, api_key, business_context, few_shot_examples)
    elif provider == "openai":
        model = model or "gpt-5-nano"
        return _classify_with_openai_sdk(
            finding, api_key,
            model=model,
            business_context=business_context,
            few_shot_examples=few_shot_examples
        )
    elif provider == "local":
        model = model or "deepseek-r1:14b"
        base_url = os.environ.get('LOCAL_OPENAI_BASE_URL') or "http://localhost:11434/v1"
        return _classify_with_openai_sdk(
            finding, api_key,
            base_url=base_url,
            model=model,
            business_context=business_context,
            few_shot_examples=few_shot_examples
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _is_transient_error(error_msg: str) -> bool:
    """Check if error is transient and should be retried."""
    return any(x in error_msg for x in [
        "503", "UNAVAILABLE", "overloaded",
        "500", "502", "504",
        "timeout", "Timeout",
        "Empty response from LLM - will retry",
        "Connection", "connection",
        "No complete JSON object found",
        "Invalid JSON in response",
        "Unterminated string"
    ])


def _is_rate_limit_error(error_msg: str) -> bool:
    """Check if error is a rate limit error."""
    return "RateLimitReached" in error_msg or "429" in error_msg


def classify_single(finding: dict, provider: str = "auto",
                    api_key: Optional[str] = None,
                    model: Optional[str] = None,
                    business_context: Optional[dict] = None,
                    few_shot_examples: Optional[str] = None,
                    max_retries: int = 3,
                    backoff_base: float = 2.0) -> dict:
    """
    Classify a single finding using specified provider with automatic retry.

    Args:
        finding: Dictionary from parser.to_dict_list()
        provider: "gemini", "github", "openai", "local", or "auto"
        api_key: API key (if None, reads from environment)
        model: Model name (provider-specific)
        business_context: Optional business rules and environment context
        few_shot_examples: Optional formatted few-shot examples (Phase 3)
        max_retries: Maximum number of retry attempts for transient errors (default: 3)
        backoff_base: Base for exponential backoff calculation (default: 2.0)

    Returns:
        Finding dict enriched with classification fields

    Raises:
        RuntimeError: If classification fails after all retries
        ValueError: If provider is unknown or API key is missing
    """
    # Auto-detect provider
    if provider == "auto":
        if os.environ.get('GITHUB_TOKEN'):
            provider = "github"
        elif os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'):
            provider = "gemini"
        elif os.environ.get('OPENAI_API_KEY'):
            provider = "openai"
        elif os.environ.get('LOCAL_OPENAI_BASE_URL'):
            provider = "local"
        else:
            raise RuntimeError("No API key found in environment. Set GITHUB_TOKEN, GEMINI_API_KEY, OPENAI_API_KEY, or LOCAL_OPENAI_BASE_URL")

    # Get API key
    if api_key is None:
        if provider == "github":
            api_key = os.environ.get('GITHUB_TOKEN')
            if not api_key:
                raise RuntimeError("GITHUB_TOKEN not set")
        elif provider == "gemini":
            api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not set")
        elif provider == "openai":
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
        elif provider == "local":
            api_key = os.environ.get('LOCAL_OPENAI_API_KEY') or "local"

    # Attempt classification with retry logic
    last_error = None

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            # Perform classification
            classification = _perform_classification(
                finding, provider, api_key, model,
                business_context, few_shot_examples
            )

            # Success! Merge and return
            enriched = finding.copy()
            enriched.update(classification)

            if attempt > 0:
                print(f"      ✅ Retry {attempt} successful", file=sys.stderr)

            return enriched

        except Exception as e:
            last_error = e
            error_msg = str(e)

            # Check error type
            is_transient = _is_transient_error(error_msg)
            is_rate_limit = _is_rate_limit_error(error_msg)

            # Handle rate limit errors (don't retry)
            if is_rate_limit:
                print(f"\n[ERROR] Rate limit exceeded!", file=sys.stderr)
                print(f"[ERROR] {error_msg[:200]}", file=sys.stderr)
                print(f"\n[!] Cannot continue without LLM classification.", file=sys.stderr)
                print(f"[!] Please wait for rate limit to reset or switch to a different provider.", file=sys.stderr)
                raise RuntimeError(f"Rate limit exceeded: {error_msg}") from e

            # Handle transient errors (retry with backoff)
            if is_transient and attempt < max_retries:
                wait_time = backoff_base ** attempt  # 1s, 2s, 4s for base=2.0

                if attempt == 0:
                    print(f"      ⚠️  Transient error, retrying with exponential backoff...", file=sys.stderr)

                print(f"      Retry {attempt + 1}/{max_retries} in {wait_time:.1f}s... (error: {error_msg[:80]})", file=sys.stderr)
                time.sleep(wait_time)
                continue

            # Non-transient error or exhausted retries
            if is_transient and attempt == max_retries:
                print(f"\n[ERROR] All {max_retries} retries exhausted: {error_msg}", file=sys.stderr)
                print(f"[!] Cannot continue without LLM classification.", file=sys.stderr)
                raise RuntimeError(f"Classification failed after {max_retries} retries: {error_msg}") from e
            else:
                # Non-transient error on first attempt
                print(f"\n[ERROR] LLM classification failed: {error_msg}", file=sys.stderr)
                print(f"[!] Cannot continue without LLM classification.", file=sys.stderr)
                raise RuntimeError(f"Classification failed: {error_msg}") from e

    # Should never reach here, but just in case
    raise RuntimeError(f"Classification failed: {last_error}") from last_error


def classify_findings(findings: list[dict], provider: str = "auto",
                     model: Optional[str] = None,
                     business_context: Optional[dict] = None,
                     enable_few_shot: bool = True,
                     metrics: Optional[Any] = None) -> list[dict]:
    """
    Classify a batch of findings with rate limiting, metrics, and few-shot examples.

    Args:
        findings: List of dictionaries from parser.to_dict_list()
        provider: "gemini", "github", "openai", or "auto"
        model: Model name (provider-specific)
        business_context: Optional business rules and environment context
        enable_few_shot: Enable dynamic few-shot example selection (Phase 3)
        metrics: Optional ClassificationMetrics instance for tracking performance

    Returns:
        List of enriched finding dictionaries
    """
    # Initialize Phase 3 components
    from pathlib import Path

    few_shot_selector = None

    # Use provided metrics or create new one
    try:
        from phase3_enhancements import DynamicFewShotSelector, ClassificationMetrics

        if metrics is None:
            metrics = ClassificationMetrics()

        if enable_few_shot and Path("examples.json").exists():
            try:
                few_shot_selector = DynamicFewShotSelector()
                print("[*] Few-shot examples enabled", file=sys.stderr)
            except ImportError as e:
                print(f"[WARNING] Few-shot selection disabled: {e}", file=sys.stderr)
                few_shot_selector = None
            except Exception as e:
                print(f"[WARNING] Few-shot selector initialization failed: {e}", file=sys.stderr)
                few_shot_selector = None
    except ImportError:
        print("[*] Phase 3 enhancements not available (metrics/few-shot disabled)", file=sys.stderr)
    
    # Detect actual provider
    actual_provider = provider
    if provider == "auto":
        if os.environ.get('GITHUB_TOKEN'):
            actual_provider = "github"
        elif os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'):
            actual_provider = "gemini"
        elif os.environ.get('OPENAI_API_KEY'):
            actual_provider = "openai"
        elif os.environ.get('LOCAL_OPENAI_BASE_URL'):
            actual_provider = "local"
        else:
            print("\n[!] No LLM API key detected in environment. Using offline heuristic classification mode.")
            actual_provider = "heuristic"
    
    provider_names = {
        "github": "GitHub Models",
        "gemini": "Google Gemini",
        "openai": "OpenAI",
        "local": "Local (Ollama/LM Studio)",
        "heuristic": "Offline Rule-Based Heuristics",
        "offline": "Offline Rule-Based Heuristics"
    }
    
    print(f"[*] Classifying with {provider_names.get(actual_provider, actual_provider)}")
    
    enriched = []
    total = len(findings)
    
    for i, finding in enumerate(findings, 1):
        title = finding.get('title', 'Unknown')[:50]
        print(f"[*] Classifying finding {i}/{total}: {title}...")
        
        if actual_provider in ("heuristic", "offline"):
            heuristic_res = _heuristic_fallback(finding)
            merged = {**finding, **heuristic_res}
            if metrics:
                metrics.add_classification(0.001, heuristic_res.get('severity_bucket'), is_valid=True)
            enriched.append(merged)
            continue
        
        # Phase 3: Select few-shot examples if enabled
        few_shot_text = None
        if few_shot_selector:
            try:
                description = finding.get('description', '')
                examples = few_shot_selector.select_examples(description, k=3)
                few_shot_text = few_shot_selector.format_examples_for_prompt(examples)
            except Exception as e:
                print(f"      [WARNING] Few-shot selection failed: {e}", file=sys.stderr)
        
        # Classify with metrics tracking
        start_time = time.time()
        try:
            classified = classify_single(finding, provider=actual_provider, model=model, 
                                       business_context=business_context,
                                       few_shot_examples=few_shot_text)
            latency = time.time() - start_time
            
            # Track metrics
            if metrics:
                severity = classified.get('severity_bucket')
                metrics.add_classification(latency, severity, is_valid=True)
            
            enriched.append(classified)
        except Exception as e:
            latency = time.time() - start_time
            if metrics:
                metrics.add_classification(latency, None, is_valid=False)
            raise  # Re-raise to maintain existing error handling
        
        # Rate limiting: wait between requests
        # Gemini free tier: 15 requests/min = 4s between requests to be safe
        # GitHub/OpenAI: More conservative 0.5s delay
        if i < total:
            if actual_provider == "gemini":
                time.sleep(5)  # 5s = 12 requests/min (safely under 15 RPM limit)
            else:
                time.sleep(0.5)  # 0.5s for GitHub/OpenAI
    
    print(f"[+] Classified {len(enriched)} findings", file=sys.stderr)

    # Phase 3: Print metrics summary (only if metrics was created internally)
    # If metrics was passed from caller, let caller handle printing
    if metrics and 'metrics_printed' not in dir(metrics):
        metrics.metrics_printed = True  # Mark to avoid duplicate printing
        # Metrics summary will be printed by caller (experiment.py)

    return enriched


def main() -> None:
    """Test harness using parser module."""
    print("[*] Testing classifier_v2 module\n")
    
    try:
        import parser
    except ImportError:
        print("[ERROR] Could not import parser.py", file=sys.stderr)
        sys.exit(1)
    
    # Parse sample data
    xml_file = "auvap_nessus_25_findings.xml"
    try:
        findings = parser.parse_nessus_xml(xml_file)
        findings_dicts = parser.to_dict_list(findings)
        print(f"[+] Loaded {len(findings_dicts)} findings\n")
    except Exception as e:
        print(f"[ERROR] Failed to parse XML: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Classify first 3 findings
    sample = findings_dicts[:3]
    enriched = classify_findings(sample)
    
    print("\n[*] Classification results:\n")
    for i, finding in enumerate(enriched, 1):
        print(f"--- Finding {i} ---")
        print(f"Title: {finding['title']}")
        print(f"Severity: {finding.get('severity_bucket', 'N/A')}")
        print(f"Attack Vector: {finding.get('attack_vector', 'N/A')}")
        print(f"Component: {finding.get('vuln_component', 'N/A')}")
        print(f"Automation: {finding.get('automation_candidate', False)}")
        print()


if __name__ == "__main__":
    main()
