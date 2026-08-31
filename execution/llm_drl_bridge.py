"""
LLM↔DRL Integration Bridge (Priority 1, Item 3)

Orchestrates the LLM→DRL→LLM feedback loop:
- LLM generates initial exploit script
- DRL agent executes with refinement attempts
- On failure, feed execution trace back to LLM
- Store successful attempts in persistent memory
- Retrieve similar past attempts for context
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import tempfile
import time

from parser import VAFinding
from task_manager import ExploitTask
from execution.sandbox_executor import SandboxExecutor, ExecutionResult
from execution.persistent_memory import PersistentMemory


@dataclass
class ScriptGenerationRequest:
    """Request for LLM to generate exploit script"""
    finding: VAFinding
    task: ExploitTask
    similar_attempts: List = None
    refinement_iteration: int = 0
    previous_errors: List[str] = None
    execution_trace: str = ""


@dataclass
class ScriptGenerationResponse:
    """Response from LLM with generated script"""
    script_content: str
    confidence: float
    reasoning: str
    metadata: Dict


class LLMDRLBridge:
    """
    Bridge between LLM (script generation) and DRL (execution/learning).

    Implements iterative refinement loop:
    1. LLM generates initial exploit script
    2. DRL agent attempts execution
    3. If fails, feed error back to LLM for refinement
    4. Repeat up to max iterations
    5. Store successful attempts in memory
    """

    def __init__(self,
                 sandbox_executor: SandboxExecutor,
                 persistent_memory: PersistentMemory,
                 llm_provider: str = "openai",
                 max_refinement_iterations: int = 3,
                 use_memory_context: bool = True,
                 verbose: bool = True):
        """
        Initialize LLM-DRL bridge.

        Args:
            sandbox_executor: Sandbox for execution
            persistent_memory: Memory for history
            llm_provider: LLM provider ("openai", "gemini", "local")
            max_refinement_iterations: Max script refinement attempts
            use_memory_context: Use past attempts as context
            verbose: Print detailed progress
        """
        self.sandbox = sandbox_executor
        self.memory = persistent_memory
        self.llm_provider = llm_provider
        self.max_iterations = max_refinement_iterations
        self.use_memory = use_memory_context
        self.verbose = verbose

        # Initialize LLM client
        self._init_llm_client()

    def _init_llm_client(self):
        """Initialize LLM client by resolving provider and API key."""
        self.api_key = None
        self.base_url = None
        self.model = None

        provider = self.llm_provider

        # Auto-detect provider if set to 'auto'
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
                if self.verbose:
                    print("⚠ No LLM API key found – bridge will use placeholder scripts")
                self.llm_provider = "none"
                return
            self.llm_provider = provider

        # Resolve API key and endpoint per provider
        if provider == "github":
            self.api_key = os.environ.get('GITHUB_TOKEN')
            self.base_url = "https://models.inference.ai.azure.com"
            self.model = "gpt-4o-mini"
        elif provider == "gemini":
            self.api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        elif provider == "openai":
            self.api_key = os.environ.get('OPENAI_API_KEY')
            self.base_url = "https://api.openai.com/v1"
            self.model = "gpt-4o-mini"
        elif provider == "local":
            self.api_key = os.environ.get('LOCAL_OPENAI_API_KEY') or "local"
            self.base_url = os.environ.get('LOCAL_OPENAI_BASE_URL') or "http://localhost:11434/v1"
            self.model = "deepseek-r1:14b"
        else:
            if self.verbose:
                print(f"⚠ Unknown provider '{provider}' – bridge will use placeholder scripts")
            self.llm_provider = "none"
            return

        if not self.api_key:
            if self.verbose:
                print(f"⚠ API key not set for {provider} – bridge will use placeholder scripts")
            self.llm_provider = "none"
            return

        if self.verbose:
            print(f"✓ LLM bridge initialized (provider: {self.llm_provider})")

    def plan_and_execute(self, finding: VAFinding, task: ExploitTask) -> Tuple[bool, ExecutionResult, str]:
        """
        Main LLM→DRL→LLM loop.

        Args:
            finding: Vulnerability finding
            task: Exploitation task

        Returns:
            Tuple of (success, execution_result, final_script)
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"LLM-DRL Bridge: {task.task_id}")
            print(f"{'='*70}")
            print(f"Target: {task.target_host}:{task.target_port}")
            print(f"Vulnerability: {task.vulnerability_id}")
            print(f"CVSS: {task.cvss_score}")

        iteration = 0
        previous_errors: List[str] = []
        execution_trace = ""

        # Try up to max iterations
        while iteration < self.max_iterations:
            iteration += 1

            if self.verbose:
                print(f"\n[Iteration {iteration}/{self.max_iterations}]")

            # Step 1: Generate exploit script via LLM
            script_response = self._generate_script(
                finding=finding,
                task=task,
                iteration=iteration,
                previous_errors=previous_errors,
                execution_trace=execution_trace
            )

            if not script_response or not script_response.script_content:
                if self.verbose:
                    print("  ✗ Failed to generate script")
                continue

            if self.verbose:
                print(f"  ✓ Script generated (confidence: {script_response.confidence:.2f})")

            # Save script to temporary file
            script_path = self._save_script(task.task_id, script_response.script_content, iteration)

            # Step 2: Execute script in sandbox
            if self.verbose:
                print(f"  ⚙ Executing script...")

            exec_result = self.sandbox.execute_task(
                task_id=f"{task.task_id}_iter{iteration}",
                script_path=script_path,
                timeout=30
            )

            # Step 3: Check result
            if exec_result.status == "success" and exec_result.exit_code == 0:
                if self.verbose:
                    print(f"  ✓ Execution successful!")

                # Store successful attempt in memory
                self._store_success(finding, task, script_response.script_content, exec_result)

                return True, exec_result, script_response.script_content

            # Step 4: Collect error information for refinement
            if self.verbose:
                print(f"  ✗ Execution failed")
                if exec_result.stderr:
                    print(f"    Error: {exec_result.stderr[:200]}")

            previous_errors.append(exec_result.stderr)
            execution_trace = f"{exec_result.stdout}\n{exec_result.stderr}"

            # Check for safety violations (abort immediately)
            if exec_result.safety_violations:
                if self.verbose:
                    print(f"  🚨 Safety violations detected: {exec_result.safety_violations}")
                return False, exec_result, script_response.script_content

        # All iterations failed
        if self.verbose:
            print(f"\n  ✗ All {self.max_iterations} refinement attempts failed")

        return False, exec_result, script_response.script_content

    def _generate_script(self,
                        finding: VAFinding,
                        task: ExploitTask,
                        iteration: int,
                        previous_errors: List[str],
                        execution_trace: str) -> Optional[ScriptGenerationResponse]:
        """
        Generate exploit script via LLM.

        Args:
            finding: Vulnerability finding
            task: Exploitation task
            iteration: Current iteration number
            previous_errors: Errors from previous attempts
            execution_trace: Execution trace from previous attempt

        Returns:
            ScriptGenerationResponse or None
        """
        # Build prompt for LLM
        prompt = self._build_script_generation_prompt(
            finding, task, iteration, previous_errors, execution_trace
        )

        # Get similar attempts from memory for context
        context_attempts = []
        if self.use_memory and iteration == 1:
            context_attempts = self.memory.get_similar_attempts(
                finding_type=finding.plugin_name,
                cve=finding.plugin_id,
                service=finding.protocol,
                limit=3,
                successful_only=True
            )

        # Generate script (placeholder - actual LLM call would go here)
        script_content = self._call_llm_for_script(prompt, context_attempts)

        if not script_content:
            return None

        return ScriptGenerationResponse(
            script_content=script_content,
            confidence=0.8,  # Placeholder
            reasoning="Generated based on vulnerability characteristics",
            metadata={'iteration': iteration}
        )

    def _build_script_generation_prompt(self,
                                       finding: VAFinding,
                                       task: ExploitTask,
                                       iteration: int,
                                       previous_errors: List[str],
                                       execution_trace: str) -> str:
        """Build prompt for LLM script generation"""
        prompt = f"""Generate an exploitation script for the following vulnerability:

Target: {task.target_host}:{task.target_port}
Service: {finding.protocol}
Vulnerability: {finding.plugin_name}
CVE: {finding.plugin_id}
CVSS Score: {finding.cvss_base_score}
Severity: {finding.severity}
Description: {finding.synopsis}

Requirements:
- Python 3 script
- Safe and controlled execution
- No destructive operations
- Include error handling
- Return status code 0 on success

"""

        # Add refinement context if this is not the first iteration
        if iteration > 1 and previous_errors:
            prompt += f"""
REFINEMENT ITERATION {iteration}

Previous attempts failed with the following errors:
{chr(10).join(f"- {err[:200]}" for err in previous_errors[-2:])}

Execution trace from last attempt:
{execution_trace[:500]}

Please generate an improved script that addresses these issues.
"""

        return prompt

    def _call_llm_for_script(self, prompt: str, context_attempts: List) -> Optional[str]:
        """
        Call LLM to generate an exploit script.

        Uses the same multi-provider infrastructure as classifier_v2.py.
        Falls back to a placeholder script if no LLM provider is configured.

        Args:
            prompt: Generation prompt
            context_attempts: Similar past attempts for context

        Returns:
            Generated script content or None
        """
        # Add context from similar successful attempts
        if context_attempts:
            context_section = "\n\nSuccessful approaches from similar vulnerabilities:\n"
            for i, attempt in enumerate(context_attempts[:2], 1):
                context_section += f"\nExample {i}:\n{attempt.script_content[:300]}...\n"
            prompt = context_section + prompt

        # ---- Real LLM call (multi-provider) ----
        if self.llm_provider != "none" and self.api_key:
            try:
                response_text = self._query_llm(prompt)
                if response_text:
                    script = self._extract_script_from_response(response_text)
                    if script:
                        return script
                    # If extraction failed, return the raw response as-is
                    # (it might already be a plain script without fences)
                    return response_text
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ LLM call failed ({e}), falling back to placeholder", file=sys.stderr)

        # ---- Fallback placeholder script ----
        return self._placeholder_script()

    def _query_llm(self, prompt: str) -> Optional[str]:
        """
        Send a prompt to the configured LLM provider and return raw text.

        Supports: openai, github, local (OpenAI-compatible), gemini.
        """
        provider = self.llm_provider

        if provider in ("openai", "github", "local"):
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")

            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": ("You are a penetration testing expert. "
                                 "Generate a safe, well-structured Python 3 exploit "
                                 "script based on the user's description. "
                                 "Output ONLY the script inside a ```python code fence.")},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            raw = response.choices[0].message.content
            if not raw:
                raise ValueError("Empty response from LLM")
            return raw.strip()

        elif provider == "gemini":
            try:
                from google import genai  # type: ignore[import]
                from google.genai import types  # type: ignore[import]
            except ImportError:
                raise ImportError("google-genai package not installed. Run: pip install google-genai")

            gemini_client: Any = genai.Client(api_key=self.api_key)
            response_gemini: Any = gemini_client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=2000
                )
            )
            raw = getattr(response_gemini, "text", None)
            if not raw:
                raise ValueError("Empty response from Gemini")
            return str(raw).strip()

        return None

    @staticmethod
    def _extract_script_from_response(response_text: str) -> Optional[str]:
        """
        Extract a Python script from an LLM response that may be wrapped in
        markdown code fences (```python ... ```).
        """
        text = response_text.strip()

        # Try to find a fenced code block
        if '```' in text:
            parts = text.split('```')
            for part in parts[1::2]:  # odd-indexed parts are inside fences
                # Strip optional language identifier (e.g. "python\n")
                lines = part.split('\n', 1)
                if len(lines) == 2:
                    lang_hint = lines[0].strip().lower()
                    body = lines[1]
                    if lang_hint in ('python', 'python3', 'py', ''):
                        return body.strip()
                # Single-line fenced block (unusual but handle gracefully)
                return part.strip()

        # No fences – if it looks like Python, return as-is
        if text.startswith('#!') or 'import ' in text[:200] or 'def ' in text[:500]:
            return text

        return None

    @staticmethod
    def _placeholder_script() -> str:
        """Return a minimal placeholder exploit script."""
        return '''#!/usr/bin/env python3
"""Placeholder exploit script – generated without LLM."""
import sys
import socket

def exploit(target_host, target_port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((target_host, int(target_port)))
        sock.close()
        return True
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    success = exploit("localhost", 80)
    sys.exit(0 if success else 1)
'''

    def _save_script(self, task_id: str, script_content: str, iteration: int) -> str:
        """Save script to a cross-platform temporary file."""
        fd, script_path = tempfile.mkstemp(
            prefix=f"exploit_{task_id}_iter{iteration}_",
            suffix=".py"
        )
        with os.fdopen(fd, 'w') as f:
            f.write(script_content)
        # Set executable permission on Unix; harmless no-op on Windows
        try:
            os.chmod(script_path, 0o755)
        except OSError:
            pass
        return script_path

    def _store_success(self,
                      finding: VAFinding,
                      task: ExploitTask,
                      script_content: str,
                      exec_result: ExecutionResult):
        """Store successful attempt in persistent memory"""
        self.memory.store_outcome(
            finding_type=finding.plugin_name,
            cve=finding.plugin_id,
            service=finding.protocol,
            target_os="unknown",  # TODO: Extract from finding
            script_content=script_content,
            success=True,
            error_message="",
            execution_trace=exec_result.stdout,
            cvss_score=finding.cvss_base_score or 0.0,
            metadata={
                'task_id': task.task_id,
                'target': task.target_host,
                'port': task.target_port,
                'duration': exec_result.duration
            }
        )

        if self.verbose:
            print("  ✓ Success stored in persistent memory")

    def get_script_from_memory(self, finding: VAFinding, task: ExploitTask) -> Optional[str]:
        """
        Try to get a known-working script from memory.

        Args:
            finding: Vulnerability finding
            task: Exploitation task

        Returns:
            Script content or None
        """
        return self.memory.get_best_script(
            finding_type=finding.plugin_name,
            cve=finding.plugin_id,
            service=finding.protocol
        )

    def get_success_rate(self, finding: VAFinding) -> float:
        """
        Get historical success rate for this vulnerability type.

        Args:
            finding: Vulnerability finding

        Returns:
            Success rate (0.0 - 1.0)
        """
        return self.memory.get_success_rate(
            finding_type=finding.plugin_name,
            cve=finding.plugin_id,
            service=finding.protocol
        )
