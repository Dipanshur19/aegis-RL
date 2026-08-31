"""
Persistent Memory System (Priority 1, Item 5)

SQLite-based persistent storage for exploitation history with:
- Store exploitation outcomes across runs
- Retrieve similar past attempts
- Learn from previous successes/failures
- Feed context to LLM for improved script generation
"""

import sqlite3
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ExploitAttempt:
    """Record of a single exploitation attempt"""
    attempt_id: str
    finding_type: str
    cve: str
    service: str
    target_os: str
    script_hash: str
    script_content: str
    success: bool
    error_message: str
    execution_trace: str
    timestamp: str
    cvss_score: float
    metadata: Dict


class PersistentMemory:
    """
    Persistent memory for storing and retrieving exploitation history.

    Database Schema:
    - attempts: Main exploitation attempt records
    - scripts: Exploit scripts (deduplicated by hash)
    - outcomes: Success/failure statistics
    """

    def __init__(self, db_path: str = "data/persistent_memory.db"):
        """
        Initialize persistent memory.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema"""
        cursor = self.conn.cursor()

        # Attempts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                finding_type TEXT NOT NULL,
                cve TEXT,
                service TEXT,
                target_os TEXT,
                script_hash TEXT NOT NULL,
                success INTEGER NOT NULL,
                error_message TEXT,
                execution_trace TEXT,
                timestamp TEXT NOT NULL,
                cvss_score REAL,
                metadata TEXT
            )
        """)

        # Scripts table (deduplicated)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scripts (
                script_hash TEXT PRIMARY KEY,
                script_content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                used_count INTEGER DEFAULT 0
            )
        """)

        # Outcomes table (aggregated statistics)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                finding_type TEXT,
                cve TEXT,
                service TEXT,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                total_attempts INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                last_updated TEXT,
                PRIMARY KEY (finding_type, cve, service)
            )
        """)

        # Indices for fast lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_finding_type ON attempts(finding_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cve ON attempts(cve)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_service ON attempts(service)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_success ON attempts(success)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON attempts(timestamp)")

        self.conn.commit()

    def store_outcome(self,
                     finding_type: str,
                     cve: str,
                     service: str,
                     target_os: str,
                     script_content: str,
                     success: bool,
                     error_message: str = "",
                     execution_trace: str = "",
                     cvss_score: float = 0.0,
                     metadata: Optional[Dict] = None) -> str:
        """
        Store an exploitation attempt outcome.

        Args:
            finding_type: Type of vulnerability
            cve: CVE identifier
            service: Target service
            target_os: Target operating system
            script_content: Exploit script content
            success: Whether exploitation succeeded
            error_message: Error message if failed
            execution_trace: Execution trace/logs
            cvss_score: CVSS score
            metadata: Additional metadata

        Returns:
            attempt_id: Unique attempt identifier
        """
        cursor = self.conn.cursor()

        # Generate IDs and hashes
        attempt_id = self._generate_attempt_id(finding_type, cve, service)
        script_hash = self._hash_script(script_content)
        timestamp = datetime.now().isoformat()

        # Store script (if not exists)
        cursor.execute("""
            INSERT OR IGNORE INTO scripts (script_hash, script_content, created_at)
            VALUES (?, ?, ?)
        """, (script_hash, script_content, timestamp))

        # Update script usage count
        cursor.execute("""
            UPDATE scripts SET used_count = used_count + 1
            WHERE script_hash = ?
        """, (script_hash,))

        # Store attempt
        cursor.execute("""
            INSERT INTO attempts (
                attempt_id, finding_type, cve, service, target_os,
                script_hash, success, error_message, execution_trace,
                timestamp, cvss_score, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            attempt_id, finding_type, cve, service, target_os,
            script_hash, int(success), error_message, execution_trace,
            timestamp, cvss_score, json.dumps(metadata or {})
        ))

        # Update outcomes statistics
        self._update_outcomes(finding_type, cve, service, success, timestamp)

        self.conn.commit()

        return attempt_id

    def _update_outcomes(self, finding_type: str, cve: str, service: str, success: bool, timestamp: str):
        """Update aggregated outcomes statistics"""
        cursor = self.conn.cursor()

        # Get current statistics
        cursor.execute("""
            SELECT success_count, failure_count, total_attempts
            FROM outcomes
            WHERE finding_type = ? AND cve = ? AND service = ?
        """, (finding_type, cve, service))

        row = cursor.fetchone()

        if row:
            success_count = row['success_count'] + (1 if success else 0)
            failure_count = row['failure_count'] + (0 if success else 1)
            total_attempts = row['total_attempts'] + 1
        else:
            success_count = 1 if success else 0
            failure_count = 0 if success else 1
            total_attempts = 1

        success_rate = success_count / total_attempts if total_attempts > 0 else 0.0

        # Update or insert
        cursor.execute("""
            INSERT OR REPLACE INTO outcomes (
                finding_type, cve, service, success_count, failure_count,
                total_attempts, success_rate, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (finding_type, cve, service, success_count, failure_count,
              total_attempts, success_rate, timestamp))

    def get_similar_attempts(self,
                            finding_type: str,
                            cve: Optional[str] = None,
                            service: Optional[str] = None,
                            target_os: Optional[str] = None,
                            limit: int = 10,
                            successful_only: bool = False) -> List[ExploitAttempt]:
        """
        Retrieve similar past exploitation attempts.

        Args:
            finding_type: Type of vulnerability
            cve: CVE identifier (optional)
            service: Target service (optional)
            target_os: Target OS (optional)
            limit: Maximum number of results
            successful_only: Return only successful attempts

        Returns:
            List of ExploitAttempt objects
        """
        cursor = self.conn.cursor()

        # Build query
        query = """
            SELECT a.*, s.script_content
            FROM attempts a
            JOIN scripts s ON a.script_hash = s.script_hash
            WHERE a.finding_type = ?
        """
        params: List[Any] = [finding_type]

        if cve:
            query += " AND a.cve = ?"
            params.append(cve)

        if service:
            query += " AND a.service = ?"
            params.append(service)

        if target_os:
            query += " AND a.target_os = ?"
            params.append(target_os)

        if successful_only:
            query += " AND a.success = 1"

        query += " ORDER BY a.timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Convert to ExploitAttempt objects
        attempts = []
        for row in rows:
            attempts.append(ExploitAttempt(
                attempt_id=row['attempt_id'],
                finding_type=row['finding_type'],
                cve=row['cve'],
                service=row['service'],
                target_os=row['target_os'],
                script_hash=row['script_hash'],
                script_content=row['script_content'],
                success=bool(row['success']),
                error_message=row['error_message'],
                execution_trace=row['execution_trace'],
                timestamp=row['timestamp'],
                cvss_score=row['cvss_score'],
                metadata=json.loads(row['metadata'])
            ))

        return attempts

    def get_success_rate(self, finding_type: str, cve: str, service: str) -> float:
        """
        Get success rate for a specific vulnerability type.

        Args:
            finding_type: Type of vulnerability
            cve: CVE identifier
            service: Target service

        Returns:
            Success rate (0.0 - 1.0)
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT success_rate
            FROM outcomes
            WHERE finding_type = ? AND cve = ? AND service = ?
        """, (finding_type, cve, service))

        row = cursor.fetchone()
        return row['success_rate'] if row else 0.0

    def get_best_script(self, finding_type: str, cve: str, service: str) -> Optional[str]:
        """
        Get the most successful script for a vulnerability.

        Args:
            finding_type: Type of vulnerability
            cve: CVE identifier
            service: Target service

        Returns:
            Script content or None
        """
        attempts = self.get_similar_attempts(
            finding_type, cve, service,
            limit=1, successful_only=True
        )

        return attempts[0].script_content if attempts else None

    def get_statistics(self) -> Dict:
        """Get overall memory statistics"""
        cursor = self.conn.cursor()

        stats = {}

        # Total attempts
        cursor.execute("SELECT COUNT(*) as count FROM attempts")
        stats['total_attempts'] = cursor.fetchone()['count']

        # Successful attempts
        cursor.execute("SELECT COUNT(*) as count FROM attempts WHERE success = 1")
        stats['successful_attempts'] = cursor.fetchone()['count']

        # Success rate
        if stats['total_attempts'] > 0:
            stats['overall_success_rate'] = stats['successful_attempts'] / stats['total_attempts']
        else:
            stats['overall_success_rate'] = 0.0

        # Unique CVEs
        cursor.execute("SELECT COUNT(DISTINCT cve) as count FROM attempts")
        stats['unique_cves'] = cursor.fetchone()['count']

        # Unique scripts
        cursor.execute("SELECT COUNT(*) as count FROM scripts")
        stats['unique_scripts'] = cursor.fetchone()['count']

        # Most successful vulnerability types
        cursor.execute("""
            SELECT finding_type, success_rate, total_attempts
            FROM outcomes
            ORDER BY success_rate DESC, total_attempts DESC
            LIMIT 5
        """)
        stats['top_vulnerabilities'] = [dict(row) for row in cursor.fetchall()]

        return stats

    def _generate_attempt_id(self, finding_type: str, cve: str, service: str) -> str:
        """Generate unique attempt ID"""
        data = f"{finding_type}_{cve}_{service}_{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _hash_script(self, script_content: str) -> str:
        """Generate hash of script content"""
        return hashlib.sha256(script_content.encode()).hexdigest()

    def close(self):
        """Close database connection"""
        self.conn.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
