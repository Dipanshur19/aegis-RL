"""
Automatic Knowledge Graph Builder from AUVAP Pipeline

Builds a cyber knowledge graph automatically from:
1. LLM-classified vulnerability report (experiment_report.json)
2. LLM-generated exploit scripts (exploits_manifest.json)

No manual configuration needed - fully automated!

Usage:
    python build_knowledge_graph.py \
        --experiment-report results/experiment_report_20251111_023313.json \
        --exploits-manifest exploits/exploits_20251111_023313/exploits_manifest.json \
        --output knowledge_graphs/kg_20251111_023313.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Optional
import networkx as nx
from enum import Enum


class RelationType(Enum):
    """Edge types in the knowledge graph"""
    VULNERABLE_TO = "VULNERABLE_TO"
    REQUIRES = "REQUIRES"
    ENABLES = "ENABLES"
    GRANTS = "GRANTS"
    CONNECTS_TO = "CONNECTS_TO"


class CapabilityType(Enum):
    """Capabilities that exploits can grant"""
    INITIAL_ACCESS = "initial_access"
    ROOT_SHELL = "root_shell"
    USER_SHELL = "user_shell"
    CREDENTIALS = "credentials"
    LATERAL_MOVEMENT = "lateral_movement"
    DATA_ACCESS = "data_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class AutoKnowledgeGraphBuilder:
    """
    Automatically builds knowledge graph from AUVAP pipeline outputs
    
    Intelligence Sources:
    1. Experiment Report:
       - Hosts, services, vulnerabilities
       - CVSS scores, feasibility ratings
       - Network topology (inferred from IPs)
    
    2. Exploits Manifest:
       - Generated scripts (proof of exploitability)
       - Script languages (Python/PowerShell/Bash)
       - Safety constraints
    
    3. CVE Intelligence:
       - Attack patterns (RCE, SQLi, XSS, etc.)
       - Exploit requirements (auth, network access)
       - Granted capabilities (shell, creds, data)
    """
    
    # CVE Pattern Recognition (expandable)
    CVE_PATTERNS = {
        # Remote Code Execution → Root/User Shell
        'rce_patterns': {
            'keywords': ['remote code execution', 'rce', 'code injection', 
                        'command injection', 'arbitrary code'],
            'grants': CapabilityType.ROOT_SHELL,
            'requires_auth': False,
            'enables_lateral': True
        },
        
        # Memory Corruption → Shell Access
        'memory_corruption': {
            'keywords': ['buffer overflow', 'heap overflow', 'use after free',
                        'memory corruption', 'stack overflow'],
            'grants': CapabilityType.ROOT_SHELL,
            'requires_auth': False,
            'enables_lateral': True
        },
        
        # Authentication Bypass → Credentials
        'auth_bypass': {
            'keywords': ['authentication bypass', 'auth bypass', 'credential',
                        'password disclosure', 'hardcoded credential'],
            'grants': CapabilityType.CREDENTIALS,
            'requires_auth': False,
            'enables_lateral': False
        },
        
        # Path Traversal → Data Access
        'path_traversal': {
            'keywords': ['path traversal', 'directory traversal', 
                        'file disclosure', 'information disclosure'],
            'grants': CapabilityType.DATA_ACCESS,
            'requires_auth': False,
            'enables_lateral': False
        },
        
        # Privilege Escalation
        'privesc': {
            'keywords': ['privilege escalation', 'elevation of privilege',
                        'gain elevated privileges', 'root access'],
            'grants': CapabilityType.PRIVILEGE_ESCALATION,
            'requires_auth': True,
            'enables_lateral': False
        },
        
        # SQL Injection → Data + Potential Shell
        'sqli': {
            'keywords': ['sql injection', 'sqli', 'database injection'],
            'grants': CapabilityType.DATA_ACCESS,
            'requires_auth': False,
            'enables_lateral': False
        }
    }
    
    # Known High-Impact CVEs (automatically grant capabilities)
    KNOWN_CVES = {
        'CVE-2020-1938': {  # Apache Tomcat Ghostcat
            'grants': CapabilityType.ROOT_SHELL,
            'requires_auth': False,
            'enables_lateral': True
        },
        'CVE-2017-0144': {  # EternalBlue
            'grants': CapabilityType.ROOT_SHELL,
            'requires_auth': False,
            'enables_lateral': True
        },
        'CVE-2014-6271': {  # Shellshock
            'grants': CapabilityType.ROOT_SHELL,
            'requires_auth': False,
            'enables_lateral': True
        },
        'CVE-2017-7269': {  # IIS WebDAV
            'grants': CapabilityType.USER_SHELL,
            'requires_auth': True,
            'enables_lateral': False
        },
        'CVE-2021-41773': {  # Apache Path Traversal
            'grants': CapabilityType.DATA_ACCESS,
            'requires_auth': False,
            'enables_lateral': False
        },
        'CVE-2021-3711': {  # OpenSSL
            'grants': CapabilityType.CREDENTIALS,
            'requires_auth': False,
            'enables_lateral': False
        }
    }
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.hosts: Set[str] = set()
        self.vulnerabilities: Set[str] = set()
        self.capabilities: Set[str] = set()
        
        print("\n" + "="*70)
        print("AUTOMATIC KNOWLEDGE GRAPH BUILDER")
        print("="*70)
    
    def build_from_auvap_pipeline(
        self,
        experiment_report_path: Path,
        exploits_manifest_path: Path
    ) -> nx.DiGraph:
        """
        Main method: Build KG from AUVAP pipeline outputs
        
        Args:
            experiment_report_path: Path to experiment_report.json
            exploits_manifest_path: Path to exploits_manifest.json
        
        Returns:
            NetworkX DiGraph with complete knowledge graph
        """
        print(f"\n[Step 1] Loading experiment report: {experiment_report_path.name}")
        with open(experiment_report_path) as f:
            exp_report = json.load(f)
        
        print(f"[Step 2] Loading exploits manifest: {exploits_manifest_path.name}")
        with open(exploits_manifest_path) as f:
            exploits_manifest = json.load(f)
        
        # Extract vulnerabilities from experiment report
        vulnerabilities = exp_report.get('feasible_findings_detailed', [])
        print(f"[Step 3] Found {len(vulnerabilities)} feasible vulnerabilities")
        
        # Extract exploit scripts from manifest
        exploit_scripts = exploits_manifest.get('manifests', [])
        print(f"[Step 4] Found {len(exploit_scripts)} generated exploit scripts")
        
        # Build the graph
        print("\n[Step 5] Building knowledge graph...")
        self._add_hosts(vulnerabilities)
        self._add_vulnerabilities(vulnerabilities)
        self._add_capabilities(vulnerabilities)
        self._link_vulnerabilities_to_capabilities(vulnerabilities)
        self._add_exploit_scripts(exploit_scripts)
        self._infer_dependencies(vulnerabilities)
        self._model_network_topology(vulnerabilities)
        
        print("\n[Step 6] Knowledge graph construction complete!")
        self._print_summary()
        
        return self.graph
    
    def _add_hosts(self, vulnerabilities: List[Dict]):
        """Extract and add all hosts from vulnerabilities"""
        # Get unique hosts
        hosts = set()
        for vuln in vulnerabilities:
            host = vuln.get('host_ip')
            if host:
                hosts.add(host)
        
        print(f"  → Adding {len(hosts)} hosts to graph...")
        
        for host in hosts:
            # All hosts initially accessible (external scanner position)
            self.graph.add_node(host, **{
                'type': 'host',
                'ip': host,
                'accessible': True,
                'compromised': False,
                'has_root': False
            })
            self.hosts.add(host)
    
    def _add_vulnerabilities(self, vulnerabilities: List[Dict]):
        """Add vulnerability nodes and link to hosts"""
        print(f"  → Adding {len(vulnerabilities)} vulnerabilities...")
        
        for vuln in vulnerabilities:
            cve = vuln.get('cve') or 'NO_CVE'
            host = vuln.get('host_ip')
            port = vuln.get('port', 0)
            
            # Create unique vulnerability ID
            vuln_id = f"{cve}@{host}:{port}"
            
            # Add vulnerability node
            self.graph.add_node(vuln_id, **{
                'type': 'vulnerability',
                'cve': cve,
                'host': host,
                'port': port,
                'cvss': vuln.get('cvss', 5.0),
                'severity': vuln.get('severity_bucket', 'Medium'),
                'title': vuln.get('title', ''),
                'service': vuln.get('service', ''),
                'requires_auth': self._detect_auth_requirement(vuln),
                'patched': False
            })
            self.vulnerabilities.add(vuln_id)
            
            # Link host to vulnerability (VULNERABLE_TO)
            self.graph.add_edge(host, vuln_id, 
                              relation=RelationType.VULNERABLE_TO.value)
    
    def _detect_auth_requirement(self, vuln: Dict) -> bool:
        """
        Detect if vulnerability requires authentication
        
        Uses:
        - CVE database lookup
        - Title/description keyword analysis
        - Service type inference
        """
        cve = vuln.get('cve')
        
        # Check known CVEs
        if cve and cve in self.KNOWN_CVES:
            return self.KNOWN_CVES[cve]['requires_auth']
        
        # Check keywords in title/description
        title = vuln.get('title', '').lower()
        description = vuln.get('exploit_notes', '').lower()
        text = f"{title} {description}"
        
        auth_keywords = [
            'authenticated', 'requires authentication', 'after authentication',
            'logged in', 'valid credentials', 'authenticated user'
        ]
        
        return any(keyword in text for keyword in auth_keywords)
    
    def _add_capabilities(self, vulnerabilities: List[Dict]):
        """Add capability nodes for each host"""
        print(f"  → Adding capabilities for {len(self.hosts)} hosts...")
        
        for host in self.hosts:
            # Add each capability type for the host
            for cap_type in CapabilityType:
                cap_id = f"{cap_type.value}@{host}"
                
                self.graph.add_node(cap_id, **{
                    'type': 'capability',
                    'capability': cap_type.value,
                    'host': host,
                    'achieved': False
                })
                self.capabilities.add(cap_id)
    
    def _link_vulnerabilities_to_capabilities(self, vulnerabilities: List[Dict]):
        """
        Create GRANTS edges: Vulnerability → Capability
        
        Uses intelligent pattern matching:
        1. Known CVE database lookup
        2. Title/description keyword analysis
        3. CVSS-based inference
        """
        print(f"  → Linking vulnerabilities to capabilities...")
        
        for vuln in vulnerabilities:
            cve = vuln.get('cve')
            host = vuln.get('host_ip')
            port = vuln.get('port', 0)
            vuln_id = f"{cve}@{host}:{port}"
            
            # Determine granted capability
            granted_cap, enables_lateral = self._infer_granted_capability(vuln)
            
            # Add GRANTS edge
            cap_id = f"{granted_cap.value}@{host}"
            self.graph.add_edge(vuln_id, cap_id,
                              relation=RelationType.GRANTS.value,
                              probability=0.85)
            
            # If enables lateral movement, also grant that capability
            if enables_lateral:
                lat_cap_id = f"{CapabilityType.LATERAL_MOVEMENT.value}@{host}"
                self.graph.add_edge(vuln_id, lat_cap_id,
                                  relation=RelationType.GRANTS.value,
                                  probability=0.9)
    
    def _infer_granted_capability(self, vuln: Dict) -> tuple:
        """
        Infer what capability a vulnerability grants
        
        Returns: (CapabilityType, enables_lateral_movement: bool)
        """
        cve = vuln.get('cve')
        
        # Check known CVEs first
        if cve and cve in self.KNOWN_CVES:
            cve_data = self.KNOWN_CVES[cve]
            return cve_data['grants'], cve_data['enables_lateral']
        
        # Pattern matching on title/description
        title = vuln.get('title', '').lower()
        description = vuln.get('exploit_notes', '').lower()
        text = f"{title} {description}"
        
        for pattern_name, pattern_data in self.CVE_PATTERNS.items():
            keywords = pattern_data['keywords']
            if any(keyword in text for keyword in keywords):
                return pattern_data['grants'], pattern_data['enables_lateral']
        
        # Fallback: CVSS-based inference
        cvss = vuln.get('cvss', 5.0)
        if cvss >= 9.0:
            return CapabilityType.ROOT_SHELL, True
        elif cvss >= 7.0:
            return CapabilityType.USER_SHELL, False
        else:
            return CapabilityType.DATA_ACCESS, False
    
    def _add_exploit_scripts(self, exploit_scripts: List[Dict]):
        """
        Enhance vulnerability nodes with exploit script metadata
        
        Shows that exploit is actually available (not just theoretical)
        """
        print(f"  → Adding exploit script metadata...")
        
        for script_data in exploit_scripts:
            cve = script_data.get('cve')
            
            # Parse target to get host and port
            target = script_data.get('target', '')
            if ':' in target:
                host, port = target.split(':')
                port = int(port)
            else:
                host = target
                port = 0
            
            vuln_id = f"{cve}@{host}:{port}"
            
            if vuln_id in self.vulnerabilities:
                # Add script metadata to vulnerability node
                vuln_node = self.graph.nodes[vuln_id]
                vuln_node['has_exploit'] = True
                vuln_node['script_path'] = script_data.get('script_path')
                vuln_node['script_language'] = self._detect_language(
                    script_data.get('script_path', '')
                )
                vuln_node['llm_generated'] = script_data.get('llm_generated', True)
                vuln_node['requires_review'] = script_data.get('requires_review', True)
    
    def _detect_language(self, script_path: str) -> str:
        """Detect script language from file extension"""
        if script_path.endswith('.py'):
            return 'python'
        elif script_path.endswith('.ps1'):
            return 'powershell'
        elif script_path.endswith('.sh'):
            return 'bash'
        return 'unknown'
    
    def _infer_dependencies(self, vulnerabilities: List[Dict]):
        """
        Infer exploit dependencies based on patterns
        
        Dependency Rules:
        1. Lateral movement: Non-accessible hosts REQUIRE lateral_movement
        2. Privilege escalation: REQUIRES user_shell or initial_access
        3. Authenticated exploits: REQUIRE credentials
        4. Same host: High-priority exploits ENABLE lower-priority ones
        """
        print(f"  → Inferring exploit dependencies...")
        
        # Group vulnerabilities by host
        vulns_by_host = {}
        for vuln in vulnerabilities:
            host = vuln.get('host_ip')
            if host not in vulns_by_host:
                vulns_by_host[host] = []
            vulns_by_host[host].append(vuln)
        
        # For each host, create ENABLES relationships
        for host, host_vulns in vulns_by_host.items():
            # Sort by CVSS (highest priority first)
            host_vulns.sort(key=lambda v: v.get('cvss', 0), reverse=True)
            
            if len(host_vulns) >= 2:
                # First exploit enables others on same host
                primary = host_vulns[0]
                primary_cve = primary.get('cve')
                primary_port = primary.get('port', 0)
                primary_id = f"{primary_cve}@{host}:{primary_port}"
                
                # Determine what primary grants
                granted_cap, _ = self._infer_granted_capability(primary)
                cap_id = f"{granted_cap.value}@{host}"
                
                # Secondary exploits ENABLED by primary's capability
                for secondary in host_vulns[1:]:
                    secondary_cve = secondary.get('cve')
                    secondary_port = secondary.get('port', 0)
                    secondary_id = f"{secondary_cve}@{host}:{secondary_port}"
                    
                    # Add ENABLES relationship
                    self.graph.add_edge(secondary_id, cap_id,
                                      relation=RelationType.ENABLES.value)
        
        # Add authentication dependencies
        self._add_auth_dependencies(vulnerabilities)
    
    def _add_auth_dependencies(self, vulnerabilities: List[Dict]):
        """Add REQUIRES edges for authenticated exploits"""
        for vuln in vulnerabilities:
            cve = vuln.get('cve')
            host = vuln.get('host_ip')
            port = vuln.get('port', 0)
            vuln_id = f"{cve}@{host}:{port}"
            
            vuln_node = self.graph.nodes[vuln_id]
            
            if vuln_node.get('requires_auth', False):
                # Requires credentials capability
                cred_cap_id = f"{CapabilityType.CREDENTIALS.value}@{host}"
                self.graph.add_edge(vuln_id, cred_cap_id,
                                  relation=RelationType.REQUIRES.value)
    
    def _model_network_topology(self, vulnerabilities: List[Dict]):
        """
        Model network connections (which hosts can reach each other)
        
        Simple heuristic: Hosts in same /24 subnet can reach each other
        """
        print(f"  → Modeling network topology...")
        
        hosts = list(self.hosts)
        
        for i, host1 in enumerate(hosts):
            for host2 in hosts[i+1:]:
                # Check if same /24 subnet
                if self._same_subnet(host1, host2):
                    # Bidirectional connection
                    self.graph.add_edge(host1, host2,
                                      relation=RelationType.CONNECTS_TO.value)
                    self.graph.add_edge(host2, host1,
                                      relation=RelationType.CONNECTS_TO.value)
    
    def _same_subnet(self, ip1: str, ip2: str) -> bool:
        """Check if two IPs are in same /24 subnet"""
        octets1 = ip1.split('.')[:3]
        octets2 = ip2.split('.')[:3]
        return octets1 == octets2
    
    def save_to_json(self, output_path: Path):
        """Save knowledge graph to JSON"""
        data = {
            'metadata': {
                'total_hosts': len(self.hosts),
                'total_vulnerabilities': len(self.vulnerabilities),
                'total_capabilities': len(self.capabilities),
                'total_edges': self.graph.number_of_edges()
            },
            'nodes': [
                {
                    'id': node,
                    **self.graph.nodes[node]
                }
                for node in self.graph.nodes
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    **self.graph.edges[u, v]
                }
                for u, v in self.graph.edges
            ]
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2))
        print(f"\n✅ Saved knowledge graph to: {output_path}")
    
    def _print_summary(self):
        """Print knowledge graph statistics"""
        print("\n" + "="*70)
        print("KNOWLEDGE GRAPH SUMMARY")
        print("="*70)
        print(f"Hosts:            {len(self.hosts)}")
        print(f"Vulnerabilities:  {len(self.vulnerabilities)}")
        print(f"Capabilities:     {len(self.capabilities)}")
        print(f"Total Nodes:      {self.graph.number_of_nodes()}")
        print(f"Total Edges:      {self.graph.number_of_edges()}")
        
        # Count relationships
        relation_counts = {}
        for u, v in self.graph.edges:
            rel = self.graph.edges[u, v].get('relation', 'UNKNOWN')
            relation_counts[rel] = relation_counts.get(rel, 0) + 1
        
        print("\nRelationships:")
        for rel, count in sorted(relation_counts.items()):
            print(f"  {rel}: {count}")
        
        # Exploitability stats
        exploitable = sum(
            1 for n in self.vulnerabilities
            if self.graph.nodes[n].get('has_exploit', False)
        )
        print(f"\nExploitability:")
        print(f"  Vulnerabilities with LLM-generated exploits: {exploitable}/{len(self.vulnerabilities)}")
        
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Build Knowledge Graph from AUVAP Pipeline'
    )
    parser.add_argument(
        '--experiment-report',
        type=Path,
        required=True,
        help='Path to experiment_report.json'
    )
    parser.add_argument(
        '--exploits-manifest',
        type=Path,
        required=True,
        help='Path to exploits_manifest.json'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('knowledge_graphs/kg.json'),
        help='Output path for knowledge graph JSON'
    )
    
    args = parser.parse_args()
    
    # Build knowledge graph
    builder = AutoKnowledgeGraphBuilder()
    kg = builder.build_from_auvap_pipeline(
        args.experiment_report,
        args.exploits_manifest
    )
    
    # Save to file
    builder.save_to_json(args.output)
    
    print("\n✅ Knowledge graph construction complete!")
    print(f"   Use this graph for PPO training with realistic dependencies")


if __name__ == '__main__':
    main()
