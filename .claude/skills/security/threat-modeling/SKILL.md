---
name: threat-modeling
description: Comprehensive threat modeling using STRIDE methodology, attack tree analysis, and mitigation mapping. Use when analyzing system security, conducting threat modeling sessions, visualizing attack paths, or creating security remediation plans.
---

# Threat Modeling

Systematic security analysis combining STRIDE methodology, attack tree construction, and threat-to-control mapping.

## When to Use This Skill

- Starting threat modeling sessions
- Analyzing system architecture security
- Visualizing attack scenarios
- Mapping threats to mitigations
- Prioritizing security investments
- Creating security documentation
- Compliance and audit preparation
- Planning defensive investments

---

## Part 1: STRIDE Analysis

### STRIDE Categories

```
S - Spoofing       → Authentication threats
T - Tampering      → Integrity threats
R - Repudiation    → Non-repudiation threats
I - Information    → Confidentiality threats
    Disclosure
D - Denial of      → Availability threats
    Service
E - Elevation of   → Authorization threats
    Privilege
```

### Threat Analysis Matrix

| Category | Question | Control Family |
|----------|----------|----------------|
| **Spoofing** | Can attacker pretend to be someone else? | Authentication |
| **Tampering** | Can attacker modify data in transit/rest? | Integrity |
| **Repudiation** | Can attacker deny actions? | Logging/Audit |
| **Info Disclosure** | Can attacker access unauthorized data? | Encryption |
| **DoS** | Can attacker disrupt availability? | Rate limiting |
| **Elevation** | Can attacker gain higher privileges? | Authorization |

### STRIDE Implementation

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict

class StrideCategory(Enum):
    SPOOFING = "S"
    TAMPERING = "T"
    REPUDIATION = "R"
    INFORMATION_DISCLOSURE = "I"
    DENIAL_OF_SERVICE = "D"
    ELEVATION_OF_PRIVILEGE = "E"


class Impact(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Likelihood(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Threat:
    id: str
    category: StrideCategory
    title: str
    description: str
    target: str
    impact: Impact
    likelihood: Likelihood
    mitigations: List[str] = field(default_factory=list)
    status: str = "open"

    @property
    def risk_score(self) -> int:
        return self.impact.value * self.likelihood.value

    @property
    def risk_level(self) -> str:
        score = self.risk_score
        if score >= 12:
            return "Critical"
        elif score >= 6:
            return "High"
        elif score >= 3:
            return "Medium"
        return "Low"


@dataclass
class ThreatModel:
    name: str
    version: str
    description: str
    threats: List[Threat] = field(default_factory=list)

    def get_threats_by_category(self, category: StrideCategory) -> List[Threat]:
        return [t for t in self.threats if t.category == category]

    def get_critical_threats(self) -> List[Threat]:
        return [t for t in self.threats if t.risk_level in ("Critical", "High")]

    def generate_report(self) -> Dict:
        return {
            "summary": {
                "name": self.name,
                "total_threats": len(self.threats),
                "critical_threats": len([t for t in self.threats if t.risk_level == "Critical"]),
            },
            "by_category": {
                cat.name: len(self.get_threats_by_category(cat))
                for cat in StrideCategory
            },
            "top_risks": [
                {"id": t.id, "title": t.title, "risk_score": t.risk_score}
                for t in sorted(self.threats, key=lambda x: x.risk_score, reverse=True)[:10]
            ]
        }
```

### STRIDE Questions by Category

```python
STRIDE_QUESTIONS = {
    StrideCategory.SPOOFING: [
        "Can an attacker impersonate a legitimate user?",
        "Are authentication tokens properly validated?",
        "Can session identifiers be predicted or stolen?",
        "Is multi-factor authentication available?",
    ],
    StrideCategory.TAMPERING: [
        "Can data be modified in transit?",
        "Can data be modified at rest?",
        "Are input validation controls sufficient?",
        "Can an attacker manipulate application logic?",
    ],
    StrideCategory.REPUDIATION: [
        "Are all security-relevant actions logged?",
        "Can logs be tampered with?",
        "Is there sufficient attribution for actions?",
        "Are timestamps reliable and synchronized?",
    ],
    StrideCategory.INFORMATION_DISCLOSURE: [
        "Is sensitive data encrypted at rest?",
        "Is sensitive data encrypted in transit?",
        "Can error messages reveal sensitive information?",
        "Are access controls properly enforced?",
    ],
    StrideCategory.DENIAL_OF_SERVICE: [
        "Are rate limits implemented?",
        "Can resources be exhausted by malicious input?",
        "Is there protection against amplification attacks?",
        "Are there single points of failure?",
    ],
    StrideCategory.ELEVATION_OF_PRIVILEGE: [
        "Are authorization checks performed consistently?",
        "Can users access other users' resources?",
        "Can privilege escalation occur through parameter manipulation?",
        "Is the principle of least privilege followed?",
    ],
}
```

### Risk Matrix

```
              IMPACT
         Low  Med  High Crit
    Low   1    2    3    4
L   Med   2    4    6    8
I   High  3    6    9    12
K   Crit  4    8   12    16
```

---

## Part 2: Attack Tree Analysis

### Attack Tree Structure

```
                    [Root Goal]
                         |
            ┌────────────┴────────────┐
            │                         │
       [Sub-goal 1]              [Sub-goal 2]
       (OR node)                 (AND node)
            │                         │
      ┌─────┴─────┐             ┌─────┴─────┐
      │           │             │           │
   [Attack]   [Attack]      [Attack]   [Attack]
    (leaf)     (leaf)        (leaf)     (leaf)
```

### Node Types

| Type | Symbol | Description |
|------|--------|-------------|
| **OR** | Oval | Any child achieves goal |
| **AND** | Rectangle | All children required |
| **Leaf** | Box | Atomic attack step |

### Attack Attributes

| Attribute | Description | Values |
|-----------|-------------|--------|
| **Cost** | Resources needed | $, $$, $$$ |
| **Time** | Duration to execute | Hours, Days, Weeks |
| **Skill** | Expertise required | Low, Medium, High |
| **Detection** | Likelihood of detection | Low, Medium, High |

### Attack Tree Implementation

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional

class NodeType(Enum):
    OR = "or"
    AND = "and"
    LEAF = "leaf"


class Difficulty(Enum):
    TRIVIAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    EXPERT = 5


class Cost(Enum):
    FREE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4


class DetectionRisk(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CERTAIN = 4


@dataclass
class AttackAttributes:
    difficulty: Difficulty = Difficulty.MEDIUM
    cost: Cost = Cost.MEDIUM
    detection_risk: DetectionRisk = DetectionRisk.MEDIUM
    time_hours: float = 8.0
    requires_insider: bool = False
    requires_physical: bool = False


@dataclass
class AttackNode:
    id: str
    name: str
    description: str
    node_type: NodeType
    attributes: AttackAttributes = field(default_factory=AttackAttributes)
    children: List['AttackNode'] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)

    def add_child(self, child: 'AttackNode') -> None:
        self.children.append(child)

    def calculate_path_difficulty(self) -> float:
        """Calculate aggregate difficulty for this path."""
        if self.node_type == NodeType.LEAF:
            return self.attributes.difficulty.value

        if not self.children:
            return 0

        child_difficulties = [c.calculate_path_difficulty() for c in self.children]

        if self.node_type == NodeType.OR:
            return min(child_difficulties)
        else:  # AND
            return max(child_difficulties)


@dataclass
class AttackTree:
    name: str
    description: str
    root: AttackNode

    def find_easiest_path(self) -> List[AttackNode]:
        """Find the path with lowest difficulty."""
        return self._find_path(self.root, minimize="difficulty")

    def get_all_leaf_attacks(self) -> List[AttackNode]:
        """Get all leaf attack nodes."""
        leaves = []
        self._collect_leaves(self.root, leaves)
        return leaves

    def _collect_leaves(self, node: AttackNode, leaves: List[AttackNode]) -> None:
        if node.node_type == NodeType.LEAF:
            leaves.append(node)
        for child in node.children:
            self._collect_leaves(child, leaves)

    def get_unmitigated_attacks(self) -> List[AttackNode]:
        """Find attacks without mitigations."""
        return [n for n in self.get_all_leaf_attacks() if not n.mitigations]
```

### Attack Tree Builder

```python
class AttackTreeBuilder:
    """Fluent builder for attack trees."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._node_stack: List[AttackNode] = []
        self._root: Optional[AttackNode] = None

    def goal(self, id: str, name: str, description: str = "") -> 'AttackTreeBuilder':
        """Set the root goal."""
        self._root = AttackNode(id=id, name=name, description=description, node_type=NodeType.OR)
        self._node_stack = [self._root]
        return self

    def or_node(self, id: str, name: str, description: str = "") -> 'AttackTreeBuilder':
        """Add an OR sub-goal."""
        node = AttackNode(id=id, name=name, description=description, node_type=NodeType.OR)
        self._current().add_child(node)
        self._node_stack.append(node)
        return self

    def and_node(self, id: str, name: str, description: str = "") -> 'AttackTreeBuilder':
        """Add an AND sub-goal (all children required)."""
        node = AttackNode(id=id, name=name, description=description, node_type=NodeType.AND)
        self._current().add_child(node)
        self._node_stack.append(node)
        return self

    def attack(self, id: str, name: str, difficulty: Difficulty = Difficulty.MEDIUM,
               cost: Cost = Cost.MEDIUM, detection: DetectionRisk = DetectionRisk.MEDIUM,
               mitigations: List[str] = None) -> 'AttackTreeBuilder':
        """Add a leaf attack node."""
        node = AttackNode(
            id=id, name=name, description="", node_type=NodeType.LEAF,
            attributes=AttackAttributes(difficulty=difficulty, cost=cost, detection_risk=detection),
            mitigations=mitigations or []
        )
        self._current().add_child(node)
        return self

    def end(self) -> 'AttackTreeBuilder':
        """Close current node, return to parent."""
        if len(self._node_stack) > 1:
            self._node_stack.pop()
        return self

    def build(self) -> AttackTree:
        """Build the attack tree."""
        return AttackTree(name=self.name, description=self.description, root=self._root)

    def _current(self) -> AttackNode:
        return self._node_stack[-1]


# Example usage
def build_account_takeover_tree() -> AttackTree:
    return (
        AttackTreeBuilder("Account Takeover", "Gain unauthorized access")
        .goal("G1", "Take Over User Account")
        .or_node("S1", "Steal Credentials")
            .attack("A1", "Phishing Attack", difficulty=Difficulty.LOW,
                    mitigations=["Security awareness training", "Email filtering"])
            .attack("A2", "Credential Stuffing", difficulty=Difficulty.TRIVIAL,
                    mitigations=["Rate limiting", "MFA", "Password breach monitoring"])
        .end()
        .or_node("S2", "Bypass Authentication")
            .attack("A3", "Session Hijacking", difficulty=Difficulty.MEDIUM,
                    mitigations=["Secure session management", "HTTPS only"])
        .end()
        .build()
    )
```

### Mermaid Diagram Export

```python
class MermaidExporter:
    """Export attack trees to Mermaid diagram format."""

    def __init__(self, tree: AttackTree):
        self.tree = tree
        self._lines: List[str] = []
        self._node_count = 0

    def export(self) -> str:
        """Export tree to Mermaid flowchart."""
        self._lines = ["flowchart TD"]
        self._export_node(self.tree.root, None)
        return "\n".join(self._lines)

    def _export_node(self, node: AttackNode, parent_id: Optional[str]) -> str:
        node_id = f"N{self._node_count}"
        self._node_count += 1

        if node.node_type == NodeType.OR:
            shape = f"{node_id}(({node.name}))"
        elif node.node_type == NodeType.AND:
            shape = f"{node_id}[{node.name}]"
        else:
            shape = f"{node_id}[/{node.name}/]"

        self._lines.append(f"    {shape}")

        if parent_id:
            self._lines.append(f"    {parent_id} --> {node_id}")

        for child in node.children:
            self._export_node(child, node_id)

        return node_id
```

---

## Part 3: Threat Mitigation Mapping

### Control Categories

```
Preventive ────► Stop attacks before they occur
   │              (Firewall, Input validation)
   │
Detective ─────► Identify attacks in progress
   │              (IDS, Log monitoring)
   │
Corrective ────► Respond and recover from attacks
                  (Incident response, Backup restore)
```

### Defense in Depth

```
                    ┌──────────────────────┐
                    │      Perimeter       │ ← Firewall, WAF
                    │   ┌──────────────┐   │
                    │   │   Network    │   │ ← Segmentation, IDS
                    │   │  ┌────────┐  │   │
                    │   │  │  Host  │  │   │ ← EDR, Hardening
                    │   │  │ ┌────┐ │  │   │
                    │   │  │ │App │ │  │   │ ← Auth, Validation
                    │   │  │ │Data│ │  │   │ ← Encryption
                    │   │  │ └────┘ │  │   │
                    │   │  └────────┘  │   │
                    │   └──────────────┘   │
                    └──────────────────────┘
```

### Security Control Model

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional

class ControlType(Enum):
    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"


class ControlLayer(Enum):
    NETWORK = "network"
    APPLICATION = "application"
    DATA = "data"
    ENDPOINT = "endpoint"
    PROCESS = "process"


class ImplementationStatus(Enum):
    NOT_IMPLEMENTED = "not_implemented"
    PARTIAL = "partial"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"


class Effectiveness(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4


@dataclass
class SecurityControl:
    id: str
    name: str
    description: str
    control_type: ControlType
    layer: ControlLayer
    effectiveness: Effectiveness
    implementation_cost: str
    status: ImplementationStatus = ImplementationStatus.NOT_IMPLEMENTED
    mitigates_threats: List[str] = field(default_factory=list)
    compliance_refs: List[str] = field(default_factory=list)

    def coverage_score(self) -> float:
        status_multiplier = {
            ImplementationStatus.NOT_IMPLEMENTED: 0.0,
            ImplementationStatus.PARTIAL: 0.5,
            ImplementationStatus.IMPLEMENTED: 0.8,
            ImplementationStatus.VERIFIED: 1.0,
        }
        return self.effectiveness.value * status_multiplier[self.status]
```

### Control Library

```python
STANDARD_CONTROLS = {
    # Authentication Controls
    "AUTH-001": SecurityControl(
        id="AUTH-001",
        name="Multi-Factor Authentication",
        description="Require MFA for all user authentication",
        control_type=ControlType.PREVENTIVE,
        layer=ControlLayer.APPLICATION,
        effectiveness=Effectiveness.HIGH,
        implementation_cost="Medium",
        mitigates_threats=["SPOOFING"],
        compliance_refs=["PCI-DSS 8.3", "NIST 800-63B"]
    ),

    # Input Validation
    "VAL-001": SecurityControl(
        id="VAL-001",
        name="Input Validation Framework",
        description="Validate and sanitize all user input",
        control_type=ControlType.PREVENTIVE,
        layer=ControlLayer.APPLICATION,
        effectiveness=Effectiveness.HIGH,
        implementation_cost="Medium",
        mitigates_threats=["TAMPERING", "INJECTION"],
        compliance_refs=["OWASP ASVS V5"]
    ),

    # Encryption
    "ENC-001": SecurityControl(
        id="ENC-001",
        name="Data Encryption at Rest",
        description="Encrypt sensitive data in storage",
        control_type=ControlType.PREVENTIVE,
        layer=ControlLayer.DATA,
        effectiveness=Effectiveness.HIGH,
        implementation_cost="Medium",
        mitigates_threats=["INFORMATION_DISCLOSURE"],
        compliance_refs=["PCI-DSS 3.4", "GDPR Art. 32"]
    ),

    # Logging
    "LOG-001": SecurityControl(
        id="LOG-001",
        name="Security Event Logging",
        description="Log all security-relevant events",
        control_type=ControlType.DETECTIVE,
        layer=ControlLayer.APPLICATION,
        effectiveness=Effectiveness.MEDIUM,
        implementation_cost="Low",
        mitigates_threats=["REPUDIATION"],
        compliance_refs=["PCI-DSS 10.2", "SOC2"]
    ),

    # Access Control
    "ACC-001": SecurityControl(
        id="ACC-001",
        name="Role-Based Access Control",
        description="Implement RBAC for authorization",
        control_type=ControlType.PREVENTIVE,
        layer=ControlLayer.APPLICATION,
        effectiveness=Effectiveness.HIGH,
        implementation_cost="Medium",
        mitigates_threats=["ELEVATION_OF_PRIVILEGE"],
        compliance_refs=["PCI-DSS 7.1", "SOC2"]
    ),

    # Availability
    "AVL-001": SecurityControl(
        id="AVL-001",
        name="Rate Limiting",
        description="Limit request rates to prevent abuse",
        control_type=ControlType.PREVENTIVE,
        layer=ControlLayer.APPLICATION,
        effectiveness=Effectiveness.MEDIUM,
        implementation_cost="Low",
        mitigates_threats=["DENIAL_OF_SERVICE"],
        compliance_refs=["OWASP API Security"]
    ),
}
```

### Mitigation Plan Analysis

```python
@dataclass
class MitigationMapping:
    threat: Threat
    controls: List[SecurityControl]

    def calculate_coverage(self) -> float:
        if not self.controls:
            return 0.0
        total_score = sum(c.coverage_score() for c in self.controls)
        max_possible = len(self.controls) * Effectiveness.VERY_HIGH.value
        return (total_score / max_possible) * 100 if max_possible > 0 else 0

    def has_defense_in_depth(self) -> bool:
        layers = set(c.layer for c in self.controls
                    if c.status != ImplementationStatus.NOT_IMPLEMENTED)
        return len(layers) >= 2

    def has_control_diversity(self) -> bool:
        types = set(c.control_type for c in self.controls
                   if c.status != ImplementationStatus.NOT_IMPLEMENTED)
        return len(types) >= 2


@dataclass
class MitigationPlan:
    name: str
    threats: List[Threat] = field(default_factory=list)
    controls: List[SecurityControl] = field(default_factory=list)
    mappings: List[MitigationMapping] = field(default_factory=list)

    def get_unmapped_threats(self) -> List[Threat]:
        mapped_ids = {m.threat.id for m in self.mappings}
        return [t for t in self.threats if t.id not in mapped_ids]

    def get_gaps(self) -> List[Dict]:
        gaps = []
        for mapping in self.mappings:
            coverage = mapping.calculate_coverage()
            if coverage < 50:
                gaps.append({
                    "threat": mapping.threat.id,
                    "issue": "Insufficient control coverage",
                    "coverage": coverage
                })
            if not mapping.has_defense_in_depth():
                gaps.append({
                    "threat": mapping.threat.id,
                    "issue": "No defense in depth"
                })
        return gaps
```

---

## Threat Model Document Template

```markdown
# Threat Model: [System Name]

## 1. System Overview
[Brief description of the system and its purpose]

### Data Flow Diagram
[User] --> [Web App] --> [API Gateway] --> [Backend Services]
                              |
                              v
                        [Database]

### Trust Boundaries
- External: Internet to DMZ
- Internal: DMZ to Internal Network
- Data: Application to Database

## 2. Assets
| Asset | Sensitivity | Description |
|-------|-------------|-------------|
| User Credentials | High | Passwords, tokens |
| Personal Data | High | PII |
| Session Data | Medium | Active sessions |

## 3. STRIDE Analysis

### Spoofing
| ID | Threat | Impact | Likelihood |
|----|--------|--------|------------|
| S1 | Session hijacking | High | Medium |

### Tampering
| ID | Threat | Impact | Likelihood |
|----|--------|--------|------------|
| T1 | SQL injection | Critical | Medium |

[Continue for R, I, D, E...]

## 4. Attack Trees
[Include Mermaid diagrams for key attack scenarios]

## 5. Mitigation Plan
| Threat | Control | Status | Coverage |
|--------|---------|--------|----------|
| S1 | MFA | Implemented | 80% |
| T1 | Input Validation | Partial | 50% |

## 6. Recommendations
1. Immediate: [Critical fixes]
2. Short-term (30 days): [High priority]
3. Long-term (90 days): [Improvements]
```

---

## Best Practices

### Do's
- **Be systematic** - Cover all STRIDE categories
- **Involve stakeholders** - Security, dev, and ops
- **Visualize attacks** - Attack trees aid understanding
- **Layer controls** - Defense in depth is essential
- **Mix control types** - Preventive, detective, corrective
- **Update regularly** - Threat models are living documents

### Don'ts
- **Don't skip categories** - Each reveals different threats
- **Don't assume security** - Question every component
- **Don't rely on single controls** - Single points of failure
- **Don't ignore people/process** - Technology alone isn't enough
- **Don't set and forget** - Continuous improvement

## Resources

- [Microsoft STRIDE](https://docs.microsoft.com/en-us/azure/security/develop/threat-modeling-tool-threats)
- [OWASP Threat Modeling](https://owasp.org/www-community/Threat_Modeling)
- [MITRE ATT&CK Framework](https://attack.mitre.org/)
- [MITRE D3FEND](https://d3fend.mitre.org/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CIS Controls](https://www.cisecurity.org/controls)
