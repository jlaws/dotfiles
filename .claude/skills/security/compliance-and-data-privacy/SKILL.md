---
name: compliance-and-data-privacy
description: "Use when implementing GDPR, PCI-DSS, SOC2, or HIPAA compliance, handling personal/payment data, building consent systems, or conducting compliance reviews."
---

# Compliance & Data Privacy

## Framework Decision Matrix

| Framework | Triggers | Scope |
|-----------|----------|-------|
| **GDPR** | Process EU personal data | All data subjects in EU/EEA |
| **PCI-DSS** | Handle payment cards | Cardholder data environment (CDE) |
| **SOC2** | SaaS / service provider | Trust principles: Security, Availability, Integrity, Confidentiality, Privacy |
| **HIPAA** | US healthcare data (PHI) | Covered entities + business associates |

## GDPR Essentials

### Legal Bases (pick exactly one per processing activity)
- **Consent**: freely given, specific, informed, withdrawable
- **Contract**: necessary for contract performance
- **Legal obligation**: required by law
- **Legitimate interest**: balanced against data subject rights (document the balancing test)

### Data Subject Rights (respond within 30 days)
Access, Rectification, Erasure, Restrict Processing, Portability, Object

### Privacy by Design Defaults
- Separate PII from behavioral data (different tables/stores)
- UUIDs not sequential IDs; hash emails for lookups
- Encrypt PII at rest with per-record key IDs
- Pseudonymize analytics with rotating pseudonym IDs
- Generalize location to country level
- Collect only fields required for stated purpose (data minimization)

### Breach Notification
- **72 hours** to notify supervisory authority (sensitive data or medium+ severity)
- **Without undue delay** to affected individuals (high/critical severity)

### Retention Policy Defaults
| Data Type | Retention | Basis | End-of-Life |
|-----------|-----------|-------|-------------|
| User accounts | 3 years from last activity | Contract | Archive then delete |
| Transactions | 7 years | Legal obligation | Archive then delete |
| Marketing consent | 2 years | Consent | Delete |
| Analytics | 1 year | Consent | Anonymize |

## PCI-DSS Essentials

### Cardinal Rules
- **NEVER store**: full track data, CVV/CVC, PIN
- **CAN store** (encrypted): PAN, cardholder name, expiration, service code
- Display PAN as first-6 + last-4 only (mask middle)

### Scope Reduction Strategy (in order of preference)
1. **Hosted payments** (Stripe Checkout, PayPal) -- SAQ A (~20 questions)
2. **Embedded JS** (Stripe Elements) -- SAQ A-EP (~180 questions)
3. **Direct handling** -- SAQ D (~300 questions); avoid if possible

### Compliance Levels
| Level | Volume | Audit |
|-------|--------|-------|
| 1 | >6M txns/year | Annual ROC + quarterly ASV scan |
| 2 | 1-6M | Annual SAQ |
| 3 | 20K-1M e-commerce | SAQ |
| 4 | <20K e-commerce | SAQ |

### Vulnerability Remediation SLAs
- Critical: 24 hours
- High: 7 days
- Medium: 30 days
- Low: 90 days

### Encryption Requirements
- At rest: AES-256-GCM (envelope encryption preferred)
- In transit: TLS 1.2+ mandatory
- Key management: rotate annually, separate key custodians

## SOC2 Key Controls

- **CC6.1 Access**: MFA for sensitive resources, RBAC with least privilege, time-limited tokens
- **Encryption**: AES-256-GCM at rest, TLS 1.2+ in transit, HSTS headers
- **Audit logging**: auth events, authz decisions, data access/export/modify/delete
- **Tamper-proof logs**: append-only storage, chained checksums, real-time critical alerts

## HIPAA Key Controls

- **Minimum Necessary**: PHI access scoped by role + purpose, time-limited tokens (24hr)
- **Encryption**: FIPS 140-2 validated, AES-256-CBC, PBKDF2 key derivation (100K iterations)
- **Transmission**: TLS 1.2+, VPN required, S/MIME or PGP for email
- **BAAs**: required for ALL third-party processors
- **Training**: annual renewal (Privacy Rule, Security Rule, PHI handling, breach notification)

## Cross-Framework Compliance Checklist

### Data Protection
- [ ] Legal basis documented for each processing activity
- [ ] Encryption at rest (AES-256) and in transit (TLS 1.2+)
- [ ] Data minimization enforced (collect only what's needed)
- [ ] Retention policies with automated enforcement

### Access Control
- [ ] RBAC with least privilege
- [ ] MFA for sensitive data access
- [ ] Access reviews quarterly
- [ ] Audit logging on all data access (who, what, when, from where)

### Consent & Rights
- [ ] Opt-in consent (not pre-checked), granular per purpose
- [ ] Consent withdrawal mechanism with immediate effect
- [ ] DSAR process: access, erasure, portability, rectification
- [ ] 30-day response SLA with tracking

### Incident Response
- [ ] Breach detection and classification process
- [ ] 72-hour authority notification (GDPR)
- [ ] Individual notification for high-severity breaches
- [ ] Post-incident review and remediation tracking

### CI/CD Integration
- [ ] PII scanner in pipeline (flag accidental PII in logs/responses)
- [ ] Encryption verification (detect plaintext sensitive fields)
- [ ] License checker (MIT, Apache-2.0, BSD-3-Clause, ISC only)
- [ ] Compliance report generated on every main branch push

## Non-Obvious Gotchas

- Legitimate interest requires a documented balancing test -- "we need it" is not sufficient
- GDPR erasure has exceptions: legal obligations (tax records) override right to delete
- PCI scope creep: if card data touches a system, that system is in scope -- network segmentation is the primary defense
- Tokenization only reduces scope if the token vault is isolated and separately secured
- SOC2 Type I (point-in-time) vs Type II (over a period) -- customers increasingly demand Type II
- HIPAA BAAs must be in place BEFORE sharing PHI, not retroactively
- Anonymized data is outside GDPR scope, but pseudonymized data is NOT -- know the difference
- Log retention itself creates compliance obligations; don't log PII unless required
