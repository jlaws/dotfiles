---
name: j-audit
description: "Security threat model and vulnerability scan — STRIDE analysis, SAST patterns, and compliance mapping. Use when reviewing code for vulnerabilities, conducting threat modeling, or mapping compliance controls. Do NOT use for quick security questions (ask directly instead)."
argument-hint: "<target-path-or-scope>"
model: opus
effort: xhigh
---

Load skill `analysis-output-patterns` for output structure rules.
Load skill `code-review-patterns` for severity labeling and vulnerability-spotting discipline.

Before invoking the analysis, perform reconnaissance:

1. **Detect project language/framework** from config files (package.json, Cargo.toml, go.mod, requirements.txt, Gemfile, etc.).
2. **Identify auth patterns** by searching for auth-related imports and middleware.
3. **Check existing security tooling** config (.semgrep.yml, .snyk, .trivyignore, .eslintrc security plugins).
4. **Get file tree overview** of the target scope (if $ARGUMENTS specifies a component/directory, scope to that; otherwise scope to the full project).

---

## Security Analysis

Apply STRIDE threat modeling, risk scoring, SAST tool selection, and compliance mapping per `references/security/security-analysis.md` -- it covers the STRIDE categories and requirements mapping, the risk-scoring formula, the controls library, SAST tool/rule selection and CI integration, the threat model document structure, and the compliance framework quick reference. For multi-trust-boundary systems, that reference's "Comprehensive Threat Modeling" section gives the sequential-perspective breakdown (STRIDE analyst, compliance mapper, attack path analyst, mitigation planner).

### References

For deep code-level vulnerability review, delegate to the `security-reviewer` agent via the Task tool, passing the reconnaissance findings and scope. It loads its skills (code-review-patterns) and the `references/security/` library, then returns findings with severity labels and file:line. Verify its findings against the code before presenting. For the review protocol itself — threat-model-before-scanning, execution-verify N/N, an independent grader, and the patch-validation gate — follow references/security/vulnerability-review-pipeline.md.

When the target includes AI/ML components, also read `references/ai-ml/ai-safety-and-alignment` for AI-specific security and safety considerations.

---

Apply the above methodology to analyze: $ARGUMENTS
