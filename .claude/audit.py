#!/usr/bin/env python3
"""
Comprehensive structural audit of the .claude knowledge base.
Checks skills, agents, commands, and references against defined standards.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

BASE_DIR = Path("/sessions/peaceful-pensive-gauss/mnt/dotfiles/.claude")

# Result tracking
results = {
    "skills": [],
    "agents": [],
    "commands": [],
    "references": [],
    "cross_refs": []
}

# Cross-reference index
all_assets = {
    "skills": {},
    "agents": {},
    "commands": {},
    "references": {}
}


def is_kebab_case(name: str) -> bool:
    """Check if name follows kebab-case convention."""
    return bool(re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", name))


def extract_yaml_frontmatter(content: str) -> Dict[str, str]:
    """Extract YAML frontmatter from markdown file."""
    frontmatter = {}
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if match:
        yaml_content = match.group(1)
        for line in yaml_content.split("\n"):
            if not line.strip():
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                frontmatter[key.strip()] = value.strip().strip('"\'')
    return frontmatter


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def get_markdown_headings(content: str) -> List[str]:
    """Extract all markdown headings from content."""
    headings = re.findall(r"^#+\s+(.+)$", content, re.MULTILINE)
    return headings


def find_references_in_content(content: str) -> List[str]:
    """Find all references to skills, agents, commands, and references."""
    references = []
    # Look for markdown links and references
    link_refs = re.findall(r"\[.*?\]\((.*?)\)", content)
    references.extend(link_refs)
    # Look for skill/agent/command references
    text_refs = re.findall(r"`([a-z0-9-]+/[a-z0-9-]+)`", content)
    references.extend(text_refs)
    return references


def audit_skill(skill_dir: Path) -> Dict[str, Any]:
    """Audit a single skill directory."""
    results = {
        "path": str(skill_dir),
        "name": skill_dir.name,
        "checks": {}
    }

    # SK-S3: Folder name is kebab-case
    results["checks"]["SK-S3"] = {
        "status": "PASS" if is_kebab_case(skill_dir.name) else "FAIL",
        "message": f"Folder name: {skill_dir.name}"
    }

    # SK-S1: SKILL.md exists with exact casing
    skill_md = skill_dir / "SKILL.md"
    results["checks"]["SK-S1"] = {
        "status": "PASS" if skill_md.exists() else "FAIL",
        "message": f"SKILL.md exists: {skill_md.exists()}"
    }

    if not skill_md.exists():
        return results

    content = skill_md.read_text()
    frontmatter = extract_yaml_frontmatter(content)

    # SK-F1: YAML frontmatter present
    results["checks"]["SK-F1"] = {
        "status": "PASS" if frontmatter else "FAIL",
        "message": "YAML frontmatter found" if frontmatter else "No YAML frontmatter"
    }

    # SK-F2: name field exists
    results["checks"]["SK-F2"] = {
        "status": "PASS" if "name" in frontmatter else "FAIL",
        "message": f"name field: {frontmatter.get('name', 'MISSING')}"
    }

    # SK-F4: name matches folder name
    name_match = frontmatter.get("name", "").lower().replace(" ", "-") == skill_dir.name
    results["checks"]["SK-F4"] = {
        "status": "PASS" if name_match else "WARN",
        "message": f"name={frontmatter.get('name', '?')} vs folder={skill_dir.name}"
    }

    # SK-F5: description field exists
    results["checks"]["SK-F5"] = {
        "status": "PASS" if "description" in frontmatter else "FAIL",
        "message": f"description field: {'PRESENT' if 'description' in frontmatter else 'MISSING'}"
    }

    # SK-F6: description under 1024 chars
    desc = frontmatter.get("description", "")
    results["checks"]["SK-F6"] = {
        "status": "PASS" if len(desc) < 1024 else "FAIL",
        "message": f"description length: {len(desc)} chars"
    }

    # SK-F9: Description has trigger phrase
    trigger_phrases = ["use when", "use to", "when you", "run this"]
    has_trigger = any(phrase in desc.lower() for phrase in trigger_phrases)
    results["checks"]["SK-F9"] = {
        "status": "PASS" if has_trigger else "WARN",
        "message": f"trigger phrase found: {has_trigger}"
    }

    # Index the skill
    all_assets["skills"][skill_dir.name] = {
        "path": str(skill_dir),
        "name": frontmatter.get("name", skill_dir.name),
        "content": content
    }

    return results


def audit_agent(agent_file: Path) -> Dict[str, Any]:
    """Audit a single agent file."""
    results = {
        "path": str(agent_file),
        "name": agent_file.stem,
        "checks": {}
    }

    # AG-S1: Filename is kebab-case
    results["checks"]["AG-S1"] = {
        "status": "PASS" if is_kebab_case(agent_file.stem) else "FAIL",
        "message": f"Filename: {agent_file.stem}"
    }

    content = agent_file.read_text()
    frontmatter = extract_yaml_frontmatter(content)

    # AG-F1: YAML frontmatter present
    results["checks"]["AG-F1"] = {
        "status": "PASS" if frontmatter else "FAIL",
        "message": "YAML frontmatter found" if frontmatter else "No YAML frontmatter"
    }

    # AG-F2: name field exists
    results["checks"]["AG-F2"] = {
        "status": "PASS" if "name" in frontmatter else "FAIL",
        "message": f"name field: {frontmatter.get('name', 'MISSING')}"
    }

    # AG-F3: name matches filename
    name_match = frontmatter.get("name", "").lower().replace(" ", "-") == agent_file.stem
    results["checks"]["AG-F3"] = {
        "status": "PASS" if name_match else "WARN",
        "message": f"name={frontmatter.get('name', '?')} vs file={agent_file.stem}"
    }

    # AG-F4: description field exists
    results["checks"]["AG-F4"] = {
        "status": "PASS" if "description" in frontmatter else "FAIL",
        "message": f"description field: {'PRESENT' if 'description' in frontmatter else 'MISSING'}"
    }

    # Index the agent
    all_assets["agents"][agent_file.stem] = {
        "path": str(agent_file),
        "name": frontmatter.get("name", agent_file.stem),
        "content": content
    }

    return results


def audit_command(command_file: Path) -> Dict[str, Any]:
    """Audit a single command file."""
    results = {
        "path": str(command_file),
        "name": command_file.stem,
        "checks": {}
    }

    # CM-S1: Filename is kebab-case
    results["checks"]["CM-S1"] = {
        "status": "PASS" if is_kebab_case(command_file.stem) else "FAIL",
        "message": f"Filename: {command_file.stem}"
    }

    content = command_file.read_text()
    frontmatter = extract_yaml_frontmatter(content)

    # CM-F1: YAML frontmatter present
    results["checks"]["CM-F1"] = {
        "status": "PASS" if frontmatter else "FAIL",
        "message": "YAML frontmatter found" if frontmatter else "No YAML frontmatter"
    }

    # CM-F2: name field exists
    results["checks"]["CM-F2"] = {
        "status": "PASS" if "name" in frontmatter else "FAIL",
        "message": f"name field: {frontmatter.get('name', 'MISSING')}"
    }

    # CM-F3: name matches filename
    name_match = frontmatter.get("name", "").lower().replace(" ", "-") == command_file.stem
    results["checks"]["CM-F3"] = {
        "status": "PASS" if name_match else "WARN",
        "message": f"name={frontmatter.get('name', '?')} vs file={command_file.stem}"
    }

    # CM-F4: description field exists
    results["checks"]["CM-F4"] = {
        "status": "PASS" if "description" in frontmatter else "FAIL",
        "message": f"description field: {'PRESENT' if 'description' in frontmatter else 'MISSING'}"
    }

    # Index the command
    all_assets["commands"][command_file.stem] = {
        "path": str(command_file),
        "name": frontmatter.get("name", command_file.stem),
        "content": content
    }

    return results


def audit_reference(ref_file: Path) -> Dict[str, Any]:
    """Audit a single reference file."""
    results = {
        "path": str(ref_file),
        "name": ref_file.stem,
        "checks": {}
    }

    # RF-S1: Filename is kebab-case
    results["checks"]["RF-S1"] = {
        "status": "PASS" if is_kebab_case(ref_file.stem) else "FAIL",
        "message": f"Filename: {ref_file.stem}"
    }

    content = ref_file.read_text()

    # RF-C1: Has at least one markdown heading
    headings = get_markdown_headings(content)
    results["checks"]["RF-C1"] = {
        "status": "PASS" if headings else "FAIL",
        "message": f"Markdown headings found: {len(headings)}"
    }

    # RF-C2: Over 50 words of content
    word_count = count_words(content)
    results["checks"]["RF-C2"] = {
        "status": "PASS" if word_count > 50 else "WARN",
        "message": f"Word count: {word_count}"
    }

    # Index the reference
    all_assets["references"][ref_file.stem] = {
        "path": str(ref_file),
        "name": ref_file.stem,
        "content": content,
        "word_count": word_count
    }

    return results


def check_cross_references() -> List[Dict[str, Any]]:
    """Verify cross-references between assets."""
    cross_ref_results = []

    # Check all assets for references to other assets
    all_content = {}
    for asset_type, assets in all_assets.items():
        for asset_name, asset_info in assets.items():
            all_content[f"{asset_type}/{asset_name}"] = asset_info["content"]

    # For each asset, find what it references
    for source_id, content in all_content.items():
        refs = find_references_in_content(content)
        for ref in refs:
            # Check if reference is valid
            ref_status = "UNKNOWN"

            # Check skills
            if any(ref.endswith(name) for name in all_assets["skills"].keys()):
                ref_status = "VALID (skill)"
            # Check agents
            elif any(ref.endswith(name) for name in all_assets["agents"].keys()):
                ref_status = "VALID (agent)"
            # Check commands
            elif any(ref.endswith(name) for name in all_assets["commands"].keys()):
                ref_status = "VALID (command)"
            # Check references
            elif any(ref.endswith(name) for name in all_assets["references"].keys()):
                ref_status = "VALID (reference)"

            if ref_status == "UNKNOWN":
                cross_ref_results.append({
                    "source": source_id,
                    "reference": ref,
                    "status": "UNRESOLVED"
                })

    return cross_ref_results


def find_orphaned_references() -> List[str]:
    """Find reference files that are not referenced by any other asset."""
    orphans = []

    for ref_name, ref_info in all_assets["references"].items():
        referenced = False
        for asset_type, assets in all_assets.items():
            if asset_type == "references":
                continue
            for asset_name, asset_info in assets.items():
                if ref_name in asset_info["content"]:
                    referenced = True
                    break
            if referenced:
                break

        # Also check if referenced by other references
        for other_ref_name, other_ref_info in all_assets["references"].items():
            if ref_name != other_ref_name and ref_name in other_ref_info["content"]:
                referenced = True
                break

        if not referenced:
            orphans.append(ref_name)

    return orphans


def run_audit() -> None:
    """Run the complete audit."""
    print("Starting audit of .claude knowledge base...\n")

    # Audit skills
    print("Auditing skills...")
    skills_dir = BASE_DIR / "skills"
    for skill_type_dir in skills_dir.iterdir():
        if not skill_type_dir.is_dir():
            continue
        for skill_dir in skill_type_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            result = audit_skill(skill_dir)
            results["skills"].append(result)

    # Audit agents
    print("Auditing agents...")
    agents_dir = BASE_DIR / "agents"
    for agent_file in agents_dir.glob("*.md"):
        result = audit_agent(agent_file)
        results["agents"].append(result)

    # Audit commands
    print("Auditing commands...")
    commands_dir = BASE_DIR / "commands"
    for cmd_type_dir in commands_dir.iterdir():
        if not cmd_type_dir.is_dir():
            continue
        for cmd_file in cmd_type_dir.glob("*.md"):
            result = audit_command(cmd_file)
            results["commands"].append(result)

    # Audit references
    print("Auditing references...")
    refs_dir = BASE_DIR / "references"
    for ref_type_dir in refs_dir.iterdir():
        if not ref_type_dir.is_dir():
            continue
        for ref_file in ref_type_dir.rglob("*.md"):
            if ref_file.parent.name == "references":
                # Skip nested reference directories for now
                continue
            if ref_file.name == "SKILL.md":
                continue
            result = audit_reference(ref_file)
            results["references"].append(result)

    # Check cross-references
    print("Checking cross-references...")
    results["cross_refs"] = check_cross_references()

    # Find orphaned references
    print("Finding orphaned references...")
    orphans = find_orphaned_references()

    print(f"\nAudit complete. Found {len(results['skills'])} skills, "
          f"{len(results['agents'])} agents, {len(results['commands'])} commands, "
          f"{len(results['references'])} references, {len(orphans)} orphaned references.\n")

    # Generate report
    generate_report(orphans)


def generate_report(orphans: List[str]) -> None:
    """Generate the audit report."""
    report_lines = []

    report_lines.append("# .claude Knowledge Base Audit Report\n")
    report_lines.append(f"Generated: {Path('/sessions/peaceful-pensive-gauss/mnt/dotfiles/.claude').stat().st_mtime}\n")

    # Summary
    report_lines.append("## Executive Summary\n")

    total_checks = 0
    passed_checks = 0
    warned_checks = 0
    failed_checks = 0

    for asset_type, asset_results in results.items():
        if asset_type == "cross_refs":
            continue
        for asset in asset_results:
            for check_name, check_result in asset.get("checks", {}).items():
                total_checks += 1
                status = check_result["status"]
                if status == "PASS":
                    passed_checks += 1
                elif status == "WARN":
                    warned_checks += 1
                elif status == "FAIL":
                    failed_checks += 1

    report_lines.append(f"- **Total Checks**: {total_checks}\n")
    report_lines.append(f"- **Passed**: {passed_checks} ({100*passed_checks//total_checks if total_checks else 0}%)\n")
    report_lines.append(f"- **Warnings**: {warned_checks}\n")
    report_lines.append(f"- **Failed**: {failed_checks}\n")
    report_lines.append(f"- **Orphaned References**: {len(orphans)}\n")

    # Skills report
    report_lines.append("\n## Skills Report\n")
    report_lines.append(f"Total skills audited: {len(results['skills'])}\n\n")

    for skill in sorted(results["skills"], key=lambda x: x["name"]):
        report_lines.append(f"### {skill['name']}\n")
        report_lines.append(f"Path: `{skill['path']}`\n\n")

        for check_name in sorted(skill["checks"].keys()):
            check = skill["checks"][check_name]
            status_marker = "✓" if check["status"] == "PASS" else ("⚠" if check["status"] == "WARN" else "✗")
            report_lines.append(f"- {status_marker} **{check_name}** [{check['status']}]: {check['message']}\n")

        report_lines.append("\n")

    # Agents report
    report_lines.append("\n## Agents Report\n")
    report_lines.append(f"Total agents audited: {len(results['agents'])}\n\n")

    for agent in sorted(results["agents"], key=lambda x: x["name"]):
        report_lines.append(f"### {agent['name']}\n")
        report_lines.append(f"Path: `{agent['path']}`\n\n")

        for check_name in sorted(agent["checks"].keys()):
            check = agent["checks"][check_name]
            status_marker = "✓" if check["status"] == "PASS" else ("⚠" if check["status"] == "WARN" else "✗")
            report_lines.append(f"- {status_marker} **{check_name}** [{check['status']}]: {check['message']}\n")

        report_lines.append("\n")

    # Commands report
    report_lines.append("\n## Commands Report\n")
    report_lines.append(f"Total commands audited: {len(results['commands'])}\n\n")

    for cmd in sorted(results["commands"], key=lambda x: x["name"]):
        report_lines.append(f"### {cmd['name']}\n")
        report_lines.append(f"Path: `{cmd['path']}`\n\n")

        for check_name in sorted(cmd["checks"].keys()):
            check = cmd["checks"][check_name]
            status_marker = "✓" if check["status"] == "PASS" else ("⚠" if check["status"] == "WARN" else "✗")
            report_lines.append(f"- {status_marker} **{check_name}** [{check['status']}]: {check['message']}\n")

        report_lines.append("\n")

    # References report
    report_lines.append("\n## References Report\n")
    report_lines.append(f"Total references audited: {len(results['references'])}\n\n")

    for ref in sorted(results["references"], key=lambda x: x["name"]):
        report_lines.append(f"### {ref['name']}\n")
        report_lines.append(f"Path: `{ref['path']}`\n\n")

        for check_name in sorted(ref["checks"].keys()):
            check = ref["checks"][check_name]
            status_marker = "✓" if check["status"] == "PASS" else ("⚠" if check["status"] == "WARN" else "✗")
            report_lines.append(f"- {status_marker} **{check_name}** [{check['status']}]: {check['message']}\n")

        report_lines.append("\n")

    # Orphaned references
    if orphans:
        report_lines.append("\n## Orphaned References\n")
        report_lines.append("The following references are not referenced by any other asset:\n\n")
        for orphan in sorted(orphans):
            report_lines.append(f"- `{orphan}`\n")
        report_lines.append("\n")

    # Unresolved cross-references
    unresolved = [cr for cr in results["cross_refs"] if cr["status"] == "UNRESOLVED"]
    if unresolved:
        report_lines.append("\n## Unresolved Cross-References\n")
        report_lines.append("The following cross-references could not be resolved:\n\n")
        for unres in sorted(unresolved, key=lambda x: x["source"]):
            report_lines.append(f"- Source: `{unres['source']}` → Reference: `{unres['reference']}`\n")
        report_lines.append("\n")

    # Write report
    report_path = BASE_DIR / "audit-report.md"
    report_path.write_text("".join(report_lines))
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    run_audit()
