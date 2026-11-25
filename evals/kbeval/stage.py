"""Stage the Claude tree into a kb or vanilla eval plugin.

`kb` carries the skills, commands, and agents, appends `.claude/CLAUDE.md` plus two harness notes
to every case's system prompt, and seeds references into each run's workspace. `vanilla` is an
empty plugin, which is plain Claude. See `docs/adr/workflow/kb-evals-run-on-a-staged-plugin.md`.

kb staging also copies `.claude/CLAUDE.md` and `evals/harness.json` into `_kb/`, and both stagings
write the prompt-note templates to `_kb/prompt-notes.txt`, so the runner's KB digest covers them; no
run reads those files.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional

from evals.kbeval.cases import FRONTMATTER, Case, discover

MARKER = ".kbeval-stage"
STAGINGS = ("kb", "vanilla")
NON_INTERACTIVE = (
    "Run non-interactively: do not ask questions; take the recommended option for every decision "
    "and state it."
)
AGENT_NOTE = "KB agents are registered as `kb:<name>`; dispatch them with that subagent_type."
REFS_NOTE = "KB references are at {where}<p>; they are not project files."
# The installed forms KB assets cite references by (see docs/adr/workflow/
# kb-reference-paths-never-resolve-against-the-project.md). A staged copy has no home tree, so
# each becomes the workspace copy the seed script writes.
INSTALLED_REFERENCE_PREFIXES = ("${CLAUDE_SKILL_DIR}/../../references/", "~/.claude/references/")
WORKSPACE_REFERENCES = "./.kb-references/"
AGENT_PREFIX = "Tell the agent: "
GIT_SHIM_DIR = "/usr/bin"  # macOS's git shim calls xcrun, which the run's sandbox blocks
GIT_NOTE = "In this sandbox `/usr/bin/git` fails; run git as `{git}`."
GIT_FIXTURE = re.compile(r"\bgit init\b")
COMMAND_DIRECTIVE = re.compile(r"^/kb:([a-z0-9-]+) ")
AGENT_DIRECTIVE = re.compile(r"^Use the kb:([a-z0-9-]+) agent to ")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


class StageError(Exception):
    """Raised when staging cannot proceed safely."""


def _hash_tree(digest, root: Path, repo: Path) -> None:
    if not root.is_dir():
        if root.is_file():
            digest.update(f"{root.relative_to(repo)}\0".encode())
            digest.update(root.read_bytes() + b"\0")
        return
    for path in sorted(root.rglob("*")):
        parts = path.relative_to(root).parts
        if not path.is_file() or "__pycache__" in parts or path.suffix == ".pyc":
            continue
        digest.update(f"{path.relative_to(repo)}\0".encode())
        digest.update(path.read_bytes() + b"\0")


def source_digest(repo: Path, staging: str, exclude_tags: frozenset) -> str:
    """Hash everything `stage()` reads, so a later edit to any of it is detectable."""
    digest = hashlib.sha256()
    digest.update(f"{staging}\0{','.join(sorted(exclude_tags))}\0".encode())
    paths = [
        repo / "evals" / "kbeval" / "stage.py",
        repo / "evals" / "kbeval" / "cases.py",
        repo / "evals" / "harness.json",
        repo / "evals" / "fixtures",
        repo / "evals" / "cases",
    ]
    if staging == "kb":
        claude = repo / ".claude"
        paths += [claude / n for n in ("skills", "commands", "agents", "references", "CLAUDE.md")]
    for path in paths:
        _hash_tree(digest, path, repo)
    return digest.hexdigest()


def marker_staging(plugin: Path) -> Optional[str]:
    """Return the staging (`kb` or `vanilla`) recorded in `plugin`'s stage marker, or None."""
    try:
        marker = json.loads((plugin.parent / MARKER).read_text(encoding="utf-8"))
        return str(marker["staging"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def staleness(repo: Path, plugin: Path) -> Optional[str]:
    """Return why `plugin` no longer matches the tree, or None when it does."""
    try:
        marker = json.loads((plugin.parent / MARKER).read_text(encoding="utf-8"))
        staged = marker["sourceDigest"]
        staging, tags = marker["staging"], frozenset(marker["excludeTags"])
    except (OSError, ValueError, KeyError, TypeError):
        return f"staged plugin {plugin} holds no source digest"
    tree = source_digest(repo, staging, tags)
    if tree == staged:
        return None
    return f"staged plugin {plugin} is stale (staged {staged[:8]}, tree {tree[:8]})"


def _localize_references(text: str) -> str:
    """Replace each installed reference prefix with the workspace copy's path."""
    for prefix in INSTALLED_REFERENCE_PREFIXES:
        text = text.replace(prefix, WORKSPACE_REFERENCES)
    return text


def stage(repo: Path, out: Path, staging: str, exclude_tags: frozenset = frozenset()) -> Path:
    """Build a staged plugin under `out` and return the plugin directory.

    `out` must be missing, or a previous staging (it holds the marker file); anything else is
    refused rather than deleted.
    """
    if staging not in STAGINGS:
        raise StageError(f"staging must be one of {', '.join(STAGINGS)}, not {staging!r}")
    if out.is_symlink() or (out.exists() and not out.is_dir()):
        raise StageError(f"{out} is a symlink or a file; refusing to stage there")
    out = out.resolve()
    sources = source_digest(repo, staging, exclude_tags)
    if out.exists():
        if any(out.iterdir()) and not (out / MARKER).is_file():
            raise StageError(
                f"{out} is not empty and holds no {MARKER} marker; refusing to replace it"
            )
        shutil.rmtree(out)
    out.mkdir(parents=True)
    marker = {"staging": staging, "excludeTags": sorted(exclude_tags), "sourceDigest": sources}
    (out / MARKER).write_text(json.dumps(marker) + "\n", encoding="utf-8")

    harness = json.loads((repo / "evals" / "harness.json").read_text(encoding="utf-8"))
    command_form = harness.get("commandForm", "slash")
    refs_mode = harness.get("refsMode", "workspace")

    plugin = out / "kb"
    (plugin / ".claude-plugin").mkdir(parents=True)
    manifest = {"name": staging, "version": "0.0.0", "description": f"KB eval staging: {staging}"}
    (plugin / ".claude-plugin" / "plugin.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    claude = repo / ".claude"
    refs = plugin / "_kb" / "references"
    if staging == "kb":
        for component in ("skills", "commands", "agents"):
            shutil.copytree(claude / component, plugin / component, ignore=IGNORE)
            for staged in sorted((plugin / component).rglob("*.md")):
                text = staged.read_text(encoding="utf-8")
                localized = _localize_references(text)
                if localized != text:
                    staged.write_text(localized, encoding="utf-8")
        if command_form == "skill":
            for command in (plugin / "commands").glob("*.md"):
                text = command.read_text(encoding="utf-8")
                command.write_text(
                    re.sub(r"(?m)^disable-model-invocation:.*\n", "", text), encoding="utf-8"
                )
        shutil.copytree(claude / "references", refs, ignore=IGNORE)
        shutil.copy2(claude / "CLAUDE.md", plugin / "_kb" / "CLAUDE.md")
        shutil.copy2(repo / "evals" / "harness.json", plugin / "_kb" / "harness.json")
    fixtures = plugin / "_kb" / "fixtures"
    if (repo / "evals" / "fixtures").is_dir():
        shutil.copytree(repo / "evals" / "fixtures", fixtures, ignore=IGNORE)
    else:
        fixtures.mkdir(parents=True)
    notes = (NON_INTERACTIVE, GIT_NOTE, AGENT_NOTE, REFS_NOTE, AGENT_PREFIX + REFS_NOTE)
    (plugin / "_kb" / "prompt-notes.txt").write_text(
        "".join(n + "\n" for n in notes), encoding="utf-8"
    )

    claude_md = (claude / "CLAUDE.md").read_text(encoding="utf-8") if staging == "kb" else ""
    suites = ("triggers", "effects", "workflows") if staging == "kb" else ("effects", "workflows")
    for case in discover(repo / "evals" / "cases"):
        if case.suite not in suites or exclude_tags & set(case.tags):
            continue
        if command_form == "skill" and case.kind == "trigger-noauto":
            continue
        dest = plugin / f"evals-{case.suite}" / case.name
        shutil.copytree(case.path, dest, ignore=IGNORE)
        add_refs = staging == "kb" and refs_mode == "add_dirs"
        if add_refs:
            shutil.copytree(refs, dest / "_refs")
        where = f"{dest / '_refs'}/" if add_refs else "./.kb-references/"
        system = None
        if staging == "kb":
            system = (
                claude_md.rstrip("\n") + "\n\n" + REFS_NOTE.format(where=where) + "\n" + AGENT_NOTE
            )
        _write_prompt(case, dest, staging, command_form, system, where)
        _write_case_yaml(case, dest, add_refs)
        seed_refs = refs if staging == "kb" and refs_mode == "workspace" else None
        _write_seed(case, dest, fixtures, seed_refs)
    return plugin


def working_git(path: Optional[str] = None) -> Optional[str]:
    """Return the first git on `path` (default: PATH) other than the system shim."""
    for entry in (os.environ.get("PATH", "") if path is None else path).split(os.pathsep):
        candidate = os.path.join(entry, "git")
        if entry and entry != GIT_SHIM_DIR and os.access(candidate, os.X_OK):
            return candidate
    return None


def _transform_body(case: Case, staging: str, command_form: str, where: str) -> str:
    body = case.body
    if not (case.suite == "workflows" or case.kind == "effect"):
        return body
    if staging == "vanilla":
        stripped = AGENT_DIRECTIVE.sub("", COMMAND_DIRECTIVE.sub("", body, count=1), count=1)
        if stripped != body:
            body = stripped[:1].upper() + stripped[1:]
    elif command_form == "skill":
        body = COMMAND_DIRECTIVE.sub(r"Use the kb:\1 command to: ", body, count=1)
    # A dispatched agent does not see the session's system prompt, so the note goes in the task.
    if staging == "kb" and AGENT_DIRECTIVE.search(case.body):
        body += "\n\n" + AGENT_PREFIX + REFS_NOTE.format(where=where)
    # The CLI builds the run's PATH itself, so git cases name a working git in both arms' prompts.
    git = working_git() if GIT_FIXTURE.search(case.fixture or "") else None
    if git:
        body += "\n\n" + GIT_NOTE.format(git=git)
    return body + "\n\n" + NON_INTERACTIVE


def _write_prompt(
    case: Case, dest: Path, staging: str, command_form: str, system, where: str
) -> None:
    text = (case.path / "prompt.md").read_text(encoding="utf-8")
    match = FRONTMATTER.match(text)
    lines: List[str] = match.group(1).splitlines() if match else []
    if system is not None:
        lines.append("append_system_prompt: " + json.dumps(system, ensure_ascii=True))
    body = _transform_body(case, staging, command_form, where)
    (dest / "prompt.md").write_text(
        "---\n" + "\n".join(lines) + "\n---\n\n" + body + "\n", encoding="utf-8"
    )


def _write_case_yaml(case: Case, dest: Path, add_refs: bool) -> None:
    context: List[str] = []
    if case.case_yaml is not None:
        inside = False
        for line in case.case_yaml.splitlines():
            if line and not line[0].isspace():
                inside = line.split(":", 1)[0].strip() == "context"
                continue
            if inside and line.strip():
                context.append("  " + line.strip())
    if add_refs:
        for index, line in enumerate(context):
            if line.startswith("  add_dirs:"):
                items = line.split(":", 1)[1].strip().strip("[]")
                listed = [item.strip() for item in items.split(",") if item.strip()]
                context[index] = "  add_dirs: [" + ", ".join([*listed, "_refs"]) + "]"
                break
        else:
            context.append("  add_dirs: [_refs]")
    text = (
        'schema_version: "1.1"\n'
        f"name: {case.name}\n"
        "context:\n"
        "  scaffold_script: seed.sh\n" + "".join(line + "\n" for line in context)
    )
    (dest / "case.yaml").write_text(text, encoding="utf-8")


def _quote(path: Path) -> str:
    return "'" + str(path).replace("'", "'\\''") + "'"


def _write_seed(case: Case, dest: Path, fixtures: Path, refs) -> None:
    # The fixture runs as the user, so keep the user's git config (signing, hooks, templates) out.
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1",
    ]
    if case.fixture is not None:
        lines.append(f"bash {_quote(dest / 'fixture.sh')} {_quote(fixtures)}")
        # Running Python in the workspace must not show up as an untracked change.
        lines.append(
            "if [ -d .git ]; then mkdir -p .git/info; printf '%s\\n' __pycache__/ >> .git/info/exclude; fi"
        )
    if refs is not None:
        lines.append("mkdir -p .kb-references")
        lines.append(f"cp -R {_quote(refs)}/. .kb-references/")
        lines.append(
            "if [ -d .git ]; then mkdir -p .git/info; printf '%s\\n' .kb-references >> "
            ".git/info/exclude; fi"
        )
    seed = dest / "seed.sh"
    seed.write_text("\n".join(lines) + "\n", encoding="utf-8")
    seed.chmod(0o755)
