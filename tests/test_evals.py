"""Tests for the KB eval harness in evals/kbeval.

Every test here is free: nothing launches `claude`. Staging and lint run on a synthetic repo built in
a temp dir, so the suite does not depend on how many cases the real tree holds; the two real-tree
tests at the end run lint and coverage on the repo itself.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

from evals.kbeval import cases as kcases
from evals.kbeval import headroom as kheadroom
from evals.kbeval import stage as kstage

REPO = Path(__file__).resolve().parents[1]

NON_INTERACTIVE = (
    "Run non-interactively: do not ask questions; take the recommended option for every decision "
    "and state it."
)
TRIGGER_FM = (
    "tags: [triggers, {kind}]\nmax_turns: 6\ntimeout_seconds: 120\n"
    "allowed_tools: [Skill, Read, Glob, Grep]\n"
)
EFFECT_FM = (
    "tags: [effects, effect]\nmax_turns: 40\ntimeout_seconds: 900\n"
    "allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]\n"
)
WORKFLOW_FM = (
    "tags: [workflows, heavy]\nmax_turns: 150\ntimeout_seconds: 3600\n"
    "allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]\n"
)
LLM_RESULT = (
    "---\ntype: llm\nfocus: { source: file, path: design.md }\n---\n\n"
    "PASS if design.md names a cache eviction policy.\n"
    "FAIL if design.md has no invalidation section.\n"
)


def skill_fired(name: str, minimum: int = 1, maximum: int | None = None) -> str:
    bound = f"min: {minimum}\n" + (f"max: {maximum}\n" if maximum is not None else "")
    return (
        "---\ntype: tool_used\ntool: Skill\n"
        f'input_match: \'"skill"\\s*:\\s*"(?:kb:)?{name}"\'\n{bound}---\n'
    )


def agent_dispatched(name: str) -> str:
    return (
        "---\ntype: tool_used\ntool: Agent\n"
        f'input_match: \'"subagent_type"\\s*:\\s*"kb:{name}"\'\n---\n'
    )


def make_repo(root: Path) -> Path:
    """Build a minimal repo: one skill, two commands (one user-only), one agent, one reference."""
    files = {
        ".claude/CLAUDE.md": '# KB rules\n\nSay "hi" and use \\n literally.\n',
        ".claude/skills/demo-skill/SKILL.md": "---\nname: demo-skill\ndescription: Use when demoing.\n---\n\nBody.\n",
        ".claude/skills/demo-skill/scripts/run.py": "print('x')\n",
        ".claude/commands/j-demo.md": "---\nname: j-demo\ndescription: Demo. Use when demoing.\n---\n\nDo the demo.\n",
        ".claude/commands/j-quiet.md": (
            "---\nname: j-quiet\ndescription: Quiet. Use when asked.\n"
            "disable-model-invocation: true\n---\n\nDo it quietly.\n"
        ),
        ".claude/agents/demo-agent.md": "---\nname: demo-agent\ndescription: Use when demoing.\nmodel: sonnet\ntools: Read\n---\n\nReview things.\n",
        ".claude/references/cat/ref.md": "# Ref\n",
        "evals/harness.json": '{"commandForm": "slash", "refsMode": "workspace"}\n',
        "evals/fixtures/py-calc/calc/ops.py": "def add(a, b):\n    return a + b\n",
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / ".claude/skills/demo-skill/__pycache__").mkdir()
    (root / ".claude/skills/demo-skill/__pycache__/run.cpython-312.pyc").write_bytes(b"\0")
    return root


def case_description(root: Path, suite: str, name: str) -> str:
    """A valid `description` for a case in the temp repo, with the ending its kind needs."""
    asset = name.split("--")[0]
    paths = [
        ".claude/skills/%s/SKILL.md" % asset,
        ".claude/commands/%s.md" % asset,
        ".claude/agents/%s.md" % asset,
    ]
    path = next((p for p in paths if (root / p).is_file()), ".claude/commands/j-demo.md")
    if suite == "triggers":
        ending = (
            "near: none"
            if "trigger-near" in name
            else "noauto"
            if "trigger-noauto" in name
            else "should"
        )
    else:
        ending = "hard because a capable model without the asset would miss it"
    return f"the demo rule, as a clause; {path}; {ending}"


def write_case(
    root: Path,
    suite: str,
    name: str,
    frontmatter: str,
    body: str,
    graders: dict,
    fixture: str | None = None,
    case_yaml: str | None = None,
) -> Path:
    folder = root / "evals" / "cases" / suite / name
    (folder / "graders").mkdir(parents=True)
    if not re.search(r"^description:", frontmatter, re.MULTILINE):
        frontmatter = f'description: "{case_description(root, suite, name)}"\n{frontmatter}'
    (folder / "prompt.md").write_text(f"---\n{frontmatter}---\n\n{body}\n")
    for grader, text in graders.items():
        (folder / "graders" / f"{grader}.md").write_text(text)
    if fixture is not None:
        (folder / "fixture.sh").write_text(fixture)
    if case_yaml is not None:
        (folder / "case.yaml").write_text(case_yaml)
    return folder


def valid_cases(root: Path) -> None:
    """One lint-clean case of each shape."""
    write_case(
        root,
        "triggers",
        "demo-skill--trigger-should-rate",
        TRIGGER_FM.format(kind="trigger-should"),
        "Help me design a rate limiter for my API.",
        {"fired": skill_fired("demo-skill")},
    )
    write_case(
        root,
        "triggers",
        "demo-skill--trigger-near-typo",
        TRIGGER_FM.format(kind="trigger-near"),
        "Fix the typo in README.",
        {"fired": skill_fired("demo-skill", 0, 0)},
    )
    write_case(
        root,
        "triggers",
        "j-quiet--trigger-noauto-ask",
        TRIGGER_FM.format(kind="trigger-noauto"),
        "Do the quiet thing for me.",
        {"fired": skill_fired("j-quiet", 0, 0)},
    )
    write_case(
        root,
        "effects",
        "j-demo--effect-cache",
        EFFECT_FM,
        "/kb:j-demo design a cache layer and save it to ./design.md.",
        {"result": LLM_RESULT, "fired": skill_fired("j-demo")},
        fixture='set -euo pipefail\ncp -R "$1/py-calc/." .\ngit init -q -b main\nmkdir -p plans\n',
        case_yaml="context:\n  add_dirs: [resources]\n",
    )
    write_case(
        root,
        "effects",
        "demo-agent--effect-review",
        EFFECT_FM,
        "Use the kb:demo-agent agent to review the cache design and save it to ./design.md.",
        {"result": LLM_RESULT, "dispatched": agent_dispatched("demo-agent")},
    )
    write_case(
        root,
        "effects",
        "demo-skill--effect-limiter",
        EFFECT_FM,
        "Design a rate limiter and save the design to ./design.md.",
        {"result": LLM_RESULT, "fired": skill_fired("demo-skill")},
    )
    write_case(
        root,
        "workflows",
        "wf--demo-chain",
        WORKFLOW_FM,
        "/kb:j-demo run the whole chain and save the design to ./design.md.",
        {"result": LLM_RESULT, "fired": skill_fired("j-demo")},
    )


def frontmatter_of(path: Path) -> str:
    match = re.match(r"\A---\n(.*?)\n---\n", path.read_text(), re.DOTALL)
    assert match is not None, path
    return match.group(1)


def body_of(path: Path) -> str:
    return re.sub(r"\A---\n.*?\n---\n", "", path.read_text(), count=1, flags=re.DOTALL).strip()


class TempRepoCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.repo = make_repo(self.tmp / "repo")
        valid_cases(self.repo)

    def set_harness(self, **values):
        config = {"commandForm": "slash", "refsMode": "workspace", **values}
        (self.repo / "evals" / "harness.json").write_text(json.dumps(config))

    def stage(self, staging: str = "kb", exclude_tags: frozenset = frozenset()) -> Path:
        return kstage.stage(self.repo, self.tmp / f"out-{staging}", staging, exclude_tags)


class StageTests(TempRepoCase):
    def test_stage_refuses_unmarked_nonempty_out(self):
        out = self.tmp / "busy"
        out.mkdir()
        (out / "keep.txt").write_text("mine")
        with self.assertRaises(kstage.StageError):
            kstage.stage(self.repo, out, "kb")
        self.assertTrue((out / "keep.txt").is_file())

    def test_stage_kb_layout(self):
        plugin = self.stage()
        manifest = json.loads((plugin / ".claude-plugin" / "plugin.json").read_text())
        self.assertEqual(manifest["name"], "kb")
        self.assertEqual(sorted(p.name for p in (plugin / "skills").iterdir()), ["demo-skill"])
        self.assertFalse((plugin / "skills" / "demo-skill" / "__pycache__").exists())
        self.assertEqual(
            sorted(p.name for p in (plugin / "commands").iterdir()), ["j-demo.md", "j-quiet.md"]
        )
        self.assertEqual(sorted(p.name for p in (plugin / "agents").iterdir()), ["demo-agent.md"])
        self.assertTrue((plugin / "_kb" / "references" / "cat" / "ref.md").is_file())
        self.assertTrue((plugin / "_kb" / "fixtures" / "py-calc" / "calc" / "ops.py").is_file())
        for suite, case in (
            ("triggers", "demo-skill--trigger-should-rate"),
            ("effects", "j-demo--effect-cache"),
            ("workflows", "wf--demo-chain"),
        ):
            self.assertTrue((plugin / f"evals-{suite}" / case / "prompt.md").is_file(), case)
        self.assertTrue((plugin.parent / ".kbeval-stage").is_file())

    def test_kb_staging_copies_claude_md_and_harness_into_kb(self):
        plugin = self.stage()
        self.assertEqual(
            (plugin / "_kb" / "CLAUDE.md").read_text(),
            (self.repo / ".claude" / "CLAUDE.md").read_text(),
        )
        self.assertEqual(
            (plugin / "_kb" / "harness.json").read_text(),
            (self.repo / "evals" / "harness.json").read_text(),
        )
        vanilla = self.stage("vanilla")
        self.assertFalse((vanilla / "_kb" / "CLAUDE.md").exists())
        self.assertFalse((vanilla / "_kb" / "harness.json").exists())

    def test_restage_over_marked_out_replaces_it(self):
        plugin = self.stage()
        (plugin / "stale.txt").write_text("old")
        plugin = self.stage()
        self.assertFalse((plugin / "stale.txt").exists())

    def test_stage_vanilla_layout(self):
        plugin = self.stage("vanilla")
        self.assertTrue((plugin / ".claude-plugin" / "plugin.json").is_file())
        for component in ("skills", "commands", "agents", "_kb/references"):
            self.assertFalse((plugin / component).exists(), component)
        self.assertFalse((plugin / "evals-triggers").exists())
        prompt = plugin / "evals-effects" / "j-demo--effect-cache" / "prompt.md"
        self.assertNotIn("append_system_prompt", frontmatter_of(prompt))
        seed = (prompt.parent / "seed.sh").read_text()
        self.assertNotIn("references", seed)

    def test_stage_kb_injects_json_encoded_system_prompt(self):
        plugin = self.stage()
        for suite in ("triggers", "effects", "workflows"):
            for prompt in (plugin / f"evals-{suite}").glob("*/prompt.md"):
                line = next(
                    ln
                    for ln in frontmatter_of(prompt).splitlines()
                    if ln.startswith("append_system_prompt: ")
                )
                text = json.loads(line[len("append_system_prompt: ") :])
                self.assertTrue(text.startswith((self.repo / ".claude/CLAUDE.md").read_text()))
                self.assertIn(
                    "KB references cited as `.claude/references/<p>` or `references/<p>` are at "
                    "./.kb-references/<p>; they are not project files.",
                    text,
                )
                self.assertIn(
                    "KB agents are registered as `kb:<name>`; dispatch them with that subagent_type.",
                    text,
                )
        first = {p.relative_to(plugin): p.read_bytes() for p in plugin.rglob("*") if p.is_file()}
        plugin = self.stage()
        second = {p.relative_to(plugin): p.read_bytes() for p in plugin.rglob("*") if p.is_file()}
        self.assertEqual(first, second)

    def test_agent_case_prompt_carries_the_reference_path_note(self):
        note = (
            "Tell the agent: KB references cited as `.claude/references/<p>` or `references/<p>` "
            "are at ./.kb-references/<p>; they are not project files."
        )
        plugin = self.stage()
        effects = plugin / "evals-effects"
        self.assertIn(note, body_of(effects / "demo-agent--effect-review" / "prompt.md"))
        self.assertNotIn("Tell the agent:", body_of(effects / "j-demo--effect-cache" / "prompt.md"))
        vanilla = self.stage("vanilla") / "evals-effects"
        self.assertNotIn(
            "Tell the agent:", body_of(vanilla / "demo-agent--effect-review" / "prompt.md")
        )

    def test_seed_runs_in_minimal_env(self):
        plugin = self.stage()
        case = plugin / "evals-effects" / "j-demo--effect-cache"
        seed = (case / "seed.sh").read_text()
        for path in re.findall(r"'(/[^']*)'", seed):
            self.assertTrue(Path(path).is_absolute())
        work = self.tmp / "work"
        work.mkdir()
        home = self.tmp / "home"
        home.mkdir()
        env = {"PATH": os.environ["PATH"], "HOME": str(home), "TMPDIR": str(self.tmp)}
        result = subprocess.run(
            ["bash", str(case / "seed.sh")],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((work / "calc" / "ops.py").is_file())
        self.assertTrue((work / ".kb-references" / "cat" / "ref.md").is_file())
        self.assertIn(".kb-references", (work / ".git" / "info" / "exclude").read_text())
        self.assertIn("__pycache__/", (work / ".git" / "info" / "exclude").read_text())
        self.assertLess(seed.index("fixture.sh"), seed.index(".kb-references"))
        vanilla = (
            self.stage("vanilla") / "evals-effects" / "j-demo--effect-cache" / "seed.sh"
        ).read_text()
        self.assertIn("__pycache__/", vanilla)

    def test_case_yaml_has_schema_name_and_scaffold(self):
        plugin = self.stage()
        text = (plugin / "evals-effects" / "j-demo--effect-cache" / "case.yaml").read_text()
        self.assertIn('schema_version: "1.1"\n', text)
        self.assertIn("name: j-demo--effect-cache\n", text)
        self.assertIn("  scaffold_script: seed.sh\n", text)
        self.assertIn("  add_dirs: [resources]\n", text)
        self.assertEqual(text.count("context:"), 1)
        trigger = plugin / "evals-triggers" / "demo-skill--trigger-should-rate" / "case.yaml"
        self.assertIn("name: demo-skill--trigger-should-rate\n", trigger.read_text())

    def test_stage_appends_non_interactive_line(self):
        for staging in ("kb", "vanilla"):
            plugin = self.stage(staging)
            for suite in ("effects", "workflows"):
                for prompt in (plugin / f"evals-{suite}").glob("*/prompt.md"):
                    self.assertTrue(body_of(prompt).endswith(NON_INTERACTIVE), (staging, prompt))
        plugin = self.stage()
        trigger = plugin / "evals-triggers" / "demo-skill--trigger-should-rate" / "prompt.md"
        self.assertEqual(body_of(trigger), "Help me design a rate limiter for my API.")

    def test_git_cases_name_a_working_git_in_both_arms(self):
        note = kstage.GIT_NOTE.format(git="/brew/bin/git")
        with mock.patch.object(kstage, "working_git", return_value="/brew/bin/git"):
            for staging in ("kb", "vanilla"):
                plugin = self.stage(staging)
                git_case = body_of(plugin / "evals-effects" / "j-demo--effect-cache" / "prompt.md")
                self.assertTrue(
                    git_case.endswith(note + "\n\n" + NON_INTERACTIVE), (staging, git_case)
                )
                plain = body_of(
                    plugin / "evals-effects" / "demo-skill--effect-limiter" / "prompt.md"
                )
                self.assertNotIn(note, plain)
        with mock.patch.object(kstage, "working_git", return_value=None):
            plugin = self.stage()
            git_case = body_of(plugin / "evals-effects" / "j-demo--effect-cache" / "prompt.md")
            self.assertNotIn("/usr/bin/git", git_case)

    def test_working_git_skips_the_system_shim(self):
        brew = self.tmp / "brew-bin"
        brew.mkdir()
        (brew / "git").write_text("#!/bin/sh\n")
        (brew / "git").chmod(0o755)
        path = os.pathsep.join(["/nowhere", "/usr/bin", str(brew)])
        self.assertEqual(kstage.working_git(path), str(brew / "git"))
        self.assertIsNone(kstage.working_git(os.pathsep.join(["/nowhere", "/usr/bin"])))

    def test_stage_vanilla_strips_kb_directives(self):
        plugin = self.stage("vanilla")
        command = body_of(plugin / "evals-effects" / "j-demo--effect-cache" / "prompt.md")
        self.assertTrue(command.startswith("Design a cache layer"), command)
        agent = body_of(plugin / "evals-effects" / "demo-agent--effect-review" / "prompt.md")
        self.assertTrue(agent.startswith("Review the cache design"), agent)
        self.assertNotIn("kb:", command + agent)

    def test_stage_command_form_skill(self):
        self.set_harness(commandForm="skill")
        plugin = self.stage()
        body = body_of(plugin / "evals-effects" / "j-demo--effect-cache" / "prompt.md")
        self.assertTrue(body.startswith("Use the kb:j-demo command to: design a cache layer"), body)
        self.assertNotIn(
            "disable-model-invocation", (plugin / "commands" / "j-quiet.md").read_text()
        )
        self.assertFalse((plugin / "evals-triggers" / "j-quiet--trigger-noauto-ask").exists())
        self.assertTrue((plugin / "evals-triggers" / "demo-skill--trigger-near-typo").exists())

    def test_stage_refs_mode_add_dirs(self):
        self.set_harness(refsMode="add_dirs")
        plugin = self.stage()
        case = plugin / "evals-effects" / "demo-skill--effect-limiter"
        self.assertTrue((case / "_refs" / "cat" / "ref.md").is_file())
        self.assertIn("  add_dirs: [_refs]\n", (case / "case.yaml").read_text())
        merged = (plugin / "evals-effects" / "j-demo--effect-cache" / "case.yaml").read_text()
        self.assertIn("  add_dirs: [resources, _refs]\n", merged)
        line = next(
            ln
            for ln in frontmatter_of(case / "prompt.md").splitlines()
            if ln.startswith("append_system_prompt: ")
        )
        self.assertIn(str(case / "_refs"), json.loads(line[len("append_system_prompt: ") :]))
        self.assertNotIn(".kb-references", (case / "seed.sh").read_text())

    def test_agent_note_uses_the_refs_path_in_add_dirs_mode(self):
        self.set_harness(refsMode="add_dirs")
        plugin = self.stage()
        case = plugin / "evals-effects" / "demo-agent--effect-review"
        note = kstage.AGENT_PREFIX + kstage.REFS_NOTE.format(where=f"{case / '_refs'}/")
        body = body_of(case / "prompt.md")
        self.assertIn(note, body)
        self.assertNotIn(".kb-references", body)

    def test_prompt_notes_are_written_to_kb_in_both_stagings(self):
        expected = [
            kstage.NON_INTERACTIVE,
            kstage.GIT_NOTE,
            kstage.AGENT_NOTE,
            kstage.REFS_NOTE,
            "Tell the agent: " + kstage.REFS_NOTE,
        ]
        for staging in ("kb", "vanilla"):
            notes = self.stage(staging) / "_kb" / "prompt-notes.txt"
            self.assertTrue(notes.is_file(), staging)
            self.assertEqual(notes.read_text().splitlines(), expected, staging)

    def test_kb_digest_changes_when_a_prompt_note_changes(self):
        before_plugin = kstage.stage(self.repo, self.tmp / "out-before", "kb")
        with mock.patch.object(kstage, "NON_INTERACTIVE", "Run without asking."):
            after_plugin = kstage.stage(self.repo, self.tmp / "out-after", "kb")

        def outside_cases(plugin: Path) -> dict:
            return {
                p.relative_to(plugin): p.read_bytes()
                for p in plugin.rglob("*")
                if p.is_file() and not p.relative_to(plugin).parts[0].startswith("evals-")
            }

        before, after = outside_cases(before_plugin), outside_cases(after_plugin)
        changed = sorted(str(rel) for rel in before if before[rel] != after.get(rel))
        self.assertEqual(changed, ["_kb/prompt-notes.txt"])
        self.assertNotEqual(
            krunner._tree_digest(before_plugin, skip_prefix="evals-"),
            krunner._tree_digest(after_plugin, skip_prefix="evals-"),
        )

    def test_stage_exclude_tags(self):
        folder = write_case(
            self.repo,
            "effects",
            "demo-skill--effect-second-mode",
            EFFECT_FM.replace("[effects, effect]", "[effects, effect, extended]"),
            "Review a rate limiter and save notes to ./design.md.",
            {"result": LLM_RESULT, "fired": skill_fired("demo-skill")},
        )
        plugin = self.stage(exclude_tags=frozenset({"extended"}))
        self.assertFalse((plugin / "evals-effects" / folder.name).exists())
        self.assertTrue((plugin / "evals-effects" / "demo-skill--effect-limiter").exists())


class LintTests(TempRepoCase):
    def problems(self) -> list:
        return kcases.lint(kcases.discover(self.repo / "evals" / "cases"), self.repo)

    def assert_problem(self, pattern: str) -> None:
        problems = self.problems()
        self.assertTrue(
            any(re.search(pattern, p) for p in problems),
            f"no problem matching {pattern!r} in {problems}",
        )

    def edit(self, rel: str, old: str, new: str) -> None:
        path = self.repo / "evals" / "cases" / rel
        text = path.read_text()
        self.assertIn(old, text)
        path.write_text(text.replace(old, new))

    def test_valid_cases_lint_clean(self):
        self.assertEqual(self.problems(), [])

    def test_discover_finds_every_case(self):
        found = kcases.discover(self.repo / "evals" / "cases")
        self.assertEqual(len(found), 7)
        effect = next(c for c in found if c.name == "j-demo--effect-cache")
        self.assertEqual((effect.suite, effect.asset, effect.kind), ("effects", "j-demo", "effect"))

    def test_lint_trigger_needs_exactly_one_skill_grader(self):
        (
            self.repo / "evals/cases/triggers/demo-skill--trigger-should-rate/graders/extra.md"
        ).write_text(LLM_RESULT)
        self.assert_problem(r"demo-skill--trigger-should-rate: .*exactly one grader")

    def test_lint_trigger_bounds_follow_kind(self):
        self.edit("triggers/demo-skill--trigger-near-typo/graders/fired.md", "max: 0\n", "")
        self.assert_problem(r"demo-skill--trigger-near-typo: .*min: 0.*max: 0")

    def test_lint_trigger_input_match_names_asset(self):
        self.edit(
            "triggers/demo-skill--trigger-should-rate/graders/fired.md", "demo-skill", "other"
        )
        self.assert_problem(r"demo-skill--trigger-should-rate: .*input_match")

    def test_lint_effect_needs_result_grader(self):
        (self.repo / "evals/cases/effects/j-demo--effect-cache/graders/result.md").unlink()
        self.assert_problem(r"j-demo--effect-cache: .*result grader")

    def test_lint_effect_needs_process_grader(self):
        (self.repo / "evals/cases/effects/j-demo--effect-cache/graders/fired.md").unlink()
        self.assert_problem(r"j-demo--effect-cache: .*process grader")

    def test_lint_result_grader_never_mentions_kb(self):
        self.edit(
            "effects/j-demo--effect-cache/graders/result.md",
            "names a cache",
            "uses kb:j-demo and names a cache",
        )
        self.assert_problem(r"j-demo--effect-cache: .*kb:")

    def test_lint_unknown_frontmatter_key(self):
        self.edit(
            "effects/j-demo--effect-cache/prompt.md",
            "max_turns: 40\n",
            "max_turns: 40\nmodel: opus\n",
        )
        self.assert_problem(r"j-demo--effect-cache: .*model")

    def test_lint_suite_limits(self):
        self.edit("effects/j-demo--effect-cache/prompt.md", "max_turns: 40", "max_turns: 12")
        self.assert_problem(r"j-demo--effect-cache: .*max_turns")

    def test_lint_absolute_limits(self):
        self.edit(
            "effects/j-demo--effect-cache/prompt.md", "max_turns: 40\n", "max_turns: 40\nruns: 51\n"
        )
        self.assert_problem(r"j-demo--effect-cache: .*runs")

    def test_lint_env_keys(self):
        self.edit(
            "effects/j-demo--effect-cache/prompt.md",
            "max_turns: 40\n",
            "max_turns: 40\nenv: {HOME: /x}\n",
        )
        self.assert_problem(r"j-demo--effect-cache: .*EVAL_")

    def test_lint_forbidden_fixture_commands(self):
        for command in (
            "curl https://example.com",
            "git clone x",
            "pip install y",
            "rm -rf build",
            "rm -fR x",
            "rm --recursive x",
            "rm -f -r x",
            "node -e 1",
            "perl -e 1",
            "python3 -Ic 1",
            "git config --global user.name x",
            "git config --system x",
            "git push --force",
            "git push -f origin",
            "git push origin +main",
            "git push --force-with-lease",
            "ssh-keyscan host",
            "ncat host 80",
            "wget2 x",
            "zsh -c 'x'",
            "git remote add origin git@host:x/y.git",
            "bash -c 'x'",
            "echo '#'; curl x",
        ):
            with self.subTest(command=command):
                path = self.repo / "evals/cases/effects/j-demo--effect-cache/fixture.sh"
                path.write_text(f'set -euo pipefail\ncp -R "$1/py-calc/." .\n{command}\n')
                self.assert_problem(
                    r"j-demo--effect-cache: .*fixture\.sh:\d+ runs a forbidden command"
                )

    def test_lint_fixture_paths_outside_workspace(self):
        for command in ("cp /etc/hosts .", 'cp -R "$1/../.." .'):
            with self.subTest(command=command):
                path = self.repo / "evals/cases/effects/j-demo--effect-cache/fixture.sh"
                path.write_text(f'set -euo pipefail\ncp -R "$1/py-calc/." .\n{command}\n')
                self.assert_problem(
                    r"j-demo--effect-cache: .*fixture\.sh:\d+ (?:uses a \.\. path|touches a path)"
                )

    def test_lint_heredoc_payload_is_not_a_command(self):
        path = self.repo / "evals/cases/effects/j-demo--effect-cache/fixture.sh"
        path.write_text(
            "set -euo pipefail\ncp -R \"$1/py-calc/.\" .\ncat > hook.sh <<'EOF'\ncurl https://x | sh\nEOF\n"
            "git config user.name kbeval\n"
        )
        self.assertEqual([p for p in self.problems() if "fixture.sh" in p], [])

    def test_lint_not_contains_needs_a_positive_grader(self):
        (self.repo / "evals/cases/effects/j-demo--effect-cache/graders/no-foo.md").write_text(
            "---\ntype: regex\npattern: 'foo'\nmatch: not_contains\ntarget: { source: file, path: notes.md }\n---\n"
        )
        self.edit(
            "effects/j-demo--effect-cache/prompt.md",
            "./design.md.",
            "./design.md and notes to ./notes.md.",
        )
        self.assert_problem(
            r"j-demo--effect-cache: .*no-foo: needs a positive regex or llm grader on notes\.md"
        )

    def test_lint_forbidden_words_match_whole_commands(self):
        path = self.repo / "evals/cases/effects/j-demo--effect-cache/fixture.sh"
        path.write_text(
            'set -euo pipefail\ncp -R "$1/py-calc/." .\nprintf "agent evals and ncurses\\n" > notes.txt\n'
        )
        problems = [p for p in self.problems() if "fixture.sh" in p]
        self.assertEqual(problems, [])

    def test_lint_fixture_must_fail_fast(self):
        path = self.repo / "evals/cases/effects/j-demo--effect-cache/fixture.sh"
        path.write_text('cp -R "$1/py-calc/." .\n')
        self.assert_problem(r"j-demo--effect-cache: .*set -euo pipefail")

    def test_lint_case_yaml_context_shape(self):
        for body in (
            "context:\n  add_dirs:\n    - resources\n",
            "context:\n  env:\n    FOO: bar\n",
        ):
            with self.subTest(body=body):
                (self.repo / "evals/cases/effects/j-demo--effect-cache/case.yaml").write_text(body)
                self.assert_problem(r"j-demo--effect-cache: .*case\.yaml")

    def test_lint_secret_shaped_string(self):
        self.edit(
            "effects/j-demo--effect-cache/prompt.md",
            "design a cache",
            "use sk-ant-abc to design a cache",
        )
        self.assert_problem(r"j-demo--effect-cache: .*sk-ant-")

    def test_lint_author_scaffold_script(self):
        self.edit(
            "effects/j-demo--effect-cache/case.yaml",
            "context:\n",
            "context:\n  scaffold_script: mine.sh\n",
        )
        self.assert_problem(r"j-demo--effect-cache: .*scaffold_script")

    def test_lint_name_prefix_names_a_real_asset(self):
        source = self.repo / "evals/cases/effects/demo-skill--effect-limiter"
        source.rename(source.parent / "no-such-asset--effect-limiter")
        self.assert_problem(r"no-such-asset--effect-limiter: .*asset")

    def test_lint_llm_grader_needs_pass_and_fail_claims(self):
        self.edit("effects/j-demo--effect-cache/graders/result.md", "FAIL if", "Also")
        self.assert_problem(r"j-demo--effect-cache: .*PASS if.*FAIL if")

    def test_lint_command_effect_prefix(self):
        self.edit("effects/j-demo--effect-cache/prompt.md", "/kb:j-demo design", "Please design")
        self.assert_problem(r"j-demo--effect-cache: .*/kb:j-demo ")

    def test_lint_agent_effect_prefix(self):
        self.edit(
            "effects/demo-agent--effect-review/prompt.md",
            "Use the kb:demo-agent agent to review",
            "Review",
        )
        self.assert_problem(r"demo-agent--effect-review: .*Use the kb:demo-agent agent to ")

    def test_lint_skill_prompt_never_names_skill(self):
        self.edit(
            "effects/demo-skill--effect-limiter/prompt.md",
            "Design a rate",
            "Use demo-skill to design a rate",
        )
        self.assert_problem(r"demo-skill--effect-limiter: .*names the skill")

    def test_lint_at_most_one_extended_case_per_asset(self):
        for slug in ("second", "third"):
            write_case(
                self.repo,
                "effects",
                f"demo-skill--effect-{slug}",
                EFFECT_FM.replace("[effects, effect]", "[effects, effect, extended]"),
                "Review a rate limiter and save notes to ./design.md.",
                {"result": LLM_RESULT, "fired": skill_fired("demo-skill")},
            )
        self.assert_problem(r"demo-skill: .*extended")

    def test_lint_malformed_quoted_scalar(self):
        self.edit(
            "effects/j-demo--effect-cache/graders/fired.md", "input_match: '", "input_match: 'it's "
        )
        self.assert_problem(r"j-demo--effect-cache: .*malformed quoted YAML")

    def test_lint_js_incompatible_regex(self):
        for pattern in ("'(?m)^Verdict'", "'(?i:verdict)'", "'\\Averdict'"):
            with self.subTest(pattern=pattern):
                (self.repo / "evals/cases/effects/j-demo--effect-cache/graders/re.md").write_text(
                    f"---\ntype: regex\npattern: {pattern}\ntarget: {{ source: file, path: design.md }}\n---\n"
                )
                self.assert_problem(r"j-demo--effect-cache: .*inline flags")

    def test_lint_mid_file_heading_anchor_needs_multiline(self):
        grader = self.repo / "evals/cases/effects/j-demo--effect-cache/graders/re.md"
        grader.write_text(
            "---\ntype: regex\npattern: '^## Risks'\ntarget: { source: file, path: design.md }\n---\n"
        )
        self.assert_problem(r"j-demo--effect-cache: grader re: .*`flags: m`")
        grader.write_text(
            "---\ntype: regex\npattern: '^## Risks'\nflags: m\ntarget: { source: file, path: design.md }\n---\n"
        )
        self.assertEqual([p for p in self.problems() if "flags: m" in p], [])

    def test_lint_graded_file_under_dot_claude(self):
        self.edit(
            "effects/j-demo--effect-cache/graders/result.md",
            "path: design.md",
            "path: .claude/settings.json",
        )
        self.assert_problem(r"j-demo--effect-cache: .*under \.claude/")

    def test_lint_valid_quoted_scalars_pass(self):
        for value in (
            "\"Merge branch 'feat/rst'\"",
            "'a''b'",
            '"tab\\tend"',
            "{ source: file, path: 'x.md' }",
        ):
            with self.subTest(value=value):
                self.assertFalse(kcases.malformed_scalar(value))

    def test_lint_invalid_double_quoted_escape(self):
        for value in ('"\\d+"', "{ tool: Write, input_match: 'it's' }"):
            with self.subTest(value=value):
                self.assertTrue(kcases.malformed_scalar(value))

    def test_lint_js_regex_in_input_match_and_flags(self):
        self.edit(
            "effects/j-demo--effect-cache/graders/fired.md",
            "input_match: '",
            "input_match: '(?P<n>x)",
        )
        self.assert_problem(r"j-demo--effect-cache: .*inline flags, \(\?P<")
        self.edit("effects/j-demo--effect-cache/graders/fired.md", "(?P<n>x)", "")
        (self.repo / "evals/cases/effects/j-demo--effect-cache/graders/re.md").write_text(
            "---\ntype: regex\npattern: 'x'\nflags: q\ntarget: { source: file, path: design.md }\n---\n"
        )
        self.assert_problem(r"j-demo--effect-cache: .*not JS regex flags")

    def test_lint_escaped_backslash_a_is_literal(self):
        self.assertIsNone(kcases.JS_UNSUPPORTED.search("\\\\Aabc"))
        self.assertIsNotNone(kcases.JS_UNSUPPORTED.search("\\Aabc"))
        self.assertIsNone(kcases.JS_UNSUPPORTED.search("(?:a)(?=b)"))

    def test_lint_required_keys_and_match_all(self):
        (self.repo / "evals/cases/effects/j-demo--effect-cache/graders/re.md").write_text(
            "---\ntype: regex\npattern: '.*'\n---\n"
        )
        self.assert_problem(r"j-demo--effect-cache: .*regex needs `target`")
        self.assert_problem(r"j-demo--effect-cache: .*matches everything")

    def test_lint_llm_claim_count(self):
        claims = "".join(f"PASS if claim {i}.\n" for i in range(6))
        (self.repo / "evals/cases/effects/j-demo--effect-cache/graders/result.md").write_text(
            "---\ntype: llm\nfocus: { source: file, path: design.md }\n---\n\n"
            + claims
            + "FAIL if nothing.\n"
        )
        self.assert_problem(r"j-demo--effect-cache: .*7 claims; use 2-5")

    def test_lint_vacuous_result_grader(self):
        (self.repo / "evals/cases/effects/j-demo--effect-cache/graders/kept.md").write_text(
            "---\ntype: regex\npattern: 'def add'\ntarget: { source: file, path: calc/ops.py }\n---\n"
        )
        self.assert_problem(r"j-demo--effect-cache: .*already passes on the seeded calc/ops\.py")

    def test_lint_exists_false_is_vacuous(self):
        (self.repo / "evals/cases/effects/j-demo--effect-cache/graders/none.md").write_text(
            "---\ntype: file_exists\npath: design.md\nexists: false\n---\n"
        )
        self.assert_problem(r"j-demo--effect-cache: .*exists: false")

    def test_lint_graded_path_must_be_named_or_seeded(self):
        self.edit(
            "effects/j-demo--effect-cache/graders/result.md", "path: design.md", "path: notes.md"
        )
        self.assert_problem(r"j-demo--effect-cache: .*'notes\.md' is neither named")

    def test_lint_non_md_grader_file(self):
        (self.repo / "evals/cases/effects/j-demo--effect-cache/graders/extra.yml").write_text(
            "type: llm\n"
        )
        self.assert_problem(r"j-demo--effect-cache: .*extra\.yml is not a \.md grader")

    def test_lint_double_md_extension(self):
        graders = self.repo / "evals/cases/effects/j-demo--effect-cache/graders"
        (graders / "result.md").rename(graders / "result.md.md")
        self.assert_problem(r"j-demo--effect-cache: .*double \.md")

    DESCRIPTION = "description: "

    def set_description(self, rel: str, line: str | None) -> None:
        """Replace (or with None, drop) the description line of a case's prompt.md."""
        path = self.repo / "evals" / "cases" / rel / "prompt.md"
        lines = [
            ln
            for ln in path.read_text().splitlines(keepends=True)
            if not ln.startswith(self.DESCRIPTION)
        ]
        if line is not None:
            lines.insert(1, line + "\n")
        path.write_text("".join(lines))

    def test_case_without_description_is_a_problem(self):
        self.set_description("effects/j-demo--effect-cache", None)
        self.assert_problem(
            r"j-demo--effect-cache: description must be <text>; <asset path>; <ending>"
        )

    def test_empty_quoted_description_is_a_problem(self):
        self.set_description("effects/j-demo--effect-cache", 'description: ""')
        self.assert_problem(r"j-demo--effect-cache: description must be")

    BASE = "a rule; .claude/commands/j-demo.md; hard because it is easy to miss"

    def test_description_with_an_invalid_escape_is_malformed(self):
        text = self.BASE.replace("rule", "ru\\le")
        self.set_description("effects/j-demo--effect-cache", f'description: "{text}"')
        self.assert_problem(
            r"j-demo--effect-cache: `description` is a malformed quoted YAML scalar"
        )

    def test_description_with_an_escaped_quote_or_backslash_is_a_problem(self):
        for text in (self.BASE.replace("rule", 'ru\\"le'), self.BASE.replace("rule", "ru\\\\le")):
            with self.subTest(text=text):
                self.set_description("effects/j-demo--effect-cache", f'description: "{text}"')
                self.assert_problem(
                    r"j-demo--effect-cache: description must be <text>; <asset path>; <ending>"
                )

    def assert_description_problem(
        self, line: str, pattern: str, rel: str = "effects/j-demo--effect-cache"
    ) -> None:
        self.set_description(rel, line)
        problems = [p for p in self.problems() if p.startswith(rel.split("/")[1] + ":")]
        self.assertTrue(
            any(re.search(pattern, p) for p in problems), f"no {pattern!r} in {problems}"
        )

    def test_unquoted_description_is_a_problem(self):
        self.assert_description_problem(
            "description: x; .claude/agents/demo-agent.md; hard because y", r"description must be"
        )

    def test_single_quoted_description_is_a_problem(self):
        self.assert_description_problem(
            "description: 'x; .claude/agents/demo-agent.md; hard because y'", r"description must be"
        )

    def test_description_with_an_absolute_path_is_a_problem(self):
        self.assert_description_problem(
            'description: "x; /etc/hosts; hard because y"', r"description must be"
        )

    def test_description_with_a_parent_segment_is_a_problem(self):
        (self.tmp / "dotfiles").mkdir()
        (self.tmp / "dotfiles" / "README.md").write_text("# outside\n")
        self.assert_description_problem(
            'description: "x; ../dotfiles/README.md; hard because y"', r"description must be"
        )
        self.assert_description_problem(
            'description: "x; .claude/../.claude/agents/demo-agent.md; hard because y"',
            r"description must be",
        )

    def test_description_path_outside_claude_is_a_problem(self):
        (self.repo / "evals" / "README.md").write_text("# evals\n")
        self.assert_description_problem(
            'description: "x; evals/README.md; hard because y"', r"description must be"
        )

    def test_description_path_that_is_a_directory_is_a_problem(self):
        self.assert_description_problem(
            'description: "x; .claude/agents; hard because y"',
            r"description names a path that does not exist",
        )

    def test_description_path_with_a_semicolon_is_a_problem(self):
        (self.repo / ".claude/references/cat/a;b.md").write_text("# odd\n")
        self.assert_description_problem(
            'description: "x; .claude/references/cat/a;b.md; hard because y"',
            r"description must be",
        )

    def test_description_with_a_trailing_newline_is_a_problem(self):
        self.assert_description_problem(
            'description: "x; .claude/agents/demo-agent.md; hard because y\n  "',
            r"description must be",
        )
        self.assert_description_problem(
            'description: "x; .claude/skills/demo-skill/SKILL.md; should\n  "',
            r"description must be",
            "triggers/demo-skill--trigger-should-rate",
        )

    def test_description_problem_names_the_expected_ending_and_the_path(self):
        self.assert_description_problem(
            'description: "x; .claude/skills/demo-skill/SKILL.md; hard because y"',
            r"description must be <text>; <asset path>; <ending>.*should",
            "triggers/demo-skill--trigger-should-rate",
        )
        self.assert_description_problem(
            'description: "x; .claude/commands/missing.md; hard because y"',
            r"description names a path that does not exist.*\.claude/commands/missing\.md",
        )

    def test_trigger_near_description_needs_a_near_ending(self):
        rel = "triggers/demo-skill--trigger-near-typo"
        self.assert_description_problem(
            'description: "x; .claude/skills/demo-skill/SKILL.md; should"',
            r"description must be",
            rel,
        )
        self.assert_description_problem(
            'description: "x; .claude/skills/demo-skill/SKILL.md; near:"',
            r"description must be",
            rel,
        )
        self.set_description(
            rel, 'description: "x; .claude/skills/demo-skill/SKILL.md; near: j-demo"'
        )
        self.assertFalse(
            any(p.startswith("demo-skill--trigger-near-typo:") for p in self.problems())
        )

    def test_trigger_noauto_description_needs_noauto(self):
        rel = "triggers/j-quiet--trigger-noauto-ask"
        self.assert_description_problem(
            'description: "x; .claude/commands/j-quiet.md; should"', r"description must be", rel
        )
        self.set_description(rel, 'description: "x; .claude/commands/j-quiet.md; noauto"')
        self.assertFalse(any(p.startswith("j-quiet--trigger-noauto-ask:") for p in self.problems()))

    def test_workflow_description_needs_hard_because(self):
        rel = "workflows/wf--demo-chain"
        self.assert_description_problem(
            'description: "x; .claude/commands/j-demo.md; should"', r"description must be", rel
        )
        self.set_description(rel, 'description: "x; .claude/commands/j-demo.md; hard because y"')
        self.assertFalse(any(p.startswith("wf--demo-chain:") for p in self.problems()))

    def test_description_path_must_exist(self):
        self.set_description(
            "effects/j-demo--effect-cache",
            'description: "a rule; .claude/commands/missing.md; hard because it is easy to miss"',
        )
        self.assert_problem(r"j-demo--effect-cache: description names a path that does not exist")

    def test_trigger_description_needs_a_trigger_ending(self):
        name = "triggers/demo-skill--trigger-should-rate"
        self.set_description(
            name, 'description: "a rule; .claude/skills/demo-skill/SKILL.md; hard because x"'
        )
        self.assert_problem(r"demo-skill--trigger-should-rate: description must be")
        self.set_description(
            name, 'description: "a rule; .claude/skills/demo-skill/SKILL.md; near: none"'
        )
        self.assert_problem(r"demo-skill--trigger-should-rate: description must be")
        self.set_description(
            name, 'description: "a rule; .claude/skills/demo-skill/SKILL.md; should"'
        )
        self.assertFalse(any("demo-skill--trigger-should-rate" in p for p in self.problems()))

    def add_indexing_agent(self, categories: str = "cat") -> None:
        agent = self.repo / ".claude/agents/indexer.md"
        agent.write_text(
            "---\nname: indexer\ndescription: Use when indexing.\nmodel: sonnet\ntools: Read\n---\n\n"
            + "".join(f"Read references/{c}/ for guidance.\n" for c in categories.split(","))
        )

    def add_reading_case(self, input_match: str) -> None:
        write_case(
            self.repo,
            "effects",
            "indexer--effect-reads",
            EFFECT_FM,
            "Use the kb:indexer agent to review the cache design and save it to ./design.md.",
            {
                "result": LLM_RESULT,
                "dispatched": agent_dispatched("indexer"),
                "read": f"---\ntype: tool_used\ntool: Read\ninput_match: '{input_match}'\n---\n",
            },
        )

    def test_agent_with_a_reference_index_needs_a_reading_case(self):
        self.add_indexing_agent()
        self.assert_problem(r"^indexer: no effect case reads references/cat/")
        self.add_reading_case(r"\.kb-references/cat/")
        self.assertFalse(any(p.startswith("indexer:") for p in self.problems()), self.problems())

    def test_agent_without_a_reference_index_is_exempt(self):
        self.assertFalse(any(p.startswith("demo-agent:") for p in self.problems()))

    def test_alternation_input_match_counts_for_each_category_it_matches(self):
        self.add_indexing_agent("data")
        self.add_reading_case("(?:architecture|data)/")
        self.assertFalse(any(p.startswith("indexer:") for p in self.problems()), self.problems())

    def add_agent_case(self, suite: str, name: str, grader: str, prompt: str | None = None) -> None:
        frontmatter = {
            "triggers": TRIGGER_FM.format(kind="trigger-should"),
            "effects": EFFECT_FM,
        }.get(suite, WORKFLOW_FM)
        body = (
            prompt
            or "Use the kb:indexer agent to review the cache design and save it to ./design.md."
        )
        graders = (
            {"read": grader} if suite == "triggers" else {"result": LLM_RESULT, "read": grader}
        )
        write_case(self.repo, suite, name, frontmatter, body, graders)

    def indexer_problems(self) -> list:
        return [p for p in self.problems() if p.startswith("indexer:")]

    def assert_not_a_reading_case(self, suite: str, name: str, grader: str) -> None:
        self.add_indexing_agent()
        self.add_agent_case(suite, name, grader)
        self.assertEqual(len(self.indexer_problems()), 1, self.problems())

    READ = "---\ntype: tool_used\ntool: Read\n{extra}---\n"

    def test_read_grader_without_input_match_is_not_a_reading_case(self):
        self.assert_not_a_reading_case(
            "effects", "indexer--effect-reads", self.READ.format(extra="")
        )

    def test_read_grader_with_an_empty_input_match_is_not_a_reading_case(self):
        self.assert_not_a_reading_case(
            "effects", "indexer--effect-reads", self.READ.format(extra="input_match: ''\n")
        )

    def test_read_grader_with_max_0_is_not_a_reading_case(self):
        extra = "input_match: '\\.kb-references/cat/'\nmin: 0\nmax: 0\n"
        self.assert_not_a_reading_case(
            "effects", "indexer--effect-reads", self.READ.format(extra=extra)
        )

    def test_read_grader_on_another_category_is_not_a_reading_case(self):
        extra = "input_match: '\\.kb-references/other/'\n"
        self.assert_not_a_reading_case(
            "effects", "indexer--effect-reads", self.READ.format(extra=extra)
        )

    def test_non_read_grader_is_not_a_reading_case(self):
        grep = "---\ntype: tool_used\ntool: Grep\ninput_match: '\\.kb-references/cat/'\n---\n"
        self.assert_not_a_reading_case("effects", "indexer--effect-greps", grep)

    def test_regex_grader_is_not_a_reading_case(self):
        regex = "---\ntype: regex\ntarget: { source: file, path: design.md }\npattern: '\\.kb-references/cat/'\n---\n"
        self.assert_not_a_reading_case("effects", "indexer--effect-regex", regex)

    def test_trigger_case_of_the_agent_is_not_a_reading_case(self):
        extra = "input_match: '\\.kb-references/cat/'\n"
        self.assert_not_a_reading_case(
            "triggers", "indexer--trigger-should-reads", self.READ.format(extra=extra)
        )

    def test_workflow_case_is_not_a_reading_case(self):
        extra = "input_match: '\\.kb-references/cat/'\n"
        self.assert_not_a_reading_case(
            "workflows", "wf--indexer-reads", self.READ.format(extra=extra)
        )

    def test_another_agents_effect_case_is_not_a_reading_case(self):
        extra = "input_match: '\\.kb-references/cat/'\n"
        self.add_indexing_agent()
        self.add_agent_case(
            "effects",
            "demo-agent--effect-reads",
            self.READ.format(extra=extra),
            "Use the kb:demo-agent agent to review the cache design and save it to ./design.md.",
        )
        self.assertEqual(len(self.indexer_problems()), 1, self.problems())

    def test_input_match_that_python_cannot_compile_is_reported_when_nothing_reads(self):
        self.add_indexing_agent()
        self.add_reading_case(r"\.kb-references/(?<c>cat)/")
        self.assert_problem(
            r"^indexer: no effect case reads references/cat/; skipped indexer--effect-reads grader read: "
            r"input_match does not compile in Python"
        )

    def test_missing_reference_problem_names_every_category(self):
        self.add_indexing_agent("cat,data")
        self.assert_problem(r"^indexer: no effect case reads references/cat/ or references/data/$")

    def test_one_category_is_enough_for_an_agent_that_names_several(self):
        self.add_indexing_agent("cat,data")
        self.add_reading_case(r"\.kb-references/data/")
        self.assertEqual(self.indexer_problems(), [])

    def test_missing_agents_directory_passes_the_reference_check(self):
        shutil.rmtree(self.repo / ".claude" / "agents")
        found = kcases.discover(self.repo / "evals" / "cases")
        self.assertEqual(kcases._lint_agent_references(found, self.repo), [])

    def test_lint_tags_name_suite_and_kind(self):
        self.edit(
            "effects/j-demo--effect-cache/prompt.md", "tags: [effects, effect]", "tags: [effect]"
        )
        self.assert_problem(r"j-demo--effect-cache: .*tags")

    def test_lint_no_web_tools(self):
        self.edit("effects/j-demo--effect-cache/prompt.md", "TodoWrite]", "TodoWrite, WebFetch]")
        self.assert_problem(r"j-demo--effect-cache: .*WebFetch")


class CoverageTests(TempRepoCase):
    def gaps(self) -> list:
        return kcases.coverage(kcases.discover(self.repo / "evals" / "cases"), self.repo)

    def test_coverage_rules(self):
        gaps = self.gaps()
        # demo-skill: has should, near, one effect -> needs a second effect.
        self.assertTrue(any(g.startswith("demo-skill:") and "effect" in g for g in gaps), gaps)
        # j-demo is model-invocable: no trigger-should, no trigger-near, one effect.
        j_demo = [g for g in gaps if g.startswith("j-demo:")]
        self.assertTrue(any("trigger-should" in g for g in j_demo), j_demo)
        self.assertTrue(any("trigger-near" in g for g in j_demo), j_demo)
        # j-quiet is user-only: has trigger-noauto, needs effects, never needs trigger-should.
        j_quiet = [g for g in gaps if g.startswith("j-quiet:")]
        self.assertTrue(any("effect" in g for g in j_quiet), j_quiet)
        self.assertFalse(any("trigger-should" in g for g in j_quiet), j_quiet)
        # demo-agent needs two effects and no triggers.
        agent = [g for g in gaps if g.startswith("demo-agent:")]
        self.assertTrue(any("effect" in g for g in agent), agent)
        self.assertFalse(any("trigger" in g for g in agent), agent)


class RealTreeTests(unittest.TestCase):
    def test_real_cases_pass_lint(self):
        problems = kcases.lint(kcases.discover(REPO / "evals" / "cases"), REPO)
        self.assertEqual(problems, [], "\n".join(problems))

    def test_real_assets_meet_coverage(self):
        found = kcases.discover(REPO / "evals" / "cases")
        gaps = kcases.coverage(found, REPO)
        self.assertEqual(gaps, [], "\n".join(gaps))

    @staticmethod
    def real_graders(case_prefix: str, name: str) -> list:
        found = kcases.discover(REPO / "evals" / "cases")
        cases = [c for c in found if c.name.startswith(case_prefix)]
        return [g for c in cases for g in c.graders if g.name == name]

    def assert_grades(self, graders: list, failing: list, passing: list) -> None:
        """Grade in Python `re`, as lint's grounding check does: pass = found != not_contains.

        The harness grades in JavaScript; these patterns use only syntax both dialects share.
        """
        self.assertTrue(graders)
        for g in graders:
            flags = kcases._python_flags(kcases.unquote(g.fields.get("flags", "")))
            pattern = re.compile(kcases.unquote(g.fields["pattern"]), flags)
            negate = kcases.unquote(g.fields.get("match", "")) == "not_contains"
            for text in failing:
                found = pattern.search(text) is not None
                self.assertFalse(found != negate, f"{g.name} should fail: {text}")
            for text in passing:
                found = pattern.search(text) is not None
                self.assertTrue(found != negate, f"{g.name} should pass: {text}")

    def test_j_explain_self_contained_rejects_external_resources(self):
        graders = self.real_graders("j-explain--effect-", "self-contained")
        self.assertEqual(len(graders), 2)
        self.assert_grades(
            graders,
            failing=[
                '<link rel="stylesheet" href="https://cdn.x/a.css">',
                '<script src="https://x/y.js"></script>',
                '<img src="//cdn.x/i.png">',
                "<style>@import url(https://x/f.css);</style>",
                '<div style="background:url(https://x/b.png)">',
                '<img alt="x" srcset="https://x/i.png 1x, https://x/i2.png 2x">',
                '<object data="https://x/d.svg" type="image/svg+xml"></object>',
                '<svg><image href="https://x/i.png" width="9" height="9"/></svg>',
                '<svg><image xlink:href="//x/i.png"/></svg>',
                "<script>fetch('https://x/api').then(r => r.json());</script>",
                "<script>const r = new XMLHttpRequest(); r.open('GET', '/x');</script>",
                '<meta http-equiv="refresh" content="0; url=https://x/">',
                "<script type=\"module\">import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';</script>",
                '<script type="module">import x from "https://x/m.js";</script>',
                "<script>import('https://x/y.js')</script>",
                '<script type="module">import("https://x/m.js");</script>',
                '<video controls poster="https://x/p.png"></video>',
                '<input type="image" src="https://x/b.png">',
                '<body background="https://x/bg.png">',
                '<script>new Image().src = "https://x/t.gif";</script>',
                '<script>new Worker("https://x/w.js")</script>',
                '<style>b{background:image-set("https://x/a.png" 1x)}</style>',
                '<style>@import"x.css";</style>',
                '<style>@import "x.css";</style>',
                "<script>navigator.sendBeacon('/log', 'x');</script>",
                "<script>const s = new WebSocket('wss://x/s');</script>",
                "<script>const e = new EventSource('/events');</script>",
                '<video><track src="//x/a.vtt"></video>',
                '<svg><use href="https://x/s.svg#a"/></svg>',
                '<svg><filter><feImage href="https://x/a.png"/></filter></svg>',
                '<link rel="stylesheet"\n      href="https://x/a.css">',
                '<img src="img/a.png">',
                '<img src="/a.png">',
                '<audio src="a.mp3"></audio>',
                '<iframe src="page.html"></iframe>',
                "<style>b{background:url(img/a.png)}</style>",
                "<style>@font-face{font-family:f;src:url(fonts/a.woff2)}</style>",
            ],
            passing=[
                '<style>b{color:red}</style><svg></svg><a href="#s1">x</a>',
                "<script>document.title = 'x';</script>",
                '<div style="background:url(data:image/png;base64,AAAA)">',
                '<link rel="icon" href="data:image/svg+xml,%3Csvg%3E%3C/svg%3E">',
                "<style>/* no @import here: every rule is inline */ b{color:red}</style>",
                "<!-- inline only, no @import or CDN -->",
                '<img srcset="data:image/png;base64,AAAA 1x">',
                '<object data="data:text/html,hi"></object>',
                "<style>@import url(data:text/css,b{});</style>",
                '<svg><use href="#id"/></svg>',
                '<img src="data:image/png;base64,AAAA">',
                '<rect fill="url(#g)"/>',
                '<a href="https://example.com/docs">docs</a>',
                '<svg xmlns="http://www.w3.org/2000/svg"></svg>',
                '<script>const u = URL.createObjectURL(new Blob(["x"]));</script>',
            ],
        )

    def test_j_data_no_serial_ignores_sql_comments(self):
        graders = self.real_graders("j-data--effect-orders-schema", "no-serial")
        self.assertEqual(len(graders), 1)
        self.assert_grades(
            graders,
            failing=[
                "id serial PRIMARY KEY,",
                "  price money,",
                "name varchar(50),",
                "created timestamp NOT NULL,",
                "CREATE TABLE t (id bigserial, x int);",
                "id serial PRIMARY KEY, -- pk",
            ],
            passing=[
                "-- id (no serial)",
                "-- ids are uuid, not serial",
                "total numeric(12,2) NOT NULL, -- cents, not money",
                "  id uuid, -- not serial",
                "created_at timestamptz NOT NULL,",
            ],
        )


# --- runner and ledger -------------------------------------------------------------------------

import contextlib  # noqa: E402
import io  # noqa: E402
import sys  # noqa: E402
from unittest import mock  # noqa: E402

from evals.kbeval import ledger as kledger  # noqa: E402
from evals.kbeval import runner as krunner  # noqa: E402

JUDGE = "claude-fable-5-2"
MODEL = "claude-sonnet-5-5"


def result_doc(cases: dict, cost=0.5, partial=False, reason=None) -> dict:
    """A result document in the captured CLI shape: {case name: [run error or None, ...]}."""
    captured = json.loads(
        (REPO / "tests" / "fixtures" / "kbeval" / "aggregate-result.json").read_text()
    )
    template_case = captured["cases"][0]
    template_run = template_case["arms"]["with"][0]
    doc = {**captured, "partial": partial, "costUsd": cost, "cases": []}
    if reason is not None:
        doc["partialReason"] = reason
    for name, errors in cases.items():
        runs = []
        for err in errors:
            run = {**template_run, "error": err, "tracePath": None}
            if err and API_ERROR_TEXT.search(err):
                run["graders"] = []
            runs.append(run)
        doc["cases"].append({**template_case, "name": name, "arms": {"with": runs}})
    return doc


API_ERROR_TEXT = re.compile(r"(?i)api error|rate.?limit")


class FakeCli:
    """Records argv; `responder(args)` returns (doc or None, returncode, stderr)."""

    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def version(self) -> str:
        return "2.1.325 (Claude Code)"

    def run(self, args):
        self.calls.append(list(args))
        doc, code, stderr = self.responder(args)
        if doc is not None:
            path = Path(args[args.index("--json") + 1])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(doc))
        return subprocess.CompletedProcess(args, code, "", stderr)


def case_arg(args) -> str:
    return args[args.index("--case") + 1]


class RunnerCase(TempRepoCase):
    def setUp(self):
        super().setUp()
        self.plugin = self.stage()
        self.state = self.tmp / "state"
        self.ledger_path = self.tmp / "ledger.jsonl"
        self.budget: dict = {"period": "p", "ceilingUsd": 100, "smokeCapUsd": 40}
        env = mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-placeholder"})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("KBEVAL_ALLOW_PAID", None)
        if getattr(self, "real_cli", False):
            return
        refuse = mock.patch.object(
            krunner.ClaudeCli, "run", side_effect=AssertionError("real claude called")
        )
        refuse.start()
        self.addCleanup(refuse.stop)

    def run_suite(self, cli, suite="effects", assets=("j-demo",), staging="kb", **extra):
        err = io.StringIO()
        options = {
            "plugin": self.plugin,
            "state": self.state,
            "ledger_path": self.ledger_path,
            "budget": self.budget,
            "p95": 5.0,
            **extra,
        }
        with contextlib.redirect_stderr(err):
            code = krunner.run(suite, staging, MODEL, 3, "test", list(assets), 8, cli, **options)
        return code, err.getvalue()

    def entries(self) -> list:
        if not self.ledger_path.exists():
            return []
        return [json.loads(line) for line in self.ledger_path.read_text().splitlines()]


class RunnerGuardTests(RunnerCase):
    real_cli = True  # these tests exercise the real ClaudeCli guard, with claude off PATH

    def test_cli_refuses_unless_allow_paid_is_exactly_one(self):
        for value in ("0", "true", "yes", " 1", ""):
            env = {**os.environ, "KBEVAL_ALLOW_PAID": value, "PATH": str(self.tmp / "empty-bin")}
            with self.subTest(value=value), mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaises(krunner.PaidCallRefusedError):
                    krunner.ClaudeCli().run(["plugin", "eval", "."])

    def test_cli_refuses_without_allow_paid(self):
        env = {k: v for k, v in os.environ.items() if k != "KBEVAL_ALLOW_PAID"}
        env["PATH"] = str(self.tmp / "empty-bin")
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(krunner.PaidCallRefusedError):
                krunner.ClaudeCli().run(["plugin", "eval", "."])

    def test_runner_refuses_without_api_key(self):
        cli = FakeCli(lambda args: (result_doc({}), 0, ""))
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            code, err = self.run_suite(cli)
        self.assertNotEqual(code, 0)
        self.assertIn("evals bill an API key; export ANTHROPIC_API_KEY", err)
        self.assertEqual(cli.calls, [])

    def test_runner_refuses_a_model_in_the_judge_family(self):
        cli = FakeCli(lambda args: (result_doc({}), 0, ""))
        err = io.StringIO()
        options = {
            "plugin": self.plugin,
            "state": self.state,
            "ledger_path": self.ledger_path,
            "budget": self.budget,
            "p95": 5.0,
        }
        with contextlib.redirect_stderr(err):
            code = krunner.run(
                "effects", "kb", "claude-fable-5-3", 3, "test", ["j-demo"], 8, cli, **options
            )
        self.assertNotEqual(code, 0)
        self.assertIn("claude-fable-5-3", err.getvalue())
        self.assertIn(krunner.JUDGE_MODEL, err.getvalue())
        self.assertEqual(cli.calls, [])
        self.assertEqual(self.entries(), [])

    def test_runner_allows_other_families(self):
        cli = FakeCli(lambda args: (result_doc({case_arg(args).rstrip("*-"): [None]}), 0, ""))
        code, err = self.run_suite(cli)
        self.assertEqual(code, 0, err)
        self.assertTrue(cli.calls)

    def test_missing_api_key_message_wins_over_the_judge_guard(self):
        cli = FakeCli(lambda args: (result_doc({}), 0, ""))
        err = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}),
            contextlib.redirect_stderr(err),
        ):
            code = krunner.run(
                "effects",
                "kb",
                "claude-fable-5-3",
                3,
                "test",
                ["j-demo"],
                8,
                cli,
                plugin=self.plugin,
            )
        self.assertNotEqual(code, 0)
        self.assertIn("export ANTHROPIC_API_KEY", err.getvalue())
        self.assertNotIn("judge's family", err.getvalue())

    def test_runner_refuses_a_model_id_the_guard_cannot_parse(self):
        options = {
            "plugin": self.plugin,
            "state": self.state,
            "ledger_path": self.ledger_path,
            "budget": self.budget,
            "p95": 5.0,
        }
        for model in ("fable", "us.anthropic.claude-fable-5-2"):
            with self.subTest(model=model):
                cli = FakeCli(lambda args: (result_doc({}), 0, ""))
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    code = krunner.run(
                        "effects", "kb", model, 3, "test", ["j-demo"], 8, cli, **options
                    )
                self.assertNotEqual(code, 0)
                self.assertEqual(cli.calls, [])
                self.assertIn(f"model {model} is not a full model id", err.getvalue())

    def test_missing_api_key_message_wins_over_the_model_id_guard(self):
        cli = FakeCli(lambda args: (result_doc({}), 0, ""))
        err = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}),
            contextlib.redirect_stderr(err),
        ):
            code = krunner.run(
                "effects", "kb", "fable", 3, "test", ["j-demo"], 8, cli, plugin=self.plugin
            )
        self.assertNotEqual(code, 0)
        self.assertIn("export ANTHROPIC_API_KEY", err.getvalue())
        self.assertNotIn("not a full model id", err.getvalue())

    def test_judge_family_of_an_unprefixed_id_is_the_id(self):
        self.assertEqual(krunner.judge_family("claude-fable-5-3"), "fable")
        self.assertEqual(krunner.judge_family("gpt-5"), "gpt-5")


class RunnerBatchTests(RunnerCase):
    def test_batch_args(self):
        cli = FakeCli(lambda args: (result_doc({case_arg(args).rstrip("*-"): [None]}), 0, ""))
        code, _ = self.run_suite(cli)
        self.assertEqual(code, 0)
        out = self.state / "test" / "kb" / MODEL / "effects" / "j-demo"
        self.assertEqual(
            cli.calls[0],
            [
                "plugin",
                "eval",
                str(self.plugin),
                "--eval-dir",
                "evals-effects",
                "--case",
                "j-demo--*",
                "--runs",
                "3",
                "--model",
                MODEL,
                "--judge-model",
                JUDGE,
                "--ablation",
                "none",
                "--threshold",
                "0",
                "--trust-plugin",
                "--scaffold",
                "--no-publish",
                "-j",
                "8",
                "--json",
                f"{out}.json",
                "--output-dir",
                str(out),
                "--max-cost-usd",
                "60.00",
                "--allow-tools",
                "Bash",
                "Write",
                "Edit",
            ],
        )

    def test_kb_digest_changes_when_claude_md_changes(self):
        cli = FakeCli(lambda args: (result_doc({}), 0, ""))
        options = {
            "plugin": self.plugin,
            "state": self.state,
            "ledger_path": self.ledger_path,
            "budget": self.budget,
            "p95": 5.0,
        }
        before = krunner._Batches("effects", "kb", MODEL, 3, "test", 8, cli, options).kb_digest
        claude_md = self.repo / ".claude" / "CLAUDE.md"
        claude_md.write_text(claude_md.read_text() + "\nNew rule.\n")
        options["plugin"] = self.stage()
        after = krunner._Batches("effects", "kb", MODEL, 3, "test", 8, cli, options).kb_digest
        self.assertNotEqual(before, after)

    def test_runner_writes_the_kb_digest_sidecar(self):
        cli = FakeCli(lambda args: (result_doc({"j-demo--effect-cache": [None]}), 0, ""))
        self.run_suite(cli)
        out = self.state / "test" / "kb" / MODEL / "effects" / "j-demo"
        digest = krunner._tree_digest(self.plugin, skip_prefix="evals-")
        self.assertEqual(
            json.loads(out.with_suffix(".meta.json").read_text()), {"kbDigest": digest}
        )

    def test_keep_temp_traces_are_scanned_for_key(self):
        trace = self.tmp / "kept" / "out" / "trace.jsonl"

        def responder(args):
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text('{"text": "sk-ant-leaked"}\n')
            doc = result_doc({"j-demo--effect-cache": [None]})
            doc["cases"][0]["arms"]["with"][0]["tracePath"] = str(trace)
            return doc, 0, ""

        cli = FakeCli(responder)
        code, err = self.run_suite(cli, keep_temp=True)
        self.assertIn("--keep-temp", cli.calls[0])
        self.assertNotEqual(code, 0)
        self.assertIn("trace.jsonl", err)

    def test_triggers_get_no_write_tools(self):
        cli = FakeCli(lambda args: (result_doc({"x": [None]}), 0, ""))
        self.run_suite(cli, suite="triggers", assets=("demo-skill",))
        self.assertNotIn("--allow-tools", cli.calls[0])

    def test_workflows_batch_one_case_at_a_time(self):
        cli = FakeCli(lambda args: (result_doc({case_arg(args): [None]}), 0, ""))
        self.run_suite(cli, suite="workflows", assets=())
        self.assertEqual([case_arg(c) for c in cli.calls], ["wf--demo-chain"])

    def test_cap_is_remaining_minus_reserve(self):
        kledger.append({"period": "p", "costUsd": 40.0}, self.ledger_path)
        cli = FakeCli(lambda args: (result_doc({"x": [None]}), 0, ""))
        self.run_suite(cli)
        cap = cli.calls[0][cli.calls[0].index("--max-cost-usd") + 1]
        self.assertEqual(cap, "20.00")
        kledger.append({"period": "p", "costUsd": 30.0}, self.ledger_path)
        cli = FakeCli(lambda args: (result_doc({"x": [None]}), 0, ""))
        code, err = self.run_suite(cli, assets=("demo-skill",))
        self.assertEqual(code, 2)
        self.assertEqual(cli.calls, [])
        self.assertIn("budget", err)

    def test_cap_limit_bounds_the_invocation(self):
        cli = FakeCli(lambda args: (result_doc({"x": [None]}, cost=4.0), 0, ""))
        self.run_suite(cli, assets=("j-demo", "demo-skill"), cap_limit=10.0)
        caps = [c[c.index("--max-cost-usd") + 1] for c in cli.calls]
        self.assertEqual(caps, ["10.00", "6.00"])

    def test_ledger_reserves_then_settles_to_actual_cost(self):
        cli = FakeCli(lambda args: (result_doc({"x": [None]}, cost=1.25), 0, ""))
        self.run_suite(cli)
        self.assertEqual([e["charged"] for e in self.entries()], ["reserved", "settle"])
        self.assertAlmostEqual(sum(e["costUsd"] for e in self.entries()), 1.25)
        self.assertEqual(self.entries()[1]["actualUsd"], 1.25)

    def test_missing_json_charges_cap(self):
        cli = FakeCli(lambda args: (None, 143, "terminated"))
        code, _ = self.run_suite(cli)
        self.assertNotEqual(code, 0)
        self.assertEqual(
            [(e["costUsd"], e["charged"]) for e in self.entries()], [(60.0, "reserved")]
        )

    def test_null_cost_with_runs_aborts(self):
        cli = FakeCli(lambda args: (result_doc({"x": [None]}, cost=None), 0, ""))
        code, err = self.run_suite(cli, assets=("j-demo", "demo-skill"))
        self.assertNotEqual(code, 0)
        self.assertIn("cost", err)
        self.assertEqual(len(cli.calls), 1)


class LedgerTests(unittest.TestCase):
    def test_ledger_append_is_locked(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "ledger.jsonl"
        script = (
            "import sys; from pathlib import Path; from evals.kbeval import ledger\n"
            "for i in range(200): ledger.append({'period': 'p', 'costUsd': 0.01, 'i': i}, Path(sys.argv[1]))\n"
        )
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", script, str(path)], cwd=REPO, stdin=subprocess.DEVNULL
            )
            for _ in range(2)
        ]
        for proc in procs:
            self.assertEqual(proc.wait(timeout=60), 0)
        lines = path.read_text().splitlines()
        self.assertEqual(len(lines), 400)
        for line in lines:
            json.loads(line)
        self.assertAlmostEqual(kledger.spent("p", path), 4.0)

    def test_snapshot_writes_period_totals(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "ledger.jsonl"
        kledger.append({"period": "p", "purpose": "pilot", "model": "m", "costUsd": 1.5}, path)
        kledger.append({"period": "q", "purpose": "pilot", "model": "m", "costUsd": 9.0}, path)
        out = Path(tmp.name) / "spend.json"
        kledger.snapshot(out, "p", path)
        doc = json.loads(out.read_text())
        self.assertEqual(doc["period"], "p")
        self.assertEqual(doc["totals"]["totalUsd"], 1.5)
        self.assertEqual(doc["totals"]["byPurpose"], {"pilot": 1.5})


class RunnerResumeTests(RunnerCase):
    def test_runner_reruns_only_api_errors(self):
        main = {
            "j-demo--effect-cache": [
                "API error: 429 rate_limit_error",
                None,
                "timed out after 900s",
            ]
        }

        def responder(args):
            if case_arg(args) == "j-demo--*":
                return result_doc(main), 1, "ended on an API error"
            return result_doc({case_arg(args): [None, None, None]}), 0, ""

        cli = FakeCli(responder)
        code, _ = self.run_suite(cli)
        self.assertEqual(code, 0)
        self.assertEqual([case_arg(c) for c in cli.calls], ["j-demo--*", "j-demo--effect-cache"])
        again = FakeCli(responder)
        self.run_suite(again)
        self.assertEqual(again.calls, [])

    def test_skipped_complete_batch_gets_a_sidecar(self):
        cli = FakeCli(lambda args: (result_doc({"j-demo--effect-cache": [None]}), 0, ""))
        self.run_suite(cli)
        sidecar = (self.state / "test" / "kb" / MODEL / "effects" / "j-demo").with_suffix(
            ".meta.json"
        )
        expected = sidecar.read_text()
        sidecar.unlink()
        again = FakeCli(lambda args: (result_doc({"j-demo--effect-cache": [None]}), 0, ""))
        self.run_suite(again)
        self.assertEqual(again.calls, [])
        self.assertEqual(sidecar.read_text(), expected)

    def test_rerun_discards_the_stale_sidecar_before_the_cli_runs(self):
        cli = FakeCli(lambda args: (result_doc({"j-demo--effect-cache": [None]}), 0, ""))
        self.run_suite(cli)
        sidecar = (self.state / "test" / "kb" / MODEL / "effects" / "j-demo").with_suffix(
            ".meta.json"
        )
        self.assertTrue(sidecar.exists())
        prompt = self.plugin / "evals-effects" / "j-demo--effect-cache" / "prompt.md"
        prompt.write_text(prompt.read_text() + "\nMore.\n")
        seen = []

        def rerun(args):
            seen.append(sidecar.exists())
            return result_doc({"j-demo--effect-cache": [None]}), 0, ""

        self.run_suite(FakeCli(rerun))
        self.assertEqual(seen, [False])

    def test_changed_case_reruns_the_batch(self):
        cli = FakeCli(lambda args: (result_doc({"j-demo--effect-cache": [None]}), 0, ""))
        self.run_suite(cli)
        prompt = self.plugin / "evals-effects" / "j-demo--effect-cache" / "prompt.md"
        prompt.write_text(prompt.read_text() + "\nMore.\n")
        again = FakeCli(lambda args: (result_doc({"j-demo--effect-cache": [None]}), 0, ""))
        self.run_suite(again)
        self.assertEqual(len(again.calls), 1)

    def test_timeouts_and_turn_caps_are_results(self):
        doc = result_doc({"j-demo--effect-cache": ["timed out after 900s", "hit max_turns (40)"]})
        cli = FakeCli(lambda args: (doc, 0, ""))
        self.run_suite(cli)
        self.assertEqual(len(cli.calls), 1)

    def test_runner_stops_on_load_or_auth_failure(self):
        cli = FakeCli(lambda args: (None, 1, "No eval cases found"))
        code, err = self.run_suite(cli, assets=("j-demo", "demo-skill"))
        self.assertEqual(code, 1)
        self.assertIn("No eval cases found", err)
        self.assertEqual(len(cli.calls), 1)
        self.ledger_path.unlink()  # the missing JSON above charged its full cap
        cli = FakeCli(lambda args: (result_doc({}, partial=True, reason="auth_failed"), 2, ""))
        code, err = self.run_suite(cli, assets=("j-demo", "demo-skill"))
        self.assertEqual(code, 2)
        self.assertIn("auth_failed", err)
        self.assertEqual(len(cli.calls), 1)

    def test_results_scanned_for_key(self):
        def responder(args):
            out = Path(args[args.index("--output-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "transcript.jsonl").write_text('{"text": "sk-ant-leaked"}\n')
            return result_doc({"x": [None]}), 0, ""

        code, err = self.run_suite(FakeCli(responder))
        self.assertNotEqual(code, 0)
        self.assertIn("transcript.jsonl", err)
        self.assertIn("rotate the key", err)


from evals.kbeval import orchestrate as korch  # noqa: E402


class OrchestrateTests(RunnerCase):
    def setUp(self):
        super().setUp()
        (self.repo / "evals" / "budget.json").write_text(json.dumps(self.budget))
        (self.repo / "evals" / "README.md").write_text("# evals\n")
        self.budget["p95RunUsd"] = 5.0
        (self.repo / "evals" / "budget.json").write_text(json.dumps(self.budget))
        for patch in (
            mock.patch.object(korch, "REPO", self.repo),
            mock.patch.object(krunner, "REPO", self.repo),
            mock.patch.dict(os.environ, {"XDG_STATE_HOME": str(self.tmp / "xdg")}),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def calls_for(self, cli, staging: str) -> list:
        root = str(korch.state_root() / "stage" / staging)
        return [c for c in cli.calls if c[2].startswith(root + "/")]

    def test_smoke_samples_the_first_vanilla_effect_cases(self):
        for slug in ("b", "c", "d", "e"):
            write_case(
                self.repo,
                "effects",
                f"demo-skill--effect-{slug}",
                EFFECT_FM,
                "Design a rate limiter and save the design to ./design.md.",
                {"result": LLM_RESULT, "fired": skill_fired("demo-skill")},
            )
        cli = FakeCli(lambda args: (result_doc({case_arg(args): [None]}, cost=0.1), 0, ""))
        status = korch.smoke(["demo-agent", "demo-skill", "j-demo"], cli=cli)
        self.assertEqual(status, 0)
        vanilla = [case_arg(c) for c in self.calls_for(cli, "vanilla")]
        self.assertEqual(
            vanilla,
            [
                "demo-agent--effect-review",
                "demo-skill--effect-b",
                "demo-skill--effect-c",
                "demo-skill--effect-d",
                "demo-skill--effect-e",
            ],
        )
        kb_runs = {c[c.index("--runs") + 1] for c in self.calls_for(cli, "kb")}
        self.assertEqual(kb_runs, {"1"})
        self.assertTrue(all("--keep-temp" in c for c in cli.calls))

    def test_baseline_runs_vanilla_only_for_listed_models(self):
        cli = FakeCli(lambda args: (result_doc({case_arg(args): [None]}, cost=0.1), 0, ""))
        korch.baseline("claude-sonnet-5-5", vanilla_models=["claude-opus-5-5"], cli=cli)
        self.assertEqual(self.calls_for(cli, "vanilla"), [])
        self.assertEqual({c[c.index("--runs") + 1] for c in self.calls_for(cli, "kb")}, {"3"})
        cli = FakeCli(lambda args: (result_doc({case_arg(args): [None]}, cost=0.1), 0, ""))
        korch.baseline(
            "claude-opus-5-5", vanilla_runs=1, vanilla_models=["claude-opus-5-5"], cli=cli
        )
        vanilla = self.calls_for(cli, "vanilla")
        self.assertTrue(vanilla)
        self.assertEqual({c[c.index("--runs") + 1] for c in vanilla}, {"1"})

    def test_baseline_cap_is_the_purpose_remainder_at_each_step(self):
        def spend(*args, **kwargs):
            kledger.append({"period": self.budget["period"], "purpose": "baseline", "costUsd": 1.0})
            return 0

        kledger.append({"period": self.budget["period"], "purpose": "other", "costUsd": 7.0})
        with mock.patch.object(korch.runner, "run", side_effect=spend) as run:
            self.assertEqual(
                korch.baseline("claude-sonnet-5-5", vanilla_models=[], cap_usd=50.0), 0
            )
        self.assertEqual([c.kwargs["cap_limit"] for c in run.call_args_list], [50.0, 49.0, 48.0])

    def test_baseline_stops_with_two_when_the_cap_is_spent(self):
        kledger.append({"period": self.budget["period"], "purpose": "baseline", "costUsd": 5.0})
        cli = FakeCli(lambda args: (result_doc({case_arg(args): [None]}, cost=0.1), 0, ""))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            status = korch.baseline("claude-sonnet-5-5", vanilla_models=[], cap_usd=5.0, cli=cli)
        self.assertEqual(status, 2)
        self.assertEqual(cli.calls, [])
        self.assertIn(
            "baseline: purpose cap reached (cap $5.00, spent $5.00); no step started. Raise the cap to continue.",
            err.getvalue(),
        )

    def test_remainder_below_the_runner_minimum_counts_as_spent(self):
        kledger.append({"period": self.budget["period"], "purpose": "baseline", "costUsd": 4.999})
        err = io.StringIO()
        with (
            mock.patch.object(korch.runner, "run", return_value=0) as run,
            contextlib.redirect_stderr(err),
        ):
            status = korch.baseline("claude-sonnet-5-5", vanilla_models=[], cap_usd=5.0)
        self.assertEqual(status, 2)
        run.assert_not_called()
        self.assertIn("purpose cap reached (cap $5.00, spent $5.00)", err.getvalue())

    def test_cap_preamble_precedes_the_first_step(self):
        kledger.append({"period": self.budget["period"], "purpose": "baseline", "costUsd": 1.0})
        err = io.StringIO()

        def step(*args, **kwargs):
            self.assertIn("baseline: purpose cap $5.00, spent $1.00, left $4.00\n", err.getvalue())
            return 0

        with (
            mock.patch.object(korch.runner, "run", side_effect=step),
            contextlib.redirect_stderr(err),
        ):
            self.assertEqual(korch.baseline("claude-sonnet-5-5", vanilla_models=[], cap_usd=5.0), 0)
        self.assertEqual(err.getvalue().count("purpose cap $5.00, spent"), 1)
        self.assertIn("baseline: spent $1.00 of cap $5.00\n", err.getvalue())

    def test_step_spend_past_the_cap_is_called_out(self):
        def spend(*args, **kwargs):
            kledger.append({"period": self.budget["period"], "purpose": "baseline", "costUsd": 6.5})
            return 0

        err = io.StringIO()
        with (
            mock.patch.object(korch.runner, "run", side_effect=spend),
            contextlib.redirect_stderr(err),
        ):
            korch.baseline("claude-sonnet-5-5", vanilla_models=[], cap_usd=5.0)
        self.assertIn(
            "baseline: spent $6.50 of cap $5.00 (over the cap by $1.50: runs in flight finished past it)\n",
            err.getvalue(),
        )

    def test_baseline_without_cap_passes_no_limit(self):
        with mock.patch.object(korch.runner, "run", return_value=0) as run:
            korch.baseline("claude-sonnet-5-5", vanilla_models=[])
        self.assertTrue(run.called)
        self.assertTrue(all(c.kwargs.get("cap_limit") is None for c in run.call_args_list))

    def test_baseline_stops_mid_sequence_when_a_step_uses_up_the_cap(self):
        def spend(*args, **kwargs):
            kledger.append({"period": self.budget["period"], "purpose": "baseline", "costUsd": 5.0})
            return 0

        err = io.StringIO()
        with (
            mock.patch.object(korch.runner, "run", side_effect=spend) as run,
            contextlib.redirect_stderr(err),
        ):
            status = korch.baseline("claude-sonnet-5-5", vanilla_models=[], cap_usd=5.0)
        self.assertEqual(status, 2)
        self.assertEqual(run.call_count, 1)
        self.assertIn(
            "baseline: purpose cap reached (cap $5.00, spent $5.00); no step started. Raise the cap to continue.",
            err.getvalue(),
        )


# --- summary and compare ------------------------------------------------------------------------

import copy  # noqa: E402

from evals.kbeval import compare as kcompare  # noqa: E402
from evals.kbeval import summary as ksummary  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "kbeval" / "aggregate-result.json"
TWO_PROCESS = REPO / "tests" / "fixtures" / "kbeval" / "aggregate-result-two-process.json"
OPUS = "claude-opus-5-5"
SETTINGS = {"kbRuns": 3, "vanillaRuns": 3, "vanillaModels": [OPUS], "excludeTags": []}


def real_doc() -> dict:
    return json.loads(FIXTURE.read_text())


def with_runs(doc: dict, design_passes: list, errors: list | None = None) -> dict:
    """Copy the captured doc, replacing its runs with one per entry of `design_passes`."""
    doc = copy.deepcopy(doc)
    case = doc["cases"][0]
    template = case["arms"]["with"][0]
    runs = []
    for index, passed in enumerate(design_passes):
        run = copy.deepcopy(template)
        run["error"] = (errors or [None] * len(design_passes))[index]
        for grader in run["graders"]:
            if grader["name"] == "design":
                grader["passed"] = passed
        if run["error"] and "API" in run["error"]:
            run["graders"] = []
        runs.append(run)
    case["arms"] = {"with": runs}
    return doc


class SummaryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.results = Path(tmp.name)

    def put(
        self,
        staging: str,
        label: str,
        doc: dict,
        suite: str = "effects",
        keyed: bool = True,
        digest: str | None = None,
    ) -> None:
        path = self.results / staging / OPUS / suite / f"{label}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))
        if digest is not None:
            path.with_suffix(".meta.json").write_text(json.dumps({"kbDigest": digest}))
        key = path.with_suffix(".key")
        if keyed:
            key.write_text("k\n")
        elif key.exists():
            key.unlink()

    def test_unkeyed_and_partial_results_are_ignored(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True, True]))
        self.put(
            "kb",
            "j-arch@j-arch--effect-cache-layer",
            with_runs(real_doc(), [False] * 3),
            keyed=False,
        )
        partial = with_runs(real_doc(), [False] * 3)
        partial["partial"] = True
        self.put("vanilla", "j-arch", partial)
        doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)
        self.assertEqual(doc["cases"][0]["kbScore"], 1.0)
        self.assertEqual(doc["cases"][0]["vanillaRuns"], 0)

    def test_pass_boundary_is_two_thirds(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True, False]))
        self.assertTrue(
            ksummary.summarize(self.results, OPUS, "effects", SETTINGS)["cases"][0]["passed"]
        )
        self.put("kb", "j-arch", with_runs(real_doc(), [True, False, False]))
        self.assertFalse(
            ksummary.summarize(self.results, OPUS, "effects", SETTINGS)["cases"][0]["passed"]
        )

    def test_trigger_suite_scores_its_process_grader(self):
        doc = copy.deepcopy(real_doc())
        case = doc["cases"][0]
        case["name"] = "design-first--trigger-should-rate"
        case["graders"] = [g for g in case["graders"] if g["type"] == "tool_used"]
        for run in case["arms"]["with"]:
            run["graders"] = [g for g in run["graders"] if g["name"] == "delegated"]
        case["arms"]["with"] = case["arms"]["with"] * 3
        self.put("kb", "design-first", doc, suite="triggers")
        summary = ksummary.summarize(self.results, OPUS, "triggers", SETTINGS)
        (row,) = summary["cases"]
        self.assertEqual(
            (row["kind"], row["kbScore"], row["vanillaScore"]), ("trigger-should", 1.0, None)
        )
        self.assertEqual(ksummary.check(summary), [])

    def two_process(
        self, loaded: list, delegated: list, errors: list | None = None, max_zero: bool = False
    ) -> dict:
        """Return the two-process fixture with one run per entry, each grader passing as given."""
        doc = with_runs(json.loads(TWO_PROCESS.read_text()), [True] * len(loaded), errors)
        case = doc["cases"][0]
        if max_zero:
            next(g for g in case["graders"] if g["name"] == "skill-loaded")["config"]["max"] = 0
        for run, load, deleg in zip(case["arms"]["with"], loaded, delegated):
            for grader in run["graders"]:
                if grader["name"] == "skill-loaded":
                    grader["passed"] = load
                if grader["name"] == "delegated":
                    grader["passed"] = deleg
        return doc

    def row(self, doc: dict) -> dict:
        self.put("kb", "j-arch", doc)
        (row,) = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)["cases"]
        return row

    def test_load_rate_counts_only_the_own_load_grader(self):
        row = self.row(self.two_process([True] * 3, [False] * 3))
        self.assertEqual((row["loadRate"], row["behaviorRate"]), (1.0, 0.0))

    def test_behavior_rate_survives_a_missing_load_grader(self):
        row = self.row(with_runs(real_doc(), [True, True, True]))
        self.assertEqual((row["loadRate"], row["behaviorRate"]), (None, 1.0))

    def test_near_case_load_rate_inverts_a_max_zero_grader(self):
        row = self.row(self.two_process([True] * 3, [True] * 3, max_zero=True))
        self.assertEqual(row["loadRate"], 0.0)

    def test_rows_carry_no_fired_rate(self):
        row = self.row(self.two_process([True] * 3, [True] * 3))
        self.assertNotIn("firedRate", row)

    def test_capped_run_that_loaded_is_capped_not_censored(self):
        capped = [None, None, "exit 1: Reached maximum number of turns (6)"]
        base = self.row(self.two_process([True] * 3, [True] * 3))
        row = self.row(self.two_process([True] * 3, [True] * 3, capped))
        self.assertEqual((row["cappedRuns"], row["censoredRuns"]), (1, 0))
        self.assertEqual(row["kbScore"], base["kbScore"])

    def test_capped_run_that_never_loaded_is_censored(self):
        capped = [None, None, "timed out after 120s"]
        row = self.row(self.two_process([True, True, False], [True] * 3, capped))
        self.assertEqual((row["cappedRuns"], row["censoredRuns"]), (1, 1))

    def test_capped_vanilla_run_is_capped_and_never_censored(self):
        self.put(
            "vanilla",
            "j-arch",
            with_runs(real_doc(), [True] * 3, [None, None, "timed out after 120s"]),
        )
        row = self.row(self.two_process([True] * 3, [True] * 3))
        self.assertEqual((row["cappedRuns"], row["censoredRuns"]), (1, 0))

    def test_capped_near_run_is_censored_when_its_max_zero_grader_passed(self):
        capped = [None, None, "timed out after 120s"]
        row = self.row(self.two_process([True] * 3, [True] * 3, capped, max_zero=True))
        self.assertEqual(row["censoredRuns"], 1)
        row = self.row(self.two_process([True, True, False], [True] * 3, capped, max_zero=True))
        self.assertEqual(row["censoredRuns"], 0)

    def test_capped_run_with_no_graders_is_censored_and_outside_the_load_rate(self):
        doc = self.two_process([True] * 3, [True] * 3, [None, None, "timed out after 120s"])
        doc["cases"][0]["arms"]["with"][2]["graders"] = []
        row = self.row(doc)
        self.assertEqual((row["cappedRuns"], row["censoredRuns"], row["loadRate"]), (1, 1, 1.0))

    def test_error_runs_are_counted_per_case(self):
        errors = [None, "exit 1: Reached maximum number of turns (6)", "timed out after 120s"]
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True, True], errors))
        (row,) = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)["cases"]
        self.assertEqual(row["errorRuns"], 2)

    def test_write_refuses_local_paths_and_request_ids(self):
        for leak in (
            "open /Users/someone/x",
            "/home/someone/x",
            "/var/folders/ab/T/x",
            "open /private/tmp/x",
            "open /tmp/x",
            "Request ID: req_011abc",
            "Message ID: msg_011abc",
        ):
            with self.subTest(leak=leak):
                self.put(
                    "kb", "j-arch", with_runs(real_doc(), [True, True, True], [None, leak, None])
                )
                doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)
                with self.assertRaises(ValueError):
                    ksummary.write(doc, self.results / "out.json")

    def test_write_refuses_secret_shaped_text(self):
        self.put(
            "kb", "j-arch", with_runs(real_doc(), [True, True, True], [None, "sk-ant-leak", None])
        )
        doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)
        with self.assertRaises(ValueError):
            ksummary.write(doc, self.results / "out.json")

    def test_write_refuses_other_key_shapes_and_home_relative_paths(self):
        for leak in (
            "read ~/.config/tool/token",
            "key AKIAABCDEFGHIJKLMNOP",
            "token ghp_abcdefghijklmnopqrstuvwxyz0123",
            "key sk-proj-abcdefghijklmnopqrstuv",
        ):
            with self.subTest(leak=leak):
                self.put(
                    "kb", "j-arch", with_runs(real_doc(), [True, True, True], [None, leak, None])
                )
                doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)
                with self.assertRaises(ValueError):
                    ksummary.write(doc, self.results / "out.json")

    def test_write_accepts_the_cap_and_timeout_errors(self):
        errors = [None, "exit 1: Reached maximum number of turns (6)", "timed out after 120s"]
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True, True], errors))
        doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)
        ksummary.write(doc, self.results / "out.json")
        self.assertTrue((self.results / "out.json").is_file())

    def test_summary_from_real_doc(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True, False]))
        self.put("vanilla", "j-arch", with_runs(real_doc(), [True, False, False]))
        doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)
        for key in (
            "schema",
            "suite",
            "model",
            "judgeModel",
            "claudeVersion",
            "date",
            "runs",
            "costUsd",
        ):
            self.assertIn(key, doc)
        self.assertEqual(
            (doc["suite"], doc["model"], doc["judgeModel"]), ("effects", OPUS, "claude-fable-5-2")
        )
        self.assertEqual(doc["claudeVersion"], "2.1.325")
        self.assertEqual(doc["runs"], SETTINGS)
        (case,) = doc["cases"]
        self.assertEqual((case["asset"], case["kind"]), ("j-arch", "effect"))
        # Result graders only: the llm `design` grader. The tool_used `delegated` grader is a
        # process grader, reported through behaviorRate.
        self.assertAlmostEqual(case["kbScore"], 2 / 3, places=3)
        self.assertAlmostEqual(case["vanillaScore"], 1 / 3, places=3)
        self.assertAlmostEqual(case["delta"], 1 / 3, places=3)
        self.assertEqual((case["loadRate"], case["behaviorRate"]), (None, 1.0))
        self.assertEqual((case["kbRuns"], case["vanillaRuns"]), (3, 3))
        self.assertEqual(case["errors"], [])
        for arm in ("kb", "vanilla", "delta"):
            self.assertIn("mean", doc["aggregates"][arm])
            self.assertEqual(len(doc["aggregates"][arm]["ci95"]), 2)

    def test_summary_is_deterministic_and_one_case_per_line(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True, False, True]))
        doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)
        out = self.results / "summary.json"
        ksummary.write(doc, out)
        first = out.read_text()
        ksummary.write(ksummary.summarize(self.results, OPUS, "effects", SETTINGS), out)
        self.assertEqual(first, out.read_text())
        self.assertEqual(json.loads(first)["cases"][0]["asset"], "j-arch")
        self.assertTrue(
            any(line.lstrip().startswith('{"asset": "j-arch"') for line in first.splitlines())
        )

    def test_rerun_result_replaces_the_api_error_case(self):
        self.put(
            "kb",
            "j-arch",
            with_runs(real_doc(), [True, True, True], [None, "API error: 429", None]),
        )
        self.put(
            "kb", "j-arch@j-arch--effect-cache-layer", with_runs(real_doc(), [True, True, True])
        )
        doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)
        self.assertEqual(doc["cases"][0]["kbRuns"], 3)
        self.assertEqual(doc["cases"][0]["errors"], [])

    def test_summarize_check(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True]))
        self.put("vanilla", "j-arch", with_runs(real_doc(), [True, True, True]))
        problems = ksummary.check(ksummary.summarize(self.results, OPUS, "effects", SETTINGS))
        self.assertTrue(any("kbRuns" in p for p in problems), problems)
        self.put(
            "kb",
            "j-arch",
            with_runs(real_doc(), [True, True, True], [None, None, "API error: 529"]),
        )
        problems = ksummary.check(ksummary.summarize(self.results, OPUS, "effects", SETTINGS))
        self.assertTrue(any("API" in p for p in problems), problems)
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True, True]))
        self.assertEqual(
            ksummary.check(ksummary.summarize(self.results, OPUS, "effects", SETTINGS)), []
        )

    def test_check_rejects_a_self_judged_summary(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True, True]))
        self.put("vanilla", "j-arch", with_runs(real_doc(), [True, True, True]))
        doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)
        self.assertEqual(ksummary.check(doc), [])
        doc["model"] = "claude-fable-5-3"
        self.assertIn("claude-fable-5-3/effects: judged by its own family", ksummary.check(doc))
        doc["judgeModel"] = None
        self.assertIn("claude-fable-5-3/effects: judged by its own family", ksummary.check(doc))

    def test_check_skips_vanilla_for_unlisted_models(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True, True]))
        settings = {**SETTINGS, "vanillaModels": ["claude-sonnet-5-5"]}
        self.assertEqual(
            ksummary.check(ksummary.summarize(self.results, OPUS, "effects", settings)), []
        )

    def test_mixed_claude_versions_are_refused(self):
        other = real_doc()
        other["claudeVersion"] = "2.1.999"
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True, True]))
        self.put("vanilla", "j-arch", with_runs(other, [True, True, True]))
        with self.assertRaises(ValueError):
            ksummary.summarize(self.results, OPUS, "effects", SETTINGS)

    @staticmethod
    def renamed(name: str) -> dict:
        doc = with_runs(real_doc(), [True] * 3)
        doc["cases"][0]["name"] = name
        return doc

    def test_mixed_kb_digests_are_refused(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True] * 3), digest="aaa")
        self.put("kb", "j-arch--other", self.renamed("j-arch--effect-other"), digest="bbb")
        with self.assertRaisesRegex(
            ValueError,
            "results mix KB digests: aaa, bbb; rerun that staging or move the older results aside$",
        ):
            ksummary.summarize(self.results, OPUS, "effects", SETTINGS)

    def test_kb_and_vanilla_may_carry_different_digests(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True] * 3), digest="aaa")
        self.put("vanilla", "j-arch", with_runs(real_doc(), [True] * 3), digest="bbb")
        doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)
        self.assertEqual(doc["kbDigest"], {"kb": "aaa", "vanilla": "bbb"})

    def test_legacy_results_summarize_as_one_unrecorded_digest(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True] * 3))
        doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)
        self.assertEqual(doc["kbDigest"], {"kb": "unrecorded", "vanilla": None})

    def test_legacy_mixed_with_recorded_is_refused(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True] * 3))
        self.put("kb", "j-arch--other", self.renamed("j-arch--effect-other"), digest="aaa")
        with self.assertRaisesRegex(ValueError, "results mix KB digests"):
            ksummary.summarize(self.results, OPUS, "effects", SETTINGS)

    def test_mixed_vanilla_digests_are_refused(self):
        self.put("vanilla", "j-arch", with_runs(real_doc(), [True] * 3), digest="aaa")
        self.put("vanilla", "j-arch--other", self.renamed("j-arch--effect-other"), digest="bbb")
        with self.assertRaisesRegex(ValueError, "results mix KB digests"):
            ksummary.summarize(self.results, OPUS, "effects", SETTINGS)

    def test_results_for_deleted_cases_are_dropped(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True] * 3), digest="aaa")
        live = {"j-arch--effect-cache-layer"}
        doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS, live=live)
        self.assertEqual([c["name"] for c in doc["cases"]], ["j-arch--effect-cache-layer"])
        doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS, live={"other--effect-x"})
        self.assertEqual(doc["cases"], [])
        self.assertEqual(doc["kbDigest"], {"kb": None, "vanilla": None})

    def test_exact_case_rerun_newer_than_its_batch_wins_in_summarize(self):
        old = with_runs(real_doc(), [False] * 3)
        old["startedAt"] = "2026-11-15T10:00:00.000Z"
        new = with_runs(real_doc(), [True] * 3)
        new["startedAt"] = "2026-11-16T10:00:00.000Z"
        self.put("kb", "j-arch", old)
        self.put("kb", "j-arch--effect-cache-layer", new)
        (row,) = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)["cases"]
        self.assertEqual(row["kbScore"], 1.0)

    def test_fully_shadowed_document_leaves_version_and_cost_alone(self):
        old = with_runs(real_doc(), [False] * 3)
        old["startedAt"] = "2026-11-15T10:00:00.000Z"
        old["claudeVersion"] = "1.0.0"
        old["costUsd"] = 50.0
        new = with_runs(real_doc(), [True] * 3)
        new["startedAt"] = "2026-11-16T10:00:00.000Z"
        new["costUsd"] = 2.0
        self.put("kb", "j-arch", old)
        self.put("kb", "j-arch--effect-cache-layer", new)
        doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)
        self.assertEqual((doc["claudeVersion"], doc["costUsd"]), ("2.1.325", 2.0))

    def test_document_outside_live_contributes_no_cost(self):
        kept = with_runs(real_doc(), [True] * 3)
        kept["costUsd"] = 2.0
        gone = self.renamed("j-arch--effect-gone")
        gone["costUsd"] = 50.0
        self.put("kb", "j-arch", kept)
        self.put("kb", "j-arch--gone", gone)
        doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS, live={CASE})
        self.assertEqual(doc["costUsd"], 2.0)

    def test_notes_report_skipped_files_and_dropped_cases(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True] * 3))
        self.put("kb", "j-arch--gone", self.renamed("j-arch--effect-gone"))
        self.put("kb", "j-arch--late", self.renamed("j-arch--effect-late"), keyed=False)
        notes: list = []
        ksummary.summarize(self.results, OPUS, "effects", SETTINGS, live={CASE}, notes=notes)
        late = self.results / "kb" / OPUS / "effects" / "j-arch--late.json"
        self.assertEqual(
            notes,
            [
                f"kb/{OPUS}/effects: skipped {late} (no .key)",
                f"kb/{OPUS}/effects: dropped results for 1 cases no longer under evals/cases: j-arch--effect-gone",
            ],
        )

    def test_determination_load_sees_exact_case_files(self):
        old = with_runs(real_doc(), [False] * 3)
        old["startedAt"] = "2026-11-15T10:00:00.000Z"
        new = with_runs(real_doc(), [True] * 3)
        new["startedAt"] = "2026-11-16T10:00:00.000Z"
        self.put("kb", "j-arch", old)
        self.put("kb", "j-arch--effect-cache-layer", new)
        purpose = self.results.name
        with mock.patch.object(kdet, "RESULTS", self.results.parent):
            cases = kdet.load(purpose, OPUS, "j-arch") or {}
            self.assertIsNone(kdet.load(purpose, OPUS, "ghost"))
        self.assertEqual(list(cases), ["j-arch--effect-cache-layer"])
        self.assertEqual(
            [kdet.passed(r, "design") for r in kdet.runs(cases["j-arch--effect-cache-layer"])],
            [True] * 3,
        )


from evals.kbeval import results as kresults  # noqa: E402

CASE = "j-arch--effect-cache-layer"


class ResultsLoaderTests(unittest.TestCase):
    """`results.load_cases`: one precedence rule for every reader of raw results."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "p" / "kb" / OPUS / "effects"
        self.root.mkdir(parents=True)

    def put(self, label, doc, started=None, keyed=True, digest=None):
        if started is not None:
            doc["startedAt"] = started
        path = self.root / f"{label}.json"
        path.write_text(json.dumps(doc))
        if keyed:
            path.with_suffix(".key").write_text("k\n")
        if digest is not None:
            (self.root / f"{label}.meta.json").write_text(json.dumps({"kbDigest": digest}))

    def design_passes(self, loaded):
        return [kresults.passed(r, "design") for r in kresults.runs(loaded.cases[CASE])]

    def test_exact_case_rerun_wins_over_older_asset_batch(self):
        self.put("j-arch", with_runs(real_doc(), [False] * 3), "2026-11-15T10:00:00.000Z")
        self.put(CASE, with_runs(real_doc(), [True] * 3), "2026-11-16T10:00:00.000Z")
        self.assertEqual(self.design_passes(kresults.load_cases(self.root)), [True] * 3)
        self.assertEqual(
            self.design_passes(kresults.load_cases(self.root, asset="j-arch")), [True] * 3
        )

    def test_newer_asset_batch_wins_over_older_exact_case_file(self):
        self.put("j-arch", with_runs(real_doc(), [False] * 3), "2026-11-16T10:00:00.000Z")
        self.put(CASE, with_runs(real_doc(), [True] * 3), "2026-11-15T10:00:00.000Z")
        self.assertEqual(self.design_passes(kresults.load_cases(self.root)), [False] * 3)

    def test_at_rerun_wins_on_a_startedat_tie(self):
        same = "2026-11-15T10:00:00.000Z"
        self.put("j-arch", with_runs(real_doc(), [False] * 3), same)
        self.put(CASE, with_runs(real_doc(), [False] * 3), same)
        self.put(f"j-arch@{CASE}", with_runs(real_doc(), [True] * 3), same)
        self.assertEqual(self.design_passes(kresults.load_cases(self.root)), [True] * 3)

    def test_exact_case_file_wins_over_asset_batch_on_a_startedat_tie(self):
        same = "2026-11-15T10:00:00.000Z"
        self.put("j-arch", with_runs(real_doc(), [False] * 3), same)
        self.put(CASE, with_runs(real_doc(), [True] * 3), same)
        self.assertEqual(self.design_passes(kresults.load_cases(self.root)), [True] * 3)

    def test_keyless_and_partial_files_are_listed_as_skipped(self):
        self.put("j-arch", with_runs(real_doc(), [True] * 3))
        self.put(CASE, with_runs(real_doc(), [False] * 3), keyed=False)
        partial = with_runs(real_doc(), [False] * 3)
        partial["partial"] = True
        self.put(f"j-arch@{CASE}", partial)
        loaded = kresults.load_cases(self.root)
        self.assertEqual(self.design_passes(loaded), [True] * 3)
        self.assertEqual(len(loaded.docs), 1)
        self.assertEqual(
            loaded.skipped,
            [
                f"{self.root / CASE}.json (no .key)",
                f"{self.root / f'j-arch@{CASE}'}.json (partial)",
            ],
        )

    def test_sidecar_is_neither_skipped_nor_read_as_a_document(self):
        self.put("j-arch", with_runs(real_doc(), [True] * 3), digest="aaa")
        loaded = kresults.load_cases(self.root)
        self.assertEqual(loaded.skipped, [])
        self.assertEqual(len(loaded.docs), 1)
        self.assertNotIn("kbDigest", loaded.docs[0])
        self.assertEqual(loaded.digests, {"aaa"})

    def test_asset_filter_reads_only_that_asset(self):
        plan = copy.deepcopy(real_doc())
        plan["cases"][0]["name"] = "j-plan--effect-x"
        arch = real_doc()
        stray = copy.deepcopy(arch["cases"][0])
        stray["name"] = "other--effect-z"
        arch["cases"].append(stray)
        self.put("j-arch", arch)
        self.put("j-plan", plan)
        self.assertEqual(
            set(kresults.load_cases(self.root).cases), {CASE, "j-plan--effect-x", "other--effect-z"}
        )
        loaded = kresults.load_cases(self.root, asset="j-arch")
        self.assertEqual(list(loaded.cases), [CASE])
        self.assertEqual(len(loaded.docs), 1)
        self.assertEqual(kresults.load_cases(self.root, asset="ghost").docs, [])

    def test_cases_outside_live_are_dropped(self):
        self.put("j-arch", real_doc())
        self.assertEqual(kresults.load_cases(self.root, live={"other--effect-y"}).cases, {})
        self.assertEqual(list(kresults.load_cases(self.root, live={CASE}).cases), [CASE])

    def test_dropped_lists_case_names_outside_live(self):
        doc = real_doc()
        for name in ("j-arch--effect-b", "j-arch--effect-a"):
            extra = copy.deepcopy(doc["cases"][0])
            extra["name"] = name
            doc["cases"].append(extra)
        self.put("j-arch", doc)
        self.put("j-arch--effect-a", self.renamed_doc("j-arch--effect-a"))
        loaded = kresults.load_cases(self.root, live={CASE})
        self.assertEqual(loaded.dropped, ["j-arch--effect-a", "j-arch--effect-b"])
        self.assertEqual(kresults.load_cases(self.root).dropped, [])

    @staticmethod
    def renamed_doc(name):
        doc = real_doc()
        doc["cases"][0]["name"] = name
        return doc

    def test_digests_come_only_from_documents_that_supply_a_winning_case(self):
        self.put("j-arch", real_doc(), "2026-11-15T10:00:00.000Z", digest="old")
        self.put(CASE, real_doc(), "2026-11-16T10:00:00.000Z", digest="new")
        self.assertEqual(kresults.load_cases(self.root).digests, {"new"})

    def test_fully_shadowed_document_is_not_in_docs(self):
        old = real_doc()
        old["claudeVersion"] = "1.0.0"
        self.put("j-arch", old, "2026-11-15T10:00:00.000Z")
        self.put(CASE, real_doc(), "2026-11-16T10:00:00.000Z")
        loaded = kresults.load_cases(self.root)
        self.assertEqual([d["claudeVersion"] for d in loaded.docs], ["2.1.325"])

    def test_corrupt_result_file_names_its_path(self):
        self.put("j-arch", real_doc())
        path = self.root / "j-arch.json"
        path.write_text('{"cases": [', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, f"{re.escape(str(path))}: unreadable result"):
            kresults.load_cases(self.root)

    def test_corrupt_sidecar_names_its_path(self):
        self.put("j-arch", real_doc(), digest="aaa")
        sidecar = self.root / "j-arch.meta.json"
        sidecar.write_text('{"kbDig', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, f"{re.escape(str(sidecar))}: unreadable sidecar"):
            kresults.load_cases(self.root)

    def test_own_load_grader_matches_skill_and_agent_tools(self):
        case = {
            "graders": [
                {
                    "name": "other",
                    "config": {
                        "tool": "Agent",
                        "input_match": '"subagent_type"\\s*:\\s*"kb:code-reviewer"',
                    },
                },
                {
                    "name": "dispatched",
                    "config": {
                        "tool": "Agent",
                        "input_match": '"subagent_type"\\s*:\\s*"kb:security-reviewer"',
                    },
                },
                {
                    "name": "skill",
                    "config": {"tool": "Skill", "input_match": '"skill"\\s*:\\s*"(?:kb:)?j-arch"'},
                },
            ]
        }
        self.assertEqual(kresults.own_load_grader(case, "security-reviewer"), "dispatched")
        self.assertEqual(kresults.own_load_grader(case, "j-arch"), "skill")
        self.assertIsNone(kresults.own_load_grader(case, "j-plan"))

    def test_own_load_grader_does_not_match_a_hyphen_suffix_asset(self):
        prefixed = {
            "graders": [
                {
                    "name": "skill",
                    "config": {
                        "tool": "Skill",
                        "input_match": '"skill"\\s*:\\s*"(?:kb:)?j-create-pr"',
                    },
                },
            ]
        }
        self.assertIsNone(kresults.own_load_grader(prefixed, "create-pr"))
        exact = {
            "graders": [
                {
                    "name": "skill",
                    "config": {"tool": "Skill", "input_match": '"skill"\\s*:\\s*"kb:create-pr"'},
                },
            ]
        }
        self.assertEqual(kresults.own_load_grader(exact, "create-pr"), "skill")

    @staticmethod
    def scored(passes):
        return {
            CASE: {
                "graders": [{"name": "design", "type": "llm"}],
                "arms": {
                    "with": [{"graders": [{"name": "design", "passed": ok}]} for ok in passes]
                },
            }
        }

    def test_drop_of_two_thirds_on_one_model_regresses(self):
        before = {OPUS: self.scored([True] * 3), kdet.SONNET: self.scored([True] * 3)}
        after = {OPUS: self.scored([True, False, False]), kdet.SONNET: self.scored([True] * 3)}
        self.assertEqual(kresults.regressed(before, after, {}), [CASE])

    def test_drop_of_one_third_on_both_models_regresses(self):
        before = {OPUS: self.scored([True] * 3), kdet.SONNET: self.scored([True] * 3)}
        after = {m: self.scored([True, True, False]) for m in before}
        self.assertEqual(kresults.regressed(before, after, {}), [CASE])

    def test_drop_of_one_third_on_one_model_does_not(self):
        before = {OPUS: self.scored([True] * 3), kdet.SONNET: self.scored([True] * 3)}
        after = {OPUS: self.scored([True, True, False]), kdet.SONNET: self.scored([True] * 3)}
        self.assertEqual(kresults.regressed(before, after, {}), [])

    def test_one_third_rule_needs_two_models(self):
        before = {OPUS: self.scored([True] * 3)}
        self.assertEqual(
            kresults.regressed(before, {OPUS: self.scored([True, True, False])}, {}), []
        )
        self.assertEqual(
            kresults.regressed(before, {OPUS: self.scored([True, False, False])}, {}), [CASE]
        )

    def test_skipped_grader_does_not_regress(self):
        before = {OPUS: self.scored([True] * 3), kdet.SONNET: self.scored([True] * 3)}
        after = {OPUS: self.scored([True, False, False]), kdet.SONNET: self.scored([True] * 3)}
        self.assertEqual(kresults.regressed(before, after, {}), [CASE])
        self.assertEqual(kresults.regressed(before, after, {CASE: {"design"}}), [])

    def test_case_after_but_not_before_is_skipped(self):
        after = {OPUS: self.scored([False] * 3)}
        befores: list[dict[str, dict | None]] = [{OPUS: {}}, {}, {OPUS: None}]
        for before in befores:
            self.assertEqual(kresults.paired_scores(before, after, {}), [])
            self.assertEqual(kresults.regressed(before, after, {}), [])

    @staticmethod
    def three_graders(passes):
        """Return a case whose runs pass the first n of three result graders, n per run."""
        names = ("g1", "g2", "g3")
        return {
            CASE: {
                "graders": [{"name": n, "type": "llm"} for n in names],
                "arms": {
                    "with": [
                        {"graders": [{"name": n, "passed": i < count} for i, n in enumerate(names)]}
                        for count in passes
                    ]
                },
            }
        }

    def test_exact_third_drops_from_three_run_means_regress(self):
        # (1+1+2/3)/3 - (0+0+1/3)/3 and (0+2/3+1)/3 - (0+0+2/3)/3 fall a hair short of 2/3 and 1/3 in floats.
        steady = self.three_graders([3, 3, 3])
        before = {OPUS: self.three_graders([1, 3, 3]), kdet.SONNET: steady}
        after = {OPUS: self.three_graders([0, 0, 1]), kdet.SONNET: steady}
        self.assertEqual(kresults.regressed(before, after, {}), [CASE])
        before = {m: self.three_graders([0, 2, 3]) for m in (OPUS, kdet.SONNET)}
        after = {m: self.three_graders([0, 0, 2]) for m in (OPUS, kdet.SONNET)}
        self.assertEqual(kresults.regressed(before, after, {}), [CASE])

    def test_no_regression_applies_the_real_skip_table(self):
        name = "j-data--effect-orders-schema"
        self.assertIn("no-serial", kdet.SKIP[name])
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)

        def case(serial):
            return {"name": name, "graders": [{"name": "design", "type": "llm"}, {"name": "no-serial", "type": "regex"}],
                    "arms": {"with": [{"graders": [{"name": "design", "passed": True},
                                                   {"name": "no-serial", "passed": serial}]}] * 3}}  # fmt: skip

        for model in (OPUS, kdet.SONNET):
            root = tmp / "baseline" / "kb" / model / "effects"
            root.mkdir(parents=True)
            (root / "j-data.json").write_text(
                json.dumps({"partial": False, "cases": [case(True)]}), encoding="utf-8"
            )
            (root / "j-data.key").write_text("k", encoding="utf-8")
        data = {m: {name: case(False)} for m in (OPUS, kdet.SONNET)}
        with (
            mock.patch.object(kdet, "RESULTS", tmp),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            kdet.Checker("p").no_regression("j-data", data)
        self.assertTrue(
            out.getvalue().startswith("CHECK no-regression j-data PASS "), out.getvalue()
        )

    def test_no_regression_line_is_unchanged(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)

        def put(purpose, model, passes):
            root = tmp / purpose / "kb" / model / "effects"
            root.mkdir(parents=True, exist_ok=True)
            (root / "j-arch.json").write_text(
                json.dumps(
                    {"partial": False, "cases": [{"name": CASE, **self.scored(passes)[CASE]}]}
                ),
                encoding="utf-8",
            )
            (root / "j-arch.key").write_text("k", encoding="utf-8")

        for model in (OPUS, kdet.SONNET):
            put("baseline", model, [True] * 3)
        expected = {
            "regressing": (
                {OPUS: [True, False, False], kdet.SONNET: [True] * 3},
                f"CHECK no-regression j-arch FAIL {CASE}@opus=1.00->0.33 {CASE}@sonnet=1.00->1.00"
                f" regressed: {CASE}\n",
            ),
            "clean": (
                {OPUS: [True] * 3, kdet.SONNET: [True, True, False]},
                f"CHECK no-regression j-arch PASS {CASE}@opus=1.00->1.00 {CASE}@sonnet=1.00->0.67\n",
            ),
        }
        for label, (passes, line) in expected.items():
            with self.subTest(label), mock.patch.object(kdet, "RESULTS", tmp):
                data = {m: self.scored(p) for m, p in passes.items()}
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    kdet.Checker("p").no_regression("j-arch", data)
                self.assertEqual(out.getvalue(), line)


def summary_doc(cases: dict, model: str = OPUS) -> dict:
    return {
        "schema": 1, "suite": "effects", "model": model,
        "aggregates": {
            "kb": {"mean": sum(c[0] for c in cases.values()) / len(cases), "ci95": [0.4, 0.9]},
            "vanilla": {"mean": 0.3, "ci95": [0.1, 0.5]},
            "delta": {"mean": 0.3, "ci95": [0.1, 0.5]},
        },
        "cases": [
            {
                "asset": name.split("--")[0], "kind": "effect", "name": name, "kbScore": values[0],
                "passed": values[1], "vanillaScore": values[2] if len(values) > 2 else None,
            }
            for name, values in sorted(cases.items())
        ],
    }  # fmt: skip


class CompareTests(unittest.TestCase):
    def test_compare_overlap_only(self):
        old = summary_doc(
            {"a--effect-x": (1.0, True), "b--effect-y": (0.5, False), "c--effect-gone": (1.0, True)}
        )
        new = summary_doc(
            {"a--effect-x": (0.6, False), "b--effect-y": (0.5, False), "d--effect-new": (1.0, True)}
        )
        report = "\n".join(kcompare.compare(old, new))
        self.assertIn("suite mean", report)
        self.assertIn("[0.40, 0.90]", report)
        self.assertRegex(report, r"a--effect-x.*1\.00 -> 0\.60")
        self.assertRegex(report, r"a--effect-x.*pass -> fail")
        self.assertNotRegex(report, r"b--effect-y.*->")
        self.assertRegex(report, r"c--effect-gone.*not run here")
        self.assertRegex(report, r"d--effect-new.*not run here")
        self.assertNotIn("removed", report)

    def test_drop_of_exactly_one_third_is_reported(self):
        old = summary_doc({"a--effect-x": (1.0, True)})
        new = summary_doc({"a--effect-x": (2 / 3, True)})
        self.assertRegex("\n".join(kcompare.compare(old, new)), r"a--effect-x.*1\.00 -> 0\.67")

    def test_small_drop_is_not_a_regression(self):
        old = summary_doc({"a--effect-x": (1.0, True)})
        new = summary_doc({"a--effect-x": (0.75, True)})
        self.assertNotRegex("\n".join(kcompare.compare(old, new)), r"a--effect-x.*->")


def headroom_rows(*models: dict) -> dict:
    """Return rows_by_model for `classify`: each model maps case -> (kb, vanilla)."""
    names = ("m1", "m2", "m3")
    return {
        names[i]: {
            case: {"name": case, "kbScore": kb, "vanillaScore": vanilla}
            for case, (kb, vanilla) in cases.items()
        }
        for i, cases in enumerate(models)
    }


class HeadroomTests(unittest.TestCase):
    def classify(self, *models: dict) -> dict:
        return kheadroom.classify(headroom_rows(*models))

    def test_saturated_needs_one_in_both_arms_on_every_model(self):
        got = self.classify({"a--effect-x": (1.0, 1.0)}, {"a--effect-x": (1.0, 1.0)})
        self.assertEqual(got, {"a--effect-x": "saturated"})
        got = self.classify({"a--effect-x": (1.0, 1.0)}, {"a--effect-x": (1.0, 0.0)})
        self.assertEqual(got, {"a--effect-x": "discriminating"})

    def test_one_arm_below_one_is_discriminating(self):
        self.assertEqual(
            self.classify({"a--effect-x": (1.0, 2 / 3)}), {"a--effect-x": "discriminating"}
        )
        self.assertEqual(
            self.classify({"a--effect-x": (2 / 3, 1.0)}), {"a--effect-x": "kb-below-vanilla"}
        )

    def test_dead_is_below_the_pass_line_everywhere(self):
        got = self.classify({"a--effect-x": (0.0, 0.0)}, {"a--effect-x": (1 / 3, 0.0)})
        self.assertEqual(got, {"a--effect-x": "dead"})
        got = self.classify({"a--effect-x": (0.0, 0.0)}, {"a--effect-x": (2 / 3, 0.0)})
        self.assertEqual(got, {"a--effect-x": "discriminating"})

    def test_kb_below_vanilla_on_every_model(self):
        got = self.classify({"a--effect-x": (0.5, 0.8)}, {"a--effect-x": (0.7, 1.0)})
        self.assertEqual(got, {"a--effect-x": "kb-below-vanilla"})
        got = self.classify({"a--effect-x": (0.5, 0.8)}, {"a--effect-x": (0.8, 0.8)})
        self.assertEqual(got, {"a--effect-x": "discriminating"})

    def test_none_score_is_unmeasured(self):
        self.assertEqual(self.classify({"a--effect-x": (None, 1.0)}), {"a--effect-x": "unmeasured"})
        self.assertEqual(self.classify({"a--effect-x": (1.0, None)}), {"a--effect-x": "unmeasured"})
        got = self.classify({"a--effect-x": (1.0, 1.0)}, {})
        self.assertEqual(got, {"a--effect-x": "unmeasured"})

    def test_precedence_saturated_then_dead_then_kb_below_vanilla(self):
        got = self.classify(
            {"s--effect-x": (1.0, 1.0), "d--effect-x": (0.0, 0.0), "k--effect-x": (0.0, 0.5)}
        )
        self.assertEqual(got["s--effect-x"], "saturated")
        self.assertEqual(got["d--effect-x"], "dead")
        self.assertEqual(got["k--effect-x"], "dead")  # dead outranks kb-below-vanilla
        got = self.classify({"u--effect-x": (None, None)})
        self.assertEqual(got["u--effect-x"], "unmeasured")  # unmeasured outranks the rest

    def write_baselines(self, root: Path, docs: dict) -> Path:
        for model, cases in docs.items():
            for suite in kheadroom.SUITES:
                doc = summary_doc(cases.get(suite, {"z--effect-z": (0.5, True, 0.0)}), model)
                path = root / model / f"{suite}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(doc))
        return root

    def test_asset_with_only_saturated_cases_is_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write_baselines(
                Path(tmp),
                {
                    OPUS: {
                        "effects": {
                            "a--effect-x": (1.0, True, 1.0),
                            "a--effect-y": (1.0, True, 1.0),
                            "b--effect-x": (1.0, True, 1.0),
                            "b--effect-y": (1.0, True, 0.0),
                        }
                    }
                },
            )
            lines = kheadroom.report(root, {})
        self.assertIn("ASSET a no discriminating case", lines)
        self.assertNotIn("ASSET b no discriminating case", lines)
        self.assertIn(f"HEADROOM saturated a--effect-x {OPUS}=1.00/1.00", lines)
        self.assertIn(
            "4 cases: 3 saturated, 0 dead, 0 kb-below-vanilla, 1 discriminating, 0 unmeasured",
            lines,
        )

    def test_discriminating_case_sorted_before_a_saturated_one_keeps_the_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write_baselines(
                Path(tmp),
                {
                    OPUS: {
                        "effects": {
                            "a--effect-a": (1.0, True, 0.0),
                            "a--effect-b": (1.0, True, 1.0),
                        }
                    }
                },
            )
            lines = kheadroom.report(root, {})
        self.assertNotIn("ASSET a no discriminating case", lines)

    def test_baselines_with_no_model_directory_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "no model directories"):
                kheadroom.report(Path(tmp), {})

    def test_workflow_cases_get_no_asset_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write_baselines(
                Path(tmp),
                {
                    OPUS: {
                        "effects": {"e--effect-x": (1.0, True, 0.0)},
                        "workflows": {"wf--plan-to-review": (1.0, True, 1.0)},
                    }
                },
            )
            lines = kheadroom.report(root, {})
        self.assertIn(f"HEADROOM saturated wf--plan-to-review {OPUS}=1.00/1.00", lines)
        self.assertEqual([line for line in lines if line.startswith("ASSET ")], [])

    def test_missing_live_case_is_listed_and_not_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write_baselines(
                Path(tmp), {OPUS: {"effects": {"a--effect-x": (1.0, True, 0.0)}}}
            )
            lines = kheadroom.report(root, {"effects": {"a--effect-x", "a--effect-gone"}})
        self.assertIn("MISSING a--effect-gone", lines)
        self.assertNotIn("MISSING a--effect-x", lines)
        self.assertIn(
            "1 cases: 0 saturated, 0 dead, 0 kb-below-vanilla, 1 discriminating, 0 unmeasured",
            lines,
        )


class RunnerSafetyTests(RunnerCase):
    def test_truncated_json_keeps_the_cap_charged(self):
        def responder(args):
            path = Path(args[args.index("--json") + 1])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"cases": [')
            return None, 143, "killed"

        code, err = self.run_suite(FakeCli(responder))
        self.assertNotEqual(code, 0)
        self.assertIn("no readable JSON", err)
        self.assertEqual(
            [(e["charged"], e["costUsd"]) for e in self.entries()], [("reserved", 60.0)]
        )

    def test_load_error_charges_nothing(self):
        cli = FakeCli(lambda args: (None, 1, "Error: No eval cases found"))
        code, err = self.run_suite(cli)
        self.assertEqual(code, 1)
        self.assertAlmostEqual(sum(e["costUsd"] for e in self.entries()), 0.0)
        self.assertIn("charged $0", err)

    def test_interrupt_keeps_the_reservation(self):
        def responder(args):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.run_suite(FakeCli(responder))
        self.assertEqual([e["charged"] for e in self.entries()], ["reserved"])

    def test_bad_cost_values_abort_with_the_cap_charged(self):
        for cost in (float("nan"), -1.0, "1.0"):
            with self.subTest(cost=cost):
                self.ledger_path.unlink(missing_ok=True)
                shutil.rmtree(self.state, ignore_errors=True)
                cli = FakeCli(lambda args: (result_doc({"x": [None]}, cost=cost), 0, ""))
                code, _ = self.run_suite(cli)
                self.assertNotEqual(code, 0)
                self.assertEqual([e["charged"] for e in self.entries()], ["reserved"])

    def test_other_nonzero_exit_is_not_cached(self):
        cli = FakeCli(lambda args: (result_doc({"j-demo--effect-cache": [None]}), 137, ""))
        code, _ = self.run_suite(cli)
        self.assertEqual(code, 137)
        again = FakeCli(lambda args: (result_doc({"j-demo--effect-cache": [None]}), 0, ""))
        self.run_suite(again)
        self.assertEqual(len(again.calls), 1)

    def test_kb_edit_invalidates_the_cache(self):
        cli = FakeCli(lambda args: (result_doc({"j-demo--effect-cache": [None]}), 0, ""))
        self.run_suite(cli)
        skill = self.plugin / "skills" / "demo-skill" / "SKILL.md"
        skill.write_text(skill.read_text() + "\nNew rule.\n")
        again = FakeCli(lambda args: (result_doc({"j-demo--effect-cache": [None]}), 0, ""))
        self.run_suite(again)
        self.assertEqual(len(again.calls), 1)

    def test_rerunning_a_batch_discards_its_old_case_reruns(self):
        def responder(args):
            if case_arg(args) == "j-demo--*":
                return result_doc({"j-demo--effect-cache": ["API error: 529 overloaded"]}), 1, ""
            return result_doc({case_arg(args): [None]}), 0, ""

        self.run_suite(FakeCli(responder))
        base = self.state / "test" / "kb" / MODEL / "effects"
        self.assertTrue((base / "j-demo@j-demo--effect-cache.json").exists())
        prompt = self.plugin / "evals-effects" / "j-demo--effect-cache" / "prompt.md"
        prompt.write_text(prompt.read_text() + "\nChanged.\n")
        self.run_suite(FakeCli(lambda args: (result_doc({"j-demo--effect-cache": [None]}), 0, "")))
        self.assertFalse((base / "j-demo@j-demo--effect-cache.json").exists())

    def test_leaked_result_is_quarantined(self):
        doc = result_doc({"j-demo--effect-cache": ["sk-ant-oops"]})
        code, _ = self.run_suite(FakeCli(lambda args: (doc, 0, "")))
        self.assertEqual(code, 3)
        base = self.state / "test" / "kb" / MODEL / "effects"
        self.assertFalse((base / "j-demo.json").exists())
        self.assertTrue((base / "j-demo.quarantined").exists())

    def test_batch_allowance_bounds_the_cap(self):
        self.budget["ceilingUsd"] = 1000
        cli = FakeCli(lambda args: (result_doc({"x": [None]}), 0, ""))
        self.run_suite(cli)
        # one case x 3 runs x max(10 x $5 p95, $5) = $150, below the $960 budget headroom
        self.assertEqual(cli.calls[0][cli.calls[0].index("--max-cost-usd") + 1], "150.00")

    def test_api_error_pattern(self):
        for text in (
            "API error: 429",
            "rate_limit_error",
            "status 529",
            "ECONNRESET",
            "Overloaded",
        ):
            self.assertRegex(text, krunner.API_ERROR)
        for text in (
            "timed out after 500 ms",
            "hit max_turns (40)",
            "exit 1: Reached maximum number of turns (3)",
        ):
            self.assertNotRegex(text, krunner.API_ERROR)

    def test_jobs_below_one_is_refused(self):
        cli = FakeCli(lambda args: (result_doc({"x": [None]}), 0, ""))
        err = io.StringIO()
        options = {
            "plugin": self.plugin,
            "state": self.state,
            "ledger_path": self.ledger_path,
            "budget": self.budget,
            "p95": 5.0,
        }
        with contextlib.redirect_stderr(err):
            code = krunner.run("effects", "kb", MODEL, 3, "test", ["j-demo"], 0, cli, **options)
        self.assertEqual(code, 1)
        self.assertEqual(cli.calls, [])


class LedgerSafetyTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "ledger.jsonl"

    def test_reserve_refuses_past_the_ceiling(self):
        kledger.append({"period": "p", "costUsd": 50.0}, self.path)
        self.assertTrue(
            kledger.reserve(
                {"period": "p", "costUsd": 30.0, "charged": "reserved"}, 100, 20, self.path
            )
        )
        self.assertFalse(
            kledger.reserve(
                {"period": "p", "costUsd": 0.5, "charged": "reserved"}, 100, 20, self.path
            )
        )
        self.assertAlmostEqual(kledger.spent("p", self.path), 80.0)

    def test_torn_line_fails_closed_naming_the_line(self):
        kledger.append({"period": "p", "costUsd": 1.0}, self.path)
        with open(self.path, "a") as handle:
            handle.write('{"period": "p", "cost')
        with self.assertRaisesRegex(kledger.LedgerError, r"ledger\.jsonl:2"):
            kledger.spent("p", self.path)

    def test_invalid_costs_are_refused(self):
        for entry in ({"period": "p", "costUsd": float("nan")}, {"period": "p", "costUsd": -1.0}):
            with self.subTest(entry=entry), self.assertRaises(kledger.LedgerError):
                kledger.append(entry, self.path)
        kledger.append({"period": "p", "costUsd": -1.0, "charged": "settle"}, self.path)

    def test_append_waits_for_the_lock(self):
        import fcntl
        import time

        self.path.write_text("")
        script = (
            "import sys; from pathlib import Path; from evals.kbeval import ledger\n"
            "ledger.append({'period': 'p', 'costUsd': 1.0}, Path(sys.argv[1]))\n"
        )
        with open(self.path, "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            proc = subprocess.Popen(
                [sys.executable, "-c", script, str(self.path)], cwd=REPO, stdin=subprocess.DEVNULL
            )
            time.sleep(0.5)
            self.assertEqual(self.path.read_text(), "")
            fcntl.flock(handle, fcntl.LOCK_UN)
        self.assertEqual(proc.wait(timeout=60), 0)
        self.assertEqual(len(self.path.read_text().splitlines()), 1)


class OrchestratePlanTests(OrchestrateTests):
    def test_pilot_runs_both_models_once_under_its_cap(self):
        cli = FakeCli(lambda args: (result_doc({case_arg(args): [None]}, cost=0.1), 0, ""))
        with mock.patch.object(korch, "PILOT_ASSETS", ("demo-skill", "j-demo")):
            self.assertEqual(korch.pilot(cli=cli), 0)
        models = {c[c.index("--model") + 1] for c in cli.calls}
        self.assertEqual(models, set(korch.MODELS))
        self.assertEqual({c[c.index("--runs") + 1] for c in cli.calls}, {"1"})
        self.assertTrue(
            all(float(c[c.index("--max-cost-usd") + 1]) <= korch.PILOT_CAP_USD for c in cli.calls)
        )
        self.assertTrue(self.calls_for(cli, "vanilla"))
        self.assertFalse(
            any(
                c[c.index("--eval-dir") + 1] == "evals-triggers"
                for c in self.calls_for(cli, "vanilla")
            )
        )

    def test_smoke_caps_follow_the_smoke_budget(self):
        cli = FakeCli(lambda args: (result_doc({case_arg(args): [None]}, cost=15.0), 0, ""))
        korch.smoke(["demo-skill", "j-demo"], cli=cli)
        caps = [float(c[c.index("--max-cost-usd") + 1]) for c in cli.calls]
        self.assertEqual(caps[0], 40.0)
        self.assertTrue(all(cap <= 40.0 for cap in caps))
        self.assertLess(caps[-1], caps[0])

    def test_spend_snapshots_the_period(self):
        from evals.kbeval import ledger as led

        led.append({"period": "p", "purpose": "pilot", "model": "m", "costUsd": 2.5})
        out = self.tmp / "spend.json"
        self.assertEqual(korch.spend(out), 2.5)
        self.assertEqual(json.loads(out.read_text())["totals"]["totalUsd"], 2.5)


class EntryPointTests(OrchestrateTests):
    RUN_ARGS: ClassVar[list] = [
        "run",
        "--suite",
        "effects",
        "--staging",
        "kb",
        "--plugin",
        "/p",
        "--model",
        "claude-opus-5-5",
        "--purpose",
        "fanout-x",
    ]

    def test_run_cap_is_the_purpose_remainder(self):
        kledger.append({"period": self.budget["period"], "purpose": "fanout-x", "costUsd": 2.0})
        kledger.append({"period": self.budget["period"], "purpose": "other", "costUsd": 1.0})
        with mock.patch.object(kmain.runner, "run", return_value=0) as run:
            self.assertEqual(kmain.main([*self.RUN_ARGS, "--cap-usd", "5"]), 0)
        self.assertEqual(run.call_args.kwargs["cap_limit"], 3.0)

    def test_run_with_the_cap_spent_exits_two_without_calling_the_runner(self):
        for spent in (5.0, 6.0):
            purpose = f"fanout-s{int(spent)}"
            args = [*self.RUN_ARGS[:-1], purpose]
            with self.subTest(spent=spent):
                kledger.append(
                    {"period": self.budget["period"], "purpose": purpose, "costUsd": spent}
                )
                err = io.StringIO()
                with (
                    mock.patch.object(kmain.runner, "run", return_value=0) as run,
                    contextlib.redirect_stderr(err),
                ):
                    self.assertEqual(kmain.main([*args, "--cap-usd", "5"]), 2)
                run.assert_not_called()
                self.assertIn(
                    f"{purpose}: purpose cap reached (cap $5.00, spent ${spent:.2f}); "
                    "no step started. Raise the cap to continue.",
                    err.getvalue(),
                )

    def test_run_cap_must_be_a_positive_finite_amount(self):
        for value in ("nan", "inf", "0", "-3"):
            with (
                self.subTest(value=value),
                mock.patch.object(kmain.runner, "run", return_value=0) as run,
            ):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    kmain.main([*self.RUN_ARGS, "--cap-usd", value])
                run.assert_not_called()

    def test_bad_cap_values_say_what_is_accepted(self):
        for value in ("5o", "$5", "nan", "0"):
            with (
                self.subTest(value=value),
                mock.patch.object(kmain.runner, "run", return_value=0) as run,
            ):
                err = io.StringIO()
                with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                    kmain.main([*self.RUN_ARGS, "--cap-usd", value])
                run.assert_not_called()
                self.assertIn(
                    f"expected a positive number of USD such as 25 or 12.50, not {value}",
                    err.getvalue(),
                )

    BASELINE_ARGS: ClassVar[list] = ["baseline", "--model", "claude-sonnet-5-5"]

    def test_baseline_passes_the_cap_to_the_orchestrator(self):
        with mock.patch.object(kmain.orchestrate, "baseline", return_value=0) as baseline:
            self.assertEqual(kmain.main([*self.BASELINE_ARGS, "--cap-usd", "5"]), 0)
        self.assertEqual(baseline.call_args.kwargs["cap_usd"], 5.0)

    def test_baseline_rejects_a_bad_cap_before_running(self):
        for value in ("0", "nan"):
            with (
                self.subTest(value=value),
                mock.patch.object(kmain.orchestrate, "baseline", return_value=0) as baseline,
            ):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    kmain.main([*self.BASELINE_ARGS, "--cap-usd", value])
                baseline.assert_not_called()

    def test_run_without_cap_passes_no_limit(self):
        with mock.patch.object(kmain.runner, "run", return_value=0) as run:
            self.assertEqual(kmain.main(self.RUN_ARGS), 0)
        self.assertIsNone(run.call_args.kwargs.get("cap_limit"))

    def test_headroom_prints_the_tail_and_exits_zero(self):
        from evals.kbeval import __main__ as kmain

        root = self.repo / "evals" / "baselines"
        for suite in kheadroom.SUITES:
            path = root / OPUS / f"{suite}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary_doc({"demo-skill--effect-a": (1.0, True, 1.0)})))
        out = io.StringIO()
        with mock.patch.object(kmain, "REPO", self.repo), contextlib.redirect_stdout(out):
            code = kmain.main(["headroom"])
        self.assertEqual(code, 0)
        lines = out.getvalue().splitlines()
        self.assertIn(
            "1 cases: 1 saturated, 0 dead, 0 kb-below-vanilla, 0 discriminating, 0 unmeasured",
            lines,
        )

    def test_headroom_exits_one_on_an_unreadable_baseline(self):
        from evals.kbeval import __main__ as kmain

        path = self.repo / "evals" / "baselines" / OPUS / "effects.json"
        path.parent.mkdir(parents=True)
        path.write_text("{")
        err = io.StringIO()
        with mock.patch.object(kmain, "REPO", self.repo), contextlib.redirect_stderr(err):
            self.assertEqual(kmain.main(["headroom"]), 1)

    def test_headroom_lists_live_cases_missing_from_the_baselines(self):
        root = self.repo / "evals" / "baselines"
        for suite in kheadroom.SUITES:
            path = root / OPUS / f"{suite}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(summary_doc({"demo-skill--effect-limiter": (1.0, True, 1.0)}))
            )
        out = io.StringIO()
        with mock.patch.object(kmain, "REPO", self.repo), contextlib.redirect_stdout(out):
            self.assertEqual(kmain.main(["headroom"]), 0)
        lines = out.getvalue().splitlines()
        self.assertIn("MISSING demo-agent--effect-review", lines)
        self.assertIn("MISSING wf--demo-chain", lines)
        self.assertNotIn("MISSING demo-skill--effect-limiter", lines)

    def test_headroom_invalid_json_exits_one_naming_the_path(self):
        path = self.repo / "evals" / "baselines" / OPUS / "effects.json"
        path.parent.mkdir(parents=True)
        path.write_text("{")
        err = io.StringIO()
        with mock.patch.object(kmain, "REPO", self.repo), contextlib.redirect_stderr(err):
            self.assertEqual(kmain.main(["headroom"]), 1)
        self.assertTrue(err.getvalue().startswith(f"headroom: {path}: "), err.getvalue())

    def test_real_git_changes_are_parsed(self):
        git = shutil.which("git")
        if git is None:
            self.fail("git is not installed")
        work = self.tmp / "gitwork"
        work.mkdir()

        def run(*args):
            subprocess.run(
                [
                    git,
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@example.com",
                    "-c",
                    "commit.gpgsign=false",
                    *args,
                ],
                cwd=work,
                check=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
            )

        run("init", "-q", "-b", "main")
        (work / "a b.txt").write_text("one\n")
        (work / "old.txt").write_text("keep\n")
        run("add", "-A")
        run("commit", "-q", "-m", "base")
        (work / "a b.txt").write_text("two\n")
        (work / "newdir").mkdir()
        (work / "newdir" / "n.txt").write_text("new\n")
        run("mv", "old.txt", "new.txt")
        with mock.patch.object(kmain, "REPO", work):
            base = kmain._git(["merge-base", "--end-of-options", "HEAD", "HEAD"]).strip()
            diff = kmain._git(["diff", "--name-status", "--no-renames", "-z", base])
            status = kmain._git(["status", "--porcelain", "-z", "--untracked-files=all"])
        self.assertEqual(
            kmain._parse_changes(diff, status),
            [("M", "a b.txt"), ("A", "new.txt"), ("D", "old.txt"), ("A", "newdir/n.txt")],
        )

    def test_affected_exits_two_on_everything(self):
        def fake_git(args, **kwargs):
            out = {"merge-base": "abc\n", "diff": "M\0.claude/CLAUDE.md\0", "status": ""}[args[1]]
            return subprocess.CompletedProcess(args, 0, out, "")

        out, err = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(kmain, "REPO", self.repo),
            mock.patch.object(kmain.subprocess, "run", fake_git),
        ):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = kmain.main(["affected"])
        self.assertEqual(code, 2)
        self.assertIn("ALL <- .claude/CLAUDE.md; a smoke cannot cover it", err.getvalue())

    def run_affected(self, diff="", status="", fail=False):
        def fake_git(args, **kwargs):
            if fail:
                return subprocess.CompletedProcess(args, 128, "", "fatal: no merge base")
            out = {"merge-base": "abc\n", "diff": diff, "status": status, "rev-parse": "feat/x\n"}[
                args[1]
            ]
            return subprocess.CompletedProcess(args, 0, out, "")

        out, err = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(kmain, "REPO", self.repo),
            mock.patch.object(kmain.subprocess, "run", fake_git),
        ):
            with mock.patch.object(kmain.orchestrate, "_capped", lambda purpose, cap: cap):
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = kmain.main(["affected"])
        return code, out.getvalue(), err.getvalue()

    def test_affected_passes_a_dash_base_to_git_as_a_ref(self):
        calls = []

        def fake_git(args, **kwargs):
            calls.append(args)
            out = {"merge-base": "abc\n", "diff": "", "status": "", "rev-parse": "feat/x\n"}[
                args[1]
            ]
            return subprocess.CompletedProcess(args, 0, out, "")

        with (
            mock.patch.object(kmain, "REPO", self.repo),
            mock.patch.object(kmain.subprocess, "run", fake_git),
        ):
            with mock.patch.object(kmain.orchestrate, "_capped", lambda purpose, cap: cap):
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    kmain.main(["affected", "--base=-x"])
        self.assertEqual(calls[0], ["git", "merge-base", "--end-of-options", "-x", "HEAD"])

    def test_affected_with_no_changes_prints_nothing_to_run(self):
        code, out, err = self.run_affected()
        self.assertEqual(code, 0)
        self.assertEqual(out, "")
        lines = err.splitlines()
        self.assertEqual(lines[:-1], ["nothing to run"])
        self.assertTrue(
            lines[-1].startswith(
                "ESTIMATE 0 kb runs + 0 vanilla runs, about $0.00; smoke cap left $"
            ),
            err,
        )

    def test_affected_prints_sorted_names_on_one_line_and_notes_before_the_estimate(self):
        diff = (
            "M\0.claude/skills/demo-skill/SKILL.md\0M\0evals/kbeval/results.py\0M\0.codex/AGENTS.md\0"
            "M\0.claude/commands/j-demo.md\0"
        )
        code, out, err = self.run_affected(diff=diff)
        self.assertEqual(code, 0)
        self.assertEqual(out, "demo-skill j-demo wf--demo-chain\n")
        lines = err.splitlines()
        estimate = [i for i, line in enumerate(lines) if line.startswith("ESTIMATE ")]
        self.assertEqual(estimate, [len(lines) - 1])
        self.assertEqual(
            lines[:-1],
            [
                "HARNESS evals/kbeval/results.py: covered by make test",
                "UNMEASURED .codex/AGENTS.md",
            ],
        )

    def test_affected_prints_unpriced_names_before_the_estimate(self):
        (self.repo / ".claude/commands/j-new.md").write_text("---\nname: j-new\n---\n\nNew.\n")
        code, out, err = self.run_affected(diff="M\0.claude/commands/j-new.md\0")
        self.assertEqual((code, out), (0, "j-new\n"))
        lines = err.splitlines()
        self.assertEqual(lines[:-1], ["UNPRICED j-new: no case"])
        self.assertTrue(lines[-1].startswith("ESTIMATE 0 kb runs"), err)

    def test_affected_exits_one_on_a_malformed_budget(self):
        (self.repo / "evals" / "budget.json").write_text("{")
        code, out, err = self.run_affected(diff="M\0.claude/skills/demo-skill/SKILL.md\0")
        self.assertEqual((code, out), (1, ""))
        self.assertTrue(err.splitlines()[-1].startswith("affected: "), err)

    def test_affected_exits_one_on_an_unreadable_ledger(self):
        def bad_ledger(purpose, cap):
            raise kmain.kledger.LedgerError("ledger.jsonl:1: unreadable ledger line")

        def fake_git(args, **kwargs):
            out = {"merge-base": "abc\n", "diff": "", "status": "", "rev-parse": "feat/x\n"}[
                args[1]
            ]
            return subprocess.CompletedProcess(args, 0, out, "")

        out, err = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(kmain, "REPO", self.repo),
            mock.patch.object(kmain.subprocess, "run", fake_git),
        ):
            with mock.patch.object(kmain.orchestrate, "_capped", bad_ledger):
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = kmain.main(["affected"])
        self.assertEqual((code, out.getvalue()), (1, ""))
        self.assertIn("affected: ledger.jsonl:1: unreadable ledger line\n", err.getvalue())

    def test_affected_exits_one_with_nothing_on_stdout_when_git_fails(self):
        code, out, err = self.run_affected(fail=True)
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertTrue(err.startswith("affected: git "), err)

    def test_parse_changes_returns_unquoted_paths_with_their_status(self):
        diff = "M\0docs/a b.md\0D\0.claude/agents/old.md\0"
        status = " M tracked.md\0?? caf\u00e9/\u00e9.md\0A  staged.md\0?? dir/new file.md\0"
        self.assertEqual(
            kmain._parse_changes(diff, status),
            [
                ("M", "docs/a b.md"),
                ("D", ".claude/agents/old.md"),
                ("A", "caf\u00e9/\u00e9.md"),
                ("A", "dir/new file.md"),
            ],
        )

    def test_parse_changes_skips_the_source_path_of_a_staged_rename(self):
        self.assertEqual(kmain._parse_changes("", "R  new.md\0old.md\0?? x.md\0"), [("A", "x.md")])

    def test_summarize_writes_nothing_when_a_check_fails(self):
        from evals.kbeval import __main__ as kmain

        err = io.StringIO()
        with mock.patch.object(kmain, "REPO", self.repo), contextlib.redirect_stderr(err):
            code = kmain.main(["summarize", "--model", OPUS])
        self.assertEqual(code, 1)
        self.assertIn("no summary written", err.getvalue())
        self.assertFalse((self.repo / "evals" / "baselines").exists())

    def test_summarize_passes_the_live_case_set(self):
        seen = []

        def fake(results, model, suite, settings, live=None, notes=None):
            seen.append(live)
            return {"suite": suite, "model": model, "cases": [], "runs": settings}

        with (
            mock.patch.object(kmain, "REPO", self.repo),
            mock.patch.object(kmain.ksummary, "summarize", fake),
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                kmain.main(["summarize", "--model", OPUS])
        live = {c.name for c in kmain.kcases.discover(self.repo / "evals" / "cases")}
        self.assertTrue(live)
        self.assertTrue(seen)
        self.assertTrue(all(item == live for item in seen))

    def test_summarize_prints_notes_without_changing_the_exit_code(self):
        def summarize_with(note):
            def fake(results, model, suite, settings, live=None, notes=None):
                if note and notes is not None:
                    notes.append(f"kb/{model}/{suite}: {note}")
                row = {"name": "j-arch--effect-x", "kbRuns": 3, "vanillaRuns": 3, "errors": []}
                return {"suite": suite, "model": model, "cases": [row], "runs": settings}

            err = io.StringIO()
            with (
                mock.patch.object(kmain, "REPO", self.repo),
                mock.patch.object(kmain.ksummary, "summarize", fake),
            ):
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    code = kmain.main(["summarize", "--model", OPUS])
            return code, err.getvalue()

        quiet_code, _ = summarize_with(None)
        code, err = summarize_with("skipped x.json (no .key)")
        self.assertEqual((quiet_code, code), (0, 0))
        self.assertIn(f"kb/{OPUS}/effects: skipped x.json (no .key)", err)

    def test_spend_exits_one_over_the_ceiling(self):
        from evals.kbeval import __main__ as kmain
        from evals.kbeval import ledger as led

        led.append({"period": "p", "costUsd": 150.0})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(kmain.main(["spend"]), 1)
        self.assertIn("$150.00 of $100.00", out.getvalue())

    def test_jobs_must_be_positive(self):
        from evals.kbeval import __main__ as kmain

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            kmain.main(["pilot", "--jobs", "0"])


class RealFixtureTests(unittest.TestCase):
    def test_real_seed_scripts_run_in_a_minimal_env(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        plugin = kstage.stage(REPO, root / "stage", "kb")
        seeds = sorted((plugin / "evals-effects").glob("*/seed.sh"))
        self.assertTrue(seeds, "no staged effect cases")
        for seed in seeds:
            with self.subTest(case=seed.parent.name):
                work = root / "work" / seed.parent.name
                work.mkdir(parents=True)
                env = {"PATH": os.environ["PATH"], "HOME": str(root), "TMPDIR": str(root)}
                result = subprocess.run(
                    ["bash", str(seed)],
                    cwd=work,
                    env=env,
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((work / ".kb-references").is_dir())

    def test_py_calc_planned_is_py_calc_plus_a_plan(self):
        fixtures = REPO / "evals" / "fixtures"

        def files(root: Path) -> dict:
            return {
                p.relative_to(root): p.read_bytes()
                for p in root.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            }

        planned = files(fixtures / "py-calc-planned")
        plan = planned.pop(Path("plans") / "plan.md")
        self.assertEqual(planned, files(fixtures / "py-calc"))
        self.assertIn(b"mean([])", plan)

    def test_py_calc_keeps_its_seeded_bug(self):
        fixture = REPO / "evals" / "fixtures" / "py-calc"
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        bug = subprocess.run(
            [sys.executable, "-c", "from calc.ops import mean; mean([])"],
            cwd=fixture,
            env=env,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=60,
        )
        self.assertIn("ZeroDivisionError", bug.stderr)
        self.assertIn("0.0", (fixture / "calc" / "ops.py").read_text())
        suite = subprocess.run(
            [sys.executable, "-m", "unittest"],
            cwd=fixture,
            env=env,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=60,
        )
        self.assertEqual(suite.returncode, 0, suite.stderr)

    def test_py_calc_red_fails_only_on_the_empty_mean(self):
        fixture = REPO / "evals" / "fixtures" / "py-calc-red"
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        suite = subprocess.run(
            [sys.executable, "-m", "unittest"],
            cwd=fixture,
            env=env,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=60,
        )
        self.assertNotEqual(suite.returncode, 0)
        self.assertIn("test_mean_of_empty_list_is_zero", suite.stderr)
        self.assertIn("Ran 3 tests", suite.stderr)

    def test_docs_drift_rejects_the_documented_flag(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        work = Path(tmp.name)
        shutil.copytree(REPO / "evals" / "fixtures" / "docs-drift", work, dirs_exist_ok=True)
        (work / "data.csv").write_text("a,b\n1,2\n")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        old = subprocess.run(
            [sys.executable, "tool.py", "--input", "data.csv", "--out", "x.jsonl"],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=60,
        )
        self.assertEqual(old.returncode, 2)
        self.assertIn("unrecognized arguments: --out", old.stderr)
        self.assertIn("--out", (work / "README.md").read_text())


from evals.kbeval import __main__ as kmain  # noqa: E402
from evals.kbeval import determination as kdet  # noqa: E402


class RunKeepTempTests(unittest.TestCase):
    """`run --keep-temp` is forwarded to runner.run as `keep_temp`."""

    def _keep_temp(self, extra):
        recorded = {}

        def fake_run(*args, **kwargs):
            recorded.update(kwargs)
            return 0

        argv = [
            "run",
            "--suite",
            "triggers",
            "--staging",
            "kb",
            "--plugin",
            "/tmp/p",
            "--model",
            "m",
            "--purpose",
            "p",
            *extra,
        ]
        with mock.patch.object(kmain.runner, "run", fake_run):
            self.assertEqual(kmain.main(argv), 0)
        return recorded["keep_temp"]

    def test_run_command_passes_keep_temp(self):
        self.assertIs(self._keep_temp(["--keep-temp"]), True)
        self.assertIs(self._keep_temp([]), False)


class TriggerReportTests(unittest.TestCase):
    """`determination triggers` counts fired runs per trigger case."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        patcher = mock.patch.object(kdet, "RESULTS", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write(
        self, model, stem, cases, partial=False, key=True, started=None, grader_name="fired"
    ):
        root = self.tmp / "p" / "kb" / model / "triggers"
        root.mkdir(parents=True, exist_ok=True)
        # Like the real trigger cases, a near or noauto case's load grader carries `max: 0`.
        doc = {
            "partial": partial,
            "cases": [
                {
                    "name": name,
                    "graders": [
                        {
                            "name": grader_name,
                            "type": "tool_used",
                            "config": {
                                "tool": "Skill",
                                "input_match": f'"skill"\\s*:\\s*"(?:kb:)?{name.split("--")[0]}"',
                                **({} if "--trigger-should-" in name else {"max": 0}),
                            },
                        }
                    ],
                    "arms": {
                        "with": [
                            {"graders": [{"name": grader_name, "passed": ok}]} for ok in outcomes
                        ]
                    },
                }
                for name, outcomes in cases.items()
            ],
        }
        if started is not None:
            doc["startedAt"] = started
        (root / f"{stem}.json").write_text(json.dumps(doc), encoding="utf-8")
        if key:
            (root / f"{stem}.key").write_text("k", encoding="utf-8")

    def test_trigger_rows_counts_fired_runs(self):
        self._write("m", "a", {"a--trigger-should-x": [True, False, True]})
        self.assertEqual(
            kdet.trigger_rows("p", ["a"]), [("a--trigger-should-x", "trigger-should", "m", 2, 3)]
        )

    def test_trigger_rows_inverts_near_cases(self):
        self._write("m", "a", {"a--trigger-near-x": [True, True, True]})
        self.assertEqual(
            kdet.trigger_rows("p", ["a"]), [("a--trigger-near-x", "trigger-near", "m", 0, 3)]
        )

    def test_trigger_rows_filters_by_split(self):
        self._write(
            "m",
            "a",
            {
                "a--trigger-should-held-x": [True],
                "a--trigger-should-tune-y": [True],
                "a--trigger-should-z": [False],
            },
        )
        held = [row[0] for row in kdet.trigger_rows("p", ["a"], split="held")]
        tune = [row[0] for row in kdet.trigger_rows("p", ["a"], split="tune")]
        self.assertEqual(held, ["a--trigger-should-held-x"])
        self.assertEqual(tune, ["a--trigger-should-tune-y", "a--trigger-should-z"])

    def test_trigger_rows_reads_exact_case_files(self):
        self._write("m", "a--trigger-should-x", {"a--trigger-should-x": [True, True]})
        self.assertEqual(
            kdet.trigger_rows("p", ["a"]), [("a--trigger-should-x", "trigger-should", "m", 2, 2)]
        )

    def test_trigger_rows_inverts_noauto_and_mixed_near_cases(self):
        self._write(
            "m",
            "a",
            {
                "a--trigger-noauto-x": [True, False],
                "a--trigger-near-y": [True, False, False],
            },
        )
        self.assertEqual(
            kdet.trigger_rows("p", ["a"]),
            [
                ("a--trigger-near-y", "trigger-near", "m", 2, 3),
                ("a--trigger-noauto-x", "trigger-noauto", "m", 1, 2),
            ],
        )

    def test_trigger_rows_skips_partial_and_keyless_batches(self):
        self._write("m", "a", {"a--trigger-should-x": [True]}, partial=True)
        self._write("m", "a--trigger-should-y", {"a--trigger-should-y": [True]}, key=False)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(kdet.trigger_rows("p", ["a"]), [])
        self.assertIn("(partial)", err.getvalue())
        self.assertIn("(no .key)", err.getvalue())

    def test_trigger_rows_newest_startedat_wins(self):
        self._write(
            "m",
            "a--trigger-should-x",
            {"a--trigger-should-x": [False, False]},
            started="2026-11-16T10:00:00.000Z",
        )
        self._write(
            "m", "a@2", {"a--trigger-should-x": [True, True]}, started="2026-11-15T10:00:00.000Z"
        )
        self.assertEqual(
            kdet.trigger_rows("p", ["a"]), [("a--trigger-should-x", "trigger-should", "m", 0, 2)]
        )

    def test_trigger_rows_use_the_case_own_skill_grader(self):
        self._write("m", "a", {"a--trigger-should-x": [True, True, True]}, grader_name="skill")
        self.assertEqual(
            kdet.trigger_rows("p", ["a"]), [("a--trigger-should-x", "trigger-should", "m", 3, 3)]
        )

    def test_trigger_rows_keeps_assets_apart_and_drops_non_trigger_cases(self):
        self._write(
            "m",
            "a",
            {
                "a--trigger-should-x": [True],
                "a-b--trigger-should-y": [True],
                "a--effect-z": [True],
            },
        )
        self._write("m", "a-b--trigger-should-w", {"a-b--trigger-should-w": [True]})
        self.assertEqual([row[0] for row in kdet.trigger_rows("p", ["a"])], ["a--trigger-should-x"])

    def test_trigger_rows_ignores_runs_without_the_fired_grader(self):
        root = self.tmp / "p" / "kb" / "m" / "triggers"
        root.mkdir(parents=True)
        doc = {
            "partial": False,
            "cases": [
                {
                    "name": "a--trigger-near-x",
                    "graders": [
                        {
                            "name": "fired",
                            "type": "tool_used",
                            "config": {
                                "tool": "Skill",
                                "input_match": '"skill"\\s*:\\s*"a"',
                                "max": 0,
                            },
                        },
                    ],
                    "arms": {
                        "with": [
                            {"graders": [{"name": "fired", "passed": True}]},
                            {"graders": [{"name": "other", "passed": False}]},
                        ]
                    },
                }
            ],
        }
        (root / "a.json").write_text(json.dumps(doc), encoding="utf-8")
        (root / "a.key").write_text("k", encoding="utf-8")
        self.assertEqual(
            kdet.trigger_rows("p", ["a"]), [("a--trigger-near-x", "trigger-near", "m", 0, 1)]
        )

    def test_trigger_rows_reports_a_case_with_no_load_grader(self):
        root = self.tmp / "p" / "kb" / "m" / "triggers"
        root.mkdir(parents=True)
        doc = {
            "partial": False,
            "cases": [
                {
                    "name": "a--trigger-should-x",
                    "graders": [
                        {
                            "name": "fired",
                            "type": "tool_used",
                            "config": {"tool": "Skill", "input_match": '"skill"\\s*:\\s*"b"'},
                        },
                    ],
                    "arms": {"with": [{"graders": [{"name": "fired", "passed": True}]}]},
                }
            ],
        }
        (root / "a.json").write_text(json.dumps(doc), encoding="utf-8")
        (root / "a.key").write_text("k", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            rows = kdet.trigger_rows("p", ["a"])
        self.assertEqual(rows, [("a--trigger-should-x", "trigger-should", "m", 0, 0)])
        self.assertIn("NO LOAD GRADER a--trigger-should-x\n", err.getvalue())

    def test_figures_counts_an_agent_tool_load_grader(self):
        root = self.tmp / "baseline" / "kb" / OPUS / "triggers"
        root.mkdir(parents=True)
        doc = {
            "partial": False,
            "cases": [
                {
                    "name": "security-reviewer--trigger-should-x",
                    "graders": [
                        {
                            "name": "dispatched",
                            "type": "tool_used",
                            "config": {
                                "tool": "Agent",
                                "input_match": '"subagent_type"\\s*:\\s*"kb:security-reviewer"',
                            },
                        },
                    ],
                    "arms": {"with": [{"graders": [{"name": "dispatched", "passed": False}]}] * 3},
                }
            ],
        }
        (root / "security-reviewer.json").write_text(json.dumps(doc), encoding="utf-8")
        (root / "security-reviewer.key").write_text("k", encoding="utf-8")
        with mock.patch.object(kdet, "skill_assets", lambda: ["security-reviewer"]):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(kdet.figures(), 0)
        self.assertIn("\n  security-reviewer: 0/3\n", out.getvalue())

    def test_triggers_exits_one_when_nothing_matched(self):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as err,
        ):
            self.assertEqual(kdet.triggers("missing", ["a"], None), 1)
        self.assertIn("no trigger results", err.getvalue())


from evals.kbeval import hillclimb as khill  # noqa: E402


class HillclimbAdoptTests(unittest.TestCase):
    """`hillclimb adopt` computes the held-out adoption criteria from stored results."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        patcher = mock.patch.object(kdet, "RESULTS", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _put(self, purpose, model, suite, stem, cases):
        root = self.tmp / purpose / "kb" / model / suite
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{stem}.json").write_text(
            json.dumps({"partial": False, "cases": cases}), encoding="utf-8"
        )
        (root / f"{stem}.key").write_text("k", encoding="utf-8")

    def _trigger(self, purpose, model, name, fired, graded):
        """Store a trigger case with `fired` of `graded` runs loading the skill."""
        near = "--trigger-should-" not in name
        self._put(
            purpose,
            model,
            "triggers",
            name,
            [
                {
                    "name": name,
                    "graders": [
                        {
                            "name": "skill",
                            "type": "tool_used",
                            "config": {
                                "tool": "Skill",
                                "input_match": f'"skill"\\s*:\\s*"(?:kb:)?{name.split("--")[0]}"',
                                **({"max": 0} if near else {}),
                            },
                        }
                    ],
                    "arms": {
                        "with": [
                            {"graders": [{"name": "skill", "passed": (i < fired) != near}]}
                            for i in range(graded)
                        ]
                    },
                }
            ],
        )

    def _effects(self, purpose, model, passes, asset="a"):
        self._put(
            purpose,
            model,
            "effects",
            asset,
            [
                {
                    "name": f"{asset}--effect-x",
                    "graders": [{"name": "design", "type": "llm"}],
                    "arms": {
                        "with": [{"graders": [{"name": "design", "passed": ok}]} for ok in passes]
                    },
                }
            ],
        )

    def _asset(
        self,
        held=(0, 4),
        existing=(3, 3),
        near=0,
        held_graded=(6, 6),
        effects=True,
        held_before=(6, 6),
    ):
        """Store asset `a` before and after: fired counts across both models, split evenly."""
        for model, part in ((OPUS, 0), (kdet.SONNET, 1)):
            for purpose, fired in (("hb", held[0]), ("hf", held[1])):
                share = fired // 2 + (fired % 2 if part == 0 else 0)
                graded = held_graded[part] if purpose == "hf" else held_before[part]
                self._trigger(purpose, model, "a--trigger-should-held-h", share, graded)
            for purpose, fired in (("b", existing[0]), ("f", existing[1])):
                share = fired // 2 + (fired % 2 if part == 0 else 0)
                self._trigger(purpose, model, "a--trigger-should-x", share, 3)
            self._trigger("b", model, "a--trigger-near-y", 0, 3)
            self._trigger("f", model, "a--trigger-near-y", near if part == 0 else 0, 3)
            self._trigger("f", model, "a--trigger-near-held-z", 0, 3)
            if effects:
                self._effects("baseline", model, [True] * 3)
                self._effects("e", model, [True] * 3)

    def _adopt(self, before=("b", "hb"), after=("f", "hf")):
        with (
            contextlib.redirect_stdout(io.StringIO()) as out,
            contextlib.redirect_stderr(io.StringIO()) as err,
        ):
            code = khill.adopt(["a"], list(before), list(after), "e")
        self.err = err.getvalue()
        return code, out.getvalue()

    def test_held_gain_of_four_adopts(self):
        self._asset(held=(0, 4))
        code, out = self._adopt()
        self.assertEqual(code, 0)
        self.assertIn("CRITERION a held-gain PASS", out)
        self.assertIn("CRITERION a effects PASS", out)
        self.assertTrue(out.endswith("ADOPT a adopt\n"), out)

    def test_held_gain_of_three_restores(self):
        self._asset(held=(1, 4))
        code, out = self._adopt()
        self.assertEqual(code, 0)
        self.assertIn("CRITERION a held-gain FAIL", out)
        self.assertIn("ADOPT a restore\n", out)

    def test_near_fire_restores_despite_a_large_gain(self):
        self._asset(held=(0, 12), near=1)
        code, out = self._adopt()
        self.assertEqual(code, 0)
        self.assertIn("CRITERION a held-gain PASS", out)
        self.assertIn("CRITERION a near-fires FAIL", out)
        self.assertIn("ADOPT a restore\n", out)

    def test_existing_should_drop_restores(self):
        self._asset(held=(0, 6), existing=(3, 2))
        code, out = self._adopt()
        self.assertEqual(code, 0)
        self.assertIn("CRITERION a existing-should FAIL", out)
        self.assertIn("ADOPT a restore\n", out)

    def test_short_graded_runs_block(self):
        self._asset(held=(0, 6), held_graded=(6, 5))
        code, out = self._adopt()
        self.assertEqual(code, 1)
        self.assertIn("CRITERION a held-gain BLOCKED", out)
        self.assertIn("ADOPT a blocked\n", out)

    def test_effects_are_skipped_once_a_trigger_criterion_fails(self):
        self._asset(held=(1, 4), effects=False)
        code, out = self._adopt()
        self.assertEqual(code, 0)
        self.assertIn("CRITERION a effects SKIPPED", out)
        self.assertIn("ADOPT a restore\n", out)

    def test_missing_effects_results_block_an_otherwise_passing_asset(self):
        self._asset(held=(0, 4), effects=False)
        code, out = self._adopt()
        self.assertEqual(code, 1)
        self.assertIn("CRITERION a effects BLOCKED", out)
        self.assertIn("ADOPT a blocked\n", out)

    def test_effects_regression_restores(self):
        self._asset(held=(0, 4))
        self._effects("e", OPUS, [True, False, False])
        code, out = self._adopt()
        self.assertEqual(code, 0)
        self.assertIn("CRITERION a effects FAIL", out)
        self.assertIn("a--effect-x", out)
        self.assertIn("ADOPT a restore\n", out)

    def test_neighbour_drop_prints_review_and_changes_no_verdict(self):
        self._asset(held=(0, 4))
        self._trigger("baseline", kdet.SONNET, "n--trigger-should-x", 2, 3)
        self._trigger("f", kdet.SONNET, "n--trigger-should-x", 0, 3)
        self._trigger("baseline", kdet.SONNET, "n--trigger-should-w", 2, 3)
        self._trigger("f", kdet.SONNET, "n--trigger-should-w", 1, 3)
        self._trigger("f", kdet.SONNET, "n--trigger-near-y", 1, 3)
        self._trigger("baseline", OPUS, "n--trigger-should-z", 3, 3)
        self._trigger("f", OPUS, "n--trigger-should-z", 0, 3)
        self._trigger("f", OPUS, "n--trigger-near-v", 3, 3)
        code, out = self._adopt()
        self.assertEqual(code, 0)
        self.assertIn("ADOPT a adopt\n", out)
        lines = out.splitlines()
        neighbour = lines[lines.index("ADOPT a adopt") + 1 :]
        self.assertEqual(
            [" ".join(line.split()[:3]) for line in neighbour],
            [
                "CRITERION neighbours n--trigger-near-y",
                "REVIEW n--trigger-should-x should",
            ],
        )
        self.assertTrue(neighbour[0].startswith("CRITERION neighbours n--trigger-near-y FAIL "))

    def test_missing_effects_baseline_blocks(self):
        self._asset(held=(0, 4), effects=False)
        for model in (OPUS, kdet.SONNET):
            self._effects("e", model, [True] * 3)
        code, out = self._adopt()
        self.assertEqual(code, 1)
        self.assertIn(
            f"CRITERION a effects BLOCKED no baseline results for {OPUS} {kdet.SONNET}\n", out
        )
        self.assertIn("ADOPT a blocked\n", out)

    def test_effects_missing_on_one_model_block(self):
        self._asset(held=(0, 4), effects=False)
        for model in (OPUS, kdet.SONNET):
            self._effects("baseline", model, [True] * 3)
        self._effects("e", OPUS, [True] * 3)
        code, out = self._adopt()
        self.assertEqual(code, 1)
        self.assertIn(f"CRITERION a effects BLOCKED no e results for {kdet.SONNET}\n", out)
        self.assertIn("ADOPT a blocked\n", out)

    def test_purpose_listed_twice_in_after_is_refused(self):
        self._asset(held=(0, 4))
        code, out = self._adopt(after=("f", "hf", "hf"))
        self.assertEqual(code, 1)
        self.assertIn(f"BLOCKED duplicate a--trigger-should-held-h {OPUS} in hf, hf\n", self.err)
        self.assertNotIn("ADOPT", out)

    def test_purpose_listed_twice_in_before_is_refused(self):
        self._asset(held=(0, 4))
        code, out = self._adopt(before=("b", "hb", "b"))
        self.assertEqual(code, 1)
        self.assertIn(f"BLOCKED duplicate a--trigger-should-x {kdet.SONNET} in b, b\n", self.err)
        self.assertNotIn("ADOPT", out)

    def test_neighbour_case_in_two_after_purposes_is_refused(self):
        self._asset(held=(0, 4))
        for purpose in ("f", "f2"):
            self._trigger(purpose, kdet.SONNET, "n--trigger-near-y", 0, 3)
        code, out = self._adopt(after=("f", "hf", "f2"))
        self.assertEqual(code, 1)
        self.assertIn(f"BLOCKED duplicate n--trigger-near-y {kdet.SONNET} in f, f2\n", self.err)
        self.assertEqual(out, "")

    def test_two_held_out_should_cases_block(self):
        self._asset(held=(0, 4))
        for model in (OPUS, kdet.SONNET):
            self._trigger("hb", model, "a--trigger-should-held-h", 0, 3)
            self._trigger("hb", model, "a--trigger-should-held-k", 0, 3)
            self._trigger("hf", model, "a--trigger-should-held-h", 2, 3)
            self._trigger("hf", model, "a--trigger-should-held-k", 0, 3)
        code, out = self._adopt()
        self.assertEqual(code, 1)
        self.assertIn("CRITERION a held-gain BLOCKED", out)
        self.assertIn("2 held-out should cases before", out)
        self.assertIn("ADOPT a blocked\n", out)

    def test_short_graded_runs_before_block(self):
        self._asset(held=(0, 6), held_before=(6, 5))
        code, out = self._adopt()
        self.assertEqual(code, 1)
        self.assertIn("CRITERION a held-gain BLOCKED", out)
        self.assertIn("11 graded before", out)
        self.assertIn("ADOPT a blocked\n", out)

    def test_held_out_near_fire_alone_restores(self):
        self._asset(held=(0, 4))
        self._trigger("f", OPUS, "a--trigger-near-held-z", 1, 3)
        code, out = self._adopt()
        self.assertEqual(code, 0)
        self.assertIn("CRITERION a near-fires FAIL", out)
        self.assertIn("ADOPT a restore\n", out)

    def test_after_without_existing_and_near_cases_blocks(self):
        self._asset(held=(0, 4))
        code, out = self._adopt(after=("hf",))
        self.assertEqual(code, 1)
        self.assertIn(
            "CRITERION a existing-should BLOCKED existing 3/6 -> 0/0; 0 graded after\n", out
        )
        self.assertIn("CRITERION a near-fires BLOCKED near 0/0 fired; 0 graded after\n", out)
        self.assertIn("ADOPT a blocked\n", out)

    def test_existing_graded_counts_that_differ_block(self):
        self._asset(held=(0, 4))
        for model in (OPUS, kdet.SONNET):
            self._trigger("b", model, "a--trigger-should-x", 1, 1)
        code, out = self._adopt()
        self.assertEqual(code, 1)
        self.assertIn(
            "CRITERION a existing-should BLOCKED existing 2/2 -> 3/6; graded 2 before, 6 after\n",
            out,
        )
        self.assertIn("ADOPT a blocked\n", out)

    def test_effects_with_no_shared_case_block(self):
        self._asset(held=(0, 4), effects=False)
        for model in (OPUS, kdet.SONNET):
            self._put(
                "baseline",
                model,
                "effects",
                "a",
                [
                    {
                        "name": "a--effect-old",
                        "graders": [{"name": "design", "type": "llm"}],
                        "arms": {"with": [{"graders": [{"name": "design", "passed": True}]}]},
                    }
                ],
            )
            self._effects("e", model, [True] * 3)
        code, out = self._adopt()
        self.assertEqual(code, 1)
        self.assertIn("CRITERION a effects BLOCKED no effect case compared against baseline\n", out)
        self.assertIn("ADOPT a blocked\n", out)

    def test_effects_pass_names_the_pair_count(self):
        self._asset(held=(0, 4))
        code, out = self._adopt()
        self.assertEqual(code, 0)
        self.assertIn("CRITERION a effects PASS no regression vs baseline (2 pairs)\n", out)

    def test_mistyped_after_purpose_is_named(self):
        self._asset(held=(0, 4))
        self._adopt(after=("f", "hf-typo"))
        self.assertIn(f"MISSING purpose hf-typo ({self.tmp / 'hf-typo'})\n", self.err)
        self.assertNotIn("MISSING purpose f ", self.err)

    def test_neighbour_count_is_printed_with_one_neighbour(self):
        self._asset(held=(0, 4))
        self._trigger("f", kdet.SONNET, "n--trigger-near-y", 0, 3)
        _, out = self._adopt()
        self.assertTrue(self.err.endswith("NEIGHBOURS 1 assets checked\n"), self.err)
        self.assertNotIn("NEIGHBOURS", out)

    def test_neighbour_count_is_printed_with_no_neighbour(self):
        self._asset(held=(0, 4))
        self._adopt()
        self.assertTrue(self.err.endswith("NEIGHBOURS 0 assets checked\n"), self.err)

    def test_purpose_names_that_are_not_plain_are_refused(self):
        for bad in ("../x", "/abs", "a*"):
            argv = ["adopt", "--asset", "a", "--before", bad, "--after", "f", "--effects", "e"]
            with (
                contextlib.redirect_stderr(io.StringIO()) as err,
                self.assertRaises(SystemExit) as raised,
            ):
                khill.main(argv)
            self.assertEqual(raised.exception.code, 2, bad)
            self.assertIn("--before", err.getvalue(), bad)

    def test_malformed_effects_results_block_without_a_traceback(self):
        self._asset(held=(0, 4), effects=False)
        for model in (OPUS, kdet.SONNET):
            self._effects("baseline", model, [True] * 3)
            self._put(
                "e",
                model,
                "effects",
                "a",
                [
                    {
                        "name": "a--effect-x",
                        "graders": [{"name": "design", "type": "llm"}],
                        "arms": {"with": [{"graders": [{"passed": True}]}]},
                    }
                ],
            )
        code, out = self._adopt()
        self.assertEqual(code, 1)
        self.assertIn("ADOPT a blocked\n", out)
        self.assertIn("BLOCKED unreadable results for a: ", self.err)

    def test_main_end_to_end_swapping_before_and_after_changes_the_verdict(self):
        self._asset(held=(0, 4))

        def main(before, after):
            argv = [
                "adopt",
                "--asset",
                "a",
                "--before",
                *before,
                "--after",
                *after,
                "--effects",
                "e",
            ]
            with (
                contextlib.redirect_stdout(io.StringIO()) as out,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code = khill.main(argv)
            return code, out.getvalue().splitlines()[-1]

        self.assertEqual(main(["b", "hb"], ["f", "hf"]), (0, "ADOPT a adopt"))
        self.assertEqual(main(["f", "hf"], ["b", "hb"]), (0, "ADOPT a restore"))

    def test_own_near_fire_prints_no_neighbour_line(self):
        self._asset(held=(0, 4))
        self._trigger("f", kdet.SONNET, "a--trigger-near-y", 1, 3)
        _, out = self._adopt()
        self.assertIn("CRITERION a near-fires FAIL", out)
        self.assertNotIn("CRITERION neighbours", out)

    def test_neighbour_near_case_with_no_fire_prints_nothing(self):
        self._asset(held=(0, 4))
        self._trigger("f", kdet.SONNET, "n--trigger-near-y", 0, 3)
        _, out = self._adopt()
        self.assertTrue(out.endswith("ADOPT a adopt\n"), out)
        self.assertIn("NEIGHBOURS 1 assets checked", self.err)

    def test_heldx_slug_is_not_held_out(self):
        self.assertEqual(khill._split("a--trigger-should-heldx-y"), "")
        self.assertEqual(khill._split("a--trigger-should-held-y"), "held")

    def test_neighbour_should_with_differing_graded_counts_is_reviewed(self):
        self._asset(held=(0, 4))
        self._trigger("baseline", kdet.SONNET, "n--trigger-should-x", 3, 3)
        self._trigger("f", kdet.SONNET, "n--trigger-should-x", 2, 2)
        code, out = self._adopt()
        self.assertEqual(code, 0)
        self.assertIn(
            f"REVIEW n--trigger-should-x should 3/3 -> 2/2 on {kdet.SONNET} vs baseline (graded counts differ)\n",
            out,
        )


from evals.kbeval import affected as kaffected  # noqa: E402


class AffectedTests(TempRepoCase):
    def put(self, rel: str, text: str) -> None:
        path = self.repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def plan(self, *changes):
        return kaffected.affected(self.repo, list(changes))

    def test_skill_edit_maps_to_skill_preloading_agents_and_naming_commands(self):
        self.put(
            ".claude/agents/loader.md",
            "---\nname: loader\nskills:\n  - other\n  - demo-skill\n---\n\nBody.\n",
        )
        self.put(
            ".claude/agents/bystander.md",
            "---\nname: bystander\nskills:\n  - other\n---\n\nBody.\n",
        )
        self.put(
            ".claude/commands/j-names.md", "---\nname: j-names\n---\n\nUse `demo-skill` first.\n"
        )
        self.put(".claude/commands/j-bare.md", "---\nname: j-bare\n---\n\nUse demo-skill first.\n")
        plan = self.plan(("M", ".claude/skills/demo-skill/scripts/run.py"))
        self.assertEqual(plan.names, ["demo-skill", "j-names", "loader"])
        self.assertFalse(plan.everything)

    def test_command_edit_maps_to_command_and_its_workflow_cases(self):
        plan = self.plan(("M", ".claude/commands/j-demo.md"))
        self.assertEqual(plan.names, ["j-demo", "wf--demo-chain"])

    def test_agent_edit_maps_to_agent_and_naming_commands(self):
        self.put(
            ".claude/commands/j-names.md", "---\nname: j-names\n---\n\nDispatch `demo-agent`.\n"
        )
        plan = self.plan(("M", ".claude/agents/demo-agent.md"))
        self.assertEqual(plan.names, ["demo-agent", "j-names"])

    def test_reference_maps_to_citing_assets_with_reason(self):
        self.put(
            ".claude/skills/demo-skill/SKILL.md",
            "---\nname: demo-skill\n---\n\nBody.\nSee cat/ref.md.\n",
        )
        self.put(".claude/references/cat/mid.md", "# Mid\n\nBuilds on ref for details.\n")
        self.put(".claude/commands/j-deep.md", "---\nname: j-deep\n---\n\nRead mid first.\n")
        self.put(
            ".claude/commands/j-prefix.md", "---\nname: j-prefix\n---\n\nRead reference or ref-x.\n"
        )
        plan = self.plan(("M", ".claude/references/cat/ref.md"))
        self.assertEqual(plan.names, ["demo-skill", "j-deep"])
        self.assertIn(
            "AFFECTED demo-skill <- .claude/references/cat/ref.md via .claude/skills/demo-skill/SKILL.md:6",
            plan.notes,
        )
        self.assertIn(
            "AFFECTED j-deep <- .claude/references/cat/ref.md via .claude/commands/j-deep.md:5",
            plan.notes,
        )

    def test_case_edit_maps_to_exact_case_name(self):
        plan = self.plan(("M", "evals/cases/effects/j-demo--effect-cache/graders/result.md"))
        self.assertEqual(plan.names, ["j-demo--effect-cache"])

    def test_fixture_edit_maps_to_cases_that_copy_it(self):
        self.put("evals/fixtures/py-calc-red/x.py", "x = 1\n")
        plan = self.plan(("M", "evals/fixtures/py-calc/calc/ops.py"))
        self.assertEqual(plan.names, ["j-demo--effect-cache"])

    def test_claude_md_edit_is_everything(self):
        for path in (".claude/CLAUDE.md", "evals/harness.json", "evals/kbeval/stage.py"):
            plan = self.plan(("M", path))
            self.assertTrue(plan.everything, path)
            self.assertEqual(plan.names, [], path)

    def test_deleted_asset_is_noted_never_named(self):
        plan = self.plan(("D", ".claude/skills/gone/SKILL.md"), ("D", ".claude/agents/old.md"))
        self.assertEqual(plan.names, [])
        self.assertIn("DELETED gone: add an AGENT_REMOVALS entry", plan.notes)
        self.assertIn("DELETED old: add an AGENT_REMOVALS entry", plan.notes)

    def test_other_trees_are_unmeasured(self):
        paths = (".agents/skills/x/SKILL.md", ".codex/AGENTS.md", ".gemini/GEMINI.md",
                 ".claude/settings.json", ".claude/hooks/h.sh")  # fmt: skip
        plan = self.plan(*(("M", p) for p in paths))
        self.assertEqual(plan.names, [])
        self.assertEqual(plan.notes, [f"UNMEASURED {p}" for p in paths])

    def test_harness_edit_is_noted(self):
        plan = self.plan(("M", "evals/kbeval/results.py"))
        self.assertEqual(plan.names, [])
        self.assertEqual(plan.notes, ["HARNESS evals/kbeval/results.py: covered by make test"])
        self.assertFalse(plan.everything)

    def test_unrelated_path_maps_to_nothing(self):
        plan = self.plan(("M", "tests/test_evals.py"), ("M", "README.md"))
        self.assertEqual(plan, kaffected.Plan([], [], False))

    def test_names_are_sorted_and_unique(self):
        plan = self.plan(
            ("M", ".claude/commands/j-demo.md"),
            ("M", ".claude/skills/demo-skill/SKILL.md"),
            ("A", ".claude/commands/j-demo.md"),
        )
        self.assertEqual(plan.names, ["demo-skill", "j-demo", "wf--demo-chain"])

    def test_estimate_prices_each_suite_from_smoke_usd(self):
        names = [
            "demo-skill--trigger-should-rate",
            "j-demo--effect-cache",
            "demo-agent--effect-review",
            "wf--demo-chain",
        ]
        kb, vanilla, cost, _ = kaffected.estimate(self.repo, names)
        self.assertEqual((kb, vanilla), (4, 0))
        self.assertAlmostEqual(cost, 0.07 + 0.29 + 0.29 + 1.00)

    def test_estimate_adds_vanilla_runs_for_asset_names(self):
        kb, vanilla, cost, _ = kaffected.estimate(self.repo, ["demo-agent"])
        self.assertEqual((kb, vanilla), (1, 1))
        self.assertAlmostEqual(cost, 0.29 + 0.19)

    def test_estimate_samples_at_most_five_vanilla_effect_cases(self):
        for number in range(6):
            write_case(
                self.repo,
                "effects",
                f"demo-agent--effect-more{number}",
                EFFECT_FM,
                "Do it.",
                {"result": LLM_RESULT},
            )
        kb, vanilla, cost, _ = kaffected.estimate(self.repo, ["demo-agent"])
        self.assertEqual((kb, vanilla), (7, 5))
        self.assertAlmostEqual(cost, 7 * 0.29 + 5 * 0.19)

    def test_estimate_counts_a_case_named_by_asset_and_by_name_once(self):
        kb, vanilla, cost, _ = kaffected.estimate(
            self.repo, ["demo-agent", "demo-agent--effect-review"]
        )
        self.assertEqual((kb, vanilla), (1, 1))
        self.assertAlmostEqual(cost, 0.29 + 0.19)

    def test_deleted_command_is_noted_never_named(self):
        plan = self.plan(("D", ".claude/commands/j-demo.md"))
        self.assertEqual(plan.names, [])
        self.assertEqual(plan.notes, ["DELETED j-demo: add an AGENT_REMOVALS entry"])

    def test_case_edit_uses_the_suite_from_the_path(self):
        # The case exists under effects only; the same name under another suite maps to nothing.
        plan = self.plan(("M", "evals/cases/triggers/j-demo--effect-cache/graders/result.md"))
        self.assertEqual(plan.names, [])

    def test_inline_skills_list_in_agent_frontmatter_is_read(self):
        self.put(
            ".claude/agents/inline.md",
            "---\nname: inline\nskills: [other, demo-skill]\n---\n\nBody.\n",
        )
        plan = self.plan(("M", ".claude/skills/demo-skill/SKILL.md"))
        self.assertEqual(plan.names, ["demo-skill", "inline"])

    def test_deeper_harness_paths_map_to_nothing(self):
        plan = self.plan(("M", "evals/kbeval/sub/x.py"))
        self.assertEqual(plan, kaffected.Plan([], [], False))

    def test_edited_command_with_no_such_command_is_unmapped(self):
        plan = self.plan(("M", ".claude/commands/j-ghost.md"))
        self.assertEqual(
            plan,
            kaffected.Plan(
                [], ["UNMAPPED .claude/commands/j-ghost.md: no eval case names it"], False
            ),
        )

    def test_reference_nothing_cites_is_unmapped(self):
        plan = self.plan(("M", ".claude/references/cat/ref.md"))
        self.assertEqual(
            plan,
            kaffected.Plan(
                [], ["UNMAPPED .claude/references/cat/ref.md: no eval case names it"], False
            ),
        )

    def test_estimate_returns_names_with_no_case(self):
        _, _, _, unpriced = kaffected.estimate(self.repo, ["demo-skill", "demo-ghost"])
        self.assertEqual(unpriced, ["UNPRICED demo-ghost: no case"])

    def test_same_stem_references_in_two_categories_where_one_is_cited(self):
        self.put(".claude/references/other/ref.md", "# Other ref\n\nUncited.\n")
        self.put(
            ".claude/skills/demo-skill/SKILL.md",
            "---\nname: demo-skill\n---\n\nBody.\nSee cat/ref.md.\n",
        )
        plan = self.plan(("M", ".claude/references/cat/ref.md"))
        self.assertEqual(plan.names, ["demo-skill"])
        self.assertEqual(
            plan.notes,
            [
                "AFFECTED demo-skill <- .claude/references/cat/ref.md via .claude/skills/demo-skill/SKILL.md:6"
            ],
        )

    def test_affected_reads_each_kb_file_once(self):
        self.put(".claude/references/cat/mid.md", "# Mid\n\nBuilds on ref.\n")
        self.put(".claude/skills/demo-skill/SKILL.md", "---\nname: demo-skill\n---\n\nSee mid.\n")
        reads = []
        real = Path.read_text

        def counting(path, *args, **kwargs):
            if ".claude" in path.parts:
                reads.append(path)
            return real(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", counting):
            self.plan(
                ("M", ".claude/references/cat/ref.md"), ("M", ".claude/skills/demo-skill/SKILL.md")
            )
        self.assertTrue(reads)
        self.assertEqual(len(reads), len(set(reads)), sorted(map(str, reads)))

    def test_hyphen_prefixed_name_is_not_a_whole_word(self):
        self.assertIsNone(kaffected._word("Use x-demo-skill here.", "demo-skill"))
        self.assertEqual(kaffected._word("Use demo-skill here.", "demo-skill"), 1)

    def test_deleted_non_skill_md_file_names_the_skill(self):
        plan = self.plan(("D", ".claude/skills/demo-skill/scripts/run.py"))
        self.assertEqual(plan, kaffected.Plan(["demo-skill"], [], False))

    def test_command_prefixing_another_does_not_take_its_workflow_case(self):
        self.put(".claude/commands/j-dem.md", "---\nname: j-dem\n---\n\nDemo.\n")
        self.assertEqual(self.plan(("M", ".claude/commands/j-dem.md")).names, ["j-dem"])

    def test_fixture_name_inside_another_does_not_match(self):
        for fixture in ("calc", "py-calc-red"):
            self.put(f"evals/fixtures/{fixture}/x.py", "x = 1\n")
            self.assertEqual(self.plan(("M", f"evals/fixtures/{fixture}/x.py")).names, [], fixture)

    def test_estimate_adds_no_vanilla_run_for_a_workflow_case(self):
        kb, vanilla, cost, _ = kaffected.estimate(self.repo, ["wf--demo-chain"])
        self.assertEqual((kb, vanilla), (1, 0))
        self.assertAlmostEqual(cost, 1.00)

    def test_asset_reached_through_two_references_is_named_once(self):
        self.put(".claude/references/cat/one.md", "# One\n\nSee ref.\n")
        self.put(".claude/references/cat/two.md", "# Two\n\nSee ref.\n")
        self.put(
            ".claude/skills/demo-skill/SKILL.md",
            "---\nname: demo-skill\n---\n\nRead one.\nRead two.\n",
        )
        plan = self.plan(("M", ".claude/references/cat/ref.md"))
        self.assertEqual(plan.names, ["demo-skill"])
        self.assertEqual(
            plan.notes,
            [
                "AFFECTED demo-skill <- .claude/references/cat/ref.md via .claude/skills/demo-skill/SKILL.md:5"
            ],
        )

    def test_deleted_reference_still_names_its_citers(self):
        self.put(
            ".claude/skills/demo-skill/SKILL.md",
            "---\nname: demo-skill\n---\n\nBody.\nSee cat/ref.md.\n",
        )
        plan = self.plan(("D", ".claude/references/cat/ref.md"))
        self.assertEqual(plan.names, ["demo-skill"])
        self.assertEqual(
            plan.notes,
            [
                "AFFECTED demo-skill <- .claude/references/cat/ref.md via .claude/skills/demo-skill/SKILL.md:6"
            ],
        )


if __name__ == "__main__":
    unittest.main()
