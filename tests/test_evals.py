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

from evals.kbeval import cases as kcases
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
        f"input_match: '\"skill\"\\s*:\\s*\"(?:kb:)?{name}\"'\n{bound}---\n"
    )


def agent_dispatched(name: str) -> str:
    return (
        "---\ntype: tool_used\ntool: Agent\n"
        f"input_match: '\"subagent_type\"\\s*:\\s*\"kb:{name}\"'\n---\n"
    )


def make_repo(root: Path) -> Path:
    """Build a minimal repo: one skill, two commands (one user-only), one agent, one reference."""
    files = {
        ".claude/CLAUDE.md": "# KB rules\n\nSay \"hi\" and use \\n literally.\n",
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
        root, "triggers", "demo-skill--trigger-should-rate", TRIGGER_FM.format(kind="trigger-should"),
        "Help me design a rate limiter for my API.", {"fired": skill_fired("demo-skill")},
    )
    write_case(
        root, "triggers", "demo-skill--trigger-near-typo", TRIGGER_FM.format(kind="trigger-near"),
        "Fix the typo in README.", {"fired": skill_fired("demo-skill", 0, 0)},
    )
    write_case(
        root, "triggers", "j-quiet--trigger-noauto-ask", TRIGGER_FM.format(kind="trigger-noauto"),
        "Do the quiet thing for me.", {"fired": skill_fired("j-quiet", 0, 0)},
    )
    write_case(
        root, "effects", "j-demo--effect-cache", EFFECT_FM,
        "/kb:j-demo design a cache layer and save it to ./design.md.",
        {"result": LLM_RESULT, "fired": skill_fired("j-demo")},
        fixture='set -euo pipefail\ncp -R "$1/py-calc/." .\ngit init -q -b main\nmkdir -p plans\n',
        case_yaml="context:\n  add_dirs: [resources]\n",
    )
    write_case(
        root, "effects", "demo-agent--effect-review", EFFECT_FM,
        "Use the kb:demo-agent agent to review the cache design and save it to ./design.md.",
        {"result": LLM_RESULT, "dispatched": agent_dispatched("demo-agent")},
    )
    write_case(
        root, "effects", "demo-skill--effect-limiter", EFFECT_FM,
        "Design a rate limiter and save the design to ./design.md.",
        {"result": LLM_RESULT, "fired": skill_fired("demo-skill")},
    )
    write_case(
        root, "workflows", "wf--demo-chain", WORKFLOW_FM,
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
                    ln for ln in frontmatter_of(prompt).splitlines()
                    if ln.startswith("append_system_prompt: ")
                )
                text = json.loads(line[len("append_system_prompt: "):])
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
            ["bash", str(case / "seed.sh")], cwd=work, env=env, capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((work / "calc" / "ops.py").is_file())
        self.assertTrue((work / ".kb-references" / "cat" / "ref.md").is_file())
        self.assertIn(".kb-references", (work / ".git" / "info" / "exclude").read_text())
        self.assertLess(seed.index("fixture.sh"), seed.index(".kb-references"))

    def test_case_yaml_has_schema_name_and_scaffold(self):
        plugin = self.stage()
        text = (plugin / "evals-effects" / "j-demo--effect-cache" / "case.yaml").read_text()
        self.assertIn('schema_version: "1.1"\n', text)
        self.assertIn("name: j-demo--effect-cache\n", text)
        self.assertIn("  scaffold_script: seed.sh\n", text)
        self.assertIn("  add_dirs: [resources]\n", text)
        self.assertEqual(text.count("context:"), 1)
        trigger = (plugin / "evals-triggers" / "demo-skill--trigger-should-rate" / "case.yaml")
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
        self.assertNotIn("disable-model-invocation", (plugin / "commands" / "j-quiet.md").read_text())
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
            ln for ln in frontmatter_of(case / "prompt.md").splitlines()
            if ln.startswith("append_system_prompt: ")
        )
        self.assertIn(str(case / "_refs"), json.loads(line[len("append_system_prompt: "):]))
        self.assertNotIn(".kb-references", (case / "seed.sh").read_text())

    def test_stage_exclude_tags(self):
        folder = write_case(
            self.repo, "effects", "demo-skill--effect-second-mode",
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
            any(re.search(pattern, p) for p in problems), f"no problem matching {pattern!r} in {problems}"
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
        (self.repo / "evals/cases/triggers/demo-skill--trigger-should-rate/graders/extra.md").write_text(
            LLM_RESULT
        )
        self.assert_problem(r"demo-skill--trigger-should-rate: .*exactly one grader")

    def test_lint_trigger_bounds_follow_kind(self):
        self.edit("triggers/demo-skill--trigger-near-typo/graders/fired.md", "max: 0\n", "")
        self.assert_problem(r"demo-skill--trigger-near-typo: .*min: 0.*max: 0")

    def test_lint_trigger_input_match_names_asset(self):
        self.edit("triggers/demo-skill--trigger-should-rate/graders/fired.md", "demo-skill", "other")
        self.assert_problem(r"demo-skill--trigger-should-rate: .*input_match")

    def test_lint_effect_needs_result_grader(self):
        (self.repo / "evals/cases/effects/j-demo--effect-cache/graders/result.md").unlink()
        self.assert_problem(r"j-demo--effect-cache: .*result grader")

    def test_lint_effect_needs_process_grader(self):
        (self.repo / "evals/cases/effects/j-demo--effect-cache/graders/fired.md").unlink()
        self.assert_problem(r"j-demo--effect-cache: .*process grader")

    def test_lint_result_grader_never_mentions_kb(self):
        self.edit("effects/j-demo--effect-cache/graders/result.md", "names a cache", "uses kb:j-demo and names a cache")
        self.assert_problem(r"j-demo--effect-cache: .*kb:")

    def test_lint_unknown_frontmatter_key(self):
        self.edit("effects/j-demo--effect-cache/prompt.md", "max_turns: 40\n", "max_turns: 40\nmodel: opus\n")
        self.assert_problem(r"j-demo--effect-cache: .*model")

    def test_lint_suite_limits(self):
        self.edit("effects/j-demo--effect-cache/prompt.md", "max_turns: 40", "max_turns: 12")
        self.assert_problem(r"j-demo--effect-cache: .*max_turns")

    def test_lint_absolute_limits(self):
        self.edit("effects/j-demo--effect-cache/prompt.md", "max_turns: 40\n", "max_turns: 40\nruns: 51\n")
        self.assert_problem(r"j-demo--effect-cache: .*runs")

    def test_lint_env_keys(self):
        self.edit("effects/j-demo--effect-cache/prompt.md", "max_turns: 40\n", "max_turns: 40\nenv: {HOME: /x}\n")
        self.assert_problem(r"j-demo--effect-cache: .*EVAL_")

    def test_lint_forbidden_fixture_commands(self):
        for command in ("curl https://example.com", "git clone x", "pip install y",
                        "rm -rf build", "rm -fR x", "rm --recursive x", "rm -f -r x", "node -e 1", "perl -e 1",
                        "python3 -Ic 1", "git config --global user.name x", "git config --system x",
                        "git push --force", "git push -f origin", "git push origin +main",
                        "git push --force-with-lease", "ssh-keyscan host", "ncat host 80", "wget2 x",
                        "zsh -c 'x'", "git remote add origin git@host:x/y.git", "bash -c 'x'", "echo '#'; curl x"):
            with self.subTest(command=command):
                path = self.repo / "evals/cases/effects/j-demo--effect-cache/fixture.sh"
                path.write_text(f'set -euo pipefail\ncp -R "$1/py-calc/." .\n{command}\n')
                self.assert_problem(r"j-demo--effect-cache: .*fixture\.sh:\d+ runs a forbidden command")

    def test_lint_fixture_paths_outside_workspace(self):
        for command in ("cp /etc/hosts .", 'cp -R "$1/../.." .'):
            with self.subTest(command=command):
                path = self.repo / "evals/cases/effects/j-demo--effect-cache/fixture.sh"
                path.write_text(f'set -euo pipefail\ncp -R "$1/py-calc/." .\n{command}\n')
                self.assert_problem(r"j-demo--effect-cache: .*fixture\.sh:\d+ (?:uses a \.\. path|touches a path)")

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
        self.edit("effects/j-demo--effect-cache/prompt.md", "./design.md.", "./design.md and notes to ./notes.md.")
        self.assert_problem(r"j-demo--effect-cache: .*no-foo: needs a positive regex or llm grader on notes\.md")

    def test_lint_forbidden_words_match_whole_commands(self):
        path = self.repo / "evals/cases/effects/j-demo--effect-cache/fixture.sh"
        path.write_text('set -euo pipefail\ncp -R "$1/py-calc/." .\nprintf "agent evals and ncurses\\n" > notes.txt\n')
        problems = [p for p in self.problems() if "fixture.sh" in p]
        self.assertEqual(problems, [])

    def test_lint_fixture_must_fail_fast(self):
        path = self.repo / "evals/cases/effects/j-demo--effect-cache/fixture.sh"
        path.write_text('cp -R "$1/py-calc/." .\n')
        self.assert_problem(r"j-demo--effect-cache: .*set -euo pipefail")

    def test_lint_case_yaml_context_shape(self):
        for body in ("context:\n  add_dirs:\n    - resources\n", "context:\n  env:\n    FOO: bar\n"):
            with self.subTest(body=body):
                (self.repo / "evals/cases/effects/j-demo--effect-cache/case.yaml").write_text(body)
                self.assert_problem(r"j-demo--effect-cache: .*case\.yaml")

    def test_lint_secret_shaped_string(self):
        self.edit("effects/j-demo--effect-cache/prompt.md", "design a cache", "use sk-ant-abc to design a cache")
        self.assert_problem(r"j-demo--effect-cache: .*sk-ant-")

    def test_lint_author_scaffold_script(self):
        self.edit("effects/j-demo--effect-cache/case.yaml", "context:\n", "context:\n  scaffold_script: mine.sh\n")
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
        self.edit("effects/demo-agent--effect-review/prompt.md", "Use the kb:demo-agent agent to review", "Review")
        self.assert_problem(r"demo-agent--effect-review: .*Use the kb:demo-agent agent to ")

    def test_lint_skill_prompt_never_names_skill(self):
        self.edit("effects/demo-skill--effect-limiter/prompt.md", "Design a rate", "Use demo-skill to design a rate")
        self.assert_problem(r"demo-skill--effect-limiter: .*names the skill")

    def test_lint_at_most_one_extended_case_per_asset(self):
        for slug in ("second", "third"):
            write_case(
                self.repo, "effects", f"demo-skill--effect-{slug}",
                EFFECT_FM.replace("[effects, effect]", "[effects, effect, extended]"),
                "Review a rate limiter and save notes to ./design.md.",
                {"result": LLM_RESULT, "fired": skill_fired("demo-skill")},
            )
        self.assert_problem(r"demo-skill: .*extended")

    def test_lint_malformed_quoted_scalar(self):
        self.edit("effects/j-demo--effect-cache/graders/fired.md", "input_match: '", "input_match: 'it's ")
        self.assert_problem(r"j-demo--effect-cache: .*malformed quoted YAML")

    def test_lint_js_incompatible_regex(self):
        for pattern in ("'(?m)^Verdict'", "'(?i:verdict)'", "'\\Averdict'"):
            with self.subTest(pattern=pattern):
                (self.repo / "evals/cases/effects/j-demo--effect-cache/graders/re.md").write_text(
                    f"---\ntype: regex\npattern: {pattern}\ntarget: {{ source: file, path: design.md }}\n---\n"
                )
                self.assert_problem(r"j-demo--effect-cache: .*inline flags")

    def test_lint_graded_file_under_dot_claude(self):
        self.edit("effects/j-demo--effect-cache/graders/result.md", "path: design.md", "path: .claude/settings.json")
        self.assert_problem(r"j-demo--effect-cache: .*under \.claude/")

    def test_lint_valid_quoted_scalars_pass(self):
        for value in ('"Merge branch \'feat/rst\'"', "'a''b'", '"tab\\tend"', "{ source: file, path: 'x.md' }"):
            with self.subTest(value=value):
                self.assertFalse(kcases.malformed_scalar(value))

    def test_lint_invalid_double_quoted_escape(self):
        for value in ('"\\d+"', "{ tool: Write, input_match: 'it's' }"):
            with self.subTest(value=value):
                self.assertTrue(kcases.malformed_scalar(value))

    def test_lint_js_regex_in_input_match_and_flags(self):
        self.edit("effects/j-demo--effect-cache/graders/fired.md", "input_match: '", "input_match: '(?P<n>x)")
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
            "---\ntype: llm\nfocus: { source: file, path: design.md }\n---\n\n" + claims + "FAIL if nothing.\n"
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
        self.edit("effects/j-demo--effect-cache/graders/result.md", "path: design.md", "path: notes.md")
        self.assert_problem(r"j-demo--effect-cache: .*'notes\.md' is neither named")

    def test_lint_non_md_grader_file(self):
        (self.repo / "evals/cases/effects/j-demo--effect-cache/graders/extra.yml").write_text("type: llm\n")
        self.assert_problem(r"j-demo--effect-cache: .*extra\.yml is not a \.md grader")

    def test_lint_double_md_extension(self):
        graders = self.repo / "evals/cases/effects/j-demo--effect-cache/graders"
        (graders / "result.md").rename(graders / "result.md.md")
        self.assert_problem(r"j-demo--effect-cache: .*double \.md")

    def test_lint_tags_name_suite_and_kind(self):
        self.edit("effects/j-demo--effect-cache/prompt.md", "tags: [effects, effect]", "tags: [effect]")
        self.assert_problem(r"j-demo--effect-cache: .*tags")

    def test_lint_no_web_tools(self):
        self.edit("effects/j-demo--effect-cache/prompt.md", "TodoWrite]", "TodoWrite, WebFetch]")
        self.assert_problem(r"j-demo--effect-cache: .*WebFetch")


class CoverageTests(TempRepoCase):
    def gaps(self, exempt: frozenset = frozenset()) -> list:
        return kcases.coverage(kcases.discover(self.repo / "evals" / "cases"), self.repo, exempt)

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

    def test_exempt_assets_are_skipped(self):
        exempt = frozenset({"demo-skill", "j-demo", "j-quiet", "demo-agent"})
        self.assertEqual(self.gaps(exempt), [])


class RealTreeTests(unittest.TestCase):
    def test_real_cases_pass_lint(self):
        problems = kcases.lint(kcases.discover(REPO / "evals" / "cases"), REPO)
        self.assertEqual(problems, [], "\n".join(problems))

    def test_real_assets_meet_coverage(self):
        found = kcases.discover(REPO / "evals" / "cases")
        gaps = kcases.coverage(found, REPO, kcases.PENDING_EVAL_COVERAGE)
        self.assertEqual(gaps, [], "\n".join(gaps))

    def test_pending_names_only_real_assets(self):
        assets = set(kcases.assets(REPO))
        self.assertEqual(sorted(kcases.PENDING_EVAL_COVERAGE - assets), [])



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
    captured = json.loads((REPO / "tests" / "fixtures" / "kbeval" / "aggregate-result.json").read_text())
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
        refuse = mock.patch.object(krunner.ClaudeCli, "run", side_effect=AssertionError("real claude called"))
        refuse.start()
        self.addCleanup(refuse.stop)

    def run_suite(self, cli, suite="effects", assets=("j-demo",), staging="kb", **extra):
        err = io.StringIO()
        options = {"plugin": self.plugin, "state": self.state, "ledger_path": self.ledger_path,
                   "budget": self.budget, "p95": 5.0, **extra}
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


class RunnerBatchTests(RunnerCase):
    def test_batch_args(self):
        cli = FakeCli(lambda args: (result_doc({case_arg(args).rstrip("*-"): [None]}), 0, ""))
        code, _ = self.run_suite(cli)
        self.assertEqual(code, 0)
        out = self.state / "test" / "kb" / MODEL / "effects" / "j-demo"
        self.assertEqual(
            cli.calls[0],
            ["plugin", "eval", str(self.plugin), "--eval-dir", "evals-effects", "--case", "j-demo--*",
             "--runs", "3", "--model", MODEL, "--judge-model", JUDGE, "--ablation", "none",
             "--threshold", "0", "--trust-plugin", "--scaffold", "--no-publish", "-j", "8",
             "--json", f"{out}.json", "--output-dir", str(out), "--max-cost-usd", "60.00",
             "--allow-tools", "Bash", "Write", "Edit"],
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
        self.assertEqual([(e["costUsd"], e["charged"]) for e in self.entries()], [(60.0, "reserved")])

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
            subprocess.Popen([sys.executable, "-c", script, str(path)], cwd=REPO, stdin=subprocess.DEVNULL)
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
        main = {"j-demo--effect-cache": ["API error: 429 rate_limit_error", None, "timed out after 900s"]}

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
                self.repo, "effects", f"demo-skill--effect-{slug}", EFFECT_FM,
                "Design a rate limiter and save the design to ./design.md.",
                {"result": LLM_RESULT, "fired": skill_fired("demo-skill")},
            )
        cli = FakeCli(lambda args: (result_doc({case_arg(args): [None]}, cost=0.1), 0, ""))
        status = korch.smoke(["demo-agent", "demo-skill", "j-demo"], cli=cli)
        self.assertEqual(status, 0)
        vanilla = [case_arg(c) for c in self.calls_for(cli, "vanilla")]
        self.assertEqual(
            vanilla,
            ["demo-agent--effect-review", "demo-skill--effect-b", "demo-skill--effect-c",
             "demo-skill--effect-d", "demo-skill--effect-e"],
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
        korch.baseline("claude-opus-5-5", vanilla_runs=1, vanilla_models=["claude-opus-5-5"], cli=cli)
        vanilla = self.calls_for(cli, "vanilla")
        self.assertTrue(vanilla)
        self.assertEqual({c[c.index("--runs") + 1] for c in vanilla}, {"1"})


# --- summary and compare ------------------------------------------------------------------------

import copy  # noqa: E402

from evals.kbeval import compare as kcompare  # noqa: E402
from evals.kbeval import summary as ksummary  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "kbeval" / "aggregate-result.json"
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

    def put(self, staging: str, label: str, doc: dict, suite: str = "effects", keyed: bool = True) -> None:
        path = self.results / staging / OPUS / suite / f"{label}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))
        key = path.with_suffix(".key")
        if keyed:
            key.write_text("k\n")
        elif key.exists():
            key.unlink()

    def test_unkeyed_and_partial_results_are_ignored(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True, True]))
        self.put("kb", "j-arch@j-arch--effect-cache-layer", with_runs(real_doc(), [False] * 3), keyed=False)
        partial = with_runs(real_doc(), [False] * 3)
        partial["partial"] = True
        self.put("vanilla", "j-arch", partial)
        doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)
        self.assertEqual(doc["cases"][0]["kbScore"], 1.0)
        self.assertEqual(doc["cases"][0]["vanillaRuns"], 0)

    def test_pass_boundary_is_two_thirds(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True, False]))
        self.assertTrue(ksummary.summarize(self.results, OPUS, "effects", SETTINGS)["cases"][0]["passed"])
        self.put("kb", "j-arch", with_runs(real_doc(), [True, False, False]))
        self.assertFalse(ksummary.summarize(self.results, OPUS, "effects", SETTINGS)["cases"][0]["passed"])

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
        self.assertEqual((row["kind"], row["kbScore"], row["vanillaScore"]), ("trigger-should", 1.0, None))
        self.assertEqual(ksummary.check(summary), [])

    def test_write_refuses_secret_shaped_text(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True, True], [None, "sk-ant-leak", None]))
        doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)
        with self.assertRaises(ValueError):
            ksummary.write(doc, self.results / "out.json")

    def test_summary_from_real_doc(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True, False]))
        self.put("vanilla", "j-arch", with_runs(real_doc(), [True, False, False]))
        doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)
        for key in ("schema", "suite", "model", "judgeModel", "claudeVersion", "date", "runs", "costUsd"):
            self.assertIn(key, doc)
        self.assertEqual((doc["suite"], doc["model"], doc["judgeModel"]), ("effects", OPUS, "claude-fable-5-2"))
        self.assertEqual(doc["claudeVersion"], "2.1.325")
        self.assertEqual(doc["runs"], SETTINGS)
        (case,) = doc["cases"]
        self.assertEqual((case["asset"], case["kind"]), ("j-arch", "effect"))
        # Result graders only: the llm `design` grader. The tool_used `delegated` grader is the
        # process indicator, reported as firedRate.
        self.assertAlmostEqual(case["kbScore"], 2 / 3, places=3)
        self.assertAlmostEqual(case["vanillaScore"], 1 / 3, places=3)
        self.assertAlmostEqual(case["delta"], 1 / 3, places=3)
        self.assertEqual(case["firedRate"], 1.0)
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
        self.assertTrue(any(line.lstrip().startswith('{"asset": "j-arch"') for line in first.splitlines()))

    def test_rerun_result_replaces_the_api_error_case(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True, True], [None, "API error: 429", None]))
        self.put("kb", "j-arch@j-arch--effect-cache-layer", with_runs(real_doc(), [True, True, True]))
        doc = ksummary.summarize(self.results, OPUS, "effects", SETTINGS)
        self.assertEqual(doc["cases"][0]["kbRuns"], 3)
        self.assertEqual(doc["cases"][0]["errors"], [])

    def test_summarize_check(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True]))
        self.put("vanilla", "j-arch", with_runs(real_doc(), [True, True, True]))
        problems = ksummary.check(ksummary.summarize(self.results, OPUS, "effects", SETTINGS))
        self.assertTrue(any("kbRuns" in p for p in problems), problems)
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True, True], [None, None, "API error: 529"]))
        problems = ksummary.check(ksummary.summarize(self.results, OPUS, "effects", SETTINGS))
        self.assertTrue(any("API" in p for p in problems), problems)
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True, True]))
        self.assertEqual(ksummary.check(ksummary.summarize(self.results, OPUS, "effects", SETTINGS)), [])

    def test_check_skips_vanilla_for_unlisted_models(self):
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True, True]))
        settings = {**SETTINGS, "vanillaModels": ["claude-sonnet-5-5"]}
        self.assertEqual(ksummary.check(ksummary.summarize(self.results, OPUS, "effects", settings)), [])

    def test_mixed_claude_versions_are_refused(self):
        other = real_doc()
        other["claudeVersion"] = "2.1.999"
        self.put("kb", "j-arch", with_runs(real_doc(), [True, True, True]))
        self.put("vanilla", "j-arch", with_runs(other, [True, True, True]))
        with self.assertRaises(ValueError):
            ksummary.summarize(self.results, OPUS, "effects", SETTINGS)


def summary_doc(cases: dict, model: str = OPUS) -> dict:
    return {
        "schema": 1, "suite": "effects", "model": model,
        "aggregates": {
            "kb": {"mean": sum(c[0] for c in cases.values()) / len(cases), "ci95": [0.4, 0.9]},
            "vanilla": {"mean": 0.3, "ci95": [0.1, 0.5]},
            "delta": {"mean": 0.3, "ci95": [0.1, 0.5]},
        },
        "cases": [
            {"asset": name.split("--")[0], "kind": "effect", "name": name, "kbScore": kb, "passed": passed}
            for name, (kb, passed) in sorted(cases.items())
        ],
    }  # fmt: skip


class CompareTests(unittest.TestCase):
    def test_compare_overlap_only(self):
        old = summary_doc({"a--effect-x": (1.0, True), "b--effect-y": (0.5, False), "c--effect-gone": (1.0, True)})
        new = summary_doc({"a--effect-x": (0.6, False), "b--effect-y": (0.5, False), "d--effect-new": (1.0, True)})
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
        self.assertEqual([(e["charged"], e["costUsd"]) for e in self.entries()], [("reserved", 60.0)])

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
        for text in ("API error: 429", "rate_limit_error", "status 529", "ECONNRESET", "Overloaded"):
            self.assertRegex(text, krunner.API_ERROR)
        for text in ("timed out after 500 ms", "hit max_turns (40)", "exit 1: Reached maximum number of turns (3)"):
            self.assertNotRegex(text, krunner.API_ERROR)

    def test_jobs_below_one_is_refused(self):
        cli = FakeCli(lambda args: (result_doc({"x": [None]}), 0, ""))
        err = io.StringIO()
        options = {"plugin": self.plugin, "state": self.state, "ledger_path": self.ledger_path,
                   "budget": self.budget, "p95": 5.0}
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
        self.assertTrue(kledger.reserve({"period": "p", "costUsd": 30.0, "charged": "reserved"}, 100, 20, self.path))
        self.assertFalse(kledger.reserve({"period": "p", "costUsd": 0.5, "charged": "reserved"}, 100, 20, self.path))
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
            proc = subprocess.Popen([sys.executable, "-c", script, str(self.path)], cwd=REPO, stdin=subprocess.DEVNULL)
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
        self.assertTrue(all(float(c[c.index("--max-cost-usd") + 1]) <= korch.PILOT_CAP_USD for c in cli.calls))
        self.assertTrue(self.calls_for(cli, "vanilla"))
        self.assertFalse(any(c[c.index("--eval-dir") + 1] == "evals-triggers" for c in self.calls_for(cli, "vanilla")))

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
    def test_summarize_writes_nothing_when_a_check_fails(self):
        from evals.kbeval import __main__ as kmain

        err = io.StringIO()
        with mock.patch.object(kmain, "REPO", self.repo), contextlib.redirect_stderr(err):
            code = kmain.main(["summarize", "--model", OPUS])
        self.assertEqual(code, 1)
        self.assertIn("no summary written", err.getvalue())
        self.assertFalse((self.repo / "evals" / "baselines").exists())

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
                    ["bash", str(seed)], cwd=work, env=env, capture_output=True, text=True,
                    stdin=subprocess.DEVNULL, timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((work / ".kb-references").is_dir())

    def test_py_calc_keeps_its_seeded_bug(self):
        fixture = REPO / "evals" / "fixtures" / "py-calc"
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        bug = subprocess.run(
            [sys.executable, "-c", "from calc.ops import mean; mean([])"], cwd=fixture, env=env,
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60,
        )
        self.assertIn("ZeroDivisionError", bug.stderr)
        self.assertIn("0.0", (fixture / "calc" / "ops.py").read_text())
        suite = subprocess.run(
            [sys.executable, "-m", "unittest"], cwd=fixture, env=env, capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=60,
        )
        self.assertEqual(suite.returncode, 0, suite.stderr)

    def test_py_calc_red_fails_only_on_the_empty_mean(self):
        fixture = REPO / "evals" / "fixtures" / "py-calc-red"
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        suite = subprocess.run(
            [sys.executable, "-m", "unittest"], cwd=fixture, env=env, capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=60,
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
            [sys.executable, "tool.py", "--input", "data.csv", "--out", "x.jsonl"], cwd=work, env=env,
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60,
        )
        self.assertEqual(old.returncode, 2)
        self.assertIn("unrecognized arguments: --out", old.stderr)
        self.assertIn("--out", (work / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
