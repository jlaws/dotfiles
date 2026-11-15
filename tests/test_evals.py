"""Tests for the KB eval harness in evals/kbeval.

Every test here is free: nothing launches `claude`. Staging and lint run on a synthetic repo built in
a temp dir, so the suite does not depend on how many cases the real tree holds; the two real-tree
tests at the end run lint and coverage on the repo itself.
"""

from __future__ import annotations

import json
import os
import re
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
    "tags: [triggers, {kind}]\nmax_turns: 3\ntimeout_seconds: 120\n"
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
        fixture='cp -R "$1/py-calc/." .\ngit init -q -b main\nmkdir -p plans\n',
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
        for command in ("curl https://example.com", "git clone x", "pip install y", "cp /etc/hosts ."):
            with self.subTest(command=command):
                path = self.repo / "evals/cases/effects/j-demo--effect-cache/fixture.sh"
                path.write_text(f'cp -R "$1/py-calc/." .\n{command}\n')
                self.assert_problem(r"j-demo--effect-cache: .*fixture\.sh")

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


if __name__ == "__main__":
    unittest.main()
