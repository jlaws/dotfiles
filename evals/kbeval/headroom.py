"""List the effect and workflow cases whose baseline cannot show a change.

A case that scores 1.0 with and without the KB, or fails both ways, leaves no room for a KB edit
to move it. `classify` sorts every case into one class; `report` prints the non-discriminating
ones, the assets left with no discriminating case, and live cases no baseline covers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set

from evals.kbeval.summary import PASS_AT

SUITES = ("effects", "workflows")
CLASSES = ("saturated", "dead", "kb-below-vanilla", "discriminating", "unmeasured")


def classify(rows_by_model: Dict[str, Dict[str, dict]]) -> Dict[str, str]:
    """Return the class of every case named by any model's rows."""
    names = sorted({name for rows in rows_by_model.values() for name in rows})
    classes: Dict[str, str] = {}
    for name in names:
        pairs = []
        for rows in rows_by_model.values():
            row = rows.get(name)
            pairs.append((None, None) if row is None else (row.get("kbScore"), row.get("vanillaScore")))
        if any(kb is None or vanilla is None for kb, vanilla in pairs):
            classes[name] = "unmeasured"
        elif all(kb == 1.0 and vanilla == 1.0 for kb, vanilla in pairs):
            classes[name] = "saturated"
        elif all(kb < PASS_AT and vanilla < PASS_AT for kb, vanilla in pairs):
            classes[name] = "dead"
        elif all(kb < vanilla for kb, vanilla in pairs):
            classes[name] = "kb-below-vanilla"
        else:
            classes[name] = "discriminating"
    return classes


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def report(baselines: Path, live: Mapping[str, Set[str]]) -> List[str]:
    """Return the report lines for the baselines under `baselines`.

    `live` maps a suite to the names of its live cases. Raises `ValueError` on an unreadable
    baseline file.
    """
    models = sorted(path.name for path in baselines.iterdir() if path.is_dir())
    if not models:
        raise ValueError(f"{baselines}: no model directories")
    lines: List[str] = []
    for suite in SUITES:
        rows_by_model: Dict[str, Dict[str, dict]] = {}
        for model in models:
            path = baselines / model / f"{suite}.json"
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                rows_by_model[model] = {row["name"]: row for row in doc["cases"]}
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}: {error}") from error
        classes = classify(rows_by_model)
        for name, label in classes.items():
            if label == "discriminating":
                continue
            scores = " ".join(
                f"{model}={_fmt(rows_by_model[model].get(name, {}).get('kbScore'))}"
                f"/{_fmt(rows_by_model[model].get(name, {}).get('vanillaScore'))}"
                for model in models
            )
            lines.append(f"HEADROOM {label} {name} {scores}")
        if suite == "effects":  # a workflow case spans several commands, so it belongs to no one asset
            assets: Dict[str, bool] = {}
            for name, label in classes.items():
                asset = name.split("--")[0]
                assets[asset] = assets.get(asset, False) or label == "discriminating"
            lines += [f"ASSET {asset} no discriminating case" for asset, found in sorted(assets.items()) if not found]
        lines += [f"MISSING {name}" for name in sorted(live.get(suite, set()) - set(classes))]
        counts = {label: sum(1 for value in classes.values() if value == label) for label in CLASSES}
        lines.append(
            f"{len(classes)} cases: {counts['saturated']} saturated, {counts['dead']} dead, "
            f"{counts['kb-below-vanilla']} kb-below-vanilla, {counts['discriminating']} discriminating, "
            f"{counts['unmeasured']} unmeasured"
        )
    return lines
