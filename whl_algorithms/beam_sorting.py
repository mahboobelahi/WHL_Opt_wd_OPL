"""Sorting-rule loading and deterministic BeamNode ordering utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from whl_core.paths import CONFIG_DIR

from whl_algorithms.beam_node import BeamNode

ALLOWED_DIRECTIONS = {"asc", "desc"}
ALLOWED_METRICS = {
    "pick_faces",
    "interior_storage",
    "retrieval_penalty",
    "scalar_score",
}
LEGACY_METRIC_ALIASES = {
    "storage_locked": "interior_storage",
    "RetrievalPenalty": "retrieval_penalty",
}


def _normalize_metric_name(metric: str) -> str:
    return LEGACY_METRIC_ALIASES.get(metric, metric)


def normalize_sorting_rule(rule: Any) -> list[tuple[str, str]]:
    """Normalize one sorting rule to ``[(metric, direction), ...]`` tuples."""
    if not isinstance(rule, list) or not rule:
        raise ValueError("sorting rule must be a non-empty list.")

    normalized: list[tuple[str, str]] = []
    for entry in rule:
        if isinstance(entry, dict):
            metric = entry.get("metric")
            direction = entry.get("direction")
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            metric, direction = entry
        else:
            raise ValueError("rule entries must be metric/direction pairs.")

        if not isinstance(metric, str) or not isinstance(direction, str):
            raise ValueError("metric and direction must be strings.")

        clean_metric = _normalize_metric_name(metric)
        if clean_metric not in ALLOWED_METRICS:
            raise ValueError(f"unsupported sorting metric: {metric}")
        if direction not in ALLOWED_DIRECTIONS:
            raise ValueError(f"unsupported sorting direction: {direction}")
        normalized.append((clean_metric, direction))

    return normalized


def validate_sorting_rules(rules: dict) -> None:
    """Validate a sorting-rule registry."""
    if not isinstance(rules, dict) or not rules:
        raise ValueError("sorting rules must be a non-empty dictionary.")

    for name, rule in rules.items():
        if not isinstance(name, str) or not name:
            raise ValueError("sorting rule names must be non-empty strings.")
        normalize_sorting_rule(rule)


def load_sorting_rules(path: Path | None = None) -> dict[str, list[tuple[str, str]]]:
    """Load and normalize Beam Search sorting rules from YAML."""
    rules_path = CONFIG_DIR / "sorting_rules.yaml" if path is None else Path(path)
    with rules_path.open("r", encoding="utf-8") as file:
        raw_rules = yaml.safe_load(file) or {}

    validate_sorting_rules(raw_rules)
    return {
        name: normalize_sorting_rule(rule)
        for name, rule in raw_rules.items()
    }


def _node_sort_key(
    node: BeamNode,
    rule: list[tuple[str, str]],
) -> tuple:
    values: list[float | bytes] = []
    for metric, direction in rule:
        if metric not in node.metrics:
            raise KeyError(f"node is missing sorting metric: {metric}")
        value = float(node.metrics[metric])
        values.append(value if direction == "asc" else -value)
    values.append(node.signature)
    return tuple(values)


def sort_nodes_by_rule(
    nodes: list[BeamNode],
    rule: list[tuple[str, str]] | str,
    rules: dict[str, list[tuple[str, str]]] | None = None,
) -> list[BeamNode]:
    """Return nodes sorted by a Beam Search sorting rule."""
    if isinstance(rule, str):
        selected_rules = load_sorting_rules() if rules is None else rules
        if rule not in selected_rules:
            raise KeyError(f"unknown sorting rule: {rule}")
        normalized_rule = selected_rules[rule]
    else:
        normalized_rule = normalize_sorting_rule(rule)

    return sorted(nodes, key=lambda node: _node_sort_key(node, normalized_rule))


def sample_sorting_rule(
    rules: dict,
    rng: np.random.Generator,
    allowed_rule_names: list[str] | None = None,
) -> str:
    """Sample a sorting-rule name deterministically from ``rng``."""
    validate_sorting_rules(rules)
    normalized_rules = {
        name: normalize_sorting_rule(rule)
        for name, rule in rules.items()
    }

    if allowed_rule_names is None:
        candidates = sorted(normalized_rules)
    else:
        missing = set(allowed_rule_names) - set(normalized_rules)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise KeyError(f"unknown allowed sorting rule(s): {missing_text}")
        candidates = list(allowed_rule_names)

    if not candidates:
        raise ValueError("no sorting rules available to sample.")
    return str(rng.choice(candidates))


__all__ = [
    "ALLOWED_DIRECTIONS",
    "ALLOWED_METRICS",
    "LEGACY_METRIC_ALIASES",
    "load_sorting_rules",
    "normalize_sorting_rule",
    "sample_sorting_rule",
    "sort_nodes_by_rule",
    "validate_sorting_rules",
]
