#!/usr/bin/env python3
"""P4-N: extend the frozen P4-M candidate on a second untouched window.

This script does not synthesize or edit rules. It verifies the exact P4-M
candidate hash, pre-registers a new ALFWorld split disjoint from every prior
game ID recorded by P4-M, and compares old Memory against the frozen candidate
on 24 new games. Passing only preserves the candidate in the human-review
queue; it never authorizes production Memory, training labels, or actions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from collections import defaultdict
from pathlib import Path


WORKING = Path("/kaggle/working")
P4M_SCRIPT = Path(
    "/kaggle/input/datasets/placideo/jev-p4m-automated-failure-rule-replay-staging/"
    "jev_p4m_automated_failure_rule_replay_staging.py"
)
P4M_OUTPUT = WORKING / "jev_p4m_automated_failure_rule_replay_staging"
P4M_CANDIDATE = P4M_OUTPUT / "p4m_candidate_state_rules.json"
P4M_QUEUE = P4M_OUTPUT / "p4m_staging_queue.jsonl"
P4M_REPORT = P4M_OUTPUT / "p4m_report.json"
P4M_STAGING_MANIFEST = P4M_OUTPUT / "staging_manifest" / "p4g_preregistered_split_manifest.json"
MISTRAL_BASE = WORKING / "jev_p4l_mistral7b_frozen_base.py"

EXPECTED_CANDIDATE_ID = "jev-p4m-state-rules-v1"
EXPECTED_CANDIDATE_SHA256 = "e44efb29ef6b8221864ae33f683d0983ca2c99ac2ab4a2fca3f48fc6c79d7b2f"
SELECTION_SEED = 20261001
TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]
GAMES_PER_TASK_TYPE = 4
MIN_ABSOLUTE_DELTA = 0.25
MAX_REGRESSIONS = 0
MAX_PARSE_FALLBACKS = 0
MAX_DECODER_VIOLATIONS = 0

OUTPUT_ROOT = WORKING / "jev_p4n_untouched_validation_window"
MANIFEST_DIR = OUTPUT_ROOT / "preregistration"
REPLAY_DIR = OUTPUT_ROOT / "untouched_replay"
ARCHIVE_BASE = WORKING / "jev_p4n_untouched_validation_window_artifacts"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def enumerate_untouched_ids(base, excluded: set[str]) -> tuple[list[str], dict[str, list[str]]]:
    from alfworld.agents.environment import get_environment

    engine = get_environment("AlfredTWEnv")(base.config(), train_eval="eval_out_of_distribution")
    try:
        game_files = sorted(engine.game_files)
    finally:
        close = getattr(engine, "close", None)
        if callable(close):
            close()

    by_task: dict[str, list[str]] = defaultdict(list)
    for path in game_files:
        game_id = base.relative_game_id(path)
        task_type = base.task_type_for(path)
        if task_type in TASK_TYPES and game_id not in excluded:
            by_task[task_type].append(game_id)

    selected: list[str] = []
    selection_by_task: dict[str, list[str]] = {}
    for task_type in TASK_TYPES:
        ranked = sorted(
            by_task[task_type],
            key=lambda game_id: hashlib.sha256(f"{SELECTION_SEED}:{game_id}".encode()).hexdigest(),
        )
        chosen = ranked[:GAMES_PER_TASK_TYPE]
        if len(chosen) != GAMES_PER_TASK_TYPE:
            raise RuntimeError(f"Not enough untouched games for {task_type}: {len(chosen)}")
        selected.extend(chosen)
        selection_by_task[task_type] = chosen

    if len(selected) != 24 or len(selected) != len(set(selected)) or set(selected) & excluded:
        raise RuntimeError("P4-N split is not a 24-game zero-overlap window")
    return selected, selection_by_task


def main() -> None:
    required = [
        P4M_SCRIPT,
        P4M_CANDIDATE,
        P4M_QUEUE,
        P4M_REPORT,
        P4M_STAGING_MANIFEST,
        MISTRAL_BASE,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if OUTPUT_ROOT.exists() or ARCHIVE_BASE.with_suffix(".zip").exists():
        raise RuntimeError("P4-N output already exists; refusing overwrite")

    queue = read_jsonl(P4M_QUEUE)
    if len(queue) != 1 or queue[0]["candidate_id"] != EXPECTED_CANDIDATE_ID:
        raise RuntimeError("Unexpected P4-M candidate queue")
    if queue[0]["candidate_rules_sha256"] != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError("P4-M queue candidate hash mismatch")
    if sha256_file(P4M_CANDIDATE) != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError("P4-M candidate file changed after staging")
    candidate = read_json(P4M_CANDIDATE)
    if candidate["status"] != "staging_candidate":
        raise RuntimeError("P4-M candidate is not in staging_candidate status")

    p4m = load_module(P4M_SCRIPT, "p4n_frozen_p4m")
    base = load_module(MISTRAL_BASE, "p4n_selection_base")
    prior_manifest = read_json(P4M_STAGING_MANIFEST)
    excluded = set(prior_manifest["excluded_prior_game_ids"]) | set(prior_manifest["discovery_game_ids"])
    selected, by_task = enumerate_untouched_ids(base, excluded)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": "JEV_P4N_UNTOUCHED_VALIDATION_MANIFEST_V1",
        "created_before_inference": True,
        "selection_seed": SELECTION_SEED,
        "games_per_task_type": GAMES_PER_TASK_TYPE,
        "selected_game_ids": selected,
        "selection_by_task_type": by_task,
        "excluded_prior_game_ids": sorted(excluded),
        "prior_game_overlap": len(set(selected) & excluded),
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "candidate_rules_sha256": EXPECTED_CANDIDATE_SHA256,
        "rules_frozen_before_selection": True,
        "production_memory_authorized": False,
        "training_label_authorized": False,
        "business_action_authorized": False,
    }
    write_json(MANIFEST_DIR / "p4g_preregistered_split_manifest.json", {
        "schema_version": "JEV_P4M_REPLAY_MANIFEST_V1",
        "created_before_p4m_inference": True,
        "phase": "p4n_untouched_validation_window",
        "selection_seed": SELECTION_SEED,
        "discovery_game_ids": selected,
        "hidden_validation_game_ids": [],
        "excluded_prior_game_ids": sorted(excluded),
        "selection_by_task_type": by_task,
        "hidden_validation_executed": False,
        "production_memory_authorized": False,
        "training_label_authorized": False,
        "business_action_authorized": False,
    })
    write_json(OUTPUT_ROOT / "p4n_untouched_manifest.json", manifest)

    active_types = set(candidate["activated_task_types"])
    replay_report = p4m.run_phase(
        "p4n_untouched_validation_window",
        MANIFEST_DIR,
        REPLAY_DIR,
        OUTPUT_ROOT / "untouched_replay_artifacts",
        active_types,
    )

    effect = replay_report["effect"]
    interface = replay_report["interface_validation"]
    gate_passed = (
        effect["absolute_success_rate_delta"] >= MIN_ABSOLUTE_DELTA
        and len(effect["corrected_games"]) > 0
        and len(effect["regressed_games"]) <= MAX_REGRESSIONS
        and interface["parse_fallbacks"] <= MAX_PARSE_FALLBACKS
        and interface["constrained_decoder_violations"] <= MAX_DECODER_VIOLATIONS
    )

    candidate_snapshot = json.loads(json.dumps(candidate))
    candidate_snapshot["p4n_untouched_validation_gate_passed"] = gate_passed
    candidate_snapshot["status"] = (
        "staging_candidate_extended_validation_passed" if gate_passed else "rejected_after_untouched_validation"
    )
    candidate_snapshot["memory_write_authorized"] = False
    candidate_snapshot["training_label_authorized"] = False
    candidate_snapshot["business_action_authorized"] = False
    write_json(OUTPUT_ROOT / "p4n_candidate_snapshot.json", candidate_snapshot)

    queue_row = {
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "candidate_rules_sha256": EXPECTED_CANDIDATE_SHA256,
        "untouched_validation_gate_passed": gate_passed,
        "review_status": "awaiting_qualified_human_review" if gate_passed else "rejected_by_untouched_gate",
        "memory_write_authorized": False,
        "training_label_authorized": False,
        "business_action_authorized": False,
    }
    write_jsonl(OUTPUT_ROOT / "p4n_human_review_queue.jsonl", [queue_row])

    report = {
        "schema_version": "JEV_P4N_UNTOUCHED_VALIDATION_WINDOW_REPORT_V1",
        "purpose": "Extend the frozen P4-M candidate on a second, larger, zero-overlap validation window",
        "candidate": {
            "candidate_id": EXPECTED_CANDIDATE_ID,
            "candidate_rules_sha256": EXPECTED_CANDIDATE_SHA256,
            "rule_count": len(candidate["activated_rule_ids"]),
            "rules_changed_after_p4m": False,
        },
        "design": {
            "games": len(selected),
            "games_per_task_type": GAMES_PER_TASK_TYPE,
            "prior_game_overlap": 0,
            "selection_seed": SELECTION_SEED,
            "same_model_scaffold_decoder_and_games": True,
            "counterbalanced_arm_order": True,
            "only_arm_difference": "old frozen Memory versus exact frozen P4-M candidate state rules",
            "pre_registered_limits": {
                "minimum_absolute_success_rate_delta": MIN_ABSOLUTE_DELTA,
                "maximum_regressions": MAX_REGRESSIONS,
                "maximum_parse_fallbacks": MAX_PARSE_FALLBACKS,
                "maximum_decoder_violations": MAX_DECODER_VIOLATIONS,
            },
        },
        "old_memory": replay_report["old_memory"],
        "candidate": {
            "candidate_id": EXPECTED_CANDIDATE_ID,
            "candidate_rules_sha256": EXPECTED_CANDIDATE_SHA256,
            "rule_count": len(candidate["activated_rule_ids"]),
            "successes": replay_report["jev_state_controller"]["successes"],
            "games": replay_report["jev_state_controller"]["games"],
            "success_rate": replay_report["jev_state_controller"]["success_rate"],
            "rules_changed_after_p4m": False,
        },
        "effect": effect,
        "per_task_type": replay_report["per_task_type"],
        "interface_validation": interface,
        "decision": {
            "untouched_validation_gate_passed": gate_passed,
            "candidate_status": candidate_snapshot["status"],
            "qualified_human_review_required": gate_passed,
            "production_memory_authorized": False,
            "training_label_authorized": False,
            "business_action_authorized": False,
            "interpretation": (
                "The unchanged P4-M candidate generalized to a second untouched window and remains queued for qualified human review."
                if gate_passed
                else "The unchanged P4-M candidate failed the second untouched window and is rejected."
            ),
        },
        "runtime": {
            "model_id": read_json(P4M_REPORT)["runtime"]["model_id"],
            "model_revision": read_json(P4M_REPORT)["runtime"]["model_revision"],
            "selection_seed": SELECTION_SEED,
        },
    }
    write_json(OUTPUT_ROOT / "p4n_report.json", report)
    shutil.copy2(Path(__file__), OUTPUT_ROOT / Path(__file__).name)
    (OUTPUT_ROOT / "README.md").write_text(
        "# JEV P4-N untouched validation window\n\n"
        "The exact frozen P4-M candidate is evaluated on 24 new ALFWorld games. "
        "Passing preserves a human-review staging candidate only; every production authorization remains false.\n",
        encoding="utf-8",
    )
    archive = shutil.make_archive(str(ARCHIVE_BASE), "zip", root_dir=OUTPUT_ROOT.parent, base_dir=OUTPUT_ROOT.name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("P4-N archive:", archive)


if __name__ == "__main__":
    main()
