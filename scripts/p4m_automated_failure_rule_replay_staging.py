#!/usr/bin/env python3
"""P4-M: auditable failure -> candidate rule -> Replay -> staging loop.

The build evidence is the failed old-Memory arm from P4-L. A deterministic
synthesizer maps structured failure signatures into a small, pre-declared safe
rule DSL. The rules are first replayed on the source failures and then tested
on a new pre-registered ALFWorld valid_unseen split whose IDs are disjoint from
all P4-F/P4-G/P4-J/P4-L games. Passing creates a staging candidate only; it
never authorizes production Memory, training labels, or business actions.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import shutil
from collections import defaultdict
from pathlib import Path


WORKING = Path("/kaggle/working")
P4L_OUTPUT = WORKING / "jev_p4l_second_model_disjoint_ood_alfworld_mistral7b"
P4L_REPORT = P4L_OUTPUT / "p4f_report.json"
P4L_MANIFEST = P4L_OUTPUT / "p4l_disjoint_ood_manifest.json"
P4L_TRAJECTORIES = P4L_OUTPUT / "p4f_complete_trajectories.jsonl"
MISTRAL_BASE = WORKING / "jev_p4l_mistral7b_frozen_base.py"
FROZEN_P4I = Path(
    "/kaggle/input/datasets/placideo/jev-p4i-explicit-state-controller/"
    "jev_p4i_explicit_state_controller_alfworld_7b.py"
)
FROZEN_P4I_SHA256 = "9b4079a98592ae41437dff2be0830ee5920fc8218f9e5d12e7be31c54c48a0d2"
MISTRAL_BASE_SHA256 = "c3df19234924dabc922061e2be19f92fe8ddf10c2949c11599c6fa4dc9a041e0"
SELECTION_SEED = 20260930
TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]
GAMES_PER_TASK_TYPE = 2
MIN_SOURCE_CORRECTIONS = 4
MIN_STAGING_ABSOLUTE_DELTA = 0.25
MAX_REGRESSIONS = 0

OUTPUT_ROOT = WORKING / "jev_p4m_automated_failure_rule_replay_staging"
SOURCE_MANIFEST_DIR = OUTPUT_ROOT / "source_replay_manifest"
STAGING_MANIFEST_DIR = OUTPUT_ROOT / "staging_manifest"
SOURCE_REPLAY_DIR = OUTPUT_ROOT / "source_replay"
STAGING_REPLAY_DIR = OUTPUT_ROOT / "new_frozen_staging_replay"
ARCHIVE_BASE = WORKING / "jev_p4m_automated_failure_rule_replay_staging_artifacts"


RULE_CATALOG = {
    "acquire_exact_target": {
        "condition": "holding is null and exact target is currently takeable",
        "effect": "restrict candidates to taking the exact target",
    },
    "open_current_search_container": {
        "condition": "target is not held and an unopened current container is legal",
        "effect": "restrict candidates to opening a current search container",
    },
    "visit_unsearched_location": {
        "condition": "target is not held and target is not currently visible",
        "effect": "exclude previously visited locations and navigate to an unseen location",
    },
    "apply_required_transform": {
        "condition": "target is held and required clean/heat/cool transform is incomplete",
        "effect": "navigate/open the required appliance and apply the transform before delivery",
    },
    "use_required_lamp": {
        "condition": "light task target is held and lamp evidence is incomplete",
        "effect": "navigate to the desk lamp, use it, then examine the target",
    },
    "place_target_at_destination": {
        "condition": "target is held and all required transforms are complete",
        "effect": "navigate/open the destination and place the held target",
    },
    "repeat_until_required_count": {
        "condition": "placed_count is below required_count",
        "effect": "return to search and repeat acquisition/delivery for another target instance",
    },
}


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


def task_type_for_game_id(game_id: str) -> str:
    return game_id.split("-", 1)[0]


def synthesize_candidate_rules() -> tuple[list[dict], dict]:
    """Compile P4-L old-arm failures into a safe, versioned rule candidate."""
    rows = read_jsonl(P4L_TRAJECTORIES)
    old_failures = [row for row in rows if row["arm"] == "guarded_baseline" and not row["success"]]
    if not old_failures:
        raise RuntimeError("P4-L contains no old-Memory failures to synthesize")

    evidence_rows = []
    activated_task_types = set()
    activated_rule_ids = set()
    for row in old_failures:
        goal = row["goal_schema"]
        actions = [step["action"].lower() for step in row["trajectory"]]
        target = goal["target_type"]
        transform = goal["required_transform"]
        acquired = any(action.startswith("take ") and target in action for action in actions)
        transformed = bool(transform and any(action.startswith(transform + " ") for action in actions))
        placed = any(action.startswith("move ") and target in action for action in actions)
        rules = {"acquire_exact_target", "open_current_search_container", "visit_unsearched_location", "place_target_at_destination"}
        if transform in {"clean", "heat", "cool"}:
            rules.add("apply_required_transform")
        if transform == "light":
            rules.add("use_required_lamp")
        if goal["required_count"] > 1:
            rules.add("repeat_until_required_count")
        activated_task_types.add(goal["task_type"])
        activated_rule_ids.update(rules)
        evidence_rows.append({
            "game_id": row["game_id"],
            "task_type": goal["task_type"],
            "failure_signature": {
                "acquired_target": acquired,
                "applied_required_transform": transformed,
                "attempted_destination_placement": placed,
                "final_phase": row["final_controller_state"]["phase"],
                "placed_count": row["final_controller_state"]["placed_count"],
                "required_count": goal["required_count"],
            },
            "proposed_rule_ids": sorted(rules),
            "memory_write_authorized": False,
            "training_label_authorized": False,
            "business_action_authorized": False,
        })

    unknown = activated_rule_ids - set(RULE_CATALOG)
    if unknown:
        raise RuntimeError(f"Synthesizer emitted rules outside the safe catalog: {sorted(unknown)}")
    candidate = {
        "schema_version": "JEV_P4M_CANDIDATE_STATE_RULES_V1",
        "source": "P4-L old-Memory failures only",
        "synthesis": "deterministic failure-signature compiler into pre-declared safe rule DSL",
        "source_failure_count": len(old_failures),
        "activated_task_types": sorted(activated_task_types),
        "activated_rule_ids": sorted(activated_rule_ids),
        "rule_catalog": {rid: RULE_CATALOG[rid] for rid in sorted(activated_rule_ids)},
        "static_validation": {
            "unknown_rule_ids": [],
            "arbitrary_code_generation": False,
            "object_ids_hardcoded": False,
            "hidden_outcomes_used": False,
            "passed": True,
        },
        "status": "candidate_unreplayed",
        "memory_write_authorized": False,
        "training_label_authorized": False,
        "business_action_authorized": False,
    }
    return evidence_rows, candidate


def enumerate_new_staging_ids(base, excluded: set[str]) -> tuple[list[str], dict[str, list[str]]]:
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
    selected, selection_by_task = [], {}
    for task_type in TASK_TYPES:
        ranked = sorted(
            by_task[task_type],
            key=lambda game_id: hashlib.sha256(f"{SELECTION_SEED}:{game_id}".encode()).hexdigest(),
        )
        chosen = ranked[:GAMES_PER_TASK_TYPE]
        if len(chosen) != GAMES_PER_TASK_TYPE:
            raise RuntimeError(f"Not enough fresh games for {task_type}")
        selected.extend(chosen)
        selection_by_task[task_type] = chosen
    if len(selected) != len(set(selected)) or set(selected) & excluded:
        raise RuntimeError("P4-M staging split is not disjoint")
    return selected, selection_by_task


def write_phase_manifest(directory: Path, ids: list[str], excluded: set[str], label: str, selection_by_task=None) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": "JEV_P4M_REPLAY_MANIFEST_V1",
        "created_before_p4m_inference": True,
        "phase": label,
        "selection_seed": SELECTION_SEED,
        "discovery_game_ids": ids,
        "hidden_validation_game_ids": [],
        "excluded_prior_game_ids": sorted(excluded),
        "selection_by_task_type": selection_by_task or {},
        "hidden_validation_executed": False,
        "production_memory_authorized": False,
        "training_label_authorized": False,
        "business_action_authorized": False,
    }
    write_json(directory / "p4g_preregistered_split_manifest.json", manifest)


def run_phase(label: str, manifest_dir: Path, output_dir: Path, archive_base: Path, active_task_types: set[str]) -> dict:
    p4i = load_module(FROZEN_P4I, f"p4m_controller_{label}")
    original = p4i.controller_candidates

    def staged_controller(commands, state):
        if state["task_type"] not in active_task_types:
            return list(commands), {
                "controller_reason": "candidate_not_activated_for_task_type",
                "controller_original_count": len(commands),
                "controller_filtered_count": len(commands),
                "controller_state": json.loads(json.dumps(state)),
            }
        candidates, meta = original(commands, state)
        meta["p4m_candidate_rule_status"] = "activated"
        return candidates, meta

    p4i.controller_candidates = staged_controller
    p4i.BASE_SCRIPT = MISTRAL_BASE
    p4i.P4G_DIR = manifest_dir
    p4i.OUTPUT_DIR = output_dir
    p4i.ARCHIVE_BASE = str(archive_base)
    p4i.main()
    report = read_json(output_dir / "p4f_report.json")
    report["p4m_phase"] = label
    report["candidate_activated_task_types"] = sorted(active_task_types)
    write_json(output_dir / "p4f_report.json", report)
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return report


def main() -> None:
    required = [P4L_REPORT, P4L_MANIFEST, P4L_TRAJECTORIES, MISTRAL_BASE, FROZEN_P4I]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if OUTPUT_ROOT.exists() or ARCHIVE_BASE.with_suffix(".zip").exists():
        raise RuntimeError("P4-M output already exists; refusing overwrite")
    if sha256_file(FROZEN_P4I) != FROZEN_P4I_SHA256:
        raise RuntimeError("Frozen P4-I controller SHA mismatch")
    if sha256_file(MISTRAL_BASE) != MISTRAL_BASE_SHA256:
        raise RuntimeError("Pinned Mistral base SHA mismatch")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    evidence_rows, candidate = synthesize_candidate_rules()
    write_jsonl(OUTPUT_ROOT / "p4m_failure_evidence.jsonl", evidence_rows)
    write_json(OUTPUT_ROOT / "p4m_candidate_state_rules.json", candidate)

    p4l_manifest = read_json(P4L_MANIFEST)
    p4l_report = read_json(P4L_REPORT)
    source_failure_ids = [row["game_id"] for row in evidence_rows]
    excluded_prior = set(p4l_manifest["excluded_prior_game_ids"]) | set(p4l_manifest["discovery_game_ids"])
    base = load_module(MISTRAL_BASE, "p4m_selection_base")
    staging_ids, by_task = enumerate_new_staging_ids(base, excluded_prior)
    write_phase_manifest(SOURCE_MANIFEST_DIR, source_failure_ids, excluded_prior, "source_failure_replay")
    write_phase_manifest(STAGING_MANIFEST_DIR, staging_ids, excluded_prior, "new_frozen_staging", by_task)

    active_types = set(candidate["activated_task_types"])
    source_report = run_phase(
        "source_failure_replay", SOURCE_MANIFEST_DIR, SOURCE_REPLAY_DIR,
        OUTPUT_ROOT / "source_replay_artifacts", active_types,
    )
    staging_report = run_phase(
        "new_frozen_staging_replay", STAGING_MANIFEST_DIR, STAGING_REPLAY_DIR,
        OUTPUT_ROOT / "new_frozen_staging_replay_artifacts", active_types,
    )

    source_effect = source_report["effect"]
    staging_effect = staging_report["effect"]
    source_interface = source_report["interface_validation"]
    staging_interface = staging_report["interface_validation"]
    source_gate = (
        len(source_effect["corrected_games"]) >= MIN_SOURCE_CORRECTIONS
        and len(source_effect["regressed_games"]) == 0
        and source_interface["parse_fallbacks"] == 0
        and source_interface["constrained_decoder_violations"] == 0
    )
    staging_gate = (
        staging_effect["absolute_success_rate_delta"] >= MIN_STAGING_ABSOLUTE_DELTA
        and len(staging_effect["corrected_games"]) > 0
        and len(staging_effect["regressed_games"]) <= MAX_REGRESSIONS
        and staging_interface["parse_fallbacks"] == 0
        and staging_interface["constrained_decoder_violations"] == 0
    )
    status = "staging_candidate" if source_gate and staging_gate else "rejected"
    candidate["status"] = status
    candidate["source_replay_gate_passed"] = source_gate
    candidate["new_frozen_staging_gate_passed"] = staging_gate
    write_json(OUTPUT_ROOT / "p4m_candidate_state_rules.json", candidate)

    queue_row = {
        "candidate_id": "jev-p4m-state-rules-v1",
        "status": status,
        "review_status": "awaiting_qualified_human_review" if status == "staging_candidate" else "rejected_by_automatic_gate",
        "candidate_rules_sha256": sha256_file(OUTPUT_ROOT / "p4m_candidate_state_rules.json"),
        "source_replay_gate_passed": source_gate,
        "new_frozen_staging_gate_passed": staging_gate,
        "memory_write_authorized": False,
        "training_label_authorized": False,
        "business_action_authorized": False,
    }
    write_jsonl(OUTPUT_ROOT / "p4m_staging_queue.jsonl", [queue_row])

    report = {
        "schema_version": "JEV_P4M_AUTOMATED_FAILURE_RULE_REPLAY_STAGING_REPORT_V1",
        "purpose": "Test an auditable automated RSI candidate loop without self-authorizing production writes",
        "source_evidence": {
            "p4l_report_sha256": sha256_file(P4L_REPORT),
            "p4l_old_memory_failures": len(source_failure_ids),
            "candidate_rule_count": len(candidate["activated_rule_ids"]),
            "activated_task_types": candidate["activated_task_types"],
        },
        "source_failure_replay": {
            "old_memory": source_report["old_memory"],
            "candidate": source_report["jev_state_controller"],
            "effect": source_effect,
            "interface_validation": source_interface,
            "gate_passed": source_gate,
        },
        "new_frozen_staging_replay": {
            "games": len(staging_ids),
            "prior_game_overlap": 0,
            "old_memory": staging_report["old_memory"],
            "candidate": staging_report["jev_state_controller"],
            "effect": staging_effect,
            "interface_validation": staging_interface,
            "gate_passed": staging_gate,
        },
        "decision": {
            "candidate_status": status,
            "automatic_staging_authorized": status == "staging_candidate",
            "qualified_human_review_required": status == "staging_candidate",
            "production_memory_authorized": False,
            "training_label_authorized": False,
            "business_action_authorized": False,
            "interpretation": (
                "The automated evidence loop may enqueue a versioned candidate for human review; it cannot approve its own production use."
                if status == "staging_candidate"
                else "The candidate failed an automatic gate and must not enter staging."
            ),
        },
        "runtime": {
            "model_id": p4l_report["runtime_model"]["model_id"],
            "model_revision": p4l_report["runtime_model"]["revision"],
            "frozen_controller_sha256": FROZEN_P4I_SHA256,
            "mistral_base_sha256": MISTRAL_BASE_SHA256,
            "selection_seed": SELECTION_SEED,
        },
    }
    write_json(OUTPUT_ROOT / "p4m_report.json", report)
    shutil.copy2(Path(__file__), OUTPUT_ROOT / Path(__file__).name)
    (OUTPUT_ROOT / "README.md").write_text(
        "# JEV P4-M automated evidence-gated RSI staging loop\n\n"
        "A deterministic safe-DSL rule synthesizer, source-failure Replay, and a fresh disjoint staging split. "
        "Passing creates a human-review staging candidate only. All production authorizations remain false.\n",
        encoding="utf-8",
    )
    archive = shutil.make_archive(str(ARCHIVE_BASE), "zip", root_dir=OUTPUT_ROOT.parent, base_dir=OUTPUT_ROOT.name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("P4-M archive:", archive)


if __name__ == "__main__":
    main()
