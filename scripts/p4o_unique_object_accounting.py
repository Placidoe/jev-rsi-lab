#!/usr/bin/env python3
"""P4-O: failure-derived unique-object accounting candidate for JEV RSI.

The sole candidate change is an auditable state rule for multi-object tasks:
count unique object identities placed at the destination and never reacquire an
already placed identity.  The frozen P4-M controller is the baseline.  We replay
the three P4-N pick-two stable failures and a pre-registered, zero-overlap set of
eight new pick-two games.  Passing only creates an experimental staging record;
all production, training, Memory-write, and business authorizations remain false.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path


WORKING = Path("/kaggle/working")
INPUT_ROOT = Path(__file__).resolve().parent
SOURCE_CONTROLLER = INPUT_ROOT / "jev_p4i_explicit_state_controller_alfworld_7b.py"
BASE_SCRIPT = INPUT_ROOT / "jev_p4l_mistral7b_frozen_base.py"
F4_SCRIPT = INPUT_ROOT / "jev_p4f4_shared_public_demo_alfworld_7b.py"
F6_SCRIPT = INPUT_ROOT / "jev_p4f6_constrained_action_alfworld_7b.py"
P4N_MANIFEST = INPUT_ROOT / "p4n_untouched_manifest.json"
P4N_TRAJECTORIES = INPUT_ROOT / "p4n_complete_trajectories.jsonl"

OUTPUT_ROOT = WORKING / "jev_p4o_unique_object_accounting"
MANIFEST_DIR = OUTPUT_ROOT / "preregistration"
BASELINE_DIR = OUTPUT_ROOT / "p4m_frozen_controller"
CANDIDATE_DIR = OUTPUT_ROOT / "p4o_identity_controller"
ARCHIVE_BASE = WORKING / "jev_p4o_unique_object_accounting_artifacts"

CANDIDATE_ID = "jev-p4o-unique-object-accounting-v1"
SELECTION_SEED = 20261002
FRESH_GAMES = 5
MIN_SOURCE_CORRECTIONS = 2
MIN_FRESH_ABSOLUTE_DELTA = 0.25
MAX_REGRESSIONS = 0


def read_json(path: Path):
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


def source_failure_ids() -> list[str]:
    rows = read_jsonl(P4N_TRAJECTORIES)
    by_game: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_game.setdefault(row["game_id"], {})[row["arm"]] = row
    failures = []
    for game_id, arms in sorted(by_game.items()):
        candidate = arms.get("guarded_memory")
        if candidate and candidate["task_type"] == "pick_two_obj_and_place" and not candidate["success"]:
            failures.append(game_id)
    if len(failures) != 3:
        raise RuntimeError(f"Expected exactly three P4-N pick-two failures, found {len(failures)}")
    return failures


def enumerate_fresh_ids(base, excluded: set[str]) -> list[str]:
    from alfworld.agents.environment import get_environment

    engine = get_environment("AlfredTWEnv")(base.config(), train_eval="eval_out_of_distribution")
    try:
        game_files = sorted(engine.game_files)
    finally:
        close = getattr(engine, "close", None)
        if callable(close):
            close()
    candidates = []
    for path in game_files:
        game_id = base.relative_game_id(path)
        if base.task_type_for(path) == "pick_two_obj_and_place" and game_id not in excluded:
            candidates.append(game_id)
    ranked = sorted(candidates, key=lambda gid: hashlib.sha256(f"{SELECTION_SEED}:{gid}".encode()).hexdigest())
    selected = ranked[:FRESH_GAMES]
    if len(selected) != FRESH_GAMES or set(selected) & excluded:
        raise RuntimeError("Could not construct the zero-overlap P4-O fresh validation window")
    return selected


def write_combined_manifest(source_ids: list[str], fresh_ids: list[str], excluded: set[str]) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=False)
    write_json(MANIFEST_DIR / "p4g_preregistered_split_manifest.json", {
        "schema_version": "JEV_P4O_PREREGISTERED_SPLIT_MANIFEST_V1",
        "created_before_inference": True,
        "selection_seed": SELECTION_SEED,
        "phase": "source_failure_plus_zero_overlap_pick_two_validation",
        "discovery_game_ids": source_ids + fresh_ids,
        "source_failure_game_ids": source_ids,
        "fresh_validation_game_ids": fresh_ids,
        "hidden_validation_game_ids": [],
        "excluded_prior_game_ids": sorted(excluded),
        "fresh_prior_overlap": len(set(fresh_ids) & excluded),
        "candidate_id": CANDIDATE_ID,
        "candidate_rule_frozen_before_inference": True,
        "pre_registered_limits": {
            "minimum_source_corrections": MIN_SOURCE_CORRECTIONS,
            "minimum_fresh_absolute_success_rate_delta": MIN_FRESH_ABSOLUTE_DELTA,
            "maximum_regressions": MAX_REGRESSIONS,
            "maximum_parse_fallbacks": 0,
            "maximum_decoder_violations": 0,
        },
        "hidden_validation_executed": False,
        "memory_write_authorized": False,
        "training_label_authorized": False,
        "production_action_authorized": False,
        "business_action_authorized": False,
    })


def object_id_from_take(command: str) -> str | None:
    match = re.match(r"take (.+?) from ", command.lower())
    return match.group(1) if match else None


def install_identity_candidate(p4i) -> None:
    original_initial_state = p4i.initial_state

    def initial_state(goal: dict) -> dict:
        state = original_initial_state(goal)
        state["placed_object_ids"] = []
        state["identity_retake_blocks"] = 0
        state["duplicate_place_attempts"] = 0
        return state

    def controller_candidates(commands: list[str], state: dict):
        commands = list(commands)
        target = state["target_type"]
        destination = state["destination_type"]
        holding = state["holding"]
        transform = state["required_transform"]
        placed = set(state.get("placed_object_ids", []))
        reason = "policy_fallback"

        def select(items: list[str], why: str):
            nonlocal reason
            if items:
                reason = why
                return items, {
                    "controller_reason": reason,
                    "controller_original_count": len(commands),
                    "controller_filtered_count": len(items),
                    "controller_state": json.loads(json.dumps(state)),
                }
            return None

        if holding:
            if transform == "light":
                if not state["lamp_used"]:
                    result = select([c for c in commands if c.lower().startswith("use desklamp")], "use_required_lamp")
                    if result:
                        return result
                    result = select([c for c in commands if c.lower().startswith("go to desk")], "navigate_to_lamp")
                    if result:
                        return result
                result = select([c for c in commands if c.lower().startswith("examine ") and target in c.lower()], "examine_target_under_light")
                if result:
                    return result
            elif transform and not state["transformed_current"]:
                result = select([c for c in commands if c.lower().startswith(transform + " ") and holding in c.lower()], "apply_required_transform")
                if result:
                    return result
                appliance = {"clean": "sinkbasin", "heat": "microwave", "cool": "fridge"}[transform]
                result = select([c for c in commands if c.lower().startswith("open ") and appliance in c.lower()], "open_required_appliance")
                if result:
                    return result
                result = select([c for c in commands if c.lower().startswith("go to ") and appliance in c.lower()], "navigate_to_required_appliance")
                if result:
                    return result
            else:
                result = select([c for c in commands if c.lower().startswith("move ") and holding in c.lower() and destination in c.lower()], "place_unique_target_at_destination")
                if result:
                    return result
                result = select([c for c in commands if c.lower().startswith("open ") and destination in c.lower()], "open_destination")
                if result:
                    return result
                result = select([c for c in commands if c.lower().startswith("go to ") and destination in c.lower()], "navigate_to_destination")
                if result:
                    return result
        else:
            all_target_takes = [c for c in commands if p4i.is_target_command(c, target, "take")]
            new_target_takes = [c for c in all_target_takes if object_id_from_take(c) not in placed]
            state["identity_retake_blocks"] += len(all_target_takes) - len(new_target_takes)
            result = select(new_target_takes, "acquire_unplaced_exact_target")
            if result:
                return result
            result = select([c for c in commands if c.lower().startswith("open ")], "open_current_search_container")
            if result:
                return result
            visited = set(state["visited_locations"])
            unseen = [c for c in commands if c.lower().startswith("go to ") and p4i.command_location(c) not in visited]
            result = select(unseen, "visit_unsearched_location")
            if result:
                return result

        safe = []
        for command in commands:
            lowered = command.lower()
            if lowered.startswith("close "):
                continue
            if lowered.startswith("take "):
                object_id = object_id_from_take(command)
                if not p4i.is_target_command(command, target, "take") or object_id in placed:
                    continue
            safe.append(command)
        safe = safe or commands
        return safe, {
            "controller_reason": reason,
            "controller_original_count": len(commands),
            "controller_filtered_count": len(safe),
            "controller_state": json.loads(json.dumps(state)),
        }

    def advance_state(state: dict, action: str) -> None:
        lowered = action.lower()
        location = p4i.command_location(action)
        if location and location not in state["visited_locations"]:
            state["visited_locations"].append(location)
        if lowered.startswith("open "):
            container = lowered[len("open "):]
            if container not in state["opened_containers"]:
                state["opened_containers"].append(container)
        taken = object_id_from_take(action)
        if taken and state["target_type"] in taken:
            state["holding"] = taken
            state["transformed_current"] = False
            state["phase"] = "transform" if state["required_transform"] not in (None, "light") else "deliver"
        if state["holding"] and lowered.startswith(("clean ", "heat ", "cool ")) and state["holding"] in lowered:
            state["transformed_current"] = True
            state["phase"] = "deliver"
        if lowered.startswith("use desklamp"):
            state["lamp_used"] = True
            state["phase"] = "verify"
        if state["holding"] and lowered.startswith("move ") and state["holding"] in lowered and state["destination_type"] in lowered:
            object_id = state["holding"]
            if object_id in state["placed_object_ids"]:
                state["duplicate_place_attempts"] += 1
            else:
                state["placed_object_ids"].append(object_id)
            state["placed_count"] = len(state["placed_object_ids"])
            state["holding"] = None
            state["transformed_current"] = False
            state["phase"] = "search" if state["placed_count"] < state["required_count"] else "complete"
        state["controller_steps"] += 1

    p4i.initial_state = initial_state
    p4i.controller_candidates = controller_candidates
    p4i.advance_state = advance_state


def run_controller(output_dir: Path, archive_base: Path, candidate: bool) -> dict:
    p4i = load_module(SOURCE_CONTROLLER, "p4o_candidate_module" if candidate else "p4o_baseline_module")
    p4i.BASE_SCRIPT = BASE_SCRIPT
    p4i.F4_SCRIPT = F4_SCRIPT
    p4i.F6_SCRIPT = F6_SCRIPT
    p4i.P4G_DIR = MANIFEST_DIR
    p4i.OUTPUT_DIR = output_dir
    p4i.ARCHIVE_BASE = str(archive_base)
    if candidate:
        install_identity_candidate(p4i)
    p4i.main()
    report = read_json(output_dir / "p4f_report.json")
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return report


def arm_rows(directory: Path) -> dict[str, dict]:
    return {
        row["game_id"]: row
        for row in read_jsonl(directory / "p4f_complete_trajectories.jsonl")
        if row["arm"] == "guarded_memory"
    }


def compare(ids: list[str], baseline: dict[str, dict], candidate: dict[str, dict]) -> dict:
    rows = []
    for game_id in ids:
        old = baseline[game_id]
        new = candidate[game_id]
        transition = (
            "corrected" if not old["success"] and new["success"] else
            "regressed" if old["success"] and not new["success"] else
            "stable_success" if old["success"] else "stable_failure"
        )
        rows.append({
            "game_id": game_id,
            "task_type": old["task_type"],
            "p4m_success": old["success"],
            "p4o_success": new["success"],
            "transition": transition,
            "p4m_steps": old["steps"],
            "p4o_steps": new["steps"],
            "p4m_final_state": old["final_controller_state"],
            "p4o_final_state": new["final_controller_state"],
        })
    old_successes = sum(row["p4m_success"] for row in rows)
    new_successes = sum(row["p4o_success"] for row in rows)
    return {
        "games": len(rows),
        "p4m_successes": old_successes,
        "p4o_successes": new_successes,
        "p4m_success_rate": old_successes / len(rows),
        "p4o_success_rate": new_successes / len(rows),
        "absolute_success_rate_delta": (new_successes - old_successes) / len(rows),
        "corrected_games": [row["game_id"] for row in rows if row["transition"] == "corrected"],
        "regressed_games": [row["game_id"] for row in rows if row["transition"] == "regressed"],
        "stable_failure_games": [row["game_id"] for row in rows if row["transition"] == "stable_failure"],
        "rows": rows,
    }


def main() -> None:
    required = [SOURCE_CONTROLLER, BASE_SCRIPT, F4_SCRIPT, F6_SCRIPT, P4N_MANIFEST, P4N_TRAJECTORIES]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if OUTPUT_ROOT.exists() or ARCHIVE_BASE.with_suffix(".zip").exists():
        raise RuntimeError("P4-O output already exists; refusing overwrite")

    source_ids = source_failure_ids()
    p4n_manifest = read_json(P4N_MANIFEST)
    excluded = set(p4n_manifest["excluded_prior_game_ids"]) | set(p4n_manifest["selected_game_ids"])
    base = load_module(BASE_SCRIPT, "p4o_selection_base")
    fresh_ids = enumerate_fresh_ids(base, excluded)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    write_combined_manifest(source_ids, fresh_ids, excluded)

    candidate_spec = {
        "schema_version": "JEV_P4O_UNIQUE_OBJECT_ACCOUNTING_CANDIDATE_V1",
        "candidate_id": CANDIDATE_ID,
        "source_evidence": "three P4-N pick-two stable failures",
        "failure_mechanism": "the controller can re-take an already placed identity and count the same physical object twice",
        "rule": {
            "id": "unique_object_identity_count",
            "state_additions": ["placed_object_ids", "identity_retake_blocks", "duplicate_place_attempts"],
            "condition": "required_count > 1",
            "effect": "count unique placed identities, exclude already placed identities from acquisition, and complete only when the unique cardinality reaches required_count",
            "arbitrary_code_generation": False,
            "object_ids_hardcoded": False,
            "hidden_outcomes_used": False,
        },
        "status": "experimental_candidate_unreplayed",
        "memory_write_authorized": False,
        "training_label_authorized": False,
        "production_action_authorized": False,
        "business_action_authorized": False,
    }
    write_json(OUTPUT_ROOT / "p4o_candidate_rule.json", candidate_spec)
    candidate_rule_sha = sha256_file(OUTPUT_ROOT / "p4o_candidate_rule.json")

    baseline_report = run_controller(BASELINE_DIR, OUTPUT_ROOT / "p4m_frozen_controller_artifacts", candidate=False)
    candidate_report = run_controller(CANDIDATE_DIR, OUTPUT_ROOT / "p4o_identity_controller_artifacts", candidate=True)
    baseline = arm_rows(BASELINE_DIR)
    candidate = arm_rows(CANDIDATE_DIR)
    source = compare(source_ids, baseline, candidate)
    fresh = compare(fresh_ids, baseline, candidate)
    all_rows = source["rows"] + fresh["rows"]
    write_jsonl(OUTPUT_ROOT / "p4o_paired_cases.jsonl", all_rows)

    interface = {
        "baseline_parse_fallbacks": sum(baseline[gid]["parse_fallbacks"] for gid in source_ids + fresh_ids),
        "candidate_parse_fallbacks": sum(candidate[gid]["parse_fallbacks"] for gid in source_ids + fresh_ids),
        "baseline_decoder_violations": sum(step.get("parse_method") != "prefix_constrained_legal_command" for gid in source_ids + fresh_ids for step in baseline[gid]["trajectory"]),
        "candidate_decoder_violations": sum(step.get("parse_method") != "prefix_constrained_legal_command" for gid in source_ids + fresh_ids for step in candidate[gid]["trajectory"]),
        "identity_retake_blocks": sum(row["final_controller_state"].get("identity_retake_blocks", 0) for row in candidate.values()),
        "duplicate_place_attempts": sum(row["final_controller_state"].get("duplicate_place_attempts", 0) for row in candidate.values()),
    }
    source_gate = (
        len(source["corrected_games"]) >= MIN_SOURCE_CORRECTIONS
        and len(source["regressed_games"]) <= MAX_REGRESSIONS
    )
    fresh_gate = (
        fresh["absolute_success_rate_delta"] >= MIN_FRESH_ABSOLUTE_DELTA
        and len(fresh["corrected_games"]) > 0
        and len(fresh["regressed_games"]) <= MAX_REGRESSIONS
    )
    interface_gate = all(value == 0 for key, value in interface.items() if "fallbacks" in key or "violations" in key)
    passed = source_gate and fresh_gate and interface_gate
    status = "experimental_staging_candidate" if passed else "revise_or_reject"
    candidate_spec["status"] = status
    candidate_spec["source_gate_passed"] = source_gate
    candidate_spec["fresh_validation_gate_passed"] = fresh_gate
    candidate_spec["interface_gate_passed"] = interface_gate
    write_json(OUTPUT_ROOT / "p4o_candidate_rule.json", candidate_spec)

    report = {
        "schema_version": "JEV_P4O_UNIQUE_OBJECT_ACCOUNTING_REPORT_V1",
        "purpose": "Test whether failure-derived object-identity Memory repairs false progress in multi-object long-horizon tasks",
        "candidate": {
            "candidate_id": CANDIDATE_ID,
            "pre_inference_candidate_rule_sha256": candidate_rule_sha,
            "single_rule_change": True,
            "baseline": "exact frozen P4-M/P4-N controller",
            "candidate_change": "unique identity set, retake exclusion, and cardinality-based phase completion",
        },
        "design": {
            "source_failures": len(source_ids),
            "fresh_zero_overlap_games": len(fresh_ids),
            "fresh_prior_overlap": 0,
            "same_model_revision_memory_scaffold_decoder_and_games": True,
            "selection_seed": SELECTION_SEED,
            "model_id": baseline_report.get("runtime_model", {}).get("model_id", "mistralai/Mistral-7B-Instruct-v0.3"),
        },
        "source_failure_replay": {key: value for key, value in source.items() if key != "rows"},
        "fresh_zero_overlap_validation": {key: value for key, value in fresh.items() if key != "rows"},
        "interface_validation": interface,
        "decision": {
            "source_gate_passed": source_gate,
            "fresh_validation_gate_passed": fresh_gate,
            "interface_gate_passed": interface_gate,
            "candidate_status": status,
            "qualified_human_review_required": passed,
            "memory_write_authorized": False,
            "training_label_authorized": False,
            "production_action_authorized": False,
            "business_action_authorized": False,
            "interpretation": (
                "The identity-aware state rule repaired the observed failure mechanism and generalized to a fresh zero-overlap window; it may enter human-review staging only."
                if passed else
                "The identity-aware state rule did not satisfy every pre-registered gate and must be revised or rejected."
            ),
        },
    }
    write_json(OUTPUT_ROOT / "p4o_report.json", report)
    write_jsonl(OUTPUT_ROOT / "p4o_human_review_queue.jsonl", [{
        "candidate_id": CANDIDATE_ID,
        "candidate_status": status,
        "review_status": "awaiting_qualified_human_review" if passed else "not_eligible",
        "memory_write_authorized": False,
        "training_label_authorized": False,
        "production_action_authorized": False,
        "business_action_authorized": False,
    }])
    shutil.copy2(Path(__file__), OUTPUT_ROOT / Path(__file__).name)
    (OUTPUT_ROOT / "README.md").write_text(
        "# JEV P4-O unique-object accounting\n\n"
        "Failure-derived identity Memory candidate tested against the exact frozen P4-M controller on source failures and a fresh zero-overlap window. No automatic production authorization.\n",
        encoding="utf-8",
    )
    archive = shutil.make_archive(str(ARCHIVE_BASE), "zip", root_dir=OUTPUT_ROOT.parent, base_dir=OUTPUT_ROOT.name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("P4-O archive:", archive)


if __name__ == "__main__":
    main()
