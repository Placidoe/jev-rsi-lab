#!/usr/bin/env python3
"""P4-P: frozen P4-O cross-split transfer audit on ALFWorld valid_seen.

This experiment is deliberately evaluation-only.  It freezes the exact P4-O
unique-object accounting rule, selects pick-two tasks from a new public split
before inference, and compares the frozen P4-M controller with the P4-O
candidate.  No outcome can authorize Memory, training, production, or business
actions.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path


WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
OUTPUT_ROOT = WORKING / "jev_p4p_cross_split_transfer"
MANIFEST_DIR = OUTPUT_ROOT / "preregistration"
BASELINE_DIR = OUTPUT_ROOT / "p4m_frozen_controller"
CANDIDATE_DIR = OUTPUT_ROOT / "p4o_frozen_identity_controller"
PATCHED_DIR = OUTPUT_ROOT / "frozen_runtime"
ARCHIVE_BASE = WORKING / "jev_p4p_cross_split_transfer_artifacts"

CANDIDATE_ID = "jev-p4o-unique-object-accounting-v1"
EXPERIMENT_ID = "jev-p4p-cross-split-transfer-v1"
P4O_PREINFERENCE_RULE_SHA256 = "15f2f21ff8b64dcc7e494226992798b6f1d3ee2f7971e3a1e9c3f1b1ddce1f06"
SELECTION_SEED = 20261003
REQUESTED_GAMES = 12
MIN_BASELINE_FAILURES = 2
MIN_CORRECTIONS = 2
MIN_NET_CORRECTIONS = 2
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


def locate_one(filename: str, prefer_fragment: str | None = None) -> Path:
    candidates = sorted(INPUT.rglob(filename))
    if prefer_fragment:
        preferred = [p for p in candidates if prefer_fragment in str(p)]
        if preferred:
            candidates = preferred
    if not candidates:
        raise FileNotFoundError(f"Could not find {filename} below /kaggle/input")
    return candidates[0]


def patch_base_for_valid_seen(source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8")
    old_relative = '''def relative_game_id(game_file: str) -> str:\n    marker = "/valid_unseen/"\n    return game_file.split(marker, 1)[-1] if marker in game_file else Path(game_file).name\n'''
    new_relative = '''def relative_game_id(game_file: str) -> str:\n    for marker in ("/valid_seen/", "/valid_unseen/"):\n        if marker in game_file:\n            return game_file.split(marker, 1)[-1]\n    return str(Path(game_file))\n'''
    if old_relative not in text:
        raise RuntimeError("Frozen base relative_game_id signature changed; refusing an unreviewed patch")
    text = text.replace(old_relative, new_relative, 1)
    old_split = 'get_environment("AlfredTWEnv")(config(), train_eval="eval_out_of_distribution")'
    new_split = 'get_environment("AlfredTWEnv")(config(), train_eval="eval_in_distribution")'
    if text.count(old_split) != 1:
        raise RuntimeError("Frozen base environment signature changed; refusing an unreviewed patch")
    text = text.replace(old_split, new_split, 1)
    destination.write_text(text, encoding="utf-8")


def enumerate_valid_seen_pick_two(base) -> list[str]:
    from alfworld.agents.environment import get_environment

    engine = get_environment("AlfredTWEnv")(base.config(), train_eval="eval_in_distribution")
    try:
        game_files = sorted(engine.game_files)
    finally:
        close = getattr(engine, "close", None)
        if callable(close):
            close()
    ids = [
        base.relative_game_id(path)
        for path in game_files
        if base.task_type_for(path) == "pick_two_obj_and_place"
    ]
    if len(ids) < REQUESTED_GAMES:
        raise RuntimeError(f"Need {REQUESTED_GAMES} valid_seen pick-two games, found {len(ids)}")
    ranked = sorted(ids, key=lambda gid: hashlib.sha256(f"{SELECTION_SEED}:{gid}".encode()).hexdigest())
    return ranked[:REQUESTED_GAMES]


def install_frozen_p4o_candidate(p4i) -> None:
    p4o_source = locate_one("jev_p4o_unique_object_accounting.py", "p4o")
    p4o = load_module(p4o_source, "p4p_frozen_p4o_rule")
    p4o.install_identity_candidate(p4i)


def run_controller(source_controller: Path, base_script: Path, f4_script: Path, f6_script: Path,
                   output_dir: Path, archive_base: Path, candidate: bool) -> dict:
    p4i = load_module(source_controller, "p4p_candidate_module" if candidate else "p4p_baseline_module")
    p4i.BASE_SCRIPT = base_script
    p4i.F4_SCRIPT = f4_script
    p4i.F6_SCRIPT = f6_script
    p4i.P4G_DIR = MANIFEST_DIR
    p4i.OUTPUT_DIR = output_dir
    p4i.ARCHIVE_BASE = str(archive_base)
    if candidate:
        install_frozen_p4o_candidate(p4i)
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


def paired_rows(ids: list[str], baseline: dict[str, dict], candidate: dict[str, dict]) -> list[dict]:
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
            "step_delta": new["steps"] - old["steps"],
            "p4m_final_state": old["final_controller_state"],
            "p4o_final_state": new["final_controller_state"],
            "memory_write_authorized": False,
            "training_label_authorized": False,
            "production_action_authorized": False,
            "business_action_authorized": False,
        })
    return rows


def main() -> None:
    if OUTPUT_ROOT.exists() or ARCHIVE_BASE.with_suffix(".zip").exists():
        raise RuntimeError("P4-P output already exists; refusing overwrite")

    source_controller = locate_one("jev_p4i_explicit_state_controller_alfworld_7b.py", "p4o")
    source_base = locate_one("jev_p4l_mistral7b_frozen_base.py", "p4o")
    f4_script = locate_one("jev_p4f4_shared_public_demo_alfworld_7b.py", "p4o")
    f6_script = locate_one("jev_p4f6_constrained_action_alfworld_7b.py", "p4o")
    p4o_source = locate_one("jev_p4o_unique_object_accounting.py", "p4o")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=False)
    PATCHED_DIR.mkdir(parents=True, exist_ok=False)
    patched_base = PATCHED_DIR / "jev_p4p_valid_seen_frozen_base.py"
    patch_base_for_valid_seen(source_base, patched_base)
    base = load_module(patched_base, "p4p_selection_base")
    selected_ids = enumerate_valid_seen_pick_two(base)

    frozen_rule_spec = {
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
        "source_p4o_pre_inference_rule_sha256": P4O_PREINFERENCE_RULE_SHA256,
        "memory_write_authorized": False,
        "training_label_authorized": False,
        "production_action_authorized": False,
        "business_action_authorized": False,
    }
    write_json(OUTPUT_ROOT / "frozen_p4o_candidate_rule.json", frozen_rule_spec)
    manifest = {
        "schema_version": "JEV_P4P_CROSS_SPLIT_PREREGISTRATION_V1",
        "experiment_id": EXPERIMENT_ID,
        "created_before_inference": True,
        "candidate_id": CANDIDATE_ID,
        "candidate_rule_frozen_before_inference": True,
        "source_split": "ALFWorld valid_unseen / eval_out_of_distribution",
        "target_split": "ALFWorld valid_seen / eval_in_distribution",
        "selection_seed": SELECTION_SEED,
        "selection_method": "SHA256 rank of all valid_seen pick_two_obj_and_place game IDs",
        "requested_games": REQUESTED_GAMES,
        "selected_game_ids": selected_ids,
        "prior_split_overlap_by_construction": 0,
        "frozen_p4o_pre_inference_candidate_rule_sha256": P4O_PREINFERENCE_RULE_SHA256,
        "attached_p4o_implementation_sha256": sha256_file(p4o_source),
        "pre_registered_limits": {
            "minimum_baseline_failures_for_diagnostic_power": MIN_BASELINE_FAILURES,
            "minimum_corrected_games": MIN_CORRECTIONS,
            "minimum_net_corrections": MIN_NET_CORRECTIONS,
            "maximum_regressions": MAX_REGRESSIONS,
            "maximum_parse_fallbacks": 0,
            "maximum_decoder_violations": 0,
        },
        "outcomes_hidden_at_registration": True,
        "memory_write_authorized": False,
        "training_label_authorized": False,
        "production_action_authorized": False,
        "business_action_authorized": False,
        "hidden_validation_game_ids": [],
        "hidden_validation_executed": False,
        "discovery_game_ids": selected_ids,
    }
    write_json(MANIFEST_DIR / "p4g_preregistered_split_manifest.json", manifest)
    shutil.copy2(p4o_source, OUTPUT_ROOT / "frozen_p4o_implementation.py")

    baseline_report = run_controller(
        source_controller, patched_base, f4_script, f6_script,
        BASELINE_DIR, OUTPUT_ROOT / "p4m_cross_split_artifacts", candidate=False,
    )
    candidate_report = run_controller(
        source_controller, patched_base, f4_script, f6_script,
        CANDIDATE_DIR, OUTPUT_ROOT / "p4o_cross_split_artifacts", candidate=True,
    )
    baseline = arm_rows(BASELINE_DIR)
    candidate = arm_rows(CANDIDATE_DIR)
    rows = paired_rows(selected_ids, baseline, candidate)
    write_jsonl(OUTPUT_ROOT / "p4p_paired_cases.jsonl", rows)

    baseline_successes = sum(row["p4m_success"] for row in rows)
    candidate_successes = sum(row["p4o_success"] for row in rows)
    baseline_failures = len(rows) - baseline_successes
    corrected = [row["game_id"] for row in rows if row["transition"] == "corrected"]
    regressed = [row["game_id"] for row in rows if row["transition"] == "regressed"]
    stable_failures = [row["game_id"] for row in rows if row["transition"] == "stable_failure"]
    net_corrections = len(corrected) - len(regressed)
    joint_successes = [row for row in rows if row["p4m_success"] and row["p4o_success"]]
    mean_old_steps = (sum(row["p4m_steps"] for row in joint_successes) / len(joint_successes)) if joint_successes else None
    mean_new_steps = (sum(row["p4o_steps"] for row in joint_successes) / len(joint_successes)) if joint_successes else None
    efficiency_delta = ((mean_new_steps - mean_old_steps) / mean_old_steps) if mean_old_steps else None

    interface = {
        "baseline_parse_fallbacks": sum(baseline[gid]["parse_fallbacks"] for gid in selected_ids),
        "candidate_parse_fallbacks": sum(candidate[gid]["parse_fallbacks"] for gid in selected_ids),
        "baseline_decoder_violations": sum(
            step.get("parse_method") != "prefix_constrained_legal_command"
            for gid in selected_ids for step in baseline[gid]["trajectory"]
        ),
        "candidate_decoder_violations": sum(
            step.get("parse_method") != "prefix_constrained_legal_command"
            for gid in selected_ids for step in candidate[gid]["trajectory"]
        ),
        "identity_retake_blocks": sum(candidate[gid]["final_controller_state"].get("identity_retake_blocks", 0) for gid in selected_ids),
        "duplicate_place_attempts": sum(candidate[gid]["final_controller_state"].get("duplicate_place_attempts", 0) for gid in selected_ids),
    }
    diagnostic_power = baseline_failures >= MIN_BASELINE_FAILURES
    effect_gate = len(corrected) >= MIN_CORRECTIONS and net_corrections >= MIN_NET_CORRECTIONS and len(regressed) <= MAX_REGRESSIONS
    interface_gate = all(
        value == 0 for key, value in interface.items()
        if "fallbacks" in key or "violations" in key or "duplicate_place_attempts" in key
    )
    passed = diagnostic_power and effect_gate and interface_gate
    if passed:
        status = "cross_split_replication_passed_pending_human_review"
    elif not diagnostic_power:
        status = "inconclusive_insufficient_baseline_failures"
    else:
        status = "cross_split_replication_failed"

    report = {
        "schema_version": "JEV_P4P_CROSS_SPLIT_TRANSFER_REPORT_V1",
        "purpose": "Test whether the frozen P4-O identity rule transfers to a new ALFWorld distribution without outcome-driven edits",
        "candidate": {
            "candidate_id": CANDIDATE_ID,
            "rule_changed_after_p4o": False,
            "frozen_source_rule_hash": manifest["frozen_p4o_pre_inference_candidate_rule_sha256"],
        },
        "design": {
            "target_split": manifest["target_split"],
            "games": len(rows),
            "task_type": "pick_two_obj_and_place",
            "selection_seed": SELECTION_SEED,
            "same_model_revision_scaffold_decoder_and_games": True,
            "baseline": "frozen P4-M explicit-state controller",
            "candidate_only_change": "frozen P4-O unique-object identity accounting",
            "baseline_runtime_model": baseline_report.get("runtime_model", {}),
            "candidate_runtime_model": candidate_report.get("runtime_model", {}),
        },
        "results": {
            "baseline_successes": baseline_successes,
            "candidate_successes": candidate_successes,
            "baseline_success_rate": baseline_successes / len(rows),
            "candidate_success_rate": candidate_successes / len(rows),
            "absolute_success_rate_delta": (candidate_successes - baseline_successes) / len(rows),
            "baseline_failures": baseline_failures,
            "corrected_games": corrected,
            "regressed_games": regressed,
            "stable_failure_games": stable_failures,
            "net_corrections": net_corrections,
            "joint_success_games": len(joint_successes),
            "joint_success_mean_p4m_steps": mean_old_steps,
            "joint_success_mean_p4o_steps": mean_new_steps,
            "joint_success_relative_step_delta": efficiency_delta,
        },
        "interface_validation": interface,
        "decision": {
            "diagnostic_power_gate_passed": diagnostic_power,
            "effect_gate_passed": effect_gate,
            "interface_gate_passed": interface_gate,
            "candidate_status": status,
            "qualified_human_review_required": passed,
            "memory_write_authorized": False,
            "training_label_authorized": False,
            "production_action_authorized": False,
            "business_action_authorized": False,
            "interpretation": (
                "The frozen identity rule replicated across the new split and may enter qualified human review; it is not automatically promoted."
                if passed else
                "The frozen identity rule did not establish a cross-split replication result under every preregistered gate."
            ),
        },
    }
    write_json(OUTPUT_ROOT / "p4p_report.json", report)
    write_jsonl(OUTPUT_ROOT / "p4p_human_review_queue.jsonl", [{
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
        "# JEV P4-P cross-split transfer\n\n"
        "Evaluation-only audit of the frozen P4-O unique-object accounting rule on ALFWorld valid_seen. "
        "No result automatically authorizes Memory, training, production, or business actions.\n",
        encoding="utf-8",
    )
    archive = shutil.make_archive(str(ARCHIVE_BASE), "zip", root_dir=OUTPUT_ROOT.parent, base_dir=OUTPUT_ROOT.name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("P4-P archive:", archive)


if __name__ == "__main__":
    main()
