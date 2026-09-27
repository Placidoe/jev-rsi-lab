#!/usr/bin/env python3
"""P4-H: evolve one evidence-bounded Memory candidate on P4-G discovery only.

The P4-G hidden validation split remains unopened.  This script compares the
frozen old Memory with one candidate distilled from discovery failure patterns.
Both arms use the same model, prefix-constrained legal-command decoder, fresh
action state, public scaffold, loop guard, games, and counterbalanced order.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
P4G_DIR = Path(os.environ.get("JEV_P4G_DIR", "results/p4g_preregistered_discovery_alfworld_7b"))
BASE_SCRIPT = Path(os.environ.get("JEV_P4F3_SCRIPT", str(SCRIPT_DIR / "p4f3_exact_command_alfworld_7b.py")))
F4_SCRIPT = Path(os.environ.get("JEV_P4F4_SCRIPT", str(SCRIPT_DIR / "p4f4_shared_public_demo_alfworld_7b.py")))
F5_SCRIPT = Path(os.environ.get("JEV_P4F5_SCRIPT", str(SCRIPT_DIR / "p4f5_fresh_action_state_alfworld_7b.py")))
F6_SCRIPT = Path(os.environ.get("JEV_P4F6_SCRIPT", str(SCRIPT_DIR / "p4f6_constrained_action_alfworld_7b.py")))
OUTPUT_DIR = Path(os.environ.get("JEV_OUTPUT_DIR", "results/p4h_discovery_memory_evolution_alfworld_7b"))
ARCHIVE_BASE = os.environ.get(
    "JEV_ARCHIVE_BASE", str(OUTPUT_DIR.with_name(f"{OUTPUT_DIR.name}_artifacts"))
)


COMMON_STATE_MACHINE = (
    "First parse the goal into target object noun(s), required transformation, and final receptacle. "
    "Maintain those as the only active subgoals. While searching, visit each new location, open a closed "
    "container immediately, leave searched containers open, and never take or manipulate a non-target object. "
    "When the target appears, take that exact target immediately. Once holding it, stop searching and execute "
    "only the next unmet transformation or final-placement subgoal. Never abandon the target for a distractor. "
    "Prefer the allowed command that directly advances the next unmet subgoal; avoid examine/close actions unless "
    "they are necessary to reveal the target or complete the goal. "
)

CANDIDATE_RULES = {
    "pick_and_place_simple": COMMON_STATE_MACHINE
    + "State machine: SEARCH target -> TAKE exact target -> GO TO requested receptacle -> MOVE exact target there.",
    "look_at_obj_in_light": COMMON_STATE_MACHINE
    + "State machine: SEARCH target -> TAKE exact target -> GO TO the desk lamp -> USE the lamp -> EXAMINE the held target. Do not pick up or examine distractors or furniture.",
    "pick_clean_then_place_in_recep": COMMON_STATE_MACHINE
    + "State machine: SEARCH target -> TAKE exact target -> GO TO a sinkbasin -> CLEAN exact target with that sinkbasin -> GO TO requested receptacle -> MOVE exact target there. Never clean or carry a distractor.",
    "pick_heat_then_place_in_recep": COMMON_STATE_MACHINE
    + "State machine: SEARCH target -> TAKE exact target -> GO TO the heating appliance -> HEAT exact target with it -> GO TO requested receptacle -> MOVE exact target there. Never heat or carry a distractor.",
    "pick_cool_then_place_in_recep": COMMON_STATE_MACHINE
    + "State machine: SEARCH target -> TAKE exact target -> GO TO the fridge -> COOL exact target with it -> GO TO requested receptacle -> MOVE exact target there. Never cool or carry a distractor.",
    "pick_two_obj_and_place": COMMON_STATE_MACHINE
    + "Track progress explicitly as 0/2, 1/2, 2/2. SEARCH and TAKE one requested target instance, MOVE it to the requested receptacle, then search for and place the second requested instance. Do not stop after the first and never carry a distractor.",
}


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_sha(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def exact_mcnemar_p(corrected: int, regressed: int) -> float:
    n = corrected + regressed
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(corrected, regressed) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def discovery_evidence(rows: list[dict]) -> dict:
    """Summarize only observable discovery failures; never inspect hidden games."""
    failures = [row for row in rows if not row["success"]]
    by_task = {}
    for task_type in CANDIDATE_RULES:
        subset = [row for row in failures if row["task_type"] == task_type]
        actions = [step["action"] for row in subset for step in row["trajectory"]]
        by_task[task_type] = {
            "failed_arms": len(subset),
            "steps": sum(row["steps"] for row in subset),
            "take_actions": sum(action.startswith("take ") for action in actions),
            "close_actions": sum(action.startswith("close ") for action in actions),
            "examine_actions": sum(action.startswith("examine ") for action in actions),
            "transformation_actions": sum(
                action.startswith(("clean ", "heat ", "cool ", "use ")) for action in actions
            ),
            "unique_action_examples": sorted(set(actions))[:20],
        }
    return {
        "source": "P4-G discovery trajectories only",
        "failed_arms": len(failures),
        "hidden_validation_accessed": False,
        "by_task_type": by_task,
    }


def main() -> None:
    required = [BASE_SCRIPT, F4_SCRIPT, F5_SCRIPT, F6_SCRIPT, P4G_DIR / "p4g_preregistered_split_manifest.json", P4G_DIR / "p4f_complete_trajectories.jsonl"]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    base = load_module(BASE_SCRIPT, "p4h_base")
    f4 = load_module(F4_SCRIPT, "p4h_f4")
    f5 = load_module(F5_SCRIPT, "p4h_f5")
    f6 = load_module(F6_SCRIPT, "p4h_f6")

    manifest = json.loads((P4G_DIR / "p4g_preregistered_split_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("hidden_validation_executed") is not False:
        raise RuntimeError("P4-G hidden-validation boundary is not intact")
    discovery_ids = list(manifest["discovery_game_ids"])
    discovery_set = set(discovery_ids)
    if discovery_set & set(manifest["hidden_validation_game_ids"]):
        raise RuntimeError("Discovery/hidden overlap")

    p4g_rows = read_jsonl(P4G_DIR / "p4f_complete_trajectories.jsonl")
    if {row["game_id"] for row in p4g_rows} != discovery_set:
        raise RuntimeError("P4-G trajectory IDs do not equal the frozen discovery set")
    evidence = discovery_evidence(p4g_rows)
    old_rules = dict(base.MEMORY_RULES)
    old_sha = canonical_sha(old_rules)
    candidate_sha = canonical_sha(CANDIDATE_RULES)

    def choose_discovery(game_files):
        by_id = {base.relative_game_id(path): path for path in game_files}
        missing = [gid for gid in discovery_ids if gid not in by_id]
        if missing:
            raise RuntimeError(f"Missing frozen discovery games: {missing}")
        return [by_id[gid] for gid in discovery_ids]

    fresh = f5.build_fresh_run_arm(base)

    def run_old_vs_candidate(engine, game_file, arm, model, tokenizer):
        rules = old_rules if arm == "guarded_baseline" else CANDIDATE_RULES
        label = "old_memory" if arm == "guarded_baseline" else "candidate_memory"
        saved = base.MEMORY_RULES
        try:
            base.MEMORY_RULES = rules
            row = fresh(engine, game_file, "guarded_memory", model, tokenizer)
        finally:
            base.MEMORY_RULES = saved
        row["arm"] = arm
        row["comparison_arm"] = label
        row["memory_version_sha256"] = old_sha if label == "old_memory" else candidate_sha
        row["memory_injected"] = rules[row["task_type"]]
        return row

    base.OUTPUT_DIR = OUTPUT_DIR
    base.ARCHIVE_BASE = ARCHIVE_BASE
    base.choose_holdout_games = choose_discovery
    base.choose_action = f6.build_constrained_choose_action(base, f4)
    base.run_arm = run_old_vs_candidate
    base.main()

    rows = read_jsonl(OUTPUT_DIR / "p4f_complete_trajectories.jsonl")
    paired = read_jsonl(OUTPUT_DIR / "p4f_paired_cases.jsonl")
    corrected = sum(row["transition"] == "corrected" for row in paired)
    regressed = sum(row["transition"] == "regressed" for row in paired)
    old_successes = sum(row["baseline_success"] for row in paired)
    candidate_successes = sum(row["memory_success"] for row in paired)
    per_task = {}
    for task_type in CANDIDATE_RULES:
        subset = [row for row in paired if row["task_type"] == task_type]
        per_task[task_type] = {
            "games": len(subset),
            "old_memory_successes": sum(row["baseline_success"] for row in subset),
            "candidate_memory_successes": sum(row["memory_success"] for row in subset),
            "corrected": sum(row["transition"] == "corrected" for row in subset),
            "regressed": sum(row["transition"] == "regressed" for row in subset),
        }

    promote = candidate_successes > old_successes and regressed == 0
    report = json.loads((OUTPUT_DIR / "p4f_report.json").read_text(encoding="utf-8"))
    report.update({
        "schema_version": "JEV_P4H_DISCOVERY_MEMORY_EVOLUTION_ALFWORLD_7B_REPORT_V1",
        "purpose": "Evolve and replay one evidence-bounded Memory candidate using P4-G discovery trajectories only",
        "design": {
            **report["design"],
            "split": "P4-G discovery only",
            "games": len(paired),
            "old_memory_sha256": old_sha,
            "candidate_memory_sha256": candidate_sha,
            "candidate_generation": "fixed task-agnostic state-machine template triggered by aggregate discovery failure signals",
            "object_ids_or_hidden_outcomes_in_candidate": False,
            "hidden_validation_executed": False,
            "action_state_refresh": True,
            "action_selection": "prefix-constrained trie over current legal commands",
            "fallback_action_policy": "none",
            "only_arm_difference": "old Memory versus candidate Memory text",
        },
        "old_memory": {"successes": old_successes, "games": len(paired), "success_rate": old_successes / len(paired)},
        "candidate_memory": {"successes": candidate_successes, "games": len(paired), "success_rate": candidate_successes / len(paired)},
        "effect": {
            "corrected_games": [row["game_id"] for row in paired if row["transition"] == "corrected"],
            "regressed_games": [row["game_id"] for row in paired if row["transition"] == "regressed"],
            "net_corrections": corrected - regressed,
            "absolute_success_rate_delta": (candidate_successes - old_successes) / len(paired),
            "paired_exact_mcnemar_two_sided_p": exact_mcnemar_p(corrected, regressed),
        },
        "per_task_type": per_task,
        "interface_validation": {
            "dynamic_manipulation_steps": sum(row["dynamic_manipulation_steps"] for row in rows),
            "parse_fallbacks": sum(row["parse_fallbacks"] for row in rows),
            "constrained_decoder_violations": sum(
                step.get("parse_method") != "prefix_constrained_legal_command"
                for row in rows for step in row["trajectory"]
            ),
        },
        "decision": {
            "candidate_wins_discovery_gate": promote,
            "freeze_candidate_before_hidden_validation": promote,
            "next_gate": "Open the pre-registered P4-G hidden validation once with this frozen candidate" if promote else "Reject candidate; revise from discovery evidence without opening hidden validation",
            "production_memory_authorized": False,
            "training_label_authorized": False,
            "business_action_authorized": False,
        },
    })
    (OUTPUT_DIR / "p4f_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "p4h_old_memory_rules.json").write_text(json.dumps(old_rules, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "p4h_candidate_memory_rules.json").write_text(json.dumps(CANDIDATE_RULES, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "p4h_discovery_evidence_summary.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "p4g_preregistered_split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), OUTPUT_DIR / Path(__file__).name)
    for source in (BASE_SCRIPT, F4_SCRIPT, F5_SCRIPT, F6_SCRIPT):
        shutil.copy2(source, OUTPUT_DIR / source.name)
    (OUTPUT_DIR / "README.md").write_text(
        "# JEV P4-H discovery Memory evolution\n\n"
        "Old and candidate Memory are replayed on the frozen P4-G discovery set. Hidden validation is unopened. "
        "All authorization fields remain false.\n",
        encoding="utf-8",
    )
    archive = shutil.make_archive(ARCHIVE_BASE, "zip", root_dir=OUTPUT_DIR.parent, base_dir=OUTPUT_DIR.name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("P4-H archive:", archive)


if __name__ == "__main__":
    main()

