#!/usr/bin/env python3
"""P4-G: pre-register a stratified discovery/hidden-validation RSI split.

Twelve discovery games (two per ALFWorld task type) are executed now. Twelve
additional games are frozen in a manifest but not executed. Any evolved Memory
must be generated only from discovery trajectories before the hidden split is
opened. This script retains the repaired action state and constrained decoder.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = Path(os.environ.get("JEV_P4F3_SCRIPT", str(SCRIPT_DIR / "p4f3_exact_command_alfworld_7b.py")))
F4_SCRIPT = Path(os.environ.get("JEV_P4F4_SCRIPT", str(SCRIPT_DIR / "p4f4_shared_public_demo_alfworld_7b.py")))
F5_SCRIPT = Path(os.environ.get("JEV_P4F5_SCRIPT", str(SCRIPT_DIR / "p4f5_fresh_action_state_alfworld_7b.py")))
F6_SCRIPT = Path(os.environ.get("JEV_P4F6_SCRIPT", str(SCRIPT_DIR / "p4f6_constrained_action_alfworld_7b.py")))
OUTPUT_DIR = Path(os.environ.get("JEV_OUTPUT_DIR", "results/p4g_preregistered_discovery_alfworld_7b"))
ARCHIVE_BASE = os.environ.get(
    "JEV_ARCHIVE_BASE", str(OUTPUT_DIR.with_name(f"{OUTPUT_DIR.name}_artifacts"))
)
PER_TYPE_PER_SPLIT = int(os.environ.get("JEV_PER_TYPE_PER_SPLIT", "2"))

# Hashes preserve exclusion membership without publishing public task identifiers.
P4F_EXCLUDED_GAME_SHA256 = {
    "b137832c00a5b386cc1cd4feba8e06dc837d5e8ee050d504aac7ee93a639ddbf",
    "7ee14d9445e8cdbe7d2dd03128c7965d843abb70d6e47b8afa0f1bd6ab1751fe",
    "c825d45c94001033b17d2d6c44804779a3cc7e2fb2c8a5975923ec7a1723a1c3",
    "34641718f173369773b0309df2b2adae19f40576b8864e89bb2b6def29c03cb0",
    "4e03a578d43b0b82567657579ec4aaf8556a86c7881e81d8a7b977359ffe424e",
    "fac148c0f0d543996c65c81765d5bab4e587cecd8a30c04d01c80eb7b007b8f6",
    "e9782828c3779018e6c8fbdb121ba3bbb304ce6475845a1b07098df9af918cde",
    "325ce2e01436259fb1ec5736dfbf37bfe5b3737895545d41b1d466456e2464c6",
}

SPLIT_MANIFEST = {}


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing required frozen script: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def exact_mcnemar_p(corrected: int, regressed: int) -> float:
    discordant = corrected + regressed
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(corrected, regressed) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def main() -> None:
    base = load_module(BASE_SCRIPT, "p4f3_base")
    f4 = load_module(F4_SCRIPT, "p4f4_scaffold")
    f5 = load_module(F5_SCRIPT, "p4f5_fresh_state")
    f6 = load_module(F6_SCRIPT, "p4f6_constrained")

    task_types = list(base.MEMORY_RULES)

    def choose_discovery_games(game_files):
        grouped = defaultdict(list)
        for game in game_files:
            gid = base.relative_game_id(game)
            task_type = base.task_type_for(game)
            if base.stable_hash(gid) not in P4F_EXCLUDED_GAME_SHA256 and task_type in task_types:
                grouped[task_type].append(game)
        discovery, validation = [], []
        for task_type in task_types:
            ranked = sorted(
                grouped[task_type],
                key=lambda path: (base.stable_hash(base.relative_game_id(path)), base.relative_game_id(path)),
            )
            needed = PER_TYPE_PER_SPLIT * 2
            if len(ranked) < needed:
                raise RuntimeError(f"Not enough games for {task_type}: {len(ranked)} < {needed}")
            discovery.extend(ranked[:PER_TYPE_PER_SPLIT])
            validation.extend(ranked[PER_TYPE_PER_SPLIT:needed])
        SPLIT_MANIFEST.update(
            {
                "schema_version": "JEV_P4G_PREREGISTERED_SPLIT_V1",
                "selection": "exclude prior game hashes; rank SHA256(game_id); first 2/type discovery; next 2/type hidden validation",
                "task_types": task_types,
                "discovery_game_ids": [base.relative_game_id(x) for x in discovery],
                "hidden_validation_game_ids": [base.relative_game_id(x) for x in validation],
                "excluded_prior_game_sha256": sorted(P4F_EXCLUDED_GAME_SHA256),
                "hidden_validation_executed": False,
                "memory_candidate_access_authorized": False,
            }
        )
        return discovery

    base.OUTPUT_DIR = OUTPUT_DIR
    base.ARCHIVE_BASE = ARCHIVE_BASE
    base.choose_holdout_games = choose_discovery_games
    base.choose_action = f6.build_constrained_choose_action(base, f4)
    base.run_arm = f5.build_fresh_run_arm(base)
    base.main()

    report_path = OUTPUT_DIR / "p4f_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    paired = [
        json.loads(line)
        for line in (OUTPUT_DIR / "p4f_paired_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [
        json.loads(line)
        for line in (OUTPUT_DIR / "p4f_complete_trajectories.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    corrected = sum(x["transition"] == "corrected" for x in paired)
    regressed = sum(x["transition"] == "regressed" for x in paired)
    per_task = {}
    for task_type in task_types:
        subset = [x for x in paired if x["task_type"] == task_type]
        per_task[task_type] = {
            "games": len(subset),
            "baseline_successes": sum(x["baseline_success"] for x in subset),
            "memory_successes": sum(x["memory_success"] for x in subset),
            "corrected": sum(x["transition"] == "corrected" for x in subset),
            "regressed": sum(x["transition"] == "regressed" for x in subset),
        }

    report["schema_version"] = "JEV_P4G_PREREGISTERED_DISCOVERY_ALFWORLD_7B_REPORT_V1"
    report["purpose"] = (
        "Run the pre-registered public ALFWorld discovery split across all six task types, "
        "while keeping a disjoint validation split unopened for evolved-Memory testing"
    )
    report["design"].update(
        {
            "split": "discovery",
            "stratification": "2 games per each of 6 ALFWorld task types",
            "prior_p4f_overlap": 0,
            "hidden_validation_games_frozen": len(SPLIT_MANIFEST["hidden_validation_game_ids"]),
            "hidden_validation_executed": False,
            "shared_public_scaffold": True,
            "action_state_refresh": True,
            "action_selection": "prefix-constrained trie over current legal commands",
            "fallback_action_policy": "none",
        }
    )
    report["per_task_type"] = per_task
    report["paired_exact_mcnemar"] = {
        "corrected": corrected,
        "regressed": regressed,
        "two_sided_exact_p": exact_mcnemar_p(corrected, regressed),
        "note": "Discovery inference only; not a confirmatory hidden-set claim.",
    }
    report["interface_validation"] = {
        "dynamic_manipulation_steps": sum(x["dynamic_manipulation_steps"] for x in rows),
        "parse_fallbacks": sum(x["parse_fallbacks"] for x in rows),
        "constrained_decoder_violations": sum(
            int(step.get("parse_method") != "prefix_constrained_legal_command")
            for row in rows
            for step in row["trajectory"]
        ),
        "action_state_refresh_observed": any(x["dynamic_manipulation_steps"] > 0 for x in rows),
    }
    report["rsi_boundary"] = {
        "memory_candidate_may_use": "discovery trajectories only",
        "memory_candidate_may_not_use": "hidden validation trajectories or outcomes",
        "production_memory_authorized": False,
        "training_label_authorized": False,
        "business_action_authorized": False,
    }
    report["decision"]["production_memory_authorized"] = False
    report["decision"]["next_gate"] = (
        "Generate one versioned Memory candidate from discovery errors; replay old vs candidate "
        "on discovery; freeze the winner before opening hidden validation."
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "p4g_preregistered_split_manifest.json").write_text(
        json.dumps(SPLIT_MANIFEST, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "shared_public_demo.txt").write_text(f4.SHARED_PUBLIC_SCAFFOLD, encoding="utf-8")
    (OUTPUT_DIR / "public_demo_source.json").write_text(
        json.dumps(f4.PUBLIC_SOURCE, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for source in (BASE_SCRIPT, F4_SCRIPT, F5_SCRIPT, F6_SCRIPT, Path(__file__)):
        shutil.copy2(source, OUTPUT_DIR / source.name)
    (OUTPUT_DIR / "README.md").write_text(
        "# JEV P4-G pre-registered discovery split\n\n"
        "Only discovery cases were executed. Hidden validation IDs are frozen in the manifest "
        "and may not be opened until a candidate Memory is versioned and frozen.\n",
        encoding="utf-8",
    )
    archive = shutil.make_archive(
        ARCHIVE_BASE, "zip", root_dir=OUTPUT_DIR.parent, base_dir=OUTPUT_DIR.name
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("P4-G archive:", archive)


if __name__ == "__main__":
    main()

