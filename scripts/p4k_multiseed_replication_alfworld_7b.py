#!/usr/bin/env python3
"""P4-K: replicate the frozen P4-J hidden comparison across three seeds.

The already-completed P4-J run is seed 20260927. This launcher runs only the
two additional pre-registered seeds, preserving the model, controller, task
IDs, prompts, legal-action decoder, thresholds, and arm ordering. It then
builds an auditable aggregate report and archive. No output authorizes
production Memory writes, training labels, or business actions.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


SEEDS = [20260927, 20260928, 20260929]
REFERENCE_SEED = 20260927
WORKING = Path("/kaggle/working")
REFERENCE_DIR = WORKING / "jev_p4j_frozen_hidden_validation_alfworld_7b"
REFERENCE_RUNNER = WORKING / "jev_p4j_frozen_hidden_validation_runner.py"
REFERENCE_BASE = WORKING / "jev_p4f3_exact_command_alfworld_7b.py"
REFERENCE_P4G_DIR = WORKING / "jev_p4g_preregistered_discovery_alfworld_7b"
OUTPUT_DIR = WORKING / "jev_p4k_multiseed_replication_alfworld_7b"
ARCHIVE_BASE = WORKING / "jev_p4k_multiseed_replication_alfworld_7b_artifacts"


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one source match, got {count}: {old!r}")
    return source.replace(old, new, 1)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_prediction_hash(rows: list[dict]) -> str:
    compact = [
        {
            "game_id": row["game_id"],
            "arm": row["arm"],
            "success": row["success"],
            "actions": [step["action"] for step in row["trajectory"]],
        }
        for row in sorted(rows, key=lambda r: (r["game_id"], r["arm"]))
    ]
    payload = json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_seed(seed: int) -> Path:
    seed_dir = WORKING / f"jev_p4k_seed_{seed}_alfworld_7b"
    seed_archive = WORKING / f"jev_p4k_seed_{seed}_alfworld_7b_artifacts"
    seed_runner = WORKING / f"jev_p4k_seed_{seed}_runner.py"
    seed_base = WORKING / f"jev_p4k_seed_{seed}_base.py"
    seed_prereg_dir = WORKING / f"jev_p4k_seed_{seed}_preregistration"
    if seed_dir.exists() or seed_archive.with_suffix(".zip").exists():
        raise RuntimeError(f"Seed output already exists; refusing overwrite: {seed}")

    base_source = REFERENCE_BASE.read_text(encoding="utf-8")
    base_source = replace_once(base_source, f"SEED = {REFERENCE_SEED}", f"SEED = {seed}")
    seed_base.write_text(base_source, encoding="utf-8")

    # P4-J marks its own pre-registration manifest as executed after the
    # one-shot hidden run.  Each planned replication therefore gets an
    # isolated copy of that frozen manifest with only the execution flag reset.
    # The game IDs, controller, prompts, decoder and thresholds are unchanged.
    manifest = read_json(REFERENCE_P4G_DIR / "p4g_preregistered_split_manifest.json")
    if manifest.get("hidden_validation_executed") is not True:
        raise RuntimeError("Reference P4-J manifest was not finalized")
    manifest["hidden_validation_executed"] = False
    manifest["replication_seed"] = seed
    manifest["replication_parent"] = "P4-J frozen hidden validation"
    seed_prereg_dir.mkdir(parents=True, exist_ok=False)
    (seed_prereg_dir / "p4g_preregistered_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    runner_source = REFERENCE_RUNNER.read_text(encoding="utf-8")
    runner_source = replace_once(
        runner_source,
        'BASE_SCRIPT = Path("/kaggle/working/jev_p4f3_exact_command_alfworld_7b.py")',
        f'BASE_SCRIPT = Path("{seed_base}")',
    )
    runner_source = replace_once(
        runner_source,
        'P4G_DIR = Path("/kaggle/working/jev_p4g_preregistered_discovery_alfworld_7b")',
        f'P4G_DIR = Path("{seed_prereg_dir}")',
    )
    runner_source = replace_once(
        runner_source,
        'OUTPUT_DIR = Path("/kaggle/working/jev_p4j_frozen_hidden_validation_alfworld_7b")',
        f'OUTPUT_DIR = Path("{seed_dir}")',
    )
    runner_source = replace_once(
        runner_source,
        'ARCHIVE_BASE = "/kaggle/working/jev_p4j_frozen_hidden_validation_alfworld_7b_artifacts"',
        f'ARCHIVE_BASE = "{seed_archive}"',
    )
    runner_source = replace_once(
        runner_source,
        '"schema_version": "JEV_P4J_FROZEN_HIDDEN_VALIDATION_ALFWORLD_7B_REPORT_V1",',
        '"schema_version": "JEV_P4K_MULTISEED_REPLICATION_RUN_V1",',
    )
    runner_source = replace_once(
        runner_source,
        '"purpose": "One-shot hidden validation of the frozen P4-I typed JEV state controller",',
        f'"purpose": "P4-K frozen-controller replication seed {seed}",',
    )
    seed_runner.write_text(runner_source, encoding="utf-8")
    subprocess.run([sys.executable, str(seed_runner)], check=True)
    return seed_dir


def summarize(seed: int, seed_dir: Path) -> dict:
    report = read_json(seed_dir / "p4j_hidden_validation_report.json")
    rows = read_jsonl(seed_dir / "p4f_complete_trajectories.jsonl")
    effect = report["effect"]
    interface = report["interface_validation"]
    return {
        "seed": seed,
        "old_memory_successes": report["old_memory"]["successes"],
        "state_controller_successes": report["jev_state_controller"]["successes"],
        "games": report["old_memory"]["games"],
        "corrected": len(effect["corrected_games"]),
        "regressed": len(effect["regressed_games"]),
        "absolute_success_rate_delta": effect["absolute_success_rate_delta"],
        "mcnemar_p": effect["paired_exact_mcnemar_two_sided_p"],
        "parse_fallbacks": interface["parse_fallbacks"],
        "decoder_violations": interface["constrained_decoder_violations"],
        "prediction_sha256": canonical_prediction_hash(rows),
    }


def main() -> None:
    required = [
        REFERENCE_DIR / "p4j_hidden_validation_report.json",
        REFERENCE_DIR / "p4f_complete_trajectories.jsonl",
        REFERENCE_RUNNER,
        REFERENCE_BASE,
        REFERENCE_P4G_DIR / "p4g_preregistered_split_manifest.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if OUTPUT_DIR.exists() or ARCHIVE_BASE.with_suffix(".zip").exists():
        raise RuntimeError("P4-K aggregate already exists; refusing overwrite")

    OUTPUT_DIR.mkdir(parents=True)
    seed_dirs = {REFERENCE_SEED: REFERENCE_DIR}
    for seed in SEEDS[1:]:
        seed_dirs[seed] = run_seed(seed)

    runs = [summarize(seed, seed_dirs[seed]) for seed in SEEDS]
    hashes = {run["prediction_sha256"] for run in runs}
    all_positive = all(
        run["state_controller_successes"] > run["old_memory_successes"]
        and run["regressed"] == 0
        and run["parse_fallbacks"] == 0
        and run["decoder_violations"] == 0
        for run in runs
    )
    report = {
        "schema_version": "JEV_P4K_MULTISEED_REPLICATION_ALFWORLD_7B_REPORT_V1",
        "purpose": "Test whether the frozen P4-J gain survives three deterministic seeds",
        "design": {
            "seeds": SEEDS,
            "reference_seed_reused": REFERENCE_SEED,
            "additional_runs": SEEDS[1:],
            "same_hidden_ids_model_controller_prompts_decoder_and_thresholds": True,
            "only_planned_change": "random seed",
            "production_memory_authorized": False,
            "training_label_authorized": False,
            "business_action_authorized": False,
        },
        "runs": runs,
        "aggregate": {
            "all_seeds_positive": all_positive,
            "identical_action_trajectory_hash_across_seeds": len(hashes) == 1,
            "unique_prediction_hashes": sorted(hashes),
            "mean_old_memory_success_rate": sum(r["old_memory_successes"] / r["games"] for r in runs) / len(runs),
            "mean_state_controller_success_rate": sum(r["state_controller_successes"] / r["games"] for r in runs) / len(runs),
            "total_corrected": sum(r["corrected"] for r in runs),
            "total_regressed": sum(r["regressed"] for r in runs),
        },
        "decision": {
            "multiseed_gate_passed": all_positive,
            "interpretation": "A pass supports deterministic robustness in this fixed setting; it does not establish cross-model or OOD generalization.",
            "next_gate": "Replicate with a second open model and a disjoint OOD long-horizon set" if all_positive else "Reject or revise the controller using discovery-only evidence",
            "production_memory_authorized": False,
            "training_label_authorized": False,
            "business_action_authorized": False,
        },
    }
    (OUTPUT_DIR / "p4k_multiseed_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "p4k_seed_runs.jsonl").write_text(
        "".join(json.dumps(run, ensure_ascii=False) + "\n" for run in runs), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), OUTPUT_DIR / Path(__file__).name)
    for seed, seed_dir in seed_dirs.items():
        shutil.copy2(seed_dir / "p4j_hidden_validation_report.json", OUTPUT_DIR / f"seed_{seed}_report.json")
        shutil.copy2(seed_dir / "p4j_hidden_paired_cases.jsonl", OUTPUT_DIR / f"seed_{seed}_paired_cases.jsonl")
    (OUTPUT_DIR / "README.md").write_text(
        "# JEV P4-K multi-seed replication\n\nFrozen P4-J controller, three seeds. All authorization gates remain closed.\n",
        encoding="utf-8",
    )
    archive = shutil.make_archive(str(ARCHIVE_BASE), "zip", root_dir=OUTPUT_DIR.parent, base_dir=OUTPUT_DIR.name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("P4-K archive:", archive)


if __name__ == "__main__":
    main()
