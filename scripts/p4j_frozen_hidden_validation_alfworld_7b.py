#!/usr/bin/env python3
"""P4-J: open a pre-registered ALFWorld hidden split exactly once.

The launcher verifies the frozen P4-I controller and its passed discovery
gate, mechanically changes only the selected split and report labels, then
runs the generated hidden-only runner. It intentionally refuses to overwrite
an existing P4-J output directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FROZEN_P4I_SHA256 = os.environ.get(
    "JEV_FROZEN_P4I_SHA256",
    "438b834563c193d3bc892900ed8462882e7b66e95e12c40bdb6ede00cc407b50",
)
P4I_INPUT = Path(os.environ.get(
    "JEV_P4I_SCRIPT", str(SCRIPT_DIR / "p4i_explicit_state_controller_alfworld_7b.py")
))
P4G_DIR = Path(os.environ.get("JEV_P4G_DIR", "results/p4g_preregistered_discovery_alfworld_7b"))
P4I_DISCOVERY_REPORT = Path(os.environ.get(
    "JEV_P4I_DISCOVERY_REPORT", str(Path("results/p4i_explicit_state_controller_alfworld_7b") / "p4f_report.json")
))
P4J_OUTPUT = Path(os.environ.get("JEV_OUTPUT_DIR", "results/p4j_frozen_hidden_validation_alfworld_7b"))
P4J_RUNNER = Path(os.environ.get(
    "JEV_P4J_RUNNER", str(P4J_OUTPUT.parent / "p4j_frozen_hidden_validation_runner.py")
))
P4J_ARCHIVE_BASE = os.environ.get(
    "JEV_ARCHIVE_BASE", str(P4J_OUTPUT.with_name(f"{P4J_OUTPUT.name}_artifacts"))
)


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one frozen source match, got {count}: {old[:100]!r}")
    return source.replace(old, new, 1)


def main() -> None:
    archive = Path(f"{P4J_ARCHIVE_BASE}.zip")
    if P4J_OUTPUT.exists() or archive.exists() or P4J_RUNNER.exists():
        raise FileExistsError("P4-J output already exists; refusing a second view of the hidden split")
    for path in (P4I_INPUT, P4I_DISCOVERY_REPORT, P4G_DIR / "p4g_preregistered_split_manifest.json"):
        if not path.exists():
            raise FileNotFoundError(path)

    frozen_bytes = P4I_INPUT.read_bytes()
    observed_sha = hashlib.sha256(frozen_bytes).hexdigest()
    if observed_sha != FROZEN_P4I_SHA256:
        raise RuntimeError(f"Frozen P4-I SHA mismatch: {observed_sha}")

    discovery = json.loads(P4I_DISCOVERY_REPORT.read_text(encoding="utf-8"))
    if discovery.get("decision", {}).get("candidate_wins_discovery_gate") is not True:
        raise RuntimeError("P4-I discovery gate did not pass; hidden validation is not authorized")
    if discovery.get("design", {}).get("hidden_validation_executed") is not False:
        raise RuntimeError("P4-I report does not preserve the unopened hidden boundary")

    source = frozen_bytes.decode("utf-8")
    source = replace_once(
        source,
        '"""P4-I: compare frozen text Memory with an explicit JEV task-state controller.',
        '"""P4-J: one-shot hidden validation of the frozen P4-I state controller.',
    )
    source = replace_once(
        source,
        "The P4-G hidden validation split remains unopened.  Both arms use the same\n"
        "Qwen2.5-7B policy, public scaffold, prefix-constrained legal-action decoder,\n"
        "fresh environment state, discovery games, and counterbalanced order.  The\n"
        "candidate arm adds a typed state compiler and phase-aware legal-action filter;\n"
        "it does not learn from or inspect hidden games.",
        "The controller, model, Memory, decoder, prompts, thresholds and arm order are\n"
        "frozen from P4-I. This runner changes only the selected split to the\n"
        "pre-registered hidden IDs and records that the one-shot gate was opened.",
    )
    source = replace_once(
        source,
        'discovery_ids = list(manifest["discovery_game_ids"])',
        'hidden_ids = list(manifest["hidden_validation_game_ids"])',
    )
    source = replace_once(source, "def choose_discovery(game_files):", "def choose_hidden(game_files):")
    source = replace_once(
        source,
        'if set(discovery_ids) & set(manifest["hidden_validation_game_ids"]):',
        'if set(manifest["discovery_game_ids"]) & set(hidden_ids):',
    )
    source = replace_once(source, "gid for gid in discovery_ids", "gid for gid in hidden_ids")
    source = replace_once(source, "by_id[gid] for gid in discovery_ids", "by_id[gid] for gid in hidden_ids")
    source = source.replace("Missing discovery games", "Missing hidden-validation games")
    source = replace_once(source, "base.choose_holdout_games = choose_discovery", "base.choose_holdout_games = choose_hidden")
    source = replace_once(source, 'write_jsonl(OUTPUT_DIR / "p4i_paired_cases.jsonl", paired)', 'write_jsonl(OUTPUT_DIR / "p4j_hidden_paired_cases.jsonl", paired)')
    source = replace_once(source, '"schema_version": "JEV_P4I_EXPLICIT_STATE_CONTROLLER_ALFWORLD_7B_REPORT_V1",', '"schema_version": "JEV_P4J_FROZEN_HIDDEN_VALIDATION_ALFWORLD_7B_REPORT_V1",')
    source = replace_once(source, '"purpose": "Test whether a typed JEV state compiler improves long-horizon action selection beyond text Memory alone",', '"purpose": "One-shot hidden validation of the frozen P4-I typed JEV state controller",')
    source = replace_once(source, '"split": "P4-G discovery only",', '"split": "P4-G pre-registered hidden validation",')
    source = replace_once(source, '"hidden_validation_executed": False,', '"hidden_validation_executed": True,')
    source = replace_once(source, '"candidate_wins_discovery_gate": promote,', '"candidate_wins_hidden_gate": promote,')
    source = replace_once(source, '"freeze_candidate_before_hidden_validation": promote,', '"frozen_candidate_was_used": True,')
    source = replace_once(
        source,
        '"next_gate": "Open pre-registered hidden validation once with the frozen controller" if promote else "Reject or revise the controller using discovery evidence only; keep hidden validation unopened",',
        '"next_gate": "Authorize a controlled RSI prototype replication; keep production, training-label, and business-action gates closed" if promote else "Reject generalization and return to discovery with a new version",',
    )
    source = replace_once(source, '(OUTPUT_DIR / "p4f_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")', '(OUTPUT_DIR / "p4j_hidden_validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")')
    source = replace_once(
        source,
        '(OUTPUT_DIR / "p4g_preregistered_split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")',
        'validated_manifest = dict(manifest)\n'
        '    validated_manifest["hidden_validation_executed"] = True\n'
        '    validated_manifest["hidden_validation_runner_sha256"] = "' + FROZEN_P4I_SHA256 + '"\n'
        '    (OUTPUT_DIR / "p4g_preregistered_split_manifest.json").write_text(json.dumps(validated_manifest, ensure_ascii=False, indent=2), encoding="utf-8")',
    )
    source = replace_once(
        source,
        '"# JEV P4-I explicit state controller\\n\\nDiscovery-only causal comparison. Hidden validation is unopened and every authorization field is false.\\n",',
        '"# JEV P4-J frozen hidden validation\\n\\nOne-shot pre-registered hidden comparison. Every production, training-label, and business-action authorization remains false.\\n",',
    )
    source = replace_once(source, 'print("P4-I archive:", archive)', 'print("P4-J archive:", archive)')

    P4J_RUNNER.parent.mkdir(parents=True, exist_ok=True)
    P4J_RUNNER.write_text(source, encoding="utf-8")
    run_env = os.environ.copy()
    run_env.update({
        "JEV_OUTPUT_DIR": str(P4J_OUTPUT),
        "JEV_ARCHIVE_BASE": P4J_ARCHIVE_BASE,
        "JEV_P4G_DIR": str(P4G_DIR),
        "JEV_P4F3_SCRIPT": str(SCRIPT_DIR / "p4f3_exact_command_alfworld_7b.py"),
        "JEV_P4F4_SCRIPT": str(SCRIPT_DIR / "p4f4_shared_public_demo_alfworld_7b.py"),
        "JEV_P4F6_SCRIPT": str(SCRIPT_DIR / "p4f6_constrained_action_alfworld_7b.py"),
    })
    print(json.dumps({
        "frozen_p4i_sha256": observed_sha,
        "discovery_gate_passed": True,
        "hidden_boundary_before_run": False,
        "hidden_runner": str(P4J_RUNNER),
    }, indent=2))
    subprocess.run([sys.executable, str(P4J_RUNNER)], check=True, env=run_env)


if __name__ == "__main__":
    main()
