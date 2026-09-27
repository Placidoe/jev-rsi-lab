#!/usr/bin/env python3
"""P4-L: second-model replication on a disjoint ALFWorld valid_unseen set.

This evaluation freezes the P4-I typed state controller and changes two
pre-registered dimensions together: the policy model becomes Mistral-7B-
Instruct-v0.3 at a pinned revision, and the 12 evaluation games are selected
from ALFWorld valid_unseen after excluding every P4-F/P4-G/P4-J game ID.

The result is evidence about cross-model transfer on a disjoint public OOD
set. It does not authorize production Memory writes, training labels, or
business actions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from collections import defaultdict
from pathlib import Path


WORKING = Path("/kaggle/working")
FROZEN_P4I_INPUT = Path(
    "/kaggle/input/datasets/placideo/jev-p4i-explicit-state-controller/"
    "jev_p4i_explicit_state_controller_alfworld_7b.py"
)
FROZEN_P4I_SHA256 = "9b4079a98592ae41437dff2be0830ee5920fc8218f9e5d12e7be31c54c48a0d2"
REFERENCE_BASE = WORKING / "jev_p4f3_exact_command_alfworld_7b.py"
REFERENCE_MANIFEST = (
    WORKING
    / "jev_p4g_preregistered_discovery_alfworld_7b"
    / "p4g_preregistered_split_manifest.json"
)
SECOND_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
SECOND_MODEL_REVISION = "b0693ea4ce84f1a6a70ee5ac7c8efb0df82875f6"
SELECTION_SEED = 20260927
TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]
GAMES_PER_TASK_TYPE = 2
MIN_ABSOLUTE_SUCCESS_DELTA = 0.25
MAX_REGRESSIONS = 0

MODIFIED_BASE = WORKING / "jev_p4l_mistral7b_frozen_base.py"
P4L_MANIFEST_DIR = WORKING / "jev_p4l_disjoint_ood_preregistration"
OUTPUT_DIR = WORKING / "jev_p4l_second_model_disjoint_ood_alfworld_mistral7b"
ARCHIVE_BASE = WORKING / "jev_p4l_second_model_disjoint_ood_alfworld_mistral7b_artifacts"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one source match, got {count}: {old!r}")
    return source.replace(old, new, 1)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def prepare_second_model_base() -> None:
    source = REFERENCE_BASE.read_text(encoding="utf-8")
    source = replace_once(
        source,
        'MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"',
        f'MODEL_ID = "{SECOND_MODEL_ID}"\nMODEL_REVISION = "{SECOND_MODEL_REVISION}"',
    )
    source = replace_once(
        source,
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)",
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)",
    )
    source = replace_once(
        source,
        "model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,",
        "model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16,",
    )
    MODIFIED_BASE.write_text(source, encoding="utf-8")


def build_disjoint_manifest(base) -> dict:
    reference = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    excluded = set(reference["discovery_game_ids"])
    excluded.update(reference["hidden_validation_game_ids"])
    excluded.update(reference.get("excluded_prior_game_ids", []))

    from alfworld.agents.environment import get_environment

    engine = get_environment("AlfredTWEnv")(base.config(), train_eval="eval_out_of_distribution")
    try:
        all_game_files = sorted(engine.game_files)
    finally:
        close = getattr(engine, "close", None)
        if callable(close):
            close()

    by_task: dict[str, list[str]] = defaultdict(list)
    for path in all_game_files:
        game_id = base.relative_game_id(path)
        task_type = base.task_type_for(path)
        if game_id not in excluded and task_type in TASK_TYPES:
            by_task[task_type].append(game_id)

    selected = []
    selection_by_task = {}
    for task_type in TASK_TYPES:
        ranked = sorted(
            by_task[task_type],
            key=lambda game_id: hashlib.sha256(
                f"{SELECTION_SEED}:{game_id}".encode("utf-8")
            ).hexdigest(),
        )
        chosen = ranked[:GAMES_PER_TASK_TYPE]
        if len(chosen) != GAMES_PER_TASK_TYPE:
            raise RuntimeError(f"Not enough disjoint games for {task_type}: {len(chosen)}")
        selected.extend(chosen)
        selection_by_task[task_type] = chosen

    if set(selected) & excluded:
        raise RuntimeError("P4-L selection overlaps a prior P4-F/P4-G/P4-J game")
    if len(selected) != len(set(selected)):
        raise RuntimeError("P4-L selection contains duplicate game IDs")

    return {
        "schema_version": "JEV_P4L_DISJOINT_OOD_PREREGISTRATION_V1",
        "created_before_model_inference": True,
        "dataset_split": "ALFWorld valid_unseen",
        "selection": (
            "exclude all P4-F/P4-G/P4-J IDs; rank remaining IDs within each task "
            "type by SHA256(selection_seed:game_id); take first two per type"
        ),
        "selection_seed": SELECTION_SEED,
        "task_types": TASK_TYPES,
        "games_per_task_type": GAMES_PER_TASK_TYPE,
        "discovery_game_ids": selected,
        "hidden_validation_game_ids": [],
        "excluded_prior_game_ids": sorted(excluded),
        "selection_by_task_type": selection_by_task,
        "second_model_id": SECOND_MODEL_ID,
        "second_model_revision": SECOND_MODEL_REVISION,
        "frozen_controller_sha256": FROZEN_P4I_SHA256,
        "gate": {
            "minimum_absolute_success_rate_delta": MIN_ABSOLUTE_SUCCESS_DELTA,
            "maximum_regressions": MAX_REGRESSIONS,
            "require_zero_parser_fallbacks": True,
            "require_zero_decoder_violations": True,
        },
        "hidden_validation_executed": False,
        "production_memory_authorized": False,
        "training_label_authorized": False,
        "business_action_authorized": False,
    }


def main() -> None:
    for path in (FROZEN_P4I_INPUT, REFERENCE_BASE, REFERENCE_MANIFEST):
        if not path.exists():
            raise FileNotFoundError(path)
    if OUTPUT_DIR.exists() or ARCHIVE_BASE.with_suffix(".zip").exists():
        raise RuntimeError("P4-L output already exists; refusing overwrite")

    frozen_p4i_bytes = FROZEN_P4I_INPUT.read_bytes()
    observed_p4i_sha = sha256_bytes(frozen_p4i_bytes)
    if observed_p4i_sha != FROZEN_P4I_SHA256:
        raise RuntimeError(f"Frozen P4-I SHA mismatch: {observed_p4i_sha}")

    prepare_second_model_base()
    modified_base_sha = sha256_bytes(MODIFIED_BASE.read_bytes())
    base = load_module(MODIFIED_BASE, "p4l_second_model_base")
    manifest = build_disjoint_manifest(base)
    P4L_MANIFEST_DIR.mkdir(parents=True, exist_ok=False)
    manifest_path = P4L_MANIFEST_DIR / "p4g_preregistered_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    p4i = load_module(FROZEN_P4I_INPUT, "p4l_frozen_controller")
    p4i.BASE_SCRIPT = MODIFIED_BASE
    p4i.P4G_DIR = P4L_MANIFEST_DIR
    p4i.OUTPUT_DIR = OUTPUT_DIR
    p4i.ARCHIVE_BASE = str(ARCHIVE_BASE)
    p4i.main()

    report_path = OUTPUT_DIR / "p4f_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    paired_src = OUTPUT_DIR / "p4i_paired_cases.jsonl"
    paired_dst = OUTPUT_DIR / "p4l_paired_cases.jsonl"
    shutil.copy2(paired_src, paired_dst)

    effect = report["effect"]
    interface = report["interface_validation"]
    corrected = len(effect["corrected_games"])
    regressed = len(effect["regressed_games"])
    delta = effect["absolute_success_rate_delta"]
    gate_passed = (
        delta >= MIN_ABSOLUTE_SUCCESS_DELTA
        and corrected > 0
        and regressed <= MAX_REGRESSIONS
        and interface["parse_fallbacks"] == 0
        and interface["constrained_decoder_violations"] == 0
    )
    report["schema_version"] = "JEV_P4L_SECOND_MODEL_DISJOINT_OOD_ALFWORLD_REPORT_V1"
    report["purpose"] = (
        "Replicate the frozen typed JEV state controller with a second open model "
        "on a disjoint ALFWorld valid_unseen OOD set"
    )
    report["runtime_model"] = {
        "model_id": SECOND_MODEL_ID,
        "revision": SECOND_MODEL_REVISION,
        "modified_base_sha256": modified_base_sha,
        "frozen_controller_sha256": observed_p4i_sha,
    }
    report["design"].update(
        {
            "split": "P4-L disjoint ALFWorld valid_unseen OOD set",
            "games": len(manifest["discovery_game_ids"]),
            "prior_game_overlap": 0,
            "second_model": True,
            "selection_manifest_created_before_inference": True,
            "same_controller_memory_scaffold_decoder_and_thresholds": True,
            "changed_dimensions": ["policy model", "disjoint valid_unseen game IDs"],
        }
    )
    report["decision"] = {
        "second_model_disjoint_ood_gate_passed": gate_passed,
        "pre_registered_gate": manifest["gate"],
        "interpretation": (
            "A pass supports cross-model transfer on a disjoint public valid_unseen set; "
            "it does not establish production-domain generalization."
            if gate_passed
            else "The controller did not pass the second-model disjoint-OOD gate; do not promote it."
        ),
        "next_gate": (
            "Run an automated failure-to-rule-to-Replay staging loop on a new frozen split"
            if gate_passed
            else "Use only this disjoint set as discovery evidence for a versioned controller revision"
        ),
        "production_memory_authorized": False,
        "training_label_authorized": False,
        "business_action_authorized": False,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "p4l_disjoint_ood_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), OUTPUT_DIR / Path(__file__).name)
    shutil.copy2(MODIFIED_BASE, OUTPUT_DIR / MODIFIED_BASE.name)
    (OUTPUT_DIR / "README.md").write_text(
        "# JEV P4-L second-model disjoint-OOD replication\n\n"
        "Mistral-7B-Instruct-v0.3 at a pinned revision, 12 disjoint ALFWorld "
        "valid_unseen games, and the frozen P4-I state controller. All authorization "
        "gates remain false.\n",
        encoding="utf-8",
    )
    archive_path = ARCHIVE_BASE.with_suffix(".zip")
    if archive_path.exists():
        archive_path.unlink()
    archive = shutil.make_archive(
        str(ARCHIVE_BASE), "zip", root_dir=OUTPUT_DIR.parent, base_dir=OUTPUT_DIR.name
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("P4-L archive:", archive)


if __name__ == "__main__":
    main()
