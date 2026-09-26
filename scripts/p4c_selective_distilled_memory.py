#!/usr/bin/env python3
"""P4-C: selective, distilled JEV Memory gate on the frozen P4-B set.

This is a representation/gating ablation, not a new test-set search.  It reuses
the exact P4-B baseline outputs, retrieves with the same public task prompts,
and invokes Memory only when the pre-registered top-1 TF-IDF similarity is at
least 0.08.  The injected text is a short task-agnostic rule distilled from the
public Reflexion evidence.  All other tasks take the baseline fallback.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any


SEED = 20260927
THRESHOLD = 0.08
OUTPUT_DIR = Path(os.environ.get("JEV_OUTPUT_DIR", "results/p4c_distilled_selective_memory"))
ARCHIVE_BASE = os.environ.get("JEV_ARCHIVE_BASE", str(OUTPUT_DIR.with_name(f"{OUTPUT_DIR.name}_artifacts")))
P4B_ROWS = Path(os.environ.get("JEV_P4B_ROWS", "results/p4b_frozen_humaneval/p4b_frozen_case_results.jsonl"))

DISTILLED_RULES = {
    "HumanEval/41": "For pairwise interactions, derive the count from all participating pairs; do not assume one-to-one matching without proof.",
    "HumanEval/91": "Follow the exact observable specification. Do not add semantic conditions that are absent from the tests or contract.",
    "HumanEval/115": "Before aggregating capacities, decide whether resources are global, per item, or simultaneous; never interchange these models.",
    "HumanEval/120": "Separate item selection from output ordering. Verify both which elements are chosen and the required final order.",
    "HumanEval/129": "For path search, verify revisit rules, completeness, stopping conditions, and lexicographic tie-breaking before choosing DFS, BFS, or a heap.",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def load_p4b_module():
    candidates = [
        Path(os.environ.get(
            "JEV_P4B_SCRIPT",
            str(Path(__file__).with_name("p4b_frozen_humaneval_memory_transfer.py")),
        )),
    ]
    script = next((path for path in candidates if path.exists()), None)
    if script is None:
        raise FileNotFoundError("P4-B script is required as a frozen dependency")
    spec = importlib.util.spec_from_file_location("jev_p4b", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, script


def main() -> None:
    import torch
    import transformers

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError("P4-C requires CUDA")
    if not P4B_ROWS.exists():
        raise FileNotFoundError(f"Run P4-B first; missing {P4B_ROWS}")

    p4b, p4b_script = load_p4b_module()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    repo = p4b.clone_source()
    raw_memory, _ = p4b.build_public_memory(repo)
    tasks = p4b.select_frozen_tasks(repo)
    raw_retrieval = p4b.retrieve_memories(raw_memory, tasks, top_k=1)

    baseline_rows = {
        row["task_id"]: row
        for row in read_jsonl(P4B_ROWS)
        if row["arm"] == "baseline"
    }
    if set(baseline_rows) != {task["task_id"] for task in tasks}:
        raise RuntimeError("P4-B baseline task set does not match the frozen P4-C set")

    decisions: dict[str, dict[str, Any]] = {}
    for task in tasks:
        candidate = raw_retrieval[task["task_id"]][0]
        source = candidate["source_task_id"]
        score = float(candidate["similarity"])
        inject = score >= THRESHOLD
        decisions[task["task_id"]] = {
            "inject": inject,
            "threshold": THRESHOLD,
            "retrieval_score": score,
            "source_task_id": source,
            "source_memory_id": candidate["memory_id"],
            "distilled_rule": DISTILLED_RULES[source],
        }

    model, tokenizer = p4b.load_model()
    p4b.generate(model, tokenizer, tasks[0], None)  # unmeasured warm-up
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_id = task["task_id"]
        decision = decisions[task_id]
        baseline = baseline_rows[task_id]
        if decision["inject"]:
            memory = [{
                "memory_id": decision["source_memory_id"],
                "source_task_id": decision["source_task_id"],
                "similarity": decision["retrieval_score"],
                "reflection": decision["distilled_rule"],
            }]
            completion, seconds, input_tokens, output_tokens = p4b.generate(model, tokenizer, task, memory)
            evaluation = p4b.evaluate(task, completion)
            route = "distilled_memory"
        else:
            memory = []
            completion = baseline["completion"]
            seconds = 0.0
            input_tokens = baseline["input_tokens"]
            output_tokens = baseline["output_tokens"]
            evaluation = baseline["evaluation"]
            route = "baseline_fallback"
        row = {
            "schema_version": "JEV_P4C_SELECTIVE_MEMORY_V1",
            "task_id": task_id,
            "arm": "selective_distilled_memory",
            "route": route,
            "decision": decision,
            "retrieved_memory": memory,
            "completion": completion,
            "evaluation": evaluation,
            "latency_seconds": round(seconds, 6),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "baseline_passed": bool(baseline["evaluation"]["passed"]),
        }
        rows.append(row)
        print(task_id, route, evaluation["passed"], round(seconds, 3), flush=True)

    corrected = [r["task_id"] for r in rows if not r["baseline_passed"] and r["evaluation"]["passed"]]
    regressed = [r["task_id"] for r in rows if r["baseline_passed"] and not r["evaluation"]["passed"]]
    passed = sum(bool(r["evaluation"]["passed"]) for r in rows)
    injected = [r for r in rows if r["route"] == "distilled_memory"]
    report = {
        "schema_version": "JEV_P4C_SELECTIVE_MEMORY_REPORT_V1",
        "purpose": "Test whether distillation plus selective routing removes P4-B false transfer",
        "source": {"repository": p4b.REFLEXION_URL, "commit": p4b.REFLEXION_COMMIT, "license": "MIT"},
        "runtime": {
            "model": p4b.MODEL_ID,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "seed": SEED,
        },
        "design": {
            "public_data_only": True,
            "frozen_tasks": len(rows),
            "memory_build_tasks": len(raw_memory),
            "top_k": 1,
            "similarity_threshold": THRESHOLD,
            "threshold_pre_registered_before_run": True,
            "baseline_reused_from_p4b": True,
            "distillation": "fixed human-readable task-agnostic rules derived only from public Reflexion evidence",
            "injected_tasks": len(injected),
            "fallback_tasks": len(rows) - len(injected),
        },
        "baseline": {"passed": 24, "tasks": len(rows), "pass_rate": round(24 / len(rows), 6)},
        "p4b_raw_top3_memory": {"passed": 19, "tasks": len(rows), "pass_rate": round(19 / len(rows), 6), "net_corrections": -5},
        "p4c_selective_distilled_memory": {
            "passed": passed,
            "tasks": len(rows),
            "pass_rate": round(passed / len(rows), 6),
            "injected_tasks": len(injected),
            "mean_injected_latency_seconds": round(sum(r["latency_seconds"] for r in injected) / len(injected), 6) if injected else 0,
            "injected_input_tokens": sum(r["input_tokens"] for r in injected),
        },
        "effect_vs_baseline": {
            "corrected_tasks": corrected,
            "regressed_tasks": regressed,
            "net_corrections": len(corrected) - len(regressed),
            "absolute_pass_rate_delta": round(passed / len(rows) - 24 / len(rows), 6),
        },
        "decision": {
            "false_transfer_removed": len(regressed) == 0,
            "promote_to_long_horizon_pilot": len(regressed) == 0 and len(corrected) > 0,
            "production_memory_authorized": False,
        },
        "limitations": [
            "The distilled rules are fixed from public evidence but were manually phrased; automated distillation remains a later experiment.",
            "Baseline fallbacks reuse P4-B outputs exactly; only gated tasks incur a new generation.",
            "HumanEval is not a long-horizon environment.",
        ],
    }

    write_jsonl(OUTPUT_DIR / "p4c_case_results.jsonl", rows)
    write_jsonl(OUTPUT_DIR / "p4c_gate_decisions.jsonl", [{"task_id": k, **v} for k, v in decisions.items()])
    (OUTPUT_DIR / "p4c_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "README.md").write_text(
        "# P4-C selective distilled Memory\n\n"
        f"Baseline: 24/{len(rows)}\n\n"
        f"P4-B raw top-3 Memory: 19/{len(rows)}\n\n"
        f"P4-C selective distilled Memory: {passed}/{len(rows)}\n\n"
        f"Corrected: {corrected}; regressed: {regressed}.\n",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), OUTPUT_DIR / Path(__file__).name)
    shutil.copy2(p4b_script, OUTPUT_DIR / "p4b_frozen_dependency.py")
    shutil.copy2(P4B_ROWS, OUTPUT_DIR / "p4b_baseline_source_rows.jsonl")
    archive = shutil.make_archive(ARCHIVE_BASE, "zip", root_dir=OUTPUT_DIR.parent, base_dir=OUTPUT_DIR.name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Archive: {archive}")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

