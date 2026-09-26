#!/usr/bin/env python3
"""P4-A: Convert official Reflexion runs into auditable JEV-RSI Memory evidence.

The script is intentionally model-free.  It audits already-published public
results before any new GPU experiment is allowed to consume their reflections.

Inputs:
  * noahshinn/reflexion, cloned from the official public GitHub repository
  * HumanEval simple and Reflexion run JSONL files shipped by that repository
  * WebShop base and Reflexion cumulative trial summaries shipped by it

Outputs never claim causality that the logs cannot support.  HumanEval has a
paired baseline/reflexion comparison with public unit tests.  WebShop summaries
only contain cumulative success and memory text, so they are treated as
observational evidence and are not used to authorize production Memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SEED = 20260927
SOURCE_URL = "https://github.com/noahshinn/reflexion"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_split(task_id: str) -> str:
    value = int(hashlib.sha256(f"{SEED}|{task_id}".encode()).hexdigest()[:8], 16) % 10
    return "memory_build" if value < 7 else "frozen_test"


def bootstrap_paired_delta(pairs: list[tuple[int, int]], samples: int = 20000) -> dict[str, float]:
    rng = random.Random(SEED)
    n = len(pairs)
    deltas: list[float] = []
    for _ in range(samples):
        picked = [pairs[rng.randrange(n)] for _ in range(n)]
        deltas.append(sum(after - before for before, after in picked) / n)
    deltas.sort()
    return {
        "mean": round(sum(after - before for before, after in pairs) / n, 6),
        "ci95_low": round(deltas[int(0.025 * samples)], 6),
        "ci95_high": round(deltas[int(0.975 * samples)], 6),
        "bootstrap_samples": samples,
    }


def exact_mcnemar(baseline_only: int, reflexion_only: int) -> dict[str, Any]:
    discordant = baseline_only + reflexion_only
    if discordant == 0:
        return {"discordant": 0, "two_sided_exact_p": 1.0}
    lower = min(baseline_only, reflexion_only)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
    return {"discordant": discordant, "two_sided_exact_p": round(min(1.0, 2 * tail), 8)}


def index_unique(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row["task_id"]
        if task_id in indexed:
            raise ValueError(f"Duplicate {task_id} in {name}")
        indexed[task_id] = row
    return indexed


def audit_humaneval(repo: Path, output_dir: Path) -> dict[str, Any]:
    baseline_path = repo / "programming_runs/root/simple_human_eval_py_logging/humaneval-py..gz_simple_1_gpt-4_pass_at_k_1_py.jsonl"
    reflexion_path = repo / "programming_runs/root/reflexion_humaneval_py_pass_at_1/reflexion_humaneval_py_pass_at_1.jsonl"
    baseline = index_unique(read_jsonl(baseline_path), "HumanEval baseline")
    reflexion = index_unique(read_jsonl(reflexion_path), "HumanEval reflexion")
    task_ids = sorted(set(baseline) & set(reflexion), key=lambda x: int(x.split("/")[-1]))
    if len(task_ids) != 164:
        raise ValueError(f"Expected 164 paired HumanEval tasks, found {len(task_ids)}")

    joined: list[dict[str, Any]] = []
    memory: list[dict[str, Any]] = []
    counts = Counter()
    for task_id in task_ids:
        before = baseline[task_id]
        after = reflexion[task_id]
        before_ok = bool(before["is_solved"])
        after_ok = bool(after["is_solved"])
        reflections = [str(x).strip() for x in after.get("reflections", []) if str(x).strip()]
        if not before_ok and after_ok:
            transition = "corrected"
        elif before_ok and not after_ok:
            transition = "regressed"
        elif before_ok and after_ok:
            transition = "stable_success"
        else:
            transition = "stable_failure"
        counts[transition] += 1

        row_split = task_split(task_id)
        if transition == "corrected" and reflections and row_split == "memory_build":
            gate = "evidence_backed_candidate"
            disposition = "staging_only"
        elif transition == "corrected" and reflections:
            gate = "frozen_evidence_withheld"
            disposition = "do_not_write"
        elif transition == "stable_failure" and reflections:
            gate = "observed_ineffective"
            disposition = "quarantine"
        elif transition == "regressed":
            gate = "possible_harm"
            disposition = "reject"
        else:
            gate = "no_incremental_memory_evidence"
            disposition = "do_not_write"

        evidence_id = hashlib.sha256(
            json.dumps(
                {"task_id": task_id, "before": before_ok, "after": after_ok, "reflections": reflections},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        row = {
            "schema_version": "JEV_PUBLIC_REFLEXION_AUDIT_V1",
            "source": "noahshinn/reflexion",
            "source_url": SOURCE_URL,
            "domain": "HumanEval",
            "task_id": task_id,
            "split": row_split,
            "baseline_solved": before_ok,
            "reflexion_solved": after_ok,
            "transition": transition,
            "reflections": reflections,
            "memory_gate": gate,
            "memory_disposition": disposition,
            "evidence_id": evidence_id,
            "production_memory_authorized": False,
        }
        joined.append(row)
        if reflections:
            memory.append(
                {
                    **row,
                    "task_prompt": after["prompt"],
                    "entry_point": after["entry_point"],
                    "reflection_text": "\n".join(reflections),
                    "final_solution": after["solution"],
                    "objective_evaluator": "HumanEval public unit tests",
                }
            )

    pairs = [(int(row["baseline_solved"]), int(row["reflexion_solved"])) for row in joined]
    baseline_only = counts["regressed"]
    reflexion_only = counts["corrected"]
    write_jsonl(output_dir / "p4a_humaneval_paired_cases.jsonl", joined)
    write_jsonl(output_dir / "p4a_humaneval_memory_evidence.jsonl", memory)
    return {
        "paired_tasks": len(joined),
        "baseline_solved": sum(row["baseline_solved"] for row in joined),
        "baseline_pass_rate": round(sum(row["baseline_solved"] for row in joined) / len(joined), 6),
        "reflexion_solved": sum(row["reflexion_solved"] for row in joined),
        "reflexion_pass_rate": round(sum(row["reflexion_solved"] for row in joined) / len(joined), 6),
        "paired_delta": bootstrap_paired_delta(pairs),
        "transitions": dict(counts),
        "mcnemar": exact_mcnemar(baseline_only, reflexion_only),
        "tasks_with_reflections": sum(bool(row["reflections"]) for row in joined),
        "evidence_backed_reflections_total": sum(
            row["transition"] == "corrected" and bool(row["reflections"]) for row in joined
        ),
        "eligible_memory_build_candidates": sum(row["memory_gate"] == "evidence_backed_candidate" for row in joined),
        "frozen_evidence_withheld": sum(row["memory_gate"] == "frozen_evidence_withheld" for row in joined),
        "quarantined_ineffective_reflections": sum(row["memory_gate"] == "observed_ineffective" for row in joined),
        "source_files": {
            "baseline": {"path": str(baseline_path), "sha256": sha256(baseline_path)},
            "reflexion": {"path": str(reflexion_path), "sha256": sha256(reflexion_path)},
        },
    }


def load_trials(folder: Path) -> list[list[dict[str, Any]]]:
    files = sorted(folder.glob("env_results_trial_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


def trial_summary(trials: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    previous = 0
    result = []
    for index, rows in enumerate(trials):
        solved = sum(bool(row["is_success"]) for row in rows)
        result.append(
            {
                "trial": index,
                "tasks": len(rows),
                "cumulative_success": solved,
                "cumulative_success_rate": round(solved / len(rows), 6),
                "new_successes": solved - previous,
                "tasks_with_memory": sum(bool(row.get("memory")) for row in rows),
                "memory_entries": sum(len(row.get("memory", [])) for row in rows),
            }
        )
        previous = solved
    return result


def audit_webshop(repo: Path, output_dir: Path) -> dict[str, Any]:
    base_folder = repo / "webshop_runs/base_run_logs_2"
    reflexion_folder = repo / "webshop_runs/reflexion_run_logs_2"
    base_trials = load_trials(base_folder)
    reflexion_trials = load_trials(reflexion_folder)
    if len(base_trials) != 4 or len(reflexion_trials) != 4:
        raise ValueError("Expected four published WebShop trials for base_run_logs_2 and reflexion_run_logs_2")

    evidence: list[dict[str, Any]] = []
    for trial_index in range(len(reflexion_trials) - 1):
        current = {row["name"]: row for row in reflexion_trials[trial_index]}
        nxt = {row["name"]: row for row in reflexion_trials[trial_index + 1]}
        for name in sorted(current, key=lambda x: int(x.split("_")[-1])):
            before = current[name]
            after = nxt[name]
            before_mem = before.get("memory", [])
            after_mem = after.get("memory", [])
            appended = after_mem[len(before_mem) :] if after_mem[: len(before_mem)] == before_mem else []
            converted = not before["is_success"] and bool(after["is_success"])
            evidence.append(
                {
                    "schema_version": "JEV_PUBLIC_REFLEXION_AUDIT_V1",
                    "source": "noahshinn/reflexion",
                    "source_url": SOURCE_URL,
                    "domain": "WebShop",
                    "task_id": name,
                    "trial_before": trial_index,
                    "trial_after": trial_index + 1,
                    "before_solved": bool(before["is_success"]),
                    "after_solved": bool(after["is_success"]),
                    "converted_after_memory": converted,
                    "memory_visible_before_retry": before_mem[-3:],
                    "memory_appended_after_trial": appended,
                    "memory_gate": "observational_positive" if converted else "insufficient_or_no_incremental_evidence",
                    "memory_disposition": "staging_only" if converted else "do_not_promote",
                    "production_memory_authorized": False,
                    "limitation": "Published summary omits action trajectory and reward magnitude; conversion is association, not isolated causality.",
                }
            )
    write_jsonl(output_dir / "p4a_webshop_memory_transitions.jsonl", evidence)
    base = trial_summary(base_trials)
    reflexion = trial_summary(reflexion_trials)
    return {
        "base_trials": base,
        "reflexion_trials": reflexion,
        "trial3_cumulative_delta": round(
            reflexion[-1]["cumulative_success_rate"] - base[-1]["cumulative_success_rate"], 6
        ),
        "reflexion_conversions_observed": sum(row["converted_after_memory"] for row in evidence),
        "interpretation_boundary": "Observational audit only; success carry-forward and missing action logs prevent a clean causal JEV estimate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("work/reflexion_public_pinned"))
    parser.add_argument("--output", type=Path, default=Path("results/p4a_public_reflexion_audit"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()

    report = {
        "schema_version": "JEV_P4A_PUBLIC_REFLEXION_REPORT_V1",
        "purpose": "Audit public LLM self-reflection data for JEV-RSI Memory admission and long-horizon follow-up experiments",
        "source": {"repository": SOURCE_URL, "commit": commit, "license": "MIT"},
        "seed": SEED,
        "humaneval": audit_humaneval(repo, output_dir),
        "webshop": audit_webshop(repo, output_dir),
        "decision": {
            "public_data_only": True,
            "production_memory_authorized": False,
            "next_experiment": "Frozen HumanEval baseline vs JEV retrieval-augmented memory using the same open model and objective unit tests",
            "why": "The official logs can identify evidence-backed candidate reflections, but only a new frozen-task intervention can measure JEV transfer.",
        },
    }
    (output_dir / "p4a_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = f"""# P4-A public Reflexion audit

- Source: {SOURCE_URL}
- Commit: `{commit}`
- License: MIT
- HumanEval paired tasks: {report['humaneval']['paired_tasks']}
- Baseline pass rate: {report['humaneval']['baseline_pass_rate']:.2%}
- Reflexion pass rate: {report['humaneval']['reflexion_pass_rate']:.2%}
- Paired uplift: {report['humaneval']['paired_delta']['mean']:.2%}
- 95% bootstrap CI: [{report['humaneval']['paired_delta']['ci95_low']:.2%}, {report['humaneval']['paired_delta']['ci95_high']:.2%}]
- Corrected tasks: {report['humaneval']['transitions'].get('corrected', 0)}
- Regressed tasks: {report['humaneval']['transitions'].get('regressed', 0)}
- Evidence-backed reflections before split: {report['humaneval']['evidence_backed_reflections_total']}
- Eligible memory-build candidates: {report['humaneval']['eligible_memory_build_candidates']}
- Frozen evidence withheld from Memory: {report['humaneval']['frozen_evidence_withheld']}
- Ineffective reflections quarantined: {report['humaneval']['quarantined_ineffective_reflections']}

This archive is an audit of public evidence, not proof that JEV itself caused an
improvement.  P4-B must run the same open model on a frozen set with and without
JEV-retrieved memory under identical generation and execution settings.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    shutil.copy2(Path(__file__), output_dir / Path(__file__).name)
    archive = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Archive: {archive}")


if __name__ == "__main__":
    main()

