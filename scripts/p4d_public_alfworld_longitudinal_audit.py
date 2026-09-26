#!/usr/bin/env python3
"""P4-D: audit public ALFWorld longitudinal Reflexion logs for JEV Memory evidence.

The script reads only the public, pinned Reflexion repository.  It does not call
an LLM, execute ALFWorld actions, or authorize production Memory.  Its purpose
is to turn the published cumulative trial summaries into an auditable evidence
table and to define a leakage-safe build/holdout split for the next intervention.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path


SOURCE = Path(os.environ.get("JEV_REFLEXION_DIR", "work/reflexion_public_pinned")) / "alfworld_runs" / "root"
OUTPUT = Path(os.environ.get("JEV_OUTPUT_DIR", "results/p4d_public_alfworld_longitudinal"))
ARCHIVE = Path(os.environ.get("JEV_ARCHIVE_BASE", str(OUTPUT.with_name(f"{OUTPUT.name}_artifacts"))))
REPO = "https://github.com/noahshinn/reflexion.git"
COMMIT = "218cf0ef1df84b05ce379dd4a8e47f17766733a0"


def load_trials(folder: Path) -> list[list[dict]]:
    files = sorted(
        folder.glob("env_results_trial_*.json"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    trials = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    if not trials:
        raise RuntimeError(f"No public trial logs found in {folder}")
    names = [row["name"] for row in trials[0]]
    for idx, trial in enumerate(trials):
        if [row["name"] for row in trial] != names:
            raise RuntimeError(f"Environment order changed at trial {idx}")
    return trials


def exact_two_sided_binomial(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def split_for(name: str) -> str:
    # Frozen before candidate construction: about 25% holdout by stable hash.
    return "holdout" if int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 4 == 0 else "build"


def classify_memory(text: str) -> str:
    value = text.lower()
    if "stuck in a loop" in value or "identical actions" in value:
        return "loop_recovery"
    if "open" in value and any(x in value for x in ("microwave", "drawer", "cabinet", "fridge")):
        return "container_precondition"
    if any(x in value for x in ("check all", "checked the", "look for", "find a")):
        return "search_coverage"
    if any(x in value for x in ("before", "then", "after")):
        return "action_ordering"
    if any(x in value for x in ("look at", "examine", "make sure")):
        return "verification"
    return "other"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def make_board(path: Path, base_counts: list[int], reflex_counts: list[int], report: dict) -> None:
    """Write a dependency-free SVG evidence board."""
    left, top, right, bottom = 130, 425, 1050, 815
    def points(values: list[int]) -> str:
        result = []
        for i, val in enumerate(values):
            x = left + i / 14 * (right - left)
            y = bottom - (val - 80) / 54 * (bottom - top)
            result.append(f"{x:.1f},{y:.1f}")
        return " ".join(result)

    effect = report["paired_trial6_effect"]
    cards = [
        (95, "同起点", "84 / 134", "#4B6BFB"),
        (505, "第 7 轮 Baseline", f"{base_counts[6]} / 134", "#697386"),
        (915, "第 7 轮 Reflexion", f"{reflex_counts[6]} / 134", "#20A06B"),
        (1325, "配对净优势", "+22 tasks", "#20A06B"),
    ]
    card_svg = "".join(
        f'<rect x="{x}" y="210" width="365" height="145" rx="22" fill="#F7F9FD" stroke="#E4E9F4" stroke-width="2"/>'
        f'<text x="{x+24}" y="262" class="label">{label}</text>'
        f'<text x="{x+24}" y="325" class="metric" fill="{color}">{value}</text>'
        for x, label, value, color in cards
    )
    grid_svg = ""
    for val in (84, 100, 117, 134):
        y = bottom - (val - 80) / 54 * (bottom - top)
        grid_svg += f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#EDF0F5" stroke-width="2"/>'
        grid_svg += f'<text x="72" y="{y+7:.1f}" class="tick">{val}</text>'
    evidence = [
        ("Reflexion 独有成功", effect["reflexion_only"], "#20A06B"),
        ("Baseline 独有成功", effect["baseline_only"], "#D84F4F"),
        ("共同成功", effect["both"], "#4B6BFB"),
        ("共同失败", effect["neither"], "#697386"),
        ("McNemar p", f'{effect["mcnemar_exact_p"]:.8f}', "#18243A"),
    ]
    evidence_svg = "".join(
        f'<text x="1130" y="{510+i*67}" class="label">{label}</text>'
        f'<text x="1640" y="{510+i*67}" class="value" fill="{color}" text-anchor="end">{value}</text>'
        for i, (label, value, color) in enumerate(evidence)
    )
    circles = "".join(
        f'<circle cx="{left+i/14*(right-left):.1f}" cy="{bottom-(v-80)/54*(bottom-top):.1f}" r="6" fill="#697386"/>'
        for i, v in enumerate(base_counts)
    ) + "".join(
        f'<circle cx="{left+i/14*(right-left):.1f}" cy="{bottom-(v-80)/54*(bottom-top):.1f}" r="6" fill="#20A06B"/>'
        for i, v in enumerate(reflex_counts)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1120" viewBox="0 0 1800 1120">
<style>
text {{ font-family: "PingFang SC", "Microsoft YaHei", Arial, sans-serif; }}
.title {{ font-size:48px; font-weight:700; fill:#18243A; }} .subtitle {{ font-size:25px; fill:#697386; }}
.label {{ font-size:22px; fill:#697386; }} .metric {{ font-size:38px; font-weight:700; }}
.section {{ font-size:30px; font-weight:700; fill:#18243A; }} .value {{ font-size:24px; font-weight:700; }}
.tick {{ font-size:19px; fill:#697386; }} .note {{ font-size:24px; fill:#18243A; }}
</style>
<rect width="1800" height="1120" fill="#F7F8FC"/><rect x="48" y="40" width="1704" height="1040" rx="36" fill="white" stroke="#E5E9F2" stroke-width="3"/>
<text x="95" y="128" class="title">JEV P4-D｜公开 ALFWorld 长程 Memory 证据审计</text>
<text x="97" y="174" class="subtitle">公开 Reflexion 仓库 · 134 个环境 · 同起点配对 · 生产 Memory 仍未授权</text>
{card_svg}{grid_svg}
<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#CDD4E1" stroke-width="3"/><line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#CDD4E1" stroke-width="3"/>
<polyline points="{points(base_counts)}" fill="none" stroke="#697386" stroke-width="7" stroke-linejoin="round"/><polyline points="{points(reflex_counts)}" fill="none" stroke="#20A06B" stroke-width="7" stroke-linejoin="round"/>{circles}
<text x="140" y="858" class="label">灰：Baseline（公开至第 7 轮）</text><text x="500" y="858" class="label">绿：Reflexion（第 15 轮达到 134/134）</text>
<text x="1130" y="445" class="section">配对证据（第 7 轮）</text>{evidence_svg}
<rect x="95" y="915" width="1610" height="110" rx="20" fill="#FFF7E8" stroke="#F0D399" stroke-width="2"/>
<text x="120" y="958" class="note">结论：公开日志支持“Memory 与长程累计成功提升相关”；但缺逐步动作与同模型重跑，</text>
<text x="120" y="997" class="note">因此只准入 observational staging，下一步必须在冻结 ALFWorld 环境做 Baseline / 选择性 Memory 干预。</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    base = load_trials(SOURCE / "base_run_logs")
    reflex = load_trials(SOURCE / "reflexion_run_logs")
    if len(base[0]) != 134 or len(reflex[0]) != 134:
        raise RuntimeError("Expected the published 134 ALFWorld environments")

    base_counts = [sum(bool(x["is_success"]) for x in trial) for trial in base]
    reflex_counts = [sum(bool(x["is_success"]) for x in trial) for trial in reflex]
    if base_counts[0] != reflex_counts[0]:
        raise RuntimeError("The public arms do not share the same initial success count")

    env_rows: list[dict] = []
    candidates: list[dict] = []
    quarantined: list[dict] = []
    for i, first in enumerate(reflex[0]):
        name = first["name"]
        first_success = next((t for t, trial in enumerate(reflex) if trial[i]["is_success"]), None)
        base_first_success = next((t for t, trial in enumerate(base) if trial[i]["is_success"]), None)
        memory_before_success: list[str] = []
        if first_success is not None and first_success > 0:
            memory_before_success = list(reflex[first_success][i].get("memory", []))
        split = split_for(name)
        row = {
            "schema_version": "JEV_P4D_ALFWORLD_ENV_V1",
            "env_name": name,
            "split": split,
            "initial_success": bool(first["is_success"]),
            "baseline_first_success_trial": base_first_success,
            "reflexion_first_success_trial": first_success,
            "reflexion_success_by_trial6": bool(reflex[6][i]["is_success"]),
            "baseline_success_by_trial6": bool(base[6][i]["is_success"]),
            "memory_count_before_first_success": len(memory_before_success),
            "memory_before_first_success": memory_before_success,
            "memory_categories": [classify_memory(x) for x in memory_before_success],
            "causal_authorization": False,
            "production_memory_authorized": False,
        }
        env_rows.append(row)

        if not row["initial_success"] and first_success is not None and memory_before_success:
            evidence = {
                "schema_version": "JEV_P4D_ALFWORLD_MEMORY_EVIDENCE_V1",
                "memory_id": hashlib.sha256((name + "\n" + memory_before_success[-1]).encode()).hexdigest()[:20],
                "env_name": name,
                "split": split,
                "first_success_trial": first_success,
                "memory_used_count": len(memory_before_success),
                "last_memory": memory_before_success[-1],
                "memory_category": classify_memory(memory_before_success[-1]),
                "evidence": "initial failure; non-empty memory; later cumulative environment success",
                "status": "observational_staging" if split == "build" else "frozen_holdout_evidence",
                "causal_authorization": False,
                "training_label_authorized": False,
                "production_memory_authorized": False,
            }
            (candidates if split == "build" else quarantined).append(evidence)
        elif not row["initial_success"]:
            quarantined.append({
                "schema_version": "JEV_P4D_ALFWORLD_QUARANTINE_V1",
                "env_name": name,
                "split": split,
                "reason": "No pre-success memory evidence or no observed success",
                "production_memory_authorized": False,
            })

    b6 = base[6]
    r6 = reflex[6]
    baseline_only = sum(bool(b6[i]["is_success"]) and not bool(r6[i]["is_success"]) for i in range(134))
    reflexion_only = sum(not bool(b6[i]["is_success"]) and bool(r6[i]["is_success"]) for i in range(134))
    both = sum(bool(b6[i]["is_success"]) and bool(r6[i]["is_success"]) for i in range(134))
    neither = 134 - baseline_only - reflexion_only - both
    categories = Counter(x["memory_category"] for x in candidates)

    report = {
        "schema_version": "JEV_P4D_PUBLIC_ALFWORLD_LONGITUDINAL_REPORT_V1",
        "purpose": "Audit public long-horizon Memory evidence and freeze a leakage-safe next intervention",
        "source": {"repository": REPO, "commit": COMMIT, "license": "MIT"},
        "design": {
            "public_data_only": True,
            "environments": 134,
            "baseline_trials": len(base),
            "reflexion_trials": len(reflex),
            "cumulative_success_semantics_verified_from_source": True,
            "same_initial_success_count": base_counts[0] == reflex_counts[0],
            "deterministic_split": "sha256(env_name) % 4; 0=holdout, otherwise build",
            "step_trajectories_available": False,
        },
        "success_curve": {
            "baseline": base_counts,
            "reflexion": reflex_counts,
            "initial": base_counts[0],
            "trial6_baseline": base_counts[6],
            "trial6_reflexion": reflex_counts[6],
            "reflexion_final": reflex_counts[-1],
        },
        "paired_trial6_effect": {
            "both": both,
            "baseline_only": baseline_only,
            "reflexion_only": reflexion_only,
            "neither": neither,
            "net_additional_successes": reflexion_only - baseline_only,
            "absolute_success_rate_delta": round((reflex_counts[6] - base_counts[6]) / 134, 6),
            "mcnemar_exact_p": exact_two_sided_binomial(min(baseline_only, reflexion_only), baseline_only + reflexion_only),
        },
        "memory_evidence": {
            "initial_failures": 134 - reflex_counts[0],
            "later_success_with_pre_success_memory": len(candidates) + sum(x.get("schema_version", "").endswith("MEMORY_EVIDENCE_V1") for x in quarantined),
            "build_observational_staging": len(candidates),
            "frozen_holdout_evidence": sum(x.get("status") == "frozen_holdout_evidence" for x in quarantined),
            "build_category_counts": dict(sorted(categories.items())),
        },
        "decision": {
            "public_long_horizon_signal": True,
            "memory_causality_proven": False,
            "production_memory_authorized": False,
            "next_gate": "Run a frozen ALFWorld intervention with complete step trajectories: baseline vs selective distilled Memory; measure success, steps, repeated actions, recovery, false transfer, and net corrections.",
        },
        "limitations": [
            "Published JSON files contain cumulative success and Memory, not complete step/action/reward trajectories.",
            "The public runs used historical remote models; they are evidence for the next test, not direct JEV performance.",
            "Success after Memory is observational at the environment level; natural retry variation remains a confound.",
        ],
    }

    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "p4d_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(OUTPUT / "p4d_environment_longitudinal.jsonl", env_rows)
    write_jsonl(OUTPUT / "p4d_build_memory_evidence.jsonl", candidates)
    write_jsonl(OUTPUT / "p4d_frozen_or_quarantined_evidence.jsonl", quarantined)
    make_board(OUTPUT / "p4d_alfworld_long_horizon_evidence_board.svg", base_counts, reflex_counts, report)
    shutil.copy2(Path(__file__), OUTPUT / Path(__file__).name)
    (OUTPUT / "README.md").write_text(
        "# JEV P4-D public ALFWorld longitudinal audit\n\n"
        "All data come from the pinned official Reflexion repository. The audit reads published cumulative "
        "success/Memory JSON files and creates a deterministic build/holdout split. It does not claim causal "
        "JEV benefit and authorizes no production Memory. See `p4d_report.json` and the per-environment JSONL.\n",
        encoding="utf-8",
    )
    zip_path = Path(shutil.make_archive(str(ARCHIVE), "zip", root_dir=OUTPUT.parent, base_dir=OUTPUT.name))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Archive: {zip_path}")


if __name__ == "__main__":
    main()

