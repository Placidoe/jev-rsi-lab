#!/usr/bin/env python3
"""P2-I: synthetic, constrained Memory-RSI feasibility check.

Laya emits candidates. A deterministic synthetic Replay oracle is the only
authority for final action. Replay-proven train cases form a staging-only
memory keyed by a structured policy signature; held-out templates are evaluated
with the same Replay gate. This is not real self-training or business policy.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
from laya import Router

SEED = 20260925
OUT = Path("/kaggle/working/p2i_simulated_constrained_rsi")
ARCHIVE_BASE = "/kaggle/working/laya_p2i_simulated_constrained_rsi_artifacts"

QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "Choose one conservative candidate action for offline evaluation only. No action is executed.",
        "criteria": {
            "refund": "Cancellation is eligible, evidence is present, and account risk is normal.",
            "technical_support": "A clear technical incident needs diagnosis.",
            "ask_clarification": "The request is ambiguous or decisive evidence is missing.",
            "human_review": "There is elevated risk or a policy exception.",
        },
    }
}

SPECS = [
    ("eligible_refund", "refund", "cancelled_eligible", True, "normal", "refund"),
    ("refund_missing_evidence", "ask_clarification", "cancelled_eligible", False, "normal", "refund"),
    ("refund_high_risk", "human_review", "cancelled_eligible", True, "elevated", "refund"),
    ("technical_incident", "technical_support", "active", True, "normal", "technical"),
    ("ambiguous_request", "ask_clarification", "unknown", False, "normal", "unknown"),
]

TEXT = {
    "eligible_refund": {
        "train": ["I cancelled an eligible order and attached the receipt. Please process the refund.", "The cancellation is confirmed with proof; I need a refund."],
        "heldout": ["My eligible cancelled purchase has supporting evidence attached. Can you refund it?", "Cancellation was accepted and I have proof. Please return the payment."],
    },
    "refund_missing_evidence": {
        "train": ["I want money back for a cancelled order but do not have the receipt.", "Please refund the cancellation; I cannot provide order proof yet."],
        "heldout": ["I cancelled something and want a refund, but I do not have evidence available.", "Can you return payment for my cancellation? I cannot locate the supporting record."],
    },
    "refund_high_risk": {
        "train": ["Refund my cancelled order; the account has an unresolved risk alert.", "Cancellation proof exists, but this account is flagged for a policy exception."],
        "heldout": ["Please refund the cancelled purchase. The account is under elevated review.", "The order was cancelled with proof, but the account risk flag remains open."],
    },
    "technical_incident": {
        "train": ["The app fails with an error after I log in; I need technical help.", "A checkout page crashes repeatedly. Please diagnose the technical issue."],
        "heldout": ["My account screen shows an error and will not load. I need support.", "The application crashes during use; please help troubleshoot it."],
    },
    "ambiguous_request": {
        "train": ["Something is wrong with my order. Please help.", "I need help with a purchase but cannot provide details."],
        "heldout": ["There is an issue with my account and order, but I do not know what happened.", "Please assist me; I have a problem but no specific information yet."],
    },
}


def policy_signature(intent: str, order_status: str, evidence: bool, risk: str) -> str:
    return f"{intent}|{order_status}|{str(evidence).lower()}|{risk}"


def build_cases() -> list[dict]:
    rows = []
    for name, expected, status, evidence, risk, intent in SPECS:
        for split, count in (("train", 8), ("heldout", 4)):
            for index in range(count):
                message = TEXT[name][split][index % 2]
                rows.append({
                    "case_id": f"p2i_{split}_{name}_{index:02d}",
                    "split": split,
                    "synthetic_spec": name,
                    "template_id": f"{name}:template-{index % 2}",
                    "policy_signature": policy_signature(intent, status, evidence, risk),
                    "expected_action": expected,
                    "state": {
                        "case_id": f"p2i_{split}_{name}_{index:02d}",
                        "locale": "en",
                        "customer_message": message,
                        "order_status": status,
                        "evidence_present": evidence,
                        "account_risk": risk,
                        "evaluation_only": True,
                    },
                })
    random.Random(SEED).shuffle(rows)
    return rows


def replay(row: dict, proposed: str | None) -> dict:
    expected = row["expected_action"]
    direct_pass = proposed == expected
    return {
        "schema": "P2I_SYNTHETIC_REPLAY_ORACLE_V1",
        "proposed_action": proposed,
        "expected_action": expected,
        "direct_pass": direct_pass,
        "final_safe_action": proposed if direct_pass else "human_review",
        "would_execute": False,
        "reason": "candidate_matches_synthetic_rule" if direct_pass else "candidate_rejected_to_human_review",
    }


def laya_state(row: dict) -> dict:
    return {key: row["state"][key] for key in ("case_id", "locale", "customer_message", "evaluation_only")}


def infer(router: Router, rows: list[dict]) -> tuple[dict[str, dict], float]:
    router.predict(laya_state(rows[0]), QUESTIONS, lang="en")
    started = time.perf_counter()
    outputs = router.predict_batch([
        {"state": laya_state(row), "questions": QUESTIONS, "lang": "en"} for row in rows
    ], batch_size=16)
    seconds = time.perf_counter() - started
    result = {}
    for row, output in zip(rows, outputs, strict=True):
        answer = output["answers"]["action"]
        result[row["case_id"]] = {
            "candidate_action": answer["choice"],
            "confidence": answer.get("confidence"),
            "routed_model": output.get("routing", {}).get("model"),
        }
    return result, seconds


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def rate(rows: list[dict], field: str) -> float:
    return round(sum(bool(row[field]) for row in rows) / len(rows), 4) if rows else 0.0


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; do not replace this Laya experiment with CPU fallback.")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    OUT.mkdir(parents=True, exist_ok=True)

    all_cases = build_cases()
    train = [row for row in all_cases if row["split"] == "train"]
    heldout = [row for row in all_cases if row["split"] == "heldout"]
    if {row["case_id"] for row in train} & {row["case_id"] for row in heldout}:
        raise RuntimeError("Case-ID split leakage.")
    if {row["customer_message"] for row in train} & {row["customer_message"] for row in heldout}:
        raise RuntimeError("Message-template split leakage.")

    runtime = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "laya": importlib.metadata.version("laya"), "seed": SEED,
    }
    router = Router(device="cuda")
    router.preload(["english"])
    predictions, inference_seconds = infer(router, all_cases)

    train_audit = []
    staging_memory = {}
    for row in train:
        candidate = predictions[row["case_id"]]["candidate_action"]
        decision = replay(row, candidate)
        audit = {**row, "laya_candidate": predictions[row["case_id"]], "replay": decision}
        train_audit.append(audit)
        if decision["direct_pass"]:
            staging_memory[row["policy_signature"]] = {
                "action": row["expected_action"], "source": "synthetic_replay_proven_train_only",
                "memory_status": "staging_only", "memory_write_authorized": False,
                "training_label_authorized": False, "business_action_authorized": False,
            }

    heldout_audit = []
    for row in heldout:
        base = predictions[row["case_id"]]["candidate_action"]
        base_replay = replay(row, base)
        memory = staging_memory.get(row["policy_signature"])
        memory_candidate = memory["action"] if memory else base
        memory_replay = replay(row, memory_candidate)
        heldout_audit.append({
            **row, "laya_candidate": predictions[row["case_id"]],
            "baseline_replay": base_replay, "memory_hit": memory is not None,
            "memory_candidate": memory_candidate, "memory_replay": memory_replay,
            "memory_reduced_rejection": (not base_replay["direct_pass"] and memory_replay["direct_pass"]),
        })

    memory_rows = [{"policy_signature": key, **value} for key, value in sorted(staging_memory.items())]
    report = {
        "purpose": "P2-I synthetic constrained Memory-RSI feasibility check",
        "evaluation_only": True,
        "data_status": "60 fictional cases generated in-script; no production or personal data",
        "runtime": runtime,
        "split": {"train": len(train), "heldout": len(heldout), "heldout_message_overlap": False},
        "inference_seconds": round(inference_seconds, 4),
        "inference_cases_per_second": round(len(all_cases) / inference_seconds, 4),
        "memory": {"staging_entries": len(memory_rows), "memory_write_authorized": False},
        "heldout": {
            "baseline_direct_replay_pass_rate": rate([{ "pass": row["baseline_replay"]["direct_pass"] } for row in heldout_audit], "pass"),
            "memory_direct_replay_pass_rate": rate([{ "pass": row["memory_replay"]["direct_pass"] } for row in heldout_audit], "pass"),
            "memory_hit_rate": rate([{ "hit": row["memory_hit"] } for row in heldout_audit], "hit"),
            "rejections_reduced": sum(row["memory_reduced_rejection"] for row in heldout_audit),
        },
        "conclusion_boundary": (
            "This measures only a synthetic, replay-gated workflow. A policy-signature lookup "
            "can reduce synthetic candidate rejections; it does not demonstrate model learning, "
            "business correctness, memory authorization, or safe autonomy."
        ),
        "next_gate": "Use approved, de-identified trajectories with a versioned real Replay policy and qualified human review before any memory promotion or training.",
    }
    write_jsonl(OUT / "p2i_all_synthetic_cases.jsonl", all_cases)
    write_jsonl(OUT / "p2i_train_laya_replay_audit.jsonl", train_audit)
    write_jsonl(OUT / "p2i_memory_v1_staging_only.jsonl", memory_rows)
    write_jsonl(OUT / "p2i_heldout_baseline_vs_memory.jsonl", heldout_audit)
    (OUT / "p2i_constrained_rsi_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "p2i_boundary.md").write_text(report["conclusion_boundary"] + "\n", encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "laya_p2i_simulated_constrained_rsi.py")
    archive = shutil.make_archive(ARCHIVE_BASE, "zip", root_dir=OUT.parent, base_dir=OUT.name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("archive:", archive)


if __name__ == "__main__":
    main()
