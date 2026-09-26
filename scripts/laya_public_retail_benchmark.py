"""Public-data semantic-routing benchmark for Laya on Kaggle T4.

Purpose
-------
This script evaluates *semantic routing only* on a public e-commerce dataset.
It does NOT decide whether a refund may be executed, and it never writes Memory,
trains a model, or calls a business tool.

Data source
-----------
bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset
The data is English and its labels are mapped into three coarse semantic routes:
  * refund_related     <- request_refund / refund_status / refund_policy
  * technical_support  <- technical_issue
  * human_review       <- human_agent

Run in Kaggle
-------------
1. Settings -> Accelerator -> GPU T4 x2
2. Turn Internet ON (first run downloads packages, Laya weights, and public data)
3. Upload this file or paste it into /kaggle/working, then run:
       !python /kaggle/working/laya_public_retail_benchmark.py

Outputs are written to /kaggle/working/public_retail_benchmark/.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATASET_ID = "bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset"
OUTPUT_DIR = Path("/kaggle/working/public_retail_benchmark")
SEED = "jev-laya-public-retail-v1"
PER_ROUTE_LIMIT = 180  # 108 train / 36 validation / 36 frozen-test per route
BATCH_SIZE = 8

# This is intentionally semantic routing, NOT permission to execute an action.
INTENT_TO_ROUTE = {
    "request_refund": "refund_related",
    "refund_status": "refund_related",
    "refund_policy": "refund_related",
    "technical_issue": "technical_support",
    "human_agent": "human_review",
}

QUESTIONS = {
    "semantic_route": {
        "type": "choice",
        "instructions": (
            "Classify the customer's primary support intent. This is semantic routing only; "
            "do not infer permissions, eligibility, or execute an action."
        ),
        "criteria": {
            "refund_related": "The user asks about requesting, tracking, or understanding a refund/return.",
            "technical_support": "The primary problem is an app, website, login, or product technical issue.",
            "human_review": "The user explicitly requests a human support agent or escalation.",
        },
    }
}


def install_and_check_runtime() -> dict[str, Any]:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "laya", "datasets"]
    )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Enable Kaggle Settings -> Accelerator -> GPU T4 x2.")
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


def stable_rank(row: dict[str, Any]) -> str:
    payload = f"{SEED}|{row['source_intent']}|{row['instruction']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_public_splits() -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    from datasets import load_dataset

    loaded = load_dataset(DATASET_ID)
    split_name = "train" if "train" in loaded else next(iter(loaded.keys()))
    source = loaded[split_name]

    required = {"instruction", "intent"}
    missing = required - set(source.column_names)
    if missing:
        raise RuntimeError(
            f"Public dataset schema changed; missing {sorted(missing)}. "
            f"Found columns: {source.column_names}"
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_intent_counts = Counter()
    for row in source:
        intent = str(row["intent"]).strip()
        route = INTENT_TO_ROUTE.get(intent)
        if not route:
            continue
        instruction = str(row["instruction"]).strip()
        if not instruction:
            continue
        source_intent_counts[intent] += 1
        grouped[route].append(
            {
                "dataset_id": DATASET_ID,
                "dataset_split": split_name,
                "source_intent": intent,
                "semantic_route": route,
                "instruction": instruction,
                "source_category": row.get("category"),
                "source_response": row.get("response"),
            }
        )

    missing_routes = set(INTENT_TO_ROUTE.values()) - set(grouped)
    if missing_routes:
        raise RuntimeError(
            f"Required routes absent: {sorted(missing_routes)}. "
            f"Available mapped intents: {dict(source_intent_counts)}"
        )

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "frozen_test": []}
    for route, rows in grouped.items():
        selected = sorted(rows, key=stable_rank)[:PER_ROUTE_LIMIT]
        if len(selected) < PER_ROUTE_LIMIT:
            raise RuntimeError(f"Only {len(selected)} usable rows for {route}; expected {PER_ROUTE_LIMIT}.")
        train_end = int(PER_ROUTE_LIMIT * 0.60)
        validation_end = int(PER_ROUTE_LIMIT * 0.80)
        splits["train"].extend(selected[:train_end])
        splits["validation"].extend(selected[train_end:validation_end])
        splits["frozen_test"].extend(selected[validation_end:])

    # A stable pseudo-ID prevents raw source position from becoming a later training key.
    for split_name, rows in splits.items():
        for index, row in enumerate(rows):
            row["case_id"] = f"public-retail-{split_name}-{index:04d}"
            row["evidence_level"] = "public_synthetic_semantic_benchmark_not_memory_or_tool_evidence"
            row["execution_allowed"] = False

    return splits, dict(source_intent_counts)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted(set(INTENT_TO_ROUTE.values()))
    matrix = {expected: {predicted: 0 for predicted in labels} for expected in labels}
    for row in rows:
        matrix[row["semantic_route"]][row["laya"]["candidate_route"]] += 1
    correct = sum(row["semantic_route"] == row["laya"]["candidate_route"] for row in rows)
    return {
        "count": len(rows),
        "accuracy": round(correct / len(rows), 4),
        "correct": correct,
        "confusion_matrix": matrix,
    }


def main() -> None:
    runtime = install_and_check_runtime()
    splits, source_intent_counts = build_public_splits()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Persist the split BEFORE inference. frozen_test must not be used for tuning.
    for split_name, rows in splits.items():
        path = OUTPUT_DIR / f"{split_name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    from laya import Router

    print("Runtime:", json.dumps(runtime, ensure_ascii=False))
    print("Loading only the English Laya checkpoint for this English public benchmark...")
    router = Router(device="cuda")
    router.preload(["english"])

    frozen_rows = splits["frozen_test"]
    warmup_requests = [
        {
            "state": {"locale": "en-US", "user_message": row["instruction"]},
            "questions": QUESTIONS,
            "lang": "en",
        }
        for row in frozen_rows[:2]
    ]
    _ = router.predict_batch(warmup_requests, batch_size=2)

    requests = [
        {
            "state": {"locale": "en-US", "user_message": row["instruction"]},
            "questions": QUESTIONS,
            "lang": "en",
        }
        for row in frozen_rows
    ]
    started = time.perf_counter()
    decisions = router.predict_batch(requests, batch_size=BATCH_SIZE)
    elapsed = time.perf_counter() - started

    for row, decision in zip(frozen_rows, decisions):
        answer = decision["answers"]["semantic_route"]
        row["laya"] = {
            "routed_model": decision["routing"]["model"],
            "candidate_route": answer["choice"],
            "confidence_observed_not_used_for_gating": answer["confidence"],
        }
        row["match"] = row["semantic_route"] == answer["choice"]

    frozen_path = OUTPUT_DIR / "frozen_test_with_laya.jsonl"
    frozen_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in frozen_rows), encoding="utf-8"
    )
    errors = [row for row in frozen_rows if not row["match"]]
    errors_path = OUTPUT_DIR / "frozen_test_errors.jsonl"
    errors_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in errors), encoding="utf-8")

    report = {
        "purpose": "public e-commerce semantic-routing benchmark; not an executable refund benchmark",
        "dataset": {
            "id": DATASET_ID,
            "language": "English",
            "source_intent_counts_before_sampling": source_intent_counts,
            "intent_to_route": INTENT_TO_ROUTE,
            "split_counts": {name: len(rows) for name, rows in splits.items()},
        },
        "runtime": runtime,
        "inference": {
            "checkpoint_preloaded": ["english"],
            "frozen_test_elapsed_seconds": round(elapsed, 4),
            "frozen_test_cases_per_second": round(len(frozen_rows) / elapsed, 4),
            "batch_size": BATCH_SIZE,
        },
        "frozen_test_metrics": metrics(frozen_rows),
        "safety_boundary": [
            "Public labels test semantic route only; no public row authorizes a business action.",
            "No tool call, replay pass, Memory write, prompt update, or model training occurs in this script.",
            "Do not use observed confidence for automatic execution until calibrated on an approved evaluation set.",
        ],
        "artifacts": {
            "train": str(OUTPUT_DIR / "train.jsonl"),
            "validation": str(OUTPUT_DIR / "validation.jsonl"),
            "frozen_test_source": str(OUTPUT_DIR / "frozen_test.jsonl"),
            "frozen_test_with_laya": str(frozen_path),
            "errors": str(errors_path),
        },
    }
    report_path = OUTPUT_DIR / "public_retail_laya_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Public retail semantic-routing benchmark ===")
    print(json.dumps(report["frozen_test_metrics"], ensure_ascii=False, indent=2))
    print(f"\nSaved report: {report_path}")
    print(f"Saved errors: {errors_path}")


if __name__ == "__main__":
    main()
