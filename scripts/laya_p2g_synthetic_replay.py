"""P2-G: JEV + Laya candidate decision + deterministic Replay fixtures.

Kaggle run (GPU T4 x2 + Internet on):
    !python /kaggle/working/laya_p2g_synthetic_replay.py

The script needs no external dataset. It creates 50 fictional cases (25 exchange,
25 complaint), runs bare Laya on the P2-F frozen minimal semantic input, and then
uses deterministic JEV Replay/tool mocks to choose a final safe action.

This is a prototype evaluation only: no real customer data, Memory write, model
training, or business tool call is performed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20260924
CASE_COUNT = 50
OUTPUT_DIR = Path("/kaggle/working/p2g_synthetic_replay")
ARCHIVE_BASE = "/kaggle/working/laya_p2g_synthetic_replay_artifacts"
LAYA_INPUT_SCHEMA_VERSION = "JEV_STATE_SCHEMA_V1_MINIMAL_SEMANTIC_ROUTING"
LAYA_INPUT_FIELDS = ("case_id", "locale", "customer_message", "evaluation_only")
REPLAY_SCHEMA_VERSION = "JEV_REPLAY_FIXTURE_V1"

QUESTIONS = {
    "candidate_action": {
        "type": "choice",
        "instructions": (
            "Choose only a candidate next support action from the customer message. "
            "A downstream deterministic safety gate will validate it; this does not authorize any action."
        ),
        "criteria": {
            "start_exchange": "The request appears to be a routine exchange that may be eligible.",
            "ask_clarification": "Required details are missing or the request is ambiguous.",
            "human_review": "The request is a complaint, high-risk issue, policy exception, or needs human judgment.",
        },
    }
}


def install_runtime() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "laya"])


def runtime_metadata() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Kaggle: Settings -> Accelerator -> GPU T4 x2, then restart.")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "laya_package_version": importlib.metadata.version("laya"),
        "deterministic_controls": {"seed": SEED, "cudnn_benchmark": False, "cudnn_deterministic": True},
    }


def fixture(case_id: str, domain: str, message: str, **state: object) -> dict:
    return {
        "case_id": case_id,
        "domain": domain,
        "locale": state.pop("locale", "zh"),
        "customer_message": message,
        "business_state": state,
        "fixture_only": True,
    }


def build_fixtures() -> list[dict]:
    """Build a balanced, fictional fixture set; these are not business policy labels."""
    rows: list[dict] = []
    # 8 routine eligible exchanges -> start_exchange.
    for i in range(8):
        rows.append(fixture(
            f"exchange_eligible_{i:02d}", "exchange",
            "收到的商品尺寸不合适，想换一个尺码。",
            order_status="delivered", delivery_days=2 + i % 5, exchange_window_days=14,
            inventory_available=True, prior_exchange_count=0, identity_verified=True,
            account_risk="normal", order_id_present=True, requested_size_present=True,
        ))
    # 7 incomplete exchange requests -> ask_clarification.
    for i in range(7):
        rows.append(fixture(
            f"exchange_missing_{i:02d}", "exchange",
            "商品不太合适，我想换货。" if i % 2 == 0 else "我想换一个颜色。",
            order_status="delivered", delivery_days=3, exchange_window_days=14,
            inventory_available=True, prior_exchange_count=0, identity_verified=True,
            account_risk="normal", order_id_present=(i % 2 == 0), requested_size_present=False,
        ))
    # 5 high-risk exchanges -> human_review.
    for i in range(5):
        rows.append(fixture(
            f"exchange_risk_{i:02d}", "exchange",
            "我要马上换货，否则我会投诉。",
            order_status="delivered", delivery_days=4, exchange_window_days=14,
            inventory_available=True, prior_exchange_count=1, identity_verified=(i % 2 == 0),
            account_risk="high", order_id_present=True, requested_size_present=True,
        ))
    # 5 policy exceptions (window expired or stock unavailable) -> human_review.
    for i in range(5):
        rows.append(fixture(
            f"exchange_exception_{i:02d}", "exchange",
            "这个商品我想换货，请帮我处理。",
            order_status="delivered", delivery_days=20 if i % 2 == 0 else 4, exchange_window_days=14,
            inventory_available=(i % 2 == 0), prior_exchange_count=0, identity_verified=True,
            account_risk="normal", order_id_present=True, requested_size_present=True,
        ))
    # 10 routine complaints with missing evidence -> ask_clarification.
    for i in range(10):
        rows.append(fixture(
            f"complaint_missing_{i:02d}", "complaint",
            "这次服务体验很差，我很不满意。",
            ticket_count=1, sla_hours_overdue=0, chargeback_threat=False,
            abuse_or_safety_flag=False, account_risk="normal", order_id_present=False,
            complaint_severity="low", evidence_attached=False,
        ))
    # 10 escalated complaints -> human_review.
    for i in range(10):
        rows.append(fixture(
            f"complaint_escalated_{i:02d}", "complaint",
            "问题已经反复反馈多次仍未解决，我要向平台投诉。",
            ticket_count=3 + i % 3, sla_hours_overdue=24 + i, chargeback_threat=(i % 2 == 0),
            abuse_or_safety_flag=False, account_risk="normal", order_id_present=True,
            complaint_severity="high", evidence_attached=True,
        ))
    # 5 safety/abuse/account-risk complaints -> human_review.
    for i in range(5):
        rows.append(fixture(
            f"complaint_risk_{i:02d}", "complaint",
            "我的账号和订单出现异常，我要求马上处理并投诉。",
            ticket_count=2, sla_hours_overdue=8, chargeback_threat=False,
            abuse_or_safety_flag=True, account_risk="high", order_id_present=True,
            complaint_severity="critical", evidence_attached=True,
        ))
    if len(rows) != CASE_COUNT:
        raise AssertionError(f"Expected {CASE_COUNT} fixtures, got {len(rows)}")
    random.Random(SEED).shuffle(rows)
    return rows


def laya_semantic_state(case: dict) -> dict:
    state = {
        "case_id": case["case_id"],
        "locale": case["locale"],
        "customer_message": case["customer_message"],
        "evaluation_only": True,
    }
    if tuple(state) != LAYA_INPUT_FIELDS:
        raise AssertionError(f"Laya state schema drift: {tuple(state)}")
    return state


def expected_safe_action(case: dict) -> tuple[str, str]:
    """Deterministic prototype policy; replace only after business-policy review."""
    state = case["business_state"]
    if case["domain"] == "complaint":
        if state["abuse_or_safety_flag"] or state["account_risk"] == "high":
            return "human_review", "account_or_safety_risk"
        if state["complaint_severity"] in {"high", "critical"} or state["ticket_count"] >= 3 or state["chargeback_threat"]:
            return "human_review", "escalated_complaint"
        return "ask_clarification", "missing_order_or_evidence"

    if state["account_risk"] != "normal" or not state["identity_verified"]:
        return "human_review", "exchange_risk_or_identity"
    if not state["order_id_present"] or not state["requested_size_present"]:
        return "ask_clarification", "missing_exchange_details"
    if state["delivery_days"] > state["exchange_window_days"] or not state["inventory_available"] or state["prior_exchange_count"] > 0:
        return "human_review", "exchange_policy_exception"
    if state["order_status"] == "delivered":
        return "start_exchange", "mock_eligibility_pass"
    return "human_review", "unsupported_order_state"


def replay_validate(case: dict, candidate: str) -> dict:
    expected, reason = expected_safe_action(case)
    accepted = candidate == expected
    return {
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "candidate_action": candidate,
        "expected_safe_action": expected,
        "replay_pass": accepted,
        "replay_reason": "candidate_matches_mock_policy" if accepted else reason,
        "final_action": candidate if accepted else expected,
        "final_action_matches_policy": True,
        "tool_mock": {
            "name": "mock_exchange_or_complaint_gate",
            "external_call": False,
            "result": reason,
        },
    }


def run_laya(router, cases: list[dict]) -> tuple[list[dict], float]:
    warm = cases[0]
    router.predict(laya_semantic_state(warm), QUESTIONS, lang=warm["locale"])
    requests = [
        {"state": laya_semantic_state(case), "questions": QUESTIONS, "lang": case["locale"]}
        for case in cases
    ]
    started = time.perf_counter()
    outputs = router.predict_batch(requests, batch_size=16)
    elapsed = time.perf_counter() - started
    predictions = []
    for case, output in zip(cases, outputs):
        answer = output["answers"]["candidate_action"]
        predictions.append({
            "case_id": case["case_id"],
            "candidate_action": answer["choice"],
            "reported_confidence": answer.get("confidence"),
            "routed_model": output.get("routing", {}).get("model"),
        })
    return predictions, elapsed


def summary(trajectories: list[dict]) -> dict:
    by_domain: dict[str, Counter] = defaultdict(Counter)
    reasons = Counter()
    for row in trajectories:
        bucket = by_domain[row["domain"]]
        bucket["count"] += 1
        bucket["candidate_matches_policy"] += int(row["candidate_matches_policy"])
        bucket["replay_rejections"] += int(not row["replay"]["replay_pass"])
        bucket["final_matches_policy"] += int(row["replay"]["final_action_matches_policy"])
        reasons[row["replay"]["replay_reason"]] += 1

    def divide(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    total = len(trajectories)
    candidates = sum(row["candidate_matches_policy"] for row in trajectories)
    rejected = sum(not row["replay"]["replay_pass"] for row in trajectories)
    return {
        "case_count": total,
        "candidate_action_match_rate": divide(candidates, total),
        "replay_rejection_count": rejected,
        "replay_rejection_rate": divide(rejected, total),
        "final_action_policy_match_rate": 1.0,
        "by_domain": {
            domain: {
                "count": values["count"],
                "candidate_action_match_rate": divide(values["candidate_matches_policy"], values["count"]),
                "replay_rejection_rate": divide(values["replay_rejections"], values["count"]),
                "final_action_policy_match_rate": divide(values["final_matches_policy"], values["count"]),
            }
            for domain, values in sorted(by_domain.items())
        },
        "replay_reason_counts": dict(sorted(reasons.items())),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def archive_self_or_notebook_cell() -> Path:
    destination = OUTPUT_DIR / "laya_p2g_synthetic_replay.py"
    source_name = globals().get("__file__")
    if source_name and Path(source_name).is_file():
        shutil.copy2(Path(source_name), destination)
        return destination
    try:
        cell_source = get_ipython().history_manager.input_hist_raw[-1]  # type: ignore[name-defined]
    except Exception:
        cell_source = ""
    destination.write_text(cell_source or "# Source unavailable: interactive notebook cell.\n", encoding="utf-8")
    return destination


def main() -> None:
    install_runtime()
    runtime = runtime_metadata()
    cases = build_fixtures()
    print("Runtime:", json.dumps(runtime, ensure_ascii=False))
    print("Fixture domains:", dict(Counter(case["domain"] for case in cases)))
    print("Laya schema:", LAYA_INPUT_SCHEMA_VERSION, list(LAYA_INPUT_FIELDS))

    from laya import Router

    router = Router(device="cuda")
    router.preload(["english", "multilingual"])
    predictions, inference_seconds = run_laya(router, cases)
    by_id = {row["case_id"]: row for row in predictions}
    trajectories = []
    for case in cases:
        prediction = by_id[case["case_id"]]
        replay = replay_validate(case, prediction["candidate_action"])
        trajectories.append({
            **case,
            "laya_input_schema_version": LAYA_INPUT_SCHEMA_VERSION,
            "laya_input_state": laya_semantic_state(case),
            "laya_candidate": prediction,
            "expected_safe_action": replay["expected_safe_action"],
            "candidate_matches_policy": prediction["candidate_action"] == replay["expected_safe_action"],
            "replay": replay,
        })

    report = {
        "purpose": "P2-G fictional exchange/complaint JEV Replay prototype",
        "evaluation_only": True,
        "data_status": "50 fictional fixtures generated in-script; no personal or production data",
        "runtime": runtime,
        "seed": SEED,
        "laya_input_schema": {"version": LAYA_INPUT_SCHEMA_VERSION, "fields": list(LAYA_INPUT_FIELDS)},
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "question_sha256": hashlib.sha256(json.dumps(QUESTIONS, sort_keys=True).encode()).hexdigest(),
        "inference_seconds": round(inference_seconds, 4),
        "inference_cases_per_second": round(len(cases) / inference_seconds, 4),
        "summary": summary(trajectories),
        "safety_boundary": "Laya is candidate-only. Replay uses fictional deterministic mocks. No output authorizes real tools, Memory writes, training labels, or business actions.",
        "next_gate": "Replace only after review with de-identified, business-state-enriched trajectories and approved Replay policy rules.",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_DIR / "p2g_fixture_cases.jsonl", cases)
    write_jsonl(OUTPUT_DIR / "p2g_replay_trajectories.jsonl", trajectories)
    write_jsonl(OUTPUT_DIR / "p2g_replay_rejections.jsonl", [row for row in trajectories if not row["replay"]["replay_pass"]])
    (OUTPUT_DIR / "p2g_replay_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    archived_script = archive_self_or_notebook_cell()
    archive_path = Path(shutil.make_archive(ARCHIVE_BASE, "zip", root_dir=OUTPUT_DIR.parent, base_dir=OUTPUT_DIR.name))

    print("=== P2-G synthetic JEV Replay ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved fixtures:", OUTPUT_DIR / "p2g_fixture_cases.jsonl")
    print("Saved trajectories:", OUTPUT_DIR / "p2g_replay_trajectories.jsonl")
    print("Saved replay rejections:", OUTPUT_DIR / "p2g_replay_rejections.jsonl")
    print("Saved report:", OUTPUT_DIR / "p2g_replay_report.json")
    print("Saved executed script:", archived_script)
    print("Saved reproducibility archive:", archive_path)


if __name__ == "__main__":
    main()
