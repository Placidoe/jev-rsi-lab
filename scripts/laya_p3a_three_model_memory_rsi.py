#!/usr/bin/env python3
"""P3-A: three-model, policy-gated Memory-RSI in a near-real synthetic shop.

Model A (business actor): bare Laya sees only the customer message.
Model B (teacher): SmolLM2-1.7B independently applies a versioned policy to full state.
Model C (reviewer): SmolLM2-360M independently repeats the decision without seeing
the teacher output. A deterministic policy engine is the sole synthetic oracle.

No real action is executed. Memory remains staging-only. This experiment measures
whether replay-validated policy memories improve a template-disjoint held-out set.
It does not prove production safety or model-weight self-improvement.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SEED = 20260925
OUTPUT_DIR = Path("/kaggle/working/p3a_three_model_memory_rsi")
ARCHIVE_BASE = "/kaggle/working/laya_p3a_three_model_memory_rsi_artifacts"
POLICY_FILENAME = "p3a_synthetic_business_policy_v1.json"
TEACHER_ID = os.environ.get("P3A_TEACHER_ID", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
REVIEWER_ID = os.environ.get("P3A_REVIEWER_ID", "HuggingFaceTB/SmolLM2-360M-Instruct")

INTENTS = [
    "complaint_handling",
    "exchange_request",
    "payment_issue",
    "refund_request",
    "return_policy",
    "voucher_issue",
    "unknown",
]
ACTIONS = [
    "refund",
    "exchange",
    "technical_support",
    "explain_policy",
    "ask_clarification",
    "human_review",
    "deny",
]
AUTO_ACTIONS = {"refund", "exchange"}


def find_policy() -> Path:
    candidates = list(Path("/kaggle/input").rglob(POLICY_FILENAME))
    # Support both a Kaggle Input mount and a normal checkout of this repository.
    for local in (
        Path(__file__).with_name(POLICY_FILENAME),
        Path(__file__).parents[1] / "data" / "synthetic" / "p3a" / POLICY_FILENAME,
    ):
        if local.exists():
            candidates.append(local)
    unique = sorted({path.resolve() for path in candidates})
    if len(unique) != 1:
        raise RuntimeError(f"Expected exactly one {POLICY_FILENAME}; found: {unique}")
    return unique[0]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


BASE_FACTS = {
    "order_id_present": True,
    "payment_status": "captured",
    "order_status": "delivered",
    "evidence_present": True,
    "risk_level": "low",
    "amount_cny": 88,
    "age_days": 3,
    "inventory_available": True,
    "voucher_status": "valid",
    "voucher_scope_match": True,
    "system_error_present": False,
    "complaint_count": 0,
    "fulfillment_status": "delivered",
}


SCENARIOS = [
    {
        "name": "eligible_refund",
        "facts": {"order_status": "cancelled_eligible", "payment_status": "captured", "amount_cny": 128},
        "train": [
            ("en", "My eligible order was cancelled three days ago. The receipt is attached; please refund the captured payment."),
            ("zh", "订单三天前已按规则取消，付款已扣款且凭证齐全，请退款。"),
            ("id", "Pesanan yang memenuhi syarat dibatalkan tiga hari lalu, pembayaran sudah ditagih dan bukti terlampir. Mohon pengembalian dana."),
        ],
        "test": [
            ("ms", "Pesanan layak dibatalkan tiga hari lalu. Bayaran telah diambil dan resit dilampirkan; saya mahu bayaran balik."),
            ("en", "Cancellation is confirmed and I supplied the order proof. The CNY 128 charge needs to be returned."),
        ],
    },
    {
        "name": "refund_missing_evidence",
        "facts": {"order_status": "cancelled_eligible", "evidence_present": False, "amount_cny": 98},
        "train": [
            ("en", "I cancelled the order and want a refund, but I cannot find the receipt or transaction proof."),
            ("zh", "订单取消了，我想退款，但暂时找不到订单凭证和支付记录。"),
            ("id", "Pesanan sudah dibatalkan dan saya ingin refund, tetapi bukti pesanan belum ada."),
        ],
        "test": [
            ("ms", "Saya mahu bayaran balik untuk pesanan yang dibatalkan, tetapi belum boleh beri resit."),
            ("en", "Please return the payment; the cancellation happened, although no supporting document is available yet."),
        ],
    },
    {
        "name": "refund_high_risk",
        "facts": {"order_status": "cancelled_eligible", "risk_level": "elevated", "amount_cny": 168},
        "train": [
            ("en", "The cancelled order has proof, but my account shows an unresolved risk review. I still want a refund."),
            ("zh", "取消订单的凭证齐全，但账号仍处于风险审核中，我申请退款。"),
            ("id", "Bukti pembatalan lengkap, namun akun sedang ditinjau karena risiko. Saya meminta refund."),
        ],
        "test": [
            ("ms", "Bukti pembatalan ada, tetapi akaun masih ditanda berisiko. Tolong semak permintaan bayaran balik."),
            ("en", "There is a live account-risk flag even though cancellation evidence exists; review my refund request."),
        ],
    },
    {
        "name": "eligible_exchange",
        "facts": {"order_status": "delivered_merchant_fault", "amount_cny": 76, "inventory_available": True},
        "train": [
            ("en", "The delivered shirt is the wrong color. I uploaded a photo and the requested color is in stock; can it be exchanged?"),
            ("zh", "收到的衣服颜色发错了，照片凭证已上传，目标颜色有库存，请换货。"),
            ("id", "Warna barang yang diterima salah. Foto sudah ada dan stok pengganti tersedia; saya ingin tukar barang."),
        ],
        "test": [
            ("ms", "Saiz kasut yang diterima salah. Bukti gambar ada dan saiz ganti masih ada stok; mohon pertukaran."),
            ("en", "I received the wrong variant within the exchange window, supplied evidence, and confirmed replacement stock."),
        ],
    },
    {
        "name": "exchange_no_inventory",
        "facts": {"order_status": "delivered_merchant_fault", "inventory_available": False, "amount_cny": 92},
        "train": [
            ("en", "The size is wrong and I want an exchange, but the replacement inventory is currently unavailable."),
            ("zh", "商品尺码发错了，想换货，但目前目标尺码没有库存。"),
            ("id", "Ukuran barang salah dan saya ingin menukar, tetapi stok pengganti belum tersedia."),
        ],
        "test": [
            ("ms", "Barang yang diterima salah warna, namun stok warna gantian belum tersedia. Apa langkah seterusnya?"),
            ("en", "The wrong item arrived with evidence attached, although no replacement stock is available yet."),
        ],
    },
    {
        "name": "duplicate_payment",
        "facts": {"payment_status": "duplicate_confirmed", "amount_cny": 188},
        "train": [
            ("en", "The same order was charged twice. Both transaction records confirm the duplicate CNY 188 payment."),
            ("zh", "同一订单被重复扣款，两笔 188 元交易记录都已确认。"),
            ("id", "Pesanan yang sama ditagih dua kali dan kedua transaksi sebesar CNY 188 sudah terkonfirmasi."),
        ],
        "test": [
            ("ms", "Pesanan sama dicaj dua kali; dua rekod transaksi CNY 188 telah disahkan."),
            ("en", "Two settled charges correspond to one order, and the duplicate-payment evidence is attached."),
        ],
    },
    {
        "name": "payment_pending",
        "facts": {"payment_status": "pending", "evidence_present": True, "amount_cny": 64},
        "train": [
            ("en", "Payment is still pending and checkout will not complete, although the bank authorization is visible."),
            ("zh", "支付一直处于处理中，结账无法完成，但银行预授权记录可见。"),
            ("id", "Pembayaran masih pending dan checkout tidak selesai, meski otorisasi bank terlihat."),
        ],
        "test": [
            ("ms", "Bayaran masih tergantung dan pesanan tidak selesai walaupun kebenaran bank telah muncul."),
            ("en", "The processor has not finalized the charge and the order remains stuck at checkout."),
        ],
    },
    {
        "name": "invalid_voucher",
        "facts": {"voucher_status": "expired", "voucher_scope_match": False, "evidence_present": True},
        "train": [
            ("en", "My voucher is expired and the product is outside its eligible category. Why was the discount rejected?"),
            ("zh", "优惠券已经过期，而且商品不在适用范围内，为什么不能使用？"),
            ("id", "Voucher sudah kedaluwarsa dan produknya di luar cakupan. Mengapa diskon ditolak?"),
        ],
        "test": [
            ("ms", "Baucar telah tamat tempoh dan item tidak termasuk dalam skop. Mengapa diskaun gagal?"),
            ("en", "The code is no longer valid for this category; please explain the voucher restriction."),
        ],
    },
    {
        "name": "repeated_delivery_complaint",
        "facts": {"complaint_count": 3, "fulfillment_status": "delivered_not_received", "amount_cny": 152},
        "train": [
            ("en", "This is my third complaint: tracking says delivered, but the parcel never arrived."),
            ("zh", "这是第三次投诉，物流显示已送达，但我根本没有收到包裹。"),
            ("id", "Ini keluhan ketiga; pelacakan menyatakan terkirim tetapi paket tidak pernah diterima."),
        ],
        "test": [
            ("ms", "Ini aduan kali ketiga. Sistem kata dihantar tetapi bungkusan tidak diterima."),
            ("en", "The parcel is marked delivered but missing, and two earlier complaints remain unresolved."),
        ],
    },
    {
        "name": "ambiguous_request",
        "facts": {"order_id_present": False, "evidence_present": False, "payment_status": "unknown", "order_status": "unknown"},
        "train": [
            ("en", "Something is wrong with my purchase. Please fix it."),
            ("zh", "我的订单有点问题，麻烦处理一下。"),
            ("id", "Ada masalah dengan pesanan saya. Tolong bantu."),
        ],
        "test": [
            ("ms", "Ada sesuatu yang tidak kena dengan pembelian saya. Tolong bantu."),
            ("en", "I need help with an order, but I do not have any details right now."),
        ],
    },
]


SCENARIO_INTENT = {
    "eligible_refund": "refund_request",
    "refund_missing_evidence": "refund_request",
    "refund_high_risk": "refund_request",
    "eligible_exchange": "exchange_request",
    "exchange_no_inventory": "exchange_request",
    "duplicate_payment": "payment_issue",
    "payment_pending": "payment_issue",
    "invalid_voucher": "voucher_issue",
    "repeated_delivery_complaint": "complaint_handling",
    "ambiguous_request": "unknown",
}


def policy_decision(intent: str, facts: dict[str, Any]) -> dict[str, Any]:
    """Apply the deterministic policy to a simulator-owned ground-truth intent.

    Models never receive this intent. It is part of the hidden synthetic world state,
    analogous to a qualified adjudicated label in a real evaluation.
    """

    if intent == "unknown":
        return {"intent": intent, "action": "ask_clarification", "rule_id": "POL-FALLBACK-001", "missing": []}
    if intent == "return_policy":
        return {"intent": intent, "action": "explain_policy", "rule_id": "POL-RETURN-001", "missing": []}
    if intent == "complaint_handling":
        missing = [key for key in ["order_id_present", "complaint_count", "fulfillment_status"] if key not in facts]
        if missing:
            return {"intent": intent, "action": "ask_clarification", "rule_id": "POL-EXCHANGE-002", "missing": missing}
        action = "human_review" if facts["complaint_count"] >= 2 or facts["fulfillment_status"] == "delivered_not_received" else "ask_clarification"
        return {"intent": intent, "action": action, "rule_id": "POL-COMPLAINT-001", "missing": []}
    if intent == "refund_request":
        required = ["order_id_present", "payment_status", "order_status", "evidence_present", "risk_level", "amount_cny", "age_days"]
        missing = [key for key in required if key not in facts or (key == "order_id_present" and not facts[key])]
        if missing or not facts.get("evidence_present"):
            return {"intent": intent, "action": "ask_clarification", "rule_id": "POL-EXCHANGE-002", "missing": missing or ["evidence_present"]}
        if facts["risk_level"] in {"elevated", "high"} or facts["amount_cny"] > 200:
            return {"intent": intent, "action": "human_review", "rule_id": "POL-REFUND-002", "missing": []}
        eligible = facts["order_status"] in {"cancelled_eligible", "delivered_merchant_fault"} and facts["payment_status"] in {"captured", "duplicate_confirmed"} and facts["age_days"] <= 7
        return {"intent": intent, "action": "refund" if eligible else "deny", "rule_id": "POL-REFUND-001", "missing": []}
    if intent == "exchange_request":
        required = ["order_id_present", "evidence_present", "inventory_available", "age_days", "risk_level"]
        missing = [key for key in required if key not in facts or (key == "order_id_present" and not facts[key])]
        if missing or not facts.get("evidence_present") or not facts.get("inventory_available"):
            return {"intent": intent, "action": "ask_clarification", "rule_id": "POL-EXCHANGE-002", "missing": missing or (["evidence_present"] if not facts.get("evidence_present") else ["inventory_available"])}
        if facts["risk_level"] != "low":
            return {"intent": intent, "action": "human_review", "rule_id": "POL-REFUND-002", "missing": []}
        return {"intent": intent, "action": "exchange" if facts["age_days"] <= 15 else "deny", "rule_id": "POL-EXCHANGE-001", "missing": []}
    if intent == "payment_issue":
        required = ["order_id_present", "payment_status"]
        missing = [key for key in required if key not in facts or (key == "order_id_present" and not facts[key])]
        if missing:
            return {"intent": intent, "action": "ask_clarification", "rule_id": "POL-EXCHANGE-002", "missing": missing}
        if facts["payment_status"] == "duplicate_confirmed":
            if facts.get("risk_level") == "low" and facts.get("evidence_present") and facts.get("amount_cny", 10**9) <= 500:
                return {"intent": intent, "action": "refund", "rule_id": "POL-PAYMENT-001", "missing": []}
            return {"intent": intent, "action": "human_review", "rule_id": "POL-REFUND-002", "missing": []}
        if facts["payment_status"] in {"pending", "failed", "unknown_processor_state"}:
            return {"intent": intent, "action": "technical_support", "rule_id": "POL-PAYMENT-002", "missing": []}
        return {"intent": intent, "action": "human_review", "rule_id": "POL-PAYMENT-002", "missing": []}
    if intent == "voucher_issue":
        required = ["voucher_status", "voucher_scope_match"]
        missing = [key for key in required if key not in facts]
        if missing:
            return {"intent": intent, "action": "ask_clarification", "rule_id": "POL-EXCHANGE-002", "missing": missing}
        if facts["voucher_status"] in {"expired", "invalid"} or not facts["voucher_scope_match"]:
            return {"intent": intent, "action": "explain_policy", "rule_id": "POL-VOUCHER-001", "missing": []}
        if facts["voucher_status"] == "valid" and facts["voucher_scope_match"] and facts.get("system_error_present"):
            return {"intent": intent, "action": "technical_support", "rule_id": "POL-VOUCHER-002", "missing": []}
        return {"intent": intent, "action": "explain_policy", "rule_id": "POL-VOUCHER-001", "missing": []}
    raise AssertionError(intent)


def build_cases() -> list[dict[str, Any]]:
    cases = []
    for scenario in SCENARIOS:
        facts = {**BASE_FACTS, **scenario["facts"]}
        for split, templates, repeats in [("train", scenario["train"], 2), ("test", scenario["test"], 2)]:
            for template_index, (locale, message) in enumerate(templates):
                for repeat in range(repeats):
                    case_id = f"p3a_{split}_{scenario['name']}_{template_index}_{repeat}"
                    variant = message if repeat == 0 else message + " Reference channel: customer chat."
                    oracle = policy_decision(SCENARIO_INTENT[scenario["name"]], facts)
                    cases.append({
                        "case_id": case_id,
                        "split": split,
                        "scenario": scenario["name"],
                        "template_family_id": f"{split}:{scenario['name']}:{template_index}",
                        "locale": locale,
                        "customer_message": variant,
                        "observable_state": {**facts, "case_id": case_id, "locale": locale, "evaluation_only": True},
                        "oracle": oracle,
                    })
    random.Random(SEED).shuffle(cases)
    return cases


POLICY_BRIEF = """Use exactly one intent and one action.
POL-REFUND-001: eligible cancellation/merchant fault, evidence, low risk, <=CNY200, <=7d -> refund.
POL-REFUND-002: elevated/high risk or refund amount >CNY200 -> human_review.
POL-RETURN-001: policy-information request -> explain_policy.
POL-EXCHANGE-001: wrong variant, evidence, inventory, low risk, <=15d -> exchange.
POL-EXCHANGE-002: required state/evidence/inventory missing -> ask_clarification.
POL-PAYMENT-001: confirmed duplicate, evidence, low risk, <=CNY500 -> refund.
POL-PAYMENT-002: payment pending/failed/unknown processor -> technical_support.
POL-VOUCHER-001: expired/invalid/out-of-scope voucher -> explain_policy.
POL-VOUCHER-002: valid in-scope voucher plus system error -> technical_support.
POL-COMPLAINT-001: repeated complaint or delivered-not-received -> human_review.
POL-FALLBACK-001: ambiguous request -> ask_clarification."""


DECISION_OPTIONS = {
    "A": {"intent": "refund_request", "action": "refund", "rule_id": "POL-REFUND-001"},
    "B": {"intent": "refund_request", "action": "ask_clarification", "rule_id": "POL-EXCHANGE-002"},
    "C": {"intent": "refund_request", "action": "human_review", "rule_id": "POL-REFUND-002"},
    "D": {"intent": "exchange_request", "action": "exchange", "rule_id": "POL-EXCHANGE-001"},
    "E": {"intent": "exchange_request", "action": "ask_clarification", "rule_id": "POL-EXCHANGE-002"},
    "F": {"intent": "payment_issue", "action": "refund", "rule_id": "POL-PAYMENT-001"},
    "G": {"intent": "payment_issue", "action": "technical_support", "rule_id": "POL-PAYMENT-002"},
    "H": {"intent": "voucher_issue", "action": "explain_policy", "rule_id": "POL-VOUCHER-001"},
    "I": {"intent": "complaint_handling", "action": "human_review", "rule_id": "POL-COMPLAINT-001"},
    "J": {"intent": "unknown", "action": "ask_clarification", "rule_id": "POL-FALLBACK-001"},
}
OPTION_CATALOG = "\n".join(
    f"{code}: {value['intent']} | {value['action']} | {value['rule_id']}"
    for code, value in DECISION_OPTIONS.items()
)


def option_prompt(row: dict[str, Any], role: str) -> list[dict[str, str]]:
    state = {key: value for key, value in row["observable_state"].items() if key not in {"case_id", "locale", "evaluation_only"}}
    system = (
        f"You are the independent {role} in a fictional offline evaluation. "
        "Select exactly one option under the policy. Never execute a tool.\n"
        f"{POLICY_BRIEF}\nOPTIONS:\n{OPTION_CATALOG}\nReturn only the option letter."
    )
    user = json.dumps({"locale": row["locale"], "message": row["customer_message"], "state": state}, ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def run_option_model(model_id: str, device: str, rows: list[dict[str, Any]], role: str, batch_size: int = 8) -> tuple[dict[str, dict], float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device).eval()
    option_ids = {}
    for code in DECISION_OPTIONS:
        encoded = tokenizer.encode(code, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"Option {code} is not a single token for {model_id}: {encoded}")
        option_ids[code] = encoded[0]

    # Contextual calibration removes the model's unconditional preference for
    # particular option letters.  Without this subtraction both small reviewer
    # models collapse to option A even when the policy/state evidence differs.
    neutral = {
        "locale": "en",
        "customer_message": "",
        "observable_state": {},
    }
    neutral_rendered = tokenizer.apply_chat_template(
        option_prompt(neutral, role), tokenize=False, add_generation_prompt=True
    )
    neutral_tokens = tokenizer(
        neutral_rendered, return_tensors="pt", truncation=True, max_length=1800
    ).to(device)
    with torch.inference_mode():
        neutral_logits = model(**neutral_tokens, use_cache=False).logits[:, -1, :].float()[0]
    neutral_option_logits = torch.stack(
        [neutral_logits[option_ids[code]] for code in DECISION_OPTIONS]
    )

    results: dict[str, dict] = {}
    started = time.perf_counter()
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset: offset + batch_size]
        rendered = [tokenizer.apply_chat_template(option_prompt(row, role), tokenize=False, add_generation_prompt=True) for row in batch]
        tokens = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True, max_length=1800).to(device)
        with torch.inference_mode():
            logits = model(**tokens, use_cache=False).logits[:, -1, :].float()
        option_logits = torch.stack([logits[:, option_ids[code]] for code in DECISION_OPTIONS], dim=1)
        option_logits = option_logits - neutral_option_logits.unsqueeze(0)
        top_values, top_indices = torch.topk(option_logits, k=2, dim=1)
        codes = list(DECISION_OPTIONS)
        for index, row in enumerate(batch):
            code = codes[int(top_indices[index, 0])]
            decision = DECISION_OPTIONS[code]
            results[row["case_id"]] = {
                **decision,
                "parse_ok": True,
                "selected_option": code,
                "logit_margin": round(float(top_values[index, 0] - top_values[index, 1]), 6),
                "scoring_method": "contextual_calibrated_next_token",
                "raw": f"contextual_calibrated_option:{code}",
            }
    seconds = time.perf_counter() - started
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results, seconds


LAYA_QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "Identify the customer's semantic intent for offline evaluation.",
        "criteria": {intent: intent.replace("_", " ") for intent in INTENTS},
    },
    "action": {
        "type": "choice",
        "instructions": "Choose one conservative candidate action. No action is executed.",
        "criteria": {
            "refund": "Issue a policy-eligible refund.",
            "exchange": "Approve a policy-eligible exchange.",
            "technical_support": "Diagnose a payment, voucher, or system incident.",
            "explain_policy": "Explain a policy restriction without executing an action.",
            "ask_clarification": "Request missing facts or evidence.",
            "human_review": "Escalate risk, conflict, complaint, or exception.",
            "deny": "The request is clearly outside policy.",
        },
    },
}


def run_laya(rows: list[dict[str, Any]]) -> tuple[dict[str, dict], float]:
    from laya import Router

    router = Router(device="cuda")
    router.preload(["english", "multilingual"])
    warm = rows[0]
    warm_state = {"case_id": warm["case_id"], "locale": warm["locale"], "customer_message": warm["customer_message"], "evaluation_only": True}
    router.predict(warm_state, LAYA_QUESTIONS, lang=warm["locale"])
    requests = []
    for row in rows:
        state = {"case_id": row["case_id"], "locale": row["locale"], "customer_message": row["customer_message"], "evaluation_only": True}
        requests.append({"state": state, "questions": LAYA_QUESTIONS, "lang": row["locale"]})
    started = time.perf_counter()
    outputs = router.predict_batch(requests, batch_size=16)
    seconds = time.perf_counter() - started
    result = {}
    for row, output in zip(rows, outputs, strict=True):
        result[row["case_id"]] = {
            "intent": output["answers"]["intent"]["choice"],
            "intent_confidence": output["answers"]["intent"].get("confidence"),
            "action": output["answers"]["action"]["choice"],
            "action_confidence": output["answers"]["action"].get("confidence"),
            "routed_model": output.get("routing", {}).get("model"),
        }
    return result, seconds


def bucket_amount(value: int | float) -> str:
    if value <= 200:
        return "le_200"
    if value <= 500:
        return "201_500"
    return "gt_500"


def memory_signature(intent: str, facts: dict[str, Any]) -> str:
    fields = [
        intent,
        str(facts.get("order_status")),
        str(facts.get("payment_status")),
        str(bool(facts.get("order_id_present"))).lower(),
        str(bool(facts.get("evidence_present"))).lower(),
        str(facts.get("risk_level")),
        bucket_amount(facts.get("amount_cny", 0)),
        "le7" if facts.get("age_days", 999) <= 7 else ("le15" if facts.get("age_days", 999) <= 15 else "gt15"),
        str(bool(facts.get("inventory_available"))).lower(),
        str(facts.get("voucher_status")),
        str(bool(facts.get("voucher_scope_match"))).lower(),
        str(bool(facts.get("system_error_present"))).lower(),
        "repeat" if facts.get("complaint_count", 0) >= 2 else "first",
        str(facts.get("fulfillment_status")),
    ]
    return "|".join(fields)


def replay(row: dict[str, Any], proposed_action: str | None) -> dict[str, Any]:
    oracle = row["oracle"]
    passed = proposed_action == oracle["action"]
    unsafe_wrong = proposed_action in AUTO_ACTIONS and not passed
    return {
        "policy_id": "JEV_SYNTHETIC_ECOM_AFTERSALES_V1",
        "rule_id": oracle["rule_id"],
        "oracle_intent": oracle["intent"],
        "expected_action": oracle["action"],
        "proposed_action": proposed_action,
        "direct_pass": passed,
        "unsafe_wrong_candidate": unsafe_wrong,
        "final_safe_action": proposed_action if passed else "human_review",
        "would_execute": False,
    }


def rate(values: list[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def macro_f1(rows: list[dict[str, Any]], predicted_key: str, expected_key: str, labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(row[predicted_key] == label and row[expected_key] == label for row in rows)
        fp = sum(row[predicted_key] == label and row[expected_key] != label for row in rows)
        fn = sum(row[predicted_key] != label and row[expected_key] == label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(sum(scores) / len(scores), 4)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("P3-A requires Kaggle GPU T4 x2; two CUDA devices were not found.")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    policy_path = find_policy()
    policy_bytes = policy_path.read_bytes()
    policy = json.loads(policy_bytes)
    if policy["policy_id"] != "JEV_SYNTHETIC_ECOM_AFTERSALES_V1" or policy["status"] != "fictional_evaluation_only":
        raise RuntimeError("Unexpected policy identity or status.")

    cases = build_cases()
    train = [row for row in cases if row["split"] == "train"]
    test = [row for row in cases if row["split"] == "test"]
    if {row["case_id"] for row in train} & {row["case_id"] for row in test}:
        raise RuntimeError("Case-ID leakage.")
    if {row["template_family_id"] for row in train} & {row["template_family_id"] for row in test}:
        raise RuntimeError("Template-family leakage.")
    if {row["customer_message"] for row in train} & {row["customer_message"] for row in test}:
        raise RuntimeError("Exact-message leakage.")

    runtime = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "laya": importlib.metadata.version("laya"),
        "transformers": importlib.metadata.version("transformers"),
        "huggingface_hub": importlib.metadata.version("huggingface_hub"),
        "seed": SEED,
    }
    print("Runtime:", json.dumps(runtime, ensure_ascii=False))
    print(f"Cases: {len(cases)}; train={len(train)}; test={len(test)}")

    laya, laya_seconds = run_laya(cases)
    teacher, teacher_seconds = run_option_model(TEACHER_ID, "cuda:0", cases, "teacher")
    reviewer, reviewer_seconds = run_option_model(REVIEWER_ID, "cuda:1", cases, "reviewer")

    train_audit = []
    staging_memory: dict[str, dict[str, Any]] = {}
    for row in train:
        cid = row["case_id"]
        oracle = row["oracle"]
        laya_out, teacher_out, reviewer_out = laya[cid], teacher[cid], reviewer[cid]
        independently_agree = (
            teacher_out["parse_ok"] and reviewer_out["parse_ok"]
            and teacher_out["intent"] == reviewer_out["intent"]
            and teacher_out["action"] == reviewer_out["action"]
        )
        replay_proven = (
            independently_agree
            and teacher_out["intent"] == oracle["intent"]
            and teacher_out["action"] == oracle["action"]
            and teacher_out["rule_id"] == oracle["rule_id"]
            and reviewer_out["rule_id"] == oracle["rule_id"]
        )
        online_key_usable = laya_out["intent"] == oracle["intent"]
        promotable_to_staging = replay_proven and online_key_usable and not oracle["missing"]
        signature = memory_signature(laya_out["intent"], row["observable_state"])
        audit = {
            **row,
            "business_model": laya_out,
            "teacher_model": teacher_out,
            "reviewer_model": reviewer_out,
            "teacher_reviewer_agree": independently_agree,
            "deterministic_replay_proven": replay_proven,
            "online_key_usable": online_key_usable,
            "promotable_to_staging": promotable_to_staging,
            "memory_signature": signature,
        }
        train_audit.append(audit)
        if promotable_to_staging:
            existing = staging_memory.get(signature)
            if existing and existing["action"] != oracle["action"]:
                raise RuntimeError(f"Conflicting replay-proven action for {signature}")
            if existing:
                existing["support_count"] += 1
                existing["source_case_ids"].append(cid)
            else:
                staging_memory[signature] = {
                    "memory_id": "p3a_staging_" + hashlib.sha256(signature.encode()).hexdigest()[:12],
                    "policy_signature": signature,
                    "intent": oracle["intent"],
                    "action": oracle["action"],
                    "rule_id": oracle["rule_id"],
                    "support_count": 1,
                    "source_case_ids": [cid],
                    "evidence": "teacher_reviewer_independent_agreement_plus_deterministic_policy_replay",
                    "memory_status": "staging_only",
                    "memory_write_authorized": False,
                    "training_label_authorized": False,
                    "business_action_authorized": False,
                }

    test_audit = []
    for row in test:
        cid = row["case_id"]
        laya_out = laya[cid]
        signature = memory_signature(laya_out["intent"], row["observable_state"])
        memory = staging_memory.get(signature)
        baseline = replay(row, laya_out["action"])
        rsi_candidate = memory["action"] if memory else laya_out["action"]
        rsi = replay(row, rsi_candidate)
        test_audit.append({
            **row,
            "business_model": laya_out,
            "teacher_model_audit_only": teacher[cid],
            "reviewer_model_audit_only": reviewer[cid],
            "memory_signature": signature,
            "memory_hit": memory is not None,
            "memory_id": memory["memory_id"] if memory else None,
            "baseline_replay": baseline,
            "rsi_candidate": rsi_candidate,
            "rsi_replay": rsi,
            "rsi_corrected_baseline_error": (not baseline["direct_pass"] and rsi["direct_pass"]),
            "rsi_regressed_baseline_success": (baseline["direct_pass"] and not rsi["direct_pass"]),
        })

    baseline_pass = [row["baseline_replay"]["direct_pass"] for row in test_audit]
    rsi_pass = [row["rsi_replay"]["direct_pass"] for row in test_audit]
    memory_hits = [row["memory_hit"] for row in test_audit]
    baseline_unsafe = [row["baseline_replay"]["unsafe_wrong_candidate"] for row in test_audit]
    rsi_unsafe = [row["rsi_replay"]["unsafe_wrong_candidate"] for row in test_audit]
    teacher_rows = [{"pred": teacher[r["case_id"]]["action"], "gold": r["oracle"]["action"]} for r in cases]
    reviewer_rows = [{"pred": reviewer[r["case_id"]]["action"], "gold": r["oracle"]["action"]} for r in cases]

    report = {
        "purpose": "P3-A three-model policy-gated Memory-RSI effectiveness test in a near-real synthetic e-commerce environment",
        "evaluation_only": True,
        "policy": {
            "policy_id": policy["policy_id"],
            "version": policy["version"],
            "sha256": sha256_bytes(policy_bytes),
            "status": policy["status"],
        },
        "models": {"business": "Laya Router", "teacher": TEACHER_ID, "reviewer": REVIEWER_ID},
        "runtime": runtime,
        "data": {
            "all_cases": len(cases),
            "train_cases": len(train),
            "test_cases": len(test),
            "train_template_families": len({row["template_family_id"] for row in train}),
            "test_template_families": len({row["template_family_id"] for row in test}),
            "case_id_overlap": 0,
            "template_family_overlap": 0,
            "status": "fictional multilingual cases generated in-script; no real customer or business data",
        },
        "latency": {
            "laya_seconds": round(laya_seconds, 4),
            "teacher_seconds": round(teacher_seconds, 4),
            "reviewer_seconds": round(reviewer_seconds, 4),
            "laya_cases_per_second": round(len(cases) / laya_seconds, 4),
        },
        "model_quality": {
            "laya_semantic_accuracy_all": rate([laya[row["case_id"]]["intent"] == row["oracle"]["intent"] for row in cases]),
            "laya_action_accuracy_all": rate([laya[row["case_id"]]["action"] == row["oracle"]["action"] for row in cases]),
            "teacher_parse_rate": rate([teacher[row["case_id"]]["parse_ok"] for row in cases]),
            "teacher_action_accuracy": rate([teacher[row["case_id"]]["action"] == row["oracle"]["action"] for row in cases]),
            "teacher_action_macro_f1": macro_f1(teacher_rows, "pred", "gold", ACTIONS),
            "reviewer_parse_rate": rate([reviewer[row["case_id"]]["parse_ok"] for row in cases]),
            "reviewer_action_accuracy": rate([reviewer[row["case_id"]]["action"] == row["oracle"]["action"] for row in cases]),
            "reviewer_action_macro_f1": macro_f1(reviewer_rows, "pred", "gold", ACTIONS),
            "teacher_reviewer_exact_agreement": rate([teacher[row["case_id"]]["intent"] == reviewer[row["case_id"]]["intent"] and teacher[row["case_id"]]["action"] == reviewer[row["case_id"]]["action"] for row in cases]),
        },
        "memory_build": {
            "train_cases": len(train),
            "staging_memory_entries": len(staging_memory),
            "promoted_train_cases": sum(row["promotable_to_staging"] for row in train_audit),
            "memory_write_authorized": False,
            "training_label_authorized": False,
            "business_action_authorized": False,
        },
        "heldout_effect": {
            "baseline_direct_policy_pass_rate": rate(baseline_pass),
            "rsi_direct_policy_pass_rate": rate(rsi_pass),
            "absolute_uplift": round(rate(rsi_pass) - rate(baseline_pass), 4),
            "memory_hit_rate": rate(memory_hits),
            "baseline_unsafe_wrong_candidate_rate": rate(baseline_unsafe),
            "rsi_unsafe_wrong_candidate_rate": rate(rsi_unsafe),
            "errors_corrected": sum(row["rsi_corrected_baseline_error"] for row in test_audit),
            "successes_regressed": sum(row["rsi_regressed_baseline_success"] for row in test_audit),
            "final_unsafe_execution_count": 0,
        },
        "decision": {
            "memory_rsi_effective_in_this_synthetic_environment": rate(rsi_pass) > rate(baseline_pass),
            "production_rsi_authorized": False,
            "model_weight_rsi_demonstrated": False,
            "interpretation": "A positive uplift demonstrates replay-validated policy-memory reuse under intentional policy-signature overlap. It is not evidence that the models can self-certify truth or safely act in production.",
            "next_gate": "Repeat with approved de-identified business trajectories, real policy evidence IDs, qualified human adjudication, adversarial counterexamples, and a locked untouched test set.",
        },
    }

    memory_rows = [value for _, value in sorted(staging_memory.items())]
    write_jsonl(OUTPUT_DIR / "p3a_all_synthetic_cases.jsonl", cases)
    write_jsonl(OUTPUT_DIR / "p3a_train_three_model_audit.jsonl", train_audit)
    write_jsonl(OUTPUT_DIR / "p3a_staging_memory.jsonl", memory_rows)
    write_jsonl(OUTPUT_DIR / "p3a_heldout_baseline_vs_rsi.jsonl", test_audit)
    write_jsonl(OUTPUT_DIR / "p3a_heldout_errors.jsonl", [row for row in test_audit if not row["rsi_replay"]["direct_pass"]])
    (OUTPUT_DIR / "p3a_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / POLICY_FILENAME).write_bytes(policy_bytes)
    (OUTPUT_DIR / "p3a_boundary.md").write_text(
        "# P3-A boundary\n\nSynthetic policy and fictional cases only. No output authorizes Memory writes, training labels, business actions, refunds, exchanges, or production deployment.\n",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), OUTPUT_DIR / Path(__file__).name)
    archive = shutil.make_archive(ARCHIVE_BASE, "zip", root_dir=OUTPUT_DIR.parent, base_dir=OUTPUT_DIR.name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("archive:", archive)


if __name__ == "__main__":
    main()
