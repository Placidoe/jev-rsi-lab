#!/usr/bin/env python3
"""P4-F4: shared public ALFWorld demonstration scaffold.

This wrapper reuses the frozen games, model, decoding, exact-command action
interface, and loop guard from P4-F3.  Both arms receive the same fixed
public-demo-derived control scaffold.  The only arm difference remains the
pre-existing task-type JEV Memory text.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import time
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = Path(os.environ.get("JEV_P4F3_SCRIPT", str(SCRIPT_DIR / "p4f3_exact_command_alfworld_7b.py")))
OUTPUT_DIR = Path(os.environ.get("JEV_OUTPUT_DIR", "results/p4f4_shared_public_demo_alfworld_7b"))
ARCHIVE_BASE = os.environ.get(
    "JEV_ARCHIVE_BASE", str(OUTPUT_DIR.with_name(f"{OUTPUT_DIR.name}_artifacts"))
)

PUBLIC_SOURCE = {
    "repository": "https://github.com/noahshinn/reflexion",
    "commit": "218cf0ef1df84b05ce379dd4a8e47f17766733a0",
    "file": "alfworld_runs/reflexion_few_shot_examples.txt",
    "sha256": "5040995a3cdc88bce30fa26bec3ca4875ced8db98b7b2859f5c1614deec716e7",
    "license": "MIT",
}

SHARED_PUBLIC_SCAFFOLD = """Public ALFWorld control scaffold (identical in both arms):
1. Parse the requested object and final device/receptacle from the goal.
2. Search receptacles systematically. Navigate to an unsearched location; if it is closed, open it. When the requested object is visible, take it immediately.
3. For look-at-under-light tasks, success requires holding the requested object while the desk lamp is on. Locate and take the object, locate the desk lamp, then use the desk lamp. Do not repeatedly examine the lamp.
4. For pick-and-place tasks, take the requested object, navigate to the requested destination, then choose the exact admissible move/put command.
5. Treat observations as evidence. If an action changes nothing, choose a different admissible command. Copy commands exactly; never invent syntax.
"""


def load_base():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(f"Missing frozen P4-F3 base script: {BASE_SCRIPT}")
    spec = importlib.util.spec_from_file_location("p4f3_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_choose_action(base):
    def choose_action(model, tokenizer, task, observation, history, commands, memory):
        recent = history[-8:]
        history_text = "\n".join(
            f"{x['step']}: {x['action']} -> {base.normalize(x['observation'], 360)}"
            for x in recent
        ) or "(none)"
        command_text = "\n".join(f"- {cmd}" for cmd in commands)
        memory_text = memory or "No external JEV Memory is provided."
        messages = [
            {
                "role": "system",
                "content": (
                    "You control ALFWorld. Use the shared public scaffold and current evidence to make concrete progress. "
                    "Never invent a command. Copy exactly one allowed command verbatim. "
                    "Return only that command, without an index or explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{SHARED_PUBLIC_SCAFFOLD}\n"
                    f"Goal and initial scene:\n{base.normalize(task)}\n\n"
                    f"Optional JEV Memory (the only between-arm difference):\n{memory_text}\n\n"
                    f"Recent evidence:\n{history_text}\n\n"
                    f"Current observation:\n{base.normalize(observation)}\n\n"
                    f"Allowed commands after the shared loop guard:\n{command_text}\n\nCommand:"
                ),
            },
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=6144).to(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=24,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        raw = tokenizer.decode(
            output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        cleaned = raw.strip().strip("`\"'").strip()
        cleaned = re.sub(r"^(command|action)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
        exact = {cmd.lower(): cmd for cmd in commands}
        selected = exact.get(cleaned.lower())
        parse_method = "exact"
        if selected is None:
            embedded = [cmd for cmd in commands if cmd.lower() in cleaned.lower()]
            if embedded:
                selected = max(embedded, key=len)
                parse_method = "embedded_exact_command"
        parse_ok = selected is not None
        if selected is None:
            selected = commands[0]
            parse_method = "fallback_first_command"
        return selected, {
            "raw_model_output": raw,
            "parsed_command": selected,
            "parse_method": parse_method,
            "parse_ok": parse_ok,
            "latency_seconds": round(elapsed, 6),
            "input_tokens": int(encoded["input_ids"].shape[1]),
            "output_tokens": int(output.shape[1] - encoded["input_ids"].shape[1]),
            "shared_public_scaffold_applied": True,
        }

    return choose_action


def main() -> None:
    base = load_base()
    base.OUTPUT_DIR = OUTPUT_DIR
    base.ARCHIVE_BASE = ARCHIVE_BASE
    base.choose_action = build_choose_action(base)
    base.main()

    report_path = OUTPUT_DIR / "p4f_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["schema_version"] = "JEV_P4F4_SHARED_PUBLIC_DEMO_ALFWORLD_7B_REPORT_V1"
    report["purpose"] = (
        "Raise the shared ALFWorld capability floor with a fixed public demonstration scaffold, "
        "then identify the incremental effect of existing JEV Memory"
    )
    report["public_demonstration_source"] = PUBLIC_SOURCE
    report["design"]["shared_public_scaffold"] = True
    report["design"]["shared_public_scaffold_sha256"] = base.stable_hash(SHARED_PUBLIC_SCAFFOLD)
    report["design"]["only_arm_difference"] = "pre-existing task-type JEV Memory text"
    report["decision"]["production_memory_authorized"] = False
    report["decision"]["interpretation"] = (
        "If neither arm succeeds, Memory effect remains unidentified. If either arm succeeds, "
        "paired transitions identify whether the current Memory helps, is neutral, or negatively transfers."
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "shared_public_demo.txt").write_text(SHARED_PUBLIC_SCAFFOLD, encoding="utf-8")
    (OUTPUT_DIR / "public_demo_source.json").write_text(
        json.dumps(PUBLIC_SOURCE, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), OUTPUT_DIR / Path(__file__).name)
    shutil.copy2(BASE_SCRIPT, OUTPUT_DIR / BASE_SCRIPT.name)
    archive = shutil.make_archive(
        ARCHIVE_BASE, "zip", root_dir=OUTPUT_DIR.parent, base_dir=OUTPUT_DIR.name
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("P4-F4 archive:", archive)


if __name__ == "__main__":
    main()

