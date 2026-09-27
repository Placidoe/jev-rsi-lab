#!/usr/bin/env python3
"""P4-F6: eliminate parser fallback with prefix-constrained legal actions.

This reruns the repaired P4-F5 paired experiment on the same frozen eight
ALFWorld games.  The model can emit only one of the current admissible
commands, so no invalid free-form output can silently become the first command.
The only between-arm difference remains the frozen task-type JEV Memory.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import time
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = Path(os.environ.get("JEV_P4F3_SCRIPT", str(SCRIPT_DIR / "p4f3_exact_command_alfworld_7b.py")))
F4_SCRIPT = Path(os.environ.get("JEV_P4F4_SCRIPT", str(SCRIPT_DIR / "p4f4_shared_public_demo_alfworld_7b.py")))
F5_SCRIPT = Path(os.environ.get("JEV_P4F5_SCRIPT", str(SCRIPT_DIR / "p4f5_fresh_action_state_alfworld_7b.py")))
OUTPUT_DIR = Path(os.environ.get("JEV_OUTPUT_DIR", "results/p4f6_constrained_action_alfworld_7b"))
ARCHIVE_BASE = os.environ.get(
    "JEV_ARCHIVE_BASE", str(OUTPUT_DIR.with_name(f"{OUTPUT_DIR.name}_artifacts"))
)


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing required frozen script: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_constrained_choose_action(base, f4):
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
                    "You control ALFWorld. Use the shared public scaffold and current evidence. "
                    "Choose exactly one currently allowed command. Output only that command."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{f4.SHARED_PUBLIC_SCAFFOLD}\n"
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
        prompt_len = int(encoded["input_ids"].shape[1])

        sequence_to_command = {}
        for command in commands:
            for rendered in (command, " " + command):
                ids = tuple(tokenizer.encode(rendered, add_special_tokens=False))
                if ids:
                    sequence_to_command.setdefault(ids, command)
        sequences = tuple(sequence_to_command)
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id

        def allowed_tokens(_batch_id, input_ids):
            prefix = tuple(input_ids[prompt_len:].tolist())
            allowed = set()
            for sequence in sequences:
                if len(prefix) < len(sequence) and sequence[: len(prefix)] == prefix:
                    allowed.add(sequence[len(prefix)])
                elif prefix == sequence:
                    allowed.add(eos_id)
            return sorted(allowed) if allowed else [eos_id]

        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=max(len(x) for x in sequences) + 1,
                do_sample=False,
                prefix_allowed_tokens_fn=allowed_tokens,
                pad_token_id=eos_id,
            )
        elapsed = time.perf_counter() - started
        generated = output[0, prompt_len:].tolist()
        while generated and generated[-1] in {eos_id, pad_id}:
            generated.pop()
        selected = sequence_to_command.get(tuple(generated))
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if selected is None:
            exact = {cmd.lower(): cmd for cmd in commands}
            selected = exact.get(raw.lower())
        if selected is None:
            raise RuntimeError(
                f"Constrained decoder emitted a non-candidate sequence: {generated!r} / {raw!r}"
            )
        return selected, {
            "raw_model_output": raw,
            "parsed_command": selected,
            "parse_method": "prefix_constrained_legal_command",
            "parse_ok": True,
            "latency_seconds": round(elapsed, 6),
            "input_tokens": prompt_len,
            "output_tokens": int(output.shape[1] - prompt_len),
            "shared_public_scaffold_applied": True,
            "constrained_decoder": True,
        }

    return choose_action


def main() -> None:
    base = load_module(BASE_SCRIPT, "p4f3_base")
    f4 = load_module(F4_SCRIPT, "p4f4_scaffold")
    f5 = load_module(F5_SCRIPT, "p4f5_fresh_state")
    base.OUTPUT_DIR = OUTPUT_DIR
    base.ARCHIVE_BASE = ARCHIVE_BASE
    base.choose_action = build_constrained_choose_action(base, f4)
    base.run_arm = f5.build_fresh_run_arm(base)
    base.main()

    report_path = OUTPUT_DIR / "p4f_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (OUTPUT_DIR / "p4f_complete_trajectories.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    report["schema_version"] = "JEV_P4F6_CONSTRAINED_ACTION_ALFWORLD_7B_REPORT_V1"
    report["purpose"] = (
        "Remove parser-fallback confounding and verify the paired JEV Memory effect "
        "with decoding constrained to current legal actions"
    )
    report["design"].update(
        {
            "shared_public_scaffold": True,
            "shared_public_scaffold_sha256": base.stable_hash(f4.SHARED_PUBLIC_SCAFFOLD),
            "action_state_refresh": "assign infos returned by env.step to the next iteration",
            "action_selection": "prefix-constrained trie over current legal commands",
            "fallback_action_policy": "none; a constraint violation aborts the run",
            "only_arm_difference": "pre-existing task-type JEV Memory text",
        }
    )
    report["p4f5_reference"] = {
        "baseline_successes": 6,
        "memory_successes": 8,
        "absolute_success_rate_delta": 0.25,
        "parse_fallbacks": 62,
    }
    report["interface_validation"] = {
        "dynamic_manipulation_steps": sum(x["dynamic_manipulation_steps"] for x in rows),
        "parse_fallbacks": sum(x["parse_fallbacks"] for x in rows),
        "constrained_decoder_violations": sum(
            int(step.get("parse_method") != "prefix_constrained_legal_command")
            for row in rows
            for step in row["trajectory"]
        ),
        "action_state_refresh_observed": any(x["dynamic_manipulation_steps"] > 0 for x in rows),
    }
    report["decision"]["production_memory_authorized"] = False
    report["decision"]["next_gate"] = (
        "If the paired effect remains positive, expand to a larger stratified frozen public set; "
        "otherwise treat P4-F5 as parser-mediated and redesign Memory before RSI promotion."
    )
    report["decision"]["interpretation"] = (
        "P4-F6 is the first run without parser fallback. It remains evaluation-only and does not "
        "authorize Memory writes, training labels, or production actions."
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "shared_public_demo.txt").write_text(f4.SHARED_PUBLIC_SCAFFOLD, encoding="utf-8")
    (OUTPUT_DIR / "public_demo_source.json").write_text(
        json.dumps(f4.PUBLIC_SOURCE, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), OUTPUT_DIR / Path(__file__).name)
    shutil.copy2(BASE_SCRIPT, OUTPUT_DIR / BASE_SCRIPT.name)
    shutil.copy2(F4_SCRIPT, OUTPUT_DIR / F4_SCRIPT.name)
    shutil.copy2(F5_SCRIPT, OUTPUT_DIR / F5_SCRIPT.name)
    (OUTPUT_DIR / "README.md").write_text(
        "# JEV P4-F6 constrained-action ALFWorld audit\n\n"
        "Every selected action is generated inside a legal-command trie; parser fallback is disabled. "
        "All authorization fields remain false.\n",
        encoding="utf-8",
    )
    archive = shutil.make_archive(
        ARCHIVE_BASE, "zip", root_dir=OUTPUT_DIR.parent, base_dir=OUTPUT_DIR.name
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("P4-F6 archive:", archive)


if __name__ == "__main__":
    main()

