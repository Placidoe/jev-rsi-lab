#!/usr/bin/env python3
"""P4-F3: ALFWorld capability-floor calibration with an exact-command interface.

Both arms receive the same non-learning control scaffold: diagnostic commands
are removed and a command already attempted in the identical textual state is
temporarily blocked.  The only arm difference is the fixed task-type Memory.
This keeps the Memory comparison causal while testing whether P4-E's 0/8 was
caused by a broken action interface rather than by the Memory itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from alfworld.agents.environment import get_environment


SEED = 20260927
# All raw tasks, model outputs and trajectories remain under the ignored results/ root.
MODEL_ID = os.environ.get("JEV_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
DATA_DIR = Path(os.environ.get("JEV_ALFWORLD_DATA_DIR", "work/alfworld_data"))
OUTPUT_DIR = Path(os.environ.get("JEV_OUTPUT_DIR", "results/p4f3_exact_command_alfworld_7b"))
ARCHIVE_BASE = os.environ.get(
    "JEV_ARCHIVE_BASE", str(OUTPUT_DIR.with_name(f"{OUTPUT_DIR.name}_artifacts"))
)
PILOT_GAMES = int(os.environ.get("JEV_PILOT_GAMES", "8"))
MAX_STEPS = int(os.environ.get("JEV_MAX_STEPS", "40"))
DIAGNOSTIC_COMMANDS = {"help", "inventory"}

MEMORY_RULES = {
    "pick_and_place_simple": "Find the requested object, take it, navigate to the requested receptacle, and move it there. Use observations as evidence and do not retry an unchanged failed action.",
    "look_at_obj_in_light": "Find the requested object and the desk lamp. Turn on the lamp with the admissible use command, then examine the requested object. Search new receptacles when evidence is missing.",
    "pick_clean_then_place_in_recep": "Obtain the requested object, clean it at a sinkbasin, then place it in the requested receptacle. Preserve this order.",
    "pick_heat_then_place_in_recep": "Obtain the requested object, heat it with the required appliance, then place it in the requested receptacle. Preserve this order.",
    "pick_cool_then_place_in_recep": "Obtain the requested object, cool it with the fridge, then place it in the requested receptacle. Preserve this order.",
    "pick_two_obj_and_place": "Track both requested objects. Obtain and place both in the requested receptacle; do not stop after the first.",
}


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def config() -> dict:
    return {
        "dataset": {
            "data_path": str(DATA_DIR / "json_2.1.1" / "train"),
            "eval_id_data_path": str(DATA_DIR / "json_2.1.1" / "valid_seen"),
            "eval_ood_data_path": str(DATA_DIR / "json_2.1.1" / "valid_unseen"),
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "logic": {
            "domain": str(DATA_DIR / "logic" / "alfred.pddl"),
            "grammar": str(DATA_DIR / "logic" / "alfred.twl2"),
        },
        "env": {
            "goal_desc_human_anns_prob": 0.0,
            "task_types": [1, 2, 3, 4, 5, 6],
            "domain_randomization": False,
            "expert_type": "handcoded",
        },
        "general": {"training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": MAX_STEPS}},
    }


def relative_game_id(game_file: str) -> str:
    marker = "/valid_unseen/"
    return game_file.split(marker, 1)[-1] if marker in game_file else Path(game_file).name


def choose_holdout_games(game_files: list[str]) -> list[str]:
    selected = []
    for path in sorted(game_files):
        gid = relative_game_id(path)
        if int(stable_hash(gid)[:8], 16) % 4 == 0:
            selected.append(path)
        if len(selected) == PILOT_GAMES:
            break
    if len(selected) != PILOT_GAMES:
        raise RuntimeError(f"Expected {PILOT_GAMES} deterministic games, found {len(selected)}")
    return selected


def task_type_for(game_file: str) -> str:
    path = str(game_file)
    return next((key for key in MEMORY_RULES if key in path), "pick_and_place_simple")


def normalize(text: str, limit: int = 1400) -> str:
    return text.replace("\r", " ").strip()[-limit:]


def state_key(observation: str) -> str:
    return stable_hash(normalize(observation, 2000))


def guarded_commands(commands: list[str], observation: str, tried: dict[str, set[str]]) -> tuple[list[str], dict]:
    original = sorted(str(x) for x in commands)
    candidates = [x for x in original if x not in DIAGNOSTIC_COMMANDS]
    removed_diagnostics = len(original) - len(candidates)
    key = state_key(observation)
    novel = [x for x in candidates if x not in tried[key]]
    exhausted_reset = False
    if novel:
        candidates = novel
    elif candidates:
        # The environment may legitimately require revisiting a state later.
        # Reset only after every admissible action in this exact state was tried.
        tried[key].clear()
        exhausted_reset = True
    if not candidates:
        candidates = ["look"] if "look" in original else original[:1]
    return candidates, {
        "state_key": key,
        "original_command_count": len(original),
        "guarded_command_count": len(candidates),
        "removed_diagnostic_count": removed_diagnostics,
        "blocked_previously_tried_count": max(0, len(original) - removed_diagnostics - len(candidates)),
        "exhausted_state_reset": exhausted_reset,
    }


def choose_action(model, tokenizer, task: str, observation: str, history: list[dict], commands: list[str], memory: str | None) -> tuple[str, dict]:
    recent = history[-8:]
    history_text = "\n".join(
        f"{x['step']}: {x['action']} -> {normalize(x['observation'], 360)}"
        for x in recent
    ) or "(none)"
    command_text = "\n".join(f"- {cmd}" for cmd in commands)
    memory_text = memory or "No external Memory is provided."
    messages = [
        {
            "role": "system",
            "content": (
                "You control ALFWorld. Choose one command that makes concrete progress toward the task. "
                "Prefer navigation/search, then object manipulation, then the required final operation. "
                "Never invent a command. Copy exactly one allowed command verbatim. "
                "Return only that command, without an index or explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Goal and initial scene:\n{normalize(task)}\n\n"
                f"Optional JEV Memory:\n{memory_text}\n\n"
                f"Recent evidence:\n{history_text}\n\n"
                f"Current observation:\n{normalize(observation)}\n\n"
                f"Allowed commands after shared loop guard:\n{command_text}\n\nCommand:"
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started
    raw = tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
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
    }


def run_arm(engine, game_file: str, arm: str, model, tokenizer) -> dict:
    engine.game_files = [game_file]
    engine.num_games = 1
    env = engine.init_env(batch_size=1)
    obs, info = env.reset()
    initial = str(obs[0])
    current = initial
    task_type = task_type_for(game_file)
    memory = MEMORY_RULES[task_type] if arm == "guarded_memory" else None
    trajectory = []
    tried: dict[str, set[str]] = defaultdict(set)
    repeated_actions = 0
    parse_fallbacks = 0
    won = False
    done = False
    last_action = None
    max_score = 0.0
    try:
        for step in range(MAX_STEPS):
            candidates, guard_meta = guarded_commands(info["admissible_commands"][0], current, tried)
            action, model_meta = choose_action(model, tokenizer, initial, current, trajectory, candidates, memory)
            tried[guard_meta["state_key"]].add(action)
            repeated_actions += int(action == last_action)
            parse_fallbacks += int(not model_meta["parse_ok"])
            before = current
            next_obs, scores, dones, infos = env.step([action])
            current = str(next_obs[0])
            score = float(scores[0])
            max_score = max(max_score, score)
            won = bool(infos["won"][0])
            done = bool(dones[0])
            trajectory.append({
                "step": step,
                "observation_before": normalize(before),
                "admissible_commands_after_guard": candidates,
                "action": action,
                "action_repeated": action == last_action,
                "observation": current,
                "score": score,
                "won": won,
                "done": done,
                **guard_meta,
                **model_meta,
            })
            last_action = action
            if done:
                break
    finally:
        env.close()

    action_counts = Counter(x["action"] for x in trajectory)
    return {
        "schema_version": "JEV_P4F3_EXACT_COMMAND_ALFWORLD_TRAJECTORY_V1",
        "game_id": relative_game_id(game_file),
        "game_sha256": stable_hash(relative_game_id(game_file)),
        "task_type": task_type,
        "arm": arm,
        "memory_injected": memory,
        "initial_observation": initial,
        "success": won,
        "done": done,
        "steps": len(trajectory),
        "max_score": max_score,
        "unique_actions": len(action_counts),
        "top_actions": action_counts.most_common(8),
        "repeated_actions": repeated_actions,
        "parse_fallbacks": parse_fallbacks,
        "diagnostic_commands_removed": sum(x["removed_diagnostic_count"] for x in trajectory),
        "previously_tried_actions_blocked": sum(x["blocked_previously_tried_count"] for x in trajectory),
        "input_tokens": sum(x["input_tokens"] for x in trajectory),
        "output_tokens": sum(x["output_tokens"] for x in trajectory),
        "latency_seconds": round(sum(x["latency_seconds"] for x in trajectory), 6),
        "trajectory": trajectory,
        "memory_write_authorized": False,
        "training_label_authorized": False,
        "production_action_authorized": False,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError("P4-F requires the enabled Kaggle GPU runtime")

    engine = get_environment("AlfredTWEnv")(config(), train_eval="eval_out_of_distribution")
    games = choose_holdout_games(sorted(engine.game_files))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="balanced", max_memory={0: "13GiB", 1: "13GiB", "cpu": "16GiB"}, low_cpu_mem_usage=True)
    model.eval()

    rows = []
    for idx, game in enumerate(games):
        arms = ["guarded_baseline", "guarded_memory"] if idx % 2 == 0 else ["guarded_memory", "guarded_baseline"]
        for arm in arms:
            row = run_arm(engine, game, arm, model, tokenizer)
            rows.append(row)
            print(row["game_id"], arm, row["success"], row["steps"], row["max_score"], row["unique_actions"], flush=True)

    pairs = {}
    for row in rows:
        pairs.setdefault(row["game_id"], {})[row["arm"]] = row
    paired = []
    for game_id, arms in sorted(pairs.items()):
        baseline = arms["guarded_baseline"]
        memory = arms["guarded_memory"]
        transition = (
            "corrected" if not baseline["success"] and memory["success"] else
            "regressed" if baseline["success"] and not memory["success"] else
            "stable_success" if baseline["success"] else "stable_failure"
        )
        paired.append({
            "game_id": game_id,
            "task_type": baseline["task_type"],
            "baseline_success": baseline["success"],
            "memory_success": memory["success"],
            "transition": transition,
            "baseline_max_score": baseline["max_score"],
            "memory_max_score": memory["max_score"],
            "baseline_unique_actions": baseline["unique_actions"],
            "memory_unique_actions": memory["unique_actions"],
            "baseline_repeated_actions": baseline["repeated_actions"],
            "memory_repeated_actions": memory["repeated_actions"],
        })

    baseline_success = sum(x["baseline_success"] for x in paired)
    memory_success = sum(x["memory_success"] for x in paired)
    corrected = [x["game_id"] for x in paired if x["transition"] == "corrected"]
    regressed = [x["game_id"] for x in paired if x["transition"] == "regressed"]
    floor_reached = baseline_success > 0 or memory_success > 0
    report = {
        "schema_version": "JEV_P4F3_EXACT_COMMAND_ALFWORLD_7B_REPORT_V1",
        "purpose": "Remove numeric-index positional bias and calibrate ALFWorld capability floor before RSI Memory conclusions",
        "runtime": {
            "model": MODEL_ID,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "seed": SEED,
        },
        "design": {
            "frozen_games": len(paired),
            "max_steps": MAX_STEPS,
            "same_model_decoding_and_shared_guard": True,
            "counterbalanced_arm_order": True,
            "shared_guard": "remove help/inventory and block an already-tried action in the identical text state",
            "action_interface": "model copies one exact admissible command; numeric option indices are not shown",
            "only_arm_difference": "task-type Memory text",
        },
        "p4e_reference": {
            "baseline_successes": 0,
            "memory_successes": 0,
            "baseline_repeated_actions": 207,
            "memory_repeated_actions": 169,
        },
        "guarded_baseline": {
            "successes": baseline_success,
            "games": len(paired),
            "success_rate": baseline_success / len(paired),
            "repeated_actions": sum(x["baseline_repeated_actions"] for x in paired),
            "mean_unique_actions": sum(x["baseline_unique_actions"] for x in paired) / len(paired),
        },
        "guarded_memory": {
            "successes": memory_success,
            "games": len(paired),
            "success_rate": memory_success / len(paired),
            "repeated_actions": sum(x["memory_repeated_actions"] for x in paired),
            "mean_unique_actions": sum(x["memory_unique_actions"] for x in paired) / len(paired),
        },
        "effect": {
            "corrected_games": corrected,
            "regressed_games": regressed,
            "net_corrections": len(corrected) - len(regressed),
            "absolute_success_rate_delta": (memory_success - baseline_success) / len(paired),
        },
        "decision": {
            "capability_floor_reached": floor_reached,
            "memory_effect_identifiable": floor_reached,
            "pilot_positive": len(corrected) > len(regressed),
            "production_memory_authorized": False,
            "next_gate": (
                "Replicate with two more seeds and a larger frozen set" if floor_reached
                else "If capability floor remains zero, add a shared public ALFWorld demonstration scaffold before testing Memory"
            ),
        },
    }

    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing run output: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)
    write_jsonl(OUTPUT_DIR / "p4f_complete_trajectories.jsonl", rows)
    write_jsonl(OUTPUT_DIR / "p4f_paired_cases.jsonl", paired)
    (OUTPUT_DIR / "p4f_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "p4f_memory_rules.json").write_text(json.dumps(MEMORY_RULES, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), OUTPUT_DIR / Path(__file__).name)
    (OUTPUT_DIR / "README.md").write_text(
        "# JEV P4-F3 exact-command ALFWorld 7B capability-floor calibration\n\n"
        "Both arms share the same exact-command interface and loop guard; only Memory differs. All authorization fields remain false.\n",
        encoding="utf-8",
    )
    archive = shutil.make_archive(ARCHIVE_BASE, "zip", root_dir=OUTPUT_DIR.parent, base_dir=OUTPUT_DIR.name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Archive:", archive)


if __name__ == "__main__":
    main()

