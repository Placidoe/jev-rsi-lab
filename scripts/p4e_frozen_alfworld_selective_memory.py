#!/usr/bin/env python3
"""P4-E: public ALFWorld frozen intervention, baseline vs selective JEV Memory.

This pilot uses only public ALFWorld games and generic rules distilled from the
public Reflexion build evidence.  It saves complete step trajectories.  No
result authorizes production Memory, training labels, or real tool actions.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from alfworld.agents.environment import get_environment


SEED = 20260927
MODEL_ID = os.environ.get("JEV_MODEL_ID", "Qwen/Qwen2.5-Coder-3B-Instruct")
DATA_DIR = Path(os.environ.get("JEV_ALFWORLD_DATA_DIR", "work/alfworld_data"))
OUTPUT_DIR = Path(os.environ.get("JEV_OUTPUT_DIR", "results/p4e_frozen_alfworld"))
ARCHIVE_BASE = os.environ.get("JEV_ARCHIVE_BASE", str(OUTPUT_DIR.with_name(f"{OUTPUT_DIR.name}_artifacts")))
PILOT_GAMES = int(os.environ.get("JEV_PILOT_GAMES", "8"))
MAX_STEPS = int(os.environ.get("JEV_MAX_STEPS", "30"))

MEMORY_RULES = {
    "pick_and_place_simple": "Follow the requested object and final receptacle exactly. Search systematically; after a failed action, use the new observation and choose a different admissible action.",
    "look_at_obj_in_light": "Find and activate the lamp, then examine the requested object under it. Search unvisited receptacles systematically and never repeat an unchanged failed action.",
    "pick_clean_then_place_in_recep": "Preserve the required order: obtain the object, clean it at a sinkbasin, then place it in the requested receptacle. Open a closed container before using it.",
    "pick_heat_then_place_in_recep": "Preserve the required order: obtain the object, open the heating appliance if needed, heat it, then place it in the requested receptacle. Do not repeat a failed action.",
    "pick_cool_then_place_in_recep": "Preserve the required order: obtain the object, open the fridge if needed, cool it, then place it in the requested receptacle. Do not repeat a failed action.",
    "pick_two_obj_and_place": "Track both requested objects separately. Find and take each one, then place both in the requested receptacle; do not stop after the first object.",
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


def task_type_for(game_file: str) -> str:
    name = Path(game_file).parent.name
    for key in MEMORY_RULES:
        if name.startswith(key):
            return key
    # The task-type directory is sometimes the grandparent, depending on archive layout.
    path = str(game_file)
    return next((key for key in MEMORY_RULES if key in path), "pick_and_place_simple")


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
        raise RuntimeError(f"Expected {PILOT_GAMES} deterministic holdout games, found {len(selected)}")
    return selected


def normalize_observation(text: str) -> str:
    text = text.replace("\r", " ").strip()
    return text[-1200:]


def model_action(model, tokenizer, task: str, observation: str, history: list[dict], commands: list[str], memory: str | None) -> tuple[str, dict]:
    recent = history[-6:]
    history_text = "\n".join(
        f"Step {x['step']}: action={x['action']}\nobservation={normalize_observation(x['observation'])}"
        for x in recent
    ) or "(none)"
    command_text = "\n".join(f"{i}: {cmd}" for i, cmd in enumerate(commands))
    memory_text = memory if memory else "No external Memory is provided."
    messages = [
        {
            "role": "system",
            "content": "You control a text household environment. Choose exactly one admissible command. Reply with only its integer index; no explanation.",
        },
        {
            "role": "user",
            "content": (
                f"Task and initial scene:\n{normalize_observation(task)}\n\n"
                f"JEV Memory:\n{memory_text}\n\n"
                f"Recent trajectory:\n{history_text}\n\n"
                f"Current observation:\n{normalize_observation(observation)}\n\n"
                f"Admissible commands:\n{command_text}\n\nIndex:"
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=6,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started
    raw = tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    match = re.search(r"\d+", raw)
    parsed = int(match.group()) if match else None
    parse_ok = parsed is not None and 0 <= parsed < len(commands)
    if not parse_ok:
        parsed = commands.index("look") if "look" in commands else 0
    return commands[parsed], {
        "raw_model_output": raw,
        "parsed_index": parsed,
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
    initial_observation = str(obs[0])
    current_observation = initial_observation
    task_type = task_type_for(game_file)
    memory = MEMORY_RULES[task_type] if arm == "selective_memory" else None
    trajectory: list[dict] = []
    last_action = None
    repeated_actions = 0
    parse_fallbacks = 0
    won = False
    done = False
    try:
        for step in range(MAX_STEPS):
            commands = sorted(str(x) for x in info["admissible_commands"][0])
            action, meta = model_action(
                model, tokenizer, initial_observation, current_observation,
                trajectory, commands, memory,
            )
            repeat = action == last_action
            repeated_actions += int(repeat)
            parse_fallbacks += int(not meta["parse_ok"])
            next_obs, scores, dones, infos = env.step([action])
            current_observation = str(next_obs[0])
            won = bool(infos["won"][0])
            done = bool(dones[0])
            trajectory.append({
                "step": step,
                "observation_before": normalize_observation(obs[0] if step == 0 else trajectory[-1]["observation"]),
                "admissible_commands": commands,
                "action": action,
                "action_repeated": repeat,
                "observation": current_observation,
                "score": float(scores[0]),
                "won": won,
                "done": done,
                **meta,
            })
            last_action = action
            if done:
                break
    finally:
        env.close()

    return {
        "schema_version": "JEV_P4E_ALFWORLD_TRAJECTORY_V1",
        "game_id": relative_game_id(game_file),
        "game_sha256": stable_hash(relative_game_id(game_file)),
        "split": "holdout",
        "task_type": task_type,
        "arm": arm,
        "memory_injected": memory,
        "initial_observation": initial_observation,
        "success": won,
        "done": done,
        "steps": len(trajectory),
        "repeated_actions": repeated_actions,
        "parse_fallbacks": parse_fallbacks,
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
        raise RuntimeError("P4-E requires the enabled Kaggle GPU runtime")

    engine = get_environment("AlfredTWEnv")(config(), train_eval="eval_out_of_distribution")
    engine.game_files = sorted(engine.game_files)
    games = choose_holdout_games(engine.game_files)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda:0")
    model.eval()

    rows: list[dict] = []
    for idx, game in enumerate(games):
        arms = ["baseline", "selective_memory"] if idx % 2 == 0 else ["selective_memory", "baseline"]
        for arm in arms:
            result = run_arm(engine, game, arm, model, tokenizer)
            rows.append(result)
            print(result["game_id"], arm, result["success"], result["steps"], result["repeated_actions"], flush=True)

    pairs = {}
    for row in rows:
        pairs.setdefault(row["game_id"], {})[row["arm"]] = row
    paired = []
    for game_id, arms in sorted(pairs.items()):
        baseline = arms["baseline"]
        memory = arms["selective_memory"]
        paired.append({
            "game_id": game_id,
            "task_type": baseline["task_type"],
            "baseline_success": baseline["success"],
            "memory_success": memory["success"],
            "transition": (
                "corrected" if not baseline["success"] and memory["success"] else
                "regressed" if baseline["success"] and not memory["success"] else
                "stable_success" if baseline["success"] else "stable_failure"
            ),
            "baseline_steps": baseline["steps"],
            "memory_steps": memory["steps"],
            "baseline_repeated_actions": baseline["repeated_actions"],
            "memory_repeated_actions": memory["repeated_actions"],
        })
    corrected = [x["game_id"] for x in paired if x["transition"] == "corrected"]
    regressed = [x["game_id"] for x in paired if x["transition"] == "regressed"]
    baseline_success = sum(x["baseline_success"] for x in paired)
    memory_success = sum(x["memory_success"] for x in paired)
    report = {
        "schema_version": "JEV_P4E_FROZEN_ALFWORLD_REPORT_V1",
        "purpose": "Causal pilot of selective distilled JEV Memory on frozen public ALFWorld games",
        "sources": {
            "alfworld": {"repository": "https://github.com/alfworld/alfworld", "version": "0.4.2", "license": "MIT"},
            "memory_evidence": {"repository": "https://github.com/noahshinn/reflexion", "commit": "218cf0ef1df84b05ce379dd4a8e47f17766733a0", "license": "MIT"},
        },
        "runtime": {
            "model": MODEL_ID,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "seed": SEED,
        },
        "design": {
            "public_data_only": True,
            "frozen_games": len(paired),
            "max_steps": MAX_STEPS,
            "same_model_and_decoding": True,
            "counterbalanced_arm_order": True,
            "action_space": "ALFWorld admissible commands; model chooses integer index",
            "complete_step_trajectories_saved": True,
        },
        "baseline": {"successes": baseline_success, "games": len(paired), "success_rate": baseline_success / len(paired)},
        "selective_memory": {"successes": memory_success, "games": len(paired), "success_rate": memory_success / len(paired)},
        "effect": {
            "corrected_games": corrected,
            "regressed_games": regressed,
            "net_corrections": len(corrected) - len(regressed),
            "absolute_success_rate_delta": (memory_success - baseline_success) / len(paired),
            "baseline_repeated_actions": sum(x["baseline_repeated_actions"] for x in paired),
            "memory_repeated_actions": sum(x["memory_repeated_actions"] for x in paired),
        },
        "decision": {
            "pilot_positive": len(corrected) > len(regressed),
            "production_memory_authorized": False,
            "next_gate": "Replicate on a larger pre-registered frozen set and at least two seeds before any Memory promotion.",
        },
        "limitations": [
            "Eight games are a mechanism pilot, not a powered benchmark.",
            "The agent selects from admissible commands, which is easier than unconstrained action generation.",
            "Generic rules were manually distilled from public build evidence.",
        ],
    }

    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)
    write_jsonl(OUTPUT_DIR / "p4e_complete_trajectories.jsonl", rows)
    write_jsonl(OUTPUT_DIR / "p4e_paired_cases.jsonl", paired)
    (OUTPUT_DIR / "p4e_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "p4e_memory_rules.json").write_text(json.dumps(MEMORY_RULES, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), OUTPUT_DIR / Path(__file__).name)
    (OUTPUT_DIR / "README.md").write_text(
        "# JEV P4-E frozen ALFWorld selective Memory pilot\n\n"
        "Public data only. See the report, paired cases, and complete step trajectories. "
        "All production/training authorization fields remain false.\n",
        encoding="utf-8",
    )
    archive = shutil.make_archive(ARCHIVE_BASE, "zip", root_dir=OUTPUT_DIR.parent, base_dir=OUTPUT_DIR.name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Archive:", archive)


if __name__ == "__main__":
    main()

