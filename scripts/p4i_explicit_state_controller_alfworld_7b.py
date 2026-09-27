#!/usr/bin/env python3
"""P4-I: compare frozen text Memory with an explicit JEV task-state controller.

The P4-G hidden validation split remains unopened.  Both arms use the same
Qwen2.5-7B policy, public scaffold, prefix-constrained legal-action decoder,
fresh environment state, discovery games, and counterbalanced order.  The
candidate arm adds a typed state compiler and phase-aware legal-action filter;
it does not learn from or inspect hidden games.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
P4G_DIR = Path(os.environ.get("JEV_P4G_DIR", "results/p4g_preregistered_discovery_alfworld_7b"))
BASE_SCRIPT = Path(os.environ.get("JEV_P4F3_SCRIPT", str(SCRIPT_DIR / "p4f3_exact_command_alfworld_7b.py")))
F4_SCRIPT = Path(os.environ.get("JEV_P4F4_SCRIPT", str(SCRIPT_DIR / "p4f4_shared_public_demo_alfworld_7b.py")))
F6_SCRIPT = Path(os.environ.get("JEV_P4F6_SCRIPT", str(SCRIPT_DIR / "p4f6_constrained_action_alfworld_7b.py")))
OUTPUT_DIR = Path(os.environ.get("JEV_OUTPUT_DIR", "results/p4i_explicit_state_controller_alfworld_7b"))
ARCHIVE_BASE = os.environ.get(
    "JEV_ARCHIVE_BASE", str(OUTPUT_DIR.with_name(f"{OUTPUT_DIR.name}_artifacts"))
)


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def exact_mcnemar_p(corrected: int, regressed: int) -> float:
    n = corrected + regressed
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(corrected, regressed) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def parse_goal(game_file: str, base) -> dict:
    game_id = base.relative_game_id(game_file)
    head = game_id.split("/", 1)[0]
    parts = head.rsplit("-", 4)
    if len(parts) != 5:
        raise ValueError(f"Unexpected ALFWorld game id: {game_id}")
    task_type, target, _unused, destination, _scene = parts
    transform = {
        "pick_clean_then_place_in_recep": "clean",
        "pick_heat_then_place_in_recep": "heat",
        "pick_cool_then_place_in_recep": "cool",
        "look_at_obj_in_light": "light",
    }.get(task_type)
    return {
        "task_type": task_type,
        "target_type": target.lower(),
        "destination_type": destination.lower(),
        "required_transform": transform,
        "required_count": 2 if task_type == "pick_two_obj_and_place" else 1,
    }


def initial_state(goal: dict) -> dict:
    return {
        **goal,
        "phase": "search",
        "holding": None,
        "transformed_current": False,
        "lamp_used": False,
        "placed_count": 0,
        "visited_locations": [],
        "opened_containers": [],
        "controller_steps": 0,
        "controller_singleton_steps": 0,
    }


def command_location(command: str) -> str | None:
    match = re.match(r"go to (.+)$", command.lower())
    return match.group(1) if match else None


def is_target_command(command: str, target: str, verb: str | None = None) -> bool:
    lowered = command.lower()
    return (verb is None or lowered.startswith(verb + " ")) and target in lowered


def controller_candidates(commands: list[str], state: dict) -> tuple[list[str], dict]:
    """Compile typed state into a safe, progress-oriented legal-action subset."""
    commands = list(commands)
    target = state["target_type"]
    destination = state["destination_type"]
    holding = state["holding"]
    transform = state["required_transform"]
    reason = "policy_fallback"

    def select(items: list[str], why: str) -> tuple[list[str], dict] | None:
        nonlocal reason
        if items:
            reason = why
            return items, {
                "controller_reason": reason,
                "controller_original_count": len(commands),
                "controller_filtered_count": len(items),
                "controller_state": json.loads(json.dumps(state)),
            }
        return None

    if holding:
        if transform == "light":
            if not state["lamp_used"]:
                result = select([c for c in commands if c.lower().startswith("use desklamp")], "use_required_lamp")
                if result:
                    return result
                result = select([c for c in commands if c.lower().startswith("go to desk")], "navigate_to_lamp")
                if result:
                    return result
            result = select([c for c in commands if c.lower().startswith("examine ") and target in c.lower()], "examine_target_under_light")
            if result:
                return result
        elif transform and not state["transformed_current"]:
            result = select([c for c in commands if c.lower().startswith(transform + " ") and holding in c.lower()], "apply_required_transform")
            if result:
                return result
            appliance = {"clean": "sinkbasin", "heat": "microwave", "cool": "fridge"}[transform]
            result = select([c for c in commands if c.lower().startswith("open ") and appliance in c.lower()], "open_required_appliance")
            if result:
                return result
            result = select([c for c in commands if c.lower().startswith("go to ") and appliance in c.lower()], "navigate_to_required_appliance")
            if result:
                return result
        else:
            result = select([c for c in commands if c.lower().startswith("move ") and holding in c.lower() and destination in c.lower()], "place_target_at_destination")
            if result:
                return result
            result = select([c for c in commands if c.lower().startswith("open ") and destination in c.lower()], "open_destination")
            if result:
                return result
            result = select([c for c in commands if c.lower().startswith("go to ") and destination in c.lower()], "navigate_to_destination")
            if result:
                return result
    else:
        result = select([c for c in commands if is_target_command(c, target, "take")], "acquire_exact_target")
        if result:
            return result
        result = select([c for c in commands if c.lower().startswith("open ")], "open_current_search_container")
        if result:
            return result
        visited = set(state["visited_locations"])
        unseen = [c for c in commands if c.lower().startswith("go to ") and command_location(c) not in visited]
        result = select(unseen, "visit_unsearched_location")
        if result:
            return result

    safe = [
        c for c in commands
        if not c.lower().startswith(("close ", "take "))
        or is_target_command(c, target, "take")
    ]
    safe = safe or commands
    return safe, {
        "controller_reason": reason,
        "controller_original_count": len(commands),
        "controller_filtered_count": len(safe),
        "controller_state": json.loads(json.dumps(state)),
    }


def advance_state(state: dict, action: str) -> None:
    lowered = action.lower()
    location = command_location(action)
    if location and location not in state["visited_locations"]:
        state["visited_locations"].append(location)
    if lowered.startswith("open "):
        container = lowered[len("open "):]
        if container not in state["opened_containers"]:
            state["opened_containers"].append(container)
    take = re.match(r"take (.+?) from ", lowered)
    if take and state["target_type"] in take.group(1):
        state["holding"] = take.group(1)
        state["transformed_current"] = False
        state["phase"] = "transform" if state["required_transform"] not in (None, "light") else "deliver"
    if state["holding"] and lowered.startswith(("clean ", "heat ", "cool ")) and state["holding"] in lowered:
        state["transformed_current"] = True
        state["phase"] = "deliver"
    if lowered.startswith("use desklamp"):
        state["lamp_used"] = True
        state["phase"] = "verify"
    if state["holding"] and lowered.startswith("move ") and state["holding"] in lowered and state["destination_type"] in lowered:
        state["placed_count"] += 1
        state["holding"] = None
        state["transformed_current"] = False
        state["phase"] = "search" if state["placed_count"] < state["required_count"] else "complete"
    state["controller_steps"] += 1


def main() -> None:
    required = [BASE_SCRIPT, F4_SCRIPT, F6_SCRIPT, P4G_DIR / "p4g_preregistered_split_manifest.json"]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    base = load_module(BASE_SCRIPT, "p4i_base")
    f4 = load_module(F4_SCRIPT, "p4i_f4")
    f6 = load_module(F6_SCRIPT, "p4i_f6")
    constrained_choose = f6.build_constrained_choose_action(base, f4)

    manifest = json.loads((P4G_DIR / "p4g_preregistered_split_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("hidden_validation_executed") is not False:
        raise RuntimeError("P4-G hidden-validation boundary is not intact")
    discovery_ids = list(manifest["discovery_game_ids"])
    if set(discovery_ids) & set(manifest["hidden_validation_game_ids"]):
        raise RuntimeError("Discovery/hidden overlap")

    def choose_discovery(game_files):
        by_id = {base.relative_game_id(path): path for path in game_files}
        missing = [gid for gid in discovery_ids if gid not in by_id]
        if missing:
            raise RuntimeError(f"Missing discovery games: {missing}")
        return [by_id[gid] for gid in discovery_ids]

    def run_arm(engine, game_file: str, arm: str, model, tokenizer) -> dict:
        engine.game_files = [game_file]
        engine.num_games = 1
        env = engine.init_env(batch_size=1)
        obs, info = env.reset()
        initial = str(obs[0])
        current = initial
        goal = parse_goal(game_file, base)
        state = initial_state(goal)
        memory = base.MEMORY_RULES[goal["task_type"]]
        trajectory = []
        tried: dict[str, set[str]] = defaultdict(set)
        last_action = None
        repeated_actions = 0
        dynamic_steps = 0
        won = done = False
        max_score = 0.0
        try:
            for step in range(base.MAX_STEPS):
                candidates, guard_meta = base.guarded_commands(info["admissible_commands"][0], current, tried)
                dynamic_steps += int(any(c.lower().startswith(("take ", "open ", "use ", "move ", "clean ", "heat ", "cool ")) for c in candidates))
                controller_meta = {
                    "controller_reason": "disabled_old_memory_arm",
                    "controller_original_count": len(candidates),
                    "controller_filtered_count": len(candidates),
                    "controller_state": json.loads(json.dumps(state)),
                }
                policy_candidates = candidates
                if arm == "guarded_memory":
                    policy_candidates, controller_meta = controller_candidates(candidates, state)
                    state["controller_singleton_steps"] += int(len(policy_candidates) == 1)
                action, model_meta = constrained_choose(model, tokenizer, initial, current, trajectory, policy_candidates, memory)
                tried[guard_meta["state_key"]].add(action)
                before = current
                next_obs, scores, dones, infos = env.step([action])
                current = str(next_obs[0])
                info = infos
                score = float(scores[0])
                max_score = max(max_score, score)
                won = bool(infos["won"][0])
                done = bool(dones[0])
                advance_state(state, action)
                trajectory.append({
                    "step": step,
                    "observation_before": base.normalize(before),
                    "admissible_commands_after_guard": candidates,
                    "admissible_commands_after_controller": policy_candidates,
                    "action": action,
                    "action_repeated": action == last_action,
                    "observation": current,
                    "score": score,
                    "won": won,
                    "done": done,
                    **guard_meta,
                    **controller_meta,
                    **model_meta,
                })
                repeated_actions += int(action == last_action)
                last_action = action
                if done:
                    break
        finally:
            env.close()

        counts = Counter(row["action"] for row in trajectory)
        return {
            "schema_version": "JEV_P4I_EXPLICIT_STATE_CONTROLLER_TRAJECTORY_V1",
            "game_id": base.relative_game_id(game_file),
            "game_sha256": base.stable_hash(base.relative_game_id(game_file)),
            "task_type": goal["task_type"],
            "arm": arm,
            "comparison_arm": "old_memory" if arm == "guarded_baseline" else "jev_state_controller",
            "memory_injected": memory,
            "goal_schema": goal,
            "final_controller_state": state,
            "initial_observation": initial,
            "success": won,
            "done": done,
            "steps": len(trajectory),
            "max_score": max_score,
            "unique_actions": len(counts),
            "top_actions": counts.most_common(8),
            "repeated_actions": repeated_actions,
            "parse_fallbacks": 0,
            "dynamic_manipulation_steps": dynamic_steps,
            "controller_filtered_steps": sum(row["controller_filtered_count"] < row["controller_original_count"] for row in trajectory),
            "controller_singleton_steps": state["controller_singleton_steps"],
            "input_tokens": sum(row["input_tokens"] for row in trajectory),
            "output_tokens": sum(row["output_tokens"] for row in trajectory),
            "latency_seconds": round(sum(row["latency_seconds"] for row in trajectory), 6),
            "trajectory": trajectory,
            "memory_write_authorized": False,
            "training_label_authorized": False,
            "production_action_authorized": False,
        }

    base.OUTPUT_DIR = OUTPUT_DIR
    base.ARCHIVE_BASE = ARCHIVE_BASE
    base.choose_holdout_games = choose_discovery
    base.run_arm = run_arm
    base.main()

    rows = read_jsonl(OUTPUT_DIR / "p4f_complete_trajectories.jsonl")
    pairs_by_game = {}
    for row in rows:
        pairs_by_game.setdefault(row["game_id"], {})[row["arm"]] = row
    paired = []
    for game_id, arms in sorted(pairs_by_game.items()):
        old = arms["guarded_baseline"]
        candidate = arms["guarded_memory"]
        transition = (
            "corrected" if not old["success"] and candidate["success"] else
            "regressed" if old["success"] and not candidate["success"] else
            "stable_success" if old["success"] else "stable_failure"
        )
        paired.append({
            "game_id": game_id,
            "task_type": old["task_type"],
            "old_memory_success": old["success"],
            "state_controller_success": candidate["success"],
            "transition": transition,
            "old_memory_steps": old["steps"],
            "state_controller_steps": candidate["steps"],
        })
    write_jsonl(OUTPUT_DIR / "p4i_paired_cases.jsonl", paired)

    old_successes = sum(row["old_memory_success"] for row in paired)
    candidate_successes = sum(row["state_controller_success"] for row in paired)
    corrected = sum(row["transition"] == "corrected" for row in paired)
    regressed = sum(row["transition"] == "regressed" for row in paired)
    promote = candidate_successes > old_successes and regressed == 0
    per_task = {}
    for task_type in sorted({row["task_type"] for row in paired}):
        subset = [row for row in paired if row["task_type"] == task_type]
        per_task[task_type] = {
            "games": len(subset),
            "old_memory_successes": sum(row["old_memory_success"] for row in subset),
            "state_controller_successes": sum(row["state_controller_success"] for row in subset),
            "corrected": sum(row["transition"] == "corrected" for row in subset),
            "regressed": sum(row["transition"] == "regressed" for row in subset),
        }
    report = {
        "schema_version": "JEV_P4I_EXPLICIT_STATE_CONTROLLER_ALFWORLD_7B_REPORT_V1",
        "purpose": "Test whether a typed JEV state compiler improves long-horizon action selection beyond text Memory alone",
        "design": {
            "split": "P4-G discovery only",
            "games": len(paired),
            "hidden_validation_executed": False,
            "same_model_memory_scaffold_decoder_and_games": True,
            "counterbalanced_arm_order": True,
            "old_arm": "frozen text Memory plus constrained Qwen policy",
            "candidate_arm": "same text Memory and policy plus typed state compiler and phase-aware legal-action filter",
            "state_fields": ["target_type", "destination_type", "required_transform", "required_count", "holding", "transformed_current", "lamp_used", "placed_count", "visited_locations", "phase"],
            "object_ids_or_hidden_outcomes_hardcoded": False,
            "action_selection": "prefix-constrained trie over current legal commands",
            "fallback_action_policy": "none",
        },
        "old_memory": {"successes": old_successes, "games": len(paired), "success_rate": old_successes / len(paired)},
        "jev_state_controller": {"successes": candidate_successes, "games": len(paired), "success_rate": candidate_successes / len(paired)},
        "effect": {
            "corrected_games": [row["game_id"] for row in paired if row["transition"] == "corrected"],
            "regressed_games": [row["game_id"] for row in paired if row["transition"] == "regressed"],
            "net_corrections": corrected - regressed,
            "absolute_success_rate_delta": (candidate_successes - old_successes) / len(paired),
            "paired_exact_mcnemar_two_sided_p": exact_mcnemar_p(corrected, regressed),
        },
        "per_task_type": per_task,
        "interface_validation": {
            "dynamic_manipulation_steps": sum(row["dynamic_manipulation_steps"] for row in rows),
            "parse_fallbacks": 0,
            "constrained_decoder_violations": sum(step.get("parse_method") != "prefix_constrained_legal_command" for row in rows for step in row["trajectory"]),
            "controller_filtered_steps": sum(row["controller_filtered_steps"] for row in rows if row["arm"] == "guarded_memory"),
            "controller_singleton_steps": sum(row["controller_singleton_steps"] for row in rows if row["arm"] == "guarded_memory"),
        },
        "decision": {
            "candidate_wins_discovery_gate": promote,
            "freeze_candidate_before_hidden_validation": promote,
            "next_gate": "Open pre-registered hidden validation once with the frozen controller" if promote else "Reject or revise the controller using discovery evidence only; keep hidden validation unopened",
            "production_memory_authorized": False,
            "training_label_authorized": False,
            "business_action_authorized": False,
        },
    }
    (OUTPUT_DIR / "p4f_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "p4g_preregistered_split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), OUTPUT_DIR / Path(__file__).name)
    for source in (BASE_SCRIPT, F4_SCRIPT, F6_SCRIPT):
        shutil.copy2(source, OUTPUT_DIR / source.name)
    (OUTPUT_DIR / "README.md").write_text(
        "# JEV P4-I explicit state controller\n\nDiscovery-only causal comparison. Hidden validation is unopened and every authorization field is false.\n",
        encoding="utf-8",
    )
    archive = shutil.make_archive(ARCHIVE_BASE, "zip", root_dir=OUTPUT_DIR.parent, base_dir=OUTPUT_DIR.name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("P4-I archive:", archive)


if __name__ == "__main__":
    main()

