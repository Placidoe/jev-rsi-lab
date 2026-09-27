#!/usr/bin/env python3
"""P4-F5: repair stale ALFWorld action-state propagation and rerun the frozen pair.

P4-F3/F4 called ``env.step`` but did not assign the returned ``infos`` to the
next iteration.  Consequently every decision saw the initial navigation-only
admissible-command set.  This script changes exactly that interface defect,
while preserving the frozen games, model, decoding, shared public scaffold,
loop guard, arm order, and task-type Memory contrast from P4-F4.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = Path(os.environ.get("JEV_P4F3_SCRIPT", str(SCRIPT_DIR / "p4f3_exact_command_alfworld_7b.py")))
F4_SCRIPT = Path(os.environ.get("JEV_P4F4_SCRIPT", str(SCRIPT_DIR / "p4f4_shared_public_demo_alfworld_7b.py")))
OUTPUT_DIR = Path(os.environ.get("JEV_OUTPUT_DIR", "results/p4f5_fresh_action_state_alfworld_7b"))
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


def build_fresh_run_arm(base):
    def run_arm(engine, game_file: str, arm: str, model, tokenizer) -> dict:
        engine.game_files = [game_file]
        engine.num_games = 1
        env = engine.init_env(batch_size=1)
        obs, info = env.reset()
        initial = str(obs[0])
        current = initial
        task_type = base.task_type_for(game_file)
        memory = base.MEMORY_RULES[task_type] if arm == "guarded_memory" else None
        trajectory = []
        tried: dict[str, set[str]] = defaultdict(set)
        repeated_actions = 0
        parse_fallbacks = 0
        won = False
        done = False
        last_action = None
        max_score = 0.0
        dynamic_manipulation_steps = 0
        try:
            for step in range(base.MAX_STEPS):
                candidates, guard_meta = base.guarded_commands(
                    info["admissible_commands"][0], current, tried
                )
                dynamic_manipulation_steps += int(
                    any(
                        command.lower().startswith(
                            ("take ", "open ", "close ", "use ", "move ", "put ", "clean ", "heat ", "cool ")
                        )
                        for command in candidates
                    )
                )
                action, model_meta = base.choose_action(
                    model, tokenizer, initial, current, trajectory, candidates, memory
                )
                tried[guard_meta["state_key"]].add(action)
                repeated_actions += int(action == last_action)
                parse_fallbacks += int(not model_meta["parse_ok"])
                before = current
                next_obs, scores, dones, infos = env.step([action])
                current = str(next_obs[0])
                info = infos  # P4-F5 FIX: advance the action-state with the observation.
                score = float(scores[0])
                max_score = max(max_score, score)
                won = bool(infos["won"][0])
                done = bool(dones[0])
                trajectory.append({
                    "step": step,
                    "observation_before": base.normalize(before),
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
            "schema_version": "JEV_P4F5_FRESH_ACTION_STATE_ALFWORLD_TRAJECTORY_V1",
            "game_id": base.relative_game_id(game_file),
            "game_sha256": base.stable_hash(base.relative_game_id(game_file)),
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
            "dynamic_manipulation_steps": dynamic_manipulation_steps,
            "diagnostic_commands_removed": sum(x["removed_diagnostic_count"] for x in trajectory),
            "previously_tried_actions_blocked": sum(x["blocked_previously_tried_count"] for x in trajectory),
            "input_tokens": sum(x["input_tokens"] for x in trajectory),
            "output_tokens": sum(x["output_tokens"] for x in trajectory),
            "latency_seconds": round(sum(x["latency_seconds"] for x in trajectory), 6),
            "trajectory": trajectory,
            "action_state_refresh_fixed": True,
            "memory_write_authorized": False,
            "training_label_authorized": False,
            "production_action_authorized": False,
        }

    return run_arm


def main() -> None:
    base = load_module(BASE_SCRIPT, "p4f3_base")
    f4 = load_module(F4_SCRIPT, "p4f4_scaffold")
    base.OUTPUT_DIR = OUTPUT_DIR
    base.ARCHIVE_BASE = ARCHIVE_BASE
    base.choose_action = f4.build_choose_action(base)
    base.run_arm = build_fresh_run_arm(base)
    base.main()

    report_path = OUTPUT_DIR / "p4f_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["schema_version"] = "JEV_P4F5_FRESH_ACTION_STATE_ALFWORLD_7B_REPORT_V1"
    report["purpose"] = (
        "Repair stale admissible-action propagation, then rerun the frozen paired "
        "ALFWorld comparison before drawing any JEV Memory conclusion"
    )
    report["design"]["shared_public_scaffold"] = True
    report["design"]["shared_public_scaffold_sha256"] = base.stable_hash(
        f4.SHARED_PUBLIC_SCAFFOLD
    )
    report["design"]["action_state_refresh"] = (
        "assign infos returned by env.step to the next decision iteration"
    )
    report["design"]["only_arm_difference"] = "pre-existing task-type JEV Memory text"
    report["invalidated_prior_evidence"] = {
        "experiments": ["P4-F3", "P4-F4"],
        "reason": "every decision reused reset-time admissible commands",
        "prior_candidate_manipulation_steps": 0,
        "interpretation": "their 0/8 scores cannot estimate model or Memory capability",
    }
    rows = [json.loads(line) for line in (OUTPUT_DIR / "p4f_complete_trajectories.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    report["interface_validation"] = {
        "dynamic_manipulation_steps": sum(x["dynamic_manipulation_steps"] for x in rows),
        "parse_fallbacks": sum(x["parse_fallbacks"] for x in rows),
        "action_state_refresh_observed": any(x["dynamic_manipulation_steps"] > 0 for x in rows),
    }
    report["decision"]["production_memory_authorized"] = False
    report["decision"]["interpretation"] = (
        "Only this refreshed-action-state run is eligible for capability-floor and paired Memory conclusions. "
        "All outputs remain evaluation-only."
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "shared_public_demo.txt").write_text(
        f4.SHARED_PUBLIC_SCAFFOLD, encoding="utf-8"
    )
    (OUTPUT_DIR / "public_demo_source.json").write_text(
        json.dumps(f4.PUBLIC_SOURCE, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), OUTPUT_DIR / Path(__file__).name)
    shutil.copy2(BASE_SCRIPT, OUTPUT_DIR / BASE_SCRIPT.name)
    shutil.copy2(F4_SCRIPT, OUTPUT_DIR / F4_SCRIPT.name)
    (OUTPUT_DIR / "README.md").write_text(
        "# JEV P4-F5 fresh action-state ALFWorld 7B audit\n\n"
        "This run repairs the stale `infos` propagation bug found in P4-F3/F4. "
        "Both arms remain evaluation-only and all authorization fields are false.\n",
        encoding="utf-8",
    )
    archive = shutil.make_archive(
        ARCHIVE_BASE, "zip", root_dir=OUTPUT_DIR.parent, base_dir=OUTPUT_DIR.name
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("P4-F5 archive:", archive)


if __name__ == "__main__":
    main()

