#!/usr/bin/env python3
"""One-time Kaggle setup for the public ALFWorld P4-E pilot."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


DATA_DIR = Path(os.environ.get("JEV_ALFWORLD_DATA_DIR", "work/alfworld_data"))


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "alfworld==0.4.2",
        "textworld[pddl]==1.6.2",
    ])
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-deps",
        "transformers==4.48.3",
        "huggingface_hub==0.34.6",
        "tokenizers==0.21.0",
    ])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not (DATA_DIR / "json_2.1.1" / "valid_unseen").exists():
        run(["alfworld-download", "--data-dir", str(DATA_DIR)])

    smoke = r'''
import json, os, torch, transformers, textworld
from pathlib import Path
from alfworld.agents.environment import get_environment

data = Path(os.environ["JEV_ALFWORLD_DATA_DIR"])
config = {
 "dataset": {
   "data_path": str(data / "json_2.1.1" / "train"),
   "eval_id_data_path": str(data / "json_2.1.1" / "valid_seen"),
   "eval_ood_data_path": str(data / "json_2.1.1" / "valid_unseen"),
   "num_train_games": -1, "num_eval_games": -1,
 },
 "logic": {"domain": str(data / "logic" / "alfred.pddl"), "grammar": str(data / "logic" / "alfred.twl2")},
 "env": {"goal_desc_human_anns_prob": 0.0, "task_types": [1,2,3,4,5,6], "domain_randomization": False, "expert_type": "handcoded"},
 "general": {"training_method": "dagger"},
 "dagger": {"training": {"max_nb_steps_per_episode": 30}},
}
engine = get_environment("AlfredTWEnv")(config, train_eval="eval_out_of_distribution")
print(json.dumps({
 "torch": torch.__version__, "cuda": torch.version.cuda,
 "transformers": transformers.__version__, "textworld": getattr(textworld, "__version__", "unknown"),
 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
 "alfworld_valid_unseen_games": len(engine.game_files),
}, indent=2))
assert torch.cuda.is_available()
assert len(engine.game_files) >= 100
'''
    subprocess.run([sys.executable, "-c", smoke], check=True, env={**os.environ, "JEV_ALFWORLD_DATA_DIR": str(DATA_DIR)})
    print("P4-E setup passed. Run the pilot as a fresh Python subprocess.")


if __name__ == "__main__":
    main()

