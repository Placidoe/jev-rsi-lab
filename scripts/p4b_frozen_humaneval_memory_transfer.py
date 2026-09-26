#!/usr/bin/env python3
"""P4-B: frozen HumanEval baseline vs JEV retrieval-memory intervention.

Public inputs are cloned at a pinned commit from noahshinn/reflexion.  The
Memory bank contains only reflections with objective before/after evidence in
the official published runs.  A hash split prevents the evaluated task from
contributing its own reflection.  The same open model, generation settings and
unit-test harness are used in both arms.

This experiment tests cross-task Memory transfer.  It is not an ALFWorld-style
long-horizon evaluation and it does not authorize production Memory.
"""

from __future__ import annotations

import ast
import gc
import hashlib
import json
import os
import random
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


SEED = 20260927
REFLEXION_COMMIT = "218cf0ef1df84b05ce379dd4a8e47f17766733a0"
REFLEXION_URL = "https://github.com/noahshinn/reflexion.git"
MODEL_ID = os.environ.get("JEV_MODEL_ID", "Qwen/Qwen2.5-Coder-3B-Instruct")
MAX_TASKS = int(os.environ.get("JEV_MAX_TASKS", "36"))
OUTPUT_DIR = Path(os.environ.get("JEV_OUTPUT_DIR", "results/p4b_frozen_humaneval"))
ARCHIVE_BASE = os.environ.get("JEV_ARCHIVE_BASE", str(OUTPUT_DIR.with_name(f"{OUTPUT_DIR.name}_artifacts")))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def split(task_id: str) -> str:
    value = int(hashlib.sha256(f"{SEED}|{task_id}".encode()).hexdigest()[:8], 16) % 10
    return "memory_build" if value < 7 else "frozen_test"


def clone_source() -> Path:
    target = Path(os.environ.get("JEV_REFLEXION_DIR", "work/reflexion_public_pinned"))
    if not target.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", REFLEXION_URL, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", REFLEXION_COMMIT], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", REFLEXION_COMMIT], check=True)
    actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if actual != REFLEXION_COMMIT:
        raise RuntimeError(f"Pinned commit mismatch: {actual}")
    return target


def build_public_memory(repo: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_path = repo / "programming_runs/root/simple_human_eval_py_logging/humaneval-py..gz_simple_1_gpt-4_pass_at_k_1_py.jsonl"
    refl_path = repo / "programming_runs/root/reflexion_humaneval_py_pass_at_1/reflexion_humaneval_py_pass_at_1.jsonl"
    baseline = {row["task_id"]: row for row in read_jsonl(base_path)}
    reflexion = {row["task_id"]: row for row in read_jsonl(refl_path)}
    memory: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for task_id in sorted(set(baseline) & set(reflexion), key=lambda x: int(x.split("/")[-1])):
        before = bool(baseline[task_id]["is_solved"])
        after = bool(reflexion[task_id]["is_solved"])
        reflections = [str(x).strip() for x in reflexion[task_id].get("reflections", []) if str(x).strip()]
        eligible = split(task_id) == "memory_build" and not before and after and bool(reflections)
        row = {
            "memory_id": hashlib.sha256(f"{REFLEXION_COMMIT}|{task_id}|{'|'.join(reflections)}".encode()).hexdigest()[:20],
            "source_task_id": task_id,
            "source_split": split(task_id),
            "task_prompt": reflexion[task_id]["prompt"],
            "reflection": "\n".join(reflections),
            "baseline_solved": before,
            "reflexion_solved": after,
            "objective_evidence": "HumanEval public unit tests",
            "memory_gate": "staging_eligible" if eligible else "not_eligible",
            "production_memory_authorized": False,
        }
        audit.append(row)
        if eligible:
            memory.append(row)
    if not memory:
        raise RuntimeError("No evidence-backed memories survived the frozen split")
    return memory, audit


def select_frozen_tasks(repo: Path) -> list[dict[str, Any]]:
    # Use the canonical HumanEval rows embedded in the official Reflexion run.
    # programming_runs/benchmarks/humaneval-py.jsonl is a MultiPL-E export and
    # names records with `name` rather than the canonical `task_id`; mixing the
    # two would make the pre-registered hash split impossible to reproduce.
    path = repo / "programming_runs/root/reflexion_humaneval_py_pass_at_1/reflexion_humaneval_py_pass_at_1.jsonl"
    tasks = [row for row in read_jsonl(path) if split(row["task_id"]) == "frozen_test"]
    tasks.sort(key=lambda row: int(row["task_id"].split("/")[-1]))
    return tasks[:MAX_TASKS]


def retrieve_memories(memory: list[dict[str, Any]], tasks: list[dict[str, Any]], top_k: int = 3) -> dict[str, list[dict[str, Any]]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    docs = [row["task_prompt"] for row in memory]
    queries = [row["prompt"] for row in tasks]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    matrix = vectorizer.fit_transform(docs + queries)
    sims = cosine_similarity(matrix[len(docs) :], matrix[: len(docs)])
    result: dict[str, list[dict[str, Any]]] = {}
    for task, scores in zip(tasks, sims, strict=True):
        indexes = scores.argsort()[::-1][:top_k]
        result[task["task_id"]] = [
            {
                "memory_id": memory[int(index)]["memory_id"],
                "source_task_id": memory[int(index)]["source_task_id"],
                "similarity": round(float(scores[int(index)]), 6),
                "reflection": memory[int(index)]["reflection"],
            }
            for index in indexes
        ]
    return result


def messages(task: dict[str, Any], retrieved: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    system = (
        "You are solving a public HumanEval Python problem in an offline benchmark. "
        "Return only the Python function completion that must be appended after the supplied prompt. "
        "Do not repeat imports, signature, docstring, tests, or Markdown fences. Begin with four spaces."
    )
    memory_text = ""
    if retrieved:
        chunks = [f"Experience {i + 1}: {row['reflection']}" for i, row in enumerate(retrieved)]
        memory_text = (
            "\n\nPotentially relevant experience from different training tasks follows. "
            "Use it only when logically applicable; ignore it otherwise.\n" + "\n".join(chunks)
        )
    user = f"Complete this function:\n\n{task['prompt']}{memory_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def clean_completion(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
    if match:
        text = match.group(1).strip("\n")
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith(("def ", "from ", "import ")):
        # The prompt already contains the signature.  Retain only the body after
        # the first function definition when a model ignores the instruction.
        for index, line in enumerate(lines):
            if line.lstrip().startswith("def "):
                lines = lines[index + 1 :]
                break
    normalized = []
    for line in lines:
        if line.strip() and not line.startswith((" ", "\t")):
            line = "    " + line
        normalized.append(line)
    return "\n".join(normalized).rstrip()


FORBIDDEN_NAMES = {"eval", "exec", "compile", "open", "__import__", "input", "breakpoint"}
FORBIDDEN_ROOTS = {"os", "sys", "subprocess", "socket", "requests", "pathlib", "shutil"}


def ast_safe(completion: str) -> tuple[bool, str | None]:
    try:
        tree = ast.parse("def _candidate():\n" + (completion or "    pass") + "\n")
    except SyntaxError as exc:
        return False, f"syntax:{exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "generated_import"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAMES:
            return False, f"forbidden_call:{node.func.id}"
        if isinstance(node, ast.Attribute):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in FORBIDDEN_ROOTS:
                return False, f"forbidden_root:{root.id}"
    return True, None


def _limit_resources() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def evaluate(task: dict[str, Any], completion: str) -> dict[str, Any]:
    safe, reason = ast_safe(completion)
    if not safe:
        return {"passed": False, "error_type": "guard_reject", "detail": reason}
    program = task["prompt"] + completion + "\n" + task["test"] + f"\ncheck({task['entry_point']})\n"
    with tempfile.TemporaryDirectory(prefix="jev_humaneval_") as directory:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", program],
                cwd=directory,
                env={"PATH": os.environ.get("PATH", "")},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=6,
                preexec_fn=_limit_resources,
            )
            return {
                "passed": proc.returncode == 0,
                "error_type": None if proc.returncode == 0 else "test_failure",
                "detail": proc.stderr[-1000:],
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "error_type": "timeout", "detail": "6s timeout"}


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to("cuda:0").eval()
    return model, tokenizer


def generate(model, tokenizer, task: dict[str, Any], retrieved: list[dict[str, Any]] | None) -> tuple[str, float, int, int]:
    import torch

    rendered = tokenizer.apply_chat_template(messages(task, retrieved), tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=4096).to("cuda:0")
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **tokens,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    new_tokens = generated[0, tokens["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return clean_completion(text), seconds, int(tokens["input_ids"].shape[1]), int(new_tokens.shape[0])


def summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    return {
        "tasks": len(selected),
        "passed": sum(row["evaluation"]["passed"] for row in selected),
        "pass_rate": round(sum(row["evaluation"]["passed"] for row in selected) / len(selected), 6),
        "mean_latency_seconds": round(sum(row["latency_seconds"] for row in selected) / len(selected), 6),
        "input_tokens": sum(row["input_tokens"] for row in selected),
        "output_tokens": sum(row["output_tokens"] for row in selected),
    }


def main() -> None:
    import torch
    import transformers

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError("P4-B requires a CUDA GPU")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    repo = clone_source()
    memory, memory_audit = build_public_memory(repo)
    tasks = select_frozen_tasks(repo)
    retrieved = retrieve_memories(memory, tasks)
    write_jsonl(OUTPUT_DIR / "p4b_memory_bank.jsonl", memory)
    write_jsonl(OUTPUT_DIR / "p4b_memory_admission_audit.jsonl", memory_audit)

    model, tokenizer = load_model()
    rows: list[dict[str, Any]] = []
    # Warm-up is excluded from measured latency.
    generate(model, tokenizer, tasks[0], None)
    for index, task in enumerate(tasks):
        arm_order = ["baseline", "jev_memory"] if index % 2 == 0 else ["jev_memory", "baseline"]
        for arm in arm_order:
            used = retrieved[task["task_id"]] if arm == "jev_memory" else None
            completion, seconds, input_tokens, output_tokens = generate(model, tokenizer, task, used)
            result = evaluate(task, completion)
            rows.append(
                {
                    "schema_version": "JEV_P4B_FROZEN_HUMANEVAL_V1",
                    "task_id": task["task_id"],
                    "frozen_split": True,
                    "arm": arm,
                    "retrieved_memory": used or [],
                    "completion": completion,
                    "evaluation": result,
                    "latency_seconds": round(seconds, 6),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            )
            print(task["task_id"], arm, result["passed"], round(seconds, 3), flush=True)

    by_task: dict[str, dict[str, bool]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], {})[row["arm"]] = bool(row["evaluation"]["passed"])
    corrected = [task_id for task_id, values in by_task.items() if not values["baseline"] and values["jev_memory"]]
    regressed = [task_id for task_id, values in by_task.items() if values["baseline"] and not values["jev_memory"]]
    baseline = summary(rows, "baseline")
    jev = summary(rows, "jev_memory")
    report = {
        "schema_version": "JEV_P4B_FROZEN_HUMANEVAL_REPORT_V1",
        "purpose": "Measure cross-task JEV Memory transfer under identical open-model and unit-test conditions",
        "source": {"repository": REFLEXION_URL, "commit": REFLEXION_COMMIT, "license": "MIT"},
        "runtime": {
            "model": MODEL_ID,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "seed": SEED,
        },
        "design": {
            "public_data_only": True,
            "memory_build_tasks": len(memory),
            "frozen_tasks": len(tasks),
            "task_overlap": 0,
            "same_model_and_decoding": True,
            "counterbalanced_arm_order": True,
            "objective_evaluator": "HumanEval public unit tests in restricted subprocess",
        },
        "baseline": baseline,
        "jev_memory": jev,
        "effect": {
            "absolute_pass_rate_delta": round(jev["pass_rate"] - baseline["pass_rate"], 6),
            "corrected_tasks": corrected,
            "regressed_tasks": regressed,
            "net_corrections": len(corrected) - len(regressed),
            "memory_helped": len(corrected) > len(regressed),
        },
        "decision": {
            "promote_memory_mechanism_to_long_horizon_test": len(corrected) > len(regressed),
            "production_memory_authorized": False,
            "next_gate": "Run ALFWorld/WebShop with complete step trajectories; measure task success, steps, repeated actions, recovery, and false transfer.",
        },
        "limitations": [
            "HumanEval is iterative code correction, not a full long-horizon embodied task.",
            "Published Reflexion memories were produced by a different model than the evaluation actor.",
            "A small frozen set has wide uncertainty; effect sizes require replication across seeds and models.",
        ],
    }
    write_jsonl(OUTPUT_DIR / "p4b_frozen_case_results.jsonl", rows)
    (OUTPUT_DIR / "p4b_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "README.md").write_text(
        "# P4-B frozen HumanEval Memory transfer\n\n"
        f"Baseline: {baseline['passed']}/{baseline['tasks']} ({baseline['pass_rate']:.2%})\n\n"
        f"JEV Memory: {jev['passed']}/{jev['tasks']} ({jev['pass_rate']:.2%})\n\n"
        f"Net corrections: {len(corrected) - len(regressed)}\n",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), OUTPUT_DIR / Path(__file__).name)
    archive = shutil.make_archive(ARCHIVE_BASE, "zip", root_dir=OUTPUT_DIR.parent, base_dir=OUTPUT_DIR.name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Archive: {archive}")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

