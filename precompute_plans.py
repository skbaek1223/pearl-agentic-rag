from __future__ import annotations

import argparse
import json
import os
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

PEARL_HOME = os.environ.get("PEARL_HOME", os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ADAPTER_DIR = os.environ.get(
    "PEARL_PLANNER_ADAPTER_DIR",
    os.path.join(PEARL_HOME, "checkpoints", "planner_lora", "final"),
)
BASE_MODEL_ID = "Qwen/Qwen3-8B"
LORA_RANK = 16

SYSTEM_PROMPT = (
    "You are an information retrieval planning expert. Given a question, "
    "generate an ordered sequence of concrete retrieval steps required to "
    "find the answer. Each step represents one retrieval action.\n\n"
    "Write one step per line, numbered. No other text."
)


def build_system_prompt(avoid_repetition: bool = True) -> str:
    if not avoid_repetition:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + (
        "\n\nDo not repeat any phrase or restate the same idea more than once. "
        "Do not restate the same idea across different steps -- each step must "
        "add new information beyond the previous ones."
    )

_STEP_RE = re.compile(r"^\s*\d+[\.\)]\s*(.+)", re.MULTILINE)


_TOKEN_RE = re.compile(r"[^\s/\-]+")
_NONADJ_MIN_NGRAM = 15


def _truncate_repeated_phrase(text: str, min_ngram: int = 3, max_ngram: int | None = None) -> str:
    tokens = [(m.group(0), m.end()) for m in _TOKEN_RE.finditer(text)]
    words = [t[0] for t in tokens]
    n = len(words)
    upper = max_ngram if max_ngram is not None else n // 2
    for i in range(n):
        for w in range(min_ngram, upper + 1):
            if i + 2 * w > n:
                break
            if words[i:i + w] == words[i + w:i + 2 * w]:
                return text[:tokens[i + w - 1][1]]
    for w in range(_NONADJ_MIN_NGRAM, upper + 1):
        seen: dict[tuple[str, ...], int] = {}
        for i in range(n - w + 1):
            gram = tuple(words[i:i + w])
            if gram in seen:
                return text[:tokens[i - 1][1]] if i > 0 else text
            seen[gram] = i
    return text


def _truncate_cross_step_repeat(steps: list[str], min_ngram: int = _NONADJ_MIN_NGRAM) -> list[str]:
    if len(steps) < 2:
        return steps
    offsets = []
    pos = 0
    for s in steps:
        offsets.append((pos, pos + len(s)))
        pos += len(s) + 1
    joined = " ".join(steps)
    tokens = [(m.group(0), m.start()) for m in _TOKEN_RE.finditer(joined)]
    words = [t[0] for t in tokens]
    n = len(words)
    upper = n // 2
    for w in range(min_ngram, upper + 1):
        seen: dict[tuple[str, ...], int] = {}
        for i in range(n - w + 1):
            gram = tuple(words[i:i + w])
            if gram in seen:
                cut_char = tokens[i][1]
                for idx, (s_start, s_end) in enumerate(offsets):
                    if s_start <= cut_char < s_end:
                        return steps[:idx] if idx > 0 else steps
                return steps
            seen[gram] = i
    return steps


def _parse_steps(text: str, max_steps: int | None = None) -> list[str]:
    steps = [m.group(1).strip() for m in _STEP_RE.finditer(text)]
    steps = steps if steps else ([text.strip()] if text.strip() else [])
    steps = [_truncate_repeated_phrase(s) for s in steps]
    steps = _truncate_cross_step_repeat(steps)
    if max_steps is not None and len(steps) > max_steps:
        steps = steps[:max_steps]
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_data_path", required=True,
                     help="QA dataset JSON: a list of {Question, ...} items.")
    ap.add_argument("--cache_path", required=True,
                     help="JSON cache file: {question: [step, ...]}. Extended in place.")
    ap.add_argument("--adapter_dir", default=DEFAULT_ADAPTER_DIR)
    ap.add_argument("--base_model", default=BASE_MODEL_ID)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_model_len", type=int, default=1024)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--repetition_penalty", type=float, default=None,
                     help='vLLM repetition_penalty (unused by default).')
    ap.add_argument("--max_steps", type=int, default=None,
                     help='Truncate parsed plans to at most this many steps.')
    ap.add_argument("--force_regenerate", action="store_true",
                     help='Regenerate every plan even if already cached.')
    args = ap.parse_args()

    with open(args.qa_data_path, encoding="utf-8") as f:
        qa_data = json.load(f)
    questions = [item["Question"] for item in qa_data]
    unique_questions = list(dict.fromkeys(questions))
    print(f"[precompute_plans] {len(qa_data)} items, {len(unique_questions)} unique questions "
          f"in {args.qa_data_path}")

    cache: dict[str, list[str]] = {}
    if os.path.exists(args.cache_path):
        cache = json.load(open(args.cache_path, encoding="utf-8"))
        print(f"[precompute_plans] loaded {len(cache)} cached plans from {args.cache_path}")

    if args.force_regenerate:
        todo = unique_questions
        print(f"[precompute_plans] --force_regenerate: regenerating all {len(todo)} questions")
    else:
        todo = [q for q in unique_questions if q not in cache]
        print(f"[precompute_plans] {len(todo)} questions need a fresh plan")

    if todo:
        n_gpus = len([g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]) or 1
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        print(f"[precompute_plans] loading {args.base_model} via vLLM "
              f"(tensor_parallel_size={n_gpus}, LoRA adapter={args.adapter_dir})...")
        llm = LLM(model=args.base_model, tensor_parallel_size=n_gpus,
                  enable_lora=True, max_lora_rank=LORA_RANK,
                  dtype="half", max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_memory_utilization)
        lora_request = LoRARequest("planner_lora", 1, args.adapter_dir)

        system_prompt = build_system_prompt()
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": f"Question: {q}"}],
                tokenize=False, add_generation_prompt=True,
            )
            for q in todo
        ]
        sp_kwargs = dict(max_tokens=args.max_new_tokens, temperature=0.0)
        if args.repetition_penalty is not None:
            sp_kwargs["repetition_penalty"] = args.repetition_penalty
        sp = SamplingParams(**sp_kwargs)
        outs = llm.generate(prompts, sampling_params=sp, lora_request=lora_request)
        for q, out in zip(todo, outs):
            cache[q] = _parse_steps(out.outputs[0].text, max_steps=args.max_steps)

        os.makedirs(os.path.dirname(args.cache_path) or ".", exist_ok=True)
        with open(args.cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"[precompute_plans] saved {len(cache)} total cached plans -> {args.cache_path}")
    else:
        print("[precompute_plans] nothing to do, cache already covers every question")


if __name__ == "__main__":
    main()
