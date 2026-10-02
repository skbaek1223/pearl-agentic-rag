from __future__ import annotations

import argparse
import json
import math

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from planner_infer import (
    BASE_MODEL_ID,
    FEWSHOT,
    HOP_ROUTER_THRESHOLD,
    MAX_PROMPT_TOKENS,
    MULTIHOP_MIXED_FEWSHOT,
    MULTIHOP_SYSTEM_PROMPT,
    ROUTER_SYSTEM_ANSWER_FIRST,
    SYSTEM_PROMPT,
    _parse_steps,
)

_DEFAULT_LORA_ID = 1
_ROUTER_LORA_ID = 2


def _fewshot_msgs(fewshot: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        m
        for ex in fewshot
        for m in (
            {"role": "user", "content": f"Question: {ex['q']}"},
            {"role": "assistant", "content": ex["plan"]},
        )
    ]


def _select_prompt(dataset_name: str | None, is_multi_hop: bool) -> tuple[str, list[dict[str, str]]]:
    del dataset_name
    if is_multi_hop:
        return MULTIHOP_SYSTEM_PROMPT, MULTIHOP_MIXED_FEWSHOT
    return SYSTEM_PROMPT, FEWSHOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions_file", required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--base_model", default=BASE_MODEL_ID)
    p.add_argument("--adapter_dir", required=True, help="the trained planner LoRA ('default')")
    p.add_argument("--hop_router_dir", required=True, help="the hop-router LoRA")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--threshold", type=float, default=HOP_ROUTER_THRESHOLD)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    p.add_argument("--skip_single", action="store_true",
                    help='Skip plan generation for single-hop items (plan = [question]).')
    args = p.parse_args()

    with open(args.questions_file, encoding="utf-8") as f:
        payload = json.load(f)
    questions: list[str] = payload["questions"]
    dataset_name = payload.get("dataset_name")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=16,
        dtype="half",
        trust_remote_code=True,
        max_model_len=MAX_PROMPT_TOKENS + args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        disable_custom_all_reduce=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    default_lora = LoRARequest("default", _DEFAULT_LORA_ID, args.adapter_dir)
    router_lora = LoRARequest("hop_router", _ROUTER_LORA_ID, args.hop_router_dir)

    total_prompt_tokens = 0
    total_completion_tokens = 0

    single_id = tokenizer.encode(" single", add_special_tokens=False)[0]
    multi_id = tokenizer.encode(" multi", add_special_tokens=False)[0]
    router_prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": ROUTER_SYSTEM_ANSWER_FIRST},
                {"role": "user", "content": q},
            ],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ) + "Answer:"
        for q in questions
    ]
    router_sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    router_outs = llm.generate(router_prompts, router_sp, lora_request=router_lora)
    hop_labels: list[str] = []
    for out in router_outs:
        total_prompt_tokens += len(out.prompt_token_ids)
        step_logprobs = out.outputs[0].logprobs[0]
        p_single = math.exp(step_logprobs[single_id].logprob) if single_id in step_logprobs else 0.0
        p_multi = math.exp(step_logprobs[multi_id].logprob) if multi_id in step_logprobs else 0.0
        p_multi_norm = p_multi / (p_single + p_multi + 1e-9)
        hop_labels.append("multi" if p_multi_norm > args.threshold else "single")
        total_completion_tokens += len(out.outputs[0].token_ids)

    single_idx = [i for i, l in enumerate(hop_labels) if l == "single"]
    multi_idx = [i for i, l in enumerate(hop_labels) if l == "multi"]
    plans: list[list[str] | None] = [None] * len(questions)

    def _gen_subset(idx: list[int], is_multi_hop: bool, lora_request: LoRARequest | None):
        nonlocal total_prompt_tokens, total_completion_tokens
        if not idx:
            return
        system_prompt, fewshot = _select_prompt(dataset_name, is_multi_hop)
        fewshot_msgs = _fewshot_msgs(fewshot)
        subset_qs = [questions[i] for i in idx]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}]
                + fewshot_msgs
                + [{"role": "user", "content": f"Question: {q}"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for q in subset_qs
        ]
        sp = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
        outs = llm.generate(prompts, sp, lora_request=lora_request)
        for i, out in zip(idx, outs):
            total_prompt_tokens += len(out.prompt_token_ids)
            total_completion_tokens += len(out.outputs[0].token_ids)
            plans[i] = _parse_steps(out.outputs[0].text)

    if args.skip_single:
        for i in single_idx:
            plans[i] = [questions[i]]
    else:
        _gen_subset(single_idx, False, None)
    _gen_subset(multi_idx, True, default_lora)

    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump({
            "plans": plans,
            "hop_labels": hop_labels,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
        }, f)


if __name__ == "__main__":
    main()
