from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter


def load_result(pattern: str):
    matches = [m for m in glob.glob(pattern) if not m.endswith(".metrics.json")]
    if not matches:
        return None, None
    matches.sort(key=os.path.getmtime, reverse=True)
    return json.load(open(matches[0])), matches[0]


def load_result_any_split(shard_dir: str):
    for split_prefix in ("test", "dev"):
        result, path = load_result(f"{shard_dir}/{split_prefix}.*.json")
        if result is not None:
            return result, path
    return None, None


def load_jsonl(path: str) -> dict:
    items = {}
    if not os.path.exists(path):
        return items
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items[d["id"]] = d
    return items


def sum_stats(stats_list: list[dict], keys: list[str]) -> dict:
    return {k: sum((s.get(k, 0) or 0) for s in stats_list) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--plan_version", required=True, help="e.g. v10")
    ap.add_argument("--shard_dirs", nargs="+", required=True)
    ap.add_argument("--retry_dirs", nargs="*", default=[],
                     help="Same length/order as --shard_dirs; pass 'none' for a shard with no retry.")
    ap.add_argument("--max_turn", type=int, required=True,
                     help="The max_turn actually used for the FINAL (post-retry) result of every item.")
    ap.add_argument("--plan_cache", required=True,
                     help="cache/plans_{dataset}_{plan_version}.json -- for planner token stats.")
    ap.add_argument("--base_model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out_dir", default="final_results")
    ap.add_argument("--llm_judge_summary", default=None,
                     help='Optional summary from llm_judge_grade.py; adds LLM-judge accuracy to accuracy_metrics.')
    args = ap.parse_args()

    retry_dirs = args.retry_dirs or ["none"] * len(args.shard_dirs)
    assert len(retry_dirs) == len(args.shard_dirs)

    merged_all = []
    merged_traj = {}
    stat_keys = [
        "reasoning_prompt_tokens", "reasoning_completion_tokens",
        "extractor_prompt_tokens", "extractor_completion_tokens",
        "retrieval_queries_issued", "n_turns_total", "gpu_seconds_reasoning_llm",
        "planner_prompt_tokens", "planner_completion_tokens",
    ]
    stats_totals = {k: 0 for k in stat_keys}
    latency_weighted_sum = 0.0

    for shard_dir, retry_dir in zip(args.shard_dirs, retry_dirs):
        orig, _ = load_result_any_split(shard_dir)
        if orig is None:
            raise FileNotFoundError(f"no test.*.json or dev.*.json under {shard_dir}")
        orig_traj = load_jsonl(f"{shard_dir}/trajectories_live.jsonl")
        orig_stats = json.load(open(f"{shard_dir}/run_stats.json"))
        for k in stat_keys:
            stats_totals[k] += orig_stats.get(k, 0) or 0

        retry_by_id, retry_traj = {}, {}
        n_replaced_this_shard = 0
        if retry_dir != "none":
            retry, _ = load_result_any_split(retry_dir)
            if retry is not None:
                retry_by_id = {item["id"]: item for item in retry}
                n_replaced_this_shard = len(retry_by_id)
                retry_traj = load_jsonl(f"{retry_dir}/trajectories_live.jsonl")
                retry_stats = json.load(open(f"{retry_dir}/run_stats.json"))
                for k in stat_keys:
                    stats_totals[k] += retry_stats.get(k, 0) or 0
                n_kept = orig_stats["n_questions"] - n_replaced_this_shard
                latency_weighted_sum += (
                    orig_stats["avg_latency_ms_per_question"] * n_kept
                    + retry_stats["avg_latency_ms_per_question"] * n_replaced_this_shard
                )
            else:
                latency_weighted_sum += orig_stats["avg_latency_ms_per_question"] * orig_stats["n_questions"]
        else:
            latency_weighted_sum += orig_stats["avg_latency_ms_per_question"] * orig_stats["n_questions"]

        for item in orig:
            qid = item["id"]
            if qid in retry_by_id:
                merged_all.append(retry_by_id[qid])
                merged_traj[qid] = retry_traj.get(qid, orig_traj.get(qid))
            else:
                merged_all.append(item)
                merged_traj[qid] = orig_traj.get(qid)

    merged_all.sort(key=lambda it: int(it["id"].split("_")[-1]))

    n = len(merged_all)
    valid = [it for it in merged_all if it["Metrics"]["is_valid_answer"]]
    acc = sum(it["Metrics"]["acc"] for it in merged_all) / n
    em = sum(it["Metrics"]["em"] for it in merged_all) / n
    f1 = sum(it["Metrics"]["f1"] for it in merged_all) / n
    rum_rep = sum(it["Rumination"]["rumination_5gram_rep"] for it in merged_all) / n
    rum_ent = sum(it["Rumination"]["rumination_lex_entropy"] for it in merged_all) / n

    search_counts = [t["search_count"] for t in merged_traj.values() if t]
    finish_reasons = Counter(t["finish_reason"] for t in merged_traj.values() if t)
    if search_counts:
        avg_search_count = sum(search_counts) / len(search_counts)
    else:
        avg_search_count = stats_totals["retrieval_queries_issued"] / n if n else 0.0

    agent_prompt = stats_totals["reasoning_prompt_tokens"] / n
    agent_gen = stats_totals["reasoning_completion_tokens"] / n
    extr_prompt = stats_totals["extractor_prompt_tokens"] / n
    extr_comp = stats_totals["extractor_completion_tokens"] / n
    extr_total = extr_prompt + extr_comp

    planner_prompt_avg = stats_totals["planner_prompt_tokens"] / n
    planner_comp_avg = stats_totals["planner_completion_tokens"] / n
    planner_source = "run_stats.json (in-process planner, real vLLM-tracked counts)"
    if planner_prompt_avg == 0.0 and planner_comp_avg == 0.0 and os.path.exists(args.plan_cache):
        planner_source = f"re-tokenized {args.plan_cache} (offline precompute_plans.py workflow)"
        from transformers import AutoTokenizer
        from precompute_plans import build_system_prompt

        tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        plans = json.load(open(args.plan_cache))
        system_prompt = build_system_prompt()
        p_tot = c_tot = 0
        for q, steps in plans.items():
            user_prompt = tok.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": f"Question: {q}"}],
                tokenize=False, add_generation_prompt=True,
            )
            p_tot += len(tok(user_prompt, add_special_tokens=False)["input_ids"])
            plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
            c_tot += len(tok(plan_text, add_special_tokens=False)["input_ids"])
        planner_prompt_avg = p_tot / len(plans)
        planner_comp_avg = c_tot / len(plans)
    elif planner_prompt_avg == 0.0 and planner_comp_avg == 0.0:
        planner_source = f"unavailable (no in-process data in run_stats.json, and {args.plan_cache} does not exist)"

    total_tokens_avg = agent_prompt + agent_gen + extr_total + planner_prompt_avg + planner_comp_avg

    accuracy_metrics = {"acc": acc, "em": em, "f1": f1}
    if args.llm_judge_summary:
        judge = json.load(open(args.llm_judge_summary))
        judge_by_id = {g["id"]: g for g in judge["per_item"]}
        for it in merged_all:
            g = judge_by_id.get(it["id"])
            if g:
                it["LLM_Judge"] = g
        matched = [judge_by_id[it["id"]] for it in merged_all if it["id"] in judge_by_id]
        n_judged = len(matched)
        n_judge_correct = sum(1 for g in matched if g["correct"])
        accuracy_metrics["llm_judge_accuracy"] = n_judge_correct / n_judged if n_judged else None
        accuracy_metrics["llm_judge_n_correct"] = n_judge_correct
        accuracy_metrics["llm_judge_n_total"] = n_judged
        accuracy_metrics["llm_judge_model"] = judge.get("judge_model")
        if n_judged < n:
            print(f"[warn] llm_judge_summary only covers {n_judged}/{n} merged items")

    os.makedirs(args.out_dir, exist_ok=True)
    final_path = f"{args.out_dir}/{args.dataset}_{args.plan_version}_final.json"
    summary_path = f"{args.out_dir}/{args.dataset}_{args.plan_version}_final_summary.json"

    json.dump(merged_all, open(final_path, "w"), ensure_ascii=False, indent=2)

    summary = {
        "dataset": args.dataset,
        "plan_version": args.plan_version,
        "max_turn": args.max_turn,
        "n_total": n,
        "n_valid_answer": len(valid),
        "valid_answer_rate": len(valid) / n,
        "accuracy_metrics": accuracy_metrics,
        "quality_metrics": {"rumination_5gram_rep": rum_rep, "rumination_lex_entropy": rum_ent},
        "efficiency_metrics": {
            "avg_search_count_per_question": avg_search_count,
            "finish_reason_distribution": dict(finish_reasons),
            "total_retrieval_queries_issued": stats_totals["retrieval_queries_issued"],
            "total_turns": stats_totals["n_turns_total"],
            "avg_latency_ms_per_question": round(latency_weighted_sum / n, 1),
            "gpu_hours_reasoning_llm": stats_totals["gpu_seconds_reasoning_llm"] / 3600,
        },
        "avg_token_usage_per_question": {
            "agent_prompt_tokens": round(agent_prompt, 2),
            "agent_generated_tokens": round(agent_gen, 2),
            "extraction_prompt_tokens": round(extr_prompt, 2),
            "extraction_completion_tokens": round(extr_comp, 2),
            "extraction_tokens_total": round(extr_total, 2),
            "planner_prompt_tokens": round(planner_prompt_avg, 2),
            "planner_completion_tokens": round(planner_comp_avg, 2),
            "planner_tokens_total": round(planner_prompt_avg + planner_comp_avg, 2),
            "avg_total_input_tokens": round(agent_prompt + extr_prompt + planner_prompt_avg, 2),
            "avg_total_output_tokens": round(agent_gen + extr_comp + planner_comp_avg, 2),
            "avg_total_tokens": round(total_tokens_avg, 2),
            "note": (
                "agent_* = QwQ-32B reasoning turns (search-query generation + "
                "final boxed answer). extraction_* = QwQ-32B calls that extract "
                "facts from retrieved documents per search result. planner_* = "
                f"Qwen3-8B+S8000-LoRA plan generation, source: {planner_source}."
            ),
        },
    }
    json.dump(summary, open(summary_path, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {final_path} ({n} items, sorted by id)")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
