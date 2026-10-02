from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from baseline_eval import dataset_name_from_path, last_answer, score, scoring_text
from page_chunks import PageChunkSelector, query_for, remember_search, split_words
from reason_in_docs import run_batch
from webdancer_prompts import CUSTOM_USER_PROMPT, SEARCH_SCHEMA, VISIT_SCHEMA, make_system_prompt
from wiki_browse_backend import url_to_title

TOOLS = [{"type": "function", "function": SEARCH_SCHEMA}, {"type": "function", "function": VISIT_SCHEMA}]
assert "<answer> answer here </answer>" in CUSTOM_USER_PROMPT
USER_PROMPT = CUSTOM_USER_PROMPT.replace("<answer> answer here </answer>",
                                         "<answer> answer here (usually in less than 10 words) </answer>")
RID_MARKERS = ("<tool_call>", "<tool_response>")
PARSE_ERROR_MSG = ("Error: the tool call could not be parsed. Emit exactly one <tool_call> containing a "
                   "valid JSON object {\"name\": ..., \"arguments\": {...}}, or give the final answer.")


def after_think(text: str) -> str:
    return text.split("</think>")[-1] if "</think>" in text else text


def detect_tool_call(content: str) -> tuple[bool, Optional[dict]]:
    tail = after_think(content)
    start = tail.find("<tool_call>")
    if start == -1:
        return False, None
    body = tail[start + len("<tool_call>"):].split("</tool_call>")[0].strip()
    try:
        call = json.loads(body)
    except json.JSONDecodeError:
        return True, None
    if not isinstance(call, dict) or "name" not in call:
        return True, None
    return True, call


def extract_answer(final_text: str) -> str:
    return last_answer(after_think(final_text)) or ""


def format_google_search(query: str, hits: list[dict]) -> str:
    if not hits:
        return (f"No results found for query: '{query}'. Use a less specific query."
                f"No results found for '{query}'. Try with a more general query.")
    web_snippets = []
    for idx, h in enumerate(hits, 1):
        date_published = "\nDate published: " + h["date"] if h.get("date") else ""
        snippet = "\n" + h["text"] if h.get("text") else ""
        redacted = f"{idx}. [{h['title']}]({h['url']}){date_published}\n{snippet}"
        web_snippets.append(redacted.replace("Your browser can't play this video.", ""))
    return f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)


def visit_info(url: str, goal: str, info: Optional[str]) -> str:
    head = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
    if info is None:
        return (head + "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                + "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n")
    return head + info + "\n\n"


def search_info(query: str, hits: list[dict], info: str) -> str:
    sources = "\n".join(f"{i}. [{h['title']}]({h['url']})" for i, h in enumerate(hits, 1))
    return f"The useful information in the search results for '{query}' as follows: \n\n{info}\n\nSources:\n{sources}"


def trajectory_text(process: dict) -> str:
    lines = []
    for m in process["messages"][1:]:
        if m["role"] == "assistant":
            text = m["content"].replace("<think>", "").replace("</think>", "")
            if "<tool_call>" in text:
                head, _, call = text.partition("<tool_call>")
                call = call.split("</tool_call>")[0]
                text = head.rstrip() + "\n<tool_call> " + " ".join(call.split()) + " </tool_call>"
            lines.append(text.strip())
        else:
            lines.append("<tool_response> " + " ".join(m["content"].split()) + " </tool_response>")
    return "\n\n".join(lines)


def make_backend(args):
    if args.backend == "wiki":
        from wiki_browse_backend import WikiBrowseBackend
        return WikiBrowseBackend(args.retriever_url, top_k=args.top_k, page_max_chars=250000)
    from tavily_browse_backend import TavilyBrowseBackend
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise SystemExit("TAVILY_API_KEY not set in environment.")
    return TavilyBrowseBackend(key, top_k=args.top_k, page_max_chars=250000,
                               workers=args.tavily_workers, rpm=args.tavily_rpm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["wiki", "tavily"], required=True)
    ap.add_argument("--qa_data_path", required=True)
    ap.add_argument("--model_path", default=os.environ.get("PEARL_WEBDANCER_MODEL_PATH"), required=False,
                    help="Path to WebDancer-32B weights. Defaults to the "
                         "PEARL_WEBDANCER_MODEL_PATH environment variable.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--subset_num", type=int, default=-1)
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--max_llm_calls", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16"])
    ap.add_argument("--top_k", type=int, default=5, help="search results per query")
    ap.add_argument("--max_queries", type=int, default=1,
                    help="queries executed per search call (original MAX_MULTIQUERY_NUM, default 3 there)")
    ap.add_argument("--summarize_search", type=int, default=1,
                    help="1: run the extractor over each query's top-k results (as for visited pages)")
    ap.add_argument("--page_topk", type=int, default=5,
                    help="passages kept from an opened page (0 = whole page, as the original)")
    ap.add_argument("--e5_model_path", default=os.environ.get("PEARL_E5_MODEL_PATH"), required=False,
                    help="Path to the e5-base-v2 checkpoint used for page-chunk selection. "
                         "Defaults to the PEARL_E5_MODEL_PATH environment variable.")
    ap.add_argument("--e5_device", default="cuda",
                    help="device for the page-chunk e5 model, e.g. cuda:2 on cards where vLLM's "
                         "own gpu_memory_utilization leaves no room on the default device")
    ap.add_argument("--retriever_url", default="http://127.0.0.1:8765")
    ap.add_argument("--tavily_workers", type=int, default=16)
    ap.add_argument("--tavily_rpm", type=int, default=500)
    ap.add_argument("--dataset_name", default=None, help="default: inferred from --qa_data_path")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    dataset_name = args.dataset_name or dataset_name_from_path(args.qa_data_path)

    with open(args.qa_data_path, encoding="utf-8") as f:
        qa_data = json.load(f)
    if args.subset_num != -1:
        qa_data = qa_data[: args.subset_num]

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "trajectories_live.jsonl")
    out_file = open(out_path, "w", encoding="utf-8")

    t_load0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size, seed=args.seed,
              gpu_memory_utilization=0.90, dtype=args.dtype, max_model_len=args.max_model_len,
              disable_custom_all_reduce=True)
    backend = make_backend(args)
    selector = PageChunkSelector(args.e5_model_path, k=args.page_topk, device=args.e5_device) if args.page_topk > 0 else None

    stats = {"n_questions": len(qa_data), "backend": args.backend,
             "reasoning_prompt_tokens": 0, "reasoning_completion_tokens": 0,
             "extractor_prompt_tokens": 0, "extractor_completion_tokens": 0,
             "planner_prompt_tokens": 0, "planner_completion_tokens": 0,
             "retrieval_queries_issued": 0, "reason_in_docs_calls": 0,
             "n_gpus_reasoning_llm": args.tensor_parallel_size,
             "wall_time_reasoning_llm_load_s": time.time() - t_load0}

    processes = {}
    for item in qa_data:
        qid = item.get("id") or item.get("_id") or str(len(processes))
        q = item.get("Question") or item.get("question")
        processes[qid] = {"id": qid, "question": q, "finished": False, "finish_reason": "",
                          "messages": [{"role": "user", "content": (USER_PROMPT + q).strip()}],
                          "num_llm_calls": 0, "search_count": 0, "visit_count": 0, "parse_errors": 0,
                          "final_text": ""}

    def finish(p, reason, final_text=""):
        p["finished"], p["finish_reason"], p["final_text"] = True, reason, final_text

    def run_reason_in_docs(jobs):
        infos, n_in, n_out = run_batch(llm, tokenizer, jobs, RID_MARKERS, args.max_model_len)
        stats["extractor_prompt_tokens"] += n_in
        stats["extractor_completion_tokens"] += n_out
        stats["reason_in_docs_calls"] += len(jobs)
        return infos

    start_time = time.time()
    round_i = 0
    while True:
        pending = [p for p in processes.values() if not p["finished"]]
        if not pending:
            break
        round_i += 1
        extra = ""
        if hasattr(backend, "stats"):
            t = backend.stats()
            extra = (f" | tavily req={t['tavily_requests']} 429={t['tavily_rate_limited']} "
                     f"retries={t['tavily_retries']} failed={t['tavily_failed_requests']}")
        print(f"----- Round {round_i} | {len(pending)} active{extra} -----", flush=True)

        prompts, sps, active = [], [], []
        for p in pending:
            if p["num_llm_calls"] >= args.max_llm_calls:
                finish(p, "max_llm_calls")
                continue
            text = tokenizer.apply_chat_template([{"role": "system", "content": make_system_prompt()}] + p["messages"],
                                                 tools=TOOLS, add_generation_prompt=True, tokenize=False)
            n_prompt = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            if n_prompt >= args.max_model_len - 256:
                finish(p, "context_overflow")
                continue
            prompts.append(text)
            sps.append(SamplingParams(max_tokens=min(32768, args.max_model_len - n_prompt),
                                      temperature=args.temperature, top_p=args.top_p, top_k=-1,
                                      repetition_penalty=args.repetition_penalty, stop=["<|im_end|>"]))
            active.append(p)
        if not active:
            continue

        outputs = llm.generate(prompts, sampling_params=sps)

        searches, visits = [], []
        for p, out in zip(active, outputs):
            stats["reasoning_prompt_tokens"] += len(out.prompt_token_ids)
            stats["reasoning_completion_tokens"] += len(out.outputs[0].token_ids)
            p["num_llm_calls"] += 1
            content = "<think>\n" + out.outputs[0].text.strip()
            p["messages"].append({"role": "assistant", "content": content})

            has_tag, call = detect_tool_call(content)
            if not has_tag:
                finish(p, "answer", content)
                continue
            if call is None:
                p["parse_errors"] += 1
                p["messages"].append({"role": "tool", "content": PARSE_ERROR_MSG})
                continue

            name, params = call.get("name"), call.get("arguments", {})
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError:
                    params = None
            if name == "search":
                query = params.get("query") if isinstance(params, dict) else None
                if not isinstance(query, (str, list)) or not query:
                    p["messages"].append({"role": "tool", "content": "[Search] Invalid request format: Input must be a JSON object containing 'query' field"})
                    continue
                queries = [query] if isinstance(query, str) else [str(q) for q in query[:args.max_queries]]
                p["search_count"] += 1
                stats["retrieval_queries_issued"] += len(queries)
                searches.append((p, queries))
            elif name == "visit":
                if not isinstance(params, dict) or "url" not in params or "goal" not in params:
                    p["messages"].append({"role": "tool", "content": "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"})
                    continue
                url = params["url"]
                urls = [url] if isinstance(url, str) else [str(u) for u in url] if isinstance(url, list) else []
                if not urls:
                    p["messages"].append({"role": "tool", "content": "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"})
                    continue
                p["visit_count"] += 1
                visits.append((p, urls, str(params["goal"])))
            else:
                p["messages"].append({"role": "tool", "content": f"Tool {name} does not exists."})

        jobs, job_meta = [], []
        search_parts = []
        if searches:
            flat_q = [q for _, qs in searches for q in qs]
            flat_hits = backend.batch_search(flat_q)
            i = 0
            for p, qs in searches:
                parts = []
                for q, hits in zip(qs, flat_hits[i:i + len(qs)]):
                    remember_search(p, q, hits)
                    if hits and args.summarize_search:
                        parts.append(len(jobs))
                        jobs.append((trajectory_text(p), q, [{"title": h["title"], "text": h["text"]} for h in hits]))
                        job_meta.append((q, hits))
                    else:
                        parts.append(format_google_search(q, hits))
                i += len(qs)
                search_parts.append((p, parts))
        visit_slots, flat_v = [], []
        if visits:
            flat_v = [(j, u, goal) for j, (_, urls, goal) in enumerate(visits) for u in urls]
            pages = backend.batch_access([u for _, u, _ in flat_v])
            queries = [query_for(visits[j][0], u) for j, u, _ in flat_v]
            chunks = (selector.select_chunks(pages, queries) if selector
                      else [split_words(pg) if pg else [] for pg in pages])
            for (j, u, goal), q, cs in zip(flat_v, queries, chunks):
                if cs:
                    visit_slots.append(len(jobs))
                    title = url_to_title(u)
                    jobs.append((trajectory_text(visits[j][0]), q, [{"title": title, "text": c} for c in cs]))
                    job_meta.append(None)
                else:
                    visit_slots.append(None)

        infos = run_reason_in_docs(jobs) if jobs else []

        for p, parts in search_parts:
            texts = [search_info(job_meta[x][0], job_meta[x][1], infos[x]) if isinstance(x, int) else x
                     for x in parts]
            p["messages"].append({"role": "tool", "content": "\n=======\n".join(texts)})
        if visits:
            parts_by_job: list[list[str]] = [[] for _ in visits]
            for (j, u, goal), slot in zip(flat_v, visit_slots):
                parts_by_job[j].append(visit_info(u, goal, infos[slot] if slot is not None else None))
            for (p, _, _), parts in zip(visits, parts_by_job):
                p["messages"].append({"role": "tool", "content": "\n=======\n".join(parts).strip()})

        for p in pending:
            if p["finished"] and not p.get("_written"):
                p["_written"] = True
                out_file.write(json.dumps({
                    "id": p["id"], "question": p["question"], "messages": p["messages"],
                    "pred_answer": p.setdefault("final_pred", extract_answer(p["final_text"])),
                    "finish_reason": p["finish_reason"],
                    "num_llm_calls": p["num_llm_calls"], "search_count": p["search_count"],
                    "visit_count": p["visit_count"], "parse_errors": p["parse_errors"],
                }, ensure_ascii=False) + "\n")
                out_file.flush()

    out_file.close()
    wall_time_s = time.time() - start_time
    n_q = len(processes)
    stats["wall_time_total_s"] = wall_time_s
    stats["avg_latency_ms_per_question"] = (wall_time_s * 1000 / n_q) if n_q else 0.0
    stats["gpu_seconds_reasoning_llm"] = wall_time_s * args.tensor_parallel_size
    stats["gpu_hours_reasoning_llm"] = stats["gpu_seconds_reasoning_llm"] / 3600
    stats["total_tokens_all"] = (stats["reasoning_prompt_tokens"] + stats["reasoning_completion_tokens"]
                                 + stats["extractor_prompt_tokens"] + stats["extractor_completion_tokens"])
    if hasattr(backend, "stats"):
        stats.update(backend.stats())
    stats_path = os.path.join(args.output_dir, "run_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Stats saved -> {stats_path}")
    print(f"Done: {n_q} questions in {wall_time_s:.1f}s -> {out_path}")

    outputs = []
    for i, item in enumerate(qa_data):
        p = processes[item.get("id") or item.get("_id") or str(i)]
        text = "\n\n".join(m["content"] for m in p["messages"] if m["role"] == "assistant")
        pred = p.setdefault("final_pred", extract_answer(p["final_text"]))
        outputs.append(scoring_text(text, pred))
    score(qa_data, outputs, dataset_name, args.output_dir, wall_time_s, tokenizer=tokenizer, split=args.split)


if __name__ == "__main__":
    main()
