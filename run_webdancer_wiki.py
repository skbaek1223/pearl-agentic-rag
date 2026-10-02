from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import Optional
from urllib.parse import unquote, urlparse

import requests
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def make_system_prompt() -> str:
    now = datetime.now().strftime("%A, %B %d, %Y")
    return (
        "You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions."
        "And you are also a Location-Based Services (LBS) assistant designed to help users find location-specific information."
        "No matter how complex the query, you will not give up until you find the corresponding information.\n\nAs you proceed, adhere to the following principles:\n\n"
        "1. **Persistent Actions for Answers**: You will engage in many interactions, delving deeply into the topic to explore all possible aspects until a satisfactory answer is found.\n\n"
        "2. **Repeated Verification**: Before presenting a Final Answer, you will **cross-check** and **validate the information** you've gathered to confirm its accuracy and reliability.\n\n"
        "3. **Attention to Detail**: You will carefully analyze each information source to ensure that all data is current, relevant, and from credible origins.\n\n"
        f"Please note that the current datetime is [{now}]. When responding, consider the time to provide contextually relevant information.\n\n"
        "4. **Answer Language and Format**: Always respond in English. When you are ready to give your Final Answer (no further tool call needed), "
        "give ONLY the short answer phrase itself -- no tags, no restating the question, no explanation -- e.g. `Liverpool` or `1987`, not a full sentence."
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Performs batched web searches: supply an array 'query'; the tool retrieves the top 5 results for each query in one call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of query strings. Include multiple complementary search queries in a single call.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": "Visit webpage(s) and return the summary of the content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.",
                    },
                    "goal": {"type": "string", "description": "The goal of the visit for webpage(s)."},
                },
                "required": ["url", "goal"],
            },
        },
    },
]

MAX_MULTIQUERY_NUM = 3

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning**: Locate the **specific sections/data** directly related to the user's goal within the webpage content.
2. **Key Extraction**: Identify and extract the **most relevant information** from the content, you never miss any important information
3. **Summary Output**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.


**Final Output Format using JSON format**:
{{
  "rational": "string",
  "evidence": "string",
  "summary": "string",
}}
"""

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _find_json_tool_call(text: str, start_marker: str = '{"name"') -> Optional[str]:
    idx = text.find(start_marker)
    if idx == -1:
        return None
    depth = 0
    last_close = -1
    for i in range(idx, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            last_close = i
            if depth == 0:
                return text[idx:i + 1]
    if last_close == -1:
        return None
    candidate = text[idx:last_close + 1]
    remaining_depth = candidate.count("{") - candidate.count("}")
    if remaining_depth > 0:
        return candidate + ("}" * remaining_depth)
    return None


def extract_tool_call(text: str) -> Optional[dict]:
    matches = TOOL_CALL_RE.findall(text)
    if matches:
        try:
            return json.loads(matches[-1])
        except json.JSONDecodeError:
            pass
    obj_text = _find_json_tool_call(text)
    if obj_text is None:
        return None
    try:
        obj = json.loads(obj_text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) and "name" in obj and "arguments" in obj else None


def _looks_like_id(tok: str) -> bool:
    return bool(re.fullmatch(r"\d+|[a-z]{0,3}\d{3,}[a-z0-9]*", tok, re.IGNORECASE))


def _url_to_query(url: str) -> str:
    try:
        path = urlparse(url).path or url
    except ValueError:
        path = url
    seg = unquote(path.rstrip("/").rsplit("/", 1)[-1])
    seg = re.sub(r"\.(html?|php|aspx?)$", "", seg, flags=re.IGNORECASE)
    seg = re.sub(r"[_\-+]", " ", seg)
    seg = re.sub(r"[^\w\s()]", " ", seg)
    seg = re.sub(r"\s+", " ", seg).strip()
    words = [w for w in seg.split() if not _looks_like_id(w)]
    return " ".join(words)


class WikiToolBackend:

    def __init__(self, retriever_url: str, top_k: int = 5):
        self.retriever_url = retriever_url.rstrip("/")
        self.top_k = top_k
        self.seen: dict[str, str] = {}

    def batch_search(self, queries: list[str]) -> list[dict]:
        if not queries:
            return []
        resp = requests.post(f"{self.retriever_url}/batch_search",
                              json={"queries": queries, "top_k": self.top_k}, timeout=120)
        resp.raise_for_status()
        results = resp.json()["results"]
        out = []
        for q, docs in zip(queries, results):
            snippets = []
            for i, d in enumerate(docs):
                url = f"wiki://{d['id']}"
                self.seen[url] = d["contents"]
                title = d["contents"].split("\n", 1)[0].strip('"')
                body = d["contents"].split("\n", 1)[1] if "\n" in d["contents"] else d["contents"]
                snippets.append(f"{i+1}. [{title}]({url})\n{body[:500]}")
            if snippets:
                content = f"A Google search for '{q}' found {len(snippets)} results:\n\n## Web Results\n" + "\n\n".join(snippets)
            else:
                content = f"No results found for '{q}'. Try with a more general query."
            out.append(content)
        return out

    def access(self, url: str, goal: str = "") -> str:
        key = url.strip()
        if key in self.seen:
            return self.seen[key]
        query = " ".join(p for p in [_url_to_query(key), goal] if p).strip()
        if not query:
            return ""
        try:
            resp = requests.post(f"{self.retriever_url}/batch_search",
                                  json={"queries": [query], "top_k": self.top_k}, timeout=120)
            resp.raise_for_status()
            docs = resp.json()["results"][0]
        except Exception:
            return ""
        if not docs:
            return ""
        content = "\n\n=======\n\n".join(d["contents"] for d in docs)
        self.seen[key] = content
        return content


def format_search_tool_response(query_list: list[str], backend: WikiToolBackend) -> str:
    query_list = query_list[:MAX_MULTIQUERY_NUM]
    contents = backend.batch_search(query_list)
    return "\n=======\n".join(contents)


def new_process(qid: str, question: str) -> dict:
    return {"id": qid, "question": question, "messages": [], "finished": False,
            "num_llm_calls": 0, "search_count": 0, "visit_count": 0, "final_text": ""}


def build_prompt(process: dict, tokenizer) -> str:
    messages = [{"role": "system", "content": make_system_prompt()}] + process["messages"]
    return tokenizer.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, tokenize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_data_path", required=True)
    ap.add_argument("--retriever_url", default="http://127.0.0.1:8765")
    ap.add_argument("--model_path", required=True, help="Path to WebDancer-32B weights")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--subset_num", type=int, default=-1)
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--max_llm_calls", type=int, default=20, help="WebDancer's own default (extra={'max_llm_calls': 20})")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_model_len", type=int, default=32768)
    args = ap.parse_args()

    with open(args.qa_data_path, encoding="utf-8") as f:
        qa_data = json.load(f)
    if args.subset_num != -1:
        qa_data = qa_data[: args.subset_num]

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "trajectories_live.jsonl")
    out_file = open(out_path, "w", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    _t_script0 = time.time()
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=0.90, dtype="half", max_model_len=args.max_model_len,
              disable_custom_all_reduce=True)
    sp = SamplingParams(max_tokens=16384, temperature=args.temperature, top_p=args.top_p,
                         repetition_penalty=args.repetition_penalty, seed=args.seed,
                         stop=["<|im_end|>"])

    backend = WikiToolBackend(args.retriever_url, top_k=5)

    processes = {}
    for item in qa_data:
        qid = item.get("id") or item.get("_id") or str(len(processes))
        q = item.get("Question") or item.get("question")
        processes[qid] = new_process(qid, q)
        processes[qid]["messages"] = [{"role": "user", "content": q}]
        processes[qid]["_golden"] = item.get("golden_answers") or item.get("answer")

    def run_extractor(webpage_content: str, goal: str) -> str:
        prompt = EXTRACTOR_PROMPT.format(webpage_content=webpage_content[:8000], goal=goal)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        out = llm.generate([text], sampling_params=SamplingParams(
            max_tokens=1024, temperature=0.0, stop=["<|im_end|>"]))[0]
        raw = out.outputs[0].text.replace("```json", "").replace("```", "").strip()
        try:
            d = json.loads(raw)
            return f"Evidence in page: \n{d.get('evidence', '')}\n\nSummary: \n{d.get('summary', '')}\n\n"
        except json.JSONDecodeError:
            return raw

    def build_extractor_prompt(webpage_content: str, goal: str) -> str:
        prompt = EXTRACTOR_PROMPT.format(webpage_content=webpage_content[:8000], goal=goal)
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

    def format_extractor_output(text: str) -> str:
        raw = text.replace("```json", "").replace("```", "").strip()
        try:
            d = json.loads(raw)
            return f"Evidence in page: \n{d.get('evidence', '')}\n\nSummary: \n{d.get('summary', '')}\n\n"
        except json.JSONDecodeError:
            return raw

    def run_extractor_batch(ext_prompts: list[str]) -> list[str]:
        ext_sp = SamplingParams(max_tokens=1024, temperature=0.0, stop=["<|im_end|>"])
        try:
            outs = llm.generate(ext_prompts, sampling_params=ext_sp)
            stats["extractor_prompt_tokens"] += sum(len(o.prompt_token_ids) for o in outs)
            stats["extractor_completion_tokens"] += sum(len(o.outputs[0].token_ids) for o in outs)
            return [o.outputs[0].text for o in outs]
        except Exception as e:
            print(f"[extractor] batch call failed ({e!r}) -- falling back to sequential", flush=True)
            texts = []
            for t in ext_prompts:
                o = llm.generate([t], sampling_params=ext_sp)[0]
                stats["extractor_prompt_tokens"] += len(o.prompt_token_ids)
                stats["extractor_completion_tokens"] += len(o.outputs[0].token_ids)
                texts.append(o.outputs[0].text)
            return texts

    stats = {
        "n_questions": len(qa_data),
        "reasoning_prompt_tokens": 0,
        "reasoning_completion_tokens": 0,
        "extractor_prompt_tokens": 0,
        "extractor_completion_tokens": 0,
        "planner_prompt_tokens": 0,
        "planner_completion_tokens": 0,
        "retrieval_queries_issued": 0,
        "n_gpus_reasoning_llm": args.tensor_parallel_size,
    }

    start_time = time.time()
    round_i = 0
    while True:
        pending = [p for p in processes.values() if not p["finished"]]
        if not pending:
            break
        round_i += 1
        print(f"----- Round {round_i} | {len(pending)} active -----", flush=True)

        prompts, active = [], []
        for p in pending:
            if p["num_llm_calls"] >= args.max_llm_calls:
                p["finished"] = True
                continue
            prompts.append(build_prompt(p, tokenizer))
            active.append(p)
        if not active:
            continue

        outputs = llm.generate(prompts, sampling_params=sp)
        stats["reasoning_prompt_tokens"] += sum(len(o.prompt_token_ids) for o in outputs)
        stats["reasoning_completion_tokens"] += sum(len(o.outputs[0].token_ids) for o in outputs)

        ext_prompts: list[str] = []
        deferred_visits: list = []

        for p, out in zip(active, outputs):
            gen_text = out.outputs[0].text
            p["num_llm_calls"] += 1
            assistant_text = "<think>\n" + gen_text if not gen_text.lstrip().startswith("<think>") else gen_text
            p["messages"].append({"role": "assistant", "content": assistant_text})

            call = extract_tool_call(gen_text)
            if call is None:
                p["finished"] = True
                p["final_text"] = gen_text
                continue

            name = call.get("name")
            fn_args = call.get("arguments", {})
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except json.JSONDecodeError:
                    fn_args = {"query": fn_args} if name == "search" else {"url": fn_args}
            if isinstance(fn_args, list):
                fn_args = {"query": fn_args} if name == "search" else {"url": fn_args}
            if not isinstance(fn_args, dict):
                fn_args = {}
            if name == "search":
                query = fn_args.get("query", [])
                if isinstance(query, str):
                    query = [query]
                p["search_count"] += 1
                stats["retrieval_queries_issued"] += len(query)
                tool_text = format_search_tool_response(query, backend)
            elif name == "visit":
                url = fn_args.get("url", "")
                goal = fn_args.get("goal", p["question"])
                urls = url if isinstance(url, list) else [url]
                p["visit_count"] += 1
                plan = []
                for u in urls[:MAX_MULTIQUERY_NUM]:
                    page = backend.access(u, goal)
                    if page:
                        ext_prompts.append(build_extractor_prompt(page, goal))
                        plan.append((u, goal, len(ext_prompts) - 1))
                    else:
                        plan.append((u, goal, None))
                deferred_visits.append((p, name, plan))
                continue
            else:
                tool_text = f"Unknown tool '{name}'."

            p["messages"].append({"role": "function", "name": name, "content": tool_text})

        if deferred_visits:
            ext_texts = run_extractor_batch(ext_prompts) if ext_prompts else []
            for p, name, plan in deferred_visits:
                parts = []
                for u, goal, idx in plan:
                    if idx is not None:
                        info = format_extractor_output(ext_texts[idx])
                        parts.append(f"The useful information in {u} for user goal {goal} as follows: \n\n{info}")
                    else:
                        parts.append(f"The useful information in {u} for user goal {goal} as follows: \n\n"
                                      f"Evidence in page: \nThe provided webpage content could not be accessed.\n\n"
                                      f"Summary: \nThe webpage content could not be processed.\n\n")
                tool_text = "\n=======\n".join(parts).strip()
                p["messages"].append({"role": "function", "name": name, "content": tool_text})

        for p in pending:
            if p["finished"] and not p.get("_written"):
                p["_written"] = True
                record = {
                    "id": p["id"],
                    "question": p["question"],
                    "messages": p["messages"],
                    "pred_answer": p["final_text"].split("</think>")[-1].strip() if "</think>" in p["final_text"] else p["final_text"].strip(),
                    "search_count": p["search_count"],
                    "visit_count": p["visit_count"],
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_file.flush()

    out_file.close()
    wall_time_s = time.time() - _t_script0
    n_q = len(processes)
    stats["wall_time_total_s"] = wall_time_s
    stats["avg_latency_ms_per_question"] = (wall_time_s * 1000 / n_q) if n_q else 0.0
    stats["gpu_seconds_reasoning_llm"] = wall_time_s * args.tensor_parallel_size
    stats["gpu_hours_reasoning_llm"] = stats["gpu_seconds_reasoning_llm"] / 3600
    stats["total_tokens_all"] = (stats["reasoning_prompt_tokens"] + stats["reasoning_completion_tokens"]
                                  + stats["extractor_prompt_tokens"] + stats["extractor_completion_tokens"])
    stats_path = os.path.join(args.output_dir, "run_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Stats saved -> {stats_path}")
    print(f"Done: {n_q} questions in {wall_time_s:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
